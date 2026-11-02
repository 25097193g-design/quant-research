"""paper_lib — 特朗普关税对标普500反事实拟合论文的共享库

为 notebooks/paper_tariff/ 系列 notebook 提供统一的数据/方法/评估工具。
照 wq_lib.py 模板组织(模块docstring + 分区 + 顶层评估函数)。

核心方法:
  - 合成控制法 SCM (Abadie, Diamond & Hainmueller 2010) — 手写 cvxpy
  - CausalImpact (Brodersen, Gallus et al. 2015) — statsmodels.UnobservedComponents 封装(BSTS 频率派等价)
  - 事件研究 (MacKinlay 1997) — 市场模型 + CAR
  - 稳健性: placebo / leave-one-out / permutation p-value

数据约定:
  - 价格面板: pd.DataFrame, index=日期(Timestamp), columns=ticker, 值=adj_close(或 close)
  - 由 download_tariff_data.py 生成, 落 data/tariff/{group}_panel_{VERSION}.parquet
  - 事件日期: data/tariff/treaty_event_dates.csv(常量, 防未来函数)

防未来函数(铁律):
  - SCM 权重只用预处理期(pre-event)数据拟合, Z1/Z0 严格切到 event_date 之前
  - post 期缺口 = 实际 - 合成, 合成用 pre 期权重(不偷看 post)

用法:
    from paper_lib import load_tariff_data, fit_scm_pipeline
    panels = load_tariff_data()
    res = fit_scm_pipeline(panels['us_index']['^GSPC'], panels['sector_etf'],
                           event_date='2025-04-02', pre_start='2024-01-01')
    res['gap_post'].plot()  # 反事实累计缺口
"""

import os
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))  # notebooks/paper_tariff/
DATA_DIR = os.environ.get(
    'TARIFF_DATA_DIR',
    os.path.normpath(os.path.join(_HERE, '..', '..', 'data', 'tariff')),
)
EVENT_CSV = os.path.join(DATA_DIR, 'treaty_event_dates.csv')

# 美股年交易日数(年化用)
TRADING_DAYS = 252


# ════════════════════════════════════════════════════════════════
# 区1: 数据加载 / 对齐
# ════════════════════════════════════════════════════════════════

def load_tariff_data(version='20260712', data_dir=DATA_DIR):
    """加载 4 个 parquet 面板 → dict{group: wide DataFrame(index=date, columns=ticker)}。

    groups: us_index / sector_etf / overseas_index / macro
    """
    out = {}
    for tag in ['us_index', 'sector_etf', 'overseas_index', 'macro']:
        path = os.path.join(data_dir, f'{tag}_panel_{version}.parquet')
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        out[tag] = df.sort_index()
    return out


def load_event_dates(csv_path=EVENT_CSV):
    """读 treaty_event_dates.csv → DataFrame[round,date,phase,label,source]。"""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)


def to_log_returns(price):
    """价格 → 对数收益 log(p_t / p_{t-1})。pct_change 后 log1p(数值稳)。"""
    return np.log1p(price.pct_change().fillna(0.0))


def to_cumlog(price, base_date=None):
    """价格 → 累计对数收益 log(p_t) - log(p_base), 从 base_date 归一化到 0。

    SCM 匹配"走势"而非"水平": treated 和 donor 都从同一基准日归一化,
    预处理期权重匹配走势, post 期缺口即累计异常对数收益。
    base_date=None 时用序列首日; base_date 非交易日时取 >= 该日的首个交易日。
    """
    lp = np.log(price)
    base = lp.loc[lp.index >= pd.Timestamp(base_date)].iloc[0] if base_date else lp.iloc[0]
    return lp - base


def align_to_treated(treated, donors, freq='B'):
    """把 donor 面板对齐到 treated 的交易日(标普500有交易的日期)。

    海外指数节假日与美股不重合 → 用 treated 交易日为锚, donor reindex+ffill。
    返回 (treated_aligned, donors_aligned)。
    """
    idx = treated.dropna().index
    d = donors.reindex(idx).ffill().dropna(how='all')
    t = treated.reindex(idx).dropna()
    # 取两者共同非空日期
    common = t.index.intersection(d.dropna(how='all').index)
    return t.loc[common], d.loc[common]


# ════════════════════════════════════════════════════════════════
# 区2: 合成控制法 SCM(Abadie, Diamond & Hainmueller 2010)
# ════════════════════════════════════════════════════════════════

def build_predictors(cumlog_pre, ret_pre):
    """构造 SCM 协变量预测向量(Abadie 标准: 水平 + 波动 + 终点)。

    参数:
        cumlog_pre: pd.Series 预处理期累计对数收益(水平匹配)
        ret_pre   : pd.Series 预处理期对数收益(波动匹配)
    返回: np.ndarray [mean_ret, std_ret, end_cumlog]
    """
    return np.array([
        ret_pre.mean(),
        ret_pre.std(),
        cumlog_pre.iloc[-1],
    ])


def fit_scm(Z1, Z0, X1=None, X0=None, solver='CLARABEL'):
    """合成控制法拟合权重(Abadie, Diamond & Hainmueller 2010)。

    min_w  ||Z1 - Z0 w||^2 + ||X1 - X0 w||^2   s.t.  w>=0, sum(w)=1
    其中 Z 是预处理期结果路径(必填), X 是协变量预测向量(可选)。

    参数:
        Z1 : (T0,)      干预组预处理期结果路径(累计对数收益)
        Z0 : (T0, J)    donor 预处理期结果路径
        X1 : (k,)       干预组协变量(可选)
        X0 : (k, J)     donor 协变量(可选)
        solver : cvxpy 求解器(CLARABEL 实测最快最稳; 备选 ECOS/SCS/CVXOPT)
    返回:
        w : (J,)       donor 权重(和为1, 非负)
        mspe : float   预处理期拟合 MSPE(越小拟合越好)
    """
    Z1 = np.asarray(Z1, dtype=float)
    Z0 = np.asarray(Z0, dtype=float)
    T0, J = Z0.shape
    w = cp.Variable(J, nonneg=True)
    terms = [cp.sum_squares(Z1 - Z0 @ w)]
    if X1 is not None and X0 is not None:
        X1 = np.asarray(X1, dtype=float)
        X0 = np.asarray(X0, dtype=float)
        terms.append(cp.sum_squares(X1 - X0 @ w))
    prob = cp.Problem(cp.Minimize(cp.sum(terms)), [cp.sum(w) == 1])
    try:
        prob.solve(solver=solver)
    except (cp.error.SolverError, ValueError):
        prob.solve()  # 指定求解器失败 → cvxpy 自动选(仅捕获求解异常, 不吞 KeyboardInterrupt 等)
    if w.value is None:
        raise RuntimeError(f'SCM 求解失败 status={prob.status}')
    w_val = np.asarray(w.value).ravel()
    w_val = np.clip(w_val, 0, None)
    w_val = w_val / w_val.sum()   # 数值净化: 严格归一
    mspe = float(((Z1 - Z0 @ w_val) ** 2).mean())
    return w_val, mspe


def synthetic_path(w, Z0_post):
    """生成干预期合成对照序列。Z0_post: (T1, J) → (T1,)。"""
    return np.asarray(Z0_post, dtype=float) @ np.asarray(w, dtype=float)


def gap_series(Y1_post, Z0_post, w):
    """干预期缺口 = 实际 - 合成。返回 (T1,)。"""
    return np.asarray(Y1_post, dtype=float) - synthetic_path(w, Z0_post)


def fit_scm_pipeline(price_treated, price_donors, event_date, pre_start,
                     post_end=None, use_covariates=True, solver='CLARABEL'):
    """一键 SCM: 价格 → 预处理对齐 → 拟合权重 → post 合成 + 缺口。

    参数:
        price_treated : pd.Series 干预组价格(如 panels['us_index']['^GSPC'])
        price_donors  : pd.DataFrame donor 价格面板(columns=donor)
        event_date    : str/datetime 干预日(含当天起算 post)
        pre_start     : str/datetime 预处理期起点
        post_end      : str/datetime post 终点(None=数据末)
        use_covariates: bool 是否加协变量(均值/波动/终点)进目标函数
    返回: dict
        weights: pd.Series donor 权重(降序)
        mspe_pre, mspe_post, mspe_ratio
        gap_pre, gap_post: pd.Series 缺口(累计对数收益差)
        synthetic: pd.DataFrame [actual, synthetic] post 期对比
        cumlog_treated, cumlog_donors: 归一化后的预处理期序列(绘图/检查用)
    """
    event_date = pd.Timestamp(event_date)
    pre_start = pd.Timestamp(pre_start)
    post_end = pd.Timestamp(post_end) if post_end else price_treated.index[-1]

    # 对齐到 treated 交易日
    t, d = align_to_treated(price_treated, price_donors)
    # 预处理 / post 切分(event_date 含 post)
    pre_mask = (t.index >= pre_start) & (t.index < event_date)
    post_mask = (t.index >= event_date) & (t.index <= post_end)
    t_pre, d_pre = t.loc[pre_mask], d.loc[pre_mask]
    t_post, d_post = t.loc[post_mask], d.loc[post_mask]

    if len(t_pre) < 20:
        raise ValueError(f'预处理期太短({len(t_pre)} 日), pre_start 太晚或数据缺失')
    d_pre = d_pre.dropna(axis=1)          # 去掉预处理期无数据的 donor
    d_post = d_post[d_pre.columns]         # post 对齐 donor 列

    # 归一化: cumlog 从 pre_start 起
    cumlog_t = to_cumlog(t_pre, base_date=pre_start)
    cumlog_d = to_cumlog(d_pre, base_date=pre_start)
    ret_t = to_log_returns(t_pre)
    ret_d = to_log_returns(d_pre)

    Z1 = cumlog_t.values
    Z0 = cumlog_d.values
    if use_covariates:
        X1 = build_predictors(cumlog_t, ret_t)
        X0 = np.array([build_predictors(cumlog_d[c], ret_d[c]) for c in cumlog_d.columns]).T
    else:
        X1 = X0 = None

    w, mspe_pre = fit_scm(Z1, Z0, X1, X0, solver=solver)
    weights = pd.Series(w, index=cumlog_d.columns, name='weight').sort_values(ascending=False)

    # post 缺口(累计对数收益口径, 从 event_date 归一化)
    cumlog_t_post = to_cumlog(pd.concat([t_pre.tail(1), t_post]), base_date=t_pre.index[-1]).iloc[1:]
    cl_post_list = []
    for c in d_pre.columns:
        s = pd.concat([d_pre[c].tail(1), d_post[c]]).iloc[1:]
        cl_post_list.append(to_cumlog(pd.concat([d_pre[c].tail(1), d_post[c]]),
                                      base_date=d_pre.index[-1]).iloc[1:].values)
    Z0_post = np.array(cl_post_list).T   # (T1, J)
    Y1_post = cumlog_t_post.values
    gap_post = pd.Series(Y1_post - Z0_post @ w, index=t_post.index, name='gap')

    # 预处理期缺口(平行趋势检查)
    gap_pre = pd.Series(Z1 - Z0 @ w, index=t_pre.index, name='gap')

    mspe_post = float((gap_post.values ** 2).mean())
    mspe_ratio = mspe_post / mspe_pre if mspe_pre > 0 else np.inf

    return {
        'weights': weights,
        'mspe_pre': mspe_pre, 'mspe_post': mspe_post, 'mspe_ratio': mspe_ratio,
        'gap_pre': gap_pre, 'gap_post': gap_post,
        'synthetic': pd.DataFrame({'actual': Y1_post, 'synthetic': Z0_post @ w},
                                  index=t_post.index),
        'cumlog_treated': cumlog_t, 'cumlog_donors': cumlog_d,
    }


def fit_ascm_pipeline(price_treated, price_donors, event_date, pre_start,
                      post_end=None, ridge_alpha=1.0, use_covariates=True,
                      solver='CLARABEL'):
    """Augmented SCM(Ben-Michael/Feller/Rothstein 2021; Doudchenko-Imbens 2016)。

    标准 SCM 的凸+非负权重约束把反事实限制在 donor 凸包内: 当 treated 落在包"内"(本文
    行业 ETF≈标普成分切片, 标普在 donor 凸包内 by construction)时, 合成≈重建标普, gap≈0
    (nb03 donor 同源问题, §5.2/§5.13)。augmented SCM 用 ridge 回归(Doudchenko-Imbens,
    允许负权重/外推)给反事实一个不受凸性约束的形状, 再以事件日(treated 与合成相接处)为锚
    做水平对齐(事件反事实惯例, gap 在 event_date=0)。

    估计量(augmented SCM 族, ridge 外推变体):
        1. w_scm  : 凸 SCM 权重(pre 期), 复用 fit_scm —— 兼对照 + 锚定基准。
        2. ridge  : Z1_pre ~ Z0_pre(允许负权重), 外推到 Z0_post。
        3. cf_ascm(t) = ridge_cf(t) + [Y1(pre_end) − ridge_cf(pre_end)]   (水平锚定)
           gap_ascm(t) = Y1(t) − cf_ascm(t)
    与 fit_scm_pipeline 同 cumlog 口径(pre_start 归一化, post 从 pre_end 续), 数值可直接比。

    参数:
        ridge_alpha: ridge L2 惩罚(越大越收缩向等权/凸组合, 越小越外推)。默认 1.0。
            α 敏感: 报 ASCM gap 时应同时报 α 网格(如 [0.1,1,10,100])的区间, 不报单点。
        其余同 fit_scm_pipeline。
    返回: dict
        weights_scm        : pd.Series 凸 SCM 权重(对照)
        ridge_coef         : pd.Series ridge 系数(可负, 外推权重)
        ridge_intercept, ridge_alpha
        gap_post           : pd.Series post 期 ASCM 缺口(累计对数收益差)
        gap_ascm_post_end  : float post 终点 ASCM 缺口(与 fit_scm_pipeline 的 gap_post.iloc[-1] 同口径)
        synthetic_ascm     : pd.DataFrame [actual, ascm_cf] post 期对比
        mspe_pre_scm       : float 凸 SCM 预处理 MSPE(对照)
    """
    from sklearn.linear_model import Ridge
    event_date = pd.Timestamp(event_date); pre_start = pd.Timestamp(pre_start)
    post_end = pd.Timestamp(post_end) if post_end else price_treated.index[-1]

    t, d = align_to_treated(price_treated, price_donors)
    pre_mask = (t.index >= pre_start) & (t.index < event_date)
    post_mask = (t.index >= event_date) & (t.index <= post_end)
    t_pre, d_pre = t.loc[pre_mask], d.loc[pre_mask]
    t_post, d_post = t.loc[post_mask], d.loc[post_mask]
    if len(t_pre) < 20:
        raise ValueError(f'预处理期太短({len(t_pre)} 日)')
    d_pre = d_pre.dropna(axis=1)
    d_post = d_post[d_pre.columns]
    n_pre = len(t_pre)

    # cumlog 全期(pre+post)从 pre_start 归一化
    cumlog_t_all = to_cumlog(pd.concat([t_pre, t_post]), base_date=pre_start)
    cumlog_d_all = to_cumlog(pd.concat([d_pre, d_post]), base_date=pre_start)
    Z1_pre = cumlog_t_all.iloc[:n_pre].values
    Z0_pre = cumlog_d_all.iloc[:n_pre].values
    Z0_post = cumlog_d_all.iloc[n_pre:].values
    Y1_all = cumlog_t_all.values

    # 1. 凸 SCM 权重(对照 + 锚定参考), 复用 fit_scm(含协变量, 与 nb03 口径一致)
    ret_t_pre = to_log_returns(t_pre); ret_d_pre = to_log_returns(d_pre)
    if use_covariates:
        X1 = build_predictors(to_cumlog(t_pre, base_date=pre_start), ret_t_pre)
        X0 = np.array([build_predictors(to_cumlog(d_pre[c], base_date=pre_start), ret_d_pre[c])
                       for c in d_pre.columns]).T
    else:
        X1 = X0 = None
    w_scm, mspe_pre = fit_scm(Z1_pre, Z0_pre, X1, X0, solver=solver)
    weights_scm = pd.Series(w_scm, index=d_pre.columns, name='weight_scm').sort_values(ascending=False)

    # 2. ridge(Doudchenko-Imbens): 允许负权重/外推, pre 期拟合 treated ~ donors, 外推到 post
    ridge = Ridge(alpha=ridge_alpha, fit_intercept=True)
    ridge.fit(Z0_pre, Z1_pre)
    ridge_cf_all = ridge.predict(np.vstack([Z0_pre, Z0_post]))   # (n_pre+n_post,)

    # 3. 水平锚定到 pre 期末(事件日 gap=0), 口径对齐 fit_scm_pipeline
    anchor = Y1_all[n_pre - 1] - ridge_cf_all[n_pre - 1]
    cf_ascm_all = ridge_cf_all + anchor
    gap_post = pd.Series(Y1_all[n_pre:] - cf_ascm_all[n_pre:], index=t_post.index, name='gap_ascm')

    return {
        'weights_scm': weights_scm,
        'ridge_coef': pd.Series(ridge.coef_, index=d_pre.columns, name='ridge_coef'),
        'ridge_intercept': float(ridge.intercept_),
        'ridge_alpha': ridge_alpha,
        'gap_post': gap_post,
        'gap_ascm_post_end': float(gap_post.iloc[-1]),
        'synthetic_ascm': pd.DataFrame({'actual': Y1_all[n_pre:], 'ascm_cf': cf_ascm_all[n_pre:]},
                                       index=t_post.index),
        'mspe_pre_scm': mspe_pre,
    }


# ════════════════════════════════════════════════════════════════
# 区3: CausalImpact(statsmodels.UnobservedComponents 封装 = BSTS 频率派等价)
# ════════════════════════════════════════════════════════════════

def causal_impact_statsmodels(y, exog, pre_period, post_period,
                              level='local level', seasonal=None):
    """CausalImpact 的 statsmodels 实现(BSTS 频率派对应物, Brodersen 2015 思路)。

    用预处理期拟合 局部水平 + 协变量回归 的状态空间模型, post 期做反事实预测,
    effect = 实际 - 反事实, p_value = 实际落在预测 CI 外的比例。

    参数:
        y : pd.Series     被干预序列(建议累计对数收益)
        exog : pd.DataFrame 协变量(VIX/美债/donor加权), index 对齐 y
        pre_period : (start, end)   干预前区间
        post_period : (start, end)  干预期区间
        level : 'local level' / 'local linear trend' / 'random walk'
        seasonal : int/None 季节周期(美股日频可设 5)
    返回: dict(counterfactual, ci_lower, ci_upper, effect, cumulative_effect, p_value, summary)
    """
    from statsmodels.tsa.statespace.structural import UnobservedComponents as UC
    y_pre = y.loc[pre_period[0]:pre_period[1]].dropna()
    exog_pre = exog.loc[y_pre.index].fillna(method='ffill').fillna(0)
    exog_post = exog.loc[post_period[0]:post_period[1]].fillna(method='ffill').fillna(0)
    y_post = y.loc[post_period[0]:post_period[1]]

    mod = UC(y_pre, level=level, seasonal=seasonal, exog=exog_pre)
    res = mod.fit(disp=False)
    fc = res.get_forecast(steps=len(exog_post), exog=exog_post)
    cf = fc.predicted_mean.copy()
    cf.index = y_post.index
    ci = fc.conf_int(alpha=0.05)
    ci.index = y_post.index

    effect = y_post - cf
    cum = effect.cumsum()
    # p_value: 实际落在 95% CI 外的比例(tail-area 近似)
    out_of_ci = (y_post < ci.iloc[:, 0]) | (y_post > ci.iloc[:, 1])
    p = float(out_of_ci.mean())

    return {
        'counterfactual': cf, 'ci_lower': ci.iloc[:, 0], 'ci_upper': ci.iloc[:, 1],
        'effect': effect, 'cumulative_effect': cum, 'p_value': p,
        'summary': f'Avg effect={effect.mean():+.4f}  Cumulative={cum.iloc[-1]:+.4f}  '
                   f'p(CI外)={p:.2%}  (pre MSPE via resid var)',
    }


def causal_impact_tfci(data, pre_period, post_period, fit_method='vi'):
    """tfcausalimpact 封装(可选, 需 pip install tfcausalimpact)。

    自动设 TF_USE_LEGACY_KERAS=1 避 Keras3 坑。data 含 [y] + 协变量列。
    """
    import os
    os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
    try:
        from causalimpact import CausalImpact
    except ImportError:
        raise ImportError('未装 tfcausalimpact, 跑: pip install tfcausalimpact '
                          'tensorflow==2.16.1 tensorflow-probability==0.24.0')
    ci = CausalImpact(data, pre_period, post_period,
                      model_args={'fit_method': fit_method})
    return ci   # .summary() / .summary(output='report') / .plot()


# ════════════════════════════════════════════════════════════════
# 区4: 事件研究(MacKinlay 1997)
# ════════════════════════════════════════════════════════════════

def estimate_market_model(y_est, mkt_est):
    """用估计窗拟合市场模型 R_y = α + β R_mkt + ε(OLS)。

    y_est/mkt_est: pd.Series 对数收益(估计窗=事件前, 如 [-120,-21])。
    返回 (alpha, beta, resid_std)。
    """
    df = pd.concat([y_est, mkt_est], axis=1).dropna()
    y = df.iloc[:, 0].values
    x = df.iloc[:, 1].values
    n = len(y)
    beta, alpha = np.polyfit(x, y, 1)   # 一元 OLS
    resid = y - (alpha + beta * x)
    resid_std = resid.std(ddof=2)
    return float(alpha), float(beta), float(resid_std)


def car(y_event, mkt_event, alpha, beta):
    """事件窗 CAR = Σ (R_y - α - β R_mkt)。返回 (CAR, 每日 AR pd.Series)。"""
    ar = y_event - (alpha + beta * mkt_event)
    return float(ar.sum()), ar


def car_t_stat(car_value, resid_std, L):
    """CAR 的 t 统计量(假设 AR 独立同分布, 标准差 = resid_std * sqrt(L))。"""
    se = resid_std * np.sqrt(L)
    return car_value / se if se > 0 else np.nan


# ════════════════════════════════════════════════════════════════
# 区5: 反事实评估指标
# ════════════════════════════════════════════════════════════════

def cumulative_gap(gap_post):
    """post 期累计缺口(对数收益累加)。gap_post 已是累计对数收益差时即终点值。"""
    return float(gap_post.iloc[-1])


def percent_effect(obs_final, syn_final):
    """百分比效应 = (实际 - 合成) / 合成 × 100%。"""
    return (obs_final - syn_final) / abs(syn_final) * 100.0 if syn_final != 0 else np.nan


def mspe_ratio(mspe_pre, mspe_post):
    """Abadie MSPE 比 = post / pre(越大越显著, placebo 分布用此排序)。"""
    return mspe_post / mspe_pre if mspe_pre > 0 else np.inf


def pre_trend_test(gap_pre):
    """预处理期缺口均值是否≈0(平行趋势检验)。

    用 Newey-West HAC 标准误(对时序自相关稳健), 而非普通 t 检验——时序 gap 有正自相关时,
    普通 t 检验低估 SE、p 偏小。返回 (t_stat, p_value)。
    """
    import statsmodels.api as sm
    g = gap_pre.dropna().values
    if len(g) < 5:
        return float('nan'), float('nan')
    maxlags = max(1, int(np.floor(4 * (len(g) / 100) ** (2 / 9))))  # Newey-West 1994 lag 选择
    res = sm.OLS(g, np.ones(len(g))).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    return float(res.tvalues[0]), float(res.pvalues[0])


# ════════════════════════════════════════════════════════════════
# 区6: 稳健性(placebo / leave-one-out / permutation)
# ════════════════════════════════════════════════════════════════

def placebo_test(price_treated, price_donors, event_date, pre_start,
                 post_end=None, **pipeline_kwargs):
    """安慰剂检验: 对每个 donor 当干预组跑一遍 SCM, 得 placebo 缺口零分布。

    逻辑: 真干预组(标普500)的 post 缺口应显著大于"假干预组(donor 轮流当 treated)"。
    返回:
        placebo_gaps: pd.DataFrame 每列一个 placebo 的 post 缺口序列
        observed_gap : 真实干预组的 post 缺口(对比基准)
        p_value      : permutation p = placebo 终点|gap| >= 观察值的比例
    """
    cols = list(price_donors.columns)
    gaps = {}
    for c in cols:
        # c 当 treated, 其余 donor + 原 treated 当 donor pool
        others = price_donors.drop(columns=[c])
        pool = others.join(price_treated.rename('__TREATED__'))
        try:
            res = fit_scm_pipeline(price_donors[c], pool, event_date,
                                   pre_start, post_end, **pipeline_kwargs)
            gaps[c] = res['gap_post']
        except Exception as e:
            print(f'  placebo {c} 失败: {e}')
    placebo_gaps = pd.DataFrame(gaps)

    # 真实干预组
    res_real = fit_scm_pipeline(price_treated, price_donors, event_date,
                                pre_start, post_end, **pipeline_kwargs)
    observed_gap = res_real['gap_post']

    p = permutation_p_value(observed_gap, placebo_gaps)
    return {
        'placebo_gaps': placebo_gaps,
        'observed_gap': observed_gap,
        'p_value': p,
        'real_result': res_real,
    }


def leave_one_out(price_treated, price_donors, event_date, pre_start,
                  post_end=None, **pipeline_kwargs):
    """逐个剔除 donor 重拟合, 看权重和缺口对单个 donor 的敏感性。

    返回:
        weights_loo: pd.DataFrame 每行一次剔除的权重向量
        gaps_loo   : pd.DataFrame 每列一次剔除的 post 终点缺口
    """
    cols = list(price_donors.columns)
    w_rows, gap_ends = {}, {}
    for c in cols:
        pool = price_donors.drop(columns=[c])
        try:
            res = fit_scm_pipeline(price_treated, pool, event_date,
                                   pre_start, post_end, **pipeline_kwargs)
            w = res['weights'].reindex(cols).fillna(0.0)
            w_rows[f'drop_{c}'] = w
            gap_ends[f'drop_{c}'] = res['gap_post'].iloc[-1]
        except Exception as e:
            print(f'  LOO drop {c} 失败: {e}')
    return pd.DataFrame(w_rows).T, pd.Series(gap_ends, name='post_endpoint_gap')


def permutation_p_value(observed_gap_post, placebo_gaps_post):
    """Abadie permutation p-value = placebo 终点|gap| >= 观察值 的比例。

    observed_gap_post : pd.Series 真实干预期缺口
    placebo_gaps_post : pd.DataFrame 每列一个 placebo 的缺口序列
    返回: p_value (float, [0,1])
    """
    obs = abs(observed_gap_post.iloc[-1])
    plac = placebo_gaps_post.iloc[-1].abs()
    return float((plac >= obs).sum()) / len(plac) if len(plac) > 0 else np.nan


def causal_impact_placebo_intime(y, exog, pre_start, real_event, post_end,
                                 n_placebo=10, min_pre=60,
                                 level='local level', seasonal=None):
    """CausalImpact 的 in-time placebo(时间置换): 干预点滚到 pre 期多处, 得假 effect 零分布。

    动机: causal_impact_statsmodels 的 p_value 是"实际落在 95% CI 外的比例"(贝叶斯 tail-area),
    与 SCM 的 permutation_p_value(pre 期零分布分位)不同口径。本函数把 CI 的干预点前移到 pre 期,
    构造与 SCM permutation 同口径的 placebo p: 真实 effect 在"假干预点 effect 分布"中的分位。

    逻辑:
        - L = 真实 post 长度(real_event~post_end)
        - 在 [pre_start + min_pre, real_event - L] 均匀取 n_placebo 个假干预点 t0
          (留 min_pre 天拟合 + 假 post L 天不越 real_event, 不偷看真实 post)
        - 每点: 用 pre_start~t0 拟合 BSTS, out-of-sample 预测 t0+1 ~ t0+L 的反事实,
          假 effect 终点 = 该窗口实际 - 反事实
        - p_value = 假|effect| >= 真实|effect| 的比例

    参数:
        y : pd.Series      被干预序列(累计对数收益, 全期 pre+post)
        exog : pd.DataFrame 协变量(index 对齐 y)
        pre_start : 干预期起点(也是所有 placebo 的 pre 起点)
        real_event: 真实干预日
        post_end  : 真实 post 终点(决定 L 和真实 effect)
        n_placebo : 假干预点数
        min_pre   : 假干预点前最少拟合天数(防 pre 太短 BSTS 拟不稳)
        level/seasonal: 传给 causal_impact_statsmodels
    返回: dict(fake_effects np.array, real_effect float, p_value float, n_placebo, post_length L)
    """
    pre_start = pd.Timestamp(pre_start)
    real_event = pd.Timestamp(real_event)
    post_end = pd.Timestamp(post_end)
    y = y.sort_index()

    real_post = y.loc[real_event:post_end]
    L = len(real_post)
    if L < 2:
        raise ValueError(f'post 期太短(L={L})')

    pre_start_pos = y.index.get_indexer([pre_start], method='bfill')[0]
    real_event_pos = y.index.get_indexer([real_event], method='bfill')[0]
    earliest_pos = pre_start_pos + min_pre
    latest_pos = real_event_pos - L - 1      # t0 最大位置: fake_post_end(pos+L) <= real_event_pos - 1
    if latest_pos <= earliest_pos:
        raise ValueError(f'pre 期放不下 placebo(min_pre={min_pre}, L={L}, '
                         f'pre 长度={real_event_pos - pre_start_pos})')

    picks = np.linspace(earliest_pos, latest_pos, n_placebo).round().astype(int)
    picks = sorted(set(picks.tolist()))

    fake_effects = []
    for pos in picks:
        t0 = y.index[pos]
        fake_post_start = y.index[pos + 1]
        fake_post_end = y.index[pos + L]
        try:
            res = causal_impact_statsmodels(
                y, exog,
                pre_period=(pre_start, t0),
                post_period=(fake_post_start, fake_post_end),
                level=level, seasonal=seasonal)
            fake_effects.append(float(res['effect'].iloc[-1]))
        except Exception as e:
            print(f'  in-time placebo t0={t0.date()} 失败: {type(e).__name__}: {str(e)[:80]}')

    # 真实 effect(real_event 前一天为止拟合, real_event~post_end 预测, 同 nb04 调用口径)
    real_pre_end = y.index[real_event_pos - 1]
    real_res = causal_impact_statsmodels(
        y, exog,
        pre_period=(pre_start, real_pre_end),
        post_period=(real_event, post_end),
        level=level, seasonal=seasonal)
    real_effect = float(real_res['effect'].iloc[-1])

    fake_arr = np.asarray(fake_effects, dtype=float)
    p = float((np.abs(fake_arr) >= abs(real_effect)).mean()) if len(fake_arr) > 0 else np.nan

    return {
        'fake_effects': fake_arr,
        'real_effect': real_effect,
        'p_value': p,
        'n_placebo': len(fake_arr),
        'post_length': L,
        'real_ci_p': float(real_res['p_value']),   # CI 原始贝叶斯 p(对照)
    }


# ════════════════════════════════════════════════════════════════
# 区7: metrics 收编 + 绘图
# ════════════════════════════════════════════════════════════════

def metrics(ret, name='', rf=0.0):
    """统一绩效指标(年化/Sharpe/最大回撤/Calmar)。

    收编自 generate_day34.py:118 / multi_factor_v2_compare.py:101 两版,
    加 rf 无风险利率(论文事件研究可用美债^TNX均值, 默认0)。
    """
    ret = pd.Series(ret).dropna()
    nav = (1 + ret).cumprod()
    n = len(ret)
    ann = nav.iloc[-1] ** (TRADING_DAYS / n) - 1 if n > 0 else 0.0
    vol = ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann - rf) / vol if vol > 0 else 0.0
    dd = (nav / nav.cummax() - 1).min()
    calmar = ann / abs(dd) if dd < 0 else 0.0
    return dict(name=name, ann_ret=ann, sharpe=sharpe, max_dd=dd, calmar=calmar)


def plot_counterfactual(cumlog_treated, cumlog_donors, weights, gap_post,
                        event_date, title='', ax=None):
    """SCM 通用绘图: 上图实际 vs 合成(+ donor 加权), 下图缺口。

    cumlog_treated/cumlog_donors: 全期累计对数收益(pre+post, 已归一化)
    weights: pd.Series donor 权重
    gap_post: pd.Series post 缺口
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                       gridspec_kw={'height_ratios': [2, 1]})
    else:
        fig, ax1, ax2 = ax, ax[0], ax[1]

    # 拼接 pre + post 的 cumlog(调用方负责构造全期序列)
    synthetic = (cumlog_donors * weights).sum(axis=1)
    ax1.plot(cumlog_treated.index, cumlog_treated.values, lw=2,
             label='Actual (S&P 500)', color='firebrick')
    ax1.plot(synthetic.index, synthetic.values, lw=2, ls='--',
             label='Synthetic counterfactual', color='steelblue')
    ax1.axvline(pd.Timestamp(event_date), color='gray', ls=':', lw=1.5)
    ax1.set_ylabel('Cumulative log return'); ax1.legend(loc='best'); ax1.grid(alpha=0.3)
    ax1.set_title(title or 'S&P 500: Actual vs Synthetic')

    ax2.fill_between(gap_post.index, gap_post.values * 100, 0,
                     where=gap_post.values >= 0, color='firebrick', alpha=0.5,
                     label='loss from tariff')
    ax2.fill_between(gap_post.index, gap_post.values * 100, 0,
                     where=gap_post.values < 0, color='seagreen', alpha=0.5,
                     label='gain')
    ax2.axvline(pd.Timestamp(event_date), color='gray', ls=':', lw=1.5)
    ax2.axhline(0, color='k', lw=0.5)
    ax2.set_ylabel('Gap (cum log ×100)'); ax2.legend(loc='best'); ax2.grid(alpha=0.3)
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════
# 自检(直接跑本文件时做 smoke test)
# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('=== paper_lib smoke test ===')
    panels = load_tariff_data()
    print('面板:', {k: v.shape for k, v in panels.items()})
    events = load_event_dates()
    print(f'事件: {len(events)} 个, 范围 {events.date.min().date()} ~ {events.date.max().date()}')

    # SCM 主结果(2025 解放日, 行业 ETF donor)
    res = fit_scm_pipeline(panels['us_index']['^GSPC'], panels['sector_etf'],
                           event_date='2025-04-02', pre_start='2024-06-01',
                           post_end='2025-06-30')
    print('\n--- SCM 2025-04-02 (行业 ETF donor) ---')
    print('权重 Top5:')
    print(res['weights'].head().round(4).to_string())
    print(f"MSPE pre={res['mspe_pre']:.6e}  post={res['mspe_post']:.6e}  ratio={res['mspe_ratio']:.1f}")
    print(f"post 终点累计缺口={res['gap_post'].iloc[-1]*100:+.2f}% (累计对数收益)")
    t, p = pre_trend_test(res['gap_pre'])
    print(f'平行趋势: pre 缺口 t={t:.2f} p={p:.3f} (p>0.1 才算平行)')

    # Augmented SCM 对照(同一窗口, 逐窗口算 SCM 基准)
    print('\n--- Augmented SCM 2025-04-02 (行业 ETF donor, ridge α=1) ---')
    for end in ['2025-04-08', '2025-06-30']:
        scm_end = fit_scm_pipeline(panels['us_index']['^GSPC'], panels['sector_etf'],
                                   event_date='2025-04-02', pre_start='2024-06-01', post_end=end)
        ar = fit_ascm_pipeline(panels['us_index']['^GSPC'], panels['sector_etf'],
                               event_date='2025-04-02', pre_start='2024-06-01', post_end=end)
        print(f"  post {end}: SCM={scm_end['gap_post'].iloc[-1]*100:+.2f}%  "
              f"ASCM(α=1)={ar['gap_ascm_post_end']*100:+.2f}%  "
              f"ridge 负权重数={int((ar['ridge_coef']<0).sum())}/{len(ar['ridge_coef'])}")

    print('\n✅ paper_lib smoke test 通过')
