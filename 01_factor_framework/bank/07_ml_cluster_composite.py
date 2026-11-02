"""07 簇合成因子 + 机器学习对照 — ML到底加不加值?

02分出8簇, PCA说有效维度5-7(吻合)。本脚本把每簇合成1个综合因子(符号对齐后平均), 喂给4种模型:
  - baseline: 8合成因子 ICIR加权 (无ML, 要打败的基线)
  - Lasso:    线性稀疏 (自动选哪些簇重要)
  - PLS:      监督式降维 (用收益找成分, 比PCR更适合预测)
  - XGBoost:  非线性 (极浅+强正则, 小样本上易过拟合)

⚠️ 关键: 簇内因子alpha方向常相反(beta60- vs market_cap+), 必须先按滚动IC符号翻转再平均, 否则正负抵消。
全部walk-forward(滚动36月训练→预测下季), 季度调仓, 银行Q5等权, 无超跌过滤。
预期: 小样本+线性信号 → ML大概率打不过baseline; 若真打过, 报OOS增益; 若in-sample好OOS差 → 过拟合铁证。

输入: data/(factors_monthend, mainboard_price, universe_monthend_pit)
输出: outputs/ml_compare.csv (各自月份 + 共同月份两种口径)
修正记录:
  - LassoCV/ElasticNetCV 的 cv 从随机KFold改成按月分fold — 堆叠截面里同月样本
    必须落在同一fold, 否则超参选择泄露跨月信息
  - 各模型被跳过的季度不同(Lasso预测退化时nunique<5) → 直接比年化是不同窗口;
    增加"共同月份"口径(所有模型季度交集), 报结论以它为准
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LassoCV, ElasticNetCV, Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

DATA = '/Users/lichanghaodemac/quant_strategy/data'
OUT = '/Users/lichanghaodemac/quant_strategy/projects/bank_factor_workflow/outputs'

CLUSTERS = {  # 来自02层次聚类
    'C1_wqalpha': ['WorldQuant_alpha013', 'WorldQuant_alpha016'],
    'C2_sizebeta': ['beta60', 'beta250', 'market_cap_3', 'a_share_market_val_in_circulation', 'OBV'],
    'C3_value': ['dividend_yield_ttm', 'pe_ratio_ttm', 'pe_ratio_lyr'],
    'C4_quality': ['ev_to_ebitda_ttm', 'net_profit_margin_ttm', 'return_on_equity_ttm'],
    'C5_mom': ['MOM120', 'MOM_12M_1M'],
    'C6_davol': ['DAVOL20'],
    'C7_a026': ['WorldQuant_alpha026'],
    'C8_a088': ['WorldQuant_alpha088'],
}
ALLF = sorted({f for v in CLUSTERS.values() for f in v})

# === 数据加载 ===
price = pd.read_parquet(f'{DATA}/mainboard_price.parquet'); price['date'] = pd.to_datetime(price['date'])
close = price.pivot(index='date', columns='order_book_id', values='close').sort_index()
me = pd.DatetimeIndex(pd.read_parquet(f'{DATA}/monthend_dates.parquet')['monthend'])
mev = [m for m in me if m in close.index]; cm = close.loc[mev]
fwd1 = cm.pct_change(fill_method=None).shift(-1); fwd3 = cm.shift(-3) / cm - 1
def beta_p(n):
    rd = close.pct_change(fill_method=None); mk = rd.mean(axis=1)
    return rd.rolling(n).cov(mk).div(mk.rolling(n).var(), axis=0)
self_mats = {'MOM120': close.pct_change(120, fill_method=None), 'MOM_12M_1M': close.shift(20) / close.shift(240) - 1,
             'beta60': beta_p(60), 'beta250': beta_p(250)}
factors = pd.read_parquet(f'{DATA}/factors_monthend.parquet'); factors['monthend'] = pd.DatetimeIndex(factors['monthend'])
univ = pd.read_parquet(f'{DATA}/universe_monthend_pit.parquet'); univ['monthend'] = pd.DatetimeIndex(univ['monthend'])
fromdb = [f for f in ALLF if f in factors.columns]
allf = factors[['order_book_id', 'monthend'] + fromdb].copy()
for n, s in self_mats.items():
    sl = s.loc[[m for m in mev if m in s.index]].stack().rename(n).reset_index(); sl.columns = ['monthend', 'order_book_id', n]
    allf = allf.merge(sl, on=['order_book_id', 'monthend'], how='outer')
allf = allf.merge(univ[univ['industry_l1'] == '银行'][['order_book_id', 'monthend']], on=['order_book_id', 'monthend'], how='inner')
allf = allf[['order_book_id', 'monthend'] + ALLF].sort_values(['monthend', 'order_book_id']).reset_index(drop=True)
allf[ALLF] = allf[ALLF].replace([np.inf, -np.inf], np.nan)   # 清inf(alpha026)
me_sorted = sorted(allf['monthend'].unique())
WIDE = {f: allf.pivot(index='monthend', columns='order_book_id', values=f) for f in ALLF}
stks = np.array(WIDE[ALLF[0]].columns)

def zs(mat):  # 截面z(winsorize1/99)
    lo, hi = mat.quantile(0.01, axis=1), mat.quantile(0.99, axis=1); o = mat.copy()
    for c in mat.columns: o[c] = o[c].clip(lo, hi, axis=0)
    return o.sub(o.mean(axis=1), axis=0).div(o.std(axis=1).replace(0, np.nan), axis=0)
Z = {f: zs(WIDE[f]) for f in ALLF}

# 各因子滚动IC(1m) → 用于符号对齐 & ICIR
IC1 = {f: pd.Series({m: (lambda d: spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0] if len(d) >= 10 and d.iloc[:, 0].nunique() > 1 else np.nan)
                     (pd.concat([Z[f].loc[m], fwd1.loc[m]], axis=1).dropna()) for m in me_sorted if m in fwd1.index}) for f in ALLF}

def cluster_composite(m, past):  # 簇内: 符号对齐(滚动IC均值的符号) → 平均
    comp = {}
    for cname, facs in CLUSTERS.items():
        vals = []
        for f in facs:
            v = IC1[f].reindex(past).dropna()
            sgn = np.sign(v.mean()) if len(v) > 6 else 1.0
            vals.append(sgn * Z[f].loc[m])
        comp[cname] = pd.concat(vals, axis=1).mean(axis=1)
    return pd.DataFrame(comp).reindex(stks)

# walk-forward: 季度点, 每点用过去36月训练
# ★ 2026-08-15: 季度起点对齐04/06的2020-12(面板2017-01的第47个月=2020-12-31), 原2020-01起跑
qstart = pd.Timestamp('2020-12-01')
qi = next(i for i, m in enumerate(me_sorted) if m >= qstart)
quarterly = [m for i, m in enumerate(me_sorted) if i >= qi and (i - qi) % 3 == 0]

def stack_window(months, fwd):
    X, y, idx = [], [], []
    for m in months:
        if m not in fwd.index: continue
        comp = cluster_composite(m, [mm for mm in me_sorted if mm < m][-36:])
        for s in stks:
            xv = comp.loc[s].values
            yv = fwd.loc[m, s] if s in fwd.columns else np.nan
            if not np.isnan(xv).any() and pd.notna(yv): X.append(xv); y.append(yv); idx.append((m, s))
    return np.array(X), np.array(y), idx

# baseline: 8个簇composite, ICIR加权(用各簇composite在训练窗对3m前瞻的IC)
def baseline_q5(m, past):
    comp = cluster_composite(m, past); W = []
    for c in CLUSTERS:
        ics = []
        for pm in past:
            if pm not in fwd3.index: continue
            cmp_pm = cluster_composite(pm, [mm for mm in me_sorted if mm < pm][-36:])[c]
            d = pd.concat([cmp_pm, fwd3.loc[pm]], axis=1).dropna()
            if len(d) >= 10 and d.iloc[:, 0].nunique() > 1: ics.append(spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0])
        v = np.array(ics); W.append(abs(v.mean() / v.std()) if len(v) > 6 and v.std() > 0 else 0)
    W = np.array(W); W = W / W.sum() if W.sum() > 0 else np.ones(len(CLUSTERS)) / len(CLUSTERS)
    return (comp.fillna(0) * W).sum(axis=1)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

def month_folds(idx, n_splits=5):
    """按月分fold(替代随机KFold): 同一月的截面样本留在同一fold, 否则超参选择泄露跨月信息"""
    months = sorted({m for m, _ in idx})
    blocks = np.array_split(months, n_splits)
    folds = []
    for b in blocks:
        bset = set(b)
        trn = np.array([i for i, (mm, _) in enumerate(idx) if mm not in bset])
        val = np.array([i for i, (mm, _) in enumerate(idx) if mm in bset])
        folds.append((trn, val))
    return folds

def predict_q5(model_name, m, past):
    comp = cluster_composite(m, past)
    Xtr, ytr, idx = stack_window(past, fwd3)
    if len(Xtr) < 200: return None
    xs = StandardScaler().fit(Xtr)
    Xtr_s = xs.transform(Xtr)
    xm = xs.transform(comp.loc[stks].fillna(0).values)
    if model_name == 'Lasso':
        mdl = LassoCV(cv=month_folds(idx), max_iter=5000).fit(Xtr_s, ytr); pred = mdl.predict(xm)
    elif model_name == 'ElasticNet':
        mdl = ElasticNetCV(cv=month_folds(idx), max_iter=5000, l1_ratio=[0.5, 0.8]).fit(Xtr_s, ytr); pred = mdl.predict(xm)
    elif model_name == 'PLS':
        mdl = PLSRegression(n_components=min(5, len(CLUSTERS) - 1)).fit(Xtr_s, ytr); pred = mdl.predict(xm).ravel()
    elif model_name == 'XGBoost':
        if not HAS_XGB: return None
        mdl = XGBRegressor(n_estimators=60, max_depth=2, learning_rate=0.05, subsample=0.7,
                           colsample_bytree=0.7, reg_lambda=5, reg_alpha=2, n_jobs=2).fit(Xtr_s, ytr)
        pred = mdl.predict(xm)
    return pd.Series(pred, index=stks)

models = ['baseline', 'Lasso', 'ElasticNet', 'PLS'] + (['XGBoost'] if HAS_XGB else [])
COST = 0.0015; mkt3 = fwd3.mean(axis=1)
rets_by_model = {}   # model -> {month: ret}  (各模型实际有信号的季度)
for mdl in models:
    mr = {}
    for m in quarterly:
        past = [mm for mm in me_sorted if mm < m][-36:]
        if len(past) < 36 or m not in fwd3.index: continue
        sc = baseline_q5(m, past) if mdl == 'baseline' else predict_q5(mdl, m, past)
        if sc is None or sc.dropna().nunique() < 5: continue
        sc = sc.dropna()
        try: q = pd.qcut(sc.rank(method='first'), 5, labels=False)
        except: continue
        q5 = sc[q == 4].index
        if len(q5) < 3: continue
        ret = np.mean([fwd3.loc[m, o] for o in q5 if o in fwd3.columns and pd.notna(fwd3.loc[m, o])]) - COST
        mr[m] = ret
    rets_by_model[mdl] = mr

def met(mr):
    r = pd.Series(mr).sort_index()
    if len(r) < 3: return None
    ann = r.mean() * 4; sh = r.mean() / r.std() * 2 if r.std() > 0 else 0
    cum = (1 + r).cumprod(); dd = (cum / cum.cummax() - 1).min()
    return {'季度数': len(r), '年化': round(ann, 4), '夏普(季)': round(sh, 2),
            '最大回撤': round(dd, 4), '期胜率': round((r > 0).mean(), 2)}

rows = []
for mdl, mr in rets_by_model.items():
    m_ = met(mr)
    if m_: rows.append({'模型': mdl, '口径': '各自月份', **m_})
# ★ 共同月份口径: 各模型跳过的季度不同, "各自月份"直接比年化是不同窗口 — 报结论以此为准
if len(rets_by_model) > 1:
    common = sorted(set.intersection(*[set(v) for v in rets_by_model.values()]))
    for mdl, mr in rets_by_model.items():
        m_ = met({mm: mr[mm] for mm in common if mm in mr})
        if m_: rows.append({'模型': mdl, '口径': f'共同月份({len(common)}季)', **m_})
res = pd.DataFrame(rows)[['模型', '口径', '季度数', '年化', '夏普(季)', '最大回撤', '期胜率']]
res.to_csv(f'{OUT}/ml_compare.csv', index=False, encoding='utf-8-sig')
print(f'XGBoost: {"可用" if HAS_XGB else "未装, 跳过"}', flush=True)
mw = mkt3.reindex([m for m in quarterly if m in mkt3.index]).dropna()
print(f'\n基准(全市场等权3m, 与策略同窗口{str(quarterly[0])[:10]}起): 年化{mw.mean()*4:.2%}/夏普{mw.mean()/mw.std()*2:.2f}'
      f' (旧口径全样本{mkt3.mean()*4:.2%}, 含暖机期勿再用)\n', flush=True)
print('=== 簇合成因子 → 各模型 walk-forward 季度回测 ===', flush=True)
print(res.to_string(index=False), flush=True)
print('\n存 outputs/ml_compare.csv', flush=True)
print('解读: 看ML能否打过baseline; 小样本下预期ML≈baseline, XGBoost若夏普虚高则过拟合', flush=True)
