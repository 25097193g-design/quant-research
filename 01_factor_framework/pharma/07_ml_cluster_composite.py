"""07 簇合成因子 + 机器学习对照 — 医药228只, ML会不会加分?

银行41只时ML≈baseline(截面太小, 线性信号喂不饱复杂模型)。
医药228只, N是银行5倍+ → ML可能真正加分。本脚本同bank 07:
  19簇每簇合成1综合因子(符号对齐后平均) → 4模型:
  - baseline: 19合成因子 ICIR加权 (无ML, 要打败的基线)
  - Lasso:   线性稀疏(自动选哪些簇重要)
  - ElasticNet: Lasso+Ridge混合
  - PLS:     监督式降维
  - XGBoost: 非线性(极浅+强正则, 防小样本过拟合)
walk-forward(滚动36月训练→预测下季), 季度调仓, Q5等权, 无超跌过滤(04/05已证医药不需要)。
预期: N大了ML可能打过baseline; 若in-sample好OOS差 → 过拟合铁证。

输出: outputs/ml_compare.csv
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

DATA = '/Users/lichanghaodemac/quant_strategy/data'
OUT = '/Users/lichanghaodemac/quant_strategy/projects/pharma_factor_workflow/outputs'

CLUSTERS = {  # 来自02层次聚类(24因子→19簇)
    'C1_cash_debt': ['ocf_to_debt_ttm'],
    'C2_surplus_cash': ['surplus_cash_protection_multiples_ttm'],
    'C3_volatility': ['VOLT20', 'VOLT60'],
    'C4_value': ['book_to_market_ratio_lf', 'pb_ratio_lf', 'ps_ratio_ttm'],
    'C5_dividend': ['dividend_yield_ttm'],
    'C6_pe': ['pe_ratio_lyr'],
    'C7_a024': ['WorldQuant_alpha024'],
    'C8_a037': ['WorldQuant_alpha037'],
    'C9_a006_014': ['WorldQuant_alpha006', 'WorldQuant_alpha014'],
    'C10_a040': ['WorldQuant_alpha040'],
    'C11_a003': ['WorldQuant_alpha003'],
    'C12_a027': ['WorldQuant_alpha027'],
    'C13_a050': ['WorldQuant_alpha050'],
    'C14_a013_016': ['WorldQuant_alpha013', 'WorldQuant_alpha016'],
    'C15_a044': ['WorldQuant_alpha044'],
    'C16_a015': ['WorldQuant_alpha015'],
    'C17_a055': ['WorldQuant_alpha055'],
    'C18_a088': ['WorldQuant_alpha088'],
    'C19_a011': ['WorldQuant_alpha011'],
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
allf = allf.merge(univ[univ['industry_l1'] == '医药生物'][['order_book_id', 'monthend']], on=['order_book_id', 'monthend'], how='inner')
allf = allf[['order_book_id', 'monthend'] + ALLF].sort_values(['monthend', 'order_book_id']).reset_index(drop=True)
allf[ALLF] = allf[ALLF].replace([np.inf, -np.inf], np.nan)   # 清inf(WQ alpha通病)
me_sorted = sorted(allf['monthend'].unique())
WIDE = {f: allf.pivot(index='monthend', columns='order_book_id', values=f) for f in ALLF}
stks = np.array(WIDE[ALLF[0]].columns)

def zs(mat):
    lo, hi = mat.quantile(0.01, axis=1), mat.quantile(0.99, axis=1); o = mat.copy()
    for c in mat.columns: o[c] = o[c].clip(lo, hi, axis=0)
    return o.sub(o.mean(axis=1), axis=0).div(o.std(axis=1).replace(0, np.nan), axis=0)
Z = {f: zs(WIDE[f]) for f in ALLF}

IC1 = {f: pd.Series({m: (lambda d: spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0] if len(d) >= 10 and d.iloc[:, 0].nunique() > 1 else np.nan)
                     (pd.concat([Z[f].loc[m], fwd1.loc[m]], axis=1).dropna()) for m in me_sorted if m in fwd1.index}) for f in ALLF}

def cluster_composite(m, past):
    comp = {}
    for cname, facs in CLUSTERS.items():
        vals = []
        for f in facs:
            v = IC1[f].reindex(past).dropna()
            sgn = np.sign(v.mean()) if len(v) > 6 else 1.0
            vals.append(sgn * Z[f].loc[m])
        comp[cname] = pd.concat(vals, axis=1).mean(axis=1)
    return pd.DataFrame(comp).reindex(stks)

quarterly = [m for i, m in enumerate(me_sorted) if i >= 36 and (i - 36) % 3 == 0]

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

def predict_q5(model_name, m, past):
    comp = cluster_composite(m, past)
    Xtr, ytr, _ = stack_window(past, fwd3)
    if len(Xtr) < 300: return None
    xs = StandardScaler().fit(Xtr)
    Xtr_s = xs.transform(Xtr)
    xm = xs.transform(comp.loc[stks].fillna(0).values)
    if model_name == 'Lasso':
        mdl = LassoCV(cv=5, max_iter=5000).fit(Xtr_s, ytr); pred = mdl.predict(xm)
    elif model_name == 'ElasticNet':
        mdl = ElasticNetCV(cv=5, max_iter=5000, l1_ratio=[0.5, 0.8]).fit(Xtr_s, ytr); pred = mdl.predict(xm)
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
rows = []
for mdl in models:
    rets = []
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
        rets.append({'m': m, 'ret': ret})
    if not rets: continue
    d = pd.DataFrame(rets).set_index('m'); r = d['ret']; n = len(r)
    ann = r.mean() * 4; sh = r.mean() / r.std() * 2 if r.std() > 0 else 0
    cum = (1 + r).cumprod(); dd = (cum / cum.cummax() - 1).min()
    rows.append({'模型': mdl, '季度数': n, '年化': round(ann, 4), '夏普(季)': round(sh, 2),
                 '最大回撤': round(dd, 4), '期胜率': round((r > 0).mean(), 2)})
res = pd.DataFrame(rows)
res.to_csv(f'{OUT}/ml_compare.csv', index=False, encoding='utf-8-sig')
print(f'XGBoost: {"可用" if HAS_XGB else "未装, 跳过"}', flush=True)
print(f'医药截面: {len(stks)}只; 基准(全市场等权3m): 年化{mkt3.mean()*4:.2%}\n', flush=True)
print('=== 19簇合成因子 → 各模型 walk-forward 季度回测 ===', flush=True)
print(res.to_string(index=False), flush=True)
print('\n存 outputs/ml_compare.csv', flush=True)
print('解读: 医药N=228(银行41的5倍), 看ML能否打过baseline; XGBoost夏普虚高则过拟合', flush=True)
