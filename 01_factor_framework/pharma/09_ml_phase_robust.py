"""09 ML相位稳健性检验 — XGBoost的+7pp是真加分还是相位运气?

07发现: baseline 12.2%/0.55, XGBoost 19.2%/0.60, 但Lasso/EN/PLS≈baseline。
只最灵活模型赢 = 过拟合警讯。本脚本: baseline vs XGBoost 在 3个季度起始相位(0/1/2)
各跑一遍 walk-forward。XGBoost在3相位都赢 → 真加分; 只赢1相位 → 相位lucky(不可信)。

输出: outputs/ml_phase_robust.csv
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

DATA = '/Users/lichanghaodemac/quant_strategy/data'
OUT = '/Users/lichanghaodemac/quant_strategy/projects/pharma_factor_workflow/outputs'

CLUSTERS = {  # 同07(24因子→19簇)
    'C1_cash_debt': ['ocf_to_debt_ttm'], 'C2_surplus_cash': ['surplus_cash_protection_multiples_ttm'],
    'C3_volatility': ['VOLT20', 'VOLT60'], 'C4_value': ['book_to_market_ratio_lf', 'pb_ratio_lf', 'ps_ratio_ttm'],
    'C5_dividend': ['dividend_yield_ttm'], 'C6_pe': ['pe_ratio_lyr'], 'C7_a024': ['WorldQuant_alpha024'],
    'C8_a037': ['WorldQuant_alpha037'], 'C9_a006_014': ['WorldQuant_alpha006', 'WorldQuant_alpha014'],
    'C10_a040': ['WorldQuant_alpha040'], 'C11_a003': ['WorldQuant_alpha003'], 'C12_a027': ['WorldQuant_alpha027'],
    'C13_a050': ['WorldQuant_alpha050'], 'C14_a013_016': ['WorldQuant_alpha013', 'WorldQuant_alpha016'],
    'C15_a044': ['WorldQuant_alpha044'], 'C16_a015': ['WorldQuant_alpha015'], 'C17_a055': ['WorldQuant_alpha055'],
    'C18_a088': ['WorldQuant_alpha088'], 'C19_a011': ['WorldQuant_alpha011'],
}
ALLF = sorted({f for v in CLUSTERS.values() for f in v})

price = pd.read_parquet(f'{DATA}/mainboard_price.parquet'); price['date'] = pd.to_datetime(price['date'])
close = price.pivot(index='date', columns='order_book_id', values='close').sort_index()
me = pd.DatetimeIndex(pd.read_parquet(f'{DATA}/monthend_dates.parquet')['monthend'])
mev = [m for m in me if m in close.index]; cm = close.loc[mev]
fwd1 = cm.pct_change(fill_method=None).shift(-1); fwd3 = cm.shift(-3) / cm - 1
factors = pd.read_parquet(f'{DATA}/factors_monthend.parquet'); factors['monthend'] = pd.DatetimeIndex(factors['monthend'])
univ = pd.read_parquet(f'{DATA}/universe_monthend_pit.parquet'); univ['monthend'] = pd.DatetimeIndex(univ['monthend'])
fromdb = [f for f in ALLF if f in factors.columns]
allf = factors[['order_book_id', 'monthend'] + fromdb].copy()
allf = allf.merge(univ[univ['industry_l1'] == '医药生物'][['order_book_id', 'monthend']], on=['order_book_id', 'monthend'], how='inner')
allf = allf[['order_book_id', 'monthend'] + ALLF].sort_values(['monthend', 'order_book_id']).reset_index(drop=True)
allf[ALLF] = allf[ALLF].replace([np.inf, -np.inf], np.nan)
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
            v = IC1[f].reindex(past).dropna(); sgn = np.sign(v.mean()) if len(v) > 6 else 1.0
            vals.append(sgn * Z[f].loc[m])
        comp[cname] = pd.concat(vals, axis=1).mean(axis=1)
    return pd.DataFrame(comp).reindex(stks)

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

def xgb_q5(m, past):
    comp = cluster_composite(m, past); Xtr, ytr = [], []
    for pm in past:
        if pm not in fwd3.index: continue
        cmp_pm = cluster_composite(pm, [mm for mm in me_sorted if mm < pm][-36:])
        for s in stks:
            xv = cmp_pm.loc[s].values; yv = fwd3.loc[pm, s] if s in fwd3.columns else np.nan
            if not np.isnan(xv).any() and pd.notna(yv): Xtr.append(xv); ytr.append(yv)
    Xtr, ytr = np.array(Xtr), np.array(ytr)
    if len(Xtr) < 300: return None
    xs = StandardScaler().fit(Xtr)
    mdl = XGBRegressor(n_estimators=60, max_depth=2, learning_rate=0.05, subsample=0.7,
                       colsample_bytree=0.7, reg_lambda=5, reg_alpha=2, n_jobs=2).fit(xs.transform(Xtr), ytr)
    return pd.Series(mdl.predict(xs.transform(comp.loc[stks].fillna(0).values)), index=stks)

COST = 0.0015
def walk(quarterly, scorer):
    rets = []
    for m in quarterly:
        past = [mm for mm in me_sorted if mm < m][-36:]
        if len(past) < 36 or m not in fwd3.index: continue
        sc = scorer(m, past)
        if sc is None or sc.dropna().nunique() < 5: continue
        sc = sc.dropna()
        try: q = pd.qcut(sc.rank(method='first'), 5, labels=False)
        except: continue
        q5 = sc[q == 4].index
        if len(q5) < 3: continue
        r = np.mean([fwd3.loc[m, o] for o in q5 if o in fwd3.columns and pd.notna(fwd3.loc[m, o])]) - COST
        rets.append(r)
    return np.array(rets)

rows = []
for ph in [0, 1, 2]:
    quarterly = [m for i, m in enumerate(me_sorted) if i >= 36 and (i - 36) % 3 == ph]
    print(f'>>> phase={ph}: {len(quarterly)}个季度点', flush=True)
    for name, scorer in [('baseline', baseline_q5), ('XGBoost', xgb_q5)]:
        print(f'  ... {name} walk-forward (约{len(quarterly)}次模型/ICIR计算)', flush=True)
        r = walk(quarterly, scorer)
        n = len(r)
        if n < 5: continue
        r = pd.Series(r); ann = r.mean() * 4; sh = r.mean() / r.std() * 2 if r.std() > 0 else 0
        cum = (1 + r).cumprod(); dd = (cum / cum.cummax() - 1).min()
        rows.append({'模型': name, 'phase': ph, '季度数': n, '年化': round(ann, 4),
                     '夏普(季)': round(sh, 2), '最大回撤': round(dd, 4)})
res = pd.DataFrame(rows)
# 每个模型跨3相位的年化离散度
for name in ['baseline', 'XGBoost']:
    a = res[res.模型 == name]['年化']
    if len(a) >= 2: res = pd.concat([res, pd.DataFrame([{'模型': name, 'phase': '离散(max−min)', '年化': round(max(a) - min(a), 4)}])])
res.to_csv(f'{OUT}/ml_phase_robust.csv', index=False, encoding='utf-8-sig')
print('=== ML 3相位稳健性 (基线 vs XGBoost) ===', flush=True)
print(res.to_string(index=False), flush=True)
print('\nXGBoost在3相位全赢且离散小 → 真加分; 只1相位赢 → 相位lucky', flush=True)
print('存 outputs/ml_phase_robust.csv', flush=True)
