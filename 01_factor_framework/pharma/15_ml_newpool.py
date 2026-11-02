"""15 重做Stage B — 新池(core7+sat5=12F)簇合成 ML 对照: baseline/Lasso/PLS/XGBoost

设计(2026-08-16批准的B阶段): 全部干净口径——
  - 簇合成: 簇内符号=过去36月IC1均值符号(滚动, sat5翻面自动适应) → 平均 → 9簇复合因子
    (簇结构沿02裁剪: core7各单簇 + {bm,pb,ps} + {VOLT20,VOLT60}; a026不在旧聚类里,保守单簇)
  - embargo(12审计结论): 训练标签与baseline权重窗 = me_sorted[i-38:i-2](36个月, 决策时点全部已实现)
  - 季频3相位(i>=36, (i-36)%3==ph), qcut5 Q5等权, COST 0.15%/期
  - 对照 = Stage A 赢家 core7+sat5/混合透明版(中位11.5%/+6.5pp)
输出: outputs/redo_stageB_ml.csv
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LassoCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

DATA = '/Users/lichanghaodemac/quant_strategy/data'
OUT = '/Users/lichanghaodemac/quant_strategy/projects/pharma_factor_workflow/outputs'
COST = 0.0015

CLUSTERS = {  # 12F → 9簇 (core7单簇×7 + 价值簇 + 波动簇)
    'a040': ['WorldQuant_alpha040'], 'a016': ['WorldQuant_alpha016'], 'a024': ['WorldQuant_alpha024'],
    'a044': ['WorldQuant_alpha044'], 'a011': ['WorldQuant_alpha011'], 'a026': ['WorldQuant_alpha026'],
    'dividend': ['dividend_yield_ttm'],
    'value': ['book_to_market_ratio_lf', 'pb_ratio_lf', 'ps_ratio_ttm'],
    'vol': ['VOLT20', 'VOLT60'],
}
ALLF = sorted({f for v in CLUSTERS.values() for f in v})

# ===== 面板(同14口径) =====
price = pd.read_parquet(f'{DATA}/mainboard_price.parquet'); price['date'] = pd.to_datetime(price['date'])
close = price.pivot(index='date', columns='order_book_id', values='close').sort_index()
me = pd.DatetimeIndex(pd.read_parquet(f'{DATA}/monthend_dates.parquet')['monthend'])
mev = [m for m in me if m in close.index]; cm = close.loc[mev]
fwd1 = cm.pct_change(fill_method=None).shift(-1); fwd3 = cm.shift(-3) / cm - 1
factors = pd.read_parquet(f'{DATA}/factors_monthend.parquet'); factors['monthend'] = pd.DatetimeIndex(factors['monthend'])
univ = pd.read_parquet(f'{DATA}/universe_monthend_pit.parquet'); univ['monthend'] = pd.DatetimeIndex(univ['monthend'])
allf = factors[['order_book_id', 'monthend'] + ALLF]
allf = allf.merge(univ[univ['industry_l1'] == '医药生物'][['order_book_id', 'monthend']], on=['order_book_id', 'monthend'], how='inner')
allf = allf[['order_book_id', 'monthend'] + ALLF].sort_values(['monthend', 'order_book_id']).reset_index(drop=True)
allf[ALLF] = allf[ALLF].replace([np.inf, -np.inf], np.nan)
me_sorted = sorted(allf['monthend'].unique())
WIDE = {f: allf.pivot(index='monthend', columns='order_book_id', values=f) for f in ALLF}
stks = np.array(WIDE[ALLF[0]].columns)
IDX = {m: i for i, m in enumerate(me_sorted)}

def zs(mat):
    lo, hi = mat.quantile(0.01, axis=1), mat.quantile(0.99, axis=1); o = mat.copy()
    for c in mat.columns: o[c] = o[c].clip(lo, hi, axis=0)
    return o.sub(o.mean(axis=1), axis=0).div(o.std(axis=1).replace(0, np.nan), axis=0)
Z = {f: zs(WIDE[f]) for f in ALLF}
IC1 = {f: pd.Series({m: (lambda d: spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0] if len(d) >= 10 and d.iloc[:, 0].nunique() > 1 else np.nan)
                     (pd.concat([Z[f].loc[m], fwd1.loc[m]], axis=1).dropna()) for m in me_sorted if m in fwd1.index}) for f in ALLF}

# 同月基准
ids = univ[univ['industry_l1'] == '医药生物'].groupby('monthend')['order_book_id'].apply(set).to_dict()
ph3 = pd.Series({m: fwd3.loc[m, [o for o in ss if o in fwd3.columns]].dropna().mean() for m, ss in ids.items() if m in fwd3.index}).dropna()
mkt3 = fwd3.mean(axis=1)

# ===== 簇复合因子缓存(符号=过去36月IC1均值, 只依赖pm) =====
_comp = {}
def composite(pm):
    if pm in _comp: return _comp[pm]
    i = IDX[pm]; past = me_sorted[max(0, i - 36):i]
    comp = {}
    for cname, facs in CLUSTERS.items():
        vals = []
        for f in facs:
            v = IC1[f].reindex(past).dropna(); sgn = np.sign(v.mean()) if len(v) > 6 else 1.0
            vals.append(sgn * Z[f].loc[pm])
        comp[cname] = pd.concat(vals, axis=1).mean(axis=1)
    df = pd.DataFrame(comp).reindex(stks); _comp[pm] = df; return df

_sp3 = {}
def sp3(pm):
    if pm in _sp3: return _sp3[pm]
    fv = fwd3.loc[pm].rename('fwd') if pm in fwd3.index else pd.Series(dtype=float, name='fwd')
    d0 = pd.concat([composite(pm), fv], axis=1)
    out = {c: (lambda d: spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0] if len(d) >= 10 and d.iloc[:, 0].nunique() > 1 else np.nan)(d0[[c, 'fwd']].dropna()) for c in CLUSTERS}
    s = pd.Series(out); _sp3[pm] = s; return s

def win38(m):   # embargo: 36个月全实现
    i = IDX[m]; return me_sorted[max(0, i - 38):i - 2]

def baseline_score(m):
    pms = win38(m); W = []
    for c in CLUSTERS:
        v = pd.Series([sp3(pm).get(c, np.nan) for pm in pms if pm in fwd3.index]).dropna()
        W.append(abs(v.mean() / v.std()) if len(v) > 6 and v.std() > 0 else 0)
    W = np.array(W); W = W / W.sum() if W.sum() > 0 else np.ones(len(CLUSTERS)) / len(CLUSTERS)
    return (composite(m).fillna(0) * W).sum(axis=1)

def ml_score(m, model):
    pms = win38(m); Xtr, ytr = [], []
    for pm in pms:
        if pm not in fwd3.index: continue
        cmp_pm = composite(pm)
        for s in stks:
            xv = cmp_pm.loc[s].values; yv = fwd3.loc[pm, s] if s in fwd3.columns else np.nan
            if not np.isnan(xv).any() and pd.notna(yv): Xtr.append(xv); ytr.append(yv)
    if len(Xtr) < 300: return None
    Xtr, ytr = np.array(Xtr), np.array(ytr)
    xs = StandardScaler().fit(Xtr)
    xm = xs.transform(composite(m).loc[stks].fillna(0).values)
    if model == 'Lasso':
        mdl = LassoCV(cv=5, max_iter=5000).fit(xs.transform(Xtr), ytr); pred = mdl.predict(xm)
    elif model == 'PLS':
        mdl = PLSRegression(n_components=min(5, len(CLUSTERS) - 1)).fit(xs.transform(Xtr), ytr); pred = mdl.predict(xm).ravel()
    elif model == 'XGBoost':
        mdl = XGBRegressor(n_estimators=60, max_depth=2, learning_rate=0.05, subsample=0.7,
                           colsample_bytree=0.7, reg_lambda=5, reg_alpha=2, n_jobs=2).fit(xs.transform(Xtr), ytr)
        pred = mdl.predict(xm)
    return pd.Series(pred, index=stks)

def run(scorer, quarterly, name, model=None):
    rets, months, ics = [], [], []
    for m in quarterly:
        if m not in fwd3.index: continue
        sc = scorer(m) if model is None else scorer(m, model)
        if sc is None or sc.dropna().nunique() < 5: continue
        sc = sc.dropna()
        try: q = pd.qcut(sc.rank(method='first'), 5, labels=False)
        except Exception: continue
        q5 = sc[q == 4].index
        if len(q5) < 3: continue
        vals = [fwd3.loc[m, o] for o in q5 if o in fwd3.columns and pd.notna(fwd3.loc[m, o])]
        if len(vals) >= 2:
            rets.append(np.mean(vals) - COST); months.append(m)
            d = pd.concat([sc, fwd3.loc[m]], axis=1).dropna(); ics.append(spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0])
    r = pd.Series(rets, index=months)
    if len(r) < 6: return None
    ann = r.mean() * 4; sh = r.mean() / r.std() * 2 if r.std() > 0 else 0
    cum = (1 + r).cumprod(); dd = (cum / cum.cummax() - 1).min()
    bph = ph3.reindex(months).dropna().mean() * 4; bmk = mkt3.reindex(months).dropna().mean() * 4
    return {'年化': round(ann, 4), '夏普': round(sh, 2), '最大回撤': round(dd, 4),
            '超额vs医药EW': round(ann - bph, 4), '超额vs全AEW': round(ann - bmk, 4),
            '平均RankIC': round(np.nanmean(ics), 4), '期数': len(r)}

rows = []
for ph in [0, 1, 2]:
    quarterly = [m for i, m in enumerate(me_sorted) if i >= 36 and (i - 36) % 3 == ph]
    print(f'>>> phase={ph}: {len(quarterly)}季度点', flush=True)
    for name, scorer, mdl in [('baseline', baseline_score, None), ('Lasso', ml_score, 'Lasso'),
                              ('PLS', ml_score, 'PLS'), ('XGBoost', ml_score, 'XGBoost')]:
        print(f'  ... {name}', flush=True)
        r = run(scorer, quarterly, name, mdl)
        if r: rows.append({'模型': name, '相位': ph, **r})
res = pd.DataFrame(rows)
med = []
for name in ['baseline', 'Lasso', 'PLS', 'XGBoost']:
    s = res[res.模型 == name]
    med.append({'模型': name, '相位': '中位',
                '年化': round(float(s.年化.median()), 4), '夏普': round(float(s.夏普.median()), 2),
                '最大回撤': round(float(s.最大回撤.median()), 4),
                '超额vs医药EW': round(float(s['超额vs医药EW'].median()), 4),
                '超额vs全AEW': round(float(s['超额vs全AEW'].median()), 4),
                '平均RankIC': round(float(s.平均RankIC.median()), 4), '期数': int(s.期数.median())})
res = pd.concat([res, pd.DataFrame(med)], ignore_index=True)
res.to_csv(f'{OUT}/redo_stageB_ml.csv', index=False, encoding='utf-8-sig')
print('\n===== Stage B: 新池12F 簇合成 ML (干净口径, 3相位) =====', flush=True)
print(res.to_string(index=False), flush=True)
print('\n对照: Stage A 透明混合池(同池同相位) 中位 11.5% / +6.5pp / 0.52', flush=True)
print('存 outputs/redo_stageB_ml.csv', flush=True)
