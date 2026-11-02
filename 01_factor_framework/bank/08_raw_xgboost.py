"""08 原始18因子直喂XGBoost — 不经聚类, 看树模型自己处理共线性能否更强

假说: 树模型能自己挑相关因子里的代表 + 在原始因子上找非线性切分,
      先聚成8个平均反而抹平信息 → raw-18 XGBoost 可能 > cluster-8 XGBoost。
现实风险: 18特征 × 41股×115月 小样本 → 更多特征=更高过拟合方差。
对照(全部walk-forward, 季度调仓, 同数据):
  baseline     : 8簇合成 ICIR加权 (07的无ML基线)
  XGB_cluster8 : 8簇合成 → XGBoost (07的)
  XGB_raw18    : 18原始因子 → XGBoost (本脚本的主角)
  Ridge_raw18  : 18原始因子 → Ridge (线性参考, 共线性下应不如树)

输入: data/(factors_monthend, mainboard_price, universe_monthend_pit)
输出: outputs/raw_xgb_compare.csv + 对照打印
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

DATA = '/Users/lichanghaodemac/quant_strategy/data'
OUT = '/Users/lichanghaodemac/quant_strategy/projects/bank_factor_workflow/outputs'

CLUSTERS = {'C1': ['WorldQuant_alpha013', 'WorldQuant_alpha016'],
            'C2': ['beta60', 'beta250', 'market_cap_3', 'a_share_market_val_in_circulation', 'OBV'],
            'C3': ['dividend_yield_ttm', 'pe_ratio_ttm', 'pe_ratio_lyr'],
            'C4': ['ev_to_ebitda_ttm', 'net_profit_margin_ttm', 'return_on_equity_ttm'],
            'C5': ['MOM120', 'MOM_12M_1M'], 'C6': ['DAVOL20'],
            'C7': ['WorldQuant_alpha026'], 'C8': ['WorldQuant_alpha088']}
ALL18 = sorted({f for v in CLUSTERS.values() for f in v})
FROMDB = [f for f in ALL18 if f not in {'MOM120', 'MOM_12M_1M', 'beta60', 'beta250'}]

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
allf = factors[['order_book_id', 'monthend'] + FROMDB].copy()
for n, s in self_mats.items():
    sl = s.loc[[m for m in mev if m in s.index]].stack().rename(n).reset_index(); sl.columns = ['monthend', 'order_book_id', n]
    allf = allf.merge(sl, on=['order_book_id', 'monthend'], how='outer')
allf = allf.merge(univ[univ['industry_l1'] == '银行'][['order_book_id', 'monthend']], on=['order_book_id', 'monthend'], how='inner')
allf = allf[['order_book_id', 'monthend'] + ALL18].sort_values(['monthend', 'order_book_id']).reset_index(drop=True)
allf[ALL18] = allf[ALL18].replace([np.inf, -np.inf], np.nan)   # 清inf(alpha026)
me_sorted = sorted(allf['monthend'].unique()); WIDE = {f: allf.pivot(index='monthend', columns='order_book_id', values=f) for f in ALL18}
stks = np.array(WIDE[ALL18[0]].columns)

def zs(mat):
    lo, hi = mat.quantile(0.01, axis=1), mat.quantile(0.99, axis=1); o = mat.copy()
    for c in mat.columns: o[c] = o[c].clip(lo, hi, axis=0)
    return o.sub(o.mean(axis=1), axis=0).div(o.std(axis=1).replace(0, np.nan), axis=0)
Z = {f: zs(WIDE[f]) for f in ALL18}
IC1 = {f: pd.Series({m: (lambda d: spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0] if len(d) >= 10 and d.iloc[:, 0].nunique() > 1 else np.nan)
                     (pd.concat([Z[f].loc[m], fwd1.loc[m]], axis=1).dropna()) for m in me_sorted if m in fwd1.index}) for f in ALL18}

def sgn_of(f, past):
    v = IC1[f].reindex(past).dropna()
    return np.sign(v.mean()) if len(v) > 6 else 1.0

def feat_raw18(m, past):   # 18原始因子, 符号对齐后的z (树其实不在意符号, 但统一一下)
    return pd.DataFrame({f: sgn_of(f, past) * Z[f].loc[m] for f in ALL18}).reindex(stks)

def feat_cluster8(m, past):
    comp = {}
    for cname, facs in CLUSTERS.items():
        comp[cname] = pd.concat([sgn_of(f, past) * Z[f].loc[m] for f in facs], axis=1).mean(axis=1)
    return pd.DataFrame(comp).reindex(stks)

def stack(feat_fn, months, past_fn):
    X, y = [], []
    for m in months:
        if m not in fwd3.index: continue
        comp = feat_fn(m, past_fn(m))
        for s in stks:
            xv = comp.loc[s].values; yv = fwd3.loc[m, s] if s in fwd3.columns else np.nan
            if not np.isnan(xv).any() and pd.notna(yv): X.append(xv); y.append(yv)
    return np.array(X), np.array(y)

# ★ 2026-08-15: 季度起点对齐04/06/07的2020-12(面板2017-01的第47个月=2020-12-31), 原2020-01起跑
qstart = pd.Timestamp('2020-12-01')
qi = next(i for i, m in enumerate(me_sorted) if m >= qstart)
quarterly = [m for i, m in enumerate(me_sorted) if i >= qi and (i - qi) % 3 == 0]
past_fn = lambda m: [mm for mm in me_sorted if mm < m][-36:]
COST = 0.0015

def xgb_fit(Xtr, ytr):
    return XGBRegressor(n_estimators=60, max_depth=2, learning_rate=0.05, subsample=0.7,
                        colsample_bytree=0.7, reg_lambda=5, reg_alpha=2, n_jobs=2).fit(Xtr, ytr)

def run(name, feat_fn, model):
    rets = []
    for m in quarterly:
        past = past_fn(m)
        if m not in fwd3.index: continue
        Xtr, ytr = stack(feat_fn, past, past_fn)
        if len(Xtr) < 200: continue
        comp_m = feat_fn(m, past)
        xm = comp_m.loc[stks].fillna(0).values
        sc = StandardScaler().fit(Xtr)
        if model == 'baseline':
            # ICIR加权 (重算权重)
            W = []
            for c in CLUSTERS:
                ics = []
                for pm in past:
                    if pm not in fwd3.index: continue
                    d = pd.concat([feat_cluster8(pm, past_fn(pm))[c], fwd3.loc[pm]], axis=1).dropna()
                    if len(d) >= 10 and d.iloc[:, 0].nunique() > 1: ics.append(spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0])
                v = np.array(ics); W.append(abs(v.mean() / v.std()) if len(v) > 6 and v.std() > 0 else 0)
            W = np.array(W); W = W / W.sum() if W.sum() > 0 else np.ones(8) / 8
            pred = (comp_m.fillna(0).values * W).sum(axis=1)
        elif model == 'xgb':
            mdl = xgb_fit(sc.transform(Xtr), ytr); pred = mdl.predict(sc.transform(xm))
        elif model == 'ridge':
            mdl = Ridge(alpha=10).fit(sc.transform(Xtr), ytr); pred = mdl.predict(sc.transform(xm))
        scs = pd.Series(pred, index=stks)
        if scs.dropna().nunique() < 5: continue
        try: q = pd.qcut(scs.rank(method='first'), 5, labels=False)
        except: continue
        q5 = scs[q == 4].index
        if len(q5) < 3: continue
        ret = np.mean([fwd3.loc[m, o] for o in q5 if o in fwd3.columns and pd.notna(fwd3.loc[m, o])]) - COST
        rets.append(ret)
    r = pd.Series(rets)
    if len(r) < 5: return None
    cum = (1 + r).cumprod(); dd = (cum / cum.cummax() - 1).min()
    return {'模型': name, '季度数': len(r), '年化': round(r.mean() * 4, 4), '夏普(季)': round(r.mean() / r.std() * 2, 2),
            '最大回撤': round(dd, 4), '期胜率': round((r > 0).mean(), 2)}

rows = []
rows.append(run('baseline(8簇ICIR)', feat_cluster8, 'baseline'))
rows.append(run('XGB_8簇合成', feat_cluster8, 'xgb'))
rows.append(run('XGB_18原始', feat_raw18, 'xgb'))
rows.append(run('Ridge_18原始', feat_raw18, 'ridge'))
res = pd.DataFrame([r for r in rows if r])
res.to_csv(f'{OUT}/raw_xgb_compare.csv', index=False, encoding='utf-8-sig')
mkt3 = fwd3.mean(axis=1)
mw = mkt3.reindex([m for m in quarterly if m in mkt3.index]).dropna()
print(f'基准(全市场等权3m, 与策略同窗口{str(quarterly[0])[:10]}起): 年化{mw.mean() * 4:.2%}/夏普{mw.mean() / mw.std() * 2:.2f}'
      f' (旧口径全样本{mkt3.mean() * 4:.2%}, 含暖机期勿再用)\n', flush=True)
print('=== 原始18因子直喂XGBoost vs 簇合成 (walk-forward 季度调仓) ===', flush=True)
print(res.to_string(index=False), flush=True)
print('\n存 outputs/raw_xgb_compare.csv', flush=True)
print('解读: 看XGB_18原始 vs XGB_8簇合成; 树能自处理共线性, 但18特征小样本过拟合风险更高', flush=True)
