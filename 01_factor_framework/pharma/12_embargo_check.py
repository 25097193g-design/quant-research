"""12 泄漏审计 — 3m 标签重叠前视的 embargo 对照 (2026-08-15 重新审视发现)

【问题】07/09/10/11 的 walk-forward 在决策月 m 用过去36月训练:
  past = [mm for mm in me_sorted if mm < m][-36:]  → 包含 pm = m-1, m-2
  训练标签 y = fwd3.loc[pm] = pm → pm+3 的3个月收益
  pm = m-1 的标签覆盖到 m+2, 与测试窗 [m, m+3) 重叠 2 个月 → 训练标签泄漏测试期收益。
同样问题存在于季度 ICIR 权重(roll_icir/make_signals 用 fwd3 算 past 内 IC, pm=m-1/m-2 越界)。
"3相位全赢" 只排除日历运气, 不排除此泄漏(三相位泄漏机制相同)。

【对照设计】决策时点 m 真正可观测的标签要求 pm+3 ≤ m, 即 pm ≤ m-3 → 丢 past 最后2个月:
  - leaky    : 原版(past36, 含 m-1/m-2 重叠标签)
  - embargo34: past36[:-2](34月, 全部已实现)
  - window38 : [i-38 : i-2](同样36月样本量, 全部已实现) ← 与 leaky 唯一差 = 去掉重叠
覆盖: XGB19簇 / baseline19簇 @3相位 + 24F季(3m-ICIR权重) @3相位 + 信号IC(RankIC vs fwd3)
输出: outputs/embargo_check.csv
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

DATA = '/Users/lichanghaodemac/quant_strategy/data'
OUT = '/Users/lichanghaodemac/quant_strategy/projects/pharma_factor_workflow/outputs'
NEG24 = {'VOLT20', 'VOLT60', 'pb_ratio_lf', 'pe_ratio_lyr', 'ps_ratio_ttm'}

CLUSTERS = {  # 同07/09(24因子→19簇)
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

# ===== 数据加载(与09一致) =====
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

def ic_series(f, fwd, minn=10):
    return pd.Series({m: (lambda d: spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0] if len(d) >= minn and d.iloc[:, 0].nunique() > 1 else np.nan)
                      (pd.concat([Z[f].loc[m], fwd.loc[m]], axis=1).dropna()) for m in me_sorted if m in fwd.index})

print('算 IC1 / IC3 (24因子 × 月度)...', flush=True)
IC1 = {f: ic_series(f, fwd1, minn=10) for f in ALLF}          # 符号对齐用(1m, 决策时可观测)
IC3 = {f: ic_series(f, fwd3, minn=10) for f in ALLF}          # 24F季度权重用(原版含重叠)

# ===== 复合因子缓存: composite 只依赖 pm 自身(符号=IC1 trailing36) =====
print('算 19簇复合因子缓存 (每月一次)...', flush=True)
_comp_cache = {}
def composite(pm):
    if pm in _comp_cache: return _comp_cache[pm]
    i = IDX[pm]; past_pm = me_sorted[max(0, i - 36):i]
    comp = {}
    for cname, facs in CLUSTERS.items():
        vals = []
        for f in facs:
            v = IC1[f].reindex(past_pm).dropna(); sgn = np.sign(v.mean()) if len(v) > 6 else 1.0
            vals.append(sgn * Z[f].loc[pm])
        comp[cname] = pd.concat(vals, axis=1).mean(axis=1)
    df = pd.DataFrame(comp).reindex(stks); _comp_cache[pm] = df; return df

# 复合因子 vs fwd3 的 IC(给 baseline 权重用)
_sp3_cache = {}
def sp3(pm):
    if pm in _sp3_cache: return _sp3_cache[pm]
    fv = fwd3.loc[pm].rename('fwd') if pm in fwd3.index else pd.Series(dtype=float, name='fwd')
    d0 = pd.concat([composite(pm), fv], axis=1)
    out = {}
    for c in CLUSTERS:
        d = d0[[c, 'fwd']].dropna()
        out[c] = spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0] if len(d) >= 10 and d.iloc[:, 0].nunique() > 1 else np.nan
    s = pd.Series(out); _sp3_cache[pm] = s; return s

def train_months(m, mode):
    i = IDX[m]
    if mode == 'leaky':     return me_sorted[max(0, i - 36):i]          # 原版: 含 m-1/m-2 重叠标签
    if mode == 'embargo34': return me_sorted[max(0, i - 36):i - 2]      # 34月, 全部已实现
    if mode == 'window38':  return me_sorted[max(0, i - 38):i - 2]      # 36月, 全部已实现
    raise ValueError(mode)

def baseline_score(m, mode):
    pms = train_months(m, mode); W = []
    for c in CLUSTERS:
        v = pd.Series([sp3(pm).get(c, np.nan) for pm in pms if pm in fwd3.index]).dropna()
        W.append(abs(v.mean() / v.std()) if len(v) > 6 and v.std() > 0 else 0)
    W = np.array(W); W = W / W.sum() if W.sum() > 0 else np.ones(len(CLUSTERS)) / len(CLUSTERS)
    return (composite(m).fillna(0) * W).sum(axis=1)

def xgb_score(m, mode):
    pms = train_months(m, mode)
    Xtr, ytr = [], []
    for pm in pms:
        if pm not in fwd3.index: continue
        cmp_pm = composite(pm)
        for s in stks:
            xv = cmp_pm.loc[s].values; yv = fwd3.loc[pm, s] if s in fwd3.columns else np.nan
            if not np.isnan(xv).any() and pd.notna(yv): Xtr.append(xv); ytr.append(yv)
    if len(Xtr) < 300: return None
    Xtr, ytr = np.array(Xtr), np.array(ytr)
    xs = StandardScaler().fit(Xtr)
    mdl = XGBRegressor(n_estimators=60, max_depth=2, learning_rate=0.05, subsample=0.7,
                       colsample_bytree=0.7, reg_lambda=5, reg_alpha=2, n_jobs=2).fit(xs.transform(Xtr), ytr)
    return pd.Series(mdl.predict(xs.transform(composite(m).loc[stks].fillna(0).values)), index=stks)

# ===== 24F 季度 ICIR 权重 (leaky vs window38) =====
def f24_score(m, mode):
    pms = train_months(m, mode); w = []
    for f in ALLF:
        v = IC3[f].reindex(pms).dropna()
        w.append(abs(v.mean() / v.std()) if len(v) > 6 and v.std() > 0 else 0)
    w = np.array(w); w = w / w.sum() if w.sum() > 0 else np.ones(len(ALLF)) / len(ALLF)
    z = pd.DataFrame(index=stks)
    for f in ALLF:
        c = Z[f].loc[m].reindex(stks); z[f] = -c if f in NEG24 else c
    return (z.fillna(0) * w).sum(axis=1)

COST = 0.0015
def run(scorer, mode, quarterly, name):
    rets, ics = [], []
    for m in quarterly:
        if m not in fwd3.index: continue
        sc = scorer(m, mode)
        if sc is None or sc.dropna().nunique() < 5: continue
        sc = sc.dropna()
        try: q = pd.qcut(sc.rank(method='first'), 5, labels=False)
        except: continue
        q5 = sc[q == 4].index
        if len(q5) < 3: continue
        vals = [fwd3.loc[m, o] for o in q5 if o in fwd3.columns and pd.notna(fwd3.loc[m, o])]
        rets.append(np.mean(vals) - COST)
        d_ic = pd.concat([sc, fwd3.loc[m]], axis=1).dropna()
        ics.append(spearmanr(d_ic.iloc[:, 0], d_ic.iloc[:, 1])[0])
    r = pd.Series(rets)
    return {'配置': name, '季度数': len(r), '年化': round(r.mean() * 4, 4),
            '夏普(季)': round(r.mean() / r.std() * 2, 2) if r.std() > 0 else 0,
            '最大回撤': round(((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min(), 4),
            '平均RankIC': round(np.nanmean(ics), 4)}

rows = []
for ph in [0, 1, 2]:
    quarterly = [m for i, m in enumerate(me_sorted) if i >= 38 and (i - 38) % 3 == ph]
    print(f'\n>>> phase={ph}: {len(quarterly)}个季度点', flush=True)
    for label, scorer in [('baseline', baseline_score), ('XGBoost', xgb_score), ('24F', f24_score)]:
        for mode in (['leaky', 'embargo34', 'window38'] if label == 'XGBoost' else ['leaky', 'embargo34'] if label == 'baseline' else ['leaky', 'window38']):
            print(f'  ... {label}/{mode}', flush=True)
            rows.append({**run(scorer, mode, quarterly, f'{label}/{mode}'), 'phase': ph})
res = pd.DataFrame(rows)[['phase', '配置', '季度数', '年化', '夏普(季)', '最大回撤', '平均RankIC']]
res.to_csv(f'{OUT}/embargo_check.csv', index=False, encoding='utf-8-sig')

print('\n===== 泄漏审计: leaky(原版,标签重叠) vs window38(36月,无重叠) vs embargo34 =====', flush=True)
print(res.to_string(index=False), flush=True)

print('\n===== 3相位中位数汇总 =====', flush=True)
for cfg in res['配置'].unique():
    s = res[(res.配置 == cfg) & (res.phase.isin([0, 1, 2]))]
    print(f'  {cfg:<18} 年化中位{np.median(s.年化):>7.2%}  夏普中位{np.median(s["夏普(季)"]):>5.2f}  RankIC中位{np.median(s.平均RankIC):>+.4f}', flush=True)
print('\n存 outputs/embargo_check.csv', flush=True)
