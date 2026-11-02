"""27 2025超额归零的初步归因 — 因子分年IC × 截面离散度 × 卫星/核心分年贡献

三个假设检验(2025 Top20超额仅+0.4pp, 2024为+8.2pp):
  H1 因子衰减: 12因子分年1m IC —— 2025年核心7/卫星5是否集体变弱?
  H2 截面离散度收窄: 医药月度截面收益std分年 —— 分化小→选股空间小?
  H3 成分结构: core7与sat5单独打分(同12m滚动)的分年收益 —— 2025谁拖累?
输出: outputs/redo_stageN_2025_attribution.csv
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import spearmanr
import common

OUT = '/Users/lichanghaodemac/quant_strategy/projects/pharma_factor_workflow/outputs'
COST = 0.0015
CORE7 = ['WorldQuant_alpha040', 'WorldQuant_alpha016', 'WorldQuant_alpha024', 'WorldQuant_alpha044',
         'WorldQuant_alpha011', 'dividend_yield_ttm', 'WorldQuant_alpha026']
SAT5 = ['book_to_market_ratio_lf', 'pb_ratio_lf', 'ps_ratio_ttm', 'VOLT20', 'VOLT60']
FACS = CORE7 + SAT5

P = common.load_panel(FACS)
fwd1 = P['fwd1']; me_sorted = P['me_sorted']; Z = P['Z']; stks = P['stks']
IDX = {m: i for i, m in enumerate(me_sorted)}

# ===== H1: 分因子分年IC(2023/24/25/26H1) =====
ic_rows = []
for f in FACS:
    s = pd.Series({m: (lambda d: spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0] if len(d) >= 10 and d.iloc[:, 0].nunique() > 1 else np.nan)
                   (pd.concat([Z[f].loc[m], fwd1.loc[m]], axis=1).dropna()) for m in me_sorted if m in fwd1.index}).dropna()
    row = {'因子': f.replace('WorldQuant_', '').replace('_ttm', '').replace('_lf', '').replace('_ratio', '')}
    for yr in [2023, 2024, 2025, 2026]:
        v = s[s.index.year == yr]
        row[str(yr)] = round(v.mean(), 3) if len(v) >= 6 else np.nan
    ic_rows.append(row)
ic_df = pd.DataFrame(ic_rows)
print('===== H1 因子分年IC(1m) =====')
print(ic_df.to_string(index=False))

# ===== H2: 截面离散度(医药月度个股收益std)分年 =====
univ = pd.read_parquet(f'{common.DATA}/universe_monthend_pit.parquet'); univ['monthend'] = pd.DatetimeIndex(univ['monthend'])
ids = univ[univ['industry_l1'] == '医药生物'].groupby('monthend')['order_book_id'].apply(set).to_dict()
disp = {m: fwd1.loc[m, [o for o in ss if o in fwd1.columns]].dropna().std() for m, ss in ids.items() if m in fwd1.index and len([o for o in ss if o in fwd1.columns]) >= 10}
disp = pd.Series(disp)
print('\n===== H2 医药截面月收益离散度(std)分年均值 =====')
print(disp.groupby(disp.index.year).mean().round(3).to_string())

# ===== H3: core7 / sat5 单独打分的分年收益(同12m滚动配方) =====
IC1 = {f: pd.Series({m: (lambda d: spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0] if len(d) >= 10 and d.iloc[:, 0].nunique() > 1 else np.nan)
                     (pd.concat([Z[f].loc[m], fwd1.loc[m]], axis=1).dropna()) for m in me_sorted if m in fwd1.index}) for f in FACS}

def run_leg(facs):
    rets = {}
    for m in me_sorted[36:]:
        if m not in fwd1.index: continue
        i = IDX[m]; win = me_sorted[max(0, i - 12):i]
        ws, ss_ = [], []
        for f in facs:
            v = IC1[f].reindex(win).dropna()
            if len(v) > 6 and v.std() > 0:
                ws.append(abs(v.mean() / v.std())); ss_.append(1 if v.mean() > 0 else -1)
            else:
                ws.append(0.0); ss_.append(1)
        ws = np.array(ws); ws = ws / ws.sum() if ws.sum() > 0 else np.ones(len(facs)) / len(facs)
        z = pd.DataFrame(index=stks)
        for f, s in zip(facs, ss_): z[f] = s * Z[f].loc[m].reindex(stks)
        sc = (z.fillna(0) * ws).sum(axis=1).dropna()
        lst = list(sc.nlargest(20).index)
        vals = [fwd1.loc[m, o] for o in lst if o in fwd1.columns and pd.notna(fwd1.loc[m, o])]
        if len(vals) >= 3: rets[m] = np.mean(vals) - COST
    return pd.Series(rets)

core_r = run_leg(CORE7); sat_r = run_leg(SAT5)
ph_ew = pd.Series({m: fwd1.loc[m, [o for o in ss if o in fwd1.columns]].dropna().mean() for m, ss in ids.items() if m in fwd1.index}).dropna()
print('\n===== H3 core7 / sat5 单独Top20 的分年收益与超额(vs 医药EW) =====')
rows = []
for nm, r in [('core7', core_r), ('sat5', sat_r)]:
    for yr in [2023, 2024, 2025, 2026]:
        g = r[r.index.year == yr]
        p = ph_ew[ph_ew.index.year == yr]
        if len(g) >= 6:
            rows.append({'腿': nm, '年': yr, '年化': round(g.mean() * 12, 3), '超额vs医药EW': round(g.mean() * 12 - p.mean() * 12, 3)})
print(pd.DataFrame(rows).to_string(index=False))

ic_df.to_csv(f'{OUT}/redo_stageN_2025_attribution.csv', index=False, encoding='utf-8-sig')
print('\n存 outputs/redo_stageN_2025_attribution.csv')
