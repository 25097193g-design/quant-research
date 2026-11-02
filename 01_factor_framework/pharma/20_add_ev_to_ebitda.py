"""20 加因子试验 — ev_to_ebitda_ttm(两段存活/三段挂, 负IC价值型, 不在legacy24) 加入 core7+sat5

① 相关性: ev_to_ebitda 与 sat5 成员的逐月截面spearman均值(高相关=重复下注)
② 对照(月频, 2020-01起78期, 收益实现至2026-07-17, 同月双基准):
   D1(12F) 已有 13.10%/0.58/+5.91pp
   D1-ev(13F) = core7+sat5+ev_to_ebitda, 12m全滚动号+权重, 同17配方(ev的负方向由滚动号自动处理)
输出: outputs/redo_stageG_add_ev.csv
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
EV = 'ev_to_ebitda_ttm'
FACS = CORE7 + SAT5 + [EV]

P = common.load_panel(FACS)
fwd1 = P['fwd1']; me_sorted = P['me_sorted']; Z = P['Z']; stks = P['stks']
IDX = {m: i for i, m in enumerate(me_sorted)}
IC1 = {f: pd.Series({m: (lambda d: spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0] if len(d) >= 10 and d.iloc[:, 0].nunique() > 1 else np.nan)
                     (pd.concat([Z[f].loc[m], fwd1.loc[m]], axis=1).dropna()) for m in me_sorted if m in fwd1.index}) for f in FACS}

# ===== ① ev_to_ebitda vs sat5 成员: 逐月截面spearman 均值 =====
print('=== ① ev_to_ebitda 与 sat5/核心因子的月均截面相关 ===', flush=True)
for f in SAT5 + ['dividend_yield_ttm', 'WorldQuant_alpha040']:
    cs = []
    for m in me_sorted:
        d = pd.concat([Z[EV].loc[m], Z[f].loc[m]], axis=1).dropna()
        if len(d) > 30: cs.append(spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0])
    print(f'  ev_to_ebitda ~ {f:<28}: {np.nanmean(cs):+.2f}', flush=True)

# ===== ② 13F 回测(同17配方) =====
def run(facs, label):
    rets = []
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
        if sc.nunique() < 5: continue
        try: q = pd.qcut(sc.rank(method='first'), 5, labels=False)
        except Exception: continue
        q5 = sc[q == 4].index
        if len(q5) < 3: continue
        vals = [fwd1.loc[m, o] for o in q5 if o in fwd1.columns and pd.notna(fwd1.loc[m, o])]
        if len(vals) >= 2: rets.append(np.mean(vals) - COST)
    r = pd.Series(rets)
    ann = r.mean() * 12; sh = r.mean() / r.std() * np.sqrt(12)
    cum = (1 + r).cumprod(); dd = (cum / cum.cummax() - 1).min()
    univ = pd.read_parquet(f'{common.DATA}/universe_monthend_pit.parquet'); univ['monthend'] = pd.DatetimeIndex(univ['monthend'])
    ids = univ[univ['industry_l1'] == '医药生物'].groupby('monthend')['order_book_id'].apply(set).to_dict()
    ph1 = pd.Series({m: fwd1.loc[m, [o for o in ss if o in fwd1.columns]].dropna().mean() for m, ss in ids.items() if m in fwd1.index}).dropna()
    bph = ph1.iloc[36:].dropna().mean() * 12; bmk = P['mkt1'].iloc[36:].dropna().mean() * 12
    return {'配置': label, '年化': round(ann, 4), '夏普': round(sh, 2), '最大回撤': round(dd, 4),
            '超额vs医药EW': round(ann - bph, 4), '超额vs全AEW': round(ann - bmk, 4), '期数': len(r)}

rows = [run(FACS, 'D1-ev(13F): core7+sat5+ev_to_ebitda / 12m全滚动 / 月')]
res = pd.DataFrame(rows)
res.to_csv(f'{OUT}/redo_stageG_add_ev.csv', index=False, encoding='utf-8-sig')
print('\n=== ② 对照(月频, 2020-01~2026-06) ===', flush=True)
print(res.to_string(index=False), flush=True)
print('对照: D1(12F) = 13.10% / 0.58 / +5.91pp / +2.10pp (17脚本, 同配方)', flush=True)
print('存 outputs/redo_stageG_add_ev.csv', flush=True)
