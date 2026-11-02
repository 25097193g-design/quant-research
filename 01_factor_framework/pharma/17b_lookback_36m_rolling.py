"""17b 补格 — core7+sat5 用 36m 全滚动(号+权重都36月窗) 月频, 对照 D1(12m全滚动 13.10%)

同 17 的口径, 唯一差别 lb=36; sat5/36m 顺带复算校验 Stage A 的 12.91%(混合号的sat5行=全滚动)。
起点 2020-01(36m窗暖机最早); 月频78期(2020-01~2026-06), 收益实现至2026-07-17。
输出: outputs/redo_stageD2_36m.csv
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

P = common.load_panel(CORE7 + SAT5)
fwd1 = P['fwd1']; me_sorted = P['me_sorted']; Z = P['Z']; stks = P['stks']
IDX = {m: i for i, m in enumerate(me_sorted)}
IC1 = {f: pd.Series({m: (lambda d: spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0] if len(d) >= 10 and d.iloc[:, 0].nunique() > 1 else np.nan)
                     (pd.concat([Z[f].loc[m], fwd1.loc[m]], axis=1).dropna()) for m in me_sorted if m in fwd1.index}) for f in CORE7 + SAT5}

univ = pd.read_parquet(f'{common.DATA}/universe_monthend_pit.parquet'); univ['monthend'] = pd.DatetimeIndex(univ['monthend'])
ids = univ[univ['industry_l1'] == '医药生物'].groupby('monthend')['order_book_id'].apply(set).to_dict()
ph1 = pd.Series({m: fwd1.loc[m, [o for o in ss if o in fwd1.columns]].dropna().mean() for m, ss in ids.items() if m in fwd1.index and len([o for o in ss if o in fwd1.columns]) >= 10}).sort_index()
mkt1 = P['mkt1']

def run(facs, lb, label):
    rets = []
    for m in me_sorted[36:]:
        if m not in fwd1.index: continue
        i = IDX[m]; win = me_sorted[max(0, i - lb):i]
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
    bph = ph1.iloc[36:].dropna().mean() * 12; bmk = mkt1.iloc[36:].dropna().mean() * 12
    return {'配置': label, '年化': round(ann, 4), '夏普': round(sh, 2), '最大回撤': round(dd, 4),
            '超额vs医药EW': round(ann - bph, 4), '超额vs全AEW': round(ann - bmk, 4), '期数': len(r)}

rows = [run(CORE7 + SAT5, 36, 'core7+sat5 / 36m全滚动 / 月 / 2020-01'),
        run(SAT5, 36, 'sat5 / 36m全滚动 / 月 / 2020-01 (校验≈12.91%)')]
res = pd.DataFrame(rows)
res.to_csv(f'{OUT}/redo_stageD2_36m.csv', index=False, encoding='utf-8-sig')
print('===== 36m 全滚动补格 (月频, 2020-01~2026-06, 78期) =====', flush=True)
print(res.to_string(index=False), flush=True)
print('\n对照: D1 = 同池 12m全滚动 月频 → 13.10% / 0.58 / +5.91pp (17脚本)', flush=True)
print('存 outputs/redo_stageD2_36m.csv', flush=True)
