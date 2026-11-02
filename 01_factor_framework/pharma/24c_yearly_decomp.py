"""24c 分年相对表现分解 — Top20 vs 医药等权 vs 全A等权, 回答"2023/2026是不是策略的问题"

逻辑: 若坏年份里 Top20 相对医药等权的超额仍为正 → 选股没失败, 跌的是板块beta(用户归因成立);
若超额转负 → 策略自身在坏regime里失效。输出: outputs/redo_stageK3_yearly_decomp.csv
运行: cd projects/pharma_factor_workflow && conda run -n rqsdk --no-capture-output python 24c_yearly_decomp.py
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
IC1 = {f: pd.Series({m: (lambda d: spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0] if len(d) >= 10 and d.iloc[:, 0].nunique() > 1 else np.nan)
                     (pd.concat([Z[f].loc[m], fwd1.loc[m]], axis=1).dropna()) for m in me_sorted if m in fwd1.index}) for f in FACS}

univ = pd.read_parquet(f'{common.DATA}/universe_monthend_pit.parquet'); univ['monthend'] = pd.DatetimeIndex(univ['monthend'])
ids = univ[univ['industry_l1'] == '医药生物'].groupby('monthend')['order_book_id'].apply(set).to_dict()
ph1 = pd.Series({m: fwd1.loc[m, [o for o in ss if o in fwd1.columns]].dropna().mean() for m, ss in ids.items() if m in fwd1.index}).dropna()
mkt1 = P['mkt1']

rows = []
for m in me_sorted[36:]:
    if m not in fwd1.index: continue
    i = IDX[m]; win = me_sorted[max(0, i - 12):i]
    ws, ss_ = [], []
    for f in FACS:
        v = IC1[f].reindex(win).dropna()
        if len(v) > 6 and v.std() > 0:
            ws.append(abs(v.mean() / v.std())); ss_.append(1 if v.mean() > 0 else -1)
        else:
            ws.append(0.0); ss_.append(1)
    ws = np.array(ws); ws = ws / ws.sum() if ws.sum() > 0 else np.ones(len(FACS)) / len(FACS)
    z = pd.DataFrame(index=stks)
    for f, s in zip(FACS, ss_): z[f] = s * Z[f].loc[m].reindex(stks)
    sc = (z.fillna(0) * ws).sum(axis=1).dropna()
    lst = list(sc.nlargest(20).index)
    vals = [fwd1.loc[m, o] for o in lst if o in fwd1.columns and pd.notna(fwd1.loc[m, o])]
    if len(vals) >= 3: rows.append((m, np.mean(vals) - COST))

r = pd.Series(dict(rows)).sort_index()
df = pd.DataFrame({'Top20': r, '医药EW': ph1.reindex(r.index), '全AEW': mkt1.reindex(r.index)}).dropna()

out_rows = []
print(f'{"年份":<6}{"Top20":>9}{"医药EW":>9}{"全AEW":>9}{"超额vs医药":>10}{"超额vs全A":>10}')
for y, g in df.groupby(df.index.year):
    t = np.prod(1 + g['Top20']) - 1; p = np.prod(1 + g['医药EW']) - 1; a = np.prod(1 + g['全AEW']) - 1
    print(f'{y:<6}{t:>+8.1%}{p:>+9.1%}{a:>+9.1%}{t-p:>+10.1%}{t-a:>+10.1%}')
    out_rows.append({'年份': y, 'Top20': round(t, 4), '医药EW': round(p, 4), '全AEW': round(a, 4),
                     '超额vs医药': round(t - p, 4), '超额vs全A': round(t - a, 4), '月数': len(g)})
pd.DataFrame(out_rows).to_csv(f'{OUT}/redo_stageK3_yearly_decomp.csv', index=False, encoding='utf-8-sig')
print('存 outputs/redo_stageK3_yearly_decomp.csv')
