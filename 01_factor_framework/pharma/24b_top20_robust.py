"""24b Top20 稳健性 — 集中度赢家过三关: 3起始偏移 + 2018长窗 + Q5对照同测

24发现Top20(前20只)双口径胜Q5(+2.5pp, 夏普平), 但那是4选1的后验挑选, 按惯例补:
  关1: 月频3起始偏移(i≥36/37/38) — 起始月运气
  关2: 2018-01起长窗(i≥12, 含2018熊市, 12m窗暖机最早点) — regime压力
  对照Q5同测(它有D3先例+3.1%), Top20若在三关上全面≥Q5才定为新默认。
输出: outputs/redo_stageK2_top20_robust.csv
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

SCORES = {}
for m in me_sorted:
    if m not in fwd1.index: continue
    i = IDX[m]
    if i < 12: continue   # 12m窗最少需要12个月
    win = me_sorted[max(0, i - 12):i]
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
    SCORES[m] = (z.fillna(0) * ws).sum(axis=1)

def hold(sc, mode):
    sc = sc.dropna()
    if mode == 'Q5':
        try: q = pd.qcut(sc.rank(method='first'), 5, labels=False)
        except Exception: return None
        return list(sc[q == 4].index)
    return list(sc.nlargest(mode).index)

def bt(mode, months):
    rets = []
    for m in months:
        if m not in SCORES or m not in fwd1.index: continue
        lst = hold(SCORES[m], mode)
        if not lst: continue
        vals = [fwd1.loc[m, o] for o in lst if o in fwd1.columns and pd.notna(fwd1.loc[m, o])]
        if len(vals) >= 3: rets.append(np.mean(vals) - COST)
    r = pd.Series(rets)
    if len(r) < 6: return None
    return r

rows = []
# 关1: 3起始偏移 (2020窗)
for st in [36, 37, 38]:
    months = me_sorted[st:]
    for mode in ['Q5', 20]:
        r = bt(mode, months)
        rows.append({'关口': f'关1·起始偏移', '变体': f'{mode}/{st}起', '年化': round(r.mean() * 12, 4),
                     '夏普': round(r.mean() / r.std() * np.sqrt(12), 2), '期数': len(r)})
# 关2: 2018-01起长窗 (i>=12)
months = me_sorted[12:]
for mode in ['Q5', 20]:
    r = bt(mode, months)
    cum = (1 + r).cumprod(); dd = (cum / cum.cummax() - 1).min()
    rows.append({'关口': '关2·2018长窗', '变体': str(mode), '年化': round(r.mean() * 12, 4),
                 '夏普': round(r.mean() / r.std() * np.sqrt(12), 2), '期数': len(r)})
    print(f'2018长窗 {mode}: {r.mean()*12:.2%}/{r.mean()/r.std()*np.sqrt(12):.2f}/回撤{dd:.1%} ({len(r)}期)', flush=True)

res = pd.DataFrame(rows)
res.to_csv(f'{OUT}/redo_stageK2_top20_robust.csv', index=False, encoding='utf-8-sig')
print('\n===== Top20 三关稳健性 =====', flush=True)
print(res.to_string(index=False), flush=True)
print('\n判卷: Top20在三关全面≥Q5才定新默认; 关2对照: Q5的D3先例=2018起+5.2pp超额', flush=True)
print('存 outputs/redo_stageK2_top20_robust.csv', flush=True)
