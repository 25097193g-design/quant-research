"""05 因子集 × 调仓频率 矩阵 — 4F/7F/PCA × 月频/季频(医药)

04 结论: 医药24F独立, 去冗余丢alpha, 且无过滤最好(gate没用, 与银行相反)。
本脚本看: 频率维度是否补回? PCA(24因子→7成分)能否超越手挑4F/7F?
口径同 bank 05(唯一PCR+回测在 common.py):
  - 4F/7F 滚动ICIR加权(月配1m-IC, 季配3m-IC)
  - PCA = 24因子 PCR(PCA7→Ridge, walk-forward)
  - 无过滤 vs <-1% 超跌 gate 对照
输出: outputs/freq_sets_matrix.csv + equity_curve.png
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, common

OUT = '/Users/lichanghaodemac/quant_strategy/projects/pharma_factor_workflow/outputs'
P = common.load_panel()
fwd1, fwd3, mkt1, mkt3 = P['fwd1'], P['fwd3'], P['mkt1'], P['mkt3']
me_sorted = P['me_sorted']
monthly = [m for i, m in enumerate(me_sorted) if i >= 36]
quarterly = [m for i, m in enumerate(me_sorted) if i >= 36 and (i - 36) % 3 == 0]

print('算因子集score (4F/7F ICIR × 月/季 + PCA × 月/季)...', flush=True)
SETS = {
    ('4F', '月'): common.score_set(P, common.F4, common.NEG4, common.roll_icir(P, common.F4, fwd1)),
    ('4F', '季'): common.score_set(P, common.F4, common.NEG4, common.roll_icir(P, common.F4, fwd3)),
    ('7F', '月'): common.score_set(P, common.F7, common.NEG7, common.roll_icir(P, common.F7, fwd1)),
    ('7F', '季'): common.score_set(P, common.F7, common.NEG7, common.roll_icir(P, common.F7, fwd3)),
    ('24F', '月'): common.score_set(P, P['facs'], {'VOLT20', 'VOLT60', 'pb_ratio_lf', 'pe_ratio_lyr', 'ps_ratio_ttm'}, common.roll_icir(P, P['facs'], fwd1)),
    ('24F', '季'): common.score_set(P, P['facs'], {'VOLT20', 'VOLT60', 'pb_ratio_lf', 'pe_ratio_lyr', 'ps_ratio_ttm'}, common.roll_icir(P, P['facs'], fwd3)),
    ('PCA', '月'): common.pcr_scores(P, fwd1),
    ('PCA', '季'): common.pcr_scores(P, fwd3),
}
FWD = {'月': (fwd1, mkt1, 12, monthly), '季': (fwd3, mkt3, 4, quarterly)}

rows, curves = [], {}
for (setname, freq), sc in SETS.items():
    fwd, mkt, py, rebal = FWD[freq]
    for lb, thr in [('无过滤', None), ('<-1%', -0.01)]:
        r = common.backtest(P, sc, rebal, fwd, mkt, thr)
        m = common.metrics(r, py)
        if m: m.update({'因子集': setname, '调仓': freq, '过滤': lb}); rows.append(m)
        if lb == '无过滤': curves[f'{setname}/{freq}'] = (1 + r).cumprod()
res = pd.DataFrame(rows)[['因子集', '调仓', '过滤', '年化', '夏普', '最大回撤', '期胜率', '期数']]
res.to_csv(f'{OUT}/freq_sets_matrix.csv', index=False, encoding='utf-8-sig')

print(f'\n全市场等权基准: 月频{mkt1.mean()*12:.2%} / 季频{mkt3.mean()*4:.2%}\n', flush=True)
for lb in ['无过滤', '<-1%']:
    print(f'===== 超跌过滤: {lb} =====', flush=True)
    print(res[res.过滤 == lb][['因子集', '调仓', '年化', '夏普', '最大回撤', '期胜率']].to_string(index=False), flush=True)
    print(flush=True)

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK']; matplotlib.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(figsize=(14, 6))
for k, c in curves.items(): c.plot(ax=ax, label=k, lw=1.5)
ax.set_title('医药因子集×调仓频率 净值曲线 (无过滤, 对数轴)', fontsize=13); ax.set_yscale('log')
ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.2)
plt.tight_layout(); plt.savefig(f'{OUT}/equity_curve.png', dpi=110, bbox_inches='tight'); plt.close()
print('存 freq_sets_matrix.csv + equity_curve.png', flush=True)
