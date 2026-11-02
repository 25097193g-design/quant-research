"""05 因子集 × 调仓频率 矩阵 — 4F/7F/PCA × 月频/季频

用 common.py 的唯一 PCR + 回测(消除与09的drift)。
PCA = PCR(18因子→PCA7→Ridge, walk-forward)。
4F/7F 用滚动ICIR加权(月频配1m-IC, 季频配3m-IC)。
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, common

OUT = '/Users/lichanghaodemac/quant_strategy/projects/bank_factor_workflow/outputs'
P = common.load_panel()
fwd1, fwd3, mkt1, mkt3 = P['fwd1'], P['fwd3'], P['mkt1'], P['mkt3']
me_sorted = P['me_sorted']
# ★ 2026-08-15: 起跑对齐04/06/07/08的2020-12(面板2017-01的第47个月), 原为2020-01
qstart = pd.Timestamp('2020-12-01'); qi = next(i for i, m in enumerate(me_sorted) if m >= qstart)
monthly = [m for i, m in enumerate(me_sorted) if i >= qi]
quarterly = [m for i, m in enumerate(me_sorted) if i >= qi and (i - qi) % 3 == 0]

print('算因子集score (4F/7F ICIR × 月/季 + PCA × 月/季)...', flush=True)
SETS = {
    ('4F', '月'): common.score_set(P, common.F4, common.NEG4, common.roll_icir(P, common.F4, fwd1)),
    ('4F', '季'): common.score_set(P, common.F4, common.NEG4, common.roll_icir(P, common.F4, fwd3)),
    ('7F', '月'): common.score_set(P, common.F7, common.NEG7, common.roll_icir(P, common.F7, fwd1)),
    ('7F', '季'): common.score_set(P, common.F7, common.NEG7, common.roll_icir(P, common.F7, fwd3)),
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
        if lb == '<-1%': curves[f'{setname}/{freq}'] = (1 + r).cumprod()
res = pd.DataFrame(rows)[['因子集', '调仓', '过滤', '年化', '夏普', '最大回撤', '期胜率', '期数']]
res.to_csv(f'{OUT}/freq_sets_matrix.csv', index=False, encoding='utf-8-sig')

# ★ 基准必须与策略同窗口: 策略从第37个月(2020-01)起跑, 全样本均值含2017-2019暖机期, 不可比
w1 = mkt1.reindex(monthly).dropna(); w3 = mkt3.reindex(quarterly).dropna()
print(f'\n全市场等权基准(与策略同窗口, {str(monthly[0])[:10]}起): '
      f'月频年化{w1.mean()*12:.2%}/夏普{w1.mean()/w1.std()*np.sqrt(12):.2f} | '
      f'季频年化{w3.mean()*4:.2%}/夏普{w3.mean()/w3.std()*2:.2f}')
print(f'(旧口径全样本基准: 月频{mkt1.mean()*12:.2%} / 季频{mkt3.mean()*4:.2%} — 含暖机期, 超额被高估, 勿再用)\n', flush=True)
for lb in ['无过滤', '<-1%']:
    print(f'===== 超跌过滤: {lb} =====', flush=True)
    print(res[res.过滤 == lb][['因子集', '调仓', '年化', '夏普', '最大回撤', '期胜率']].to_string(index=False), flush=True)
    print(flush=True)

# 净值曲线
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK']; matplotlib.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(figsize=(14, 6))
for k, c in curves.items():
    c.plot(ax=ax, label=k, lw=1.5)
ax.set_title('超跌埋伏轮动(<-1%) 净值曲线: 因子集×调仓频率 (对数轴)', fontsize=13); ax.set_yscale('log')
ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.2)
plt.tight_layout(); plt.savefig(f'{OUT}/equity_curve.png', dpi=110, bbox_inches='tight'); plt.close()
print('存 freq_sets_matrix.csv + equity_curve.png', flush=True)
