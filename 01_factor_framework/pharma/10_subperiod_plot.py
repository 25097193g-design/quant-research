"""10_subperiod_plot — 重画 10_subperiod_ic 的散点图(读已算好的CSV, 不重算)

修正: 10版硬编码±0.28视野, 137因子IC几乎全挤在±0.1内糊成一团。
本版: 自适应视野(两轴|IC|的98分位), 超界因子钉在边上(空心圈, 不丢点), 等比坐标轴(y=x才有意义)。
运行: cd projects/pharma_factor_workflow && conda run -n rqsdk --no-capture-output python 10_subperiod_plot.py
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK']
matplotlib.rcParams['axes.unicode_minus'] = False

OUT = '/Users/lichanghaodemac/quant_strategy/projects/pharma_factor_workflow/outputs'
res = pd.read_csv(f'{OUT}/subperiod_ic_137.csv')

x = res['IC_early'].astype(float); y = res['IC_late'].astype(float)

# 自适应视野: 全部|IC|的98分位(137个里钉边约3个), 下限0.08防过度放大
vals = np.abs(pd.concat([x, y]).dropna())
lim = max(0.08, np.nanpercentile(vals, 98)); lim = float(np.ceil(lim * 20) / 20)
xc, yc = x.clip(-lim, lim), y.clip(-lim, lim)
out = (x.abs() > lim) | (y.abs() > lim)
n_out = int(out.sum())

fig, ax = plt.subplots(figsize=(9.5, 9))
colors = ['#27ae60' if s else ('#2980b9' if p else '#b3b3b3') for s, p in zip(res['存活'], res['在24池'])]
ax.scatter(xc, yc, s=26, c=colors, alpha=0.7, zorder=3)
if n_out:   # 超界因子: 空心圈钉在边界, 可见但不占视野
    ax.scatter(xc[out], yc[out], s=70, facecolors='none', edgecolors='#555555', lw=0.9, zorder=4)
ax.axhline(0, color='k', lw=0.6); ax.axvline(0, color='k', lw=0.6)
ax.plot([-lim, lim], [-lim, lim], ls='--', color='#2980b9', lw=1, label='y=x (两期同强度)')
ax.fill_between([-lim, lim], [0.03, 0.03], [lim, lim], color='#27ae60', alpha=0.06)
ax.fill_between([-lim, lim], [-lim, lim], [-0.03, -0.03], color='#27ae60', alpha=0.06, label='晚期显著区(|IC_late|≥0.03)')
for _, r in res.iterrows():
    rx, ry = min(max(r['IC_early'], -lim), lim), min(max(r['IC_late'], -lim), lim)
    if r['存活']:
        ax.annotate(r['因子'].replace('WorldQuant_', 'α'), (rx, ry), fontsize=8, color='#145a32',
                    xytext=(4, 4), textcoords='offset points', zorder=5)
    elif r['在24池']:
        ax.annotate(r['因子'].replace('WorldQuant_', 'α'), (rx, ry), fontsize=7, color='#1a5276',
                    xytext=(4, -8), textcoords='offset points', zorder=5)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
ax.set_xlabel(f'IC_early (2017-2019)'); ax.set_ylabel('IC_late (2020-2026, 策略实际交易期)')
ax.set_title(f'医药137因子分时段IC: 铁存活{int((res["存活"]&res["三段存活"]).sum())}/两段{int(res["存活"].sum())}/三段{int(res["三段存活"].sum())}个(绿), 24池=蓝, 其余=灰\n'
             f'视野±{lim:.2f}(98分位自适应), 空心圈={n_out}个超界钉边', fontsize=11)
ax.legend(fontsize=9, loc='lower right'); ax.grid(alpha=0.2)
plt.tight_layout(); plt.savefig(f'{OUT}/subperiod_ic_scatter.png', dpi=120, bbox_inches='tight'); plt.close()
print(f'存 outputs/subperiod_ic_scatter.png (视野±{lim:.2f}, 钉边{n_out}个: {", ".join(res.loc[out, "因子"].str.replace("WorldQuant_","α"))})')
