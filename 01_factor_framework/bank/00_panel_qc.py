"""00 面板质检 — 逐月截面数量曲线 + 低N月统计 + IC筛选的低N敏感性

回答三个问题(跑管线前先过这里):
  1. 银行面板逐月到底多少只?(白名单数 vs 18因子全可用的有效截面数)
  2. 银行股是陆续上市的 → 早期月份截面小, IC噪声大(SE=1/√(N-1)), 哪些月低于警戒线?
  3. "把低N月扔掉"会不会改变 01 的筛选结论? → 敏感性: 全样本 vs N≥30 vs N≥25 的 IC/ICIR

输出: outputs/panel_qc_count.png + outputs/panel_qc_sensitivity.csv
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK']
matplotlib.rcParams['axes.unicode_minus'] = False
import common

OUT = '/Users/lichanghaodemac/quant_strategy/projects/bank_factor_workflow/outputs'
P = common.load_panel()

# === 1. 三条"每月多少只"的曲线 ===
univ = pd.read_parquet(f'{common.DATA}/universe_monthend_pit.parquet')
univ['monthend'] = pd.DatetimeIndex(univ['monthend'])
wl = univ[univ['industry_l1'] == '银行'].groupby('monthend')['order_book_id'].nunique().sort_index()  # 真白名单逐月

fwd_ok = P['fwd1'].reindex(index=P['Z'][common.ALL18[0]].index, columns=P['Z'][common.ALL18[0]].columns).notna()
eff_all18 = np.logical_and.reduce([(P['Z'][f].notna() & fwd_ok).values for f in common.ALL18])  # 18因子+下月收益全非NaN(PCR口径)
ic_eff = pd.Series(eff_all18.sum(axis=1), index=P['Z'][common.ALL18[0]].index)
n_div = (P['Z']['dividend_yield_ttm'].notna() & fwd_ok).sum(axis=1)  # 单因子口径(缺失最少的基本面)

fig, ax = plt.subplots(figsize=(14, 5))
# 白名单画成绿色填充带(不画线): 股息因子几乎无缺失, IC有效截面=白名单(115月里114月完全相等),
# 两条线画在一起必然重叠遮盖 → 带子做底, 蓝虚线骑在上面, 重合处虚线可辨
ax.fill_between(wl.index, 0, wl.values, color='#d5f5e3', zorder=1, label='银行PIT白名单(逐月, 绿色区域)')
wl.plot(ax=ax, lw=0.8, color='#27ae60', zorder=1, label='_')
n_div.plot(ax=ax, lw=1.5, ls='--', color='#2980b9', zorder=3, label='IC有效截面(股息因子; 与绿区上沿重合=零缺失)')
ic_eff.plot(ax=ax, lw=1.6, color='#c0392b', zorder=2, alpha=0.9, label='PCR有效截面(18因子全非NaN)')
ax.axhline(30, color='gray', ls='--', lw=0.8); ax.text(wl.index[0], 30.5, 'N=30 警戒线(1m IC SE≈0.19)', fontsize=8, color='gray')
strat_start = P['me_sorted'][36]
ax.axvline(strat_start, color='k', ls=':', lw=1); ax.text(strat_start, 12, ' 回测起点(第37月)', fontsize=8)
ax.set_title('银行面板逐月截面数量 QC — 白名单/单因子/18因子全可用三条口径', fontsize=13)
ax.set_ylabel('股票数(只)'); ax.legend(fontsize=9); ax.grid(alpha=0.2)
plt.tight_layout(); plt.savefig(f'{OUT}/panel_qc_count.png', dpi=120, bbox_inches='tight'); plt.close()

print('=== 逐月截面数量(三条口径) ===', flush=True)
for name, s in [('白名单', wl), ('IC有效(股息)', n_div), ('PCR有效(18因子)', ic_eff)]:
    print(f'  {name:<14}: 中位{int(s.median())} 最小{s.min()} 最大{s.max()}'
          f' | 回测起点({str(strat_start)[:10]})后中位{int(s[s.index >= strat_start].median())}', flush=True)
low = ic_eff[ic_eff < 30]
print(f'  PCR口径 N<30 的月份: {len(low)}个' + (f' ({str(low.index[0])[:10]} ~ {str(low.index[-1])[:10]})' if len(low) else ''), flush=True)

# === 2. 敏感性: 扔掉低N月, 01的筛选结论变不变(1m窗口) ===
rows = []
for f in common.ALL18:
    ic = P['IC1'][f].dropna()
    n_ic = (P['Z'][f].notna() & fwd_ok[P['Z'][f].columns]).sum(axis=1).reindex(ic.index)
    r = {'因子': f}
    for lb, mask in [('全样本', pd.Series(True, index=ic.index)), ('N≥30', n_ic >= 30), ('N≥25', n_ic >= 25)]:
        sub = ic[mask]
        m, sd = sub.mean(), (sub.std() if len(sub) > 6 else np.nan)
        icir = m / sd if sd and sd > 0 else 0
        r[f'{lb}_IC'] = round(m, 3); r[f'{lb}_ICIR'] = round(icir, 2)
        r[f'{lb}_合格'] = (abs(m) >= 0.05) and (abs(icir) >= 0.3)
        r[f'{lb}_月数'] = int(mask.sum())
    rows.append(r)
sens = pd.DataFrame(rows)
sens.to_csv(f'{OUT}/panel_qc_sensitivity.csv', index=False, encoding='utf-8-sig')

print('\n=== 低N敏感性: 1m窗口, 全样本 vs N≥30 vs N≥25 (合格=|IC|≥0.05 & |ICIR|≥0.3) ===', flush=True)
print(sens[['因子', '全样本_IC', '全样本_ICIR', '全样本_合格', 'N≥30_IC', 'N≥30_ICIR', 'N≥30_合格', 'N≥30_月数', '全样本_月数']].to_string(index=False), flush=True)
flip = sens[sens['全样本_合格'] != sens['N≥30_合格']]
print(f'\nN≥30后合格状态翻转的因子: {len(flip)}个' + (f' → {", ".join(flip["因子"])}' if len(flip) else ' → 筛选结论对低N月稳健'), flush=True)
print('存 outputs/panel_qc_count.png + panel_qc_sensitivity.csv', flush=True)
