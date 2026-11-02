"""00 面板质检(医药) — 逐月截面数量曲线 + 因子覆盖变化 + IC筛选的低N敏感性

参考 bank 00_panel_qc.py, 板块换成医药生物, 三处适配:
  1. 警戒线: 银行41只用N=30(SE≈0.19); 医药截面~200只, 用N=100(SE≈0.10)做第一档, N=150(SE≈0.08)做第二档
  2. 因子池: 银行18合格因子; 医药用 common._POOL 24因子(01筛出的)
  3. 敏感性档位: 全样本 vs N≥100 vs N≥150 (银行是 30/25)
另加"因子变化": 每个因子的逐月覆盖率和"何时长齐"(首次≥90%白名单数)

输出: outputs/panel_qc_count.png + outputs/panel_qc_sensitivity.csv + outputs/panel_qc_factor_coverage.csv
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK']
matplotlib.rcParams['axes.unicode_minus'] = False
import common

OUT = '/Users/lichanghaodemac/quant_strategy/projects/pharma_factor_workflow/outputs'
P = common.load_panel()
FACS = P['facs']            # 24因子
fwd_ok = P['fwd1'].reindex(index=P['Z'][FACS[0]].index, columns=P['Z'][FACS[0]].columns).notna()

# === 1. 三条"每月多少只"的曲线 ===
univ = pd.read_parquet(f'{common.DATA}/universe_monthend_pit.parquet')
univ['monthend'] = pd.DatetimeIndex(univ['monthend'])
wl = univ[univ['industry_l1'] == '医药生物'].groupby('monthend')['order_book_id'].nunique().sort_index()

eff_all = np.logical_and.reduce([(P['Z'][f].notna() & fwd_ok).values for f in FACS])  # 24因子+下月收益全非NaN
ic_eff = pd.Series(eff_all.sum(axis=1), index=P['Z'][FACS[0]].index)
n_div = (P['Z']['dividend_yield_ttm'].notna() & fwd_ok).sum(axis=1)  # 单因子口径(缺失最少)

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(wl.index, 0, wl.values, color='#d5f5e3', zorder=1, label='医药PIT白名单(逐月, 绿色区域)')
wl.plot(ax=ax, lw=0.8, color='#27ae60', zorder=1, label='_')
n_div.plot(ax=ax, lw=1.5, ls='--', color='#2980b9', zorder=3, label='IC有效截面(股息因子; 与绿区上沿重合=零缺失)')
ic_eff.plot(ax=ax, lw=1.6, color='#c0392b', zorder=2, alpha=0.9, label=f'全因子有效截面({len(FACS)}因子全非NaN)')
ax.axhline(100, color='gray', ls='--', lw=0.8); ax.text(wl.index[0], 102, 'N=100 警戒线(1m IC SE≈0.10)', fontsize=8, color='gray')
strat_start = P['me_sorted'][36]
ax.axvline(strat_start, color='k', ls=':', lw=1); ax.text(strat_start, 30, ' 回测起点(第37月)', fontsize=8)
ax.set_title('医药面板逐月截面数量 QC — 白名单/单因子/24因子全可用三条口径', fontsize=13)
ax.set_ylabel('股票数(只)'); ax.legend(fontsize=9); ax.grid(alpha=0.2)
plt.tight_layout(); plt.savefig(f'{OUT}/panel_qc_count.png', dpi=120, bbox_inches='tight'); plt.close()

print('=== 逐月截面数量(三条口径) ===', flush=True)
for name, s in [('白名单', wl), ('IC有效(股息)', n_div), (f'全因子有效({len(FACS)}F)', ic_eff)]:
    print(f'  {name:<14}: 中位{int(s.median())} 最小{s.min()} 最大{s.max()}'
          f' | 回测起点({str(strat_start)[:10]})后中位{int(s[s.index >= strat_start].median())}', flush=True)
for th in [100, 150]:
    low = ic_eff[ic_eff < th]
    print(f'  全因子口径 N<{th} 的月份: {len(low)}个' + (f' ({str(low.index[0])[:10]} ~ {str(low.index[-1])[:10]})' if len(low) else ''), flush=True)

# === 2. 因子变化: 每因子覆盖率 + 何时"长齐" ===
rows = []
for f in FACS:
    cov = (P['Z'][f].notna() & fwd_ok[P['Z'][f].columns]).sum(axis=1)
    covr = (cov / wl.reindex(cov.index)).dropna()
    ready = covr[covr >= 0.9]
    rows.append({'因子': f, '覆盖中位%': round(covr.median(), 3), '覆盖最小%': round(covr.min(), 3),
                 '首次≥90%': str(ready.index[0].date()) if len(ready) else '从未',
                 '有效月中位N': int(cov.median())})
fcov = pd.DataFrame(rows).sort_values('覆盖中位%')
fcov.to_csv(f'{OUT}/panel_qc_factor_coverage.csv', index=False, encoding='utf-8-sig')
print('\n=== 24因子覆盖变化(按覆盖中位%升序, 最差的在最后长齐) ===', flush=True)
print(fcov.to_string(index=False), flush=True)

# === 3. 敏感性: 扔掉低N月, 01的筛选结论变不变(1m窗口, 医药阈值|IC|≥0.03 & |ICIR|≥0.3) ===
rows = []
for f in FACS:
    ic = P['IC1'][f].dropna()
    n_ic = (P['Z'][f].notna() & fwd_ok[P['Z'][f].columns]).sum(axis=1).reindex(ic.index)
    r = {'因子': f}
    for lb, mask in [('全样本', pd.Series(True, index=ic.index)), ('N≥100', n_ic >= 100), ('N≥150', n_ic >= 150)]:
        sub = ic[mask]
        m, sd = sub.mean(), (sub.std() if len(sub) > 6 else np.nan)
        icir = m / sd if sd and sd > 0 else 0
        r[f'{lb}_IC'] = round(m, 3); r[f'{lb}_ICIR'] = round(icir, 2)
        r[f'{lb}_合格'] = (abs(m) >= 0.03) and (abs(icir) >= 0.3)
        r[f'{lb}_月数'] = int(mask.sum())
    rows.append(r)
sens = pd.DataFrame(rows)
sens.to_csv(f'{OUT}/panel_qc_sensitivity.csv', index=False, encoding='utf-8-sig')

print('\n=== 低N敏感性: 1m窗口, 全样本 vs N≥100 vs N≥150 (合格=|IC|≥0.03 & |ICIR|≥0.3) ===', flush=True)
print(sens[['因子', '全样本_IC', '全样本_ICIR', '全样本_合格', 'N≥100_IC', 'N≥100_ICIR', 'N≥100_合格', 'N≥100_月数', '全样本_月数']].to_string(index=False), flush=True)
flip = sens[sens['全样本_合格'] != sens['N≥100_合格']]
print(f'\nN≥100后合格状态翻转的因子: {len(flip)}个' + (f' → {", ".join(flip["因子"])}' if len(flip) else ' → 筛选结论对低N月稳健'), flush=True)
print('存 outputs/panel_qc_count.png + panel_qc_sensitivity.csv + panel_qc_factor_coverage.csv', flush=True)
