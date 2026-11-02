"""10 分时段IC检验(医药) — 137因子全量 × (2017-2019 vs 2020-2026), 找"跨时期存活者"

fork 自 bank 10_subperiod_ic.py, 三处适配:
  1. 板块: 银行 → 医药生物; 分割点仍 2020-01, 但理由不同(银行=白名单才满30只; 医药=36月暖机后策略起跑, 面板本身一直稳定)
  2. 存活阈值: 银行 |IC_late|≥0.05; 医药口径 |IC_late|≥0.03 & |ICIR_late|≥0.3(01的0.03口径), 同时报0.05版
  3. 对照池: 现行18 → 现行24(common._POOL); 另加"晚期强但不在池"的新候选段(全样本IC被早期拖掉的)

医药早期截面184-214只(银行15-25只) → early期统计功效好, z_diff 检验可信度高。

存活者判定(四条件全过):
  ① |IC_late| ≥ 0.03 且 |ICIR_late| ≥ 0.3   — 晚期真有信号
  ② sign(IC_early) == sign(IC_late)          — 方向没翻
  ③ |z_diff| < 2                             — 无法拒绝"两期强度相同"
  ④ T_early ≥ 12                             — 早期样本够
另加三段细切(2017-19/2020-22/2023-26): 三段IC全同号 且 2020-22、2023-26两段都|IC|≥0.03 → 三段存活
(晚期79月内部不同质: 抱团牛/熊/阴跌+924/大波动, 三段判定看"晚期合格"是均匀的还是靠单段撑的)

口径: 137 = WQ + MQ29 + 自算7价量; 医药PIT白名单; 截面≥10; 月度spearman; inf→NaN; 主表1m, 附3m晚期。
输出: outputs/subperiod_ic_137.csv + outputs/subperiod_ic_scatter.png
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK']
matplotlib.rcParams['axes.unicode_minus'] = False
import common

DATA = '/Users/lichanghaodemac/quant_strategy/data'
OUT = '/Users/lichanghaodemac/quant_strategy/projects/pharma_factor_workflow/outputs'
SPLIT = pd.Timestamp('2020-01-01')
CURRENT24 = common._POOL

# === 面板(口径同bank 10) ===
price = pd.read_parquet(f'{DATA}/mainboard_price.parquet'); price['date'] = pd.to_datetime(price['date'])
close = price.pivot(index='date', columns='order_book_id', values='close').sort_index()
me = pd.DatetimeIndex(pd.read_parquet(f'{DATA}/monthend_dates.parquet')['monthend'])
mev = [m for m in me if m in close.index]; cm = close.loc[mev]

def beta_p(n):
    rd = close.pct_change(fill_method=None); mk = rd.mean(axis=1)
    return rd.rolling(n).cov(mk).div(mk.rolling(n).var(), axis=0)
SELF = [('MOM20', close.pct_change(20, fill_method=None)), ('MOM60', close.pct_change(60, fill_method=None)),
        ('MOM120', close.pct_change(120, fill_method=None)), ('MOM_12M_1M', close.shift(20) / close.shift(240) - 1),
        ('REV5D', -close.pct_change(5, fill_method=None)), ('beta60', beta_p(60)), ('beta250', beta_p(250))]
self_long = None
for n, s in SELF:
    sl = s.loc[[m for m in mev if m in s.index]].stack().rename(n).reset_index()
    sl.columns = ['monthend', 'order_book_id', n]; sl = sl.set_index(['order_book_id', 'monthend'])
    self_long = sl if self_long is None else self_long.join(sl, how='outer')
self_long = self_long.reset_index()

factors = pd.read_parquet(f'{DATA}/factors_monthend.parquet'); factors['monthend'] = pd.DatetimeIndex(factors['monthend'])
WQ = [c for c in factors.columns if c.startswith('WorldQuant')]
MQ29 = [c for c in factors.columns if c not in WQ + ['order_book_id', 'date', 'monthend']]
FACS = WQ + MQ29 + ['MOM20', 'MOM60', 'MOM120', 'MOM_12M_1M', 'REV5D', 'beta60', 'beta250']
univ = pd.read_parquet(f'{DATA}/universe_monthend_pit.parquet'); univ['monthend'] = pd.DatetimeIndex(univ['monthend'])
panel = factors.drop(columns=['date']).merge(self_long, on=['order_book_id', 'monthend'], how='outer')\
               .merge(univ[univ['industry_l1'] == '医药生物'][['order_book_id', 'monthend']], on=['order_book_id', 'monthend'], how='inner')
panel[FACS] = panel[FACS].replace([np.inf, -np.inf], np.nan)     # ★ inf清洗
fwd1 = (cm.shift(-1) / cm - 1).stack().rename('fwd1').reset_index(); fwd1.columns = ['monthend', 'order_book_id', 'fwd1']
fwd3 = (cm.shift(-3) / cm - 1).stack().rename('fwd3').reset_index(); fwd3.columns = ['monthend', 'order_book_id', 'fwd3']
panel = panel.merge(fwd1, on=['monthend', 'order_book_id'], how='left').merge(fwd3, on=['monthend', 'order_book_id'], how='left')
print(f'137因子医药面板: {panel["order_book_id"].nunique()}只 × {panel.monthend.nunique()}月', flush=True)

GROUPS = {m: g for m, g in panel.groupby('monthend')}   # 预分组, 137因子循环内不再重复groupby

def period_stats(s):
    if len(s) < 6 or s.std() == 0: return (len(s), np.nan, np.nan, np.nan)
    m, sd, T = s.mean(), s.std(), len(s)
    return (T, m, m / sd, sd / np.sqrt(T))

def to_series(d):
    if not d: return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    return pd.Series(d).sort_index()

rows = []
SEG3 = [('2017-19', '2017-01-01', '2019-12-31'), ('2020-22', '2020-01-01', '2022-12-31'), ('2023-26', '2023-01-01', '2026-12-31')]
for fac in FACS:
    ic1, ic3 = {}, {}
    for m, g in GROUPS.items():
        s = g[[fac, 'fwd1']].dropna()
        if len(s) >= 10 and s[fac].nunique() > 1: ic1[m] = spearmanr(s[fac], s['fwd1'])[0]
        s3 = g[[fac, 'fwd3']].dropna()
        if len(s3) >= 10 and s3[fac].nunique() > 1: ic3[m] = spearmanr(s3[fac], s3['fwd3'])[0]
    s1 = to_series(ic1); s3 = to_series(ic3)
    Te, ICe, ICILe, SEe = period_stats(s1[s1.index < SPLIT])
    Tl, ICl, ICILl, SEl = period_stats(s1[s1.index >= SPLIT])
    Tl3, ICl3, ICILl3, _ = period_stats(s3[s3.index >= SPLIT])
    # 三段细切: 2017-19 / 2020-22 / 2023-26 (各段1m IC统计)
    seg = {}
    for tag, a, b in SEG3:
        seg[tag] = period_stats(s1[(s1.index >= pd.Timestamp(a)) & (s1.index <= pd.Timestamp(b))])
    sgns = [np.sign(seg[t][1]) for t, _, _ in SEG3 if pd.notna(seg[t][1])]
    three_ok = (len(sgns) == 3 and len(set(sgns)) == 1 and sgns[0] != 0
                and abs(seg['2020-22'][1]) >= 0.03 and abs(seg['2023-26'][1]) >= 0.03)
    if any(pd.isna(v) for v in [ICe, ICl]):
        z = np.nan; diff = np.nan
    else:
        diff = ICl - ICe
        z = diff / np.sqrt(SEe**2 + SEl**2) if (pd.notna(SEe) and pd.notna(SEl) and SEe + SEl > 0) else np.nan
    sig03 = bool(pd.notna(ICl) and abs(ICl) >= 0.03 and abs(ICILl) >= 0.3)
    sig05 = bool(pd.notna(ICl) and abs(ICl) >= 0.05 and abs(ICILl) >= 0.3)
    dir_ok = bool(pd.notna(ICe) and pd.notna(ICl) and np.sign(ICe) == np.sign(ICl))
    surv = (pd.notna(ICe) and pd.notna(ICl) and Te >= 12 and sig03 and dir_ok and pd.notna(z) and abs(z) < 2)
    rows.append({'因子': fac, '在24池': fac in CURRENT24, 'T_early': Te, 'IC_early': round(ICe, 3) if pd.notna(ICe) else np.nan,
                 'ICIR_early': round(ICILe, 2) if pd.notna(ICILe) else np.nan,
                 'T_late': Tl, 'IC_late': round(ICl, 3) if pd.notna(ICl) else np.nan,
                 'ICIR_late': round(ICILl, 2) if pd.notna(ICILl) else np.nan,
                 'IC_late_3m': round(ICl3, 3) if pd.notna(ICl3) else np.nan, 'ICIR_late_3m': round(ICILl3, 2) if pd.notna(ICILl3) else np.nan,
                 'IC_2017-19': round(seg['2017-19'][1], 3) if pd.notna(seg['2017-19'][1]) else np.nan,
                 'ICIR_2017-19': round(seg['2017-19'][2], 2) if pd.notna(seg['2017-19'][2]) else np.nan,
                 'IC_2020-22': round(seg['2020-22'][1], 3) if pd.notna(seg['2020-22'][1]) else np.nan,
                 'ICIR_2020-22': round(seg['2020-22'][2], 2) if pd.notna(seg['2020-22'][2]) else np.nan,
                 'IC_2023-26': round(seg['2023-26'][1], 3) if pd.notna(seg['2023-26'][1]) else np.nan,
                 'ICIR_2023-26': round(seg['2023-26'][2], 2) if pd.notna(seg['2023-26'][2]) else np.nan,
                 'IC_diff': round(diff, 3) if pd.notna(diff) else np.nan,
                 'z_diff': round(z, 2) if pd.notna(z) else np.nan,
                 '晚期显著(0.03)': sig03, '晚期显著(0.05)': sig05, '方向一致': dir_ok,
                 '强度稳定': bool(pd.notna(z) and abs(z) < 2), '存活': bool(surv), '三段存活': bool(three_ok)})
res = pd.DataFrame(rows)
res.to_csv(f'{OUT}/subperiod_ic_137.csv', index=False, encoding='utf-8-sig')

# === 汇总1: 全137的跨期存活者 ===
survivors = res[res['存活']].reindex(res[res['存活']]['ICIR_late'].abs().sort_values(ascending=False).index)
surv05 = res[res['存活'] & res['晚期显著(0.05)']]
three = res[res['三段存活']].reindex(res[res['三段存活']]['ICIR_late'].abs().sort_values(ascending=False).index)
iron = res[res['存活'] & res['三段存活']]
print(f'\n=== ① 137因子: 两段存活 {len(survivors)}个 / 三段存活 {len(three)}个 / 双判定铁存活 {len(iron)}个 ===', flush=True)
print('--- 两段存活者(主切2020-01) ---', flush=True)
print(survivors[['因子', '在24池', 'IC_early', 'IC_late', 'IC_2017-19', 'IC_2020-22', 'IC_2023-26', 'z_diff', '三段存活']].to_string(index=False), flush=True)
print(f'--- 三段存活者(2017-19/2020-22/2023-26 全同号+后两段|IC|≥0.03): {len(three)}个 ---', flush=True)
print(three[['因子', '在24池', 'IC_2017-19', 'ICIR_2017-19', 'IC_2020-22', 'ICIR_2020-22', 'IC_2023-26', 'ICIR_2023-26', '存活']].to_string(index=False), flush=True)

# === 汇总2: 现行24池逐个判定 ===
print(f'\n=== ② 现行24池存活判定 (两段/三段) ===', flush=True)
cur = res[res['因子'].isin(CURRENT24)].set_index('因子').reindex(CURRENT24)
n_ok = n3 = 0
for f, r in cur.iterrows():
    mark = '✅ 存活' if r['存活'] else ('⚠️ 仅晚期强' if r['晚期显著(0.03)'] else ('❌ 方向翻' if not r['方向一致'] else '❌ 晚期已弱'))
    m3 = '✅三段' if r['三段存活'] else '·'
    if r['存活']: n_ok += 1
    if r['三段存活']: n3 += 1
    print(f'  {f:<38} 1719:{r["IC_2017-19"]:+.3f} 2022:{r["IC_2020-22"]:+.3f} 2326:{r["IC_2023-26"]:+.3f} | late{r["IC_late"]:+.3f}/{r["ICIR_late"]:+.2f} z{r["z_diff"]:+.1f}  {mark} {m3}', flush=True)
print(f'  → 24池: 两段存活 {n_ok}/24, 三段存活 {n3}/24', flush=True)

# === 汇总3: 晚期强但不在24池的新候选(全样本IC被早期拖掉的) ===
newc = res[(~res['在24池']) & res['晚期显著(0.03)'] & res['方向一致']].reindex(
       res[(~res['在24池']) & res['晚期显著(0.03)'] & res['方向一致']]['ICIR_late'].abs().sort_values(ascending=False).index)
print(f'\n=== ③ 晚期显著+方向一致 但不在24池(全样本IC被早期拖掉/1m∪3m口径漏掉): {len(newc)}个, 取前15 ===', flush=True)
print(newc.head(15)[['因子', 'IC_early', 'IC_late', 'IC_2017-19', 'IC_2020-22', 'IC_2023-26', 'ICIR_late', 'IC_late_3m', '三段存活']].to_string(index=False), flush=True)

# === 图: early vs late 散点 ===
fig, ax = plt.subplots(figsize=(10, 9))
x, y = res['IC_early'].astype(float), res['IC_late'].astype(float)
colors = ['#27ae60' if s else ('#2980b9' if p else '#95a5a6') for s, p in zip(res['存活'], res['在24池'])]
ax.scatter(x, y, s=22, c=colors, alpha=0.65)
ax.axhline(0, color='k', lw=0.6); ax.axvline(0, color='k', lw=0.6)
lim = 0.28; ax.plot([-lim, lim], [-lim, lim], ls='--', color='#2980b9', lw=1, label='y=x (两期同强度)')
ax.fill_between([-lim, lim], [0.03, 0.03], [lim, lim], color='#27ae60', alpha=0.06)
ax.fill_between([-lim, lim], [-lim, lim], [-0.03, -0.03], color='#27ae60', alpha=0.06, label='晚期显著区(|IC_late|≥0.03)')
for _, r in res.iterrows():
    if r['存活']: ax.annotate(r['因子'].replace('WorldQuant_', 'α'), (r['IC_early'], r['IC_late']), fontsize=7.5, color='#145a32',
                              xytext=(4, 4), textcoords='offset points')
    elif r['在24池']: ax.annotate(r['因子'].replace('WorldQuant_', 'α'), (r['IC_early'], r['IC_late']), fontsize=7, color='#1a5276',
                                   xytext=(4, -8), textcoords='offset points')
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel('IC_early (2017-2019)'); ax.set_ylabel('IC_late (2020-2026, 策略实际交易期)')
ax.set_title(f'医药137因子分时段IC: 跨期存活 {len(survivors)}个(绿), 24池=蓝, 其余=灰', fontsize=12)
ax.legend(fontsize=9, loc='lower right'); ax.grid(alpha=0.2)
plt.tight_layout(); plt.savefig(f'{OUT}/subperiod_ic_scatter.png', dpi=120, bbox_inches='tight'); plt.close()
print('\n存 outputs/subperiod_ic_137.csv + subperiod_ic_scatter.png', flush=True)
