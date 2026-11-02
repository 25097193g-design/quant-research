"""10 分时段IC检验 — 137因子全量 × (2017-2019 vs 2020-2026), 找"跨时期存活者"

动机(00_panel_qc 的后续): 白名单2020-01才满30只, 低N月≡2017-2019整段; 扔掉这段=删regime,
实测5个1m合格因子翻车。正确姿势不是扔, 是**分时段各算一遍, 看谁两期都活着**:

  IC_early / ICIR_early   (2017-01 ~ 2019-12, 36月, 截面15-25只, 噪声大)
  IC_late  / ICIR_late    (2020-01 ~ 2026-07, 79月, 截面36-41只, 策略实际交易期)
  IC_diff = IC_late - IC_early
  z_diff  = IC_diff / sqrt(SE_e²+SE_l²)   (SE=期内月IC的std/√T, 用经验值不用1/√(N-1)近似)

存活者判定(四个条件全过):
  ① |IC_late| ≥ 0.05 且 |ICIR_late| ≥ 0.3   — 晚期真有信号(阈值同01的小宇宙修正口径)
  ② sign(IC_early) == sign(IC_late)          — 方向没翻
  ③ |z_diff| < 2                             — 无法拒绝"两期强度相同"(不是只在早期强)
  ④ T_early ≥ 12                             — 早期样本够下结论

口径: 面板构建同 frame_17(137=WQ+MQ29+自算7价量, 银行PIT白名单, 截面≥10, 月度spearman);
     inf→NaN(WQ因子教训, 比frame_17多一步清洗); 主表用1m(不重叠最干净), 附3m晚期作参照。
输出: outputs/subperiod_ic_137.csv + outputs/subperiod_ic_scatter.png
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK']
matplotlib.rcParams['axes.unicode_minus'] = False

DATA = '/Users/lichanghaodemac/quant_strategy/data'
OUT = '/Users/lichanghaodemac/quant_strategy/projects/bank_factor_workflow/outputs'
SPLIT = pd.Timestamp('2020-01-01')
CURRENT18 = ['dividend_yield_ttm', 'ev_to_ebitda_ttm', 'pe_ratio_ttm', 'pe_ratio_lyr',
             'beta60', 'beta250', 'net_profit_margin_ttm', 'return_on_equity_ttm',
             'market_cap_3', 'a_share_market_val_in_circulation', 'MOM_12M_1M', 'MOM120',
             'OBV', 'DAVOL20', 'WorldQuant_alpha013', 'WorldQuant_alpha016',
             'WorldQuant_alpha026', 'WorldQuant_alpha088']

# === 面板(口径同frame_17) ===
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
bank = factors.drop(columns=['date']).merge(self_long, on=['order_book_id', 'monthend'], how='outer')\
              .merge(univ[univ['industry_l1'] == '银行'][['order_book_id', 'monthend']], on=['order_book_id', 'monthend'], how='inner')
bank[FACS] = bank[FACS].replace([np.inf, -np.inf], np.nan)     # ★ inf清洗(frame_17没做, spearman虽不炸但口径统一)
fwd1 = (cm.shift(-1) / cm - 1).stack().rename('fwd1').reset_index(); fwd1.columns = ['monthend', 'order_book_id', 'fwd1']
fwd3 = (cm.shift(-3) / cm - 1).stack().rename('fwd3').reset_index(); fwd3.columns = ['monthend', 'order_book_id', 'fwd3']
bank = bank.merge(fwd1, on=['monthend', 'order_book_id'], how='left').merge(fwd3, on=['monthend', 'order_book_id'], how='left')
print(f'137因子银行面板: {bank["order_book_id"].nunique()}只 × {bank.monthend.nunique()}月', flush=True)

# === 逐因子: 两期月IC序列 → 统计 ===
def period_stats(s):
    if len(s) < 6 or s.std() == 0: return (len(s), np.nan, np.nan, np.nan)
    m, sd, T = s.mean(), s.std(), len(s)
    return (T, m, m / sd, sd / np.sqrt(T))   # (T, IC, ICIR, SE)

def to_series(d):
    """dict → 按时间排序的Series; 空dict给DatetimeIndex空壳(否则RangeIndex和Timestamp比较会炸)"""
    if not d: return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    return pd.Series(d).sort_index()

rows = []
for fac in FACS:
    ic1, ic3 = {}, {}
    for m, g in bank.groupby('monthend'):
        s = g[[fac, 'fwd1']].dropna()
        if len(s) >= 10 and s[fac].nunique() > 1: ic1[m] = spearmanr(s[fac], s['fwd1'])[0]
        s3 = g[[fac, 'fwd3']].dropna()
        if len(s3) >= 10 and s3[fac].nunique() > 1: ic3[m] = spearmanr(s3[fac], s3['fwd3'])[0]
    s1 = to_series(ic1); s3 = to_series(ic3)
    Te, ICe, ICILe, SEe = period_stats(s1[s1.index < SPLIT])
    Tl, ICl, ICILl, SEl = period_stats(s1[s1.index >= SPLIT])
    Tl3, ICl3, ICILl3, _ = period_stats(s3[s3.index >= SPLIT])
    if any(pd.isna(v) for v in [ICe, ICl]):
        z = np.nan; diff = np.nan
    else:
        diff = ICl - ICe
        z = diff / np.sqrt(SEe**2 + SEl**2) if (pd.notna(SEe) and pd.notna(SEl) and SEe + SEl > 0) else np.nan
    surv = (pd.notna(ICe) and pd.notna(ICl) and Te >= 12
            and abs(ICl) >= 0.05 and abs(ICILl) >= 0.3
            and np.sign(ICe) == np.sign(ICl) and pd.notna(z) and abs(z) < 2)
    rows.append({'因子': fac, 'T_early': Te, 'IC_early': round(ICe, 3) if pd.notna(ICe) else np.nan,
                 'ICIR_early': round(ICILe, 2) if pd.notna(ICILe) else np.nan,
                 'T_late': Tl, 'IC_late': round(ICl, 3) if pd.notna(ICl) else np.nan,
                 'ICIR_late': round(ICILl, 2) if pd.notna(ICILl) else np.nan,
                 'IC_late_3m': round(ICl3, 3) if pd.notna(ICl3) else np.nan, 'ICIR_late_3m': round(ICILl3, 2) if pd.notna(ICILl3) else np.nan,
                 'IC_diff': round(diff, 3) if pd.notna(diff) else np.nan,
                 'z_diff': round(z, 2) if pd.notna(z) else np.nan,
                 '晚期显著': bool(pd.notna(ICl) and abs(ICl) >= 0.05 and abs(ICILl) >= 0.3),
                 '方向一致': bool(pd.notna(ICe) and pd.notna(ICl) and np.sign(ICe) == np.sign(ICl)),
                 '强度稳定': bool(pd.notna(z) and abs(z) < 2),
                 '存活': bool(surv)})
res = pd.DataFrame(rows)
res.to_csv(f'{OUT}/subperiod_ic_137.csv', index=False, encoding='utf-8-sig')

# === 汇总打印 ===
survivors = res[res['存活']].reindex(res[res['存活']]['ICIR_late'].abs().sort_values(ascending=False).index)
print(f'\n=== 跨时期存活者: {len(survivors)}/{len(res)} 个 (晚期显著+方向一致+z_diff<2) ===', flush=True)
print(survivors[['因子', 'IC_early', 'ICIR_early', 'IC_late', 'ICIR_late', 'IC_late_3m', 'IC_diff', 'z_diff']].to_string(index=False), flush=True)

print(f'\n=== 对照: 现行18分析对象的存活判定 ===', flush=True)
cur = res[res['因子'].isin(CURRENT18)].set_index('因子').reindex(CURRENT18)
for f, r in cur.iterrows():
    mark = '✅ 存活' if r['存活'] else ('⚠️ 仅晚期' if r['晚期显著'] else ('❌ 方向翻' if not r['方向一致'] else '❌ 晚期不显著'))
    print(f'  {f:<32} early{r["IC_early"]:+.3f}/{r["ICIR_early"]:+.2f} → late{r["IC_late"]:+.3f}/{r["ICIR_late"]:+.2f}  diff{r["IC_diff"]:+.3f} z{r["z_diff"]:+.1f}  {mark}', flush=True)

# === 图: early vs late 散点 ===
fig, ax = plt.subplots(figsize=(10, 9))
x, y = res['IC_early'].astype(float), res['IC_late'].astype(float)
ax.scatter(x, y, s=22, c=['#27ae60' if s else '#95a5a6' for s in res['存活']], alpha=0.65,
           edgecolors=['#145a32' if s else 'none' for s in res['存活']], lw=0.6)
ax.axhline(0, color='k', lw=0.6); ax.axvline(0, color='k', lw=0.6)
lim = 0.28; ax.plot([-lim, lim], [-lim, lim], ls='--', color='#2980b9', lw=1, label='y=x (两期同强度)')
ax.fill_between([-lim, lim], [0.05, 0.05], [lim, lim], color='#27ae60', alpha=0.06)
ax.fill_between([-lim, lim], [-lim, lim], [-0.05, -0.05], color='#27ae60', alpha=0.06, label='晚期显著区(|IC_late|≥0.05)')
for _, r in res.iterrows():
    if r['存活']: ax.annotate(r['因子'].replace('WorldQuant_', 'α'), (r['IC_early'], r['IC_late']), fontsize=7.5, color='#145a32',
                              xytext=(4, 4), textcoords='offset points')
    elif r['因子'] in CURRENT18: ax.annotate(r['因子'].replace('WorldQuant_', 'α'), (r['IC_early'], r['IC_late']), fontsize=7, color='#7f8c8d',
                                             xytext=(4, -8), textcoords='offset points')
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel('IC_early (2017-2019, 小截面噪声大)'); ax.set_ylabel('IC_late (2020-2026, 策略实际交易期)')
ax.set_title(f'137因子分时段IC: 跨时期存活者 {len(survivors)}个(绿) — 右上=一直正, y=x线附近=强度没变', fontsize=12)
ax.legend(fontsize=9, loc='lower right'); ax.grid(alpha=0.2)
plt.tight_layout(); plt.savefig(f'{OUT}/subperiod_ic_scatter.png', dpi=120, bbox_inches='tight'); plt.close()
print('\n存 outputs/subperiod_ic_137.csv + subperiod_ic_scatter.png', flush=True)
