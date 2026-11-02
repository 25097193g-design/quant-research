"""03 因子的其他关系 — 换手(自相关) / 有效独立维度(PCA) / 5组单调性 / 滚动IC稳定性

承接 02: 18合格因子塌成8簇, 每簇取 |ICIR| 最高的代表 → 8个"独立"因子。
但"独立"≠"有效维度=8": 相关矩阵里仍可能有弱相关结构。PCA 量化到底几个真实独立赌注。
对比 frame_18 用的7因子集, 看它的有效维度(回答"7因子是不是其实只有4个赌注")。

四项体检:
  A. 因子自相关(lag1月, 截面rank) → 换手排序。高自相关=慢因子=交易成本低; 低自相关=快因子=换手贵
  B. PCA有效维度(特征值>1的个数 + participation ratio) → 18因子矩阵 & frame_18的7因子子集
  C. 5组单调性(8代表因子, 按IC符号定向, qcut5, 月均前瞻收益) → 面试官必问"极端值稳不稳"
  D. 滚动36月IC → 合格因子应大多数月份同号, 不能大起大落

输入: 同02 + data/ic_bank_horizon.csv(IC符号)
输出: outputs/factor_autocorr.csv | factor_pca.csv | factor_monotone.csv | factor_monotone.png | factor_rolling_ic.png
"""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
from scipy.stats import spearmanr

DATA = '/Users/lichanghaodemac/quant_strategy/data'
OUT = '/Users/lichanghaodemac/quant_strategy/projects/bank_factor_workflow/outputs'
FACTORS = ['dividend_yield_ttm', 'ev_to_ebitda_ttm', 'pe_ratio_ttm', 'pe_ratio_lyr',
           'beta60', 'beta250', 'net_profit_margin_ttm', 'return_on_equity_ttm',
           'market_cap_3', 'a_share_market_val_in_circulation', 'MOM_12M_1M', 'MOM120',
           'OBV', 'DAVOL20', 'WorldQuant_alpha013', 'WorldQuant_alpha016',
           'WorldQuant_alpha026', 'WorldQuant_alpha088']
FRAME18_7 = ['dividend_yield_ttm', 'beta60', 'ev_to_ebitda_ttm', 'pe_ratio_ttm',
             'market_cap_3', 'net_profit_margin_ttm', 'MOM_12M_1M']
SURVIVORS = ['dividend_yield_ttm', 'ev_to_ebitda_ttm', 'beta60', 'MOM_12M_1M',
             'WorldQuant_alpha016', 'DAVOL20', 'WorldQuant_alpha026', 'WorldQuant_alpha088']  # 8簇代表

# === 复用02的面板构建 ===
price = pd.read_parquet(f'{DATA}/mainboard_price.parquet'); price['date'] = pd.to_datetime(price['date'])
close = price.pivot(index='date', columns='order_book_id', values='close').sort_index()
me = pd.DatetimeIndex(pd.read_parquet(f'{DATA}/monthend_dates.parquet')['monthend'])
mev = [m for m in me if m in close.index]; cm = close.loc[mev]

def beta_p(n):
    rd = close.pct_change(fill_method=None); mk = rd.mean(axis=1)
    return rd.rolling(n).cov(mk).div(mk.rolling(n).var(), axis=0)
self_series = {'MOM_12M_1M': close.shift(20) / close.shift(240) - 1, 'MOM120': close.pct_change(120, fill_method=None),
               'beta60': beta_p(60), 'beta250': beta_p(250)}
self_long = None
for n, s in self_series.items():
    sl = s.loc[[m for m in mev if m in s.index]].stack().rename(n).reset_index()
    sl.columns = ['monthend', 'order_book_id', n]; sl = sl.set_index(['order_book_id', 'monthend'])
    self_long = sl if self_long is None else self_long.join(sl, how='outer')
self_long = self_long.reset_index()
FROMDB = [f for f in FACTORS if f not in self_series]
factors = pd.read_parquet(f'{DATA}/factors_monthend.parquet'); factors['monthend'] = pd.DatetimeIndex(factors['monthend'])
panel = pd.merge(factors[['order_book_id', 'monthend'] + FROMDB], self_long, on=['order_book_id', 'monthend'], how='outer')
univ = pd.read_parquet(f'{DATA}/universe_monthend_pit.parquet'); univ['monthend'] = pd.DatetimeIndex(univ['monthend'])
panel = panel.merge(univ[univ['industry_l1'] == '银行'][['order_book_id', 'monthend']], on=['order_book_id', 'monthend'], how='inner')
panel = panel[['order_book_id', 'monthend'] + FACTORS].sort_values(['monthend', 'order_book_id']).reset_index(drop=True)
fwd = cm.pct_change(fill_method=None).shift(-1)

# IC符号定向(用3m, 回退1m)
ic = pd.read_csv(f'{DATA}/ic_bank_horizon.csv')
icsign = {}
for f in FACTORS:
    row = ic[(ic.factor == f) & (ic.horizon == '3m')]
    if row.empty or pd.isna(row['银行IC'].iloc[0]):   # 先判empty再取iloc, 顺序反了会IndexError
        row = ic[(ic.factor == f) & (ic.horizon == '1m')]
    icsign[f] = np.sign(row['银行IC'].iloc[0]) if len(row) and pd.notna(row['银行IC'].iloc[0]) else 1.0

# === A. 因子自相关(lag1月, 截面rank pooled) → 换手 ===
ac_rows = []
for f in FACTORS:
    pv = panel.pivot(index='monthend', columns='order_book_id', values=f)
    rk = pv.rank(pct=True)
    lag1 = rk.shift(1)
    both = pd.concat([rk.stack(), lag1.stack()], axis=1).dropna()
    ac = both.corr('spearman').iloc[0, 1] if len(both) > 30 else np.nan
    ac_rows.append({'factor': f, 'lag1自相关': round(ac, 2)})
ac_df = pd.DataFrame(ac_rows).sort_values('lag1自相关', ascending=False)
ac_df.to_csv(f'{OUT}/factor_autocorr.csv', index=False, encoding='utf-8-sig')
print('=== A. 因子自相关(lag1月) → 换手排序(高=慢=便宜, 低=快=贵) ===', flush=True)
print(ac_df.to_string(index=False), flush=True)

# === B. PCA有效维度 ===
corr18 = pd.read_csv(f'{OUT}/factor_corr_matrix.csv', index_col=0).reindex(index=FACTORS, columns=FACTORS)
def pca_report(mat, name):
    eig = np.linalg.eigvalsh(mat.values)[::-1]              # 降序
    n_over1 = int((eig > 1).sum())                          # Kaiser: 特征值>1
    pr = (eig.sum() ** 2) / (eig ** 2).sum()                # participation ratio(连续有效维度)
    var80 = int(np.argmax(np.cumsum(eig) / eig.sum() >= 0.8) + 1)
    print(f'  {name:<14}: N={len(eig)}  特征值>1={n_over1}  PR={pr:.1f}  解释80%方差需{var80}个', flush=True)
    return {'集合': name, '因子数': len(eig), '特征值>1': n_over1, '有效维度PR': round(pr, 1), '解释80%所需': var80,
            'top5特征值': [round(e, 2) for e in eig[:5]]}
print('=== B. PCA有效维度(相关矩阵特征值) ===', flush=True)
pca_rows = [pca_report(corr18, '18合格因子'),
            pca_report(corr18.reindex(index=FRAME18_7, columns=FRAME18_7), 'frame18的7因子')]
pd.DataFrame(pca_rows).to_csv(f'{OUT}/factor_pca.csv', index=False, encoding='utf-8-sig')
print('  → "看似N因子, 真实独立赌注远少于N" 是过拟合的核心来源', flush=True)

# === C. 5组单调性(8代表因子) ===
mono_rows = []
mono_data = {}
for f in SURVIVORS:
    sgn = icsign[f]
    pv = (panel.pivot(index='monthend', columns='order_book_id', values=f) * sgn)   # 定向后: 高=好
    rets = []
    for m in mev:
        if m not in pv.index or m not in fwd.index:
            continue
        vals = pv.loc[m].dropna(); fr = fwd.loc[m]
        s = pd.concat([vals, fr], axis=1).dropna()
        if len(s) < 15:
            continue
        s.columns = ['v', 'r']
        try:
            s['g'] = pd.qcut(s['v'].rank(method='first'), 5, labels=False)
        except Exception:
            continue
        grp = s.groupby('g')['r'].mean()
        rets.append(grp)
    if not rets:
        continue
    avg = pd.concat(rets, axis=1).mean(axis=1)
    mono_data[f] = avg.values
    # 单调性得分: Q5-Q1 spread + Spearman(group, return)
    spread = avg.iloc[-1] - avg.iloc[0]
    mono_sp = spearmanr(range(5), avg.values)[0]
    mono_rows.append({'factor': f, 'Q1': round(avg.iloc[0], 4), 'Q2': round(avg.iloc[1], 4),
                      'Q3': round(avg.iloc[2], 4), 'Q4': round(avg.iloc[3], 4), 'Q5': round(avg.iloc[4], 4),
                      'Q5-Q1': round(spread, 4), '单调性ρ': round(mono_sp, 2)})
mono_df = pd.DataFrame(mono_rows).sort_values('Q5-Q1', ascending=False)
mono_df.to_csv(f'{OUT}/factor_monotone.csv', index=False, encoding='utf-8-sig')
print('\n=== C. 5组单调性(8代表, 定向后高=好) ===', flush=True)
print(mono_df.to_string(index=False), flush=True)

# === D. 滚动36月IC(8代表) ===
roll_ic = {}
for f in SURVIVORS:
    sgn = icsign[f]
    pv = panel.pivot(index='monthend', columns='order_book_id', values=f)
    monthly_ic = []
    for m in mev:
        if m not in pv.index or m not in fwd.index:
            continue
        s = pd.concat([pv.loc[m], fwd.loc[m]], axis=1).dropna()
        if len(s) >= 10:
            monthly_ic.append(spearmanr(s.iloc[:, 0], s.iloc[:, 1])[0])
        else:
            monthly_ic.append(np.nan)
    s = pd.Series(monthly_ic, index=[m for m in mev if m in pv.index and m in fwd.index])
    roll_ic[f] = s.rolling(36, min_periods=24).mean()

# === 图1: 单调性柱状图 ===
fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
for ax, f in zip(axes.flat, SURVIVORS):
    if f in mono_data:
        ax.bar(range(1, 6), mono_data[f], color=['#c0392b' if v < 0 else '#27ae60' for v in mono_data[f]])
        ax.set_title(f, fontsize=9); ax.axhline(0, color='k', lw=0.5)
        ax.set_xticks(range(1, 6))
fig.suptitle('银行8代表因子 5组月均前瞻收益(定向后高=好) — 看单调性', fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(f'{OUT}/factor_monotone.png', dpi=120, bbox_inches='tight'); plt.close()

# === 图2: 滚动36月IC ===
fig, ax = plt.subplots(figsize=(14, 6))
for f, s in roll_ic.items():
    s.plot(ax=ax, label=f, lw=1.4)
ax.axhline(0, color='k', lw=0.6); ax.set_title('银行8代表因子 滚动36月IC(时间稳定性)', fontsize=13)
ax.set_ylabel('滚动36月IC均值'); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.2)
plt.tight_layout(); plt.savefig(f'{OUT}/factor_rolling_ic.png', dpi=120, bbox_inches='tight'); plt.close()
print('\n存 factor_monotone.png + factor_rolling_ic.png + factor_autocorr.csv + factor_pca.csv', flush=True)
print('完成 03', flush=True)
