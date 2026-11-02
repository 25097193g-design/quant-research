"""02 因子相关性 — 18个合格因子的截面rank相关时间均值矩阵 + 层次聚类 + 去冗余

核心问题: 面试官最怕你把 5 个高度相关的因子当 5 个独立 alpha → 实盘过拟合、权重翻倍。
本脚本把"哪些因子在讲同一个故事"用数据说清楚。

方法(标准因子相关性口径):
  1. 每个月末, 对银行 41 只做截面, 把每个因子转成截面 rank(剥离绝对量级, OBV/市值这种差几个数量级的也能比)
  2. rank 之间做 Pearson = Spearman; 跨月取均值 → 平均截面相关矩阵
     (跨月均值剥离单月噪音; 比单期相关稳健得多)
  3. |corr|>0.6 的因子对 → 冗余, 二选一(保留 |ICIR| 高的那个)
  4. 层次聚类(距离=1-|corr|) → 看这批因子到底有几个"独立的故事"

输入: data/factors_monthend.parquet(MQ/WQ因子), mainboard_price.parquet(自算4个),
      universe_monthend_pit.parquet(银行白名单), data/ic_bank_horizon.csv(ICIR用于去冗余决策)
输出: outputs/factor_corr_matrix.csv | factor_corr_heatmap.png | factor_redundant_pairs.csv | factor_clusters.csv
"""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

DATA = '/Users/lichanghaodemac/quant_strategy/data'
OUT = '/Users/lichanghaodemac/quant_strategy/projects/bank_factor_workflow/outputs'

# === 18个分析对象(01脚本产出: 1m合格 ∪ 3m合格) ===
FACTORS = ['dividend_yield_ttm', 'ev_to_ebitda_ttm', 'pe_ratio_ttm', 'pe_ratio_lyr',
           'beta60', 'beta250', 'net_profit_margin_ttm', 'return_on_equity_ttm',
           'market_cap_3', 'a_share_market_val_in_circulation', 'MOM_12M_1M', 'MOM120',
           'OBV', 'DAVOL20', 'WorldQuant_alpha013', 'WorldQuant_alpha016',
           'WorldQuant_alpha026', 'WorldQuant_alpha088']
SELF = ['MOM_12M_1M', 'MOM120', 'beta60', 'beta250']   # 需自算
FROMDB = [f for f in FACTORS if f not in SELF]           # 其余从 factors_monthend

# === 构建银行因子面板 ===
price = pd.read_parquet(f'{DATA}/mainboard_price.parquet'); price['date'] = pd.to_datetime(price['date'])
close = price.pivot(index='date', columns='order_book_id', values='close').sort_index()
me = pd.DatetimeIndex(pd.read_parquet(f'{DATA}/monthend_dates.parquet')['monthend'])
mev = [m for m in me if m in close.index]

# 自算4个(口径同 frame_17)
def beta_p(n):
    rd = close.pct_change(fill_method=None); mk = rd.mean(axis=1)
    return rd.rolling(n).cov(mk).div(mk.rolling(n).var(), axis=0)
self_series = {
    'MOM_12M_1M': close.shift(20) / close.shift(240) - 1,
    'MOM120': close.pct_change(120, fill_method=None),
    'beta60': beta_p(60),
    'beta250': beta_p(250),
}
self_long = None
for n, s in self_series.items():
    sl = s.loc[[m for m in mev if m in s.index]].stack().rename(n).reset_index()
    sl.columns = ['monthend', 'order_book_id', n]
    sl = sl.set_index(['order_book_id', 'monthend'])
    self_long = sl if self_long is None else self_long.join(sl, how='outer')
self_long = self_long.reset_index()

factors = pd.read_parquet(f'{DATA}/factors_monthend.parquet'); factors['monthend'] = pd.DatetimeIndex(factors['monthend'])
db = factors[['order_book_id', 'monthend'] + FROMDB]
panel = pd.merge(db, self_long, on=['order_book_id', 'monthend'], how='outer')

univ = pd.read_parquet(f'{DATA}/universe_monthend_pit.parquet'); univ['monthend'] = pd.DatetimeIndex(univ['monthend'])
bank = univ[univ['industry_l1'] == '银行'][['order_book_id', 'monthend']]
panel = panel.merge(bank, on=['order_book_id', 'monthend'], how='inner')
panel = panel[['order_book_id', 'monthend'] + FACTORS]
print(f'银行因子面板: {panel["order_book_id"].nunique()}只 × {panel.monthend.nunique()}月 × {len(FACTORS)}因子, {len(panel)}行', flush=True)

# === 平均截面 Spearman 相关矩阵(每月rank后Pearson, 再跨月均值, NaN-aware) ===
# 个别因子某些月份恒定 → spearman=NaN, 用逐cell计数平均避免传染
acc = np.zeros((len(FACTORS), len(FACTORS))); cnt = np.zeros_like(acc)
for m, g in panel.groupby('monthend'):
    if len(g) < 10:
        continue
    c = g[FACTORS].rank().corr(method='pearson').values     # rank+pearson = spearman
    mask = ~np.isnan(c); acc += np.where(mask, c, 0); cnt += mask
mean_arr = acc / np.where(cnt > 0, cnt, 1)                  # 无重叠月份的cell→0(视为正交)
mean_corr = pd.DataFrame(mean_arr, index=FACTORS, columns=FACTORS)
n_months_corr = int(cnt.max())                              # 有效月份(取最大cell)
mean_corr.to_csv(f'{OUT}/factor_corr_matrix.csv', encoding='utf-8-sig')
print(f'\n存 factor_corr_matrix.csv (跨{n_months_corr}月均值)', flush=True)

# === 冗余对(|corr|>0.6) + 去冗余决策(保留|ICIR|高的) ===
ic = pd.read_csv(f'{DATA}/ic_bank_horizon.csv')
ic1m = ic[ic['horizon'] == '1m'].set_index('factor')['银行ICIR'].abs()
pairs = []
for i in range(len(FACTORS)):
    for j in range(i + 1, len(FACTORS)):
        a, b = FACTORS[i], FACTORS[j]
        c = mean_corr.loc[a, b]
        if abs(c) > 0.6:
            ia, ib = ic1m.get(a, 0), ic1m.get(b, 0)
            keep = a if ia >= ib else b
            drop = b if keep == a else a
            pairs.append({'因子A': a, '因子B': b, '|corr|': round(abs(c), 2), 'corr': round(c, 2),
                          'A的|ICIR|': round(ia, 2), 'B的|ICIR|': round(ib, 2),
                          '保留': keep, '剔除': drop})
redun = pd.DataFrame(pairs).sort_values('|corr|', ascending=False) if pairs else pd.DataFrame()
if len(redun):
    redun.to_csv(f'{OUT}/factor_redundant_pairs.csv', index=False, encoding='utf-8-sig')
    print(f'\n=== 冗余对(|corr|>0.6): {len(redun)} 对 ===', flush=True)
    print(redun[['因子A', '因子B', '|corr|', '保留', '剔除']].to_string(index=False), flush=True)
else:
    print('\n无 |corr|>0.6 的冗余对', flush=True)

# === 层次聚类(距离=1-|corr|, 看"几个独立故事") ===
dist = 1 - mean_corr.abs().values
np.fill_diagonal(dist, 0.0)
dist = (dist + dist.T) / 2                       # 强制对称(数值噪声)
condensed = squareform(dist, checks=False)
Z = linkage(condensed, method='average')
clusters = fcluster(Z, t=0.4, criterion='distance')     # 距离<0.4 即 |corr|>0.6 同簇
cl_df = pd.DataFrame({'factor': FACTORS, 'cluster': clusters}).sort_values(['cluster', 'factor'])
cl_df.to_csv(f'{OUT}/factor_clusters.csv', index=False, encoding='utf-8-sig')
print(f'\n=== 层次聚类 → {cl_df.cluster.nunique()} 个因子组(独立故事) ===', flush=True)
for c, g in cl_df.groupby('cluster'):
    print(f'  组{c}: {", ".join(g["factor"])}', flush=True)

# === 聚类热力图 ===
fig = sns.clustermap(mean_corr, row_linkage=Z, col_linkage=Z, cmap='RdBu_r', vmin=-1, vmax=1,
                     figsize=(13, 11), annot=True, fmt='.2f', annot_kws={'size': 7},
                     cbar_kws={'label': '截面rank相关(跨月均值)'})
fig.fig.suptitle('frame_17→银行18合格因子 相关性矩阵(层次聚类排序)', fontsize=13, y=1.02)
fig.savefig(f'{OUT}/factor_corr_heatmap.png', dpi=120, bbox_inches='tight'); plt.close()
print('\n存 factor_corr_heatmap.png', flush=True)
print('完成 02', flush=True)
