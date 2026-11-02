# P0 EDA: 打开订单簿, 算WAP/RV, 亲手验证波动聚集
# 运行: cd projects/optiver_rv_2021 && conda run -n rqsdk --no-capture-output python 01_eda.py
# 产出: outputs/book_evolution.png, wap_path.png, rv_scatter.png + 控制台数字
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common import load_stock, add_wap, rv_by_timeid, load_train_targets, rmspe

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

OUT = 'outputs'
os.makedirs(OUT, exist_ok=True)
STOCK = 0

# ---------- [1] 数据规模 ----------
book = load_stock(STOCK, 'book')
trade = load_stock(STOCK, 'trade')
targets = load_train_targets()
t0 = targets[targets.stock_id == STOCK]
print(f'[1] stock_{STOCK}: 簿快照 {len(book):,} 行 | 成交 {len(trade):,} 行 | '
      f'{book.time_id.nunique()} 个10分钟桶 | train.csv答案 {len(t0)} 行')

# ---------- [2] 挑快照最多的桶, 打开簿看演化 ----------
cnt = book.groupby('time_id').size()
show_id = cnt.idxmax()
b = book[book.time_id == show_id].copy()
tr = trade[trade.time_id == show_id]
b = add_wap(b)
rv_show = float(np.sqrt((np.log(b['wap']).diff() ** 2).sum()))
tgt_show = float(t0.loc[t0.time_id == show_id, 'target'].iloc[0])
spread = (b.ask_price1 - b.bid_price1) / b['wap']

print(f'\n[2] 展示桶 time_id={show_id}: {len(b)} 张快照, '
      f'价差中位数 {spread.median()*1e4:.1f} bp')
print('    前3张快照(挂单表):')
cols = ['seconds_in_bucket', 'bid_price1', 'bid_size1', 'ask_price1', 'ask_size1', 'wap']
print(b[cols].head(3).to_string(index=False,
      formatters={c: (lambda v: f'{v:.6f}') for c in ['bid_price1', 'ask_price1', 'wap']}))
print(f'    本桶RV(我算) = {rv_show:.6f} | target(下一桶RV, 官方) = {tgt_show:.6f}')

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                         gridspec_kw={'height_ratios': [2, 1]})
ax = axes[0]
ax.plot(b.seconds_in_bucket, b.bid_price1, label='bid_price1(最优买价)', lw=1)
ax.plot(b.seconds_in_bucket, b.ask_price1, label='ask_price1(最优卖价)', lw=1)
ax.fill_between(b.seconds_in_bucket, b.bid_price1, b.ask_price1, alpha=0.15, label='价差spread')
ax.set_ylabel('价格'); ax.legend(loc='best'); ax.set_title(
    f'stock_{STOCK} time_id={show_id} 订单簿演化({len(b)}张快照)')
ax = axes[1]
ax.plot(b.seconds_in_bucket, b.bid_size1, label='bid_size1(买方挂单量)', lw=1)
ax.plot(b.seconds_in_bucket, b.ask_size1, label='ask_size1(卖方挂单量)', lw=1)
ax.set_ylabel('挂单量(股)'); ax.set_xlabel('窗口内秒数 seconds_in_bucket'); ax.legend(loc='best')
fig.tight_layout(); fig.savefig(f'{OUT}/book_evolution.png', dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(b.seconds_in_bucket, b['wap'], color='tab:blue', lw=1.2, label='WAP路径(簿算出的价格)')
if len(tr):
    ax.scatter(tr.seconds_in_bucket, tr.price, s=np.sqrt(tr['size']) * 1.5,
               color='tab:red', alpha=0.5, label='实际成交(点大小∝成交量)')
ax.set_xlabel('窗口内秒数'); ax.set_ylabel('价格')
ax.set_title(f'WAP路径 vs 实际成交 | stock_{STOCK} time_id={show_id} | 本桶RV={rv_show:.6f}')
ax.legend(loc='best'); fig.tight_layout()
fig.savefig(f'{OUT}/wap_path.png', dpi=130); plt.close(fig)

# ---------- [3] 全部桶: 本桶RV vs 下一桶RV(波动聚集验证) ----------
book = add_wap(book)
rv_all = rv_by_timeid(book).to_frame()
df = rv_all.join(t0.set_index('time_id'), how='inner')
pear = df.rv_now.corr(df.target)
spear = df.rv_now.corr(df.target, method='spearman')
print(f'\n[3] stock_{STOCK} 全部 {len(df)} 个桶: 本桶RV vs target(下一桶RV)')
print(f'    Pearson相关 = {pear:.3f} | Spearman相关 = {spear:.3f}  (高=波动聚集, 预测有物理基础)')
print(f'    RV中位数 = {df.rv_now.median():.6f} | target中位数 = {df.target.median():.6f}')

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
for ax, xscale, yscale, ttl in [
        (axes[0], 'linear', 'linear', '线性坐标'),
        (axes[1], 'log', 'log', 'log-log(右偏校正后带状更清晰)')]:
    ax.scatter(df.rv_now, df.target, s=6, alpha=0.25, color='tab:blue')
    lim = [df[['rv_now', 'target']].min().min(), df[['rv_now', 'target']].max().max()]
    ax.plot(lim, lim, 'r--', lw=1, label='y=x(完美预测)')
    ax.set_xscale(xscale); ax.set_yscale(yscale)
    ax.set_xlabel('本桶RV(可观测)'); ax.set_ylabel('target = 下一桶RV(要预测的)')
    ax.set_title(f'{ttl} | Pearson={pear:.3f}')
    ax.legend(loc='upper left')
fig.suptitle(f'stock_{STOCK} 波动聚集验证: {len(df)}个10分钟桶')
fig.tight_layout(); fig.savefig(f'{OUT}/rv_scatter.png', dpi=130); plt.close(fig)

# ---------- [4] naive预演: 下一桶=本桶, stock_0上的RMSPE ----------
score = rmspe(df.rv_now.values, df.target.values)
print(f'\n[4] naive预测(下一桶RV=本桶RV) 在 stock_{STOCK}: RMSPE = {score:.4f}')
print('    (全112股比赛级naive≈0.34, 单股会偏离; P1将跑全量)')

print(f'\n图已存 {OUT}/: book_evolution.png, wap_path.png, rv_scatter.png')
