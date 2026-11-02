# 数据复盘: 比赛数据全景实测(文件清单/train.csv结构/单股簿+成交解剖/跨股time_id共振)
# 运行: cd projects/optiver_rv_2021 && conda run -n rqsdk --no-capture-output python 21_data_review.py
import os

import numpy as np
import pandas as pd

from common import BASE, add_wap, list_stock_ids, load_stock, rv_by_timeid

print('=' * 70)
print('1) 原始数据文件清单')
print('=' * 70)
for name in sorted(os.listdir(BASE)):
    p = os.path.join(BASE, name)
    if os.path.isdir(p):
        n_stocks = len(os.listdir(p))
        print(f'  {name:<28} 目录, {n_stocks} 个 stock_id 子目录')
    else:
        print(f'  {name:<28} {os.path.getsize(p) / 1e6:8.1f} MB')

print()
print('=' * 70)
print('2) train.csv (答案表: target=下一桶RV)')
print('=' * 70)
train = pd.read_csv(f'{BASE}/train.csv')
print(f'  行数={len(train):,}  列={list(train.columns)}')
print(f'  股票数={train.stock_id.nunique()}  time_id数={train.time_id.nunique():,}')
print(f'  target分位数: ' + ' | '.join(
    f'{q}:{np.quantile(train.target, q):.5f}' for q in (.05, .25, .5, .75, .95)))
print(f'  target范围 [{train.target.min():.5f}, {train.target.max():.5f}]  (log空间跨{np.log(train.target.max()/train.target.min()):.1f}个自然单位)')
per_stock = train.groupby('stock_id')['target'].median()
print(f'  各股target中位数: 最小{per_stock.min():.5f}(stock_{per_stock.idxmin()}) '
      f'~ 最大{per_stock.max():.5f}(stock_{per_stock.idxmax()}), 倍数{per_stock.max()/per_stock.min():.1f}x')

print()
print('=' * 70)
print('3) 单股解剖: stock_0 (book + trade)')
print('=' * 70)
book = load_stock(0, 'book')
trade = load_stock(0, 'trade')
print(f'  book:  {len(book):,}行  列={list(book.columns)}')
print(f'  trade: {len(trade):,}行  列={list(trade.columns)}')
print(f'  book time_id数={book.time_id.nunique():,}  seconds_in_bucket: {book.seconds_in_bucket.min()}-{book.seconds_in_bucket.max()}')
snaps = book.groupby('time_id').size()
print(f'  每桶快照数: 中位{snaps.median():.0f} (P10={snaps.quantile(.1):.0f}, P90={snaps.quantile(.9):.0f})  ← 事件驱动非等间隔')
tsec = trade.groupby('time_id')['seconds_in_bucket'].agg(tmin='min', tmax='max', n='count')
print(f'  trade每桶成交笔数中位{tsec["n"].median():.0f}  首笔最早出现在第{tsec["tmin"].median():.0f}秒')
print()
print('  book 第一个桶前3行:')
print(book[book.time_id == book.time_id.iloc[0]].head(3).to_string(index=False))
print()
print('  trade 第一个桶前3行:')
print(trade[trade.time_id == trade.time_id.iloc[0]].head(3).to_string(index=False))
wap = add_wap(book)
rv = rv_by_timeid(wap)
print(f'  stock_0 本桶RV: 中位{rv.median():.5f} (WAP管线复算)')

print()
print('=' * 70)
print('4) 跨股共振: 同一time_id在不同股票 = 同一市场时段')
print('=' * 70)
# 抽3只股票, 看它们两两在同一time_id上的本桶RV相关性(市场时钟的裸证据)
ids = [0, 31, 77]
rvs = {}
for sid in ids:
    b = add_wap(load_stock(sid, 'book'))
    rvs[sid] = rv_by_timeid(b)
m = pd.DataFrame(rvs).dropna()
print(f'  共同time_id数={len(m):,}; 两两RV相关(Pearson):')
for i in range(len(ids)):
    for j in range(i + 1, len(ids)):
        r = np.corrcoef(np.log(m[ids[i]]), np.log(m[ids[j]]))[0, 1]
        print(f'    stock_{ids[i]:<3} vs stock_{ids[j]:<3}: r={r:.3f}')
print('  → 相关显著为正 = 同time_id跨股票共享市场时段(性质②市场时钟的根据)')
