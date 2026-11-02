# P2a: 全112股特征工程 —— 从原始簿/成交自算 ~20 个特征
# 运行: cd projects/optiver_rv_2021 && conda run -n rqsdk --no-capture-output python 03_features.py
# 产出: cache/features_v1.parquet (每行 = stock_id×time_id)
# 特征全部来自【本桶内可观测信息】(比赛设计允许看完整当前桶), 无跨桶泄漏;
# 股票层级不用target encoding(会泄漏), 用 stock_id 类别特征交给LGBM —— P2b
import os
import time

import numpy as np
import pandas as pd

from common import load_stock, add_wap, load_train_targets, list_stock_ids

CACHE = 'cache'
os.makedirs(CACHE, exist_ok=True)

t0 = time.time()
targets = load_train_targets()
STOCKS = list_stock_ids()
rows = []

for i, sid in enumerate(STOCKS):
    book = add_wap(load_stock(sid, 'book'))
    trade = load_stock(sid, 'trade')

    book['logret'] = book.groupby('time_id')['wap'].transform(lambda s: np.log(s).diff())
    book['lr2'] = book['logret'] ** 2
    sec = book['seconds_in_bucket']
    wap = book['wap']
    imb1 = (book['bid_size1'] - book['ask_size1']) / (book['bid_size1'] + book['ask_size1'])
    imb2 = ((book['bid_size1'] + book['bid_size2'] - book['ask_size1'] - book['ask_size2'])
            / (book['bid_size1'] + book['bid_size2'] + book['ask_size1'] + book['ask_size2']))
    spread_bp = (book['ask_price1'] - book['bid_price1']) / wap * 1e4
    b = pd.DataFrame({
        'time_id': book['time_id'], 'sec': sec, 'lr2': book['lr2'],
        'absret': book['logret'].abs(),
        'imb1': imb1, 'imb2': imb2, 'spread_bp': spread_bp,
        'depth': book['bid_size1'] + book['ask_size1'], 'wap': wap,
    })
    g = b.groupby('time_id')
    f = pd.DataFrame({
        'rv_now': np.sqrt(g['lr2'].sum()),
        'rv_firsthalf': np.sqrt(b.loc[b.sec < 300].groupby('time_id')['lr2'].sum()),
        'rv_last300': np.sqrt(b.loc[b.sec >= 300].groupby('time_id')['lr2'].sum()),
        'rv_last60': np.sqrt(b.loc[b.sec >= 540].groupby('time_id')['lr2'].sum()),
        'max_absret': g['absret'].max(),
        'ret_abs': (g['wap'].last() / g['wap'].first()).pipe(np.log).abs(),
        'n_snap': g['sec'].size(),
        'spread_mean_bp': g['spread_bp'].mean(),
        'spread_last_bp': g['spread_bp'].last(),
        'imb1_mean': g['imb1'].mean(),
        'imb1_last': g['imb1'].last(),
        'imb2_mean': g['imb2'].mean(),
        'log_depth_mean': np.log1p(g['depth'].mean()),
    })
    # ---- trade 家族(无成交的桶填0) ----
    t = trade
    tg = t.groupby('time_id')
    tr = pd.DataFrame({
        'trade_vol': tg['size'].sum(),
        'trade_n': tg['size'].size(),
        'trade_orders': tg['order_count'].sum(),
        'trade_avg_size': tg['size'].mean(),
        'trade_pstd': tg['price'].std() / tg['price'].mean(),
        'trade_vol_last60': t.loc[t.seconds_in_bucket >= 540].groupby('time_id')['size'].sum(),
    })
    f = f.join(tr).fillna({'trade_vol': 0, 'trade_n': 0, 'trade_orders': 0,
                           'trade_avg_size': 0, 'trade_pstd': 0, 'trade_vol_last60': 0})

    t_s = targets[targets.stock_id == sid].set_index('time_id')
    df = f.join(t_s, how='inner').reset_index()
    rows.append(df)
    if (i + 1) % 20 == 0:
        print(f'  已处理 {i+1}/{len(STOCKS)} 只, 累计 {time.time()-t0:.0f}s')

full = pd.concat(rows, ignore_index=True)
full.to_parquet(f'{CACHE}/features_v1.parquet')

print(f'\n[P2a] 特征表: {full.shape[0]:,} 行 × {full.shape[1]} 列 | 用时 {time.time()-t0:.0f}s')
feat_cols = [c for c in full.columns if c not in ('stock_id', 'time_id', 'target')]
print(f'  特征 {len(feat_cols)} 个: {feat_cols}')
nan = full[feat_cols].isna().sum()
print(f'  NaN统计(仅列出>0): {dict(nan[nan > 0]) if (nan > 0).any() else "无"}')
print(f'  rv_now 与 target 相关: {full.rv_now.corr(full.target):.3f}')
print(full[['rv_now', 'spread_mean_bp', 'trade_vol', 'target']].describe().to_string())
