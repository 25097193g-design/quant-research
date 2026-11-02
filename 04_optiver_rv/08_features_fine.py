# P4b-1: 细粒度簿特征 —— 直指个股残差R(两阶段诊断出的真瓶颈)
# 运行: cd projects/optiver_rv_2021 && conda run -n rqsdk --no-capture-output python 08_features_fine.py
# 产出: cache/features_v3.parquet
# 新增: 100s子窗口RV矩阵(w1..w6)+桶内结构 / trade价RV / 不平衡与价差的稳定性 /
#       trade价对簿价偏离 / 挂单时间跨度 / 个股过去label的expanding均值(只用过去, 部署合法) /
#       相对市场量的平方与交互
import os
import time

import numpy as np
import pandas as pd

from common import load_stock, add_wap, load_train_targets, list_stock_ids

t0 = time.time()
targets = load_train_targets()
rows = []

for i, sid in enumerate(list_stock_ids()):
    book = add_wap(load_stock(sid, 'book'))
    trade = load_stock(sid, 'trade')
    book['logret'] = book.groupby('time_id')['wap'].transform(lambda s: np.log(s).diff())
    book['lr2'] = book['logret'] ** 2
    sec = book['seconds_in_bucket']
    imb1 = (book['bid_size1'] - book['ask_size1']) / (book['bid_size1'] + book['ask_size1'])
    spread_bp = (book['ask_price1'] - book['bid_price1']) / book['wap'] * 1e4
    b = pd.DataFrame({'time_id': book['time_id'], 'sec': sec, 'lr2': book['lr2'],
                      'imb1': imb1, 'spread_bp': spread_bp, 'wap': book['wap']})
    g = b.groupby('time_id')
    f = pd.DataFrame({
        'imb1_std': g['imb1'].std(),
        'spread_bp_std': g['spread_bp'].std(),
        'sec_span': g['sec'].max() - g['sec'].min(),
        'wap_mean': g['wap'].mean(),
    })
    for w in range(6):  # 100s子窗口RV矩阵
        lo, hi = w * 100, (w + 1) * 100
        f[f'rv_w{w+1}'] = np.sqrt(b.loc[(b.sec >= lo) & (b.sec < hi)]
                                  .groupby('time_id')['lr2'].sum())
    wm = f[[f'rv_w{w+1}' for w in range(6)]]
    f['rv_w_std'] = wm.std(axis=1)
    f['rv_trend'] = np.log((wm.iloc[:, 3:].mean(axis=1) + 1e-12)
                           / (wm.iloc[:, :3].mean(axis=1) + 1e-12))  # 后半/前半桶趋势

    # trade价RV(桶内成交价路径的抖动) + 成交价对簿价偏离
    tr = trade.sort_values(['time_id', 'seconds_in_bucket'])
    tlr2 = tr.groupby('time_id')['price'].transform(lambda s: np.log(s).diff()) ** 2
    tstat = pd.DataFrame({
        'rv_trade': np.sqrt(tlr2.groupby(tr['time_id']).sum()),
        'trade_pmean': tr.groupby('time_id')['price'].mean(),
    })
    f = f.join(tstat)
    f['trade_vs_wap'] = (f['trade_pmean'] - f['wap_mean']) / f['wap_mean']
    f = f.drop(columns=['trade_pmean', 'wap_mean']).fillna({'rv_trade': 0})

    t_s = targets[targets.stock_id == sid].set_index('time_id')
    df = f.join(t_s, how='inner').reset_index()
    rows.append(df)
    if (i + 1) % 40 == 0:
        print(f'  {i+1}/112 只, {time.time()-t0:.0f}s')

full = pd.concat(rows, ignore_index=True)

# 市场相对量的非线性 + 个股过去label的expanding均值(严格只用过去time_id)
v2 = pd.read_parquet('cache/features_v2.parquet')
full = v2.merge(full.drop(columns=['target']), on=['stock_id', 'time_id'], how='inner')
full['rv_rel_mkt_sq'] = full['rv_rel_mkt'] ** 2
full['spread_rel_mkt_sq'] = full['spread_rel_mkt'] ** 2
full['log_target'] = np.log(full['target'])
full = full.sort_values('time_id').reset_index(drop=True)
full['enc_stock'] = (full.groupby('stock_id')['log_target']
                     .transform(lambda s: s.expanding().mean().shift(1)))
full['enc_stock'] = full['enc_stock'].fillna(full['log_target'].mean())
full = full.drop(columns=['log_target'])
full.to_parquet('cache/features_v3.parquet')

new_cols = ['rv_w1', 'rv_w2', 'rv_w3', 'rv_w4', 'rv_w5', 'rv_w6', 'rv_w_std', 'rv_trend',
            'rv_trade', 'trade_vs_wap', 'imb1_std', 'spread_bp_std', 'sec_span',
            'rv_rel_mkt_sq', 'spread_rel_mkt_sq', 'enc_stock']
print(f'\n[P4b-1] features_v3: {full.shape[0]:,}行 × {full.shape[1]}列 | 新增{len(new_cols)}个 | '
      f'{time.time()-t0:.0f}s')
nan = full[new_cols].isna().sum()
print(f'  新特征NaN: {dict(nan[nan > 0]) if (nan > 0).any() else "无"}')
print(f'  corr(enc_stock, log target) = '
      f'{full.enc_stock.corr(np.log(full.target)):.3f} (个股自身水平先验)')
