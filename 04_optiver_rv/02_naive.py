# P1: 全112股 naive baseline —— 下一桶RV = 本桶RV
# 运行: cd projects/optiver_rv_2021 && conda run -n rqsdk --no-capture-output python 02_naive.py
# 产出: cache/rv_now_full.parquet(P2复用) + outputs/naive_perstock.csv + 控制台锚点数字
import os
import time

import numpy as np
import pandas as pd

from common import (load_stock, add_wap, rv_by_timeid, load_train_targets,
                    rmspe, list_stock_ids)

CACHE = 'cache'
OUT = 'outputs'
os.makedirs(CACHE, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

t_start = time.time()
targets = load_train_targets()
STOCKS = list_stock_ids()
rows, per_stock = [], []
print(f'训练集 {len(STOCKS)} 只股票(编号非连续: {STOCKS[0]}~{STOCKS[-1]})')

for i, sid in enumerate(STOCKS):
    try:
        book = add_wap(load_stock(sid, 'book'))
    except IndexError:  # 该股票文件夹缺失/为空
        print(f'  ⚠ stock_{sid} 数据缺失, 跳过')
        continue
    rv = rv_by_timeid(book)
    t_s = targets[targets.stock_id == sid].set_index('time_id')
    df = rv.to_frame().join(t_s, how='inner').reset_index()  # join后自带stock_id列
    rows.append(df)
    per_stock.append({
        'stock_id': sid,
        'n_buckets': len(df),
        'rv_median': df.rv_now.median(),
        'pearson': df.rv_now.corr(df.target),
        'rmspe': rmspe(df.rv_now.values, df.target.values),
    })
    if (i + 1) % 20 == 0:
        print(f'  已处理 {i+1}/{len(STOCKS)} 只, 累计 {time.time()-t_start:.0f}s')

full = pd.concat(rows, ignore_index=True)
full.to_parquet(f'{CACHE}/rv_now_full.parquet')  # P2特征工程直接读, 不用重算簿
ps = pd.DataFrame(per_stock)
ps.to_csv(f'{OUT}/naive_perstock.csv', index=False)

# ---------- 锚点数字 ----------
pooled_rmspe = rmspe(full.rv_now.values, full.target.values)
pooled_pear = full.rv_now.corr(full.target)
print(f'\n[P1] 全量 naive(下一桶RV=本桶RV)')
print(f'  样本: {len(full):,} 行 ({full.stock_id.nunique()}只股票) | 用时 {time.time()-t_start:.0f}s')
print(f'  ★ 比赛级 naive RMSPE = {pooled_rmspe:.4f}   (社区公认锚点≈0.34)')
print(f'  汇总Pearson相关 = {pooled_pear:.3f}')
print(f'  分股票RMSPE: 中位 {ps.rmspe.median():.3f} | '
      f'p25 {ps.rmspe.quantile(.25):.3f} ~ p75 {ps.rmspe.quantile(.75):.3f} | '
      f'最好 {ps.rmspe.min():.3f}(stock_{int(ps.loc[ps.rmspe.idxmin(),"stock_id"])}) | '
      f'最差 {ps.rmspe.max():.3f}(stock_{int(ps.loc[ps.rmspe.idxmax(),"stock_id"])})')
print(f'  股票异质性(性质3): 各股RV中位数 min {ps.rv_median.min():.5f} ~ '
      f'max {ps.rv_median.max():.5f}, 相差 {ps.rv_median.max()/ps.rv_median.min():.1f} 倍')
