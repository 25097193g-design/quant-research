# 共享模块: 数据加载 + WAP/RV 计算
# P2以后所有特征/模型脚本复用这里, 防止同一逻辑多脚本drift(bank项目教训)
import glob
import re

import numpy as np
import pandas as pd

# 数据已在本地(hive分区格式: stock_id=N/*.parquet)
BASE = '/Users/lichanghaodemac/Downloads/optiver-realized-volatility-prediction'


def list_stock_ids(kind='book'):
    """实际存在的股票编号: 非连续(0~126缺15个, 缺的只在测试集), train共112只"""
    dirs = glob.glob(f'{BASE}/{kind}_train.parquet/stock_id=*')
    return sorted(int(re.search(r'stock_id=(\d+)$', d).group(1)) for d in dirs)


def load_stock(stock_id, kind='book'):
    """kind: 'book'(订单簿快照) 或 'trade'(成交记录); 返回该股票全部数据, 按桶内时间排序"""
    paths = sorted(glob.glob(f'{BASE}/{kind}_train.parquet/stock_id={stock_id}/*.parquet'))
    df = pd.concat([pd.read_parquet(p) for p in paths])
    return df.sort_values(['time_id', 'seconds_in_bucket']).reset_index(drop=True)


def add_wap(book):
    """簿快照 → WAP: 用对面挂单量加权 (bid_p1*ask_s1 + ask_p1*bid_s1)/(bid_s1+ask_s1)
    分母为0的退化快照置NaN(不静默替换)"""
    denom = book['bid_size1'] + book['ask_size1']
    out = book.copy()
    out['wap'] = ((book['bid_price1'] * book['ask_size1'] + book['ask_price1'] * book['bid_size1'])
                  / denom).where(denom > 0)
    return out


def rv_by_timeid(book_wap):
    """每个time_id(10分钟桶)的已实现波动率: sqrt(Σ log(WAP差分)^2), 不减均值"""
    lr = book_wap.groupby('time_id')['wap'].transform(lambda s: np.log(s).diff())
    return np.sqrt((lr ** 2).groupby(book_wap['time_id']).sum()).rename('rv_now')


def load_train_targets():
    """train.csv: stock_id, time_id, target(=该桶下一桶的RV, 比赛官方给的答案)"""
    return pd.read_csv(f'{BASE}/train.csv')


def rmspe(pred, target):
    """比赛官方评分: 相对误差的均方根"""
    return float(np.sqrt(np.mean(((pred - target) / target) ** 2)))
