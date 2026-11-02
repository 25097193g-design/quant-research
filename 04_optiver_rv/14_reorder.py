# A: 重建time_id真实时间顺序(第7名哈密顿路径思路的适配版) + 真时序lag特征
# 运行: cd projects/optiver_rv_2021 && conda run -n rqsdk --no-capture-output python 14_reorder.py
# 产出: cache/lag_features.parquet + cache/time_order.parquet + 重建质量诊断
# 逻辑: 每桶取首/末WAP价格向量(112维), d(i→j)=||log(end_i)-log(start_j)||;
#       pred(j)=argmin_i d → j的真实前一桶。布朗运动下真相邻桶价格向量最近。
# 合法性: 预测桶j时前驱桶已走完, 其RV是已实现历史(冠军级思路, 非泄漏)
import time

import numpy as np
import pandas as pd

from common import load_stock, add_wap, list_stock_ids

t0 = time.time()
stocks = list_stock_ids()
sidx = {s: i for i, s in enumerate(stocks)}

rows = []
for sid in stocks:
    b = add_wap(load_stock(sid, 'book'))
    g = b.groupby('time_id')['wap']
    rows.append(pd.DataFrame({'stock': sid, 'time_id': g.first().index,
                              'start': g.first().values, 'end': g.last().values}))
P = pd.concat(rows, ignore_index=True)
times = np.sort(P['time_id'].unique())
tidx = {t: i for i, t in enumerate(times)}
T, S = len(times), len(stocks)
print(f'桶数 T={T} | 股票数 S={S} | 用时 {time.time()-t0:.0f}s')

lS = np.full((T, S), np.nan)
lE = np.full((T, S), np.nan)
lS[[tidx[t] for t in P.time_id], [sidx[s] for s in P.stock]] = np.log(P.start.values)
lE[[tidx[t] for t in P.time_id], [sidx[s] for s in P.stock]] = np.log(P.end.values)
print(f'覆盖缺失: start {np.isnan(lS).sum()} / end {np.isnan(lE).sum()} (填股票均值)')
cmS = np.nanmean(lS, axis=0)
cmE = np.nanmean(lE, axis=0)
lS = np.where(np.isnan(lS), cmS, lS).astype('float32')
lE = np.where(np.isnan(lE), cmE, lE).astype('float32')

# ---- 距离矩阵 + 匈牙利一对一指派 + 环打断 ----
# (贪心argmin有大量撞车: 多个桶抢同一前驱 → 改成最小总代价的一一匹配)
from scipy.optimize import linear_sum_assignment

D = ((lE ** 2).sum(1)[:, None] + (lS ** 2).sum(1)[None, :] - 2 * lE @ lS.T)
D = np.maximum(D, 0).astype(np.float64)
BIG = D.max() * 10
np.fill_diagonal(D, BIG)
ri, ci = linear_sum_assignment(D)  # i→j 一对一, 总代价最小
pred = np.full(T, -1, dtype=np.int64)
pred[ci] = ri                       # pred[j] = j的指派前驱(i)

rng = np.random.default_rng(0)
ii, jj = rng.integers(0, T, 20000), rng.integers(0, T, 20000)
mask = ii != jj
rand_d = np.median(D[ii[mask], jj[mask]])

# 指派是置换 → 会形成环; 在每个环的最贵边处打断(那桶成为"无前驱"起点, 多半真跨日)
broken, seen = 0, np.zeros(T, dtype=bool)
for j0 in range(T):
    if seen[j0]:
        continue
    cyc = []
    j = j0
    while j != -1 and not seen[j]:
        seen[j] = True
        cyc.append(j)
        j = pred[j]
    if j != -1 and j in cyc:  # 回到环内 → 打断最贵边
        c = cyc[cyc.index(j):]
        costs = [D[pred[k], k] for k in c]
        k_break = c[int(np.argmax(costs))]
        pred[k_break] = -1
        broken += 1
pred_dist = np.where(pred >= 0, D[np.clip(pred, 0, T - 1), np.arange(T)], np.nan)
print(f'\n[重建质量] 前驱距离中位 = {np.nanmedian(pred_dist):.6f} | 随机对中位 = {rand_d:.6f} '
      f'(比值 {np.nanmedian(pred_dist)/rand_d:.3f})')
print(f'  环打断 {broken} 个 | 无前驱起点 {int((pred < 0).sum())} 个 (≈天数即合理)')
q75 = np.nanquantile(pred_dist, 0.75)
print(f'  疑似跨日(前驱距离>q75): {int(np.nansum(pred_dist > q75))} 个桶')

# ---- 真时序lag特征 ----
rv = pd.read_parquet('cache/rv_now_full.parquet')  # stock_id,time_id,rv_now
Rv = np.full((T, S), np.nan)
Rv[[tidx[t] for t in rv.time_id], [sidx[s] for s in rv.stock_id]] = rv.rv_now.values

def chain(t_i, k):
    seen = {t_i}
    for _ in range(k):
        t_i = pred[t_i]
        if t_i < 0 or t_i in seen:
            return None
        seen.add(t_i)
    return t_i

def gather_k(k):
    out = np.full((T, S), np.nan)
    for j in range(T):
        p = pred[j] if k == 1 else chain(j, k)
        if p is not None and p >= 0:
            out[j] = Rv[p]
    return out

L1, L2, L3 = gather_k(1), gather_k(2), gather_k(3)
mkt_prev1 = np.nanmedian(L1, axis=1)

feat = []
for ti, t in enumerate(times):
    feat.append({'time_id': t,
                 'pred_time': times[pred[ti]] if pred[ti] >= 0 else -1,
                 'pred_dist': pred_dist[ti],
                 'is_overnight': bool(pred_dist[ti] > q75) if pred[ti] >= 0 else True,
                 'mkt_rv_prev1': mkt_prev1[ti]})
TF = pd.DataFrame(feat)
SF = pd.DataFrame({'time_id': np.repeat(times, S), 'stock_id': np.tile(stocks, T)})
SF['rv_prev1'] = L1.reshape(-1)
SF['rv_prev2'] = L2.reshape(-1)
SF['rv_prev3'] = L3.reshape(-1)
lag = SF.merge(TF, on='time_id').dropna(subset=['rv_prev1']).reset_index(drop=True)
lag.to_parquet('cache/lag_features.parquet')
TF.to_parquet('cache/time_order.parquet')

print(f'\n[lag特征] {len(lag):,}行(有前驱的桶) | 用时 {time.time()-t0:.0f}s')
tgt = pd.read_csv('/Users/lichanghaodemac/Downloads/optiver-realized-volatility-prediction/train.csv')
m = lag.merge(tgt, on=['stock_id', 'time_id']).merge(
    rv[['stock_id', 'time_id', 'rv_now']], on=['stock_id', 'time_id'])
print(f'  ★ corr(rv_prev1, target) = {m.rv_prev1.corr(m.target):.3f}  '
      f'[对照: 乱序相邻编号 -0.013 | 本桶rv_now 0.874]')
print(f'  corr(mkt_rv_prev1, target) = {m.mkt_rv_prev1.corr(m.target):.3f}')
print(f'  corr(rv_prev1, rv_now) = {m.rv_prev1.corr(m.rv_now):.3f}  (重排后真lag聚集)')
