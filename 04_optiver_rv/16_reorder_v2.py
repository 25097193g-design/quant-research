# A-v2: 按7th密度重做时间重建 —— 厚指纹 + 路径覆盖(允许链首) + λ自动选 + tick反推
# 运行: cd projects/optiver_rv_2021 && conda run -n rqsdk --no-capture-output python 16_reorder_v2.py
# 产出: cache/lag_features_v2.parquet + cache/time_order_v2.parquet + v1/v2对比诊断
# 三处升级 vs 14_reorder.py:
#   1) 指纹: 首末各5张快照的WAP均值(单张快照噪声大)
#   2) 目标: 带阈值λ的路径覆盖 D'=min(D,λ)指派后丢弃≥λ的边 → 允许~98个链首(日首)无前驱
#   3) λ选择: 只用可观测量 corr(rv_prev1, rv_now) 最大化(不碰label)
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from common import load_stock, add_wap, list_stock_ids

t0 = time.time()
K_SNAP = 5
stocks = list_stock_ids()
sidx = {s: i for i, s in enumerate(stocks)}

rows, ticks = [], {}
for sid in stocks:
    b = add_wap(load_stock(sid, 'book'))
    b['rk_s'] = b.groupby('time_id').cumcount()
    b['rk_e'] = b.groupby('time_id').cumcount(ascending=False)
    g_s = b[b.rk_s < K_SNAP].groupby('time_id')['wap'].mean()
    g_e = b[b.rk_e < K_SNAP].groupby('time_id')['wap'].mean()
    rows.append(pd.DataFrame({'stock': sid, 'time_id': g_s.index,
                              'start': g_s.values, 'end': g_e.values}))
    pu = np.unique(b['bid_price1'].dropna().values)
    dif = np.diff(pu)
    ticks[sid] = dif[dif > 0].min() if len(dif) else np.nan
P = pd.concat(rows, ignore_index=True)
times = np.sort(P['time_id'].unique())
tidx = {t: i for i, t in enumerate(times)}
T, S = len(times), len(stocks)
print(f'桶数 T={T} | 指纹={K_SNAP}快照均值 | 用时 {time.time()-t0:.0f}s')

tk = np.array([ticks[s] for s in stocks])
scale = 0.01 / tk  # 归一化→真实价格的放大倍数(假设美股tick=0.01)
real_px = np.array([P[P.stock == s]['end'].median() for s in stocks]) * scale
print(f'[tick反推] 归一化tick中位={np.nanmedian(tk):.2e} | 恢复的真实股价: '
      f'中位 ${np.nanmedian(real_px):.1f}, 范围 ${np.nanmin(real_px):.1f}~${np.nanmax(real_px):.1f}'
      f' (像真实美股股价→tick恢复成功)')

lS = np.full((T, S), np.nan)
lE = np.full((T, S), np.nan)
lS[[tidx[t] for t in P.time_id], [sidx[s] for s in P.stock]] = np.log(P.start.values)
lE[[tidx[t] for t in P.time_id], [sidx[s] for s in P.stock]] = np.log(P.end.values)
cmS, cmE = np.nanmean(lS, axis=0), np.nanmean(lE, axis=0)
lS = np.where(np.isnan(lS), cmS, lS).astype('float32')
lE = np.where(np.isnan(lE), cmE, lE).astype('float32')

D = ((lE ** 2).sum(1)[:, None] + (lS ** 2).sum(1)[None, :] - 2 * lE @ lS.T)
D = np.maximum(D, 0).astype(np.float64)
np.fill_diagonal(D, D.max() * 10)

# ---- rv矩阵(λ选择与lag特征都要) ----
rv = pd.read_parquet('cache/rv_now_full.parquet')
Rv = np.full((T, S), np.nan)
Rv[[tidx[t] for t in rv.time_id], [sidx[s] for s in rv.stock_id]] = rv.rv_now.values
logRv = np.log(Rv)

def path_cover(lam):
    Dc = np.minimum(D, lam)
    ri, ci = linear_sum_assignment(Dc)
    pred = np.full(T, -1, dtype=np.int64)
    for i, j in zip(ri, ci):
        if D[i, j] < lam:
            pred[j] = i
    broken, seen = 0, np.zeros(T, dtype=bool)   # 残环打断
    for j0 in range(T):
        if seen[j0]:
            continue
        cyc, j = [], j0
        while j != -1 and not seen[j]:
            seen[j] = True
            cyc.append(j)
            j = pred[j]
        if j != -1 and j in cyc:
            c = cyc[cyc.index(j):]
            k_b = c[int(np.argmax([D[pred[k], k] for k in c]))]
            pred[k_b] = -1
            broken += 1
    return pred, broken

def lag_corr(pred, subsample=1500):
    rng = np.random.default_rng(0)
    js = rng.choice(T, subsample, replace=False)
    a, b_ = [], []
    for j in js:
        p = pred[j]
        if p >= 0:
            a.append(logRv[j])
            b_.append(logRv[p])
    a, b_ = np.array(a), np.array(b_)
    m = ~(np.isnan(a) | np.isnan(b_))
    return np.corrcoef(a[m], b_[m])[0, 1]

print('\n[λ选择] (只用rv可观测量, 4个候选):')
offdiag = D[~np.isinf(D)]
best = (0, None, None)
for q in (0.02, 0.05, 0.10, 0.20):
    lam = np.quantile(offdiag, q)
    pred_try, brk = path_cover(lam)
    c = lag_corr(pred_try)
    n_edge = int((pred_try >= 0).sum())
    print(f'  λ=q{q:<5}={lam:.6f} | 保留边 {n_edge}/{T} | lag相关 {c:.4f}')
    if c > best[0]:
        best = (c, lam, pred_try)
c, lam, pred = best
print(f'  → 选定 λ={lam:.6f} (lag相关 {c:.4f}; v1版只有0.650)')

# ---- 链结构诊断(应现出"日"形状: ~98条链×~39桶) ----
depth = np.zeros(T, dtype=int)
for j in range(T):
    k, d, seen2 = j, 0, set()
    while pred[k] != -1 and k not in seen2:
        seen2.add(k)
        k = pred[k]
        d += 1
    depth[j] = d
print(f'[链结构] 链数(无前驱起点) = {int((pred < 0).sum())} | 链深分布: '
      f'中位 {np.median(depth):.0f} | p90 {np.quantile(depth, .9):.0f}')
print(f'  (若≈98条链×~39桶 → 日结构被恢复)')

# ---- lag特征v2 ----
pred_dist = np.where(pred >= 0, D[np.clip(pred, 0, T - 1), np.arange(T)], np.nan)

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
TF = pd.DataFrame({'time_id': times,
                   'pred_time': [times[pred[i]] if pred[i] >= 0 else -1 for i in range(T)],
                   'pred_dist': pred_dist,
                   'is_overnight': [True if pred[i] < 0 else bool(pred_dist[i] > lam)
                                    for i in range(T)],
                   'mkt_rv_prev1': mkt_prev1})
SF = pd.DataFrame({'time_id': np.repeat(times, S), 'stock_id': np.tile(stocks, T)})
SF['rv_prev1'], SF['rv_prev2'], SF['rv_prev3'] = L1.reshape(-1), L2.reshape(-1), L3.reshape(-1)
lag = SF.merge(TF, on='time_id').dropna(subset=['rv_prev1']).reset_index(drop=True)
lag.to_parquet('cache/lag_features_v2.parquet')
TF.to_parquet('cache/time_order_v2.parquet')

tgt = pd.read_csv('/Users/lichanghaodemac/Downloads/optiver-realized-volatility-prediction/train.csv')
m = lag.merge(tgt, on=['stock_id', 'time_id']).merge(
    rv[['stock_id', 'time_id', 'rv_now']], on=['stock_id', 'time_id'])
print(f'\n[v1 vs v2 对比] (有前驱行数 {len(m):,})')
print(f'  corr(rv_prev1, rv_now) = {m.rv_prev1.corr(m.rv_now):.3f}   [v1: 0.650]')
print(f'  corr(rv_prev1, target) = {m.rv_prev1.corr(m.target):.3f}   [v1: 0.592]')
print(f'  corr(mkt_rv_prev1, target) = {m.mkt_rv_prev1.corr(m.target):.3f} [v1: 0.464]')
