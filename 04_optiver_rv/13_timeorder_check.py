# 验证: time_id编号顺序 ≈ 真实时间顺序吗? (第7名逆向工程重排时间→怀疑官方顺序被打乱)
# 逻辑: 若编号顺序=时间顺序, 相邻编号桶的RV应显著相关(波动聚集);
#       若被打乱, 相邻编号相关 ≈ 随机打乱后的相关
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
df = pd.read_parquet('cache/rv_now_full.parquet')

lag_c, rnd_c = [], []
for sid, g in df.groupby('stock_id'):
    s = np.log(g.sort_values('time_id')['rv_now'].values)
    if len(s) < 50:
        continue
    lag_c.append(np.corrcoef(s[:-1], s[1:])[0, 1])
    sp = rng.permutation(s)
    rnd_c.append(np.corrcoef(sp[:-1], sp[1:])[0, 1])

mkt = df.groupby('time_id')['rv_now'].median().sort_index()
ms = np.log(mkt.values)
m_lag = np.corrcoef(ms[:-1], ms[1:])[0, 1]
m_rnd = np.corrcoef(rng.permutation(ms)[:-1], rng.permutation(ms)[1:])[0, 1]

print('[个股层] 112只股票相邻编号桶 logRV 相关:')
print(f'  编号顺序 lag1 相关 中位 = {np.median(lag_c):.3f}')
print(f'  随机打乱后 lag1 相关 中位 = {np.median(rnd_c):.3f}   (两者接近=编号顺序无时间信息)')
print(f'\n[市场层] 全市场RV中位数序列:')
print(f'  编号顺序 lag1 = {m_lag:.3f} | 随机打乱 lag1 = {m_rnd:.3f}')
print('\n判定: 差值大→编号≈时间顺序(enc_stock合法); 差值≈0→编号被打乱(enc_stock混入未来, 需重做)')
