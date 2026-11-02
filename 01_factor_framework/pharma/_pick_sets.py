"""_pick_sets.py — 数据驱动定医药 F4/F7(每簇选|ICIR|最高代表, 再按强度取前N)

原则(和bank一致): 去冗余 = 每簇一个代表, 但bank是手工按"故事"挑;
医药24因子→19簇, 大部分WQ alpha互相独立, 手工挑易有争议 → 用|ICIR|(1m/3m均值)排序取前N, 可复现。
输出: 打印 CLUSTERS/F4/NEG4/F7/NEG7, 供填回 common.py
"""
import pandas as pd, numpy as np

DATA = '/Users/lichanghaodemac/quant_strategy/data'
OUT = '/Users/lichanghaodemac/quant_strategy/projects/pharma_factor_workflow/outputs'

ic = pd.read_csv(f'{DATA}/ic_pharma_horizon.csv')
cl = pd.read_csv(f'{OUT}/factor_clusters.csv')
ic = ic[ic['horizon'].isin(['1m', '3m'])]

# 每簇: 按 1m∪3m 平均|ICIR| 取代表
best = []
for c in sorted(cl['cluster'].unique()):
    facs = cl[cl['cluster'] == c]['factor'].tolist()
    sub = ic[ic['factor'].isin(facs)]
    g = sub.groupby('factor')['医药ICIR'].agg(lambda s: np.nanmean(s.abs())).sort_values(ascending=False)
    rep = g.index[0]
    best.append({'cluster': c, 'rep': rep, 'ICIR': round(g.iloc[0], 3), 'cluster_factors': facs})

bdf = pd.DataFrame(best).sort_values('ICIR', ascending=False).reset_index(drop=True)
print('=== 19簇代表 按平均|ICIR| 排序 ===')
for _, r in bdf.iterrows():
    print(f'  簇{r["cluster"]:<3} {r["rep"]:<38} avg|ICIR|={r["ICIR"]:.3f}  [{", ".join(r["cluster_factors"])}]')

F7 = bdf['rep'].head(7).tolist()
F4 = bdf['rep'].head(4).tolist()

# NEG: IC(1m均值)为负 → 需要反向(取负z)
def neg(facs, h='1m'):
    out = set()
    for f in facs:
        v = ic[(ic['factor'] == f) & (ic['horizon'] == h)]['医药IC']
        if len(v) and v.iloc[0] < 0: out.add(f)
    return out

print(f'\nF4 = {F4}\nNEG4 = {sorted(neg(F4))}')
print(f'F7 = {F7}\nNEG7 = {sorted(neg(F7))}')
# 全24因子 NEG(给04用: 24F ICIR加权也要符号对齐)
ALL24 = sorted(set(cl['factor']))
print(f'\nALL24 = {ALL24}\nNEG24 = {sorted(neg(ALL24))}')
