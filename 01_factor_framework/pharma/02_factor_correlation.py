"""02 医药因子相关性 + 层次聚类 — 10合格因子 → 几个独立故事? 去冗余定pharma 4F

对比银行: 医药因子结构不同(alpha040通吃 + surplus_cash板块特异), 聚类看是否也塌成几簇。
输出: outputs/factor_corr_matrix.csv + factor_corr_heatmap.png + factor_clusters.csv
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK']; matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import common

OUT = '/Users/lichanghaodemac/quant_strategy/projects/pharma_factor_workflow/outputs'
P = common.load_panel()
FACS = P['facs']
print(f'医药面板: {len(P["stks"])}只 × {len(P["me_sorted"])}月 × {len(FACS)}因子', flush=True)

# 长表用于groupby
allf_long = pd.concat([P['WIDE'][f].stack().rename(f) for f in FACS], axis=1).reset_index()
allf_long.columns = ['monthend', 'order_book_id'] + FACS
acc = np.zeros((len(FACS), len(FACS))); cnt = np.zeros_like(acc)
for m, g in allf_long.groupby('monthend'):
    if len(g) < 30: continue
    c = g[FACS].rank().corr(method='pearson').values
    mk = ~np.isnan(c); acc += np.where(mk, c, 0); cnt += mk
mean_corr = pd.DataFrame(acc / np.where(cnt > 0, cnt, 1), index=FACS, columns=FACS)
mean_corr.to_csv(f'{OUT}/factor_corr_matrix.csv', encoding='utf-8-sig')

# 冗余对
redun = [(FACS[i], FACS[j], round(mean_corr.iloc[i, j], 2)) for i in range(len(FACS)) for j in range(i + 1, len(FACS)) if abs(mean_corr.iloc[i, j]) > 0.6]
print(f'\n|corr|>0.6 冗余对: {len(redun)}', flush=True)
for a, b, c in redun: print(f'  {a} ~ {b}: {c}', flush=True)

# 层次聚类
dist = 1 - mean_corr.abs().values; np.fill_diagonal(dist, 0); dist = (dist + dist.T) / 2
Z = linkage(squareform(dist, checks=False), method='average')
cl = fcluster(Z, t=0.4, criterion='distance')
print(f'\n→ {len(set(cl))} 个独立簇:', flush=True)
for c in sorted(set(cl)):
    print(f'  组{c}: {", ".join(FACS[i] for i in range(len(FACS)) if cl[i] == c)}', flush=True)
pd.DataFrame({'factor': FACS, 'cluster': cl}).to_csv(f'{OUT}/factor_clusters.csv', index=False, encoding='utf-8-sig')

fig = sns.clustermap(mean_corr, row_linkage=Z, col_linkage=Z, cmap='RdBu_r', vmin=-1, vmax=1,
                     figsize=(9, 7), annot=True, fmt='.2f', annot_kws={'size': 8})
fig.fig.suptitle(f'医药{len(FACS)}合格因子 相关性(聚类排序)', y=1.02, fontsize=12)
fig.savefig(f'{OUT}/factor_corr_heatmap.png', dpi=110, bbox_inches='tight'); plt.close()
print('\n存 factor_corr_matrix.csv + factor_corr_heatmap.png + factor_clusters.csv', flush=True)
