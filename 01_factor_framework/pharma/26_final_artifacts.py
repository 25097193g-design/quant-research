"""26 终版展示件 — Top20月收益序列(存CSV) + 净值曲线图(v2 REPORT主图)

计算 D1-Top20 的自建月收益(与24同配方), 存 outputs/final_top20_monthly_returns.csv(此前各脚本重复计算未存档),
画净值曲线: Top20 vs Q5 vs 医药等权 vs 全A等权 vs 沪深300(bundle指数), 对数轴, 2020-01~2026-06。
输出: outputs/final_top20_monthly_returns.csv + outputs/final_equity_curve.png
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, os, h5py
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK']; matplotlib.rcParams['axes.unicode_minus'] = False
import common

OUT = '/Users/lichanghaodemac/quant_strategy/projects/pharma_factor_workflow/outputs'
COST = 0.0015
CORE7 = ['WorldQuant_alpha040', 'WorldQuant_alpha016', 'WorldQuant_alpha024', 'WorldQuant_alpha044',
         'WorldQuant_alpha011', 'dividend_yield_ttm', 'WorldQuant_alpha026']
SAT5 = ['book_to_market_ratio_lf', 'pb_ratio_lf', 'ps_ratio_ttm', 'VOLT20', 'VOLT60']
FACS = CORE7 + SAT5

P = common.load_panel(FACS)
fwd1 = P['fwd1']; me_sorted = P['me_sorted']; Z = P['Z']; stks = P['stks']
IDX = {m: i for i, m in enumerate(me_sorted)}
IC1 = {f: pd.Series({m: (lambda d: spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0] if len(d) >= 10 and d.iloc[:, 0].nunique() > 1 else np.nan)
                     (pd.concat([Z[f].loc[m], fwd1.loc[m]], axis=1).dropna()) for m in me_sorted if m in fwd1.index}) for f in FACS}

rows = []
for m in me_sorted[36:]:
    if m not in fwd1.index: continue
    i = IDX[m]; win = me_sorted[max(0, i - 12):i]
    ws, ss_ = [], []
    for f in FACS:
        v = IC1[f].reindex(win).dropna()
        if len(v) > 6 and v.std() > 0:
            ws.append(abs(v.mean() / v.std())); ss_.append(1 if v.mean() > 0 else -1)
        else:
            ws.append(0.0); ss_.append(1)
    ws = np.array(ws); ws = ws / ws.sum() if ws.sum() > 0 else np.ones(len(FACS)) / len(FACS)
    z = pd.DataFrame(index=stks)
    for f, s in zip(FACS, ss_): z[f] = s * Z[f].loc[m].reindex(stks)
    sc = (z.fillna(0) * ws).sum(axis=1).dropna()
    if sc.nunique() < 5: continue
    try: q = pd.qcut(sc.rank(method='first'), 5, labels=False)
    except Exception: continue
    q5 = sc[q == 4].index
    lst = list(sc.nlargest(20).index)
    v20 = [fwd1.loc[m, o] for o in lst if o in fwd1.columns and pd.notna(fwd1.loc[m, o])]
    vq5 = [fwd1.loc[m, o] for o in q5 if o in fwd1.columns and pd.notna(fwd1.loc[m, o])]
    if len(v20) >= 3 and len(vq5) >= 3:
        rows.append({'monthend': m, 'top20': np.mean(v20) - COST, 'q5': np.mean(vq5) - COST})

df = pd.DataFrame(rows).set_index('monthend').sort_index()

# 基准序列
univ = pd.read_parquet(f'{common.DATA}/universe_monthend_pit.parquet'); univ['monthend'] = pd.DatetimeIndex(univ['monthend'])
ids = univ[univ['industry_l1'] == '医药生物'].groupby('monthend')['order_book_id'].apply(set).to_dict()
df['pharma_ew'] = pd.Series({m: fwd1.loc[m, [o for o in ss if o in fwd1.columns]].dropna().mean() for m, ss in ids.items()}).reindex(df.index)
df['allA_ew'] = P['mkt1'].reindex(df.index)

with h5py.File(os.path.expanduser('~/.rqalpha/bundle/indexes.h5'), 'r') as f:
    ds = f['000300.XSHG'][:]
d = pd.DataFrame({'date': pd.to_datetime([str(int(x[0]))[:8] for x in ds], format='%Y%m%d'), 'close': [x[2] for x in ds]}).set_index('date')['close'].sort_index()
csi300 = d.resample('ME').last().pct_change().dropna()
df['csi300'] = csi300.reindex(df.index).fillna(0.0)

df.round(5).to_csv(f'{OUT}/final_top20_monthly_returns.csv', encoding='utf-8-sig')

fig, ax = plt.subplots(figsize=(13, 6))
for col, lab, c, lw in [('top20', 'D1-Top20（本策略）', '#c0392b', 2.0), ('q5', 'D1-Q5(前20%≈41只)', '#e67e22', 1.2),
                        ('pharma_ew', '医药等权', '#2980b9', 1.2), ('allA_ew', '全A等权', '#7f8c8d', 1.0),
                        ('csi300', '沪深300', '#95a5a6', 1.0)]:
    cum = (1 + df[col].fillna(0)).cumprod()
    cum.plot(ax=ax, label=f'{lab} ×{cum.iloc[-1]:.1f}', lw=lw, color=c)
ax.set_yscale('log')
ax.set_title('医药 D1-Top20 净值曲线（自建月频, 2020-01~2026-06, 含0.15%/期成本）', fontsize=13)
ax.legend(fontsize=10); ax.grid(alpha=0.2)
plt.tight_layout(); plt.savefig(f'{OUT}/final_equity_curve.png', dpi=120, bbox_inches='tight'); plt.close()

# 集中度柱状图(自建 vs rqalpha 双口径)
kk = pd.read_csv(f'{OUT}/redo_stageK_topN.csv')
self_map = kk[~kk['配置'].str.contains('rqalpha')].set_index('配置')['年化']
rq_map = kk[kk['配置'].str.contains('rqalpha')].set_index('配置')['年化']
sb = [self_map.get('Q5(前20%≈41只)', np.nan), self_map.get('20'), self_map.get('10'), self_map.get('5')]
# Q5的rqalpha T+1出自16号脚本(Stage C主推D1-Q5): 13.40%, 24号实验当时未并入CSV, 此处补齐
rq = [0.1340, rq_map.get('Top20(rqalpha)'), rq_map.get('Top10(rqalpha)'), rq_map.get('Top5(rqalpha)')]
x = np.arange(4); w = 0.38
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.bar(x - w/2, sb, w, color='#e67e22', label='自建（2020-01~2026-06，78期）')
ax.bar(x + w/2, rq, w, color='#2980b9', label='rqalpha T+1（2020-01~2026-05-15）')
ax.axhline(0, color='k', lw=0.6)
ax.set_xticks(x); ax.set_xticklabels(['Q5\n(前20%≈41只)', 'Top20', 'Top10', 'Top5'])
ax.set_ylabel('年化收益'); ax.set_title('持仓集中度倒U型：Top20 是甜点（12因子透明组合，月频）', fontsize=12)
for i, v in enumerate(sb):
    ax.text(i - w/2, v + 0.003, f'{v:.1%}', ha='center', fontsize=9)
for i, v in enumerate(rq):
    if pd.notna(v): ax.text(i + w/2, v + 0.003, f'{v:.1%}', ha='center', fontsize=9)
ax.text(0, rq[0] - 0.028, '(16号脚本)', ha='center', fontsize=7, color='#2980b9')
ax.legend(fontsize=10); ax.grid(alpha=0.2, axis='y')
plt.tight_layout(); plt.savefig(f'{OUT}/concentration_chart.png', dpi=120, bbox_inches='tight'); plt.close()

print(f'存 final_top20_monthly_returns.csv ({len(df)}期) + final_equity_curve.png + concentration_chart.png')
print(df.tail(3).round(4).to_string())
