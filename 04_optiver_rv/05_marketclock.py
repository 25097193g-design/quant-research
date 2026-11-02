# P3: 市场时钟特征 —— 同time_id跨股票聚合(性质②: 市场级共振)
# 运行: cd projects/optiver_rv_2021 && conda run -n rqsdk --no-capture-output python 05_marketclock.py
# 产出: cache/features_v2.parquet + outputs/lgbm_v2_importance.png + 对比数字
# 防泄漏: 只聚合其他股票【本桶可观测量】(rv_now/spread/trade_vol), 不碰任何target(=下一桶=未来);
#         部署时同一时刻全市场本桶簿都是可见的, 信息集合法(比赛公认强特征, 39名方案所谓"leak"即此)
import matplotlib
matplotlib.use('Agg')
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import rmspe

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_parquet('cache/features_v1.parquet')
base_feats = [c for c in df.columns if c not in ('stock_id', 'time_id', 'target')]

# ---------- [1] 市场时钟特征 ----------
agg = df.groupby('time_id').agg(
    mkt_rv_sum=('rv_now', 'sum'),
    mkt_rv_med=('rv_now', 'median'),
    mkt_spread_med=('spread_mean_bp', 'median'),
    mkt_tvol_sum=('trade_vol', 'sum'),
    mkt_n=('rv_now', 'size'),
)
df = df.merge(agg, on='time_id', how='left')
df['mkt_rv_mean_ex'] = (df.mkt_rv_sum - df.rv_now) / (df.mkt_n - 1)  # 剔除自身
df['rv_rel_mkt'] = df.rv_now / df.mkt_rv_med                          # 该股相对市场的抖动倍数
df['spread_rel_mkt'] = df.spread_mean_bp / df.mkt_spread_med
df['tvol_share'] = df.trade_vol / df.mkt_tvol_sum.replace(0, np.nan)
mkt_feats = ['mkt_rv_mean_ex', 'mkt_rv_med', 'mkt_spread_med', 'mkt_tvol_sum',
             'mkt_n', 'rv_rel_mkt', 'spread_rel_mkt', 'tvol_share']
all_feats = base_feats + mkt_feats
df.to_parquet('cache/features_v2.parquet')

print('[1] 市场时钟证据(不看模型的裸相关):')
print(f'  corr(mkt_rv_med, target) = {df.mkt_rv_med.corr(df.target):.3f}  '
      f'(对比: 个股自身 corr(rv_now,target)=0.874)')
mkt_tgt = df.groupby('time_id').agg(m=('target', 'mean'), r=('mkt_rv_med', 'first'))
print(f'  市场层级: corr(本桶全市场RV中位, 下桶全市场target均值) = {mkt_tgt.r.corr(mkt_tgt.m):.3f}')
print(f'  同一time_id平均覆盖 {df.mkt_n.mean():.0f} 只股票')

# ---------- [2] 同折对比: v1特征 vs +市场时钟 ----------
df['stock_id'] = df['stock_id'].astype('category')
u = np.sort(df['time_id'].unique())
thr = u[int(len(u) * 0.8)]
tr, va = df[df.time_id < thr], df[df.time_id >= thr]
ytr, yva = np.log(tr['target']), np.log(va['target'])
tgt = va['target'].values

def feval_rmspe(y_true, y_pred):
    return 'rmspe', rmspe(np.exp(y_pred), np.exp(y_true)), False

def run(feats, tag):
    m = lgb.LGBMRegressor(
        objective='regression', n_estimators=3000, learning_rate=0.05,
        num_leaves=64, min_child_samples=100, feature_fraction=0.85,
        bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
        random_state=42, verbosity=-1, n_jobs=-1)
    m.fit(tr[feats], ytr, eval_set=[(va[feats], yva)], eval_metric=feval_rmspe,
          callbacks=[lgb.early_stopping(150, verbose=False)])
    p = np.exp(m.predict(va[feats], num_iteration=m.best_iteration_))
    s = rmspe(p, tgt)
    print(f'  {tag:<14} RMSPE = {s:.4f}  (best_iter={m.best_iteration_})')
    return m, p

print('\n[2] 同折对比(分界time_id=%d):' % thr)
m1, _ = run(base_feats, f'v1({len(base_feats)}特征)')
m2, p2 = run(all_feats, f'v2(+市场时钟)')

imp = pd.Series(m2.feature_importances_, index=all_feats).sort_values(ascending=False)
print('  v2 特征重要性Top12:')
for k, v in imp.head(12).items():
    print(f'    {k:<18} {v}')
fig, ax = plt.subplots(figsize=(8, 6))
lgb.plot_importance(m2, max_num_features=22, ax=ax, importance_type='gain')
ax.set_title('LGBM v2(v1+市场时钟) 特征重要性(gain)')
fig.tight_layout(); fig.savefig('outputs/lgbm_v2_importance.png', dpi=130); plt.close(fig)

# ---------- [3] stock_31 体检 ----------
print('\n[3] stock_31 体检(P1里naive最差=1.039):')
s31 = df[df.stock_id == 31]
pct = lambda s: (s.rank(pct=True).iloc[0]) if len(s) else np.nan
print(f'  桶数 {len(s31):,} | rv_now中位 {s31.rv_now.median():.5f} '
      f'(全市场分位 {pct(df.groupby("stock_id").rv_now.median()):.2%}) | '
      f'spread中位 {s31.spread_mean_bp.median():.1f}bp '
      f'(分位 {pct(df.groupby("stock_id").spread_mean_bp.median()):.2%}) | '
      f'平均快照/桶 {s31.n_snap.mean():.0f} (全市场均值 {df.n_snap.mean():.0f})')
vp = pd.read_parquet('cache/lgbm_v1_val_pred.parquet')
ps = vp.groupby('stock_id').apply(lambda g: rmspe(g.pred.values, g.target.values))
w = ps.sort_values(ascending=False).head(3)
print(f'  v1验证集分股RMSPE最差3只: ' + ', '.join(f'stock_{int(k)}={v:.3f}' for k, v in w.items()))
