# P2b: LGBM v1 —— 时间切分CV, log空间建模对齐RMSPE
# 运行: cd projects/optiver_rv_2021 && conda run -n rqsdk --no-capture-output python 04_lgbm.py
# 产出: outputs/lgbm_v1_importance.png, pred_vs_target.png, cache/lgbm_v1_val_pred.parquet
# 设计: y=log(target)(RMSPE是相对误差, log空间MSE≈相对误差); stock_id类别特征(6.7倍异质性);
#       验证=全局最后20%个time_id(时间在后, 不shuffle防泄漏); naive同折对比
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
feat_cols = [c for c in df.columns if c not in ('stock_id', 'time_id', 'target')]
df['stock_id'] = df['stock_id'].astype('category')

u = np.sort(df['time_id'].unique())
thr = u[int(len(u) * 0.8)]
tr, va = df[df.time_id < thr], df[df.time_id >= thr]
print(f'时间切分: train {len(tr):,} 行 | val {len(va):,} 行 | 分界 time_id={thr}')

ytr, yva = np.log(tr['target']), np.log(va['target'])

def feval_rmspe(y_true, y_pred):  # sklearn API签名: (labels, preds), 均为log空间
    return 'rmspe', rmspe(np.exp(y_pred), np.exp(y_true)), False

model = lgb.LGBMRegressor(
    objective='regression', n_estimators=3000, learning_rate=0.05,
    num_leaves=64, min_child_samples=100, feature_fraction=0.85,
    bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
    random_state=42, verbosity=-1, n_jobs=-1)
model.fit(tr[feat_cols], ytr, eval_set=[(va[feat_cols], yva)],
          eval_metric=feval_rmspe, callbacks=[lgb.early_stopping(150, verbose=False)])
best = model.best_iteration_

pred = np.exp(model.predict(va[feat_cols], num_iteration=best))
tgt = va['target'].values
m_rmspe = rmspe(pred, tgt)
n_rmspe = rmspe(va['rv_now'].values, tgt)
print(f'\n[P2b] LGBM v1 (best_iter={best})')
print(f'  naive(下桶=本桶) 同折: RMSPE = {n_rmspe:.4f}')
print(f'  ★ LGBM v1:           RMSPE = {m_rmspe:.4f}  (改善 {(1-m_rmspe/n_rmspe)*100:.1f}%)')
print(f'  pred与target相关: Pearson {np.corrcoef(pred, tgt)[0,1]:.3f}')

imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
print('  特征重要性Top10:')
for k, v in imp.head(10).items():
    print(f'    {k:<18} {v}')

fig, ax = plt.subplots(figsize=(8, 6))
lgb.plot_importance(model, max_num_features=20, ax=ax, importance_type='gain')
ax.set_title('LGBM v1 特征重要性(gain)')
fig.tight_layout(); fig.savefig('outputs/lgbm_v1_importance.png', dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 6))
ax.scatter(va['rv_now'], tgt, s=5, alpha=0.2, label='naive(本桶RV)', color='tab:gray')
ax.scatter(pred, tgt, s=5, alpha=0.2, label='LGBM v1', color='tab:blue')
lim = [min(pred.min(), tgt.min()), max(pred.max(), tgt.max())]
ax.plot(lim, lim, 'r--', lw=1, label='y=x')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('预测(下一桶RV)'); ax.set_ylabel('实际 target')
ax.set_title(f'log-log: 模型把点云压向对角线 | RMSPE {n_rmspe:.3f}→{m_rmspe:.3f}')
ax.legend(loc='upper left')
fig.tight_layout(); fig.savefig('outputs/pred_vs_target.png', dpi=130); plt.close(fig)

out = va[['stock_id', 'time_id', 'rv_now', 'target']].copy()
out['pred'] = pred
out.to_parquet('cache/lgbm_v1_val_pred.parquet')
print('\n图: outputs/lgbm_v1_importance.png, outputs/pred_vs_target.png; 预测已存 cache/')
