# P4b-2: v3细特征模型 + 与两阶段blend
# 运行: cd projects/optiver_rv_2021 && conda run -n rqsdk --no-capture-output python 09_model_v3.py
import lightgbm as lgb
import numpy as np
import pandas as pd

from common import rmspe

df = pd.read_parquet('cache/features_v3.parquet')
feats = [c for c in df.columns if c not in ('stock_id', 'time_id', 'target')] + ['stock_id']

u = np.sort(df['time_id'].unique())
thr = u[int(len(u) * 0.8)]
tr = df[df.time_id < thr]
va = df[df.time_id >= thr].sort_values(['stock_id', 'time_id']).reset_index(drop=True)
tgt = va['target'].values
ytr, yva = np.log(tr['target']), np.log(va['target'])

def feval_log(y_true, y_pred):
    return 'rmspe', rmspe(np.exp(y_pred), np.exp(y_true)), False

m = lgb.LGBMRegressor(
    objective='regression', n_estimators=3000, learning_rate=0.05,
    num_leaves=64, min_child_samples=100, feature_fraction=0.85,
    bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
    random_state=42, verbosity=-1, n_jobs=-1)
m.fit(tr[feats], ytr, eval_set=[(va[feats], yva)], eval_metric=feval_log,
      callbacks=[lgb.early_stopping(150, verbose=False)])
p_v3 = np.exp(m.predict(va[feats], num_iteration=m.best_iteration_))
print(f'v3(43特征+stock_id) RMSPE = {rmspe(p_v3, tgt):.4f}  (best_iter={m.best_iteration_})')
print('  [对照] B1(v2+stock_id)=0.2493 | 两阶段=0.2489 | blend(B1+两阶段)=0.2458')

ts = pd.read_parquet('cache/twostage_val_pred.parquet').sort_values(
    ['stock_id', 'time_id']).reset_index(drop=True)
# 行序对齐断言(上一次版本这里没对齐, blend全是垃圾数字)
assert (va[['stock_id', 'time_id']].values == ts[['stock_id', 'time_id']].values).all()
p_ts = ts['p_ts'].values
p_b1 = ts['p_b1'].values
p_bl2 = np.exp(0.5 * np.log(p_v3) + 0.5 * np.log(p_ts))
p_bl3 = np.exp((np.log(p_v3) + np.log(p_ts) + np.log(p_b1)) / 3)
print(f'blend(v3+两阶段)        RMSPE = {rmspe(p_bl2, tgt):.4f}')
print(f'blend(v3+两阶段+B1)     RMSPE = {rmspe(p_bl3, tgt):.4f}')

imp = pd.Series(m.feature_importances_, index=feats).sort_values(ascending=False)
print('\n  v3重要性Top15:')
for k, v in imp.head(15).items():
    print(f'    {k:<18} {v}')
out = va[['stock_id', 'time_id', 'target']].copy()
out['p_v3'] = p_v3
out.to_parquet('cache/v3_val_pred.parquet')
