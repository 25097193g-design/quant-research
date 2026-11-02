# A-2: 真时序lag特征进模型 —— B1 vs B1+lag, 及全家族blend
# 运行: cd projects/optiver_rv_2021 && conda run -n rqsdk --no-capture-output python 15_lag_model.py
import lightgbm as lgb
import numpy as np
import pandas as pd

from common import rmspe

df = pd.read_parquet('cache/features_v2.parquet')
lag = pd.read_parquet('cache/lag_features.parquet')
LAG_FEATS = ['rv_prev1', 'rv_prev2', 'rv_prev3', 'mkt_rv_prev1', 'pred_dist', 'is_overnight']
df = df.merge(lag[['stock_id', 'time_id'] + LAG_FEATS], on=['stock_id', 'time_id'], how='inner')
base_feats = [c for c in df.columns if c not in ('stock_id', 'time_id', 'target')] + ['stock_id']
lag_feats = [c for c in base_feats if c not in LAG_FEATS] + LAG_FEATS  # = base ∪ lag
print(f'行数 {len(df):,}(有前驱桶) | lag特征 {LAG_FEATS}')

u = np.sort(df['time_id'].unique())
thr = u[int(len(u) * 0.8)]
tr, va = df[df.time_id < thr], df[df.time_id >= thr].sort_values(
    ['stock_id', 'time_id']).reset_index(drop=True)
tgt = va['target'].values
ytr, yva = np.log(tr['target']), np.log(va['target'])

def feval_log(y_true, y_pred):
    return 'rmspe', rmspe(np.exp(y_pred), np.exp(y_true)), False

def run(feats, tag):
    m = lgb.LGBMRegressor(
        objective='regression', n_estimators=3000, learning_rate=0.05,
        num_leaves=64, min_child_samples=100, feature_fraction=0.85,
        bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
        random_state=42, verbosity=-1, n_jobs=-1)
    m.fit(tr[feats], ytr, eval_set=[(va[feats], yva)], eval_metric=feval_log,
          callbacks=[lgb.early_stopping(150, verbose=False)])
    p = np.exp(m.predict(va[feats], num_iteration=m.best_iteration_))
    print(f'  {tag:<16} RMSPE = {rmspe(p, tgt):.4f}  (best_iter={m.best_iteration_})')
    return m, p

print(f'\n[同折对比] 分界time_id={thr}:')
m_base, p_base = run(base_feats, 'B1底座(复现)')
m_lag, p_lag = run(lag_feats, 'B1+lag(重排)')

imp = pd.Series(m_lag.feature_importances_, index=lag_feats).sort_values(ascending=False)
print('  +lag后重要性Top10:')
for k, v in imp.head(10).items():
    print(f'    {k:<18} {v}')

# ---- 全家族blend(键合并) ----
out = va[['stock_id', 'time_id', 'target']].copy()
out['p_lag'] = p_lag
out['p_base'] = p_base
nn2 = pd.read_parquet('cache/nn2_val_pred.parquet')[['stock_id', 'time_id', 'p_nn2']]
ts = pd.read_parquet('cache/twostage_val_pred.parquet')[['stock_id', 'time_id', 'p_ts', 'p_b1']]
mg = out.merge(nn2, on=['stock_id', 'time_id']).merge(ts, on=['stock_id', 'time_id'])
t = mg['target'].values
L = lambda c: np.log(mg[c].values)
combos = {
    '旧最优 NN2×2+B1+TS': (2*L('p_nn2') + L('p_b1') + L('p_ts')) / 4,
    'LAG+NN2+B1+TS': (L('p_lag') + L('p_nn2') + L('p_b1') + L('p_ts')) / 4,
    'LAG×2+NN2+B1+TS': (2*L('p_lag') + L('p_nn2') + L('p_b1') + L('p_ts')) / 5,
    'LAG×2+NN2×2+B1+TS': (2*L('p_lag') + 2*L('p_nn2') + L('p_b1') + L('p_ts')) / 6,
}
print('\n[blend] (键合并对齐):')
for name, c in combos.items():
    print(f'  {name:<22} RMSPE = {rmspe(np.exp(c), t):.4f}')
mg.to_parquet('cache/lag_val_pred.parquet')
