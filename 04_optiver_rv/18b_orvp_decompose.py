# 拆解第7名的4pp优势: 特征贡献 vs 加权技巧, 及与全家族blend
# 运行: cd projects/optiver_rv_2021 && conda run -n rqsdk --no-capture-output python 18b_orvp_decompose.py
import lightgbm as lgb
import numpy as np
import pandas as pd

from common import rmspe

def feval_log(y_true, y_pred):
    return 'rmspe', rmspe(np.exp(y_pred), np.exp(y_true)), False

def feval_raw(y_true, y_pred):
    return 'rmspe', rmspe(y_true, y_pred), False

ORVP_PARAMS = dict(objective='regression', n_estimators=1000, learning_rate=0.05,
                   min_child_samples=500, feature_fraction=0.5, subsample=0.72,
                   subsample_freq=4, lambda_l1=0.5, lambda_l2=1.0, max_bin=100,
                   random_state=2021, verbosity=-1, n_jobs=-1)
OUR_PARAMS = dict(objective='regression', n_estimators=3000, learning_rate=0.05,
                  num_leaves=64, min_child_samples=100, feature_fraction=0.85,
                  bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
                  random_state=42, verbosity=-1, n_jobs=-1)

his = pd.read_parquet('cache/orvp_features.parquet')
ours = pd.read_parquet('cache/features_v2.parquet')

def split(df):
    u = np.sort(df['time_id'].unique())
    thr = u[int(len(u) * 0.8)]
    tr = df[df.time_id < thr]
    va = df[df.time_id >= thr].sort_values(['stock_id', 'time_id']).reset_index(drop=True)
    return tr, va, thr

# ---- A: 他的特征 × 我们的split, 两种建模方式 ----
hf = [c for c in his.columns if c not in {'time_id', 'target', 'row_id'}]
htr, hva, thr = split(his)
t = hva['target'].values
print(f'我们的split(分界{thr}) | val {len(hva):,}行')

m = lgb.LGBMRegressor(**OUR_PARAMS)
m.fit(htr[hf], np.log(htr['target']), eval_set=[(hva[hf], np.log(hva['target']))],
      eval_metric=feval_log, callbacks=[lgb.early_stopping(150, verbose=False)])
p_a1 = np.exp(m.predict(hva[hf], num_iteration=m.best_iteration_))
print(f'A1 他的特征+我们的log协议:      RMSPE = {rmspe(p_a1, t):.4f}')

m = lgb.LGBMRegressor(**ORVP_PARAMS)
m.fit(htr[hf], htr['target'], sample_weight=1 / htr['target'] ** 2,
      eval_set=[(hva[hf], hva['target'])], eval_metric=feval_raw)
p_a2 = m.predict(hva[hf])
print(f'A2 他的特征+他的1/y²加权协议:    RMSPE = {rmspe(p_a2, t):.4f}')

# ---- B: 我们的特征 × 他的加权(隔离加权技巧) ----
of = [c for c in ours.columns if c not in ('stock_id', 'time_id', 'target')] + ['stock_id']
otr, ova, _ = split(ours)
t2 = ova['target'].values
m = lgb.LGBMRegressor(**ORVP_PARAMS)
m.fit(otr[of], otr['target'], sample_weight=1 / otr['target'] ** 2,
      eval_set=[(ova[of], ova['target'])], eval_metric=feval_raw)
p_b2 = m.predict(ova[of])
print(f'B2 我们的特征+他的1/y²加权协议:  RMSPE = {rmspe(p_b2, t2):.4f}  '
      f'[对照 B1 log协议 0.2493]')

# ---- C: 他的A2 preds 与全家族blend ----
hout = hva[['stock_id', 'time_id', 'target']].copy()
hout['p_his'] = p_a2
nn2 = pd.read_parquet('cache/nn2_val_pred.parquet')[['stock_id', 'time_id', 'p_nn2']]
ts = pd.read_parquet('cache/twostage_val_pred.parquet')[['stock_id', 'time_id', 'p_ts', 'p_b1']]
mg = hout.merge(nn2, on=['stock_id', 'time_id']).merge(ts, on=['stock_id', 'time_id'])
tt = mg['target'].values
L = lambda c: np.log(mg[c].values)
combos = {
    '旧最优 NN2×2+B1+TS': (2 * L('p_nn2') + L('p_b1') + L('p_ts')) / 4,
    'HIS': L('p_his'),
    'HIS+NN2': (L('p_his') + L('p_nn2')) / 2,
    'HIS×2+NN2+B1+TS': (2 * L('p_his') + L('p_nn2') + L('p_b1') + L('p_ts')) / 5,
}
print('\n[blend] (我们的split, 键合并):')
for k, c in combos.items():
    print(f'  {k:<22} RMSPE = {rmspe(np.exp(c), tt):.4f}')
mg.to_parquet('cache/his_val_pred.parquet')
