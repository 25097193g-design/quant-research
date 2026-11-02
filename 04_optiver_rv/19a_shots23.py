# 弹药2+3: HIS放量到3000轮 + B2预测存档 + 等强blend(HIS+B2)
# 运行: cd projects/optiver_rv_2021 && conda run -n rqsdk --no-capture-output python 19a_shots23.py
import lightgbm as lgb
import numpy as np
import pandas as pd

from common import rmspe

def feval_raw(y_true, y_pred):
    return 'rmspe', rmspe(y_true, y_pred), False

ORVP_PARAMS = dict(objective='regression', learning_rate=0.05, min_child_samples=500,
                   feature_fraction=0.5, subsample=0.72, subsample_freq=4,
                   lambda_l1=0.5, lambda_l2=1.0, max_bin=100,
                   random_state=2021, verbosity=-1, n_jobs=-1)

def split(df):
    u = np.sort(df['time_id'].unique())
    thr = u[int(len(u) * 0.8)]
    return df[df.time_id < thr], df[df.time_id >= thr].sort_values(
        ['stock_id', 'time_id']).reset_index(drop=True)

his = pd.read_parquet('cache/orvp_features.parquet')
ours = pd.read_parquet('cache/features_v2.parquet')
hf = [c for c in his.columns if c not in {'time_id', 'target', 'row_id'}]
of = [c for c in ours.columns if c not in ('stock_id', 'time_id', 'target')] + ['stock_id']

# ---- 弹药3: HIS 3000轮(原1000轮到顶未收敛) ----
htr, hva = split(his)
t = hva['target'].values
m3 = lgb.LGBMRegressor(n_estimators=3000, **ORVP_PARAMS)
m3.fit(htr[hf], htr['target'], sample_weight=1 / htr['target'] ** 2,
       eval_set=[(hva[hf], hva['target'])], eval_metric=feval_raw,
       callbacks=[lgb.early_stopping(100, verbose=False)])
p_his3 = m3.predict(hva[hf], num_iteration=m3.best_iteration_)
print(f'弹药3 HIS-3000轮:    RMSPE = {rmspe(p_his3, t):.4f}  '
      f'(best_iter={m3.best_iteration_})  [1000轮=0.2242]')

# ---- 弹药2: B2复跑并存预测 ----
otr, ova = split(ours)
t2 = ova['target'].values
m = lgb.LGBMRegressor(n_estimators=1000, **ORVP_PARAMS)
m.fit(otr[of], otr['target'], sample_weight=1 / otr['target'] ** 2)
p_b2 = m.predict(ova[of])
print(f'弹药2 B2(我们特征+1/y²): RMSPE = {rmspe(p_b2, t2):.4f}  [对照 0.2265]')

# ---- 等强blend ----
hout = hva[['stock_id', 'time_id', 'target']].copy()
hout['p_his3'] = p_his3
oout = ova[['stock_id', 'time_id']].copy()
oout['p_b2'] = p_b2
mg = hout.merge(oout, on=['stock_id', 'time_id'])
tt = mg['target'].values
L = lambda c: np.log(mg[c].values)
for name, c in [('HIS3', L('p_his3')), ('B2', L('p_b2')),
                ('HIS3+B2', (L('p_his3') + L('p_b2')) / 2),
                ('HIS3×2+B2', (2 * L('p_his3') + L('p_b2')) / 3)]:
    print(f'  blend {name:<12} RMSPE = {rmspe(np.exp(c), tt):.4f}')
mg.to_parquet('cache/shots23_val_pred.parquet')
