# P4a: 多折时间CV稳定性 + 参数敏感性
# 运行: cd projects/optiver_rv_2021 && conda run -n rqsdk --no-capture-output python 10_multifold.py
# 4个扩张窗折: 验证块 = time_id的50-60%/60-70%/70-80%/80-100%, 训练=块之前全部
# 目的: 把"0.245x"从单折运气变成 mean±std 的可信数字(bank项目教训: 排名是口径函数)
import lightgbm as lgb
import numpy as np
import pandas as pd

from common import rmspe

df = pd.read_parquet('cache/features_v2.parquet')
feats = [c for c in df.columns if c not in ('stock_id', 'time_id', 'target')] + ['stock_id']

u = np.sort(df['time_id'].unique())
blocks = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]

def feval_log(y_true, y_pred):
    return 'rmspe', rmspe(np.exp(y_pred), np.exp(y_true)), False

CFGS = {
    'B1(叶子64/lr.05)': dict(num_leaves=64, learning_rate=0.05),
    'alt(叶子127/lr.03)': dict(num_leaves=127, learning_rate=0.03),
}
results = {k: [] for k in CFGS} | {'naive': []}

for lo, hi in blocks:
    thr_lo = u[int(len(u) * lo)]
    thr_hi = u[min(int(len(u) * hi), len(u) - 1)] + 1  # +1: 含最后一个time_id
    tr = df[df.time_id < thr_lo]
    va = df[(df.time_id >= thr_lo) & (df.time_id < thr_hi)]
    tgt = va['target'].values
    ytr, yva = np.log(tr['target']), np.log(va['target'])
    results['naive'].append(rmspe(va['rv_now'].values, tgt))
    for tag, cfg in CFGS.items():
        m = lgb.LGBMRegressor(
            objective='regression', n_estimators=3000, min_child_samples=100,
            feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1,
            lambda_l2=1.0, random_state=42, verbosity=-1, n_jobs=-1, **cfg)
        m.fit(tr[feats], ytr, eval_set=[(va[feats], yva)], eval_metric=feval_log,
              callbacks=[lgb.early_stopping(150, verbose=False)])
        p = np.exp(m.predict(va[feats], num_iteration=m.best_iteration_))
        results[tag].append(rmspe(p, tgt))
    print(f'折[{lo:.0%},{hi:.0%}): naive={results["naive"][-1]:.4f} | ' +
          ' | '.join(f'{k}={results[k][-1]:.4f}' for k in CFGS))

print('\n[P4a] 4折汇总(mean ± std):')
for k, v in results.items():
    print(f'  {k:<20} {np.mean(v):.4f} ± {np.std(v):.4f}   {[round(x,4) for x in v]}')
