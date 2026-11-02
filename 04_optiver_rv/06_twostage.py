# P3+: 两阶段市场先行分解(3rd名 "life is volatile" 思路)
# 运行: cd projects/optiver_rv_2021 && conda run -n rqsdk --no-capture-output python 06_twostage.py
# 结构: log target = log M(t) + log R(stock,t)
#   S1 预测 log M(市场几何平均波动) ← 市场级本桶可观测量(0.923相关的燃料)
#   S2 预测 log R(个股相对倍数)     ← v2全部特征 + stock_id类别
#   组装 pred = exp(Mhat + Rhat); 防泄漏: logM是label的函数, 从不作特征; Mhat只在推理端出现
import lightgbm as lgb
import numpy as np
import pandas as pd

from common import rmspe

df = pd.read_parquet('cache/features_v2.parquet')
base_feats = [c for c in df.columns if c not in ('stock_id', 'time_id', 'target')]
feats_sid = base_feats + ['stock_id']

df['log_target'] = np.log(df['target'])
logM = df.groupby('time_id')['log_target'].mean().rename('logM')
df = df.merge(logM, on='time_id')

u = np.sort(df['time_id'].unique())
thr = u[int(len(u) * 0.8)]
tr, va = df[df.time_id < thr], df[df.time_id >= thr]
tgt = va['target'].values

PARAMS = dict(objective='regression', n_estimators=3000, learning_rate=0.05,
              num_leaves=64, min_child_samples=100, feature_fraction=0.85,
              bagging_fraction=0.85, bagging_freq=1, lambda_l2=1.0,
              random_state=42, verbosity=-1, n_jobs=-1)

def X(d, feats):
    Xd = d[feats].copy()
    if 'stock_id' in feats:
        Xd['stock_id'] = Xd['stock_id'].astype('category')
    return Xd

def feval_log(y_true, y_pred):
    return 'rmspe', rmspe(np.exp(y_pred), np.exp(y_true)), False

def fit_eval(feats, ytr, yva, feval, tag):
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(X(tr, feats), ytr, eval_set=[(X(va, feats), yva)], eval_metric=feval,
          callbacks=[lgb.early_stopping(150, verbose=False)])
    p = np.exp(m.predict(X(va, feats), num_iteration=m.best_iteration_))
    print(f'  {tag:<24} RMSPE = {rmspe(p, tgt):.4f}  (best_iter={m.best_iteration_})')
    return m, p

print(f'[对比] 分界time_id={thr} | val {len(va):,}行\n')

# ---- B0: v2复现(基线) / B1: v2+stock_id ----
ytr, yva = tr['log_target'], va['log_target']
_, p_b0 = fit_eval(base_feats, ytr, yva, feval_log, 'B0: v2单阶段(27特征)')
_, p_b1 = fit_eval(feats_sid, ytr, yva, feval_log, 'B1: v2+stock_id')

# ---- S1: 市场级模型(每time_id一行) ----
tm = df.groupby('time_id').agg(
    mkt_rv_med=('rv_now', 'median'), mkt_rv_std=('rv_now', 'std'),
    mkt_rv_p25=('rv_now', lambda s: s.quantile(.25)),
    mkt_rv_p75=('rv_now', lambda s: s.quantile(.75)),
    mkt_spread_med=('spread_mean_bp', 'median'), mkt_tvol_sum=('trade_vol', 'sum'),
    mkt_n=('rv_now', 'size')).join(logM)
tm['storm_frac'] = df.assign(f=df.rv_now > 2 * df.mkt_rv_med).groupby('time_id').f.mean()
S1_feats = [c for c in tm.columns if c != 'logM']
trt, vat = tm[tm.index < thr], tm[tm.index >= thr]
s1 = lgb.LGBMRegressor(objective='regression', n_estimators=2000, learning_rate=0.03,
                       num_leaves=15, min_child_samples=20, random_state=42,
                       verbosity=-1, n_jobs=-1)
s1.fit(trt[S1_feats], trt.logM, eval_set=[(vat[S1_feats], vat.logM)],
       callbacks=[lgb.early_stopping(100, verbose=False)])
logMhat = pd.Series(s1.predict(vat[S1_feats], num_iteration=s1.best_iteration_),
                    index=vat.index)
m_rmspe = rmspe(np.exp(logMhat.values), np.exp(vat.logM.values))
print(f'\n  S1市场模型: corr(logMhat, logM) = '
      f'{np.corrcoef(logMhat, vat.logM)[0,1]:.4f} | 市场级RMSPE = {m_rmspe:.4f}')

va = va.copy()
va['logMhat'] = va['time_id'].map(logMhat)

# ---- S2: 个股相对倍数, 组装 ----
mh_arr = va['logMhat'].values
lm_arr = va['logM'].values

def feval_resid(y_true, y_pred):
    p = np.exp(y_pred + mh_arr)
    t = np.exp(y_true + lm_arr)
    return 'rmspe', rmspe(p, t), False

m2 = lgb.LGBMRegressor(**PARAMS)
m2.fit(X(tr, feats_sid), tr.log_target - tr.logM,
       eval_set=[(X(va, feats_sid), va.log_target - va.logM)], eval_metric=feval_resid,
       callbacks=[lgb.early_stopping(150, verbose=False)])
p_ts = np.exp(m2.predict(X(va, feats_sid), num_iteration=m2.best_iteration_) + mh_arr)
print(f'  {"两阶段(Mhat×Rhat)":<24} RMSPE = {rmspe(p_ts, tgt):.4f}  '
      f'(best_iter={m2.best_iteration_})')

p_bl = np.exp(0.5 * np.log(p_b1) + 0.5 * np.log(p_ts))
print(f'  {"几何blend(B1+两阶段)":<24} RMSPE = {rmspe(p_bl, tgt):.4f}')

# ---- 分股诊断 ----
out = va[['stock_id', 'time_id', 'target']].copy()
out['p_b1'], out['p_ts'] = p_b1, p_ts
out.to_parquet('cache/twostage_val_pred.parquet')
ps = out.groupby('stock_id', observed=True).apply(
    lambda g: pd.Series({'b1': rmspe(g.p_b1.values, g.target.values),
                         'ts': rmspe(g.p_ts.values, g.target.values)}))
print('\n[分股] stock_31: B1=' + f'{ps.loc[31].b1:.3f}' if 31 in ps.index else '')
w = ps.sort_values('ts', ascending=False).head(3)
print('  两阶段最差3只: ' + ', '.join(f'stock_{int(k)}={v.ts:.3f}(B1 {v.b1:.3f})'
                                    for k, v in w.iterrows()))
med_imp = (ps.b1 - ps.ts).median()
print(f'  分股RMSPE改善中位数(B1→两阶段): {med_imp:+.4f}')
