# 真实结果压力测试: leave-regime-out —— 扣出全市场风暴时段(top10% mkt_rv_med)当验证
# 运行: cd projects/optiver_rv_2021 && conda run -n rqsdk --no-capture-output python 20_true_eval.py
# 逻辑: 随机折的乐观主义主要来自regime内插值; 把训练完全没见过的极端行情整体当验证,
#       是本地能构造的最接近"LB时间外推"的严格口径
import lightgbm as lgb
import numpy as np
import pandas as pd

from common import rmspe

ORVP_PARAMS = dict(objective='regression', n_estimators=1000, learning_rate=0.05,
                   min_child_samples=500, feature_fraction=0.5, subsample=0.72,
                   subsample_freq=4, lambda_l1=0.5, lambda_l2=1.0, max_bin=100,
                   random_state=2021, verbosity=-1, n_jobs=-1)

his = pd.read_parquet('cache/orvp_features.parquet')
ours = pd.read_parquet('cache/features_v2.parquet')
mkt = ours.groupby('time_id')['mkt_rv_med'].first()
storm_thr = mkt.quantile(0.90)
calm_thr = mkt.quantile(0.10)
storm_ids = set(mkt[mkt >= storm_thr].index)
calm_ids = set(mkt[mkt <= calm_thr].index)
print(f'风暴验证: {len(storm_ids)}个time_id(市场RV中位≥{storm_thr:.5f}) | '
      f'平静参照: {len(calm_ids)}个')

def regime_eval(df, feats, holdout_ids, tag):
    va_mask = df['time_id'].isin(holdout_ids)
    tr, va = df[~va_mask], df[va_mask]
    m = lgb.LGBMRegressor(**ORVP_PARAMS)
    m.fit(tr[feats], tr['target'], sample_weight=1 / tr['target'] ** 2)
    p = m.predict(va[feats])
    n = rmspe(va['rv_now'].values, va['target'].values) if 'rv_now' in va else np.nan
    return rmspe(p, va['target'].values), p, va, n

hf = [c for c in his.columns if c not in {'time_id', 'target', 'row_id'}]
of = [c for c in ours.columns if c not in ('stock_id', 'time_id', 'target')] + ['stock_id']

print('\n[风暴regime hold-out] (训练完全没见过top10%行情):')
h_s, ph, vah, _ = regime_eval(his, hf, storm_ids, 'HIS')
o_s, po, vao, _ = regime_eval(ours, of, storm_ids, 'B2')
print(f'  HIS(7th特征):   RMSPE = {h_s:.4f}')
print(f'  B2(我们特征):   RMSPE = {o_s:.4f}')
mg = vao[['stock_id', 'time_id', 'target']].copy()
mg['p_b2'] = po
mg = mg.merge(vah[['stock_id', 'time_id']].assign(p_his=ph), on=['stock_id', 'time_id'])
bl = np.exp(0.5 * np.log(mg.p_his.values) + 0.5 * np.log(mg.p_b2.values))
naive_s = rmspe(mg.target.values * 0 + vao.set_index(['stock_id', 'time_id'])
                .loc[list(zip(mg.stock_id, mg.time_id)), 'rv_now'].values, mg.target.values)
print(f'  blend(HIS+B2):  RMSPE = {rmspe(bl, mg.target.values):.4f} | naive同折 {naive_s:.4f}')

print('\n[平静regime hold-out] (对照: 扣出最平静10%):')
h_c, _, _, _ = regime_eval(his, hf, calm_ids, 'HIS')
o_c, _, _, _ = regime_eval(ours, of, calm_ids, 'B2')
print(f'  HIS: {h_c:.4f} | B2: {o_c:.4f}  (掉分幅度 vs 风暴口 = regime外推代价的方向性参照)')
