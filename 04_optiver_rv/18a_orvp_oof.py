# 复现第7名训练协议: 随机5折KFold + weighted MSE(1/y²) + 他的LGBM参数 → OOF RMSPE
# 运行: cd projects/optiver_rv_2021 && conda run -n rqsdk --no-capture-output python 18a_orvp_oof.py
# 对照: 他公开LB 0.22955 | 我们的诚实4折 0.2519(不同折法, 仅量级参考)
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

def feval_rmspe(y_pred, ds):
    return 'RMSPE', rmspe(ds.get_label(), y_pred), False

train = pd.read_parquet('cache/orvp_features.parquet')
features = [c for c in train.columns if c not in {'time_id', 'target', 'row_id'}]
y = train['target']
oof = np.zeros(len(train))
params0 = {
    'objective': 'rmse', 'boosting_type': 'gbdt', 'max_depth': -1, 'max_bin': 100,
    'min_data_in_leaf': 500, 'learning_rate': 0.05, 'subsample': 0.72,
    'subsample_freq': 4, 'feature_fraction': 0.5, 'lambda_l1': 0.5, 'lambda_l2': 1.0,
    'seed': 2021, 'feature_fraction_seed': 2021, 'bagging_seed': 2021,
    'drop_seed': 2021, 'data_random_seed': 2021, 'n_jobs': -1, 'verbose': -1}

kfold = KFold(n_splits=5, random_state=2021, shuffle=True)
for fold, (tri, vai) in enumerate(kfold.split(train)):
    Xtr, Xva = train.iloc[tri], train.iloc[vai]
    dtr = lgb.Dataset(Xtr[features], y.iloc[tri], weight=1 / np.square(y.iloc[tri]),
                      categorical_feature=['stock_id'])
    dva = lgb.Dataset(Xva[features], y.iloc[vai], weight=1 / np.square(y.iloc[vai]),
                      reference=dtr)
    model = lgb.train(params=params0, num_boost_round=1000, train_set=dtr,
                      valid_sets=[dtr, dva], feval=feval_rmspe,
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(0)])
    oof[vai] = model.predict(Xva[features])
    print(f'fold{fold+1}: RMSPE = {rmspe(y.iloc[vai], oof[vai]):.4f} '
          f'(best_iter={model.best_iteration})')

print(f'\n★ 第7名协议 OOF RMSPE = {rmspe(y, oof):.4f}   [他的公开LB 0.22955]')
