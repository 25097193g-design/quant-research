# 复现第7名simple_model: 特征管线(feature_processor.py原样) → cache/orvp_features.parquet
# 运行: cd projects/optiver_rv_2021 && conda run -n rqsdk --no-capture-output python 17_reproduce_7th.py
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(HERE, 'external/orvp/simple_model'))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, HERE)

t0 = time.time()
import orvp_fp_patched as fp  # noqa: E402 (他的管线+pandas2最小补丁: apply→transform)

df = fp.generate_df(True)
OUT = '/Users/lichanghaodemac/quant_strategy/projects/optiver_rv_2021/cache/orvp_features.parquet'
df.to_parquet(OUT)
feats = [c for c in df.columns if c not in ('time_id', 'target', 'row_id')]
print(f'\n[复现] 第7名特征: {df.shape[0]:,}行 × {len(feats)}特征 | 用时 {time.time()-t0:.0f}s')
print('  特征清单:', feats)
print('  缺失统计(前10):')
print(df[feats].isna().sum().sort_values(ascending=False).head(10).to_string())
