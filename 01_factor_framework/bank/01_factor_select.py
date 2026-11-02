"""01 因子筛选 — 从 frame_17 的 137因子×5窗口 结果里, 按"修正后"阈值筛 1m/3m 合格因子

为什么只用 1m/3m: 6m/12m 的前瞻收益高度重叠 → IC 序列自相关 → ICIR 被系统性高估
                 (长 horizon 的 ICIR 不能和短 horizon 直接比, 详见 frame_17 caveat)
                 IC 均值无偏可信, 但保守起见稳健基准只取 1m/3m 两个不重叠窗口。

阈值口径(针对银行 41 只小宇宙修正, 非全市场通用值):
  - 单月截面 IC 的 SE ≈ 1/√(N-1) = 1/√35 ≈ 0.17, 故平均 IC 的显著门槛上抬
  - 合格 = |平均 IC| ≥ 0.05  且  |ICIR| ≥ 0.3  (在某窗口)
  - 候选(做相关性分析用, 放宽以暴露冗余) = |平均 IC| ≥ 0.04 且 |ICIR| ≥ 0.25

输入: data/ic_bank_horizon.csv  (frame_17 产出)
输出: outputs/factors_qualified.csv  (1m/3m 各自合格 + 候选集)
"""
import pandas as pd
import numpy as np

DATA = '/Users/lichanghaodemac/quant_strategy/data'
OUT = '/Users/lichanghaodemac/quant_strategy/projects/bank_factor_workflow/outputs'

ic = pd.read_csv(f'{DATA}/ic_bank_horizon.csv')
print(f'frame_17 结果: {len(ic)} 行, 因子{ic["factor"].nunique()}个 × 窗口{sorted(ic["horizon"].unique())}', flush=True)

# 有效月数检查: 1m/3m 都接近满样本(114/112), 6m/12m 因前瞻缺失掉到 98/92
print('\n各窗口有效月数中位(看样本可比性):')
print(ic.groupby('horizon')['有效月数'].median().to_string(), flush=True)

QUAL_IC, QUAL_ICIR = 0.05, 0.30
CAND_IC, CAND_ICIR = 0.04, 0.25
HORIZONS = ['1m', '3m']

rows = []
for h in HORIZONS:
    sub = ic[ic['horizon'] == h].copy()
    sub['abs_IC'] = sub['银行IC'].abs()
    sub['abs_ICIR'] = sub['银行ICIR'].abs()
    qual = sub[(sub['abs_IC'] >= QUAL_IC) & (sub['abs_ICIR'] >= QUAL_ICIR)]
    cand = sub[(sub['abs_IC'] >= CAND_IC) & (sub['abs_ICIR'] >= CAND_ICIR)]
    print(f'\n=== {h} 窗口 ===', flush=True)
    print(f'  合格(|IC|≥{QUAL_IC} & |ICIR|≥{QUAL_ICIR}): {len(qual)} 个', flush=True)
    print(f'  候选(|IC|≥{CAND_IC} & |ICIR|≥{CAND_ICIR}): {len(cand)} 个', flush=True)
    print(f'  --- 合格因子(按|ICIR|降序) ---', flush=True)
    for _, r in qual.reindex(qual['abs_ICIR'].sort_values(ascending=False).index).iterrows():
        rows.append({'horizon': h, 'factor': r['factor'], '银行IC': round(r['银行IC'], 3),
                     '银行ICIR': round(r['银行ICIR'], 2), '有效月数': int(r['有效月数']),
                     '截面中位': int(r['截面中位']), '级别': '合格'})
        print(f'    {r["factor"]:<30} IC{r["银行IC"]:+.3f}  ICIR{r["银行ICIR"]:+.2f}', flush=True)
    # 候选里不属于合格的多出来的(用来暴露冗余)
    extra = cand[~cand['factor'].isin(qual['factor'])]
    for _, r in extra.iterrows():
        rows.append({'horizon': h, 'factor': r['factor'], '银行IC': round(r['银行IC'], 3),
                     '银行ICIR': round(r['银行ICIR'], 2), '有效月数': int(r['有效月数']),
                     '截面中位': int(r['截面中位']), '级别': '候选'})

out = pd.DataFrame(rows)
out.to_csv(f'{OUT}/factors_qualified.csv', index=False, encoding='utf-8-sig')
print(f'\n存 outputs/factors_qualified.csv: {len(out)} 行', flush=True)

# 给后续脚本用的因子清单: (1m合格 ∪ 3m合格) + 候选(放宽阈值以暴露冗余) = 18 个分析对象
# 注意: 合格并集本身只有14个, 另4个(alpha013/026/088, DAVOL20)是候选级 — 别再说"6∪14=18"
union = sorted(set(out['factor']))
n_qual = out[out['级别'] == '合格']['factor'].nunique()
print(f'\n>>> 相关性分析对象 = 合格并集{n_qual} + 候选{len(union) - n_qual} = {len(union)} 个因子', flush=True)
print('  ' + ', '.join(union), flush=True)
