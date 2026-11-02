"""01 医药因子筛选 — 从 frame_21 的 ic_pharma_horizon.csv 按 IC/ICIR 阈值筛 1m/3m 合格因子

医药截面 ~207 只(银行 41 的 5 倍), N 大 → IC 更显著, 但沿用同阈值(便于跨板块对比)。
输入: data/ic_pharma_horizon.csv (frame_21 产出, 列 医药IC/医药ICIR)
输出: outputs/factors_qualified.csv
"""
import pandas as pd

DATA = '/Users/lichanghaodemac/quant_strategy/data'
OUT = '/Users/lichanghaodemac/quant_strategy/projects/pharma_factor_workflow/outputs'
ic = pd.read_csv(f'{DATA}/ic_pharma_horizon.csv')
print(f'frame_21 结果: {len(ic)}行, 因子{ic["factor"].nunique()}个 × horizon {sorted(ic["horizon"].unique())}', flush=True)

QUAL_IC, QUAL_ICIR = 0.03, 0.30
rows = []
for h in ['1m', '3m']:
    sub = ic[ic['horizon'] == h].copy()
    sub['aIC'] = sub['医药IC'].abs(); sub['aICIR'] = sub['医药ICIR'].abs()
    qual = sub[(sub.aIC >= QUAL_IC) & (sub.aICIR >= QUAL_ICIR)]
    print(f'\n=== {h} 合格(|IC|≥{QUAL_IC} & |ICIR|≥{QUAL_ICIR}): {len(qual)} 个 (截面中位{int(sub["截面中位"].median())}只) ===', flush=True)
    for _, r in qual.reindex(qual.aICIR.sort_values(ascending=False).index).iterrows():
        rows.append({'horizon': h, 'factor': r['factor'], 'IC': round(r['医药IC'], 3), 'ICIR': round(r['医药ICIR'], 2)})
        print(f'  {r["factor"]:<36} IC{r["医药IC"]:+.3f}  ICIR{r["医药ICIR"]:+.2f}', flush=True)
out = pd.DataFrame(rows)
out.to_csv(f'{OUT}/factors_qualified.csv', index=False, encoding='utf-8-sig')
union = sorted(set(out['factor']))
print(f'\n>>> 相关性分析对象 = 1m ∪ 3m 合格 = {len(union)} 个', flush=True)
print('  ' + ', '.join(union), flush=True)
