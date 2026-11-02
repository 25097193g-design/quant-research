"""09 季度调仓相位敏感性检验 — 起始月份会不会严重影响结果?

用 common.py 的唯一 PCR + 唯一回测规则(消除与05的drift)。
对 4F/7F/PCA 季度无过滤, 各跑3个相位, 看年化/夏普/回撤的离散度。
离散小→季度稳健; 离散大→相位依赖(lucky phase), 单一相位不可信。
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, common

OUT = '/Users/lichanghaodemac/quant_strategy/projects/bank_factor_workflow/outputs'
P = common.load_panel()
print(f'面板: {len(P["stks"])}只 × {len(P["me_sorted"])}月 (inf已清洗)', flush=True)

# 三套因子集的月度score (季度口径: 3m-ICIR / PCA用fwd3)
print('算 4F/7F ICIR权重 + PCR...', flush=True)
SC4 = common.score_set(P, common.F4, common.NEG4, common.roll_icir(P, common.F4, P['fwd3']))
SC7 = common.score_set(P, common.F7, common.NEG7, common.roll_icir(P, common.F7, P['fwd3']))
SCP = common.pcr_scores(P, P['fwd3'])
# ★ 2026-08-15: 起跑对齐04/06/07/08的2020-12(面板2017-01的第47个月), 原为2020-01
qstart = pd.Timestamp('2020-12-01'); qi = next(i for i, m in enumerate(P['me_sorted']) if m >= qstart)
quarterly = lambda ph: [m for i, m in enumerate(P['me_sorted']) if i >= qi and (i - qi) % 3 == ph]

rows = []
for name, sc in [('4F', SC4), ('7F', SC7), ('PCA', SCP)]:
    anns = []
    for ph in [0, 1, 2]:
        r = common.backtest(P, sc, quarterly(ph), P['fwd3'], P['mkt3'])  # 无过滤
        m = common.metrics(r, 4)
        if m:
            m.update({'方法': name, 'phase': ph}); rows.append(m); anns.append(m['年化'])
    if len(anns) >= 2:
        rows.append({'方法': name, 'phase': '离散度(max−min)', '年化': round(max(anns) - min(anns), 4)})
res = pd.DataFrame(rows)[['方法', 'phase', '期数', '年化', '夏普', '最大回撤', '期胜率']]
res.to_csv(f'{OUT}/quarterly_phase_check.csv', index=False, encoding='utf-8-sig')
# ★ 基准与策略同窗口(2020-12起, 与策略起跑点一致); 全样本均值含2017-2019暖机期不可比
strat_months = [m for i, m in enumerate(P['me_sorted']) if i >= qi]
mw = P['mkt3'].reindex(strat_months).dropna()
print(f'\n基准(全市场等权3m, 同窗口{str(strat_months[0])[:10]}起): '
      f'年化{mw.mean()*4:.2%}/夏普{mw.mean()/mw.std()*2:.2f} '
      f'(旧口径全样本{P["mkt3"].mean()*4:.2%}, 勿再用)\n', flush=True)
print('=== 季度调仓 3 相位稳健性检验 ===', flush=True)
print(res.to_string(index=False), flush=True)
print('\n离散度<2pp→稳健; >4pp→相位依赖', flush=True)
