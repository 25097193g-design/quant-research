"""04 去冗余对照 — 医药 24因子(全合格) vs 7因子/4因子(去冗余), 超跌埋伏轮动

银行版结论: 7F有效维度~4, 去冗余4F夏普差不多换手更低。
医药版假设: 24因子→19簇(大部分WQ alpha互相独立), 去冗余丢的信息更多,
   4F/7F vs 24F 的差距可能比银行小 → 验证"去冗余是否真金白银"在医药是否成立。

口径(与银行04一致, 只换板块/因子集):
  - 医药228只, 月调仓, z-score(1/99%winsorize), ICIR加权, qcut5取Q5
  - 超跌过滤: 前3月收益和 < thr 才埋伏医药Q5; 否则空仓投全市场等权
  - 成本单边万7.5(双边0.15%)
两套加权: 静态ICIR(全样本, 有未来函数→bank原版口径) / 滚动ICIR(36月, 诚实walk-forward)
输出: outputs/dedup_vs_full.csv
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, common

OUT = '/Users/lichanghaodemac/quant_strategy/projects/pharma_factor_workflow/outputs'
P = common.load_panel()
fwd1, mkt1 = P['fwd1'], P['mkt1']
me_sorted = P['me_sorted']

NEG24 = {'VOLT20', 'VOLT60', 'pb_ratio_lf', 'pe_ratio_lyr', 'ps_ratio_ttm'}
SETS = {
    '24因子(全合格)': (P['facs'], NEG24),
    '7因子(去冗余)': (common.F7, common.NEG7),
    '4因子(去冗余)': (common.F4, common.NEG4),
}

def turnover(q5_sets):
    months = sorted(q5_sets); ds = []
    for a, b in zip(months, months[1:]):
        u, i = q5_sets[a], q5_sets[b]
        if u or i: ds.append(1 - len(u & i) / len(u | i))
    return np.mean(ds) if ds else np.nan

THRS = [('无过滤', None), ('<-5%', -0.05), ('<-3%', -0.03), ('<-1%', -0.01)]
rows = []
for setname, (facs, neg) in SETS.items():
    W_roll = common.roll_icir(P, facs, fwd1)
    # 静态ICIR(全样本) → 用于复现bank原版口径
    ic_all = {}
    for f in facs:
        s = pd.Series({m: P['IC1'][f].get(m, np.nan) for m in me_sorted if m in P['IC1'][f].index}).dropna()
        ic_all[f] = s.mean() / s.std() if len(s) > 6 and s.std() > 0 else 0
    W_static = np.array([abs(ic_all[f]) for f in facs]); W_static = W_static / W_static.sum() if W_static.sum() > 0 else np.ones(len(facs)) / len(facs)
    for scheme, W_src in [('静态', {m: W_static for m in me_sorted}), ('滚动', W_roll)]:
        sc = common.score_set(P, facs, neg, W_src)
        for lb, thr in THRS:
            r, q5s = common.backtest(P, sc, me_sorted[36:], fwd1, mkt1, thr, ret_sets=True)
            m = common.metrics(r, 12)
            if m is None: continue
            m.update({'因子集': setname, '加权': scheme, '过滤': lb,
                      'Q5月换手': round(turnover(q5s), 2)})
            rows.append(m)
res = pd.DataFrame(rows)[['因子集', '加权', '过滤', '年化', '夏普', '最大回撤', '期胜率', '期数', 'Q5月换手']]
res.to_csv(f'{OUT}/dedup_vs_full.csv', index=False, encoding='utf-8-sig')

print(f'全市场等权基准: 年化{mkt1.mean()*12:.2%} 夏普{mkt1.mean()/mkt1.std()*np.sqrt(12):.2f}\n', flush=True)
for scheme in ['静态', '滚动']:
    print(f'===== {scheme}ICIR 加权 ({"有未来函数, 复现bank04口径" if scheme=="静态" else "36月rolling, 无未来, 诚实版"}) =====', flush=True)
    print(f'{"因子集":<16}{"过滤":<9}{"年化":>9}{"夏普":>9}{"回撤":>9}{"换手":>9}', flush=True)
    for setname in SETS:
        for lb, _ in THRS:
            r = res[(res.因子集 == setname) & (res.加权 == scheme) & (res.过滤 == lb)].iloc[0]
            print(f'{setname:<16}{lb:<9}{r["年化"]:>9.2%}{r["夏普"]:>9.2f}{r["最大回撤"]:>9.2%}{r["Q5月换手"]:>9.0%}', flush=True)
        print(flush=True)
print('存 outputs/dedup_vs_full.csv', flush=True)
