"""14 重做Stage A — 核心池回测矩阵: 7配置 × 月/季 × 季频3相位 (2026-08-16批准设计)

池子来自 10_subperiod_ic 的存活判定, 方法修正来自 12_embargo_check 审计:
  - embargo: 季频(fwd3)的IC权重/符号窗 = [i-38, i-2](36个月全实现, 无标签重叠); 月频fwd1无泄漏用[i-36, i]
  - 方向: 静态(全样本定号, 含前视, legacy口径) / 滚动(过去36月IC均值符号, walk-forward) / 混合(核心静态+卫星滚动)
  - 相位: 季频3相位全报, 中位数为headline(不报单相位最优)

7配置:
  1 core7/静态   2 core7/滚动   3 core11/静态   4 sat5/滚动
  5 core7+sat5(12F)/混合   6 legacy24/静态(NEG24)   7 legacy24/滚动
基准(同月计算): 医药等权(主口径=选股alpha) + 全A等权(可实现性)。
输出: outputs/redo_stageA_matrix.csv
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import spearmanr
import common

OUT = '/Users/lichanghaodemac/quant_strategy/projects/pharma_factor_workflow/outputs'
COST = 0.0015

CORE7 = ['WorldQuant_alpha040', 'WorldQuant_alpha016', 'WorldQuant_alpha024', 'WorldQuant_alpha044',
         'WorldQuant_alpha011', 'dividend_yield_ttm', 'WorldQuant_alpha026']
OBS4 = ['WorldQuant_alpha013', 'WorldQuant_alpha015', 'WorldQuant_alpha027', 'WorldQuant_alpha050']
SAT5 = ['book_to_market_ratio_lf', 'pb_ratio_lf', 'ps_ratio_ttm', 'VOLT20', 'VOLT60']
LEGACY24 = common._POOL
NEG24 = {'VOLT20', 'VOLT60', 'pb_ratio_lf', 'pe_ratio_lyr', 'ps_ratio_ttm'}
FACS_ALL = sorted(set(CORE7 + OBS4 + SAT5 + LEGACY24))

CONFIGS = [  # (名, 因子列表, 静态neg集合, 滚动成员集合)
    ('core7/静态',      CORE7,               set(),        set()),
    ('core7/滚动',      CORE7,               set(),        set(CORE7)),
    ('core11/静态',     CORE7 + OBS4,        set(),        set()),
    ('sat5/滚动',       SAT5,                set(),        set(SAT5)),
    ('core7+sat5/混合', CORE7 + SAT5,        set(),        set(SAT5)),
    ('legacy24/静态',   LEGACY24,            NEG24,        set()),
    ('legacy24/滚动',   LEGACY24,            set(),        set(LEGACY24)),
]

# ===== 面板(复用common, 因子并集) =====
P = common.load_panel(FACS_ALL)
fwd1, fwd3 = P['fwd1'], P['fwd3']
me_sorted, Z, stks = P['me_sorted'], P['Z'], P['stks']
IDX = {m: i for i, m in enumerate(me_sorted)}
print(f'面板: {len(stks)}只 × {len(me_sorted)}月 × {len(P["facs"])}因子(并集)', flush=True)

def ic_series(f, fwd, minn=10):
    return pd.Series({m: (lambda d: spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0] if len(d) >= minn and d.iloc[:, 0].nunique() > 1 else np.nan)
                      (pd.concat([Z[f].loc[m], fwd.loc[m]], axis=1).dropna()) for m in me_sorted if m in fwd.index})

print('算 IC1(fwd1) / IC3(fwd3) 序列...', flush=True)
IC1 = {f: ic_series(f, fwd1) for f in P['facs']}
IC3 = {f: ic_series(f, fwd3) for f in P['facs']}

# ===== 同月基准: 医药等权 / 全A等权 =====
univ = pd.read_parquet(f'{common.DATA}/universe_monthend_pit.parquet'); univ['monthend'] = pd.DatetimeIndex(univ['monthend'])
ids = univ[univ['industry_l1'] == '医药生物'].groupby('monthend')['order_book_id'].apply(set).to_dict()
ph1 = pd.Series({m: fwd1.loc[m, [o for o in ss if o in fwd1.columns]].dropna().mean() for m, ss in ids.items() if m in fwd1.index and len([o for o in ss if o in fwd1.columns]) >= 10}).sort_index()
ph3 = pd.Series({m: fwd3.loc[m, [o for o in ss if o in fwd3.columns]].dropna().mean() for m, ss in ids.items() if m in fwd3.index and len([o for o in ss if o in fwd3.columns]) >= 10}).dropna()
mkt1, mkt3 = P['mkt1'], P['mkt3']

# ===== 打分: 权重=|滚动ICIR|, 符号=静态neg或滚动窗IC均值符号 =====
def make_scores(facs, neg_static, roll_set, freq):
    fwd = fwd1 if freq == '月' else fwd3
    IC = IC1 if freq == '月' else IC3
    out = {}
    for m in me_sorted:
        if m not in fwd.index: continue
        i = IDX[m]
        win = me_sorted[max(0, i - 36):i] if freq == '月' else me_sorted[max(0, i - 38):i - 2]   # 季频embargo
        ws, ss_ = [], []
        for f in facs:
            v = IC[f].reindex(win).dropna()
            if len(v) > 6 and v.std() > 0:
                icir = abs(v.mean() / v.std())
                s = (1 if v.mean() > 0 else -1) if f in roll_set else (-1 if f in neg_static else 1)
            else:
                icir, s = 0.0, 1
            ws.append(icir); ss_.append(s)
        ws = np.array(ws); ws = ws / ws.sum() if ws.sum() > 0 else np.ones(len(facs)) / len(facs)
        z = pd.DataFrame(index=stks)
        for f, s in zip(facs, ss_): z[f] = s * Z[f].loc[m].reindex(stks)
        out[m] = (z.fillna(0) * ws).sum(axis=1)
    return out

def bt(scores, rebal, fwd):
    rets, months, q5s = [], [], {}
    for m in rebal:
        if m not in fwd.index or m not in scores: continue
        s = scores[m].dropna()
        if s.nunique() < 5: continue
        try: q = pd.qcut(s.rank(method='first'), 5, labels=False)
        except Exception: continue
        q5 = s[q == 4].index
        if len(q5) < 3: continue
        vals = [fwd.loc[m, o] for o in q5 if o in fwd.columns and pd.notna(fwd.loc[m, o])]
        if len(vals) >= 2:
            rets.append(np.mean(vals) - COST); months.append(m); q5s[m] = set(q5)
    return pd.Series(rets, index=months), q5s

def turnover(q5s):
    ms = sorted(q5s); ds = [1 - len(q5s[a] & q5s[b]) / len(q5s[a] | q5s[b]) for a, b in zip(ms, ms[1:]) if q5s[a] or q5s[b]]
    return np.mean(ds) if ds else np.nan

def metrics(r, months, py, bench_ph, bench_mkt):
    if len(r) < 6: return None
    ann = r.mean() * py
    sh = r.mean() / r.std() * np.sqrt(py) if r.std() > 0 else 0
    cum = (1 + r).cumprod(); dd = (cum / cum.cummax() - 1).min()
    bph = bench_ph.reindex(months).dropna().mean() * py
    bmk = bench_mkt.reindex(months).dropna().mean() * py
    return {'年化': round(ann, 4), '夏普': round(sh, 2), '最大回撤': round(dd, 4),
            '超额vs医药EW': round(ann - bph, 4), '超额vs全AEW': round(ann - bmk, 4), '期数': len(r)}

monthly = [m for i, m in enumerate(me_sorted) if i >= 36]
rows = []
for name, facs, neg, roll in CONFIGS:
    for freq in ['月', '季']:
        fwd = fwd1 if freq == '月' else fwd3
        py = 12 if freq == '月' else 4
        bph = ph1 if freq == '月' else ph3
        bmk = mkt1 if freq == '月' else mkt3
        scores = make_scores(facs, neg, roll, freq)
        if freq == '月':
            r, q5s = bt(scores, monthly, fwd)
            m_ = metrics(r, r.index, py, bph, bmk)
            if m_: rows.append({'配置': name, '频率': '月', '相位': '-', **m_, '换手': round(turnover(q5s), 2)})
        else:
            per = []
            for ph in [0, 1, 2]:
                rebal = [m for i, m in enumerate(me_sorted) if i >= 36 and (i - 36) % 3 == ph]
                r, q5s = bt(scores, rebal, fwd)
                m_ = metrics(r, r.index, py, bph, bmk)
                if m_:
                    rows.append({'配置': name, '频率': '季', '相位': ph, **m_, '换手': round(turnover(q5s), 2)})
                    per.append(m_)
            if per:
                rows.append({'配置': name, '频率': '季', '相位': '中位',
                             '年化': round(float(np.median([p['年化'] for p in per])), 4),
                             '夏普': round(float(np.median([p['夏普'] for p in per])), 2),
                             '最大回撤': round(float(np.median([p['最大回撤'] for p in per])), 4),
                             '超额vs医药EW': round(float(np.median([p['超额vs医药EW'] for p in per])), 4),
                             '超额vs全AEW': round(float(np.median([p['超额vs全AEW'] for p in per])), 4),
                             '期数': int(np.median([p['期数'] for p in per])), '换手': '-'})
res = pd.DataFrame(rows)
res.to_csv(f'{OUT}/redo_stageA_matrix.csv', index=False, encoding='utf-8-sig')

print('\n===== Stage A 矩阵 (月频 / 季频3相位+中位) =====', flush=True)
show = ['配置', '频率', '相位', '年化', '夏普', '最大回撤', '超额vs医药EW', '超额vs全AEW', '期数', '换手']
print(res[show].to_string(index=False), flush=True)
print('\n===== 季频中位数排行(主口径=超额vs医药EW) =====', flush=True)
med = res[res.相位 == '中位'].sort_values('超额vs医药EW', ascending=False)
print(med[show[:8]].to_string(index=False), flush=True)
print('\n存 outputs/redo_stageA_matrix.csv', flush=True)
