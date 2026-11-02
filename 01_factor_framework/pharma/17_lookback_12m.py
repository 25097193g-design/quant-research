"""17 重做Stage D — 回看窗 36m → 12m(号+权重全滚动) × 三个起点, 对照Stage A赢家(D0)

用户批准设计(2026-08-16): 池=core7+sat5(12F), 号=滚动(sign(过去12月IC均值), 全部因子),
权重=|ICIR|过去12月归一; 唯一变量=回看窗长度与起点, 拆行隔离效应:
  D1: 12m窗 / 2020-01起(与D0同起点, 纯窗口效应)
  D2: 12m窗 / 2021-01起(用户原始提议)
  D3: 12m窗 / 2018-01起(12m暖机允许的最早起点, 含2018大熊市, 铁存活判定的压力测试)
  D0(对照, 已有): 36m窗 / 2020-01起, 混合号 → Stage A: 季中位11.5%/+6.5pp/0.52
窗口细节: 月频IC窗=[i-12,i](fwd1无泄漏); 季频IC窗=[i-14,i-2](12个月全实现, embargo沿用)
输出: outputs/redo_stageD_lookback.csv
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import spearmanr
import common

OUT = '/Users/lichanghaodemac/quant_strategy/projects/pharma_factor_workflow/outputs'
COST = 0.0015

CORE7 = ['WorldQuant_alpha040', 'WorldQuant_alpha016', 'WorldQuant_alpha024', 'WorldQuant_alpha044',
         'WorldQuant_alpha011', 'dividend_yield_ttm', 'WorldQuant_alpha026']
SAT5 = ['book_to_market_ratio_lf', 'pb_ratio_lf', 'ps_ratio_ttm', 'VOLT20', 'VOLT60']
FACS = CORE7 + SAT5

P = common.load_panel(FACS)
fwd1, fwd3 = P['fwd1'], P['fwd3']
me_sorted, Z, stks = P['me_sorted'], P['Z'], P['stks']
IDX = {m: i for i, m in enumerate(me_sorted)}
print(f'面板: {len(stks)}只 × {len(me_sorted)}月 × {len(P["facs"])}因子', flush=True)

def ic_series(f, fwd, minn=10):
    return pd.Series({m: (lambda d: spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0] if len(d) >= minn and d.iloc[:, 0].nunique() > 1 else np.nan)
                      (pd.concat([Z[f].loc[m], fwd.loc[m]], axis=1).dropna()) for m in me_sorted if m in fwd.index})

IC1 = {f: ic_series(f, fwd1) for f in FACS}
IC3 = {f: ic_series(f, fwd3) for f in FACS}

univ = pd.read_parquet(f'{common.DATA}/universe_monthend_pit.parquet'); univ['monthend'] = pd.DatetimeIndex(univ['monthend'])
ids = univ[univ['industry_l1'] == '医药生物'].groupby('monthend')['order_book_id'].apply(set).to_dict()
ph1 = pd.Series({m: fwd1.loc[m, [o for o in ss if o in fwd1.columns]].dropna().mean() for m, ss in ids.items() if m in fwd1.index and len([o for o in ss if o in fwd1.columns]) >= 10}).sort_index()
ph3 = pd.Series({m: fwd3.loc[m, [o for o in ss if o in fwd3.columns]].dropna().mean() for m, ss in ids.items() if m in fwd3.index and len([o for o in ss if o in fwd3.columns]) >= 10}).dropna()
mkt1, mkt3 = P['mkt1'], P['mkt3']

def make_scores(freq, lb=12):
    """号+权重都从过去lb个月的IC算(全滚动)。月频窗=[i-lb,i]; 季频窗=[i-lb-2,i-2](embargo)。"""
    fwd = fwd1 if freq == '月' else fwd3
    IC = IC1 if freq == '月' else IC3
    out = {}
    for m in me_sorted:
        if m not in fwd.index: continue
        i = IDX[m]
        win = me_sorted[max(0, i - lb):i] if freq == '月' else me_sorted[max(0, i - lb - 2):i - 2]
        ws, ss_ = [], []
        for f in FACS:
            v = IC[f].reindex(win).dropna()
            if len(v) > 6 and v.std() > 0:
                ws.append(abs(v.mean() / v.std())); ss_.append(1 if v.mean() > 0 else -1)
            else:
                ws.append(0.0); ss_.append(1)
        ws = np.array(ws); ws = ws / ws.sum() if ws.sum() > 0 else np.ones(len(FACS)) / len(FACS)
        z = pd.DataFrame(index=stks)
        for f, s in zip(FACS, ss_): z[f] = s * Z[f].loc[m].reindex(stks)
        out[m] = (z.fillna(0) * ws).sum(axis=1)
    return out

def bt(scores, rebal, fwd):
    rets, months = [], []
    for m in rebal:
        if m not in fwd.index or m not in scores: continue
        s = scores[m].dropna()
        if s.nunique() < 5: continue
        try: q = pd.qcut(s.rank(method='first'), 5, labels=False)
        except Exception: continue
        q5 = s[q == 4].index
        if len(q5) < 3: continue
        vals = [fwd.loc[m, o] for o in q5 if o in fwd.columns and pd.notna(fwd.loc[m, o])]
        if len(vals) >= 2: rets.append(np.mean(vals) - COST); months.append(m)
    return pd.Series(rets, index=months)

def metrics(r, py, bph_s, bmk_s):
    if len(r) < 6: return None
    ann = r.mean() * py
    sh = r.mean() / r.std() * np.sqrt(py) if r.std() > 0 else 0
    cum = (1 + r).cumprod(); dd = (cum / cum.cummax() - 1).min()
    bph = bph_s.reindex(r.index).dropna().mean() * py
    bmk = bmk_s.reindex(r.index).dropna().mean() * py
    return {'年化': round(ann, 4), '夏普': round(sh, 2), '最大回撤': round(dd, 4),
            '超额vs医药EW': round(ann - bph, 4), '超额vs全AEW': round(ann - bmk, 4), '期数': len(r)}

STARTS = [('D1:12m/2020-01', 36), ('D2:12m/2021-01', 48), ('D3:12m/2018-01', 12)]
rows = []
for freq in ['月', '季']:
    scores = make_scores(freq)
    fwd = fwd1 if freq == '月' else fwd3
    py = 12 if freq == '月' else 4
    bph = ph1 if freq == '月' else ph3
    bmk = mkt1 if freq == '月' else mkt3
    for name, S in STARTS:
        if freq == '月':
            r = bt(scores, me_sorted[S:], fwd)
            m_ = metrics(r, py, bph, bmk)
            if m_: rows.append({'配置': name, '频率': '月', '相位': '-', **m_})
        else:
            per = []
            for ph in [0, 1, 2]:
                rebal = [m for i, m in enumerate(me_sorted) if i >= S and (i - S) % 3 == ph]
                r = bt(scores, rebal, fwd)
                m_ = metrics(r, py, bph, bmk)
                if m_:
                    rows.append({'配置': name, '频率': '季', '相位': ph, **m_}); per.append(m_)
            if per:
                rows.append({'配置': name, '频率': '季', '相位': '中位',
                             '年化': round(float(np.median([p['年化'] for p in per])), 4),
                             '夏普': round(float(np.median([p['夏普'] for p in per])), 2),
                             '最大回撤': round(float(np.median([p['最大回撤'] for p in per])), 4),
                             '超额vs医药EW': round(float(np.median([p['超额vs医药EW'] for p in per])), 4),
                             '超额vs全AEW': round(float(np.median([p['超额vs全AEW'] for p in per])), 4),
                             '期数': int(np.median([p['期数'] for p in per]))})
res = pd.DataFrame(rows)
res.to_csv(f'{OUT}/redo_stageD_lookback.csv', index=False, encoding='utf-8-sig')
print('\n===== Stage D: 12m全滚动 × 三起点 (对照 D0=36m混合号/2020-01: 季中位11.5%/+6.5pp/0.52) =====', flush=True)
print(res[['配置', '频率', '相位', '年化', '夏普', '最大回撤', '超额vs医药EW', '超额vs全AEW', '期数']].to_string(index=False), flush=True)
print('\n存 outputs/redo_stageD_lookback.csv', flush=True)
