"""24 持仓集中度实验 — D1(12F透明月频)买前5/10/20家 vs Q5(前20%~41只)

用户提议: Q5等分也有几十家, 试更深的前5/10/20。同一套D1打分(core7+sat5, 12m全滚动号+权重),
唯一变量=取排名前N只等权。关注: 信号纯度(均值可能升) vs 集中度风险(波动/回撤/单票风险)。
  A. 自建月频: Q5 / Top20 / Top10 / Top5, 2020-01~2026-06(78期), 同月基准
  B. rqalpha T+1 (2020-01-23~2026-05-15): Top20/Top10/Top5 (Q5=13.40%已有)
⚠️ 四档挑最好=研究者自由度, 全部呈现不挑边; 前提是分数单调性(Q1<Q3<Q5)已由既有结果隐含。
输出: outputs/redo_stageK_topN.csv
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import spearmanr
import common

DATA = common.DATA
OUT = '/Users/lichanghaodemac/quant_strategy/projects/pharma_factor_workflow/outputs'
import os; os.makedirs(f'{OUT}/stageK', exist_ok=True)
COST = 0.0015
CORE7 = ['WorldQuant_alpha040', 'WorldQuant_alpha016', 'WorldQuant_alpha024', 'WorldQuant_alpha044',
         'WorldQuant_alpha011', 'dividend_yield_ttm', 'WorldQuant_alpha026']
SAT5 = ['book_to_market_ratio_lf', 'pb_ratio_lf', 'ps_ratio_ttm', 'VOLT20', 'VOLT60']
FACS = CORE7 + SAT5

P = common.load_panel(FACS)
fwd1 = P['fwd1']; me_sorted = P['me_sorted']; Z = P['Z']; stks = P['stks']
IDX = {m: i for i, m in enumerate(me_sorted)}
IC1 = {f: pd.Series({m: (lambda d: spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0] if len(d) >= 10 and d.iloc[:, 0].nunique() > 1 else np.nan)
                     (pd.concat([Z[f].loc[m], fwd1.loc[m]], axis=1).dropna()) for m in me_sorted if m in fwd1.index}) for f in FACS}

# D1打分(同17): 每月12m全滚动
SCORES = {}
for m in me_sorted[36:]:
    if m not in fwd1.index: continue
    i = IDX[m]; win = me_sorted[max(0, i - 12):i]
    ws, ss_ = [], []
    for f in FACS:
        v = IC1[f].reindex(win).dropna()
        if len(v) > 6 and v.std() > 0:
            ws.append(abs(v.mean() / v.std())); ss_.append(1 if v.mean() > 0 else -1)
        else:
            ws.append(0.0); ss_.append(1)
    ws = np.array(ws); ws = ws / ws.sum() if ws.sum() > 0 else np.ones(len(FACS)) / len(FACS)
    z = pd.DataFrame(index=stks)
    for f, s in zip(FACS, ss_): z[f] = s * Z[f].loc[m].reindex(stks)
    SCORES[m] = (z.fillna(0) * ws).sum(axis=1)

def top_list(sc, mode):
    sc = sc.dropna()
    if mode == 'Q5':
        try: q = pd.qcut(sc.rank(method='first'), 5, labels=False)
        except Exception: return None
        return list(sc[q == 4].index)
    lst = list(sc.nlargest(mode).index)
    return lst if len(lst) >= mode - 1 else None

def selfbuilt(mode):
    rets, months = [], []
    for m, sc in SCORES.items():
        lst = top_list(sc, mode)
        if not lst: continue
        vals = [fwd1.loc[m, o] for o in lst if o in fwd1.columns and pd.notna(fwd1.loc[m, o])]
        if len(vals) >= 3: rets.append(np.mean(vals) - COST); months.append(m)
    r = pd.Series(rets)
    return r, months

univ = pd.read_parquet(f'{DATA}/universe_monthend_pit.parquet'); univ['monthend'] = pd.DatetimeIndex(univ['monthend'])
ids = univ[univ['industry_l1'] == '医药生物'].groupby('monthend')['order_book_id'].apply(set).to_dict()
ph1 = pd.Series({m: fwd1.loc[m, [o for o in ss if o in fwd1.columns]].dropna().mean() for m, ss in ids.items() if m in fwd1.index}).dropna()

rows = []
SIGS = {}
for mode in ['Q5', 20, 10, 5]:
    r, months = selfbuilt(mode)
    ann = r.mean() * 12; sh = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    cum = (1 + r).cumprod(); dd = (cum / cum.cummax() - 1).min()
    bph = ph1.reindex(months).dropna().mean() * 12; bmk = P['mkt1'].reindex(months).dropna().mean() * 12
    rows.append({'配置': f'{mode if mode != "Q5" else "Q5(前20%≈41只)"}', '年化': round(ann, 4), '夏普': round(sh, 2),
                 '最大回撤': round(dd, 4), '超额vs医药EW': round(ann - bph, 4), '超额vs全AEW': round(ann - bmk, 4), '期数': len(r)})
    if mode != 'Q5': SIGS[str(mode)] = {m.strftime('%Y-%m-%d'): top_list(sc, mode) for m, sc in SCORES.items() if top_list(sc, mode)}
    print(f'自建 {mode}: {ann:.2%}/{sh:.2f}/回撤{dd:.1%} ({len(r)}期)', flush=True)

# ===== rqalpha T+1 =====
from rqalpha import run_func
from rqalpha.apis import *

def run_rq(signals, tag):
    pool = sorted({o for v in signals.values() for o in v})
    def init(context): context.signals = signals; context.pending = None
    def handle_bar(context, bar_dict):
        if context.pending is not None:
            target = context.pending; context.pending = None
            for o in list(context.portfolio.positions.keys()):
                if o not in target: order_target_percent(o, 0)
            w = 1.0 / len(target)
            for o in target: order_target_percent(o, w)
        today = context.now.strftime('%Y-%m-%d')
        if today in context.signals: context.pending = context.signals[today]
    config = {"base": {"start_date": "2020-01-10", "end_date": "2026-05-15",
                       "accounts": {"stock": 1_000_000}, "benchmark": "000300.XSHG", "stock_pool": pool,
                       "frequency": "1d", "run_type": "b"},
              "extra": {"log_level": "warning", "show": False},
              "mod": {"sys_analyser": {"enabled": True, "plot": False, "report_save_path": f'{OUT}/stageK/top{tag}'},
                      "sys_progress": {"enabled": False}, "sys_simulation": {"enabled": True}}}
    print(f'  rqalpha Top{tag}...', flush=True)
    res = run_func(config=config, init=init, handle_bar=handle_bar)
    return res['sys_analyser']['summary'] if isinstance(res, dict) and 'sys_analyser' in res else {}

print('\nrqalpha T+1 (2020-01-23~2026-05-15)...', flush=True)
rq_rows = []
for tag in ['20', '10', '5']:
    s = run_rq(SIGS[tag], tag)
    if s:
        rq_rows.append({'配置': f'Top{tag}(rqalpha)', '年化': round(s.get('annualized_returns'), 4), '夏普': round(s.get('sharpe'), 2),
                        '最大回撤': round(s.get('max_drawdown'), 4), '超额vs医药EW': '-', '超额vs全AEW': '-', '期数': '-'})

res = pd.DataFrame(rows + rq_rows)
res.to_csv(f'{OUT}/redo_stageK_topN.csv', index=False, encoding='utf-8-sig')
print('\n===== 持仓集中度: Q5 vs Top20/10/5 =====', flush=True)
print(res.to_string(index=False), flush=True)
print('\n对照: Q5 rqalpha T+1 = 13.40%/IR0.63/回撤30.3% (16脚本)', flush=True)
print('存 outputs/redo_stageK_topN.csv', flush=True)
