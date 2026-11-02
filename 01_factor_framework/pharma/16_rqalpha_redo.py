"""16 重做Stage C — 米筐终验: D1配置(core7+sat5, 12m全滚动, 月频) + sat5对照

信号生成与 17(make_scores, lb=12, 全滚动号+权重)完全一致, 与rqalpha执行分离:
  - 信号=每月末 Q5(qcut5 前20%)等权名单; T+1执行(信号日次一交易日bar成交)
  - 窗口 2020-01-23 ~ 2026-05-15 (bundle实测终点; 2026-05/06的信号不再触发)
  - 基准: 沪深300(rqalpha自带同窗) + 医药等权/全A等权(日历月对齐另算)
  - 报告写 outputs/stageC/(不动旧portfolio/trades等文件)
输出: outputs/redo_stageC_rqalpha.csv
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import spearmanr
import common

DATA = common.DATA
OUT = '/Users/lichanghaodemac/quant_strategy/projects/pharma_factor_workflow/outputs'
import os; os.makedirs(f'{OUT}/stageC', exist_ok=True)

CORE7 = ['WorldQuant_alpha040', 'WorldQuant_alpha016', 'WorldQuant_alpha024', 'WorldQuant_alpha044',
         'WorldQuant_alpha011', 'dividend_yield_ttm', 'WorldQuant_alpha026']
SAT5 = ['book_to_market_ratio_lf', 'pb_ratio_lf', 'ps_ratio_ttm', 'VOLT20', 'VOLT60']

P = common.load_panel(CORE7 + SAT5)
fwd1 = P['fwd1']; me_sorted = P['me_sorted']; Z = P['Z']; stks = P['stks']
IDX = {m: i for i, m in enumerate(me_sorted)}
IC1 = {f: pd.Series({m: (lambda d: spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0] if len(d) >= 10 and d.iloc[:, 0].nunique() > 1 else np.nan)
                     (pd.concat([Z[f].loc[m], fwd1.loc[m]], axis=1).dropna()) for m in me_sorted if m in fwd1.index}) for f in CORE7 + SAT5}
print('IC1序列完成', flush=True)

def signals_12m(facs, start_i=36, lb=12):
    sig = {}
    for m in me_sorted[start_i:]:
        if m not in fwd1.index: continue
        i = IDX[m]; win = me_sorted[max(0, i - lb):i]
        ws, ss_ = [], []
        for f in facs:
            v = IC1[f].reindex(win).dropna()
            if len(v) > 6 and v.std() > 0:
                ws.append(abs(v.mean() / v.std())); ss_.append(1 if v.mean() > 0 else -1)
            else:
                ws.append(0.0); ss_.append(1)
        ws = np.array(ws); ws = ws / ws.sum() if ws.sum() > 0 else np.ones(len(facs)) / len(facs)
        z = pd.DataFrame(index=stks)
        for f, s in zip(facs, ss_): z[f] = s * Z[f].loc[m].reindex(stks)
        sc = (z.fillna(0) * ws).sum(axis=1).dropna()
        if sc.nunique() < 5: continue
        try: q = pd.qcut(sc.rank(method='first'), 5, labels=False)
        except Exception: continue
        q5 = list(sc[q == 4].index)
        if len(q5) >= 3: sig[m.strftime('%Y-%m-%d')] = q5
    return sig

SIG_D1 = signals_12m(CORE7 + SAT5)
SIG_SAT = signals_12m(SAT5)
for nm, s in [('D1(core7+sat5/12m/月)', SIG_D1), ('sat5(12m/月)', SIG_SAT)]:
    print(f'{nm}: {len(s)}个月度信号, 平均持仓{np.mean([len(v) for v in s.values()]):.1f}只, {min(s)}~{max(s)}', flush=True)

# ===== 同窗基准(日历月对齐): 医药等权 / 全A等权 月频 =====
univ = pd.read_parquet(f'{DATA}/universe_monthend_pit.parquet'); univ['monthend'] = pd.DatetimeIndex(univ['monthend'])
ids = univ[univ['industry_l1'] == '医药生物'].groupby('monthend')['order_book_id'].apply(set).to_dict()
ph1 = pd.Series({m: fwd1.loc[m, [o for o in ss if o in fwd1.columns]].dropna().mean() for m, ss in ids.items() if m in fwd1.index and len([o for o in ss if o in fwd1.columns]) >= 10}).sort_index()
W0, W1 = pd.Timestamp('2020-01-23'), pd.Timestamp('2026-05-15')
ph_win = ph1[(ph1.index >= W0) & (ph1.index <= W1)]
mk_win = P['mkt1'][(P['mkt1'].index >= W0) & (P['mkt1'].index <= W1)]
print(f'同窗基准(2020-01~2026-05, 月频): 医药等权 {ph_win.mean()*12:.2%} / 全A等权 {mk_win.mean()*12:.2%}', flush=True)

# ===== rqalpha (T+1) =====
from rqalpha import run_func
from rqalpha.apis import *

def run_rq(signals, tag):
    pool = sorted({o for v in signals.values() for o in v})
    def init(context):
        context.signals = signals; context.pending = None
    def handle_bar(context, bar_dict):
        if context.pending is not None:          # T+1: 昨天的信号今天执行
            target = context.pending; context.pending = None
            for o in list(context.portfolio.positions.keys()):
                if o not in target: order_target_percent(o, 0)
            w = 1.0 / len(target)
            for o in target: order_target_percent(o, w)
        today = context.now.strftime('%Y-%m-%d')
        if today in context.signals: context.pending = context.signals[today]   # 今日出信号, 明日执行
    config = {
        "base": {"start_date": "2020-01-10", "end_date": "2026-05-15",
                 "accounts": {"stock": 1_000_000}, "benchmark": "000300.XSHG", "stock_pool": pool,
                 "frequency": "1d", "run_type": "b"},
        "extra": {"log_level": "warning", "show": False},
        "mod": {"sys_analyser": {"enabled": True, "plot": False, "report_save_path": f'{OUT}/stageC/{tag}'},
                "sys_progress": {"enabled": False}, "sys_simulation": {"enabled": True}},
    }
    print(f'\n启动 rqalpha ({tag})...', flush=True)
    res = run_func(config=config, init=init, handle_bar=handle_bar)
    if isinstance(res, dict) and 'sys_analyser' in res and 'summary' in res['sys_analyser']:
        return res['sys_analyser']['summary']
    print('!! 未返回summary', flush=True); return None

rows = []
for tag, sig in [('D1_12m月', SIG_D1), ('sat5_12m月', SIG_SAT)]:
    s = run_rq(sig, tag)
    if s:
        rows.append({'信号': tag, **{k: s.get(k) for k in
              ['annualized_returns', 'total_returns', 'sharpe', 'volatility', 'max_drawdown', 'alpha', 'beta',
               'information_ratio', 'win_rate', 'benchmark_annualized_returns', 'transaction_cost']}})
res_df = pd.DataFrame(rows)
res_df.to_csv(f'{OUT}/redo_stageC_rqalpha.csv', index=False, encoding='utf-8-sig')
print('\n===== Stage C: rqalpha T+1 (2020-01-23~2026-05-15, 基准沪深300同窗) =====', flush=True)
print(res_df.to_string(index=False), flush=True)
print(f'\n同窗对照: 医药等权 {ph_win.mean()*12:.2%} / 全A等权 {mk_win.mean()*12:.2%} (月频日历窗)', flush=True)
print('存 outputs/redo_stageC_rqalpha.csv (明细在 outputs/stageC/)', flush=True)
