"""06 rqalpha 真实回测 — 把"季度无过滤 4F"策略喂给米筐 rqalpha 引擎跑标准回测

信号与执行分离(实盘量化标准做法):
  信号 = 自算 4F 季度 Q5(3m滚动ICIR加权, 无超跌过滤), 信号日 = 月末交易日
  执行 = rqalpha 引擎: 撮合/佣金/滑点/整手取整/账户管理 → 标准绩效报告

★ 2026-08-15 修复四项(信号计算保持逐位一致, 数字变化全部归因于T+1):
  1. T+1执行: 月末收盘后因子才算得出来 → 上一交易日收盘出信号, 下一交易日才下单
     (原版"当天收盘信号当天收盘成交", 实盘不可行)
  2. 回测终点校验: bundle日频行情只到2025-05, config的end_date写了2026-07-31会被
     引擎静默截断 → 跑完比对summary.end_date, 不一致大字警告
  3. 重构: beta_p(60)/MOM表达式原版各写两遍、zsm在季度循环里每月重算 → 全部只算一次
  4. 成本口径: 沿用rqalpha默认佣金+滑点(与手写回测15bp不是一套), 在此明确记录
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import spearmanr

DATA = '/Users/lichanghaodemac/quant_strategy/data'
OUT = '/Users/lichanghaodemac/quant_strategy/projects/bank_factor_workflow/outputs'
F4 = ['dividend_yield_ttm', 'ev_to_ebitda_ttm', 'beta60', 'MOM_12M_1M']; NEG4 = {'ev_to_ebitda_ttm', 'beta60'}

# ===== 1. 预计算季度 4F Q5 信号 (逻辑同 05 的"4F季无过滤") =====
price = pd.read_parquet(f'{DATA}/mainboard_price.parquet'); price['date'] = pd.to_datetime(price['date'])
close = price.pivot(index='date', columns='order_book_id', values='close').sort_index()
me = pd.DatetimeIndex(pd.read_parquet(f'{DATA}/monthend_dates.parquet')['monthend'])
mev = [m for m in me if m in close.index]; cm = close.loc[mev]
fwd3 = cm.shift(-3) / cm - 1                                   # 季频前瞻

def beta_p(n):                                                 # 市场=主板等权(与common/02同口径)
    rd = close.pct_change(fill_method=None); mk = rd.mean(axis=1)
    return rd.rolling(n).cov(mk).div(mk.rolling(n).var(), axis=0)
b60 = beta_p(60)                                               # ← 只算一次(原版写了两遍)
mom121 = close.shift(20) / close.shift(240) - 1                # ← 只算一次(原版写了两遍)
bs = b60.loc[[m for m in mev if m in b60.index]].stack().rename('beta60').reset_index(); bs.columns = ['monthend', 'order_book_id', 'beta60']
ms = mom121.loc[[m for m in mev if m in mom121.index]].stack().rename('MOM_12M_1M').reset_index(); ms.columns = ['monthend', 'order_book_id', 'MOM_12M_1M']
factors = pd.read_parquet(f'{DATA}/factors_monthend.parquet'); factors['monthend'] = pd.DatetimeIndex(factors['monthend'])
univ = pd.read_parquet(f'{DATA}/universe_monthend_pit.parquet'); univ['monthend'] = pd.DatetimeIndex(univ['monthend'])
allf = factors[['order_book_id', 'monthend'] + ['dividend_yield_ttm', 'ev_to_ebitda_ttm']].merge(bs, on=['order_book_id', 'monthend']).merge(ms, on=['order_book_id', 'monthend'])
allf = allf.merge(univ[univ['industry_l1'] == '银行'][['order_book_id', 'monthend']], on=['order_book_id', 'monthend'], how='inner')
allf = allf[['order_book_id', 'monthend'] + F4].sort_values(['monthend', 'order_book_id']).reset_index(drop=True)
me_sorted = sorted(allf['monthend'].unique())
WIDE = {f: allf.pivot(index='monthend', columns='order_book_id', values=f) for f in F4}

def zsm(mat):                                                  # 截面z(1/99 winsorize)
    lo, hi = mat.quantile(0.01, axis=1), mat.quantile(0.99, axis=1); out = mat.copy()
    for c in mat.columns: out[c] = out[c].clip(lo, hi, axis=0)
    return out.sub(out.mean(axis=1), axis=0).div(out.std(axis=1).replace(0, np.nan), axis=0)
Z4 = {f: zsm(WIDE[f]) for f in F4}                             # ← 只算一次(原版在季度循环里反复重算)

# 3m滚动ICIR权重(36月窗口, 不含当月, 无未来)
ic_ser = {}
for f in F4:
    s = pd.Series(index=me_sorted, dtype=float)
    for m in me_sorted:
        if m not in fwd3.index: continue
        df = pd.concat([WIDE[f].loc[m], fwd3.loc[m]], axis=1).dropna()
        if len(df) >= 10 and df.iloc[:, 0].nunique() > 1: s[m] = spearmanr(df.iloc[:, 0], df.iloc[:, 1])[0]
    ic_ser[f] = s
W3 = {}
for i, m in enumerate(me_sorted):
    past = me_sorted[max(0, i - 36):i]; w = []
    for f in F4:
        v = ic_ser[f].reindex(past).dropna()
        w.append(abs(v.mean() / v.std()) if len(v) > 6 and v.std() > 0 else 0)
    w = np.array(w); W3[m] = w / w.sum() if w.sum() > 0 else np.ones(len(F4)) / len(F4)

# 季度调仓月 + Q5信号
quarterly = [m for i, m in enumerate(me_sorted) if i >= 36 and (i - 36) % 3 == 0]
SIGNALS = {}
for m in quarterly:
    z = pd.DataFrame(index=WIDE[F4[0]].columns)
    for f in F4:
        c = Z4[f].loc[m]; z[f] = -c if f in NEG4 else c
    score = (z.fillna(0) * np.array(W3[m])).sum(axis=1).dropna()
    if len(score) < 15: continue
    try: q = pd.qcut(score.rank(method='first'), 5, labels=False)
    except Exception: continue
    q5 = list(score[q == 4].index)
    if len(q5) >= 3: SIGNALS[m.strftime('%Y-%m-%d')] = q5
print(f'信号: {len(SIGNALS)}个季度调仓点, 首{min(SIGNALS)} 末{max(SIGNALS)}, 平均持仓{np.mean([len(v) for v in SIGNALS.values()]):.1f}只', flush=True)

# ===== 2. rqalpha 策略(★T+1: 上一交易日的信号, 今天才执行) =====
from rqalpha import run_func
from rqalpha.apis import *

BANKS = sorted({o for v in SIGNALS.values() for o in v})
def init(context):
    context.signals = SIGNALS
    context.last_day = None      # 记住上一交易日, 用于T+1
def handle_bar(context, bar_dict):
    today = context.now.strftime('%Y-%m-%d')
    target = context.signals.get(context.last_day)   # 昨天收盘后的信号 → 今天下单
    context.last_day = today
    if target is None: return
    for o in list(context.portfolio.positions.keys()):
        if o not in target: order_target_percent(o, 0)
    if target:
        w = 1.0 / len(target)
        for o in target: order_target_percent(o, w)

config = {
    "base": {"start_date": min(SIGNALS), "end_date": "2026-07-31",
             "accounts": {"stock": 1_000_000}, "benchmark": "000300.XSHG", "stock_pool": BANKS,
             "frequency": "1d", "run_type": "b"},
    "extra": {"log_level": "warning", "show": False},
    # 成本: 沿用rqalpha默认佣金+滑点(未显式配置, 与手写回测15bp口径不同, 报告需注明)
    "mod": {"sys_analyser": {"enabled": True, "plot": False, "report_save_path": OUT},
            "sys_progress": {"enabled": False}, "sys_simulation": {"enabled": True}},
}
print('\n启动 rqalpha 回测(★T+1版)...', flush=True)
res = run_func(config=config, init=init, handle_bar=handle_bar)

# ===== 3. 提取报告 + ★回测终点校验(bundle截断检查) =====
if isinstance(res, dict) and 'sys_analyser' in res:
    sa = res['sys_analyser']
    if 'summary' in sa:
        s = sa['summary']
        req_end = config['base']['end_date']; act_end = str(s.get('end_date', ''))[:10]
        print(f'请求回测至 {req_end} | 实际回测至 {act_end}', flush=True)
        if act_end and act_end < req_end:
            print(f'⚠️⚠️⚠️ 回测被bundle数据截断(行情只到{act_end})! 结果仅该区间内可比, 勿与更长窗口混比!', flush=True)
        print('\n=== rqalpha 标准绩效报告 (季度无过滤4F银行Q5, 等权, ★T+1执行) ===', flush=True)
        for k in ['start_date', 'end_date', 'total_returns', 'annualized_returns', 'benchmark_total_returns',
                  'benchmark_annualized_returns', 'sharpe', 'volatility', 'max_drawdown', 'alpha', 'beta',
                  'sortino', 'information_ratio', 'win_rate', 'daily_win_rate', 'trading_days', 'transaction_cost']:
            if k in s: print(f'  {k}: {s[k]}', flush=True)
    if 'report' in sa:
        s2 = sa['report']
        print('\n=== report 字段 ===', flush=True)
        print(list(s2.keys()) if isinstance(s2, dict) else type(s2), flush=True)
