"""06 rqalpha 真实回测 — 医药 24F/4F 季度Q5 喂米筐引擎

为什么用 rqalpha: 米筐开源事件驱动回测引擎, 出标准绩效报告(年化/夏普/回撤/alpha/beta),
还原真实下单(滑点/撮合/账户)。信号与执行分离: 信号=05算好的"24F季/4F季无过滤"Q5, 执行交给rqalpha。

⚠️ rqalpha bundle(stocks.h5) 生成于 2025-05, 日频行情约覆盖到 2025-05;
   前置分析用 rqdatac 实时数据到 2026 → rqalpha 区间 < 自建回测, 仅可比区间内。
⚠️ 04/05 结论: 医药24F>7F>4F(去冗余丢alpha) → 这里同时跑 24F 和 4F, 看真引擎是否复现。
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import spearmanr

DATA = '/Users/lichanghaodemac/quant_strategy/data'
OUT = '/Users/lichanghaodemac/quant_strategy/projects/pharma_factor_workflow/outputs'
SECTOR = '医药生物'
F24 = ['WorldQuant_alpha003', 'WorldQuant_alpha006', 'WorldQuant_alpha011',
       'WorldQuant_alpha013', 'WorldQuant_alpha014', 'WorldQuant_alpha015',
       'WorldQuant_alpha016', 'WorldQuant_alpha024', 'WorldQuant_alpha027',
       'WorldQuant_alpha037', 'WorldQuant_alpha040', 'WorldQuant_alpha044',
       'WorldQuant_alpha050', 'WorldQuant_alpha055', 'WorldQuant_alpha088',
       'book_to_market_ratio_lf', 'dividend_yield_ttm', 'ocf_to_debt_ttm',
       'pb_ratio_lf', 'pe_ratio_lyr', 'ps_ratio_ttm',
       'surplus_cash_protection_multiples_ttm', 'VOLT20', 'VOLT60']
NEG24 = {'VOLT20', 'VOLT60', 'pb_ratio_lf', 'pe_ratio_lyr', 'ps_ratio_ttm'}
F4 = ['WorldQuant_alpha040', 'WorldQuant_alpha016', 'surplus_cash_protection_multiples_ttm', 'WorldQuant_alpha050']
NEG4 = set()

# ===== 1. 预计算季度 Q5 信号 (3m滚动ICIR加权, 无超跌过滤) =====
price = pd.read_parquet(f'{DATA}/mainboard_price.parquet'); price['date'] = pd.to_datetime(price['date'])
close = price.pivot(index='date', columns='order_book_id', values='close').sort_index()
me = pd.DatetimeIndex(pd.read_parquet(f'{DATA}/monthend_dates.parquet')['monthend'])
mev = [m for m in me if m in close.index]; cm = close.loc[mev]
fwd3 = cm.shift(-3) / cm - 1
factors = pd.read_parquet(f'{DATA}/factors_monthend.parquet'); factors['monthend'] = pd.DatetimeIndex(factors['monthend'])
univ = pd.read_parquet(f'{DATA}/universe_monthend_pit.parquet'); univ['monthend'] = pd.DatetimeIndex(univ['monthend'])
allf = factors[['order_book_id', 'monthend'] + F24]
allf = allf.merge(univ[univ['industry_l1'] == SECTOR][['order_book_id', 'monthend']], on=['order_book_id', 'monthend'], how='inner')
allf = allf[['order_book_id', 'monthend'] + F24].sort_values(['monthend', 'order_book_id']).reset_index(drop=True)
allf[F24] = allf[F24].replace([np.inf, -np.inf], np.nan)   # 清inf
me_sorted = sorted(allf['monthend'].unique())
WIDE = {f: allf.pivot(index='monthend', columns='order_book_id', values=f) for f in F24}

def zsm(mat):
    lo, hi = mat.quantile(0.01, axis=1), mat.quantile(0.99, axis=1); out = mat.copy()
    for c in mat.columns: out[c] = out[c].clip(lo, hi, axis=0)
    return out.sub(out.mean(axis=1), axis=0).div(out.std(axis=1).replace(0, np.nan), axis=0)

def make_signals(facs, neg):
    ic_ser = {}
    for f in facs:
        s = pd.Series(index=me_sorted, dtype=float)
        for m in me_sorted:
            if m not in fwd3.index: continue
            df = pd.concat([WIDE[f].loc[m], fwd3.loc[m]], axis=1).dropna()
            if len(df) >= 10 and df.iloc[:, 0].nunique() > 1: s[m] = spearmanr(df.iloc[:, 0], df.iloc[:, 1])[0]
        ic_ser[f] = s
    W3 = {}
    for i, m in enumerate(me_sorted):
        past = me_sorted[max(0, i - 36):i]; w = []
        for f in facs:
            v = ic_ser[f].reindex(past).dropna()
            w.append(abs(v.mean() / v.std()) if len(v) > 6 and v.std() > 0 else 0)
        w = np.array(w); W3[m] = w / w.sum() if w.sum() > 0 else np.ones(len(facs)) / len(facs)
    quarterly = [m for i, m in enumerate(me_sorted) if i >= 36 and (i - 36) % 3 == 0]
    sig = {}
    for m in quarterly:
        z = pd.DataFrame(index=WIDE[facs[0]].columns)
        for f in facs:
            c = zsm(WIDE[f]).loc[m]; z[f] = -c if f in neg else c
        score = (z.fillna(0) * np.array(W3[m])).sum(axis=1).dropna()
        if len(score) < 15: continue
        try: q = pd.qcut(score.rank(method='first'), 5, labels=False)
        except: continue
        q5 = list(score[q == 4].index)
        if len(q5) >= 3: sig[m.strftime('%Y-%m-%d')] = q5
    return sig

SIGNALS24 = make_signals(F24, NEG24)
SIGNALS4 = make_signals(F4, NEG4)
print(f'24F信号: {len(SIGNALS24)}个季度点, 平均持仓{np.mean([len(v) for v in SIGNALS24.values()]):.1f}只, 区间{min(SIGNALS24)}~{max(SIGNALS24)}', flush=True)
print(f'4F信号:  {len(SIGNALS4)}个季度点, 平均持仓{np.mean([len(v) for v in SIGNALS4.values()]):.1f}只', flush=True)

# ===== 2. rqalpha 策略(24F & 4F) =====
from rqalpha import run_func
from rqalpha.apis import *

def run_rqalpha(signals, tag):
    pool = sorted({o for v in signals.values() for o in v})
    def init(context): context.signals = signals
    def handle_bar(context, bar_dict):
        today = context.now.strftime('%Y-%m-%d')
        if today not in context.signals: return
        target = context.signals[today]
        for o in list(context.portfolio.positions.keys()):
            if o not in target: order_target_percent(o, 0)
        if target:
            w = 1.0 / len(target)
            for o in target: order_target_percent(o, w)
    config = {
        "base": {"start_date": min(signals), "end_date": "2025-05-31",
                 "accounts": {"stock": 1_000_000}, "benchmark": "000300.XSHG", "stock_pool": pool,
                 "frequency": "1d", "run_type": "b"},
        "extra": {"log_level": "warning", "show": False},
        "mod": {"sys_analyser": {"enabled": True, "plot": False, "report_save_path": OUT},
                "sys_progress": {"enabled": False}, "sys_simulation": {"enabled": True}},
    }
    print(f'\n启动 rqalpha 回测 ({tag})...', flush=True)
    res = run_func(config=config, init=init, handle_bar=handle_bar)
    if isinstance(res, dict) and 'sys_analyser' in res and 'summary' in res['sys_analyser']:
        s = res['sys_analyser']['summary']
        print(f'=== rqalpha 标准绩效报告 ({tag} 无过滤医药Q5, 等权) ===', flush=True)
        for k in ['start_date', 'end_date', 'total_returns', 'annualized_returns', 'benchmark_total_returns',
                  'benchmark_annualized_returns', 'sharpe', 'volatility', 'max_drawdown', 'alpha', 'beta',
                  'sortino', 'information_ratio', 'win_rate', 'daily_win_rate', 'trading_days', 'transaction_cost']:
            if k in s: print(f'  {k}: {s[k]}', flush=True)
        return s
    print('!! rqalpha 未返回 sys_analyser, 键:', list(res.keys()) if isinstance(res, dict) else type(res), flush=True)
    return None

s24 = run_rqalpha(SIGNALS24, '24F季')
s4 = run_rqalpha(SIGNALS4, '4F季')
rows = []
for tag, s in [('24F季', s24), ('4F季', s4)]:
    if s: rows.append({'信号': tag, **{k: s.get(k) for k in
                     ['annualized_returns', 'sharpe', 'volatility', 'max_drawdown', 'alpha', 'beta', 'information_ratio', 'win_rate']}})
pd.DataFrame(rows).to_csv(f'{OUT}/rqalpha_compare.csv', index=False, encoding='utf-8-sig')
print('\n存 outputs/rqalpha_compare.csv', flush=True)
