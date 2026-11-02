"""12 rqalpha 变体验证 — {股息单因子, 4F, 7F} × {月频, 季频} 全矩阵(T+1, 口径复刻06)

动机:
  ① 10的分时段IC: 2020+银行截面只剩股息一个跨期存活因子 → 真实引擎看"单股息"几斤几两
  ② 05/09统一窗口后 7F季 14.77%/1.01 手写最优 → 复秤
  ③ 月频对照(05手写: 4F月11.94/7F月12.90) — 引擎下月频是否仍逊于季频(成本摩擦)
  ④ 4F季重跑 = 机器自检: 应≈06的16.0%/0.75/alpha0.146, 不齐说明本脚本有drift

口径: 面板同06(2017-12起), 信号2020-12起(06的i>=36约定);
     月频配1m-IC权重、季频配3m-IC权重(05的horizon匹配约定);
     qcut(rank)取Q5≥3只, T+1执行, 基准000300.XSHG, end 2026-07-31(bundle截断有校验)。
输出: outputs/rqalpha_variants.csv + 控制台矩阵
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import spearmanr

DATA = '/Users/lichanghaodemac/quant_strategy/data'
OUT = '/Users/lichanghaodemac/quant_strategy/projects/bank_factor_workflow/outputs'

SETS = {
    '股息单因子': (['dividend_yield_ttm'], set()),
    '4F': (['dividend_yield_ttm', 'ev_to_ebitda_ttm', 'beta60', 'MOM_12M_1M'],
           {'ev_to_ebitda_ttm', 'beta60'}),
    '7F': (['dividend_yield_ttm', 'ev_to_ebitda_ttm', 'pe_ratio_ttm', 'market_cap_3',
            'net_profit_margin_ttm', 'beta60', 'MOM_12M_1M'],
           {'ev_to_ebitda_ttm', 'pe_ratio_ttm', 'beta60'}),
}
ALLF = sorted({f for v in SETS.values() for f in v[0]})
DBF = [f for f in ALLF if f not in ('beta60', 'MOM_12M_1M')]

# ===== 面板(同06, 算一次, 全变体共用) =====
price = pd.read_parquet(f'{DATA}/mainboard_price.parquet'); price['date'] = pd.to_datetime(price['date'])
close = price.pivot(index='date', columns='order_book_id', values='close').sort_index()
me = pd.DatetimeIndex(pd.read_parquet(f'{DATA}/monthend_dates.parquet')['monthend'])
mev = [m for m in me if m in close.index]; cm = close.loc[mev]
fwd1 = cm.pct_change(fill_method=None).shift(-1); fwd3 = cm.shift(-3) / cm - 1

def beta_p(n):
    rd = close.pct_change(fill_method=None); mk = rd.mean(axis=1)
    return rd.rolling(n).cov(mk).div(mk.rolling(n).var(), axis=0)
b60 = beta_p(60); mom121 = close.shift(20) / close.shift(240) - 1            # 各只算一次
bs = b60.loc[[m for m in mev if m in b60.index]].stack().rename('beta60').reset_index(); bs.columns = ['monthend', 'order_book_id', 'beta60']
ms = mom121.loc[[m for m in mev if m in mom121.index]].stack().rename('MOM_12M_1M').reset_index(); ms.columns = ['monthend', 'order_book_id', 'MOM_12M_1M']
factors = pd.read_parquet(f'{DATA}/factors_monthend.parquet'); factors['monthend'] = pd.DatetimeIndex(factors['monthend'])
univ = pd.read_parquet(f'{DATA}/universe_monthend_pit.parquet'); univ['monthend'] = pd.DatetimeIndex(univ['monthend'])
allf = factors[['order_book_id', 'monthend'] + DBF].merge(bs, on=['order_book_id', 'monthend']).merge(ms, on=['order_book_id', 'monthend'])
allf = allf.merge(univ[univ['industry_l1'] == '银行'][['order_book_id', 'monthend']], on=['order_book_id', 'monthend'], how='inner')
allf = allf[['order_book_id', 'monthend'] + ALLF].sort_values(['monthend', 'order_book_id']).reset_index(drop=True)
me_sorted = sorted(allf['monthend'].unique())
WIDE = {f: allf.pivot(index='monthend', columns='order_book_id', values=f) for f in ALLF}

def zsm(mat):
    lo, hi = mat.quantile(0.01, axis=1), mat.quantile(0.99, axis=1); out = mat.copy()
    for c in mat.columns: out[c] = out[c].clip(lo, hi, axis=0)
    return out.sub(out.mean(axis=1), axis=0).div(out.std(axis=1).replace(0, np.nan), axis=0)
Z = {f: zsm(WIDE[f]) for f in ALLF}                                          # 只算一次

def ic_series(f, fwd):
    s = pd.Series(index=me_sorted, dtype=float)
    for m in me_sorted:
        if m not in fwd.index: continue
        df = pd.concat([WIDE[f].loc[m], fwd.loc[m]], axis=1).dropna()
        if len(df) >= 10 and df.iloc[:, 0].nunique() > 1: s[m] = spearmanr(df.iloc[:, 0], df.iloc[:, 1])[0]
    return s
IC1 = {f: ic_series(f, fwd1) for f in ALLF}                                  # 月频权重用
IC3 = {f: ic_series(f, fwd3) for f in ALLF}                                  # 季频权重用

def make_signals(FACS, NEG, freq):
    ics = IC1 if freq == '月' else IC3                                       # ★ horizon匹配(05约定)
    W = {}
    for i, m in enumerate(me_sorted):
        past = me_sorted[max(0, i - 36):i]
        w = [abs((lambda v: v.mean() / v.std() if len(v) > 6 and v.std() > 0 else 0)(ics[f].reindex(past).dropna())) for f in FACS]
        w = np.array(w); W[m] = w / w.sum() if w.sum() > 0 else np.ones(len(FACS)) / len(FACS)
    rebal = ([m for i, m in enumerate(me_sorted) if i >= 36 and (i - 36) % 3 == 0] if freq == '季'
             else [m for i, m in enumerate(me_sorted) if i >= 36])           # 都从2020-12起(06面板i>=36)
    SIG = {}
    for m in rebal:
        z = pd.DataFrame(index=WIDE[FACS[0]].columns)
        for f in FACS:
            c = Z[f].loc[m]; z[f] = -c if f in NEG else c
        w = np.ones(len(FACS)) if len(FACS) == 1 else np.array(W[m])         # 单因子权重恒1
        score = (z.fillna(0) * w).sum(axis=1).dropna()
        if len(score) < 15: continue
        try: q = pd.qcut(score.rank(method='first'), 5, labels=False)
        except Exception: continue
        q5 = list(score[q == 4].index)
        if len(q5) >= 3: SIG[m.strftime('%Y-%m-%d')] = q5
    return SIG

# ===== rqalpha(T+1, 同06) =====
from rqalpha import run_func
from rqalpha.apis import *

def run_engine(name, SIG):
    BANKS = sorted({o for v in SIG.values() for o in v})
    def init(context):
        context.signals = SIG; context.last_day = None
    def handle_bar(context, bar_dict):
        today = context.now.strftime('%Y-%m-%d')
        target = context.signals.get(context.last_day)     # 昨日收盘信号 → 今日执行(T+1)
        context.last_day = today
        if target is None: return
        for o in list(context.portfolio.positions.keys()):
            if o not in target: order_target_percent(o, 0)
        w = 1.0 / len(target)
        for o in target: order_target_percent(o, w)
    config = {
        "base": {"start_date": min(SIG), "end_date": "2026-07-31",
                 "accounts": {"stock": 1_000_000}, "benchmark": "000300.XSHG", "stock_pool": BANKS,
                 "frequency": "1d", "run_type": "b"},
        "extra": {"log_level": "warning", "show": False},
        "mod": {"sys_analyser": {"enabled": True, "plot": False, "report_save_path": OUT},
                "sys_progress": {"enabled": False}, "sys_simulation": {"enabled": True}},
    }
    print(f'启动 rqalpha: {name} ({len(SIG)}个调仓点, 平均持仓{np.mean([len(v) for v in SIG.values()]):.1f}只, T+1)...', flush=True)
    res = run_func(config=config, init=init, handle_bar=handle_bar)
    s = res['sys_analyser']['summary']
    req_end, act_end = config['base']['end_date'], str(s.get('end_date', ''))[:10]
    if act_end and act_end < req_end:
        print(f'  ⚠️ bundle截断: 请求至{req_end}, 实际至{act_end}', flush=True)
    return s

rows = []
for freq in ['季', '月']:
    for name, (FACS, NEG) in SETS.items():
        SIG = make_signals(FACS, NEG, freq)
        s = run_engine(f'{name}·{freq}频', SIG)
        rows.append({'变体': f'{name}·{freq}频', '年化': round(s['annualized_returns'], 4), '夏普(日)': round(s['sharpe'], 2),
                     '最大回撤': round(s['max_drawdown'], 4), 'alpha': round(s['alpha'], 3),
                     'beta': round(s['beta'], 2), 'IR': round(s['information_ratio'], 2),
                     'Sortino': round(s['sortino'], 2),
                     '基准年化': round(s['benchmark_annualized_returns'], 4),
                     '区间': f"{str(s['start_date'])[:10]}~{str(s['end_date'])[:10]}"})

res = pd.DataFrame(rows)
res.to_csv(f'{OUT}/rqalpha_variants.csv', index=False, encoding='utf-8-sig')
print('\n=== rqalpha 全矩阵(T+1, vs HS300同期-0.98%) | 自检: 4F·季应≈06的16.0%/0.75/0.146 ===', flush=True)
print(res.to_string(index=False), flush=True)
