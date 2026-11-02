"""14 月频起点稳定性检验 — 验证"月频天然免疫起始月抽签"

13 证明季频三相位离散5.3pp(现行相位恰为最高); 月频每月都调仓, 起点只影响窗口边缘一个月,
理论上方差应≈0。本脚本实证: {4F, 7F} × 月频 × 3个起点(2020-11/12/2021-01) = 6个引擎回测,
对照13的季频离散(4F 5.29pp / 7F 5.36pp)。
输出: outputs/monthly_stability.csv
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import spearmanr

DATA = '/Users/lichanghaodemac/quant_strategy/data'
OUT = '/Users/lichanghaodemac/quant_strategy/projects/bank_factor_workflow/outputs'

SETS = {
    '4F': (['dividend_yield_ttm', 'ev_to_ebitda_ttm', 'beta60', 'MOM_12M_1M'],
           {'ev_to_ebitda_ttm', 'beta60'}),
    '7F': (['dividend_yield_ttm', 'ev_to_ebitda_ttm', 'pe_ratio_ttm', 'market_cap_3',
            'net_profit_margin_ttm', 'beta60', 'MOM_12M_1M'],
           {'ev_to_ebitda_ttm', 'pe_ratio_ttm', 'beta60'}),
}
ANCHORS = {'2020-11起': '2020-11-01', '2020-12起(现行)': '2020-12-01', '2021-01起': '2021-01-01'}
ALLF = sorted({f for v in SETS.values() for f in v[0]})
DBF = [f for f in ALLF if f not in ('beta60', 'MOM_12M_1M')]

price = pd.read_parquet(f'{DATA}/mainboard_price.parquet'); price['date'] = pd.to_datetime(price['date'])
close = price.pivot(index='date', columns='order_book_id', values='close').sort_index()
me = pd.DatetimeIndex(pd.read_parquet(f'{DATA}/monthend_dates.parquet')['monthend'])
mev = [m for m in me if m in close.index]; cm = close.loc[mev]
fwd1 = cm.pct_change(fill_method=None).shift(-1)

def beta_p(n):
    rd = close.pct_change(fill_method=None); mk = rd.mean(axis=1)
    return rd.rolling(n).cov(mk).div(mk.rolling(n).var(), axis=0)
b60 = beta_p(60); mom121 = close.shift(20) / close.shift(240) - 1
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
Z = {f: zsm(WIDE[f]) for f in ALLF}

ic1 = {}
for f in ALLF:
    s = pd.Series(index=me_sorted, dtype=float)
    for m in me_sorted:
        if m not in fwd1.index: continue
        df = pd.concat([WIDE[f].loc[m], fwd1.loc[m]], axis=1).dropna()
        if len(df) >= 10 and df.iloc[:, 0].nunique() > 1: s[m] = spearmanr(df.iloc[:, 0], df.iloc[:, 1])[0]
    ic1[f] = s

def make_signals(FACS, NEG, anchor):
    ai = next(i for i, m in enumerate(me_sorted) if m >= pd.Timestamp(anchor))
    W = {}
    for i, m in enumerate(me_sorted):
        past = me_sorted[max(0, i - 36):i]
        w = np.array([abs((lambda v: v.mean() / v.std() if len(v) > 6 and v.std() > 0 else 0)(ic1[f].reindex(past).dropna())) for f in FACS])
        W[m] = w / w.sum() if w.sum() > 0 else np.ones(len(FACS)) / len(FACS)
    rebal = [m for i, m in enumerate(me_sorted) if i >= ai]      # 月频: 每个月都是调仓月
    SIG = {}
    for m in rebal:
        z = pd.DataFrame(index=WIDE[FACS[0]].columns)
        for f in FACS:
            c = Z[f].loc[m]; z[f] = -c if f in NEG else c
        score = (z.fillna(0) * np.array(W[m])).sum(axis=1).dropna()
        if len(score) < 15: continue
        try: q = pd.qcut(score.rank(method='first'), 5, labels=False)
        except Exception: continue
        q5 = list(score[q == 4].index)
        if len(q5) >= 3: SIG[m.strftime('%Y-%m-%d')] = q5
    return SIG

from rqalpha import run_func
from rqalpha.apis import *

def run_engine(name, SIG):
    BANKS = sorted({o for v in SIG.values() for o in v})
    def init(context):
        context.signals = SIG; context.last_day = None
    def handle_bar(context, bar_dict):
        today = context.now.strftime('%Y-%m-%d')
        target = context.signals.get(context.last_day)
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
    print(f'启动 rqalpha: {name} ({len(SIG)}个调仓点, T+1)...', flush=True)
    res = run_func(config=config, init=init, handle_bar=handle_bar)
    return res['sys_analyser']['summary']

rows = []
for set_name, (FACS, NEG) in SETS.items():
    anns = []
    for a_label, anchor in ANCHORS.items():
        SIG = make_signals(FACS, NEG, anchor)
        s = run_engine(f'{set_name}·月频·{a_label}', SIG)
        rows.append({'配置': f'{set_name}·月频', '起点': a_label, '年化': round(s['annualized_returns'], 4),
                     '夏普(日)': round(s['sharpe'], 2), '最大回撤': round(s['max_drawdown'], 4),
                     'alpha': round(s['alpha'], 3), 'IR': round(s['information_ratio'], 2)})
        anns.append(s['annualized_returns'])
    rows.append({'配置': f'{set_name}·月频', '起点': '离散度(max−min)', '年化': round(max(anns) - min(anns), 4)})
res = pd.DataFrame(rows)
res.to_csv(f'{OUT}/monthly_stability.csv', index=False, encoding='utf-8-sig')
print('\n=== 月频起点稳定性(vs 季频相位离散: 4F 5.29pp / 7F 5.36pp) ===', flush=True)
print(res.to_string(index=False), flush=True)
