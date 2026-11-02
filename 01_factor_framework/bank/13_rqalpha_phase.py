"""13 rqalpha 季频相位检验 — 起始月挪一格, 引擎数字稳不稳?

09 在手写回测里做相位检验(4F离散2.12pp最稳); 本脚本把同一检验搬进 rqalpha 引擎:
  phase1 = 2020-12 起(现行口径, 06/12用的就是它)
  phase0 = 2020-11 起(左邻)    phase2 = 2021-01 起(右邻)
对象: 4F·季 / 7F·季 两个 headline 配置 → 共 6 个引擎回测, T+1, vs HS300。
读法: 三相位的年化离散度 = "起始月运气"的量度; 引擎版与手写版(2.12pp/2.77pp)互为印证。
输出: outputs/rqalpha_phase.csv
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
PHASES = {'phase0(2020-11起)': '2020-11-01', 'phase1(2020-12起,现行)': '2020-12-01', 'phase2(2021-01起)': '2021-01-01'}
ALLF = sorted({f for v in SETS.values() for f in v[0]})
DBF = [f for f in ALLF if f not in ('beta60', 'MOM_12M_1M')]

# ===== 面板(同06/12) =====
price = pd.read_parquet(f'{DATA}/mainboard_price.parquet'); price['date'] = pd.to_datetime(price['date'])
close = price.pivot(index='date', columns='order_book_id', values='close').sort_index()
me = pd.DatetimeIndex(pd.read_parquet(f'{DATA}/monthend_dates.parquet')['monthend'])
mev = [m for m in me if m in close.index]; cm = close.loc[mev]
fwd3 = cm.shift(-3) / cm - 1

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

ic3 = {}
for f in ALLF:
    s = pd.Series(index=me_sorted, dtype=float)
    for m in me_sorted:
        if m not in fwd3.index: continue
        df = pd.concat([WIDE[f].loc[m], fwd3.loc[m]], axis=1).dropna()
        if len(df) >= 10 and df.iloc[:, 0].nunique() > 1: s[m] = spearmanr(df.iloc[:, 0], df.iloc[:, 1])[0]
    ic3[f] = s

def make_signals(FACS, NEG, anchor):
    ai = next(i for i, m in enumerate(me_sorted) if m >= pd.Timestamp(anchor))
    W = {}
    for i, m in enumerate(me_sorted):
        past = me_sorted[max(0, i - 36):i]
        w = np.array([abs((lambda v: v.mean() / v.std() if len(v) > 6 and v.std() > 0 else 0)(ic3[f].reindex(past).dropna())) for f in FACS])
        W[m] = w / w.sum() if w.sum() > 0 else np.ones(len(FACS)) / len(FACS)
    rebal = [m for i, m in enumerate(me_sorted) if i >= ai and (i - ai) % 3 == 0]
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
    s = res['sys_analyser']['summary']
    return s

rows = []
for set_name, (FACS, NEG) in SETS.items():
    anns = []
    for ph_label, anchor in PHASES.items():
        SIG = make_signals(FACS, NEG, anchor)
        s = run_engine(f'{set_name}·{ph_label}', SIG)
        rows.append({'配置': set_name, '相位': ph_label, '年化': round(s['annualized_returns'], 4),
                     '夏普(日)': round(s['sharpe'], 2), '最大回撤': round(s['max_drawdown'], 4),
                     'alpha': round(s['alpha'], 3), 'IR': round(s['information_ratio'], 2)})
        anns.append(s['annualized_returns'])
    rows.append({'配置': set_name, '相位': '离散度(max−min)', '年化': round(max(anns) - min(anns), 4)})
res = pd.DataFrame(rows)
res.to_csv(f'{OUT}/rqalpha_phase.csv', index=False, encoding='utf-8-sig')
print('\n=== rqalpha 季频 3 相位检验(T+1, vs HS300) | 手写参照: 4F离散2.12pp / 7F 2.77pp ===', flush=True)
print(res.to_string(index=False), flush=True)
