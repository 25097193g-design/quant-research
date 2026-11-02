"""11 rqalpha 真实回测 — XGBoost(19簇合成因子, 0.03阈值) 季度Q5 喂米筐引擎

07/09 结论: 医药N=228时 XGBoost 真加分(19.2%/0.60, 3相位全赢), 但只在自建 walk-forward 里。
06 已把 24F/4F 喂 rqalpha 复现(24F季 11.9% / alpha 12.1%)。
本脚本: 把 0.03阈值的19簇 XGBoost 季度Q5信号喂 rqalpha, 和 24F/4F 并排 —
        验证"XGBoost加分"在真实撮合(滑点/涨跌停/成本)下是否复现。

信号生成与 09 完全一致(过去36月训练→预测下季, walk-forward, phase0对齐与06相同):
  - 19簇合成因子(簇内按滚动IC符号对齐后平均) → XGBoost(浅树+强正则)打分
  - 全截面 qcut5 → 前20% ~46只等权, 季度调仓
信号与执行分离: 信号=本脚本算的Q5, 执行=rqalpha(滑点/涨跌停/停牌/成本都还原)。
⚠️ rqalpha bundle(stocks.h5) 到 2025-05, 前置分析到 2026 → 回测区间受限, 与06一致。
输出: 追加一行到 outputs/rqalpha_compare.csv (与24F/4F同表)
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

DATA = '/Users/lichanghaodemac/quant_strategy/data'
OUT = '/Users/lichanghaodemac/quant_strategy/projects/pharma_factor_workflow/outputs'

# 0.03阈值 19簇(同07/09)
CLUSTERS = {
    'C1_cash_debt': ['ocf_to_debt_ttm'], 'C2_surplus_cash': ['surplus_cash_protection_multiples_ttm'],
    'C3_volatility': ['VOLT20', 'VOLT60'], 'C4_value': ['book_to_market_ratio_lf', 'pb_ratio_lf', 'ps_ratio_ttm'],
    'C5_dividend': ['dividend_yield_ttm'], 'C6_pe': ['pe_ratio_lyr'], 'C7_a024': ['WorldQuant_alpha024'],
    'C8_a037': ['WorldQuant_alpha037'], 'C9_a006_014': ['WorldQuant_alpha006', 'WorldQuant_alpha014'],
    'C10_a040': ['WorldQuant_alpha040'], 'C11_a003': ['WorldQuant_alpha003'], 'C12_a027': ['WorldQuant_alpha027'],
    'C13_a050': ['WorldQuant_alpha050'], 'C14_a013_016': ['WorldQuant_alpha013', 'WorldQuant_alpha016'],
    'C15_a044': ['WorldQuant_alpha044'], 'C16_a015': ['WorldQuant_alpha015'], 'C17_a055': ['WorldQuant_alpha055'],
    'C18_a088': ['WorldQuant_alpha088'], 'C19_a011': ['WorldQuant_alpha011'],
}
ALLF = sorted({f for v in CLUSTERS.values() for f in v})

# ===== 数据加载(与09一致) =====
price = pd.read_parquet(f'{DATA}/mainboard_price.parquet'); price['date'] = pd.to_datetime(price['date'])
close = price.pivot(index='date', columns='order_book_id', values='close').sort_index()
me = pd.DatetimeIndex(pd.read_parquet(f'{DATA}/monthend_dates.parquet')['monthend'])
mev = [m for m in me if m in close.index]; cm = close.loc[mev]
fwd1 = cm.pct_change(fill_method=None).shift(-1); fwd3 = cm.shift(-3) / cm - 1
factors = pd.read_parquet(f'{DATA}/factors_monthend.parquet'); factors['monthend'] = pd.DatetimeIndex(factors['monthend'])
univ = pd.read_parquet(f'{DATA}/universe_monthend_pit.parquet'); univ['monthend'] = pd.DatetimeIndex(univ['monthend'])
fromdb = [f for f in ALLF if f in factors.columns]
allf = factors[['order_book_id', 'monthend'] + fromdb].copy()
allf = allf.merge(univ[univ['industry_l1'] == '医药生物'][['order_book_id', 'monthend']], on=['order_book_id', 'monthend'], how='inner')
allf = allf[['order_book_id', 'monthend'] + ALLF].sort_values(['monthend', 'order_book_id']).reset_index(drop=True)
allf[ALLF] = allf[ALLF].replace([np.inf, -np.inf], np.nan)   # 清inf(WQ alpha通病)
me_sorted = sorted(allf['monthend'].unique())
WIDE = {f: allf.pivot(index='monthend', columns='order_book_id', values=f) for f in ALLF}
stks = np.array(WIDE[ALLF[0]].columns)

def zs(mat):
    lo, hi = mat.quantile(0.01, axis=1), mat.quantile(0.99, axis=1); o = mat.copy()
    for c in mat.columns: o[c] = o[c].clip(lo, hi, axis=0)
    return o.sub(o.mean(axis=1), axis=0).div(o.std(axis=1).replace(0, np.nan), axis=0)
Z = {f: zs(WIDE[f]) for f in ALLF}
IC1 = {f: pd.Series({m: (lambda d: spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0] if len(d) >= 10 and d.iloc[:, 0].nunique() > 1 else np.nan)
                     (pd.concat([Z[f].loc[m], fwd1.loc[m]], axis=1).dropna()) for m in me_sorted if m in fwd1.index}) for f in ALLF}

def cluster_composite(m, past):
    comp = {}
    for cname, facs in CLUSTERS.items():
        vals = []
        for f in facs:
            v = IC1[f].reindex(past).dropna(); sgn = np.sign(v.mean()) if len(v) > 6 else 1.0
            vals.append(sgn * Z[f].loc[m])
        comp[cname] = pd.concat(vals, axis=1).mean(axis=1)
    return pd.DataFrame(comp).reindex(stks)

def xgb_q5(m, past):
    comp = cluster_composite(m, past); Xtr, ytr = [], []
    for pm in past:
        if pm not in fwd3.index: continue
        cmp_pm = cluster_composite(pm, [mm for mm in me_sorted if mm < pm][-36:])
        for s in stks:
            xv = cmp_pm.loc[s].values; yv = fwd3.loc[pm, s] if s in fwd3.columns else np.nan
            if not np.isnan(xv).any() and pd.notna(yv): Xtr.append(xv); ytr.append(yv)
    Xtr, ytr = np.array(Xtr), np.array(ytr)
    if len(Xtr) < 300: return None
    xs = StandardScaler().fit(Xtr)
    mdl = XGBRegressor(n_estimators=60, max_depth=2, learning_rate=0.05, subsample=0.7,
                       colsample_bytree=0.7, reg_lambda=5, reg_alpha=2, n_jobs=2).fit(xs.transform(Xtr), ytr)
    return pd.Series(mdl.predict(xs.transform(comp.loc[stks].fillna(0).values)), index=stks)

# ===== 1. 生成 XGBoost 季度Q5信号 (phase0对齐, 同06) =====
quarterly = [m for i, m in enumerate(me_sorted) if i >= 36 and (i - 36) % 3 == 0]
SIGNALS = {}
print(f'生成 XGBoost(19簇) 季度信号, 共{len(quarterly)}个季度点...', flush=True)
for m in quarterly:
    past = [mm for mm in me_sorted if mm < m][-36:]
    if len(past) < 36 or m not in fwd3.index: continue
    sc = xgb_q5(m, past)
    if sc is None or sc.dropna().nunique() < 5: continue
    sc = sc.dropna()
    try: q = pd.qcut(sc.rank(method='first'), 5, labels=False)
    except: continue
    q5 = list(sc[q == 4].index)
    if len(q5) >= 3: SIGNALS[m.strftime('%Y-%m-%d')] = q5
print(f'XGB信号: {len(SIGNALS)}个季度点, 平均持仓{np.mean([len(v) for v in SIGNALS.values()]):.1f}只, 区间{min(SIGNALS)}~{max(SIGNALS)}', flush=True)

# ===== 2. rqalpha 真实撮合 (与06同配置) =====
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
        print(f'=== rqalpha 标准绩效报告 ({tag} 医药Q5, 等权) ===', flush=True)
        for k in ['start_date', 'end_date', 'total_returns', 'annualized_returns', 'benchmark_total_returns',
                  'benchmark_annualized_returns', 'sharpe', 'volatility', 'max_drawdown', 'alpha', 'beta',
                  'sortino', 'information_ratio', 'win_rate', 'daily_win_rate', 'trading_days', 'transaction_cost']:
            if k in s: print(f'  {k}: {s[k]}', flush=True)
        return s
    print('!! rqalpha 未返回 sys_analyser, 键:', list(res.keys()) if isinstance(res, dict) else type(res), flush=True)
    return None

s_xgb = run_rqalpha(SIGNALS, 'XGB19')

# ===== 3. 与 24F/4F 并排 (追加一行到 rqalpha_compare.csv) =====
old = pd.read_csv(f'{OUT}/rqalpha_compare.csv')
if s_xgb:
    row = {'信号': 'XGB19(19簇)', **{k: s_xgb.get(k) for k in
          ['annualized_returns', 'sharpe', 'volatility', 'max_drawdown', 'alpha', 'beta', 'information_ratio', 'win_rate']}}
    combined = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    combined.to_csv(f'{OUT}/rqalpha_compare.csv', index=False, encoding='utf-8-sig')
print('\n=== rqalpha 真引擎对比 (24F/4F vs XGBoost19簇) ===', flush=True)
print(pd.read_csv(f'{OUT}/rqalpha_compare.csv').to_string(index=False), flush=True)
print('\n存 rqalpha_compare.csv (含XGB行)', flush=True)
