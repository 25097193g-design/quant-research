"""13 rqalpha XGB embargo 复验 — 12 发现 07/09/11 的 walk-forward 训练标签与测试窗重叠(pm=m-1/m-2 的
fwd3 标签延伸到 m+2), 11 的 18.8% 继承了该泄漏信号。本脚本用干净窗口(window38: 训练月
[i-38, i-2], 36个月且全部已实现)重新生成 XGB19 季度Q5信号 → 喂 rqalpha 同配置回测。
输出: outputs/rqalpha_xgb_embargo.csv (独立文件, 不动 rqalpha_compare.csv)
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

DATA = '/Users/lichanghaodemac/quant_strategy/data'
OUT = '/Users/lichanghaodemac/quant_strategy/projects/pharma_factor_workflow/outputs'

CLUSTERS = {  # 同07/09/11(24因子→19簇)
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

# ===== 数据(与11一致) =====
price = pd.read_parquet(f'{DATA}/mainboard_price.parquet'); price['date'] = pd.to_datetime(price['date'])
close = price.pivot(index='date', columns='order_book_id', values='close').sort_index()
me = pd.DatetimeIndex(pd.read_parquet(f'{DATA}/monthend_dates.parquet')['monthend'])
mev = [m for m in me if m in close.index]; cm = close.loc[mev]
fwd1 = cm.pct_change(fill_method=None).shift(-1); fwd3 = cm.shift(-3) / cm - 1
factors = pd.read_parquet(f'{DATA}/factors_monthend.parquet'); factors['monthend'] = pd.DatetimeIndex(factors['monthend'])
univ = pd.read_parquet(f'{DATA}/universe_monthend_pit.parquet'); univ['monthend'] = pd.DatetimeIndex(univ['monthend'])
allf = factors[['order_book_id', 'monthend'] + ALLF]
allf = allf.merge(univ[univ['industry_l1'] == '医药生物'][['order_book_id', 'monthend']], on=['order_book_id', 'monthend'], how='inner')
allf = allf[['order_book_id', 'monthend'] + ALLF].sort_values(['monthend', 'order_book_id']).reset_index(drop=True)
allf[ALLF] = allf[ALLF].replace([np.inf, -np.inf], np.nan)
me_sorted = sorted(allf['monthend'].unique())
WIDE = {f: allf.pivot(index='monthend', columns='order_book_id', values=f) for f in ALLF}
stks = np.array(WIDE[ALLF[0]].columns)
IDX = {m: i for i, m in enumerate(me_sorted)}

def zs(mat):
    lo, hi = mat.quantile(0.01, axis=1), mat.quantile(0.99, axis=1); o = mat.copy()
    for c in mat.columns: o[c] = o[c].clip(lo, hi, axis=0)
    return o.sub(o.mean(axis=1), axis=0).div(o.std(axis=1).replace(0, np.nan), axis=0)
Z = {f: zs(WIDE[f]) for f in ALLF}
IC1 = {f: pd.Series({m: (lambda d: spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0] if len(d) >= 10 and d.iloc[:, 0].nunique() > 1 else np.nan)
                     (pd.concat([Z[f].loc[m], fwd1.loc[m]], axis=1).dropna()) for m in me_sorted if m in fwd1.index}) for f in ALLF}

_comp_cache = {}
def composite(pm):
    if pm in _comp_cache: return _comp_cache[pm]
    i = IDX[pm]; past_pm = me_sorted[max(0, i - 36):i]
    comp = {}
    for cname, facs in CLUSTERS.items():
        vals = []
        for f in facs:
            v = IC1[f].reindex(past_pm).dropna(); sgn = np.sign(v.mean()) if len(v) > 6 else 1.0
            vals.append(sgn * Z[f].loc[pm])
        comp[cname] = pd.concat(vals, axis=1).mean(axis=1)
    df = pd.DataFrame(comp).reindex(stks); _comp_cache[pm] = df; return df

def xgb_q5_clean(m):
    """window38: 训练月 = me_sorted[i-38 : i-2] → 36个月, 最后标签在 m-3 结束(决策时点已全部实现)。
    网格与11完全一致(i>=36, (i-36)%3==0) — 首点只有34个训练月(全部已实现), 不与leaky差窗口。"""
    i = IDX[m]; pms = me_sorted[max(0, i - 38):i - 2]
    comp = composite(m); Xtr, ytr = [], []
    for pm in pms:
        if pm not in fwd3.index: continue
        cmp_pm = composite(pm)
        for s in stks:
            xv = cmp_pm.loc[s].values; yv = fwd3.loc[pm, s] if s in fwd3.columns else np.nan
            if not np.isnan(xv).any() and pd.notna(yv): Xtr.append(xv); ytr.append(yv)
    if len(Xtr) < 300: return None
    Xtr, ytr = np.array(Xtr), np.array(ytr)
    xs = StandardScaler().fit(Xtr)
    mdl = XGBRegressor(n_estimators=60, max_depth=2, learning_rate=0.05, subsample=0.7,
                       colsample_bytree=0.7, reg_lambda=5, reg_alpha=2, n_jobs=2).fit(xs.transform(Xtr), ytr)
    return pd.Series(mdl.predict(xs.transform(comp.loc[stks].fillna(0).values)), index=stks)

# ===== 信号(phase0, 与06/11同网格 i>=36 (i-36)%3==0) =====
quarterly = [m for i, m in enumerate(me_sorted) if i >= 36 and (i - 36) % 3 == 0]
SIGNALS = {}
print(f'生成 XGB(window38, 无泄漏) 季度信号, 共{len(quarterly)}个季度点...', flush=True)
for m in quarterly:
    if m not in fwd3.index: continue
    sc = xgb_q5_clean(m)
    if sc is None or sc.dropna().nunique() < 5: continue
    sc = sc.dropna()
    try: q = pd.qcut(sc.rank(method='first'), 5, labels=False)
    except: continue
    q5 = list(sc[q == 4].index)
    if len(q5) >= 3: SIGNALS[m.strftime('%Y-%m-%d')] = q5
print(f'XGB(embargo)信号: {len(SIGNALS)}个季度点, 平均持仓{np.mean([len(v) for v in SIGNALS.values()]):.1f}只, 区间{min(SIGNALS)}~{max(SIGNALS)}', flush=True)

# ===== rqalpha (与11同配置) =====
from rqalpha import run_func
from rqalpha.apis import *

pool = sorted({o for v in SIGNALS.values() for o in v})
def init(context): context.signals = SIGNALS
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
    "base": {"start_date": min(SIGNALS), "end_date": "2025-05-31",
             "accounts": {"stock": 1_000_000}, "benchmark": "000300.XSHG", "stock_pool": pool,
             "frequency": "1d", "run_type": "b"},
    "extra": {"log_level": "warning", "show": False},
    "mod": {"sys_analyser": {"enabled": True, "plot": False, "report_save_path": OUT},
            "sys_progress": {"enabled": False}, "sys_simulation": {"enabled": True}},
}
res = run_func(config=config, init=init, handle_bar=handle_bar)
if isinstance(res, dict) and 'sys_analyser' in res and 'summary' in res['sys_analyser']:
    s = res['sys_analyser']['summary']
    keys = ['annualized_returns', 'sharpe', 'volatility', 'max_drawdown', 'alpha', 'beta', 'information_ratio', 'win_rate']
    pd.DataFrame([{'信号': 'XGB19_embargo(window38)', **{k: s.get(k) for k in keys}}]).to_csv(f'{OUT}/rqalpha_xgb_embargo.csv', index=False, encoding='utf-8-sig')
    print('\n=== rqalpha: XGB19 原版(leaky) vs embargo(window38) ===', flush=True)
    print(pd.read_csv(f'{OUT}/rqalpha_compare.csv').to_string(index=False), flush=True)
    print(pd.read_csv(f'{OUT}/rqalpha_xgb_embargo.csv').to_string(index=False), flush=True)
    print('存 outputs/rqalpha_xgb_embargo.csv', flush=True)
else:
    print('!! rqalpha 未返回 sys_analyser', flush=True)
