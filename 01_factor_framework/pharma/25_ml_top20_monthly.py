"""25 终格实验 — 12F XGB 月频 × Top20(用户要求的最后一块空格)

已有: XGB月频×Q5 = 自建10.33%/rqalpha7.62%(19); 透明D1×Top20 = 自建15.77%/rqalpha15.69%(24)。
本脚本补: 12F XGB月频(fwd1标签, 训练[i-36,i), 每月重训) × Top20等权。
输出: outputs/redo_stageL_xgb_top20.csv
判卷: 若Top20能救月频XGB到≥透明D1-Top20(15.77%)才翻案; 预期不能(Q5版10.33%差距太大)。
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

DATA = '/Users/lichanghaodemac/quant_strategy/data'
OUT = '/Users/lichanghaodemac/quant_strategy/projects/pharma_factor_workflow/outputs'
import os; os.makedirs(f'{OUT}/stageL', exist_ok=True)
COST = 0.0015

CLUSTERS = {
    'a040': ['WorldQuant_alpha040'], 'a016': ['WorldQuant_alpha016'], 'a024': ['WorldQuant_alpha024'],
    'a044': ['WorldQuant_alpha044'], 'a011': ['WorldQuant_alpha011'], 'a026': ['WorldQuant_alpha026'],
    'dividend': ['dividend_yield_ttm'],
    'value': ['book_to_market_ratio_lf', 'pb_ratio_lf', 'ps_ratio_ttm'],
    'vol': ['VOLT20', 'VOLT60'],
}
ALLF = sorted({f for v in CLUSTERS.values() for f in v})

price = pd.read_parquet(f'{DATA}/mainboard_price.parquet'); price['date'] = pd.to_datetime(price['date'])
close = price.pivot(index='date', columns='order_book_id', values='close').sort_index()
me = pd.DatetimeIndex(pd.read_parquet(f'{DATA}/monthend_dates.parquet')['monthend'])
mev = [m for m in me if m in close.index]; cm = close.loc[mev]
fwd1 = cm.pct_change(fill_method=None).shift(-1)
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

_comp = {}
def composite(pm):
    if pm in _comp: return _comp[pm]
    i = IDX[pm]; past = me_sorted[max(0, i - 36):i]
    comp = {}
    for cname, facs in CLUSTERS.items():
        vals = []
        for f in facs:
            v = IC1[f].reindex(past).dropna(); sgn = np.sign(v.mean()) if len(v) > 6 else 1.0
            vals.append(sgn * Z[f].loc[pm])
        comp[cname] = pd.concat(vals, axis=1).mean(axis=1)
    df = pd.DataFrame(comp).reindex(stks); _comp[pm] = df; return df

# ===== 自建: 每月重训(fwd1标签, [i-36,i)), Top20 =====
SIG = {}; rets = []
for m in me_sorted[36:]:
    if m not in fwd1.index: continue
    i = IDX[m]; pms = me_sorted[max(0, i - 36):i]
    Xtr, ytr = [], []
    for pm in pms:
        if pm not in fwd1.index: continue
        cp = composite(pm)
        for s in stks:
            xv = cp.loc[s].values; yv = fwd1.loc[pm, s] if s in fwd1.columns else np.nan
            if not np.isnan(xv).any() and pd.notna(yv): Xtr.append(xv); ytr.append(yv)
    if len(Xtr) < 300: continue
    Xtr, ytr = np.array(Xtr), np.array(ytr)
    xs = StandardScaler().fit(Xtr)
    mdl = XGBRegressor(n_estimators=60, max_depth=2, learning_rate=0.05, subsample=0.7,
                       colsample_bytree=0.7, reg_lambda=5, reg_alpha=2, n_jobs=2).fit(xs.transform(Xtr), ytr)
    sc = pd.Series(mdl.predict(xs.transform(composite(m).loc[stks].fillna(0).values)), index=stks).dropna()
    if sc.nunique() < 5: continue
    lst = list(sc.nlargest(20).index)
    vals = [fwd1.loc[m, o] for o in lst if o in fwd1.columns and pd.notna(fwd1.loc[m, o])]
    if len(vals) >= 3:
        SIG[m.strftime('%Y-%m-%d')] = lst
        rets.append(np.mean(vals) - COST)

r = pd.Series(rets)
print(f'自建 XGB-月频-Top20: {len(r)}期, 年化{r.mean()*12:.2%} 夏普{r.mean()/r.std()*np.sqrt(12):.2f}', flush=True)

# ===== rqalpha T+1 =====
from rqalpha import run_func
from rqalpha.apis import *
pool = sorted({o for v in SIG.values() for o in v})
def init(context): context.signals = SIG; context.pending = None
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
          "mod": {"sys_analyser": {"enabled": True, "plot": False, "report_save_path": f'{OUT}/stageL/xgb_m_top20'},
                  "sys_progress": {"enabled": False}, "sys_simulation": {"enabled": True}}}
print('启动 rqalpha (XGB月频Top20)...', flush=True)
res = run_func(config=config, init=init, handle_bar=handle_bar)
s = res['sys_analyser']['summary'] if isinstance(res, dict) and 'sys_analyser' in res else {}

rows = [{'配置': 'XGB月频×Top20 自建', '年化': round(r.mean() * 12, 4), '夏普': round(r.mean() / r.std() * np.sqrt(12), 2)}]
if s:
    rows.append({'配置': 'XGB月频×Top20 rqalpha', '年化': round(s.get('annualized_returns'), 4), '夏普': round(s.get('sharpe'), 2)})
res_df = pd.DataFrame(rows)
res_df.to_csv(f'{OUT}/redo_stageL_xgb_top20.csv', index=False, encoding='utf-8-sig')
print('\n===== 终格: 12F XGB 月频 × Top20 =====', flush=True)
print(res_df.to_string(index=False), flush=True)
if s:
    print(f"rqalpha明细: alpha{s.get('alpha'):.3f} beta{s.get('beta'):.2f} IR{s.get('information_ratio'):.2f} 回撤{s.get('max_drawdown'):.1%}", flush=True)
print('\n对照: 透明D1×Top20 = 自建15.77%/rqalpha15.69% | XGB月频×Q5 = 自建10.33%/rqalpha7.62%', flush=True)
print('存 outputs/redo_stageL_xgb_top20.csv', flush=True)
