"""22 ①补跑12F新池XGB季频rqalpha(对照表的空格) ②基准全家福(用户指定三基准+右偏量化)

① 12F XGB = 15的CLUSTERS_NEW(9簇), win38×fwd3, phase0季频 → rqalpha T+1 2020-01~2026-05-15
② 基准(bundles indexes.h5真指数 + 面板自建, 全部同窗 2020-01-23~2026-05-15 月频):
   货基 1.75%/年(常数) | 沪深300(000300) | 申万医药(000808) | 中证医药(000933)
   医药等权(旧主口径, 保留作参照) | 医药市值加权(market_cap_3, ETF实盘形态) | 医药截面中位数(右偏免疫的"典型股票")
   右偏量化: 等权均值年化 vs 中位数股票年化 的差 = 妖股尾巴贡献
输出: outputs/redo_stageI_bench_xgb12f.csv
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, os, h5py
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

DATA = '/Users/lichanghaodemac/quant_strategy/data'
OUT = '/Users/lichanghaodemac/quant_strategy/projects/pharma_factor_workflow/outputs'
import os as _os; _os.makedirs(f'{OUT}/stageI', exist_ok=True)
COST = 0.0015
W0, W1 = pd.Timestamp('2020-01-23'), pd.Timestamp('2026-05-15')

# ============ ① 12F XGB 季频 ============
CLUSTERS_NEW = {
    'a040': ['WorldQuant_alpha040'], 'a016': ['WorldQuant_alpha016'], 'a024': ['WorldQuant_alpha024'],
    'a044': ['WorldQuant_alpha044'], 'a011': ['WorldQuant_alpha011'], 'a026': ['WorldQuant_alpha026'],
    'dividend': ['dividend_yield_ttm'],
    'value': ['book_to_market_ratio_lf', 'pb_ratio_lf', 'ps_ratio_ttm'],
    'vol': ['VOLT20', 'VOLT60'],
}
ALLF = sorted({f for v in CLUSTERS_NEW.values() for f in v})

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

_comp = {}
def composite(pm):
    if pm in _comp: return _comp[pm]
    i = IDX[pm]; past = me_sorted[max(0, i - 36):i]
    comp = {}
    for cname, facs in CLUSTERS_NEW.items():
        vals = []
        for f in facs:
            v = IC1[f].reindex(past).dropna(); sgn = np.sign(v.mean()) if len(v) > 6 else 1.0
            vals.append(sgn * Z[f].loc[pm])
        comp[cname] = pd.concat(vals, axis=1).mean(axis=1)
    df = pd.DataFrame(comp).reindex(stks); _comp[pm] = df; return df

def xgb_score(m):
    i = IDX[m]; pms = me_sorted[max(0, i - 38):i - 2]
    Xtr, ytr = [], []
    for pm in pms:
        if pm not in fwd3.index: continue
        cp = composite(pm)
        for s in stks:
            xv = cp.loc[s].values; yv = fwd3.loc[pm, s] if s in fwd3.columns else np.nan
            if not np.isnan(xv).any() and pd.notna(yv): Xtr.append(xv); ytr.append(yv)
    if len(Xtr) < 300: return None
    Xtr, ytr = np.array(Xtr), np.array(ytr)
    xs = StandardScaler().fit(Xtr)
    mdl = XGBRegressor(n_estimators=60, max_depth=2, learning_rate=0.05, subsample=0.7,
                       colsample_bytree=0.7, reg_lambda=5, reg_alpha=2, n_jobs=2).fit(xs.transform(Xtr), ytr)
    return pd.Series(mdl.predict(xs.transform(composite(m).loc[stks].fillna(0).values)), index=stks)

SIG = {}
for m in [m for i, m in enumerate(me_sorted) if i >= 36 and (i - 36) % 3 == 0]:
    if m not in fwd3.index: continue
    sc = xgb_score(m)
    if sc is None: continue
    sc = sc.dropna()
    if sc.nunique() < 5: continue
    try: q = pd.qcut(sc.rank(method='first'), 5, labels=False)
    except Exception: continue
    q5 = sc[q == 4].index
    if len(q5) >= 3: SIG[m.strftime('%Y-%m-%d')] = list(q5)
print(f'12F XGB phase0信号: {len(SIG)}个季度点', flush=True)

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
          "mod": {"sys_analyser": {"enabled": True, "plot": False, "report_save_path": f'{OUT}/stageI/xgb12f_q'},
                  "sys_progress": {"enabled": False}, "sys_simulation": {"enabled": True}}}
print('启动 rqalpha (12F XGB 季频)...', flush=True)
res = run_func(config=config, init=init, handle_bar=handle_bar)
s12 = res['sys_analyser']['summary'] if isinstance(res, dict) and 'sys_analyser' in res else {}

# ============ ② 基准全家福 ============
# 指数: bundle indexes.h5 日收盘 → 月末 → 月收益
def index_monthly(code):
    with h5py.File(os.path.expanduser('~/.rqalpha/bundle/indexes.h5'), 'r') as f:
        ds = f[code][:]
    d = pd.DataFrame({'date': pd.to_datetime([str(int(x[0]))[:8] for x in ds], format='%Y%m%d'),
                      'close': [x[1] if len(x) > 1 else x['close'] for x in ds]})
    d = d.set_index('date')['close'].sort_index()
    m = d.resample('ME').last().dropna()
    return m.pct_change().dropna()

idx300 = index_monthly('000300.XSHG'); idx808 = index_monthly('000808.XSHG'); idx933 = index_monthly('000933.XSHG')

# 医药面板基准
ids = univ[univ['industry_l1'] == '医药生物'].groupby('monthend')['order_book_id'].apply(set).to_dict()
ph_rows = []
mc = factors.set_index(['order_book_id', 'monthend'])['market_cap_3'] if 'market_cap_3' in factors.columns else None
for m, ss in sorted(ids.items()):
    if m not in fwd1.index: continue
    v = fwd1.loc[m, [o for o in ss if o in fwd1.columns]].dropna()
    if len(v) < 10: continue
    r = {'monthend': m, 'ew': v.mean(), 'med': v.median()}
    if mc is not None:
        try:
            w = np.array([mc.get((o, m), np.nan) for o in v.index])
            ok = ~np.isnan(w)
            if ok.sum() > 10 and w[ok].sum() > 0:
                r['cap'] = float(np.average(v.values[ok], weights=w[ok]))
            else: r['cap'] = np.nan
        except Exception: r['cap'] = np.nan
    else: r['cap'] = np.nan
    ph_rows.append(r)
phb = pd.DataFrame(ph_rows).set_index('monthend')

def ann(series):
    s = series[(series.index >= W0) & (series.index <= W1)].dropna()
    return s.mean() * 12, len(s)

bmarks = {}
bmarks['货基(1.75%常数)'] = (0.0175, 0)
for nm, s in [('沪深300', idx300), ('申万医药(000808)', idx808), ('中证医药(000933)', idx933),
              ('医药等权', phb['ew']), ('医药市值加权', phb['cap']), ('医药截面中位数', phb['med'])]:
    a, n = ann(s)
    bmarks[nm] = (round(a, 4), n)

print('\n===== ② 基准全家福 (同窗 2020-01-23~2026-05-15, 月频年化) =====', flush=True)
for k, (a, n) in bmarks.items():
    print(f'  {k:<16}: {a:.2%}  (n={n})', flush=True)
gap = bmarks['医药等权'][0] - bmarks['医药截面中位数'][0]
print(f'\n右偏量化: 等权均值 {bmarks["医药等权"][0]:.2%} − 中位数股票 {bmarks["医药截面中位数"][0]:.2%} = {gap:+.2%}/年 ← 妖股尾巴贡献', flush=True)

# 策略对基准超额表
strategies = {'透明D1月频(rqalpha)': 0.1340, '24F XGB季频(rqalpha)': 0.1515, '26F XGB季频(rqalpha)': 0.1087}
if s12: strategies['12F XGB季频(rqalpha,本次)'] = round(s12.get('annualized_returns'), 4)
rows = []
for snm, sr in strategies.items():
    row = {'策略': snm, '年化': f'{sr:.2%}'}
    for bnm, (ba, _) in bmarks.items():
        row[f'vs {bnm}'] = f'{(sr - ba):+.1%}'
    rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(f'{OUT}/redo_stageI_bench_xgb12f.csv', index=False, encoding='utf-8-sig')
print('\n===== 策略 × 基准 超额矩阵 =====', flush=True)
print(out.to_string(index=False), flush=True)
if s12:
    print(f"\n12F XGB rqalpha 明细: 年化{s12.get('annualized_returns'):.2%} alpha{s12.get('alpha'):.3f} "
          f"beta{s12.get('beta'):.2f} IR{s12.get('information_ratio'):.2f} 回撤{s12.get('max_drawdown'):.1%}", flush=True)
print('存 outputs/redo_stageI_bench_xgb12f.csv', flush=True)
