"""21 XGB 26因子池 — legacy 24 + 清洗后a026 + ev_to_ebitda (用户提议)

簇 = legacy 19簇 + a026单簇 + ev单簇 = 21簇(两者与其余因子相关<0.6, 单簇处理与新池一致)
配方同13/19: 季频 phase0, win38[i-38,i-2]×fwd3标签(embargo), XGB(60/2/0.05/强正则), qcut5 Q5等权
跑两件事:
  A. 自建3相位(相位稳健性——legacy 24F版三相位=13.3/16.8/8.6, 相位差是它的老毛病)
  B. rqalpha T+1 2020-01-23~2026-05-15, 对照: 24F版15.15% / 透明D1 13.40% / 新池XGB中位13.4%
输出: outputs/redo_stageH_xgb26.csv
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

DATA = '/Users/lichanghaodemac/quant_strategy/data'
OUT = '/Users/lichanghaodemac/quant_strategy/projects/pharma_factor_workflow/outputs'
import os; os.makedirs(f'{OUT}/stageH', exist_ok=True)
COST = 0.0015

CLUSTERS = {  # legacy 19簇 + 2个新单簇 = 21簇, 26因子
    'C1': ['ocf_to_debt_ttm'], 'C2': ['surplus_cash_protection_multiples_ttm'],
    'C3': ['VOLT20', 'VOLT60'], 'C4': ['book_to_market_ratio_lf', 'pb_ratio_lf', 'ps_ratio_ttm'],
    'C5': ['dividend_yield_ttm'], 'C6': ['pe_ratio_lyr'], 'C7': ['WorldQuant_alpha024'],
    'C8': ['WorldQuant_alpha037'], 'C9': ['WorldQuant_alpha006', 'WorldQuant_alpha014'],
    'C10': ['WorldQuant_alpha040'], 'C11': ['WorldQuant_alpha003'], 'C12': ['WorldQuant_alpha027'],
    'C13': ['WorldQuant_alpha050'], 'C14': ['WorldQuant_alpha013', 'WorldQuant_alpha016'],
    'C15': ['WorldQuant_alpha044'], 'C16': ['WorldQuant_alpha015'], 'C17': ['WorldQuant_alpha055'],
    'C18': ['WorldQuant_alpha088'], 'C19': ['WorldQuant_alpha011'],
    'a026': ['WorldQuant_alpha026'], 'ev': ['ev_to_ebitda_ttm'],
}
ALLF = sorted({f for v in CLUSTERS.values() for f in v})
assert len(ALLF) == 26, len(ALLF)

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
allf[ALLF] = allf[ALLF].replace([np.inf, -np.inf], np.nan)   # ★ inf清洗(a026必须)
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

XGB_P = dict(n_estimators=60, max_depth=2, learning_rate=0.05, subsample=0.7,
             colsample_bytree=0.7, reg_lambda=5, reg_alpha=2, n_jobs=2)

def xgb_score(m):
    """win38 embargo: 训练月 = [i-38, i-2], fwd3标签决策时点全部已实现。"""
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
    mdl = XGBRegressor(**XGB_P).fit(xs.transform(Xtr), ytr)
    return pd.Series(mdl.predict(xs.transform(composite(m).loc[stks].fillna(0).values)), index=stks)

def q5_of(sc):
    sc = sc.dropna()
    if sc.nunique() < 5: return None
    try: q = pd.qcut(sc.rank(method='first'), 5, labels=False)
    except Exception: return None
    q5 = sc[q == 4]
    return q5 if len(q5) >= 3 else None

# ===== A. 自建3相位 =====
rows = []
for ph in [0, 1, 2]:
    quarterly = [m for i, m in enumerate(me_sorted) if i >= 36 and (i - 36) % 3 == ph]
    rets, months = [], []
    for m in quarterly:
        if m not in fwd3.index: continue
        q5 = q5_of(xgb_score(m))
        if q5 is None: continue
        vals = [fwd3.loc[m, o] for o in q5.index if o in fwd3.columns and pd.notna(fwd3.loc[m, o])]
        if len(vals) >= 2: rets.append(np.mean(vals) - COST); months.append(m)
    r = pd.Series(rets)
    ann = r.mean() * 4; sh = r.mean() / r.std() * 2 if r.std() > 0 else 0
    cum = (1 + r).cumprod(); dd = (cum / cum.cummax() - 1).min()
    ids = univ[univ['industry_l1'] == '医药生物'].groupby('monthend')['order_book_id'].apply(set).to_dict()
    ph3 = pd.Series({m: fwd3.loc[m, [o for o in ss if o in fwd3.columns]].dropna().mean() for m, ss in ids.items() if m in fwd3.index}).dropna()
    bph = ph3.reindex(months).dropna().mean() * 4
    rows.append({'模型': 'XGB-26F', '相位': ph, '年化': round(ann, 4), '夏普': round(sh, 2),
                 '最大回撤': round(dd, 4), '超额vs医药EW': round(ann - bph, 4), '期数': len(r)})
    print(f'phase={ph}: {ann:.2%}/{sh:.2f} ({len(r)}期)', flush=True)
med = {'模型': 'XGB-26F', '相位': '中位',
       '年化': round(float(np.median([x['年化'] for x in rows])), 4), '夏普': round(float(np.median([x['夏普'] for x in rows])), 2),
       '最大回撤': round(float(np.median([x['最大回撤'] for x in rows])), 4),
       '超额vs医药EW': round(float(np.median([x['超额vs医药EW'] for x in rows])), 4), '期数': int(np.median([x['期数'] for x in rows]))}

# ===== B. rqalpha T+1 phase0 =====
SIG = {}
for m in [m for i, m in enumerate(me_sorted) if i >= 36 and (i - 36) % 3 == 0]:
    if m not in fwd3.index: continue
    q5 = q5_of(xgb_score(m))
    if q5 is not None: SIG[m.strftime('%Y-%m-%d')] = list(q5.index)
print(f'phase0信号: {len(SIG)}个季度点', flush=True)

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
          "mod": {"sys_analyser": {"enabled": True, "plot": False, "report_save_path": f'{OUT}/stageH/xgb26_q'},
                  "sys_progress": {"enabled": False}, "sys_simulation": {"enabled": True}}}
print('启动 rqalpha (XGB-26F 季频)...', flush=True)
res = run_func(config=config, init=init, handle_bar=handle_bar)
s = res['sys_analyser']['summary'] if isinstance(res, dict) and 'sys_analyser' in res else {}

res_df = pd.DataFrame(rows + [med])
res_df.to_csv(f'{OUT}/redo_stageH_xgb26.csv', index=False, encoding='utf-8-sig')
print('\n===== XGB-26F (legacy24+a026+ev) 结果 =====', flush=True)
print(res_df.to_string(index=False), flush=True)
if s:
    print(f"rqalpha T+1 phase0 同窗: 年化{s.get('annualized_returns'):.2%} alpha{s.get('alpha'):.3f} "
          f"beta{s.get('beta'):.2f} IR{s.get('information_ratio'):.2f} 回撤{s.get('max_drawdown'):.1%}", flush=True)
print('\n对照: 24F XGB同窗 15.15%/IR0.64 | 透明D1 13.40%/IR0.63 | 新池XGB季中位 13.4% | 24F XGB自建3相位 13.3/16.8/8.6', flush=True)
print('存 outputs/redo_stageH_xgb26.csv', flush=True)
