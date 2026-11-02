"""23 网格/相位统一修正 — ①24F自建3相位补i≥36网格 ②12F XGB rqalpha补phase1/2 ③12F XGB月频3起始偏移

修正23前对照表的两处不严谨: 24F的3相位数字来自12审计(i≥38网格, 2020-03起), 与12F/26F(i≥36, 2020-01起)不可比;
rqalpha列全是phase0单网格。本脚本统一:
  ① 24F(19簇legacy) 自建3相位, i≥36, (i-36)%3==ph, fwd3, win38 embargo → 与15/21同网格
  ② 12F(9簇新池) rqalpha T+1 phase1/phase2 (phase0=14.63%已有, 22脚本)
  ③ 12F XGB月频(fwd1标签) 3个起始偏移 i≥36/37/38 自建 (19单起点=10.33%)
输出: outputs/redo_stageJ_grid_fix.csv
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

DATA = '/Users/lichanghaodemac/quant_strategy/data'
OUT = '/Users/lichanghaodemac/quant_strategy/projects/pharma_factor_workflow/outputs'
import os; os.makedirs(f'{OUT}/stageJ', exist_ok=True)
COST = 0.0015

CL_NEW = {  # 12F 9簇
    'a040': ['WorldQuant_alpha040'], 'a016': ['WorldQuant_alpha016'], 'a024': ['WorldQuant_alpha024'],
    'a044': ['WorldQuant_alpha044'], 'a011': ['WorldQuant_alpha011'], 'a026': ['WorldQuant_alpha026'],
    'dividend': ['dividend_yield_ttm'],
    'value': ['book_to_market_ratio_lf', 'pb_ratio_lf', 'ps_ratio_ttm'],
    'vol': ['VOLT20', 'VOLT60'],
}
CL_LEG = {  # legacy 19簇
    'C1': ['ocf_to_debt_ttm'], 'C2': ['surplus_cash_protection_multiples_ttm'],
    'C3': ['VOLT20', 'VOLT60'], 'C4': ['book_to_market_ratio_lf', 'pb_ratio_lf', 'ps_ratio_ttm'],
    'C5': ['dividend_yield_ttm'], 'C6': ['pe_ratio_lyr'], 'C7': ['WorldQuant_alpha024'],
    'C8': ['WorldQuant_alpha037'], 'C9': ['WorldQuant_alpha006', 'WorldQuant_alpha014'],
    'C10': ['WorldQuant_alpha040'], 'C11': ['WorldQuant_alpha003'], 'C12': ['WorldQuant_alpha027'],
    'C13': ['WorldQuant_alpha050'], 'C14': ['WorldQuant_alpha013', 'WorldQuant_alpha016'],
    'C15': ['WorldQuant_alpha044'], 'C16': ['WorldQuant_alpha015'], 'C17': ['WorldQuant_alpha055'],
    'C18': ['WorldQuant_alpha088'], 'C19': ['WorldQuant_alpha011'],
}
ALLF = sorted({f for v in list(CL_NEW.values()) + list(CL_LEG.values()) for f in v})

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

def composite_factory(clusters, tag):
    cache = {}
    def comp(pm):
        if pm in cache: return cache[pm]
        i = IDX[pm]; past = me_sorted[max(0, i - 36):i]
        c = {}
        for cn, facs in clusters.items():
            vals = []
            for f in facs:
                v = IC1[f].reindex(past).dropna(); sgn = np.sign(v.mean()) if len(v) > 6 else 1.0
                vals.append(sgn * Z[f].loc[pm])
            c[cn] = pd.concat(vals, axis=1).mean(axis=1)
        df = pd.DataFrame(c).reindex(stks); cache[pm] = df; return df
    return comp

comp_new = composite_factory(CL_NEW, 'new'); comp_leg = composite_factory(CL_LEG, 'leg')

def xgb_score(m, comp, fwd, lb=36):
    i = IDX[m]
    pms = me_sorted[max(0, i - lb):i] if fwd is fwd1 else me_sorted[max(0, i - lb - 2):i - 2]
    Xtr, ytr = [], []
    for pm in pms:
        if pm not in fwd.index: continue
        cp = comp(pm)
        for s in stks:
            xv = cp.loc[s].values; yv = fwd.loc[pm, s] if s in fwd.columns else np.nan
            if not np.isnan(xv).any() and pd.notna(yv): Xtr.append(xv); ytr.append(yv)
    if len(Xtr) < 300: return None
    Xtr, ytr = np.array(Xtr), np.array(ytr)
    xs = StandardScaler().fit(Xtr)
    mdl = XGBRegressor(n_estimators=60, max_depth=2, learning_rate=0.05, subsample=0.7,
                       colsample_bytree=0.7, reg_lambda=5, reg_alpha=2, n_jobs=2).fit(xs.transform(Xtr), ytr)
    return pd.Series(mdl.predict(xs.transform(comp(m).loc[stks].fillna(0).values)), index=stks)

def q5_of(sc):
    if sc is None: return None
    sc = sc.dropna()
    if sc.nunique() < 5: return None
    try: q = pd.qcut(sc.rank(method='first'), 5, labels=False)
    except Exception: return None
    q5 = sc[q == 4]
    return q5 if len(q5) >= 3 else None

def backtest_quarterly(comp, ph):
    rets, months = [], []
    for m in [m for i, m in enumerate(me_sorted) if i >= 36 and (i - 36) % 3 == ph]:
        if m not in fwd3.index: continue
        q5 = q5_of(xgb_score(m, comp, fwd3))
        if q5 is None: continue
        vals = [fwd3.loc[m, o] for o in q5.index if o in fwd3.columns and pd.notna(fwd3.loc[m, o])]
        if len(vals) >= 2: rets.append(np.mean(vals) - COST); months.append(m)
    r = pd.Series(rets)
    return r.mean() * 4, (r.mean() / r.std() * 2 if r.std() > 0 else 0), len(r), months

rows = []
# ===== ① 24F 3相位, i>=36 网格(修正与15/21同网格) =====
print('① 24F(legacy) 3相位 @ i>=36 网格...', flush=True)
for ph in [0, 1, 2]:
    a, s, n, _ = backtest_quarterly(comp_leg, ph)
    rows.append({'实验': '①24F自建3相位(i≥36网格)', '相位': ph, '年化': round(a, 4), '夏普': round(s, 2), '期数': n})
    print(f'  phase={ph}: {a:.2%}/{s:.2f} ({n}期)', flush=True)

# ===== ③ 12F XGB 月频 3起始偏移 =====
print('③ 12F XGB 月频(fwd1标签) 3起始偏移...', flush=True)
for st in [36, 37, 38]:
    rets = []
    for m in me_sorted[st:]:
        if m not in fwd1.index: continue
        q5 = q5_of(xgb_score(m, comp_new, fwd1))
        if q5 is None: continue
        vals = [fwd1.loc[m, o] for o in q5.index if o in fwd1.columns and pd.notna(fwd1.loc[m, o])]
        if len(vals) >= 2: rets.append(np.mean(vals) - COST)
    r = pd.Series(rets)
    rows.append({'实验': '③12F XGB月频·起始偏移', '相位': f'起i≥{st}', '年化': round(r.mean() * 12, 4),
                 '夏普': round(r.mean() / r.std() * np.sqrt(12), 2), '期数': len(r)})
    print(f'  起i≥{st}: {r.mean()*12:.2%}/{r.mean()/r.std()*np.sqrt(12):.2f} ({len(r)}期)', flush=True)

# ===== ② 12F XGB rqalpha phase1/2 =====
from rqalpha import run_func
from rqalpha.apis import *

def rq_phase(ph):
    SIG = {}
    for m in [m for i, m in enumerate(me_sorted) if i >= 36 and (i - 36) % 3 == ph]:
        if m not in fwd3.index: continue
        q5 = q5_of(xgb_score(m, comp_new, fwd3))
        if q5 is not None: SIG[m.strftime('%Y-%m-%d')] = list(q5.index)   # ★Series→index(修: list(q5)迭代的是分数)
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
              "mod": {"sys_analyser": {"enabled": True, "plot": False, "report_save_path": f'{OUT}/stageJ/xgb12f_ph{ph}'},
                      "sys_progress": {"enabled": False}, "sys_simulation": {"enabled": True}}}
    print(f'  rqalpha phase={ph} ({len(SIG)}信号)...', flush=True)
    res = run_func(config=config, init=init, handle_bar=handle_bar)
    s = res['sys_analyser']['summary'] if isinstance(res, dict) and 'sys_analyser' in res else {}
    return s

print('② 12F XGB rqalpha phase1/2 (phase0=14.63%已有)...', flush=True)
for ph in [1, 2]:
    s = rq_phase(ph)
    if s:
        rows.append({'实验': '②12F XGB rqalpha T+1', '相位': ph, '年化': round(s.get('annualized_returns'), 4),
                     '夏普': round(s.get('sharpe'), 2), '期数': '-'})
        print(f"  phase={ph}: {s.get('annualized_returns'):.2%} sharpe{s.get('sharpe'):.2f} IR{s.get('information_ratio'):.2f} 回撤{s.get('max_drawdown'):.1%}", flush=True)

res_df = pd.DataFrame(rows)
res_df.to_csv(f'{OUT}/redo_stageJ_grid_fix.csv', index=False, encoding='utf-8-sig')
print('\n===== 全部结果 =====', flush=True)
print(res_df.to_string(index=False), flush=True)
print('\n对照: 12F自建3相位(15)=14.2/12.4/13.4 | 24F旧网格3相位(12审计,i≥38)=13.3/16.8/8.6 | 12F月频单起点(19)=10.33%/rqalpha7.62%', flush=True)
print('存 outputs/redo_stageJ_grid_fix.csv', flush=True)
