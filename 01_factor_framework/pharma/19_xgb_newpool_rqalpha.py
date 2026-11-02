"""19 XGB 新池(core7+sat5)月频链路 + 与13旧池XGB同窗rqalpha对账

三件事:
  A. 自建: 新池9簇XGB月频walk-forward — 训练=[i-36,i)的composite×fwd1标签(全部已实现), 预测月m打分
     对照 = 透明D1月频 13.10%/+5.91pp (17脚本)
  B. 季频3相位: 直接引用15的已算结果(redo_stageB_ml.csv: XGB 14.2/12.4/13.4, 中位13.4%), 不重算
  C. rqalpha T+1 (2020-01-23~2026-05-15, 基准沪深300):
     - 新池XGB月频(本脚本A的信号)
     - 旧池XGB季频(13口径: 19簇legacy, window38 embargo, phase0网格) — 同窗重跑, 对比13原结果(16.9%, 窗被截到2025-05)
输出: outputs/redo_stageF_xgb_rq.csv
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

DATA = '/Users/lichanghaodemac/quant_strategy/data'
OUT = '/Users/lichanghaodemac/quant_strategy/projects/pharma_factor_workflow/outputs'
import os; os.makedirs(f'{OUT}/stageF', exist_ok=True)
COST = 0.0015

CLUSTERS_NEW = {  # 12F → 9簇(同15)
    'a040': ['WorldQuant_alpha040'], 'a016': ['WorldQuant_alpha016'], 'a024': ['WorldQuant_alpha024'],
    'a044': ['WorldQuant_alpha044'], 'a011': ['WorldQuant_alpha011'], 'a026': ['WorldQuant_alpha026'],
    'dividend': ['dividend_yield_ttm'],
    'value': ['book_to_market_ratio_lf', 'pb_ratio_lf', 'ps_ratio_ttm'],
    'vol': ['VOLT20', 'VOLT60'],
}
CLUSTERS_LEG = {  # legacy 24因子→19簇(同13)
    'C1': ['ocf_to_debt_ttm'], 'C2': ['surplus_cash_protection_multiples_ttm'],
    'C3': ['VOLT20', 'VOLT60'], 'C4': ['book_to_market_ratio_lf', 'pb_ratio_lf', 'ps_ratio_ttm'],
    'C5': ['dividend_yield_ttm'], 'C6': ['pe_ratio_lyr'], 'C7': ['WorldQuant_alpha024'],
    'C8': ['WorldQuant_alpha037'], 'C9': ['WorldQuant_alpha006', 'WorldQuant_alpha014'],
    'C10': ['WorldQuant_alpha040'], 'C11': ['WorldQuant_alpha003'], 'C12': ['WorldQuant_alpha027'],
    'C13': ['WorldQuant_alpha050'], 'C14': ['WorldQuant_alpha013', 'WorldQuant_alpha016'],
    'C15': ['WorldQuant_alpha044'], 'C16': ['WorldQuant_alpha015'], 'C17': ['WorldQuant_alpha055'],
    'C18': ['WorldQuant_alpha088'], 'C19': ['WorldQuant_alpha011'],
}
ALLF = sorted({f for v in list(CLUSTERS_NEW.values()) + list(CLUSTERS_LEG.values()) for f in v})

# ===== 面板 =====
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

_comp = {}   # (池标签, pm) → df; 2026-08-16审查修正: 弃用id()键(全局对象虽不回收, 但写法脆弱)
def composite(pm, clusters):
    key = ('new' if clusters is CLUSTERS_NEW else 'leg', pm)
    if key in _comp: return _comp[key]
    i = IDX[pm]; past = me_sorted[max(0, i - 36):i]
    comp = {}
    for cname, facs in clusters.items():
        vals = []
        for f in facs:
            v = IC1[f].reindex(past).dropna(); sgn = np.sign(v.mean()) if len(v) > 6 else 1.0
            vals.append(sgn * Z[f].loc[pm])
        comp[cname] = pd.concat(vals, axis=1).mean(axis=1)
    df = pd.DataFrame(comp).reindex(stks); _comp[key] = df; return df

XGB_P = dict(n_estimators=60, max_depth=2, learning_rate=0.05, subsample=0.7,
             colsample_bytree=0.7, reg_lambda=5, reg_alpha=2, n_jobs=2)

def fit_pred(pms, m, clusters, fwd):
    Xtr, ytr = [], []
    for pm in pms:
        if pm not in fwd.index: continue
        cp = composite(pm, clusters)
        for s in stks:
            xv = cp.loc[s].values; yv = fwd.loc[pm, s] if s in fwd.columns else np.nan
            if not np.isnan(xv).any() and pd.notna(yv): Xtr.append(xv); ytr.append(yv)
    if len(Xtr) < 300: return None
    Xtr, ytr = np.array(Xtr), np.array(ytr)
    xs = StandardScaler().fit(Xtr)
    mdl = XGBRegressor(**XGB_P).fit(xs.transform(Xtr), ytr)
    return pd.Series(mdl.predict(xs.transform(composite(m, clusters).loc[stks].fillna(0).values)), index=stks)

def q5_of(sc):
    sc = sc.dropna()
    if sc.nunique() < 5: return None
    try: q = pd.qcut(sc.rank(method='first'), 5, labels=False)
    except Exception: return None
    q5 = sc[q == 4]
    return q5 if len(q5) >= 3 else None

# ===== A. 新池XGB月频(训练=[i-36,i)×fwd1标签, 全部已实现) =====
print('A. 新池XGB月频 walk-forward...', flush=True)
sig_new, rets_new = {}, []
for m in me_sorted[36:]:
    if m not in fwd1.index: continue
    i = IDX[m]; pms = me_sorted[max(0, i - 36):i]
    sc = fit_pred(pms, m, CLUSTERS_NEW, fwd1)
    q5 = None if sc is None else q5_of(sc)
    if q5 is None: continue
    vals = [fwd1.loc[m, o] for o in q5.index if o in fwd1.columns and pd.notna(fwd1.loc[m, o])]
    if len(vals) >= 2:
        sig_new[m.strftime('%Y-%m-%d')] = list(q5.index); rets_new.append(np.mean(vals) - COST)
r_new = pd.Series(rets_new)
print(f'   {len(r_new)}期 {me_sorted[36].date()}~ : 年化{r_new.mean()*12:.2%} 夏普{r_new.mean()/r_new.std()*np.sqrt(12):.2f}', flush=True)

# ===== C1. 旧池XGB季频信号(13口径: win38×fwd3, phase0) =====
print('C1. 旧池XGB季频信号(window38 embargo)...', flush=True)
sig_leg = {}
for m in [m for i, m in enumerate(me_sorted) if i >= 36 and (i - 36) % 3 == 0]:
    if m not in fwd3.index: continue
    i = IDX[m]; pms = me_sorted[max(0, i - 38):i - 2]
    sc = fit_pred(pms, m, CLUSTERS_LEG, fwd3)
    q5 = None if sc is None else q5_of(sc)
    if q5 is not None: sig_leg[m.strftime('%Y-%m-%d')] = list(q5.index)
print(f'   {len(sig_leg)}个季度信号', flush=True)

# ===== 基准(2020-01~2026-05 月频日历窗, 同16) =====
ids = univ[univ['industry_l1'] == '医药生物'].groupby('monthend')['order_book_id'].apply(set).to_dict()
ph1 = pd.Series({m: fwd1.loc[m, [o for o in ss if o in fwd1.columns]].dropna().mean() for m, ss in ids.items() if m in fwd1.index and len([o for o in ss if o in fwd1.columns]) >= 10}).sort_index()
W1 = pd.Timestamp('2020-01-23'); W2 = pd.Timestamp('2026-05-15')
ph_win = ph1[(ph1.index >= W1) & (ph1.index <= W2)].mean() * 12
mk_win = fwd1.mean(axis=1); mk_win = mk_win[(mk_win.index >= W1) & (mk_win.index <= W2)].mean() * 12

# ===== C2. rqalpha T+1 两段 =====
from rqalpha import run_func
from rqalpha.apis import *

def run_rq(signals, tag):
    pool = sorted({o for v in signals.values() for o in v})
    def init(context): context.signals = signals; context.pending = None
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
              "mod": {"sys_analyser": {"enabled": True, "plot": False, "report_save_path": f'{OUT}/stageF/{tag}'},
                      "sys_progress": {"enabled": False}, "sys_simulation": {"enabled": True}}}
    print(f'启动 rqalpha ({tag})...', flush=True)
    res = run_func(config=config, init=init, handle_bar=handle_bar)
    return res['sys_analyser']['summary'] if isinstance(res, dict) and 'sys_analyser' in res else None

rows = []
s_new = run_rq(sig_new, 'xgb_new_12m_monthly')
s_leg = run_rq(sig_leg, 'xgb_legacy_q')
for tag, s in [('新池XGB/月频', s_new), ('旧池XGB/季频(13口径)', s_leg)]:
    if s:
        rows.append({'信号': tag, **{k: s.get(k) for k in
              ['annualized_returns', 'sharpe', 'max_drawdown', 'alpha', 'beta', 'information_ratio',
               'win_rate', 'benchmark_annualized_returns']}})
res_df = pd.DataFrame(rows)
res_df.to_csv(f'{OUT}/redo_stageF_xgb_rq.csv', index=False, encoding='utf-8-sig')

print('\n===== 汇总: XGB 月频新链路 vs 13旧池XGB (全部rqalpha T+1, 基准沪深300) =====', flush=True)
print(res_df.to_string(index=False), flush=True)
print(f'\n同窗基准(2020-01~2026-05月频日历窗): 医药等权{ph_win:.2%} / 全A等权{mk_win:.2%}', flush=True)
print('参照: 13原结果 旧池XGB embargo 16.9%/alpha17.1%/IR0.85 (窗2020-01~2025-05, 被截断, 不可直接比)', flush=True)
print('参照: 透明D1月频 rqalpha 13.40%/alpha10.7%/IR0.63 (16脚本, 同窗)', flush=True)
print('参照(自建): 新池XGB季3相位 15脚本 = 14.2/12.4/13.4 (中位13.4%); 新池XGB月频本脚本A', flush=True)
print('存 outputs/redo_stageF_xgb_rq.csv', flush=True)
