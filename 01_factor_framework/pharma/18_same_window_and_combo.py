"""18 收尾双实验 — ①医药重做同窗对账(2020-12起=银行窗) ②银行7F季+医药D1月 50/50组合

①同窗: 医药D1(core7+sat5, 12m全滚动, 月频)限定 2020-12-01 起 → 自建指标 + rqalpha T+1(2020-12~2026-05-15)
   对手 = 银行 rqalpha T+1 同窗 16.0%/alpha14.6%/IR0.99/beta0.39(REPORT口径)
②组合: 银行7F季(bank common F7/NEG7, roll_icir fwd3, 2020-12网格phase0, 与银行05一致)
   映射成月收益(持仓Q5未来3个月的fwd1月均, 成本0.15%/季摊3个月) + 医药D1月收益,
   50/50等权逐月, 窗口 2020-12 ~ 2026-07。
输出: outputs/redo_stageE_samewindow_combo.csv + combo_equity.png
"""
import warnings; warnings.filterwarnings('ignore')
import sys, importlib.util
import pandas as pd, numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK']; matplotlib.rcParams['axes.unicode_minus'] = False
import common as pc   # 医药 common

DATA = pc.DATA
OUT = '/Users/lichanghaodemac/quant_strategy/projects/pharma_factor_workflow/outputs'
COST = 0.0015
W_START = pd.Timestamp('2020-12-01')

CORE7 = ['WorldQuant_alpha040', 'WorldQuant_alpha016', 'WorldQuant_alpha024', 'WorldQuant_alpha044',
         'WorldQuant_alpha011', 'dividend_yield_ttm', 'WorldQuant_alpha026']
SAT5 = ['book_to_market_ratio_lf', 'pb_ratio_lf', 'ps_ratio_ttm', 'VOLT20', 'VOLT60']

# ===== 医药 D1 月频信号与收益(12m全滚动, 同17) =====
P = pc.load_panel(CORE7 + SAT5)
fwd1 = P['fwd1']; me_sorted = P['me_sorted']; Z = P['Z']; stks = P['stks']
IDX = {m: i for i, m in enumerate(me_sorted)}
IC1 = {f: pd.Series({m: (lambda d: spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0] if len(d) >= 10 and d.iloc[:, 0].nunique() > 1 else np.nan)
                     (pd.concat([Z[f].loc[m], fwd1.loc[m]], axis=1).dropna()) for m in me_sorted if m in fwd1.index}) for f in CORE7 + SAT5}

def d1_monthly_returns():
    rets = {}
    sig_cache = {}
    for m in me_sorted[36:]:
        if m not in fwd1.index: continue
        i = IDX[m]; win = me_sorted[max(0, i - 12):i]
        ws, ss_ = [], []
        for f in CORE7 + SAT5:
            v = IC1[f].reindex(win).dropna()
            if len(v) > 6 and v.std() > 0:
                ws.append(abs(v.mean() / v.std())); ss_.append(1 if v.mean() > 0 else -1)
            else:
                ws.append(0.0); ss_.append(1)
        ws = np.array(ws); ws = ws / ws.sum() if ws.sum() > 0 else np.ones(12) / 12
        z = pd.DataFrame(index=stks)
        for f, s in zip(CORE7 + SAT5, ss_): z[f] = s * Z[f].loc[m].reindex(stks)
        sc = (z.fillna(0) * ws).sum(axis=1).dropna()
        if sc.nunique() < 5: continue
        try: q = pd.qcut(sc.rank(method='first'), 5, labels=False)
        except Exception: continue
        q5 = sc[q == 4].index
        if len(q5) < 3: continue
        vals = [fwd1.loc[m, o] for o in q5 if o in fwd1.columns and pd.notna(fwd1.loc[m, o])]
        if len(vals) >= 2:
            rets[m] = np.mean(vals) - COST
            sig_cache[m.strftime('%Y-%m-%d')] = list(q5)
    return pd.Series(rets).sort_index(), sig_cache

ph_ret, PH_SIG = d1_monthly_returns()
print(f'医药D1月收益: {len(ph_ret)}期 {ph_ret.index[0].date()}~{ph_ret.index[-1].date()}', flush=True)

# 基准
univ = pd.read_parquet(f'{DATA}/universe_monthend_pit.parquet'); univ['monthend'] = pd.DatetimeIndex(univ['monthend'])
ids = univ.groupby('monthend').apply(lambda g: g[g['industry_l1'] == '医药生物']['order_book_id'].tolist()).to_dict()
ph_ew = pd.Series({m: fwd1.loc[m, [o for o in ss if o in fwd1.columns]].dropna().mean() for m, ss in ids.items() if m in fwd1.index and len([o for o in ss if o in fwd1.columns]) >= 10}).sort_index()
mkt1 = P['mkt1']

def met(r, py=12):
    ann = r.mean() * py; sh = r.mean() / r.std() * np.sqrt(py) if r.std() > 0 else 0
    cum = (1 + r).cumprod(); dd = (cum / cum.cummax() - 1).min()
    return ann, sh, dd

# ===== ① 同窗对账(2020-12起) =====
ph_win = ph_ret[ph_ret.index >= W_START]
b_ph = ph_ew[ph_ew.index >= W_START].mean() * 12; b_mk = mkt1[mkt1.index >= W_START].mean() * 12
a, s, d = met(ph_win)
print(f'\n① 医药D1 同窗自建(2020-12~{ph_win.index[-1].date()}): 年化{a:.2%} 夏普{s:.2f} 回撤{d:.1%} | 医药EW {b_ph:.2%} / 全AEW {b_mk:.2%}', flush=True)

# rqalpha 同窗(信号限定>=2020-12)
from rqalpha import run_func
from rqalpha.apis import *
SIG_SAME = {k: v for k, v in PH_SIG.items() if pd.Timestamp(k) >= W_START}
pool = sorted({o for v in SIG_SAME.values() for o in v})
def init(context): context.signals = SIG_SAME; context.pending = None
def handle_bar(context, bar_dict):
    if context.pending is not None:
        target = context.pending; context.pending = None
        for o in list(context.portfolio.positions.keys()):
            if o not in target: order_target_percent(o, 0)
        w = 1.0 / len(target)
        for o in target: order_target_percent(o, w)
    today = context.now.strftime('%Y-%m-%d')
    if today in context.signals: context.pending = context.signals[today]
config = {"base": {"start_date": "2020-12-01", "end_date": "2026-05-15",
                   "accounts": {"stock": 1_000_000}, "benchmark": "000300.XSHG", "stock_pool": pool,
                   "frequency": "1d", "run_type": "b"},
          "extra": {"log_level": "warning", "show": False},
          "mod": {"sys_analyser": {"enabled": True, "plot": False, "report_save_path": f'{OUT}/stageC/pharma_202012'},
                  "sys_progress": {"enabled": False}, "sys_simulation": {"enabled": True}}}
print('启动 rqalpha 医药同窗(2020-12~2026-05-15)...', flush=True)
rq_res = run_func(config=config, init=init, handle_bar=handle_bar)
s_ph = rq_res['sys_analyser']['summary'] if isinstance(rq_res, dict) and 'sys_analyser' in rq_res else {}
print(f"医药rqalpha同窗: 年化{s_ph.get('annualized_returns', float('nan')):.2%} alpha{s_ph.get('alpha', float('nan')):.3f} "
      f"IR{s_ph.get('information_ratio', float('nan')):.2f} beta{s_ph.get('beta', float('nan')):.2f}", flush=True)

# ===== ② 银行7F季 → 月收益映射 + 50/50组合 =====
spec = importlib.util.spec_from_file_location('bank_common', '/Users/lichanghaodemac/quant_strategy/projects/bank_factor_workflow/common.py')
bk = importlib.util.module_from_spec(spec); spec.loader.exec_module(bk)
P_b = bk.load_panel()
qi = next(i for i, m in enumerate(P_b['me_sorted']) if m >= W_START)
quarterly_b = [m for i, m in enumerate(P_b['me_sorted']) if i >= qi and (i - qi) % 3 == 0]
W_b = bk.roll_icir(P_b, bk.F7, P_b['fwd3'])
sc_b = bk.score_set(P_b, bk.F7, bk.NEG7, W_b)
# 内联提取Q5(照银行backtest的thr=None路径: nunique≥5, qcut5, Q5), 兼算季度收益做校验
q5s_b, rq_rets = {}, []
for m in quarterly_b:
    if m not in P_b['fwd3'].index or m not in sc_b: continue
    s = sc_b[m].dropna()
    if s.nunique() < 5: continue
    try: q = pd.qcut(s.rank(method='first'), 5, labels=False)
    except Exception: continue
    q5 = s[q == 4].index
    if len(q5) < 3: continue
    vals = [P_b['fwd3'].loc[m, o] for o in q5 if o in P_b['fwd3'].columns and pd.notna(P_b['fwd3'].loc[m, o])]
    if len(vals) >= 2:
        q5s_b[m] = set(q5); rq_rets.append(np.mean(vals) - bk.COST)
r_q = pd.Series(rq_rets)
print(f'银行7F季: {len(r_q)}期, 年化{r_q.mean()*4:.2%}(银行REPORT口径14.8%)', flush=True)

me_b = P_b['me_sorted']; idx_b = {m: i for i, m in enumerate(me_b)}
bk_month = {}
for m, S in sorted(q5s_b.items()):
    i0 = idx_b[m]
    for k in (1, 2, 3):
        if i0 + k < len(me_b):
            mm = me_b[i0 + k]
            if mm in P_b['fwd1'].index:
                v = P_b['fwd1'].loc[mm, list(S)].dropna()
                if len(v) >= 2: bk_month[mm] = v.mean() - COST / 3   # 0.15%/季摊到3个月
bk_ret = pd.Series(bk_month).sort_index()
print(f'银行月映射: {len(bk_ret)}期 {bk_ret.index[0].date()}~{bk_ret.index[-1].date()}', flush=True)

both = pd.concat({'银行': bk_ret, '医药': ph_ret}, axis=1).dropna()
both = both[(both.abs() < 1).all(axis=1)]
combo = (0.5 * both['银行'] + 0.5 * both['医药'])
corr = both['银行'].corr(both['医药'])
rows = []
for nm, r in [('银行7F季(月映射)', both['银行']), ('医药D1月', both['医药']), ('50/50组合(月再平衡)', combo)]:
    a, s, d = met(r)
    bmk = mkt1[mkt1.index.isin(r.index)].mean() * 12
    rows.append({'配置': nm, '年化': round(a, 4), '夏普': round(s, 2), '最大回撤': round(d, 4),
                 '超额vs全AEW': round(a - bmk, 4), '期数': len(r)})
resE = pd.DataFrame(rows)
resE.to_csv(f'{OUT}/redo_stageE_samewindow_combo.csv', index=False, encoding='utf-8-sig')
print(f'\n② 50/50组合窗口: {both.index[0].date()}~{both.index[-1].date()}, 两腿相关{corr:.2f}', flush=True)
print(resE.to_string(index=False), flush=True)

fig, ax = plt.subplots(figsize=(13, 6))
for nm, r, c in [('银行7F季', both['银行'], '#8e44ad'), ('医药D1', both['医药'], '#2980b9'), ('50/50组合', combo, '#c0392b')]:
    cum = (1 + r).cumprod(); cum.plot(ax=ax, label=f'{nm} (年化{r.mean()*12:.1%})', lw=1.6, color=c)
cum_m = (1 + mkt1[mkt1.index.isin(both.index)]).cumprod()
cum_m.iloc[0] = 1; cum_m.plot(ax=ax, label='全A等权', lw=1.0, color='gray', ls='--')
ax.set_title(f'银行×医药 50/50 组合 (2020-12~{both.index[-1].date()}, 月再平衡, 两腿相关{corr:.2f})', fontsize=13)
ax.set_yscale('log'); ax.legend(fontsize=10); ax.grid(alpha=0.2)
plt.tight_layout(); plt.savefig(f'{OUT}/combo_equity.png', dpi=120, bbox_inches='tight'); plt.close()
print('存 outputs/redo_stageE_samewindow_combo.csv + combo_equity.png', flush=True)
