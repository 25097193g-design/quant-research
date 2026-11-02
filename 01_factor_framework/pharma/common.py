"""common.py — 医药板块因子工作流共享模块(从 bank_factor_workflow/common.py fork, 板块参数化)

与银行版同一框架, 只改3处板块相关:
  SECTOR = '医药生物'        (银行是 '银行')
  IC column = '医药IC'/'医药ICIR'  (银行是 '银行IC')
  IC file = ic_pharma_horizon.csv (银行是 ic_bank_horizon.csv)

⚠️ 医药截面 ~207 只(银行 41), 是银行的 5 倍 → ML 可能有效(银行 41 只时不行)
⚠️ inf 清洗沿用(alpha026 等 WQ 因子同样有 inf 问题, 见 [[factor-backtest-check-inf-nan]])
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

DATA = '/Users/lichanghaodemac/quant_strategy/data'
COST = 0.0015
SECTOR = '医药生物'
IC_PREFIX = '医药'          # IC列名前缀: 医药IC / 医药ICIR
IC_FILE = 'ic_pharma_horizon.csv'

# 医药簇结构(02脚本算出: 24因子→19簇) — 大部分WQ alpha互相独立, 不像银行塌成5簇
CLUSTERS = {'C1_cash_debt': ['ocf_to_debt_ttm'],
            'C2_surplus_cash': ['surplus_cash_protection_multiples_ttm'],
            'C3_volatility': ['VOLT20', 'VOLT60'],
            'C4_value': ['book_to_market_ratio_lf', 'pb_ratio_lf', 'ps_ratio_ttm'],
            'C5_dividend': ['dividend_yield_ttm'],
            'C6_pe': ['pe_ratio_lyr'],
            'C7_a024': ['WorldQuant_alpha024'],
            'C8_a037': ['WorldQuant_alpha037'],
            'C9_a006_014': ['WorldQuant_alpha006', 'WorldQuant_alpha014'],
            'C10_a040': ['WorldQuant_alpha040'],
            'C11_a003': ['WorldQuant_alpha003'],
            'C12_a027': ['WorldQuant_alpha027'],
            'C13_a050': ['WorldQuant_alpha050'],
            'C14_a013_016': ['WorldQuant_alpha013', 'WorldQuant_alpha016'],
            'C15_a044': ['WorldQuant_alpha044'],
            'C16_a015': ['WorldQuant_alpha015'],
            'C17_a055': ['WorldQuant_alpha055'],
            'C18_a088': ['WorldQuant_alpha088'],
            'C19_a011': ['WorldQuant_alpha011']}
# 医药24个合格因子(01脚本筛出: 1m∪3m, |IC|≥0.03 & |ICIR|≥0.3)
_POOL = ['WorldQuant_alpha003', 'WorldQuant_alpha006', 'WorldQuant_alpha011',
         'WorldQuant_alpha013', 'WorldQuant_alpha014', 'WorldQuant_alpha015',
         'WorldQuant_alpha016', 'WorldQuant_alpha024', 'WorldQuant_alpha027',
         'WorldQuant_alpha037', 'WorldQuant_alpha040', 'WorldQuant_alpha044',
         'WorldQuant_alpha050', 'WorldQuant_alpha055', 'WorldQuant_alpha088',
         'book_to_market_ratio_lf', 'dividend_yield_ttm', 'ocf_to_debt_ttm',
         'pb_ratio_lf', 'pe_ratio_lyr', 'ps_ratio_ttm',
         'surplus_cash_protection_multiples_ttm', 'VOLT20', 'VOLT60']
# F4/F7 = _pick_sets.py 数据驱动: 每簇取avg|ICIR|最高代表, 按强度取前4/前7
F4 = ['WorldQuant_alpha040', 'WorldQuant_alpha016', 'surplus_cash_protection_multiples_ttm', 'WorldQuant_alpha050']; NEG4 = set()
F7 = ['WorldQuant_alpha040', 'WorldQuant_alpha016', 'surplus_cash_protection_multiples_ttm', 'WorldQuant_alpha050',
      'dividend_yield_ttm', 'WorldQuant_alpha027', 'WorldQuant_alpha006']; NEG7 = set()


def load_panel(factor_list=None):
    """加载价格+因子+医药白名单 → 因子面板(inf已清洗)。factor_list默认用 _POOL。"""
    facs = factor_list or _POOL
    price = pd.read_parquet(f'{DATA}/mainboard_price.parquet'); price['date'] = pd.to_datetime(price['date'])
    close = price.pivot(index='date', columns='order_book_id', values='close').sort_index()
    me = pd.DatetimeIndex(pd.read_parquet(f'{DATA}/monthend_dates.parquet')['monthend'])
    mev = [m for m in me if m in close.index]; cm = close.loc[mev]
    fwd1 = cm.pct_change(fill_method=None).shift(-1); fwd3 = cm.shift(-3) / cm - 1
    stock_ret = cm.pct_change(fill_method=None)

    def beta_p(n):
        rd = close.pct_change(fill_method=None); mk = rd.mean(axis=1)
        return rd.rolling(n).cov(mk).div(mk.rolling(n).var(), axis=0)
    self_mats = {'MOM120': close.pct_change(120, fill_method=None),
                 'MOM_12M_1M': close.shift(20) / close.shift(240) - 1,
                 'beta60': beta_p(60), 'beta250': beta_p(250)}
    factors = pd.read_parquet(f'{DATA}/factors_monthend.parquet'); factors['monthend'] = pd.DatetimeIndex(factors['monthend'])
    univ = pd.read_parquet(f'{DATA}/universe_monthend_pit.parquet'); univ['monthend'] = pd.DatetimeIndex(univ['monthend'])
    fromdb = [f for f in facs if f in factors.columns and f not in self_mats]
    allf = factors[['order_book_id', 'monthend'] + fromdb].copy()
    for n, s in self_mats.items():
        if n in facs:
            sl = s.loc[[m for m in mev if m in s.index]].stack().rename(n).reset_index(); sl.columns = ['monthend', 'order_book_id', n]
            allf = allf.merge(sl, on=['order_book_id', 'monthend'], how='outer')
    allf = allf.merge(univ[univ['industry_l1'] == SECTOR][['order_book_id', 'monthend']], on=['order_book_id', 'monthend'], how='inner')
    keep = [f for f in facs if f in allf.columns]
    allf = allf[['order_book_id', 'monthend'] + keep].sort_values(['monthend', 'order_book_id']).reset_index(drop=True)
    allf[keep] = allf[keep].replace([np.inf, -np.inf], np.nan)   # ★ inf清洗

    me_sorted = sorted(allf['monthend'].unique())
    WIDE = {f: allf.pivot(index='monthend', columns='order_book_id', values=f) for f in keep}
    stks = np.array(WIDE[keep[0]].columns)

    def zs(mat):
        lo, hi = mat.quantile(0.01, axis=1), mat.quantile(0.99, axis=1); o = mat.copy()
        for c in mat.columns: o[c] = o[c].clip(lo, hi, axis=0)
        return o.sub(o.mean(axis=1), axis=0).div(o.std(axis=1).replace(0, np.nan), axis=0)
    Z = {f: zs(WIDE[f]) for f in keep}
    IC1 = {f: pd.Series({m: (lambda d: spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0] if len(d) >= 30 and d.iloc[:, 0].nunique() > 1 else np.nan)
                         (pd.concat([Z[f].loc[m], fwd1.loc[m]], axis=1).dropna()) for m in me_sorted if m in fwd1.index}) for f in keep}
    return {'close': close, 'mev': mev, 'cm': cm, 'fwd1': fwd1, 'fwd3': fwd3, 'stock_ret': stock_ret,
            'me_sorted': me_sorted, 'WIDE': WIDE, 'Z': Z, 'IC1': IC1, 'stks': stks, 'facs': keep,
            'mkt1': fwd1.mean(axis=1), 'mkt3': fwd3.mean(axis=1), 'sector': SECTOR}


def roll_icir(P, facs, fwd):
    ic = {f: pd.Series({m: (lambda d: spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0] if len(d) >= 30 and d.iloc[:, 0].nunique() > 1 else np.nan)
                        (pd.concat([P['Z'][f].loc[m], fwd.loc[m]], axis=1).dropna()) for m in P['me_sorted'] if m in fwd.index}) for f in facs}
    W = {}
    for i, m in enumerate(P['me_sorted']):
        past = P['me_sorted'][max(0, i - 36):i]; w = [abs((lambda v: v.mean() / v.std() if len(v) > 6 and v.std() > 0 else 0)(ic[f].reindex(past).dropna())) for f in facs]
        w = np.array(w); W[m] = w / w.sum() if w.sum() > 0 else np.ones(len(facs)) / len(facs)
    return W


def score_set(P, facs, neg, W):
    out = {}
    for m in P['me_sorted']:
        if m not in W: continue
        z = pd.DataFrame(index=P['stks'])
        for f in facs:
            if f not in P['Z']: continue
            c = P['Z'][f].loc[m].reindex(P['stks']); z[f] = -c if f in neg else c
        out[m] = (z.fillna(0) * np.array(W[m][:len(z.columns)])).sum(axis=1)
    return out


def pcr_scores(P, fwd, k=7, alpha=1.0):
    Z = P['Z']; stks = P['stks']; me_sorted = P['me_sorted']; facs = P['facs']
    scores = {}
    for i, m in enumerate(me_sorted):
        if i < 36 or m not in fwd.index: continue
        past = me_sorted[i - 36:i]; Xp, yp = [], []
        for pm in past:
            if pm not in fwd.index: continue
            Xmat = np.column_stack([Z[f].loc[pm].reindex(stks).values for f in facs])
            yv = fwd.loc[pm].reindex(stks).values
            ok = ~np.isnan(Xmat).any(axis=1) & ~np.isnan(yv)
            if ok.sum(): Xp.append(Xmat[ok]); yp.append(yv[ok])
        if not Xp: continue
        Xp, yp = np.vstack(Xp), np.concatenate(yp)
        if len(Xp) < 200: continue
        pca = PCA(n_components=min(k, len(facs) - 1)); Xpc = pca.fit_transform(Xp)
        reg = Ridge(alpha=alpha).fit(Xpc, yp)
        xm = np.column_stack([Z[f].loc[m].reindex(stks).values for f in facs])
        mask = ~np.isnan(xm).any(axis=1)
        if mask.sum() < 30: continue
        scores[m] = pd.Series(reg.predict(pca.transform(xm[mask])), index=stks[mask])
    return scores


def backtest(P, score_dict, rebal_months, fwd, mkt, thr=None, min_nunique=5, min_q5=3, ret_sets=False):
    """回测。ret_sets=True 时同时返回 (rets, q5_sets:{m:set(股票)}) 供算换手。"""
    rets, q5_sets = [], {}
    for m in rebal_months:
        if m not in fwd.index or m not in score_dict: continue
        s = score_dict[m].dropna()
        if s.nunique() < min_nunique: continue
        try: q = pd.qcut(s.rank(method='first'), 5, labels=False)
        except: continue
        q5 = s[q == 4]
        held = True
        if thr is not None:
            mi = P['mev'].index(m) if m in P['mev'] else -1
            if mi >= 3:
                prev3 = P['mev'][mi - 3:mi]
                keep = [o for o in q5.index if o in P['stock_ret'].columns and P['stock_ret'].loc[prev3, o].notna().all() and P['stock_ret'].loc[prev3, o].sum() < thr]
                q5 = q5[q5.index.isin(keep)]; held = len(q5) >= min_q5
        if held:
            vals = [fwd.loc[m, o] for o in q5.index if o in fwd.columns and pd.notna(fwd.loc[m, o])]
            r = np.mean(vals) - COST if len(vals) >= 2 else np.nan
        else:
            mk = mkt.get(m, np.nan); r = mk - COST if pd.notna(mk) else np.nan
        if pd.notna(r): rets.append(r)
        if ret_sets and held: q5_sets[m] = set(q5.index)
    if ret_sets: return pd.Series(rets), q5_sets
    return pd.Series(rets)


def metrics(r, periods_yr):
    if len(r) < 3: return None
    ann = r.mean() * periods_yr; sh = r.mean() / r.std() * np.sqrt(periods_yr) if r.std() > 0 else 0
    cum = (1 + r).cumprod(); dd = (cum / cum.cummax() - 1).min()
    return {'年化': round(ann, 4), '夏普': round(sh, 2), '最大回撤': round(dd, 4), '期胜率': round((r > 0).mean(), 2), '期数': len(r)}
