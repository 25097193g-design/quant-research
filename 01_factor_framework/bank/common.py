"""common.py — 银行因子工作流共享模块(单一真相源)

所有脚本(05/07/08/09/notebook)从这里取数据清洗 + 因子定义 + PCR + 回测,
消除之前 05(pcr_scores) 和 09(pcr_monthly) 两份独立代码的 drift(~1pp不一致)。

关键修正:
  - inf 清洗: alpha026 有 82 个 inf(银行) → 全部 replace 成 NaN(在 load_panel 里统一做)
  - PCR 唯一: 只此一份 pcr_scores(), 大家调用
  - 回测规则唯一: backtest() 用统一跳过规则(nunique<5 跳, q5<3 跳), 不再各写各的
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

DATA = '/Users/lichanghaodemac/quant_strategy/data'
COST = 0.0015

CLUSTERS = {'C1_wqalpha': ['WorldQuant_alpha013', 'WorldQuant_alpha016'],
            'C2_sizebeta': ['beta60', 'beta250', 'market_cap_3', 'a_share_market_val_in_circulation', 'OBV'],
            'C3_value': ['dividend_yield_ttm', 'pe_ratio_ttm', 'pe_ratio_lyr'],
            'C4_quality': ['ev_to_ebitda_ttm', 'net_profit_margin_ttm', 'return_on_equity_ttm'],
            'C5_mom': ['MOM120', 'MOM_12M_1M'], 'C6_davol': ['DAVOL20'],
            'C7_a026': ['WorldQuant_alpha026'], 'C8_a088': ['WorldQuant_alpha088']}
ALL18 = sorted({f for v in CLUSTERS.values() for f in v})
F4 = ['dividend_yield_ttm', 'ev_to_ebitda_ttm', 'beta60', 'MOM_12M_1M']; NEG4 = {'ev_to_ebitda_ttm', 'beta60'}
F7 = ['dividend_yield_ttm', 'ev_to_ebitda_ttm', 'pe_ratio_ttm', 'market_cap_3', 'net_profit_margin_ttm', 'beta60', 'MOM_12M_1M']; NEG7 = {'ev_to_ebitda_ttm', 'pe_ratio_ttm', 'beta60'}
_SELF = ['MOM120', 'MOM_12M_1M', 'beta60', 'beta250']
FROMDB = [f for f in ALL18 if f not in _SELF]


def load_panel():
    """加载价格+因子+银行白名单 → 18因子面板(inf已清洗) + 衍生对象。返回 dict。"""
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
    allf = factors[['order_book_id', 'monthend'] + FROMDB].copy()
    for n, s in self_mats.items():
        sl = s.loc[[m for m in mev if m in s.index]].stack().rename(n).reset_index(); sl.columns = ['monthend', 'order_book_id', n]
        allf = allf.merge(sl, on=['order_book_id', 'monthend'], how='outer')
    allf = allf.merge(univ[univ['industry_l1'] == '银行'][['order_book_id', 'monthend']], on=['order_book_id', 'monthend'], how='inner')
    allf = allf[['order_book_id', 'monthend'] + ALL18].sort_values(['monthend', 'order_book_id']).reset_index(drop=True)
    allf[ALL18] = allf[ALL18].replace([np.inf, -np.inf], np.nan)   # ★ inf清洗(alpha026等)

    me_sorted = sorted(allf['monthend'].unique())
    WIDE = {f: allf.pivot(index='monthend', columns='order_book_id', values=f) for f in ALL18}
    stks = np.array(WIDE[ALL18[0]].columns)

    def zs(mat):
        lo, hi = mat.quantile(0.01, axis=1), mat.quantile(0.99, axis=1); o = mat.copy()
        for c in mat.columns: o[c] = o[c].clip(lo, hi, axis=0)
        return o.sub(o.mean(axis=1), axis=0).div(o.std(axis=1).replace(0, np.nan), axis=0)
    Z = {f: zs(WIDE[f]) for f in ALL18}
    IC1 = {f: pd.Series({m: (lambda d: spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0] if len(d) >= 10 and d.iloc[:, 0].nunique() > 1 else np.nan)
                         (pd.concat([Z[f].loc[m], fwd1.loc[m]], axis=1).dropna()) for m in me_sorted if m in fwd1.index}) for f in ALL18}
    return {'close': close, 'mev': mev, 'cm': cm, 'fwd1': fwd1, 'fwd3': fwd3, 'stock_ret': stock_ret,
            'me_sorted': me_sorted, 'WIDE': WIDE, 'Z': Z, 'IC1': IC1, 'stks': stks,
            'mkt1': fwd1.mean(axis=1), 'mkt3': fwd3.mean(axis=1)}


def roll_icir(P, facs, fwd):
    """滚动36月 |ICIR| 权重 → {month: weight_vec}"""
    ic = {f: pd.Series({m: (lambda d: spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0] if len(d) >= 10 and d.iloc[:, 0].nunique() > 1 else np.nan)
                        (pd.concat([P['Z'][f].loc[m], fwd.loc[m]], axis=1).dropna()) for m in P['me_sorted'] if m in fwd.index}) for f in facs}
    W = {}
    for i, m in enumerate(P['me_sorted']):
        past = P['me_sorted'][max(0, i - 36):i]; w = [abs((lambda v: v.mean() / v.std() if len(v) > 6 and v.std() > 0 else 0)(ic[f].reindex(past).dropna())) for f in facs]
        w = np.array(w); W[m] = w / w.sum() if w.sum() > 0 else np.ones(len(facs)) / len(facs)
    return W


def score_set(P, facs, neg, W):
    """因子集月度score → {month: Series}, 符号对齐+winsorize z+ICIR加权"""
    out = {}
    for m in P['me_sorted']:
        if m not in W: continue
        z = pd.DataFrame(index=P['stks'])
        for f in facs:
            c = P['Z'][f].loc[m].reindex(P['stks']); z[f] = -c if f in neg else c
        out[m] = (z.fillna(0) * np.array(W[m])).sum(axis=1)
    return out


def pcr_scores(P, fwd, k=7, alpha=1.0):
    """★ 唯一PCR: 18因子→PCA(k)→Ridge, walk-forward(滚动36月训练), 返回 {month: Series}"""
    Z = P['Z']; stks = P['stks']; me_sorted = P['me_sorted']
    scores = {}
    for i, m in enumerate(me_sorted):
        if i < 36 or m not in fwd.index: continue
        past = me_sorted[i - 36:i]; Xp, yp = [], []
        for pm in past:
            if pm not in fwd.index: continue
            Xmat = np.column_stack([Z[f].loc[pm].reindex(stks).values for f in ALL18])
            yv = fwd.loc[pm].reindex(stks).values
            ok = ~np.isnan(Xmat).any(axis=1) & ~np.isnan(yv)
            if ok.sum(): Xp.append(Xmat[ok]); yp.append(yv[ok])
        if not Xp: continue
        Xp, yp = np.vstack(Xp), np.concatenate(yp)
        if len(Xp) < 200: continue
        pca = PCA(n_components=k); Xpc = pca.fit_transform(Xp)
        reg = Ridge(alpha=alpha).fit(Xpc, yp)
        xm = np.column_stack([Z[f].loc[m].reindex(stks).values for f in ALL18])
        mask = ~np.isnan(xm).any(axis=1)
        if mask.sum() < 15: continue
        scores[m] = pd.Series(reg.predict(pca.transform(xm[mask])), index=stks[mask])
    return scores


def backtest(P, score_dict, rebal_months, fwd, mkt, thr=None, min_nunique=5, min_q5=3):
    """★ 唯一回测规则: 超跌过滤thr(None=一直持仓Q5), 统一跳过(nunique<min 或 q5<min 跳)。
    返回按调仓月(datetime)索引的收益Series — 被跳过的月不出现, 净值曲线/基准对齐都靠这个索引。
    组合=持仓月Q5等权, 空仓月全市场等权。"""
    idx, rets = [], []
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
        if pd.notna(r): idx.append(m); rets.append(r)
    return pd.Series(rets, index=pd.DatetimeIndex(idx))


def metrics(r, periods_yr):
    """年化/夏普/回撤/胜率"""
    if len(r) < 3: return None
    ann = r.mean() * periods_yr; sh = r.mean() / r.std() * np.sqrt(periods_yr) if r.std() > 0 else 0
    cum = (1 + r).cumprod(); dd = (cum / cum.cummax() - 1).min()
    return {'年化': round(ann, 4), '夏普': round(sh, 2), '最大回撤': round(dd, 4), '期胜率': round((r > 0).mean(), 2), '期数': len(r)}
