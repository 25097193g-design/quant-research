"""04 去冗余对照 — frame_18 超跌埋伏轮动: 7因子 vs 去冗余4因子

问题: 02/03 发现 frame_18 的7因子有效维度只有~4, 且 beta60 & market_cap_3 同簇重复加权。
       去冗余成4因子(dividend_yield+ev_to_ebitda+beta60+MOM_12M_1M, 每簇一个代表)后,
       夏普/换手实际差多少? 去冗余有没有真金白银收益?

口径(与 frame_18 完全一致, 只换因子集):
  - 银行41只, 月调仓, z-score(1/99%winsorize), ICIR加权, qcut5取Q5
  - 超跌过滤: 前3月收益和 < thr 才埋伏银行Q5; 否则空仓投全市场等权
  - 成本单边万7.5(双边0.15%)
两套加权:
  - 静态ICIR(全样本|ICIR|, 有未来函数→frame_18原版, 用于复现baseline)
  - 滚动ICIR(36月rolling, 无未来→诚实walk-forward版)
额外: Q5成分月度换手(Jaccard距离均值) → 选股稳定性

输出: outputs/dedup_vs_full.csv + 对照表打印
"""
import pandas as pd, numpy as np
from scipy.stats import spearmanr

DATA = '/Users/lichanghaodemac/quant_strategy/data'
OUT = '/Users/lichanghaodemac/quant_strategy/projects/bank_factor_workflow/outputs'
COST = 0.0015

price = pd.read_parquet(f'{DATA}/mainboard_price.parquet'); price['date'] = pd.to_datetime(price['date'])
close = price.pivot(index='date', columns='order_book_id', values='close').sort_index()
me = pd.DatetimeIndex(pd.read_parquet(f'{DATA}/monthend_dates.parquet')['monthend'])
mev = [m for m in me if m in close.index]; cm = close.loc[mev]
fwd = cm.pct_change(fill_method=None).shift(-1)
stock_ret = cm.pct_change(fill_method=None)
mkt = cm.pct_change(fill_method=None).shift(-1).mean(axis=1)   # 全市场等权前瞻

def beta_p(n):
    rd = close.pct_change(fill_method=None); mk = rd.mean(axis=1)
    return rd.rolling(n).cov(mk).div(mk.rolling(n).var(), axis=0)
mom121 = close.shift(20) / close.shift(240) - 1
ms = mom121.loc[[m for m in mev if m in mom121.index]].stack().rename('MOM_12M_1M').reset_index(); ms.columns = ['monthend', 'order_book_id', 'MOM_12M_1M']
bs = beta_p(60); bs = bs.loc[[m for m in mev if m in bs.index]].stack().rename('beta60').reset_index(); bs.columns = ['monthend', 'order_book_id', 'beta60']
factors = pd.read_parquet(f'{DATA}/factors_monthend.parquet'); factors['monthend'] = pd.DatetimeIndex(factors['monthend'])
univ = pd.read_parquet(f'{DATA}/universe_monthend_pit.parquet'); univ['monthend'] = pd.DatetimeIndex(univ['monthend'])
bank_univ = univ[univ['industry_l1'] == '银行'][['order_book_id', 'monthend']]

ALLNEED = ['dividend_yield_ttm', 'ev_to_ebitda_ttm', 'pe_ratio_ttm', 'market_cap_3', 'net_profit_margin_ttm', 'beta60', 'MOM_12M_1M']
allf = factors[['order_book_id', 'monthend'] + ['dividend_yield_ttm', 'ev_to_ebitda_ttm', 'pe_ratio_ttm', 'market_cap_3', 'net_profit_margin_ttm']]\
    .merge(bs, on=['order_book_id', 'monthend']).merge(ms, on=['order_book_id', 'monthend']).merge(bank_univ, on=['order_book_id', 'monthend'])
fwd_s = fwd.stack().rename('fwd').reset_index(); fwd_s.columns = ['monthend', 'order_book_id', 'fwd']
allf = allf.merge(fwd_s, on=['monthend', 'order_book_id'], how='left').dropna(subset=['fwd']).sort_values(['monthend', 'order_book_id']).reset_index(drop=True)
me_sorted = sorted(allf['monthend'].unique())

SETS = {
    '7因子(frame18)': (['dividend_yield_ttm', 'ev_to_ebitda_ttm', 'pe_ratio_ttm', 'market_cap_3', 'net_profit_margin_ttm', 'beta60', 'MOM_12M_1M'],
                       {'ev_to_ebitda_ttm', 'pe_ratio_ttm', 'beta60'}),
    '4因子(去冗余)': (['dividend_yield_ttm', 'ev_to_ebitda_ttm', 'beta60', 'MOM_12M_1M'],
                       {'ev_to_ebitda_ttm', 'beta60'}),
}

def make_z(df, facs, neg):
    def wz(g):
        out = pd.DataFrame(index=g.index)
        for f in facs:
            s = g[f].clip(g[f].quantile(0.01), g[f].quantile(0.99)); sd = s.std()
            out[f + '_z'] = (s - s.mean()) / sd if sd > 0 else 0
            if f in neg: out[f + '_z'] = -out[f + '_z']
        return out
    return pd.concat([df, df.groupby('monthend', group_keys=False).apply(wz)], axis=1)

def icir_series(df, f, min_n=10):
    out = {}
    for mn, g in df.groupby('monthend'):
        if f not in g.columns: continue
        s = g[[f, 'fwd']].dropna()
        if len(s) >= min_n and s[f].nunique() > 1: out[mn] = spearmanr(s[f], s['fwd'])[0]
    return pd.Series(out)

def run(df, zcol, thr, W_func):
    rets, q5_sets = [], {}
    for i, m in enumerate(me_sorted):
        if i < 36: continue
        g = df[df['monthend'] == m].copy()
        if len(g) < 15: continue
        W = W_func(m); g['score'] = (g[zcol].values * W).sum(axis=1)
        s = g[['order_book_id', 'score', 'fwd']].dropna()
        if len(s) < 15: continue
        try: s['q'] = pd.qcut(s['score'], 5, labels=False)
        except: continue
        q5 = s[s.q == 4].copy()
        q5_sets[m] = set(q5['order_book_id'])
        if thr is not None:
            mi = mev.index(m) if m in mev else -1
            if mi >= 3:
                prev3 = mev[mi - 3:mi]
                keep = [o for o in q5['order_book_id'] if o in stock_ret.columns and stock_ret.loc[prev3, o].notna().all() and stock_ret.loc[prev3, o].sum() < thr]
                q5 = q5[q5['order_book_id'].isin(keep)]
        if len(q5) < 2: rets.append({'m': m, 'ret': 0.0, 'held': False})
        else: rets.append({'m': m, 'ret': q5['fwd'].mean() - COST, 'held': True})
    return pd.DataFrame(rets).set_index('m'), q5_sets

def turnover(q5_sets):
    months = sorted(q5_sets)
    ds = []
    for a, b in zip(months, months[1:]):
        u, i = q5_sets[a], q5_sets[b]
        if u or i: ds.append(1 - len(u & i) / len(u | i))   # Jaccard距离
    return np.mean(ds) if ds else np.nan

def perf(d):
    hold = d['held']; rh = d.loc[hold, 'ret']
    ann_h = rh.mean() * 12 if len(rh) else 0
    sh_h = rh.mean() / rh.std() * np.sqrt(12) if len(rh) > 1 and rh.std() > 0 else 0
    mk = mkt.reindex(d.index)
    comb = pd.Series(np.where(hold, d['ret'], mk - COST), index=d.index).dropna()
    ann_c = comb.mean() * 12; sh_c = comb.mean() / comb.std() * np.sqrt(12) if comb.std() > 0 else 0
    return ann_h, sh_h, hold.mean(), ann_c, sh_c

THRS = [('无过滤', None), ('<-7%', -0.07), ('<-5%', -0.05), ('<-3%', -0.03), ('<-1%', -0.01), ('<0%', 0.0)]
rows = []
for setname, (facs, neg) in SETS.items():
    df = make_z(allf, facs, neg); zcol = [f + '_z' for f in facs]
    ic_ser = {f: icir_series(df, f) for f in facs}
    icir_full = {f: (ic_ser[f].mean() / ic_ser[f].std() if ic_ser[f].std() > 0 else 0) for f in facs}
    W_static = np.array([abs(icir_full[f]) for f in facs]); W_static = W_static / W_static.sum() if W_static.sum() > 0 else np.ones(len(facs)) / len(facs)
    W_roll = {}
    for m in me_sorted:
        past = [x for x in me_sorted if x < m][-36:]
        w = np.array([abs((lambda s: s.mean() / s.std() if len(s) > 6 and s.std() > 0 else 0)(ic_ser[f].reindex(past).dropna())) for f in facs])
        W_roll[m] = w / w.sum() if w.sum() > 0 else np.ones(len(facs)) / len(facs)
    for scheme, W_func in [('静态', lambda m, ws=W_static: ws), ('滚动', lambda m: W_roll.get(m, np.ones(len(facs)) / len(facs)))]:
        for lb, thr in THRS:
            d, q5s = run(df, zcol, thr, W_func)
            ann_h, sh_h, rate, ann_c, sh_c = perf(d)
            tov = turnover({m: q5s[m] for m in q5s if m in d.index})
            rows.append({'因子集': setname, '加权': scheme, '过滤': lb, '组合年化': round(ann_c, 4),
                         '组合夏普': round(sh_c, 2), '持仓年化': round(ann_h, 4), '持仓夏普': round(sh_h, 2),
                         '持仓占比': round(rate, 2), 'Q5月换手': round(tov, 2)})
res = pd.DataFrame(rows)
res.to_csv(f'{OUT}/dedup_vs_full.csv', index=False, encoding='utf-8-sig')

# ★ 基准必须与策略同窗口: 策略从 me_sorted[36](2020-12)起跑, 全样本均值含2017-2019, 不可比
mk_full = mkt.reindex(me_sorted).dropna(); mk_win = mkt.reindex(me_sorted[36:]).dropna()
print(f'全市场等权基准: 全样本{mk_full.mean() * 12:.2%}/夏普{mk_full.mean() / mk_full.std() * np.sqrt(12):.2f}'
      f' | ★策略同窗口({str(me_sorted[36])[:10]}起){mk_win.mean() * 12:.2%}/夏普{mk_win.mean() / mk_win.std() * np.sqrt(12):.2f}'
      f' — 报超额必须用同窗口\n', flush=True)
for scheme in ['静态', '滚动']:
    print(f'===== {scheme}ICIR 加权 ({"有未来函数, 复现frame_18 baseline" if scheme=="静态" else "36月rolling, 无未来, 诚实版"}) =====', flush=True)
    print(f'{"因子集":<16}{"过滤":<9}{"组合年化":>9}{"组合夏普":>9}{"持仓占比":>9}{"Q5月换手":>9}', flush=True)
    for setname in SETS:
        for lb, _ in THRS:
            r = res[(res.因子集 == setname) & (res.加权 == scheme) & (res.过滤 == lb)].iloc[0]
            print(f'{setname:<16}{lb:<9}{r["组合年化"]:>9.2%}{r["组合夏普"]:>9.2f}{r["持仓占比"]:>9.0%}{r["Q5月换手"]:>9.0%}', flush=True)
        print(flush=True)
print('存 outputs/dedup_vs_full.csv', flush=True)
