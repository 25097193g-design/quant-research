"""10 IC阈值敏感性 — 选因子阈值从|IC|≥0.03收紧到≥0.05(银行原口径), 三大结论还成立吗?

动机: 01 把阈值从0.05(银行)放宽到0.03, 24因子里14个是0.03~0.05之间的弱因子(多为WQ alpha),
"19独立簇/去冗余丢alpha/ML加分"这三个故事有多少是这14个弱因子撑起来的?
把阈值收回去, 只留|IC|≥0.05 & |ICIR|≥0.3的强因子, 整条链重跑, 和0.03结果并排对比。

复用 common.py(load_panel/roll_icir/score_set/backtest/metrics/pcr_scores) — 单一真相源。
聚类逻辑=02, F4/F7逻辑=_pick_sets, ML逻辑=07/09, 只在"因子池"一处换。
输出: outputs/ic050_clusters.csv, ic050_dedup.csv, ic050_freq.csv, ic050_ml.csv, ic_threshold_compare.csv
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import common

OUT = '/Users/lichanghaodemac/quant_strategy/projects/pharma_factor_workflow/outputs'
IC_THRESH = 0.05

# ===== 1. |IC|≥0.05 重新选因子 (1m∪3m, 同01) =====
ic = pd.read_csv(f'{common.DATA}/ic_pharma_horizon.csv')
sub = ic[ic['horizon'].isin(['1m', '3m'])].copy()
sub['aIC'] = sub['医药IC'].abs(); sub['aICIR'] = sub['医药ICIR'].abs()
qual = sub[(sub.aIC >= IC_THRESH) & (sub.aICIR >= 0.30)]
pool = sorted(set(qual['factor']))
print(f'=== |IC|≥{IC_THRESH} & |ICIR|≥0.3: 1m={len(qual[qual.horizon=="1m"])}, 3m={len(qual[qual.horizon=="3m"])}, 并集={len(pool)} ===', flush=True)
print('  ' + ', '.join(pool), flush=True)

def neg_set(facs, h='1m'):
    out = set()
    for f in facs:
        v = ic[(ic['factor'] == f) & (ic['horizon'] == h)]['医药IC']
        if len(v) and v.iloc[0] < 0: out.add(f)
    return out

NEG = neg_set(pool)
print(f'  NEG(1m IC为负需反向): {sorted(NEG)}', flush=True)

# ===== 2. 面板 + 相关性 + 层次聚类 (同02, 复用 common) =====
P = common.load_panel(pool)
FACS = P['facs']
allf_long = pd.concat([P['WIDE'][f].stack().rename(f) for f in FACS], axis=1).reset_index()
allf_long.columns = ['monthend', 'order_book_id'] + FACS
acc = np.zeros((len(FACS), len(FACS))); cnt = np.zeros_like(acc)
for m, g in allf_long.groupby('monthend'):
    if len(g) < 30: continue
    c = g[FACS].rank().corr(method='pearson').values
    mk = ~np.isnan(c); acc += np.where(mk, c, 0); cnt += mk
mean_corr = pd.DataFrame(acc / np.where(cnt > 0, cnt, 1), index=FACS, columns=FACS)
dist = 1 - mean_corr.abs().values; np.fill_diagonal(dist, 0); dist = (dist + dist.T) / 2
Z = linkage(squareform(dist, checks=False), method='average')
cl = fcluster(Z, t=0.4, criterion='distance')
ncl = len(set(cl))
print(f'\n→ {ncl} 个独立簇 (0.03版是19):', flush=True)
CLUSTERS = {}
for c in sorted(set(cl)):
    mem = [FACS[i] for i in range(len(FACS)) if cl[i] == c]
    CLUSTERS[f'C{c}'] = mem
    print(f'  组{c}: {", ".join(mem)}', flush=True)
pd.DataFrame({'factor': FACS, 'cluster': cl}).to_csv(f'{OUT}/ic050_clusters.csv', index=False, encoding='utf-8-sig')

# ===== 3. 数据驱动 F4/F7 (同_pick_sets) =====
best = []
for c in sorted(set(cl)):
    facs = [FACS[i] for i in range(len(FACS)) if cl[i] == c]
    g = ic[ic['factor'].isin(facs)].groupby('factor')['医药ICIR'].agg(lambda s: np.nanmean(s.abs())).sort_values(ascending=False)
    best.append({'cluster': c, 'rep': g.index[0], 'ICIR': round(g.iloc[0], 3)})
bdf = pd.DataFrame(best).sort_values('ICIR', ascending=False).reset_index(drop=True)
F7 = bdf['rep'].head(7).tolist(); F4 = bdf['rep'].head(4).tolist()
print(f'\nF4 = {F4}\nF7 = {F7}\nNEG4 = {sorted(neg_set(F4))}  NEG7 = {sorted(neg_set(F7))}', flush=True)

# ===== 4. 去冗余对照 (同04): 全池(10F) vs 7F vs 4F × 滚动ICIR × 4档gate, 月频 =====
fwd1, mkt1 = P['fwd1'], P['mkt1']
me_sorted = P['me_sorted']
SETS = {f'全池({len(pool)}F)': (FACS, NEG), '7F': (F7, neg_set(F7)), '4F': (F4, neg_set(F4))}
THRS = [('无过滤', None), ('<-5%', -0.05), ('<-3%', -0.03), ('<-1%', -0.01)]
rows = []
for setname, (facs, neg) in SETS.items():
    W = common.roll_icir(P, facs, fwd1)
    sc = common.score_set(P, facs, neg, W)
    for lb, thr in THRS:
        r, q5s = common.backtest(P, sc, me_sorted[36:], fwd1, mkt1, thr, ret_sets=True)
        m = common.metrics(r, 12)
        if m is None: continue
        m.update({'因子集': setname, '过滤': lb})
        rows.append(m)
dedup = pd.DataFrame(rows)[['因子集', '过滤', '年化', '夏普', '最大回撤', '期胜率', '期数']]
dedup.to_csv(f'{OUT}/ic050_dedup.csv', index=False, encoding='utf-8-sig')
print('\n=== 去冗余对照 (滚动ICIR, 月频, |IC|≥0.05) ===', flush=True)
print(dedup.to_string(index=False), flush=True)

# ===== 5. 频率×因子集矩阵 (同05): 全池/7F/4F/PCA × 月/季 × 无过滤/<-1% =====
fwd3, mkt3 = P['fwd3'], P['mkt3']
monthly = [m for i, m in enumerate(me_sorted) if i >= 36]
quarterly = [m for i, m in enumerate(me_sorted) if i >= 36 and (i - 36) % 3 == 0]
SC = {
    (f'全池({len(pool)}F)', '月'): common.score_set(P, FACS, NEG, common.roll_icir(P, FACS, fwd1)),
    (f'全池({len(pool)}F)', '季'): common.score_set(P, FACS, NEG, common.roll_icir(P, FACS, fwd3)),
    ('7F', '月'): common.score_set(P, F7, neg_set(F7), common.roll_icir(P, F7, fwd1)),
    ('7F', '季'): common.score_set(P, F7, neg_set(F7), common.roll_icir(P, F7, fwd3)),
    ('4F', '月'): common.score_set(P, F4, neg_set(F4), common.roll_icir(P, F4, fwd1)),
    ('4F', '季'): common.score_set(P, F4, neg_set(F4), common.roll_icir(P, F4, fwd3)),
    ('PCA', '月'): common.pcr_scores(P, fwd1),
    ('PCA', '季'): common.pcr_scores(P, fwd3),
}
FWD = {'月': (fwd1, mkt1, 12, monthly), '季': (fwd3, mkt3, 4, quarterly)}
frows = []
for (setname, freq), sc in SC.items():
    fwd, mkt, py, rebal = FWD[freq]
    for lb, thr in [('无过滤', None), ('<-1%', -0.01)]:
        r = common.backtest(P, sc, rebal, fwd, mkt, thr)
        m = common.metrics(r, py)
        if m: m.update({'因子集': setname, '调仓': freq, '过滤': lb}); frows.append(m)
freq_res = pd.DataFrame(frows)[['因子集', '调仓', '过滤', '年化', '夏普', '最大回撤', '期胜率', '期数']]
freq_res.to_csv(f'{OUT}/ic050_freq.csv', index=False, encoding='utf-8-sig')
print('\n=== 频率×因子集矩阵 (|IC|≥0.05) ===', flush=True)
print(freq_res[freq_res['过滤'] == '无过滤'][['因子集', '调仓', '年化', '夏普', '最大回撤']].to_string(index=False), flush=True)

# ===== 6. ML对照 (同07/09): 簇合成因子 → baseline vs XGBoost, 3相位 =====
def cluster_composite(m, past):
    comp = {}
    for cname, facs in CLUSTERS.items():
        vals = []
        for f in facs:
            v = P['IC1'][f].reindex(past).dropna()
            sgn = np.sign(v.mean()) if len(v) > 6 else 1.0
            vals.append(sgn * P['Z'][f].loc[m])
        comp[cname] = pd.concat(vals, axis=1).mean(axis=1)
    return pd.DataFrame(comp).reindex(P['stks'])

def baseline_q5(m, past):
    comp = cluster_composite(m, past); W = []
    for c in CLUSTERS:
        ics = []
        for pm in past:
            if pm not in fwd3.index: continue
            cmp_pm = cluster_composite(pm, [mm for mm in me_sorted if mm < pm][-36:])[c]
            d = pd.concat([cmp_pm, fwd3.loc[pm]], axis=1).dropna()
            if len(d) >= 10 and d.iloc[:, 0].nunique() > 1: ics.append(spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0])
        v = np.array(ics); W.append(abs(v.mean() / v.std()) if len(v) > 6 and v.std() > 0 else 0)
    W = np.array(W); W = W / W.sum() if W.sum() > 0 else np.ones(len(CLUSTERS)) / len(CLUSTERS)
    return (comp.fillna(0) * W).sum(axis=1)

def xgb_q5(m, past):
    comp = cluster_composite(m, past); Xtr, ytr = [], []
    for pm in past:
        if pm not in fwd3.index: continue
        cmp_pm = cluster_composite(pm, [mm for mm in me_sorted if mm < pm][-36:])
        for s in P['stks']:
            xv = cmp_pm.loc[s].values; yv = fwd3.loc[pm, s] if s in fwd3.columns else np.nan
            if not np.isnan(xv).any() and pd.notna(yv): Xtr.append(xv); ytr.append(yv)
    Xtr, ytr = np.array(Xtr), np.array(ytr)
    if len(Xtr) < 300: return None
    xs = StandardScaler().fit(Xtr)
    mdl = XGBRegressor(n_estimators=60, max_depth=2, learning_rate=0.05, subsample=0.7,
                       colsample_bytree=0.7, reg_lambda=5, reg_alpha=2, n_jobs=2).fit(xs.transform(Xtr), ytr)
    return pd.Series(mdl.predict(xs.transform(comp.loc[P['stks']].fillna(0).values)), index=P['stks'])

def walk(quarterly, scorer):
    rets = []
    for m in quarterly:
        past = [mm for mm in me_sorted if mm < m][-36:]
        if len(past) < 36 or m not in fwd3.index: continue
        sc = scorer(m, past)
        if sc is None or sc.dropna().nunique() < 5: continue
        sc = sc.dropna()
        try: q = pd.qcut(sc.rank(method='first'), 5, labels=False)
        except: continue
        q5 = sc[q == 4].index
        if len(q5) < 3: continue
        rets.append(np.mean([fwd3.loc[m, o] for o in q5 if o in fwd3.columns and pd.notna(fwd3.loc[m, o])]) - common.COST)
    return np.array(rets)

ml_rows = []
for ph in [0, 1, 2]:
    quarterly = [m for i, m in enumerate(me_sorted) if i >= 36 and (i - 36) % 3 == ph]
    print(f'\n>>> ML phase={ph} ({len(quarterly)}季度点)', flush=True)
    for name, scorer in [('baseline', baseline_q5), ('XGBoost', xgb_q5)]:
        r = walk(quarterly, scorer)
        if len(r) < 5: continue
        r = pd.Series(r); ann = r.mean() * 4; sh = r.mean() / r.std() * 2 if r.std() > 0 else 0
        cum = (1 + r).cumprod(); dd = (cum / cum.cummax() - 1).min()
        ml_rows.append({'模型': name, 'phase': ph, '年化': round(ann, 4), '夏普(季)': round(sh, 2), '最大回撤': round(dd, 4)})
ml = pd.DataFrame(ml_rows)
ml.to_csv(f'{OUT}/ic050_ml.csv', index=False, encoding='utf-8-sig')
print('\n=== ML 3相位 (|IC|≥0.05, 簇数={}) ==='.format(ncl), flush=True)
print(ml.to_string(index=False), flush=True)

# ===== 7. 0.03 vs 0.05 并排对比 =====
d03 = pd.read_csv(f'{OUT}/dedup_vs_full.csv')
d05 = pd.read_csv(f'{OUT}/ic050_dedup.csv')
f03 = pd.read_csv(f'{OUT}/freq_sets_matrix.csv')
f05 = pd.read_csv(f'{OUT}/ic050_freq.csv')
m03 = pd.read_csv(f'{OUT}/ml_phase_robust.csv')
print('\n========== |IC|≥0.03 vs ≥0.05 并排对比 ==========\n', flush=True)
print('-- 去冗余(月频,滚动ICIR,无过滤) --', flush=True)
for row in d03[(d03.加权 == '滚动') & (d03.过滤 == '无过滤')].itertuples():
    print(f'  0.03 {row.因子集:<18} 年化{row.年化:>6.2%} 夏普{row.夏普:>5.2f} 回撤{row.最大回撤:>7.2%}', flush=True)
for row in d05[d05.过滤 == '无过滤'].itertuples():
    print(f'  0.05 {row.因子集:<18} 年化{row.年化:>6.2%} 夏普{row.夏普:>5.2f} 回撤{row.最大回撤:>7.2%}', flush=True)
print('-- 频率矩阵(无过滤, 最优组合) --', flush=True)
for df, tag in [(f03, '0.03'), (f05, '0.05')]:
    best = df[df.过滤 == '无过滤'].sort_values('夏普', ascending=False).iloc[0]
    print(f'  {tag} 最优: {best.因子集}/{best.调仓} 年化{best.年化:.2%} 夏普{best.夏普:.2f} 回撤{best.最大回撤:.2%}', flush=True)
print('-- ML(季度, 3相位) --', flush=True)
for df, tag in [(m03, '0.03'), (ml, '0.05')]:
    for name in ['baseline', 'XGBoost']:
        s = df[(df.模型 == name) & (df['phase'].astype(str).isin(['0', '1', '2']))]
        if len(s): print(f'  {tag} {name:<8} 年化中位{np.median(s.年化):.2%} 夏普中位{np.median(s["夏普(季)"]):.2f} (n={len(s)})', flush=True)
print('\n存 ic050_clusters/dedup/freq/ml + ic050_dedup.csv 等', flush=True)
