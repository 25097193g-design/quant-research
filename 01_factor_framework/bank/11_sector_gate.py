"""11 板块级超跌/横盘 gate — 直接检验"银行横盘/超跌后未来几个月有一波行情"这个命题

背景: 04 的"超跌过滤"实际是个股级(Q5里每只股票自己的前3月收益<阈值), 不是板块级信号。
本脚本把三个变体放同一张表对质(全部持币2%/年做空仓收益, 不用纸面等权全市场):

  A. 无过滤:     一直满仓Q5
  B. 个股级gate:  Q5里保留"自己前3月收益<thr"的(04的实现, 空仓月改持币)
  C. 板块级gate:  银行等权指数前3月收益<thr → 满仓Q5(整组, 不做个股过滤), 否则持币  ← thesis的直接检验
  D. 纯板块择时:  同C的条件, 但触发时买"银行等权整个板块"(不选股) — 剥离选股贡献, 纯择时
  E. 横盘gate:    |银行等权前3月收益|<band → 满仓Q5, 否则持币 — "横盘后突破"的直接检验

银行等权指数 = 自建(PIT白名单逐月等权, stock_ret[银行].mean(axis=1)), 不下载官方指数:
成分PIT一致+与策略同源。策略窗口 2020-12~2026-07(对齐04/06/07/08, 原2020-01), 月频, 持仓月扣15bp。
输出: outputs/sector_gate_check.csv + 对照打印
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import common

OUT = '/Users/lichanghaodemac/quant_strategy/projects/bank_factor_workflow/outputs'
CASH = 0.02 / 12          # 空仓月持币(货基2%年化)
COST = 0.0015             # 持仓月固定成本(与04/common同口径)

P = common.load_panel()
fwd1, me_sorted = P['fwd1'], P['me_sorted']
# ★ 2026-08-15: 起跑对齐04/06/07/08的2020-12(面板2017-01的第47个月), 原为2020-01
qstart = pd.Timestamp('2020-12-01'); qi = next(i for i, m in enumerate(me_sorted) if m >= qstart)
monthly = [m for i, m in enumerate(me_sorted) if i >= qi]
SC4 = common.score_set(P, common.F4, common.NEG4, common.roll_icir(P, common.F4, fwd1))

# 自建银行等权月收益(PIT白名单, 逐月等权可用成分)
bank_eq = P['stock_ret'][list(P['stks'])].mean(axis=1)
mkt1 = P['mkt1']

def q5_of(m):
    s = SC4[m].dropna()
    if s.nunique() < 5: return None
    try: q = pd.qcut(s.rank(method='first'), 5, labels=False)
    except Exception: return None
    return s[q == 4].index

def prev3_sum(series, m, i):
    idx = me_sorted[max(0, i - 3):i]
    return series.reindex(idx).sum()

def run_variant(mode, thr=None):
    """mode: 'none' | 'stock' | 'sector' | 'sector_eq' | 'range'; 返回(月收益Series, 持仓mask)
    ★ 2026-08-15 修复: 原版用monthly的局部序号i去切全局me_sorted → "前3月"实际回看到2017-18年,
      B/C/D/E全部中毒(A无gate不受影响)。改为全局索引ii。"""
    rets, held_flags = [], []
    for m in monthly:
        if m not in fwd1.index: continue
        ii = me_sorted.index(m)                    # ★ 全局索引: gate回看窗=真正的"前3个月"
        q5 = q5_of(m)
        if q5 is None: continue
        if mode == 'none':
            held, members = True, q5
        elif mode == 'stock':                      # 04的实现: 个股自己的前3月
            p3 = me_sorted[max(0, ii-3):ii]
            keep = [o for o in q5 if o in P['stock_ret'].columns
                    and P['stock_ret'].loc[p3, o].notna().all()
                    and P['stock_ret'].loc[p3, o].sum() < thr]
            members = [o for o in keep if o in fwd1.columns and pd.notna(fwd1.loc[m, o])]
            held = len(members) >= 3
        elif mode in ('sector', 'sector_eq'):      # thesis: 板块整体前3月
            held = prev3_sum(bank_eq, m, ii) < thr
            members = [o for o in q5 if o in fwd1.columns and pd.notna(fwd1.loc[m, o])]
        elif mode == 'range':                      # 横盘: |前3月| < band
            held = abs(prev3_sum(bank_eq, m, ii)) < thr
            members = [o for o in q5 if o in fwd1.columns and pd.notna(fwd1.loc[m, o])]
        if held:
            if mode == 'sector_eq':
                r = bank_eq.get(m, np.nan)
            else:
                vals = [fwd1.loc[m, o] for o in members]
                r = np.mean(vals) if len(vals) >= 2 else np.nan
            if pd.isna(r): r, held = CASH, False
            else: r = r - COST
        else:
            r = CASH
        rets.append(r); held_flags.append(held)
    return pd.Series(rets, index=pd.DatetimeIndex(monthly[:len(rets)])), pd.Series(held_flags, index=pd.DatetimeIndex(monthly[:len(rets)]))

def met(r, held):
    ann = r.mean() * 12; sh = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else 0
    cum = (1 + r).cumprod(); dd = (cum / cum.cummax() - 1).min()
    eng = r[held].mean() * 12 if held.any() else np.nan     # 持仓月引擎年化
    return {'年化': round(ann, 4), '夏普': round(sh, 2), '最大回撤': round(dd, 4),
            '持仓占比': round(held.mean(), 2), '持仓月引擎年化': round(eng, 4), '月数': len(r)}

rows = []
for name, mode, thr in [
    ('A 无过滤(一直Q5)', 'none', None),
    ('B 个股级gate<-1%(04版)', 'stock', -0.01),
    ('C 板块gate<-5%', 'sector', -0.05), ('C 板块gate<-3%', 'sector', -0.03),
    ('C 板块gate<-1%', 'sector', -0.01), ('C 板块gate<0%', 'sector', 0.0),
    ('C 板块gate<+2%', 'sector', 0.02),
    ('E 横盘|前3月|<1%', 'range', 0.01), ('E 横盘|前3月|<2%', 'range', 0.02),
    ('D 纯板块择时<-3%(不选股)', 'sector_eq', -0.03), ('D 纯板块择时<-1%(不选股)', 'sector_eq', -0.01),
]:
    r, held = run_variant(mode, thr)
    rows.append({'变体': name, **met(r, held)})
res = pd.DataFrame(rows)
res.to_csv(f'{OUT}/sector_gate_check.csv', index=False, encoding='utf-8-sig')

win = bank_eq.reindex(monthly).dropna(); mktw = mkt1.reindex(monthly).dropna()
print(f'窗口: {str(monthly[0])[:10]} ~ {str(monthly[-1])[:10]} 月频, 空仓=持币2%/年, 持仓月扣15bp', flush=True)
print(f'参照: 银行等权(持有不动) {win.mean()*12:.2%}/夏普{win.mean()/win.std()*np.sqrt(12):.2f} | '
      f'全市场等权(纸面) {mktw.mean()*12:.2%} | 现金2%', flush=True)
print('\n=== 板块级 vs 个股级 gate 对质(4F滚动ICIR月频Q5) ===', flush=True)
print(res.to_string(index=False), flush=True)
print('\n解读: C/D/E是"银行超跌/横盘后有一波行情"的直接检验; 对比A看gate是加分还是减分', flush=True)
