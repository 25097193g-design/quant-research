"""[ETF板块轮动] 相对动量 Top-K 月末轮动回测 (板块成分股代理收益)
============================================================
标的 : A股ETF净值(rqdatac教育版)不可得, 用 31板块等权日收益代理
       sector_return_matrix.parquet (PIT月度重构, 无未来函数) — 这反而是更干净的标的
信号 : 相对动量 RS = 板块过去N日累计收益 - 截面均值
规则 : 每月末选 RS 最高的 K 个板块等权持有 (可选: 跨簇约束 / 趋势过滤)
评判 : vs 等权全持有31板块基准  ← 关键! 不是standalone夏普
       (空仓≠0收益, 银行轮动教训; 高相关0.69说明全持有基准很难打败, 抬高门槛)
成本 : 换手双边 COST 扣除; 信号 shift(1) 确保t月末信号t+1才生效(无未来函数)

用法 : cd projects/etf_sector_rotation && conda run -n rqsdk --no-capture-output python rotation_backtest.py
输出 : rotation_nav.csv(净值) / rotation_stats.csv(统计) / rotation_scan.csv(参数扫描) / rotation_nav.png(图)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无显示环境也能存图
import matplotlib.pyplot as plt
# 中文(本机已识别CJK字体, 设了立即生效)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang HK', 'Heiti TC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
SEC = os.path.join(HERE, '..', 'week1_sector_structure')

# ============ 参数(可改) ============
LOOKBACK = 60            # 动量回看窗口(交易日) ~3个月
K = 4                    # 持仓板块数
COST = 0.004             # 换手双边成本(0.4%)
CROSS_CLUSTER = False   # True=强制Top-K来自不同簇(用5簇聚类结果)
TREND_FILTER = False     # True=只买过去N日累计收益>0的板块(绝对动量趋势过滤, 避免接刀)
MODE = 'reversal'        # 'momentum'=追强势板块(经典美股思路) | 'reversal'=买超跌板块(A股动量是反转的, MOM负IC)

# ============ 数据 ============
rets = pd.read_parquet(os.path.join(SEC, 'sector_return_matrix.parquet'))
mem = pd.read_csv(os.path.join(SEC, 'cluster_membership.csv'))
cluster_map = dict(zip(mem['sector'], mem['capw_full']))  # 板块 -> 簇(1-5, 用capw_full)
print(f'板块收益代理: {rets.shape} | {rets.index[0].date()}~{rets.index[-1].date()} | {len(rets.columns)}个板块')

# ============ 信号: 相对动量 RS =过去N日累计收益(含t), log-sum加速
log_cum = np.log1p(rets).rolling(LOOKBACK).sum()
cum = np.expm1(log_cum)                       # 绝对累计动量
rs = cum.sub(cum.mean(axis=1), axis=0)        # 减截面均值(=31板块等权代理); 注: 排序等价于绝对动量

# ============ 月末换仓日(每月最后交易日) ============
last_td = rets.index.to_series().groupby(rets.index.to_period('M')).max()
reb_dates = pd.DatetimeIndex(last_td.values)

def select_topk(row, k, cum_row, cross_cluster=False, trend=False, cmap=None, ascending=False):
    """选K个板块. ascending=False=动量追强 / True=反转买超跌; cross_cluster=贪心强制不同簇; trend=剔除弱势"""
    s = row.sort_values(ascending=ascending)
    if trend:
        s = s[cum_row.reindex(s.index) > 0]    # 只留绝对动量为正
    if not cross_cluster or cmap is None:
        return list(s.head(k).index)
    picked, used = [], set()
    for sec in s.index:
        c = cmap.get(sec)
        if c in used:
            continue
        picked.append(sec); used.add(c)
        if len(picked) >= k:
            break
    return picked

# ============ 构造持仓 + 回测 ============
holdings = pd.DataFrame(0.0, index=reb_dates, columns=rets.columns)
held_log = []
for t in reb_dates:
    if t not in rs.index:
        continue
    sel = select_topk(rs.loc[t], K, cum.loc[t], CROSS_CLUSTER, TREND_FILTER, cluster_map,
                      ascending=(MODE == 'reversal'))
    if sel:
        holdings.loc[t, sel] = 1.0 / len(sel)   # 等权(若趋势过滤后不足K个, 仓位<1)
    held_log.append((t, sel, [cluster_map.get(s) for s in sel]))

# forward fill到日频, shift(1): t月末信号 -> t+1才生效 (无未来函数)
daily_w = holdings.reindex(rets.index).ffill().shift(1).fillna(0)
port_ret = (daily_w * rets).sum(axis=1)
turnover = daily_w.diff().abs().sum(axis=1) / 2     # 单边换手率
port_net = port_ret - turnover * COST

# ============ 基准: 等权全持有31板块(月末再平衡) ============
bench_hold = pd.DataFrame(1.0 / len(rets.columns), index=reb_dates, columns=rets.columns)
daily_wb = bench_hold.reindex(rets.index).ffill().shift(1).fillna(0)
bench_ret = (daily_wb * rets).sum(axis=1)
bench_turn = daily_wb.diff().abs().sum(axis=1) / 2
bench_net = bench_ret - bench_turn * COST

# ============ 统计 ============
def stats(r, name, turn):
    r = r.dropna()
    n = len(r)
    ann = (1 + r).prod() ** (252 / n) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else np.nan
    cumv = (1 + r).cumprod()
    mdd = (cumv / cumv.cummax() - 1).min()
    return dict(策略=name, 年化=ann, 波动=vol, 夏普=sharpe,
                最大回撤=mdd, 日均换手=turn.loc[r.index].mean())

st_port = stats(port_net, '轮动', turnover)
st_bench = stats(bench_net, '基准', bench_turn)
st_df = pd.DataFrame([st_port, st_bench])
print('\n=== 回测统计 ===')
print(st_df.to_string(index=False))

# 超额(信息比率)
excess = port_net - bench_net
ir = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else np.nan
print(f'\n超额(轮动-基准): 年化{(1+excess).prod()**(252/len(excess))-1:.4f} | '
      f'信息比率IR={ir:.3f} | 累计超额净值{(1+excess).prod()-1:+.4f}')

# 持仓分簇分布(看是否集中)
held_df = pd.DataFrame(held_log, columns=['date', 'sectors', 'clusters'])
clust_count = pd.Series([c for cls in held_df['clusters'] for c in cls]).value_counts().sort_index()
print('\n=== 持仓分簇频次(整段, 看是否集中某簇) ===')
print(clust_count.to_string())

# ============ 参数扫描(稳健性/过拟合检查) ============
print('\n=== 参数扫描 (动量 vs 反转 × lookback × K; 反转是否系统性转正=核心验证) ===')
scan_rows = []
for mode_asc, mode_name in [(False, '动量'), (True, '反转')]:
    for L in [20, 60, 120, 200]:
        for k in [2, 4, 6]:
            cs = np.expm1(np.log1p(rets).rolling(L).sum())
            rss = cs.sub(cs.mean(axis=1), axis=0)
            h = pd.DataFrame(0.0, index=reb_dates, columns=rets.columns)
            for t in reb_dates:
                if t not in rss.index:
                    continue
                sel = list(rss.loc[t].sort_values(ascending=mode_asc).head(k).index)
                h.loc[t, sel] = 1.0 / k
            dw = h.reindex(rets.index).ffill().shift(1).fillna(0)
            pr = (dw * rets).sum(axis=1) - dw.diff().abs().sum(axis=1) / 2 * COST
            s = stats(pr, f'{mode_name}L{L}K{k}', dw.diff().abs().sum(axis=1) / 2)
            s.update({'mode': mode_name, 'lookback': L, 'K': k})
            scan_rows.append(s)
scan_df = pd.DataFrame(scan_rows)
print(scan_df[['mode', 'lookback', 'K', '年化', '夏普', '最大回撤', '日均换手']].to_string(index=False))

# ============ 输出 ============
nav = pd.DataFrame({
    '轮动': (1 + port_net).cumprod(),
    '基准': (1 + bench_net).cumprod(),
    '超额': (1 + excess).cumprod(),
})
nav.to_csv(os.path.join(HERE, 'rotation_nav.csv'))
st_df.to_csv(os.path.join(HERE, 'rotation_stats.csv'), index=False)
scan_df.to_csv(os.path.join(HERE, 'rotation_scan.csv'), index=False)

# ============ 图 ============
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                         gridspec_kw={'height_ratios': [3, 1]})
axes[0].plot(nav.index, nav['轮动'], label=f'{MODE}轮动(L={LOOKBACK},K={K})', lw=1.6)
axes[0].plot(nav.index, nav['基准'], label='等权全持有31板块基准', lw=1.6, alpha=0.7)
axes[0].set_ylabel('净值 (起始=1)'); axes[0].legend(loc='upper left'); axes[0].grid(alpha=0.3)
axes[0].set_title(f'板块{MODE}轮动 vs 等权基准 | 双边成本{COST:.1%} | 跨簇={CROSS_CLUSTER} 趋势过滤={TREND_FILTER}')
axes[1].plot(nav.index, nav['超额'], color='darkred', lw=1.2)
axes[1].axhline(1, color='gray', ls='--', lw=0.8)
axes[1].set_ylabel('超额净值'); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(HERE, 'rotation_nav.png'), dpi=130)
print(f'\n输出完成: rotation_nav.csv / rotation_stats.csv / rotation_scan.csv / rotation_nav.png')
