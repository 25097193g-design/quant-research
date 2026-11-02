# 订单簿数据结构讲解页: 从 book_train.parquet 出图 + 组装自包含 HTML
# 用法: /opt/homebrew/Caskroom/miniconda/base/envs/rqsdk/bin/python build_book_explainer.py
# 产出: outputs/book_explainer.html (图片base64内嵌, 单文件可发人) + outputs/book_*.png
import base64
import glob
from datetime import date

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE = '/Users/lichanghaodemac/Downloads/optiver-realized-volatility-prediction'
OUT = '/Users/lichanghaodemac/quant_strategy/projects/optiver_rv_2021/outputs'
STOCK, TID = 0, 5

C_BID, C_ASK = '#2e7d32', '#c62828'
C_WAP, C_MID = '#1565c0', '#9e9e9e'

df = pd.read_parquet(glob.glob(f'{BASE}/book_train.parquet/stock_id={STOCK}/*.parquet')[0])
sub = df[df.time_id == TID].sort_values('seconds_in_bucket').reset_index(drop=True)
snap0 = sub.iloc[0]

wap = (sub.bid_price1 * sub.ask_size1 + sub.ask_price1 * sub.bid_size1) / (sub.bid_size1 + sub.ask_size1)
lr = np.log(wap).diff()
rv = float(np.sqrt((lr ** 2).sum()))


def save(fig, name):
    fig.savefig(f'{OUT}/{name}', dpi=150, bbox_inches='tight')
    plt.close(fig)
    return name


def fig_ladder():
    """秒0那张快照: 簿的阶梯(4个价位, 挂单量) + 价格轴上标 mid/WAP"""
    fig, (ax, axp) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={'width_ratios': [1, 1.6]})

    levels = [
        ('ask_2 (卖二)', snap0.ask_price2, snap0.ask_size2, C_ASK),
        ('ask_1 (卖一)', snap0.ask_price1, snap0.ask_size1, C_ASK),
        ('bid_1 (买一)', snap0.bid_price1, snap0.bid_size1, C_BID),
        ('bid_2 (买二)', snap0.bid_price2, snap0.bid_size2, C_BID),
    ]
    y = np.arange(len(levels))[::-1]
    for (name, price, size, color), yi in zip(levels, y):
        ax.barh(yi, size, color=color, alpha=.85, height=.62)
        ax.text(size + 4, yi, f'{size:.0f} 股', va='center', fontsize=10)
        ax.text(-4, yi, name, va='center', ha='right', fontsize=10)
    ax.set_xlim(-115, 265)
    ax.set_ylim(-.7, 3.7)
    ax.set_xlabel('挂单量 (股)')
    ax.set_title(f'簿的阶梯 — stock_{STOCK}, time_id={TID}, 第 0 秒快照', fontsize=12)
    ax.grid(axis='x', alpha=.25)

    for (name, price, size, color), yi in zip(levels, y):
        axp.hlines(yi, 1.0010, price, color=color, lw=2.5, alpha=.8)
        axp.plot(price, yi, 'o', color=color, ms=7)
        axp.text(price, yi + .22, f'{price:.6f}', ha='center', fontsize=9, color=color)
    mid, spread = (snap0.ask_price1 + snap0.bid_price1) / 2, snap0.ask_price1 - snap0.bid_price1
    w0 = (snap0.bid_price1 * snap0.ask_size1 + snap0.ask_price1 * snap0.bid_size1) / (snap0.bid_size1 + snap0.ask_size1)
    axp.axvline(mid, color=C_MID, ls='--', lw=1.4)
    axp.axvline(w0, color=C_WAP, ls='-', lw=2)
    axp.text(mid, 3.62, ' mid', color=C_MID, fontsize=10, ha='left')
    axp.text(w0, 3.28, ' WAP', color=C_WAP, fontsize=10, ha='left')
    axp.axvspan(snap0.bid_price1, snap0.ask_price1, color='#f9a825', alpha=.18)
    axp.annotate(f'spread = {spread:.6f}\n(≈{spread / snap0.bid_price1 * 1e4:.1f} bp)',
                 xy=(mid, 1.5), xytext=(1.00190, 2.6), fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='#8d6e00'))
    axp.text(w0, -.65, f'WAP={w0:.6f}\n被226股卖单压向买一', color=C_WAP, fontsize=10, ha='left')
    axp.set_xlim(1.0010, 1.0026)
    axp.set_ylim(-1.1, 4.0)
    axp.set_yticks([])
    axp.set_xlabel('价格')
    axp.set_title('同一快照的价格轴视角', fontsize=12)
    axp.grid(axis='x', alpha=.25)
    return save(fig, 'book_ladder.png')


def fig_evolution():
    """前40秒: 价格基本不动, 量在动 — 簿的更新大多数是排队变化"""
    e = sub[sub.seconds_in_bucket <= 40]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True,
                                 gridspec_kw={'hspace': .28})
    for col, color, lbl in [('bid_price1', C_BID, 'bid_1'), ('ask_price1', C_ASK, 'ask_1'),
                            ('bid_price2', '#81c784', 'bid_2'), ('ask_price2', '#ef9a9a', 'ask_2')]:
        a1.step(e.seconds_in_bucket, e[col], where='post', color=color, lw=1.6, label=lbl)
    a1.set_ylabel('价格')
    a1.set_title(f'前 40 秒簿的演化 — 价格阶梯几乎不动, 簿在"排队层"刷新', fontsize=12)
    a1.legend(ncol=4, fontsize=9, loc='right')
    a1.grid(alpha=.25)

    a2.step(e.seconds_in_bucket, e.bid_size1, where='post', color=C_BID, lw=1.8, label='bid_size1')
    a2.step(e.seconds_in_bucket, e.ask_size1, where='post', color=C_ASK, lw=1.8, label='ask_size1')
    a2.annotate('第1秒: 卖一挂单 226→100 股\n(被吃掉/撤单126股), 价格未动',
                xy=(1.2, 110), xytext=(8, 205), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='#555'))
    a2.set_xlabel('seconds_in_bucket (窗口内秒数)')
    a2.set_ylabel('最优档挂单量 (股)')
    a2.legend(ncol=2, fontsize=9)
    a2.grid(alpha=.25)
    return save(fig, 'book_evolution_early.png')


def fig_wap_rv():
    """整桶: WAP路径 + 累计RV长到最终值"""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True,
                                 gridspec_kw={'hspace': .3})
    a1.plot(sub.seconds_in_bucket, wap, color=C_WAP, lw=1.1)
    a1.set_ylabel('WAP')
    a1.set_title(f'整桶 WAP 路径 (302 张快照) 与 RV 的累计 — stock_{STOCK}, time_id={TID}', fontsize=12)
    a1.grid(alpha=.25)

    cum = np.sqrt((lr ** 2).fillna(0).cumsum())
    a2.plot(sub.seconds_in_bucket, cum, color='#6a1b9a', lw=1.6)
    a2.axhline(rv, color='#6a1b9a', ls='--', lw=1, alpha=.6)
    a2.text(302, rv, f'  本桶 RV = {rv:.6f}\n  (=10分钟价格抖动≈{rv * 1e4:.0f} bp)',
            fontsize=10, va='center', color='#6a1b9a')
    a2.set_xlabel('seconds_in_bucket')
    a2.set_ylabel(r'累计 $\sqrt{\sum r_t^2}$')
    a2.grid(alpha=.25)
    return save(fig, 'book_wap_rv.png')


def fig_sampling():
    """快照是事件驱动: 间隔分布"""
    dts = sub.seconds_in_bucket.diff().dropna()
    fig, ax = plt.subplots(figsize=(7, 2.6))
    vc = dts.value_counts().sort_index()
    ax.bar(vc.index[:10], vc.values[:10], color='#37474f', alpha=.85)
    for x, v in list(vc.items())[:4]:
        ax.text(x, v + 2, str(v), ha='center', fontsize=9)
    ax.set_xlabel('相邻快照的间隔 (秒)')
    ax.set_ylabel('次数')
    ax.set_title(f'快照间隔分布 — 不是每秒一行, 簿没变化就不存 (本桶共 {len(sub)} 张快照)', fontsize=11)
    ax.grid(axis='y', alpha=.25)
    return save(fig, 'book_sampling.png')


def b64(name):
    return base64.b64encode(open(f'{OUT}/{name}', 'rb').read()).decode()


rows10 = sub.head(10)[['seconds_in_bucket', 'bid_price1', 'ask_price1', 'bid_size1', 'ask_size1',
                       'bid_price2', 'ask_price2', 'bid_size2', 'ask_size2']]
tbl = rows10.to_html(index=False, border=0, float_format=lambda x: f'{x:.6f}',
                     classes='data')

rows300 = sub[(sub.seconds_in_bucket >= 301) & (sub.seconds_in_bucket <= 305)][
    ['seconds_in_bucket', 'bid_price1', 'ask_price1', 'bid_size1', 'ask_size1']]
tbl300 = rows300.to_html(index=False, border=0, float_format=lambda x: f'{x:.6f}', classes='data')

n_by_tid = df.groupby('time_id').size()
html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>读懂订单簿 — Optiver RV 2021 数据结构</title>
<style>
  :root {{ --ink:#1a1a1a; --sub:#5f6368; --line:#e0e0e0; --accent:#1565c0; --bg:#fafafa; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,"PingFang SC","Hiragino Sans GB",sans-serif; color:var(--ink);
         margin:0; background:var(--bg); line-height:1.75; }}
  .wrap {{ max-width:960px; margin:0 auto; padding:48px 28px 80px; }}
  header {{ border-bottom:3px solid var(--accent); padding-bottom:18px; margin-bottom:36px; }}
  h1 {{ font-size:30px; margin:0 0 6px; }}
  .meta {{ color:var(--sub); font-size:13.5px; }}
  h2 {{ font-size:21px; margin-top:52px; border-left:4px solid var(--accent); padding-left:12px; }}
  h3 {{ font-size:16.5px; margin-top:28px; }}
  code, pre {{ font-family:"SF Mono",Menlo,Consolas,monospace; font-size:13px; }}
  code {{ background:#eef2f7; padding:1px 6px; border-radius:4px; }}
  pre {{ background:#0f1720; color:#d8e2ee; padding:16px 18px; border-radius:10px; overflow-x:auto; line-height:1.55; }}
  img {{ width:100%; border:1px solid var(--line); border-radius:10px; background:#fff; margin:14px 0 6px; }}
  .cap {{ color:var(--sub); font-size:12.5px; margin-bottom:22px; }}
  table.data {{ border-collapse:collapse; font-family:Menlo,Consolas,monospace; font-size:12.5px; width:100%; }}
  table.data th {{ background:#eef2f7; font-weight:600; }}
  table.data th, table.data td {{ padding:5px 9px; border-bottom:1px solid var(--line); text-align:right; }}
  table.data th:first-child, table.data td:first-child {{ text-align:left; }}
  .note {{ background:#fff8e1; border-left:4px solid #f9a825; padding:10px 16px; border-radius:0 8px 8px 0; font-size:14px; margin:18px 0; }}
  .key {{ background:#e8f0fe; border-left:4px solid var(--accent); padding:10px 16px; border-radius:0 8px 8px 0; font-size:14px; margin:18px 0; }}
  .ladder-text {{ font-family:Menlo,Consolas,monospace; font-size:13.5px; background:#fff; border:1px solid var(--line);
                 border-radius:10px; padding:14px 20px; display:inline-block; line-height:1.9; }}
  .formula {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:14px 20px; font-size:14.5px; margin:14px 0; }}
  ul {{ padding-left:22px; }} li {{ margin:6px 0; }}
  footer {{ margin-top:70px; color:var(--sub); font-size:12.5px; border-top:1px solid var(--line); padding-top:14px; }}
</style>
</head>
<body><div class="wrap">

<header>
<h1>读懂订单簿：Optiver 波动率预测赛的数据长什么样</h1>
<div class="meta">Kaggle Optiver Realized Volatility Prediction (2021) · 数据 book_train.parquet ·
真实示例 stock_0 / time_id_5 · 生成于 {date.today()} · 脚本 build_book_explainer.py</div>
</header>

<h2>1 · 文件在磁盘上的组织</h2>
<p><code>book_train.parquet</code> 不是单个文件，是<strong>按股票分区的目录树</strong>（hive 分区格式），112 只匿名股票（编号 0~126，缺 15 个——缺的只出现在测试集）：</p>
<pre>book_train.parquet/
├── stock_id=0/c439ef22….parquet   ← 股票0的全部簿数据(一个文件)
├── stock_id=1/…
├── … 共 112 个目录</pre>
<p><code>stock_id=0</code> 这一个 parquet 实测：<strong>917,553 行、3,830 个 time_id</strong>、10 列纯平表。每个 time_id 是一个 10 分钟窗口，窗口内快照数均值 240、中位 229、范围 73~549。</p>

<h2>2 · 一行数据 = 一张"拍照"</h2>
<p>订单簿（limit order book）是<strong>还没成交的限价单排成的队</strong>：任何时刻有人挂买单（"我出 1.001422 买"）、有人挂卖单（"我要 1.002301 才卖"），全部排队情况就是簿。比赛给的是 <strong>L2 深度</strong>——买卖各 2 档：</p>
<table class="data">
<tr><th style="text-align:left">列</th><th style="text-align:left">含义</th></tr>
<tr><td style="text-align:left"><code>time_id</code></td><td style="text-align:left">10 分钟窗口编号；同一 time_id 跨股票 = 同一市场时段</td></tr>
<tr><td style="text-align:left"><code>seconds_in_bucket</code></td><td style="text-align:left">窗口内第几秒（0~599）</td></tr>
<tr><td style="text-align:left"><code>bid_price1 / ask_price1</code></td><td style="text-align:left">最优买价 / 最优卖价（买方最贵 / 卖方最便宜的一档）</td></tr>
<tr><td style="text-align:left"><code>bid_size1 / ask_size1</code></td><td style="text-align:left">该价位上的挂单量（股数）</td></tr>
<tr><td style="text-align:left"><code>bid_price2 / ask_size2 …</code></td><td style="text-align:left">第二档（次优价位）的价格与量</td></tr>
</table>
<div class="key"><strong>一行 = 某一秒末整本簿的一张快照</strong>——不是一笔委托、也不是一笔成交。所有特征工程都从"把连续快照读成动态演化"开始。</div>

<h2>3 · 真实例子：第 0 秒这本簿长什么样</h2>
<p>stock_0 / time_id_5 / 第 0 秒快照，写成阶梯（价格越靠内越优）：</p>
<div class="ladder-text">
卖二档: <span style="color:#c62828">1.002353 卖 100 股</span><br>
卖一档: <span style="color:#c62828">1.002301 卖 226 股</span><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;——— 中间是价差 spread = 0.000879 ———<br>
买一档: <span style="color:#2e7d32">1.001422 买&nbsp;&nbsp;3 股</span><br>
买二档: <span style="color:#2e7d32">1.001370 买&nbsp;&nbsp;2 股</span>
</div>
<img src="data:image/png;base64,{b64(fig_ladder())}">
<div class="cap">左：四个价位的挂单量（阶梯）。右：同一快照放到价格轴上——bid 永远低于 ask（否则早成交了）；WAP 被 226 股卖单压得几乎贴住买一，比 mid 更贴近真实均衡价。</div>

<h2>4 · 逐行看簿的演化（前 10 行原文）</h2>
{tbl}
<ul>
<li><strong>第 0→1 秒发生了什么</strong>：ask_size1 从 226 变 100——最优卖价上被吃掉/撤掉 126 股，<em>价格没动，是量在动</em>。簿的大部分更新是这种排队变化，价格跳变是少数事件。</li>
<li><strong>秒 2、3、4 没有行</strong>——那几秒簿没变化就不存。快照是<strong>事件驱动</strong>的，不是每秒一行。</li>
</ul>
<img src="data:image/png;base64,{b64(fig_evolution())}">
<div class="cap">前 40 秒：最优价阶梯几乎不动（上），挂单量持续刷新（下）——"簿的脉搏"主要在量的层面。</div>

<h3>同一桶，第 300~305 秒：簿已经完全换了一轮</h3>
{tbl300}
<p>对比开头：买一从 1.001422 抬到 1.003698（≈+23 bp），量价结构整体换血。把这 302 张快照的"价格"连起来，就是这个 10 分钟的走势线。</p>

<h2>5 · 从簿到价格：spread / mid / WAP（真实数字）</h2>
<div class="formula">
<strong>spread</strong> = ask₁ − bid₁ = 1.002301 − 1.001422 = <strong>0.000879</strong>（≈8.8 bp，簿的温度计）<br><br>
<strong>mid</strong> = (ask₁ + bid₁) / 2 = <strong>1.001862</strong>（最朴素的价格估计，但忽略两边挂单量的悬殊）<br><br>
<strong>WAP</strong> = (bid₁×ask_s₁ + ask₁×bid_s₁) / (bid_s₁ + ask_s₁)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= (1.001422×226 + 1.002301×3) / 229 = <strong>1.001434</strong>
</div>
<p>买方只挂 3 股、卖方挂 226 股 → 真实均衡价应贴近买一。WAP 按<strong>对面量</strong>加权，被那 226 股卖单拉到 1.001434（几乎贴住买一 1.001422），比 mid 合理得多。且 WAP 对挂单量敏感：第 1 秒卖单 226→100 后 WAP 从 1.001434 漂到 1.001448——<strong>簿深度的变化直接进入价格信号</strong>，这正是簿数据比成交价信息更丰富的地方。</p>

<h2>6 · 从簿到比赛目标：WAP 路径 → RV</h2>
<p>把整桶 302 张快照的 WAP 连成路径，取相邻对数差分 r<sub>t</sub> = ln(WAP<sub>t</sub>/WAP<sub>t−1</sub>)，则：</p>
<div class="formula" style="text-align:center"><strong>RV = sqrt( Σ r<sub>t</sub>² )</strong>（不减均值）= <strong>{rv:.6f}</strong> —— 本桶 10 分钟价格抖动 ≈ {rv*1e4:.0f} bp</div>
<p>比赛的 <code>target</code> 就是<strong>下一个 10 分钟桶的同一个数</strong>。整套建模 = 快照 → WAP 路径 → RV → 预测下一桶 RV（评分 RMSPE，相对误差）。</p>
<img src="data:image/png;base64,{b64(fig_wap_rv())}">
<div class="cap">上：整桶 WAP 路径。下：累计 sqrt(Σr²) 一路长到最终 RV——RV 是"路径长出来的量"，不是终起点收益。</div>

<h2>7 · 两个必须知道的数据事实</h2>
<img src="data:image/png;base64,{b64(fig_sampling())}">
<div class="cap">stock_0/time_id_5 的快照间隔：1 秒 183 次、2 秒 58 次、3 秒 25 次……簿安静时跳秒。</div>
<ul>
<li><strong>事件驱动采样</strong>：假设规则采样的"稳健化"会适得其反（本项目 16_reorder_v2 实测：5 快照均值指纹把相关性 0.650 毁到 0.525——桶末状态被抹花）。单张末快照才是真·最新状态。</li>
<li><strong>薄簿退化</strong>：stock_0 这类活跃票第二档 100% 非空；薄簿会出现二档缺失、甚至两档同价的退化快照，做 spread/depth 特征必须处理 NaN，不能静默填 0。</li>
</ul>

<h2>8 · 为什么这份数据值钱：三个结构性质</h2>
<ul>
<li><strong>波动聚集</strong>：下一桶 RV ≈ 本桶 RV（naive 基线 RMSPE≈0.34，pooled Pearson 0.874）→ 上一桶 RV 是最重要的单一特征。</li>
<li><strong>市场时钟共振</strong>：同一 time_id 跨股票 = 同一市场时段，市场级风暴全员共振 → 同 time_id 其他股票的 vol 是最强特征族之一（裸证据：市场层级 corr 0.923）。</li>
<li><strong>股票异质</strong>：各股 RV 中位数差 6.7 倍 → per-stock encoding / stock_id 类别特征是不可缺的杠杆（单加 stock_id 即 −0.95pp）。</li>
</ul>
<div class="note">延伸（面试加分）：RV 是对数价格路径的离散<strong>二次变差</strong>，采样加密 → 收敛积分方差；但采样过密时 bid-ask bounce 贡献假跳（微观结构噪声），需要 noise-robust 估计器（two-scales RV / realized kernels）。比赛 ~1s 快照已避开最毒区间。</div>

<footer>
数据: ~/Downloads/optiver-realized-volatility-prediction (Kaggle 官方, 2.5GB) ·
比赛: kaggle.com/competitions/optiver-realized-volatility-prediction ·
图/页生成: projects/optiver_rv_2021/build_book_explainer.py ·
上游口径: common.py (WAP/RV 与比赛官方定义一致)
</footer>
</div></body></html>"""

path = f'{OUT}/book_explainer.html'
open(path, 'w').write(html)
print(f'OK {path} ({len(html)/1024:.0f} KB, RV={rv:.6f})')
