"""Generate 15_donor_confounding.ipynb: donor 同源诊断 —— Exclude XLK LOO

§5.2 SCM 选 XLK 35%(pre 期 AI 行情)。本 notebook 剔除 XLK 重跑 SCM,
定量 AI/科技板块对 gap 的贡献(donor 同源诊断)。回应外部评审"剔除 AI 成分"建议。
比 AI proxy 正交化(NVDA/AI ETF, yfinance 限流不可得)更干净直接。
只用 行业 donor + 标普, 不依赖海外/AI proxy。

风格 A。
"""
import json
import os

cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t})


def code(t):
    cells.append({"cell_type": "code", "execution_count": None,
                  "metadata": {}, "outputs": [], "source": t})


# ============ Cell 1: 标题 + 故事 ============
md('''# nb15: donor 同源诊断 —— Exclude XLK LOO(AI 板块贡献)

> **承接 §5.2/§5.12**: SCM 选 XLK 35%(pre 期 AI 行情, 科技与标普日收益相关 0.91)。但这是 **AI 共同因子把 XLK 和标普绑一起**, 非"科技代表标普"。本 notebook **剔除 XLK 重跑 SCM**, 定量 AI/科技板块对 gap 的贡献。

## 诊断逻辑

pre 期 AI 狂牛(标普 5000→5700, AI 七姐妹驱动)→ SCM 给 XLK 高权重(走势最像标普)。
但 post 期(4/9 暂停后)**科技领涨反弹 +21%**(AI 独立行情)→ 含 XLK 的合成捕捉科技行情 → gap 小(+0.68%); 剔除 XLK → 合成错过科技 → gap 大(+7.51%)。

**Exclude XLK LOO 定量这**: XLK 贡献 −6.83pp, 即 AI 板块 donor 同源把 gap 低估了 6.83pp。

## 为什么不做 AI proxy 正交化(NVDA/AI ETF 残差)
1. **Exclude XLK LOO 已直接定量** AI 板块贡献(−6.83pp), 比 NVDA 残差正交化更干净
2. yfinance 限流(NVDA/BOTZ/ARKK 全失败), AI proxy 不可得
3. AI 正交化 ad-hoc(同 nb10 VIX 正交化批评); Exclude XLK 是标准 LOO 的特例''')

# ============ Cell 2: Exclude XLK LOO ============
code('''import sys
sys.path.insert(0, '.')  # 相对 cwd, 复现: 先 cd notebooks/paper_tariff
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
%matplotlib inline
from paper_lib import load_tariff_data, fit_scm_pipeline, pre_trend_test

panels = load_tariff_data()
gspc = panels['us_index']['^GSPC']; sector = panels['sector_etf']
EV='2025-04-02'; PRE='2024-06-01'; SHORT='2025-04-08'; LONG='2025-06-30'

# 完整 SCM(含 XLK 35%) vs 剔除 XLK
r_full_s = fit_scm_pipeline(gspc, sector, EV, PRE, SHORT); r_full_l = fit_scm_pipeline(gspc, sector, EV, PRE, LONG)
sector_noxlk = sector.drop(columns=['XLK'])
r_noxlk_s = fit_scm_pipeline(gspc, sector_noxlk, EV, PRE, SHORT); r_noxlk_l = fit_scm_pipeline(gspc, sector_noxlk, EV, PRE, LONG)

print('=== Exclude XLK LOO: 完整 vs 剔除 XLK ===')
print(f"{'设定':<16}{'短gap':>10}{'长gap':>10}{'平行p(长)':>12}")
print(f"{'完整(含XLK35%)':<16}{r_full_s['gap_post'].iloc[-1]*100:>+9.2f}%{r_full_l['gap_post'].iloc[-1]*100:>+9.2f}%{pre_trend_test(r_full_l['gap_pre'])[1]:>11.3f}")
print(f"{'剔除 XLK':<16}{r_noxlk_s['gap_post'].iloc[-1]*100:>+9.2f}%{r_noxlk_l['gap_post'].iloc[-1]*100:>+9.2f}%{pre_trend_test(r_noxlk_l['gap_pre'])[1]:>11.3f}")
xlk_contrib = (r_full_l['gap_post'].iloc[-1] - r_noxlk_l['gap_post'].iloc[-1]) * 100
print(f"\\n→ XLK 对长窗口 gap 贡献: {xlk_contrib:+.2f}pp(剔除 XLK 后 gap 从 +0.68% 跳到 +7.51%, 10x)")
print(f"\\n剔除 XLK 后 SCM 权重(替代 donor):")
print(r_noxlk_l['weights'][r_noxlk_l['weights']>0.05].round(3).to_string())''')

# ============ Cell 3: 图 ============
code('''fig, ax = plt.subplots(figsize=(8.5, 4.5))
labels = ['完整\\n(含XLK 35%)', '剔除 XLK']
gaps = [r_full_l['gap_post'].iloc[-1]*100, r_noxlk_l['gap_post'].iloc[-1]*100]
colors = ['steelblue', 'firebrick']
ax.bar(labels, gaps, color=colors)
ax.axhline(0, color='k', lw=0.5)
ax.set_ylabel('长窗口 SCM gap % (4/2→6/30)')
ax.set_title(f'Exclude XLK LOO: 剔除 AI/科技板块, gap 从 +0.68% → +7.51%\\n(XLK 贡献 {xlk_contrib:+.2f}pp → AI donor 同源低估 gap)')
for i, g in enumerate(gaps): ax.text(i, g + 0.3*(1 if g>0 else -1), f'{g:+.2f}%', ha='center', fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.show()''')

# ============ Cell 4: 结论 ============
md('''## 结论 + 论文定位(§5.13)

**核心发现**: 剔除 XLK(AI/科技), 长窗口 SCM gap 从 +0.68% 跳到 **+7.51%**(10x), **XLK 贡献 −6.83pp**——XLK 主导 donor 同源。

### 机制(AI donor 同源的代价)
- pre 期 AI 狂牛 → SCM 给 XLK 35%(走势最像标普)
- post 期(4/9 后)科技领涨 +21%(AI 独立行情) → 含 XLK 的合成捕捉科技行情 → 合成≈标普 → gap 小(+0.68%)
- 剔除 XLK → 合成(XLY/XLV/XLB/XLE)错过科技 → gap 大(+7.51%)
- **SCM gap 小不是"关税无害"**, 是 **AI 板块把 post 期科技独立行情计入合成、抵消关税缺口**

### 论文价值(donor 同源诊断第三支柱)
配合恐慌通道(§5.11 中介 56%) + 不确定性(§7 四重敏感性), 本节定量 **"AI 因子 → donor 同源 → gap 低估 6.83pp"**。这是 §5.2"donor 同源致 gap 小"的**机制诊断**(不只是说 gap 小, 还定位到 AI 板块 XLK)。

### 诚实局限
- Exclude XLK LOO 是 **donor 剔除**(非 AI 因子正交化); AI proxy 正交化(NVDA/AI ETF 残差)因 yfinance 限流未做, 但 Exclude XLK 已直接定量
- XLK 含 AI + 非半导体科技(如苹果/微软), 非纯 AI proxy——剔除 XLK 是"剔除科技板块", 严格 AI 因子正交化需 NVDA/AI ETF 残差
- 平行趋势 p=0.207(剔除 XLK, 边缘通过), 完整含 XLK p=0.957(更好)''')

# ============ 组装 ============
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '15_donor_confounding.ipynb')
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f'Generated {len(cells)} cells -> {OUT}')
