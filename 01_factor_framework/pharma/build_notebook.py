"""build_notebook.py — 构建 pharma_factor_workflow.ipynb（读 outputs/ 预计算结果, 快速可复现）

叙事 = REPORT.md: 两阶段筛选 → 构造 → ML对照 → 集中度 → 终验 → 基准 → 分年。
执行: cd projects/pharma_factor_workflow && conda run -n rqsdk python build_notebook.py
然后: conda run -n rqsdk jupyter nbconvert --to notebook --execute --inplace pharma_factor_workflow.ipynb
      conda run -n rqsdk jupyter nbconvert --to html --no-input pharma_factor_workflow.ipynb
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []

def md(s): C.append(nbf.v4.new_markdown_cell(s))
def code(s): C.append(nbf.v4.new_code_cell(s))

md("""# A 股医药生物板块因子策略研究 · 全流程

> 求职展示版 | Li Changhao | 2026-08
> 详细叙事见 [REPORT.md](REPORT.md)（英文版 [REPORT_EN.md](REPORT_EN.md)）；每个数字有对应脚本与 CSV

**最终策略**：12 因子组合（**非等权**）= 7 个核心因子（跨期存活，方向锁定）+ 5 个卫星因子（风格类，方向每月滚动重估）。每月末按**过去 12 个月 |ICIR|** 加权打分，全医药截面（~207 只）排序，**买前 20 名等权**，月频调仓。容量：单日调仓 ≈3 亿、分 3 日 ≈10 亿（2025 板块日成交 520 亿、单票中位 1.58 亿，参与率 10% 计）。当前状态：2026H1 超额回正（+0.7pp）、股息 IC 回 +0.101，正常运转区间。

| 口径 | 数值 |
|---|---|
| **rqalpha T+1（2020-01-23~2026-05-15）** | **年化 15.69% / 夏普 0.62 / alpha 13.0% / IR 0.73 / 回撤 −31.3%** |
| 自建（2020-01~2026-06，78 期） | 15.77%（真引擎摩擦 ≈ 0） |
| 2018 起长窗压力测试（102 期） | 15.37% / 0.61 |
| vs 货基 / 沪深300 / 医药指数 | **+13.9 / +13.1 / +19.1pp**（同期医药指数 −3.4%/年） |
| 选股超额（vs 医药等权） | **7/7 年为正** |

**一页速览（按读者）**：给投资人——年化 15.7%、月频 20 只、容量数亿量级非容量约束；给研究员——137→24→7 两阶段筛选 + 滚动符号卫星，XGBoost 全矩阵后弃用；给风控——回撤 −31.3%（587 天）、beta 0.71、2025 超额归零为首要监控项（§7 归因）。

**样本口径（全文统一）**：115 个月全样本（2017-01~2026-07）用于因子筛选（前 36 个月=滚动窗暖机）；终验窗 78 个月（自建 2020-01~2026-06）；rqalpha 窗 2020-01-23~2026-05-15（数据包终点）。""")

code("""%matplotlib inline
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from IPython.display import Image, display
import matplotlib; import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif']=['Arial Unicode MS','PingFang HK']; matplotlib.rcParams['axes.unicode_minus']=False
OUT='/Users/lichanghaodemac/quant_strategy/projects/pharma_factor_workflow/outputs'
pd.set_option('display.width', 160)
print('医药面板: 228只 PIT白名单, 2017-01~2026-07 共115个月末, 137因子')""")

md("""## 1. 面板 QC 与方法论防线

白名单 2017 年起即 184 只、2019 年起稳定 210-219（中位 207）——**无低截面问题**；候选因子零缺失；收录滞后上市 12-13 个月 = 次新排除特性。

**全程执行的四条防线**（防四类常见虚高）：①标签隔离（embargo，3m 标签训练窗截至决策点前 3 个月）；②相位稳健（季度结论一律 3 相位中位，月频天然无相位）；③同窗基准 + 几何年化统一口径；④双引擎对账（自建扫矩阵、rqalpha 终验，差值=真实摩擦）。""")
code("""display(Image(f'{OUT}/panel_qc_count.png'))
sens = pd.read_csv(f'{OUT}/panel_qc_sensitivity.csv')
print(f"低N敏感性: N≥100/N≥150 档合格状态翻转的因子 {int((sens['全样本_合格']!=sens['N≥100_合格']).sum())} 个 → 筛选结论对低N稳健")""")

md("""## 2. 两阶段因子筛选：137 → 24 → 铁存活 7

**阶段一（广度）**：全样本 IC 阈值初筛（|IC|≥0.03 & |ICIR|≥0.3；阈值随截面规模设定——银行版 41 只用 0.05，医药 207 只标准误更小按比例下调；**敏感性**：收紧到 0.05 时 7 个铁存活仅剩 alpha040 与股息率，其余 5 个为有意的"弱而稳"组合化设计）→ 24 候选。
**阶段二（时间稳健）**：分时段存活检验（两段 + 三段细切）。**漏斗：137 → 24 → 两段12 → 三段9 → 铁存活 7**。

三种淘汰模式：**早期衰减**（surplus_cash 等 11 个，全样本均值被早期撑住）/ **风格反转**（bm/pb/ps/VOLT，2020 年翻面且持续加强——regime 切换非噪声）/ **靠中段撑**（4 个，2023-26 塌掉）。

**因子定义与方向规则**（公式取自实算代码 wq_lib.py；H=最高价 V=成交量 C=收盘 VWAP=量加权均价；"正向"=因子值高看多，负号已在公式内）：alpha040=低价波动×价量同向 / alpha016/044/026=高价放量·量价同步极值时看空（过热反转族）/ alpha011=偏离VWAP极端度×量变共振 / alpha024=横盘超跌反弹 / dividend=股息率 —— **以上 7 个方向锁定（三段 IC 全正）**；bm/pb/ps=估值便宜度、VOLT20/60=低波动异象 —— **以上 5 个滚动方向**（12 个月 IC 均值定号，2020 年曾翻面；bm 常正向、pb/ps/VOLT 常负向）。

**漏斗最后一档**："铁存活" = 两段判定 ∩ 三段判定（9 个三段存活者中 2 个未过两段、12 个两段存活者中 5 个三段挂掉，交集 7）。""")
code("""sub = pd.read_csv(f'{OUT}/subperiod_ic_137.csv')
iron = sub[sub['存活'] & sub['三段存活']].sort_values('ICIR_late', ascending=False)
print(f'铁存活 {len(iron)} 个:')
print(iron[['因子','IC_2017-19','IC_2020-22','IC_2023-26','IC_late','ICIR_late']].to_string(index=False))
print(f"\\n初筛24因子: 两段存活 {int(sub[sub['在24池']]['存活'].sum())}/24, 三段存活 {int(sub[sub['在24池']]['三段存活'].sum())}/24 → 全样本IC会把单段故事选进来")""")
code("""display(Image(f'{OUT}/subperiod_ic_scatter.png'))""")

md("""### 2.2 数据管线个案：alpha026 的漏检与入选

alpha026 阶段一就被漏掉——不是因为信号弱：其公式含除法，原始数据 4,154 个 inf；上游 IC 汇总表未做 inf 清洗，截面 std 遇 inf 变 NaN → 它的 IC 全被记为 NaN → 筛选从未看见它。阶段二**绕开汇总表、带清洗从因子面板重算全量 137 因子**，a026 以三段全正（+.040/+.037/+.056）**入选核心池**。教训已成管线规则：除法类因子必须 inf→NaN；清洗规则变更后必须重算上游汇总。

### 2.3 池外候选的边界测试

池外最强候选 ev_to_ebitda（两段存活、与价值簇相关仅 ±0.3、三段衰减）加入实验：13 因子 12.90% vs 12 因子 13.10%——**−0.2pp 噪声级，不纳入**。池子边界到此。""")

md("""## 3. 组合构造：月频 + 12m 滚动 + 前 20 只

每月末：方向 s = 过去 12 个月 IC 均值符号（7 个铁存活因子用**固定方向**，5 个风格因子**每月重估**）；权重 w = |过去 12 个月 ICIR| / Σ|ICIR|；截面 winsorize+z-score 加权合成 → 全截面排序**买前 20 名等权（各 5%）**，持有一个月（换手 60-70%）。滚动方向**无平滑**——12 个月 IC 均值过零即刻反向，"6-12 个月滞后"是窗口的自然检测时间而非设计延迟；全程无拟合参数。""")
code("""d17 = pd.read_csv(f'{OUT}/redo_stageD_lookback.csv')
print('回看窗对照(月频, 2020-01起78期):')
print(d17[d17['频率']=='月'][['配置','年化','夏普','超额vs医药EW','期数']].to_string(index=False))
print('\\n36个月窗同池 = 11.81% (17b) → 12个月窗 13.10%')""")

md("""## 4. 持仓集中度：倒 U 型，Top20 是甜点

Q5(前20%≈41只) 13.10% → **Top20 15.77%** → Top10 13.66% → Top5 10.21%。三关稳健性全过：3 起始偏移极差 0.4pp / 2018 长窗 15.37% vs Q5 的 11.35% / rqalpha 零摩擦。机制 = 砍掉宽组合后半段稀释名 + 20 只≈分散化饱和点。""")
code("""k = pd.read_csv(f'{OUT}/redo_stageK_topN.csv')
print(k.to_string(index=False))
print('\\nQ5 的 rqalpha T+1 = 13.40%(16号脚本Stage C主推), 集中度四档双引擎: Q5 13.10/13.40 → Top20 15.77/15.69 → Top10 13.66/12.00 → Top5 10.21/8.89')
print('三关稳健性(24b): 3起始偏移 15.77/15.68/16.07(极差0.4pp); 2018长窗 Top20 15.37%/0.61 vs Q5 11.35%/0.48')""")
code("""display(Image(f'{OUT}/concentration_chart.png'))
display(Image(f'{OUT}/final_equity_curve.png'))
r = pd.read_csv(f'{OUT}/final_top20_monthly_returns.csv', parse_dates=['monthend'])
print(f"Top20月收益序列: {len(r)}期 {r.monthend.min().date()}~{r.monthend.max().date()}")""")

md("""## 5. 为什么不用机器学习：全矩阵对照

12 因子按 9 簇合成喂 XGBoost（浅树强正则，超参不经本板块调优），与透明组合同防线对照：

| 配置 | 自建 | rqalpha T+1 | 死因 |
|---|---|---|---|
| **透明 × Top20** | **15.77%** | **15.69% / −31.3%** | —（冠军） |
| XGB 季频×Q5（3相位） | 14.2/12.4/13.4 | 13.94%中位（极差4.2pp）/回撤−32.5% | 相位极差 4.2pp + 黑箱 |
| XGB 月频×Q5（3起始） | 10.33/10.52/10.36 | 7.62% | 1 个月标签对树≈噪声 |
| XGB 月频×Top20 | 12.19% | 8.88% | 真引擎摩擦 −3.3pp |
| XGB 26因子×Q5 | 中位 12.0% | 10.87% | 给浅树加料=稀释有效分裂 |

最优 ML 配置仅胜透明 +0.5pp，代价是相位风险 + 更深回撤 + 不可解释。**结论：这个截面大小与样本量级上，透明 ICIR 加权就是最优解。**""")
code("""print('终格(25): XGB月频×Top20 = 自建12.19% / rqalpha8.88% → 集中度救不动月频ML')
print('标签隔离审计(12): 训练窗含未来标签时 XGB 19.2% → 干净口径 12-13% 中位, RankIC 虚高25%')""")

md("""## 6. 终验与基准：三件套 + 右偏量化

**rqalpha T+1（2020-01-23~2026-05-15）**：年化 15.69%，总收益 +142.8%，夏普 0.62，索提诺 0.85，alpha 13.0%，beta 0.71，IR 0.73，回撤 −31.3%（2023-05~2024-12，587 天）。自建 15.77% → **摩擦 ≈ 0**。""")
code("""b = pd.read_csv(f'{OUT}/redo_stageI_bench_xgb12f.csv')
print(b.to_string(index=False))
print('''
右偏量化: 医药等权 +4.84% vs 典型股票(截面中位数) -8.60% = 13.4pp/年 全由妖股右尾贡献
(跌最多-100%、涨可数倍 → 均值被右尾抬高) → 等权降级为"无技能选股期望"参照, 主口径=三件套''')""")

md("""## 7. 分年分解与 2025 归因""")
code("""y = pd.read_csv(f'{OUT}/redo_stageK3_yearly_decomp.csv')
y = y.rename(columns={'医药EW':'板块·医药等权','全AEW':'全A等权'})
y['板块·申万医药指数'] = ['+55.1%','-13.0%','-23.7%','-12.4%','-14.5%','+5.3%','-4.3%']
print(y.to_string(index=False))
print('''
①选股超额 7/7 年为正(2023板块-19.8%/-12.4% vs 策略-12.4% → 亏的是beta一目了然)
②2020 一年 +35pp 是大头(剔除后年均~+4.3pp); ③2025 超额归零(下格归因)
④等权与指数口径分年差异极大(2025: +24.8% vs +5.3%) — 等权被小盘右尾抬高, 指数没有
⑤记账约定: 调仓月m的收益覆盖m→m+1行情, 故1月行情记在上一年行''')""")
code("""att = pd.read_csv(f'{OUT}/redo_stageN_2025_attribution.csv')
print('2025归因(27脚本): 12因子中9个IC仍显著; 股息率转负(-0.007, 七年唯一)是主异常;')
print('全A等权+44.4%小盘泡沫年, 医药等权被右尾极端抬高(截面离散度0.101为2021后最高)')
print('→ 2025=基准通胀年+个别因子疲软, 非引擎失效; 2026H1股息IC回+0.101、超额回正(+0.7pp)')""")

md("""## 8. 执行路径与风险披露

**执行**：月频、20 只、月换手 60-70% ≈ 每月 12-14 对订单——自动化复杂度最低档。信号引擎与执行层分离；T+1/涨跌停/真实费用已验证摩擦 ≈ 0。实盘路径：券商量化 API（QMT 类）全自动月调仓。

**风险（主动披露）**：① 卫星池是风格赌注（翻面时滚动号滞后约 1 年）；② 2025 超额近乎归零需监控；③ 样本 6.3 年含一个疫情极端年；④ beta 0.71，板块熊市亏 −12% 量级；⑤ 因子池存在选择自由度（已用相位/起始/长窗/双引擎缓解）。

---

## 附录：脚本与输出索引

```
00_panel_qc / 10_subperiod_ic(★筛选核心) / 12_embargo_check(标签隔离审计)
14_core_pool_backtest(池矩阵) / 15_ml / 16_rqalpha_redo(T+1终验) / 17_lookback(窗长)
19/21/22/23/25(XGB全矩阵/基准/网格统一) / 20(池外边界) / 24/24b/24c(集中度+稳健性+分年★)
26_final_artifacts(净值曲线/集中度图/月收益存档) / common.py(单一真相源)
outputs/ 下每个结论有对应 CSV/PNG
```""")

nb['cells'] = C
with open('pharma_factor_workflow.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('存 pharma_factor_workflow.ipynb (%d cells)' % len(C))
