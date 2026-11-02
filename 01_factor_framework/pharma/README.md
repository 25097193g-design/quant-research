# 医药生物板块因子策略研究

> 与 [银行版](../bank_factor_workflow/) 同一框架的第二板块研究。完整叙事见 [REPORT.md](REPORT.md)，英文版 [REPORT_EN.md](REPORT_EN.md) → `pharma_report_en.pdf`。

## 最终策略

**12 因子透明组合（7 跨期存活核心 + 5 风格卫星，滚动符号），月频，12 个月滚动 IC 定号定权，买前 20 名等权。**

| 口径 | 数值 |
|---|---|
| rqalpha T+1（2020-01-23~2026-05-15） | **年化 15.69% / 夏普 0.62 / alpha 13.0% / IR 0.73 / 回撤 −31.3%** |
| 自建（2020-01~2026-06，78 期） | 15.77%（真引擎摩擦 ≈ 0） |
| 2018 起长窗压力测试（102 期） | 15.37% / 0.61 |
| vs 货基 / 沪深300 / 医药指数 | +13.9 / +13.1 / **+19.1pp**（同期医药指数 −3.4%/年） |
| 选股超额（vs 医药等权） | 7/7 年为正（2020 占大头 +35pp；2025 变薄 +0.4pp 为警讯） |

## 研究链

```
00_panel_qc.py            面板QC: 无低N/零缺失/次新12月收录特性
10_subperiod_ic.py        两阶段筛选: 137→24(全样本IC初筛)→铁存活7(分时段) ★
14_core_pool_backtest.py  池矩阵: 核心+卫星混合胜出
17_lookback_12m.py        回看窗: 12m>36m, 三起点隔离验证
24/24b/24c                集中度倒U→Top20定稿+三关稳健性+分年分解 ★
16_rqalpha_redo.py        rqalpha T+1 终验
19/21/22/23/25            XGB全矩阵(全败)/基准全家福/网格统一
12_embargo_check.py       标签隔离审计(方法论防线①的量化依据)
20_add_ev_to_ebitda.py    池外候选边界测试
26_final_artifacts.py     终版净值/集中度图/月收益存档
common.py                 单一真相源(数据清洗+回测)
```

## 运行方式

```bash
cd projects/pharma_factor_workflow
conda run -n rqsdk --no-capture-output python 00_panel_qc.py
conda run -n rqsdk --no-capture-output python 10_subperiod_ic.py
conda run -n rqsdk --no-capture-output python 16_rqalpha_redo.py    # 需 rqalpha bundle
python3 build_report_pdf.py            # 中文 PDF
python3 build_report_pdf.py REPORT_EN.md   # 英文 PDF
```

## 数据依赖

| 数据 | 路径 | 说明 |
|---|---|---|
| 因子面板 | `data/factors_monthend.parquet` | WQ101+MQ29 月频（**含 inf 需清洗**，见 alpha026 案例 REPORT §2.3） |
| 价格 | `data/mainboard_price.parquet` | 全主板日频复权价（含退市股） |
| 白名单 | `data/universe_monthend_pit.parquet` | PIT 行业归属，`industry_l1=='医药生物'` → 228 只 |
| rqalpha bundle | `~/.rqalpha/bundle/` | 行情至 2026-05-15（含申万医药 000808/中证医药 000933 指数） |

## 核心结论

1. **两阶段筛选是必要的**：全样本 IC 初筛的 24 因子只有 10 个跨两段存活——全样本口径会把单段故事选进来
2. **透明胜 ML**：{XGB}×{频率}×{持仓数}×{相位/起始}×双口径全矩阵，无一配置双口径胜透明组合
3. **集中度倒 U**：Q5 13.1% → **Top20 15.8%** → Top10 13.7% → Top5 10.2%（20 只≈分散饱和点）
4. **基准要三件套**：货基/沪深300/医药指数；等权被右尾抬高 13.4pp/年，只配当"无技能选股期望"参照
5. **坏年份是板块 beta 不是选股失效**：2023/2026 超额反而为正
