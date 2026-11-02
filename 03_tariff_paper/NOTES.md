# 全球冲击、污染对照:特朗普2018与2025关税事件中反事实方法的失效诊断

> 工作论文 · **M1 rev.2 (2026-09) SSRN v1 投递** · 目标 SSRN + arXiv q-fin
> 作者: lc (香港理工大学商业分析硕士)

## 一句话

特朗普两任期关税(2018-19 渐进 vs 2025-04 解放日休克)同时处理了所有候选对照——行业 ETF 是标普成分切片、海外指数同期齐跌。本文不报效应,而是**诊断 SCM/BSTS/DL 反事实方法在全局污染对照下的失效**,并给出不依赖 donor 的波动率描述性事实。

## 核心发现 (M1 rev.2)

1. **构造性无检验力**:donor 同源 → 合成≈标普 → placebo p=100%(真实 gap 绝对值全场最小),该设计下 SCM 永远检不出效应
2. **缺口随对照构造排序**:数据驱动 V(11x 散布)、剔 XLK(+0.68%→+7.51%)、ASCM ridge 旋钮(凸界→原始收益,单调)——统计量追踪构造而非处理;跨设定散布是**不同 estimand**,不是不确定性区间
3. **VIX"通道"分解被否定**:VIX 是标普期权隐含波动率,收益~ΔVIX 回归是同期机械协动(杠杆/反馈);"56%"只作描述性协动份额,不称通道/中介/因果
4. **幸存事实(donor-free,描述性)**:2025 VIX 即时 +140%、半衰期 1 天 vs 2018 单节点零即时反应(学习仅为三种解读之一,未识别);行业原始跌幅排行被 beta 主导(raw~残差相关 0.24)
5. **海外对照弃用/降级**:海外 SCM 权重集中(加拿大 0.76)+pre趋势崩而弃用;BSTS 降为标普 vs 海外的相对表现会计
6. **阳性控制(加州99号提案)**(`positive_control.py`):四步诊断在干净对照下全部通过——RMSPE 比值口径 p=0.026(39 单元中最极端,对照关税 p=100% 全场最小)、凸 SCM 复现已发表效应(−25.5 包)、sweep 只调量级不制造结论 → 协议可证伪
7. **2026 法律后续**(IEEPA 被推翻 → Section 122 到期 → Section 301 接档)作政策为移动靶的背景讨论

## 复现

### 环境
Python 3.9（conda env）。依赖：
```bash
pip install yfinance statsmodels cvxpy scikit-learn pyarrow pysyncon arch torch
```

### 数据下载
```bash
cd notebooks/paper_tariff
python download_tariff_data.py   # yfinance 拉 4 面板 → ../../data/tariff/
```
数据源 yfinance（免费）。海外 donor 15 个发达市场（2026-07-12 扩池，剔 EEM/HSI）。若 yfinance 限流，用 `download_overseas_standalone.py`（单 ticker 慢速 fallback）或换 IP/手机热点。

### 跑 notebook
所有 notebook 由 `generate_0X_*.py` 生成（风格 A，json.dump）。复现：
```bash
cd notebooks/paper_tariff          # 重要：相对路径基准
python generate_01_data_explore.py
jupyter nbconvert --to notebook --execute --inplace 01_data_explore.ipynb
# 依次 01-15（nb12 LSTM 多 seed 较慢，timeout 1200s）
```
**路径已相对化**：`paper_lib.py` DATA_DIR 用 `os.path.dirname(__file__)`；generate 的 OUT/sys.path 相对。复现时 `cd notebooks/paper_tariff` 跑即可。

### 中国敞口数据
`china_exposure_vix_futures.md`（PIIE/BEA 综合估算 proxy，非精确 10-K）。nb14 用。

## 目录结构
```
notebooks/paper_tariff/
├── README.md                  本文件
├── paper_lib.py               共享库(SCM/augmented SCM/CausalImpact/事件研究/中介/placebo, to_cumlog已修复)
├── ascm_sweep.py              augmented SCM(§5.14) ridge α 敏感性一键复现
├── positive_control.py        阳性控制:加州烟草四步诊断(paper §6.2,缓存后瞬时)
├── smoking_california.csv     ADH 加州面板镜像(positive_control 用)
├── aligned_windows.py         两轮对齐视界(+4/+20/+60交易日)SCM对比(paper §5.5)
├── make_vix_figure.py         §5.6两轮VIX路径双栏图 → fig_vix_paths.png
├── download_tariff_data.py    数据下载(yfinance 4面板)
├── download_overseas_standalone.py  海外慢速 fallback(yfinance限流时)
├── generate_0X_*.py           notebook 生成脚本(15个)
├── 0X_*.ipynb                 notebook(01-15)
├── 论文.md                     工作笔记原稿(M1,保留)
├── 论文_正式版.md               正式中文稿(笔记体扩写为连贯学术行文)
├── paper.tex / paper.pdf      英文投稿版(SSRN/arXiv, latexmk -pdf paper.tex 编译)
├── references.bib             参考文献(13条BibTeX)
├── china_exposure_vix_futures.md  行业中国敞口proxy+VIX期货说明
└── data_findings_summary.md   数据可得性记录
data/tariff/                   (gitignore) 4面板parquet
```

## notebook 路线
| # | 主题 | 核心 |
|---|---|---|
| 01-02 | 数据探索/EDA | 解放日剧情(8日−12.1%) |
| 03 | SCM 行业 donor + pysyncon V 敏感 | XLK 35%/+0.68%，V 敏感(11x) |
| 04 | CausalImpact + in-time placebo | 海外 CI −3.54%/+4.35% |
| 05 | 事件研究 CAR | XLK beta 幻觉(+5.19%) |
| 06 | placebo/LOO | p=100%(donor 同源) |
| 07 | 海外 donor SCM（降级） | 加拿大 0.76，平行趋势崩 |
| 08 | 两轮对比 | 2018 渐进 vs 2025 休克(SCM gap 都小) |
| 09 | 复盘 | 五大发现 |
| 10 | VIX 正交化(SCM gap 口径恐慌) | 不反转，0.18pp |
| 11 | GARCH 波动率体制 | VIX 形状铁证 |
| 12 | LSTM/MLP DL 反事实诊断 | 函数形式敏感(99x) |
| 13 | 中介分析 | 恐慌通道 56% |
| 14 | SCM 权重中国敞口 | XLK 35% 高估 |
| 15 | Exclude XLK donor 同源 | AI 板块 −6.83pp |

## 局限 + 投稿
- donor 内生性 / V 敏感 / 函数形式敏感 / 海外 SCM 降级 / GDELT 情感弱 / 中国敞口 proxy / VIX 期货不可得
- **M1 (8/31) SSRN v1**：2018 vs 2025 两轮 + 2026 法律背景
- **M2 (12/31) SSRN v2 + arXiv**：加 2026 法律战第三形态 + 法院逆事件 + 海外重诊断（全球首篇三轮政策形态整合）

## 引用
```bibtex
@misc{tariff_scm_2026,
  author = {lc},
  title = {Counterfactual Estimation of Trump's Tariffs on the S\&P 500: 2018 vs 2025},
  year = {2026},
  howpublished = {SSRN Working Paper}
}
```
