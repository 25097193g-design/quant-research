# Optiver Realized Volatility Prediction (2021) 复现项目

**比赛**: https://www.kaggle.com/competitions/optiver-realized-volatility-prediction
**数据(已在本地)**: `~/Downloads/optiver-realized-volatility-prediction` (2.5GB,112只股票)
**环境**: `conda run -n rqsdk --no-capture-output python xxx.py`

## 任务一句话
给你某股票**当前10分钟**的订单簿快照+成交流水,预测**下一个10分钟**的已实现波动率(realized volatility)。回归任务,评分 RMSPE(相对误差)。

## 为什么 Optiver 出这个题
做市商双边挂单赚价差,被动持仓的库存风险 ∝ 波动率。vol 预测准 → 报价宽度/挂单量/对冲成本全跟着调。是做市主营业务,不是学术题。

## 数据结构(2026-08-17 本地实测)
```
book_train.parquet/stock_id=N/*.parquet   ← 每股簿数据
  列: time_id, seconds_in_bucket,
      bid_price1, ask_price1, bid_price2, ask_price2,
      bid_size1, ask_size1, bid_size2, ask_size2
trade_train.parquet/stock_id=N/*.parquet  ← 每股成交
  列: time_id, seconds_in_bucket, price, size, order_count
train.csv  (428,932行)                    ← 答案表
  列: stock_id, time_id, target(=下一桶RV)
```
stock_0 实测:3,830 个 time_id,每桶平均 ~240 张快照,seconds_in_bucket 0-599。

## 名词表
| 名词 | 含义 |
|---|---|
| stock_0..111 | 匿名股票编号 |
| time_id(桶) | 10分钟窗口编号;**同一 time_id 跨股票=同一市场时段** |
| seconds_in_bucket | 窗口内第几秒(0-599) |
| 订单簿/簿 | 未成交限价单的排队表;一行数据=一张快照 |
| bid/ask | 买方/卖方挂单;第1档=最优价,第2档=次优(共L2两档) |
| size | 挂单数量 |
| spread | ask−bid,簿的温度计 |
| mid | (bid+ask)/2 |
| WAP | (bid_p1×ask_s1 + ask_p1×bid_s1)/(bid_s1+ask_s1),按对面量加权的真实价格估计 |
| WAP 路径 | 一个桶内几百张快照的 WAP 连成的价格走势线 |
| log return | ln(WAP_t/WAP_{t−1}) |
| RV | sqrt(Σ log_return²),不减均值;一桶价格抖动程度的单值 |
| target | 下一桶的 RV(train.csv 给好) |
| RMSPE | sqrt(mean(((pred−target)/target)²)),相对误差→log空间建模自然 |

## 三个结构性质(全部打法从此长出)
1. **波动聚集**:下桶RV≈本桶RV,naive baseline RMSPE≈0.34
2. **市场时钟共振**:同 time_id 跨股票 = 同一市场时段,市场级风暴全员共振(=A股31板块高相关的微观版)→ 同 time_id 其他股票的 vol 是最强特征之一
3. **股票异质**:每股有自己的 vol 量级 → per-stock target encoding/log空间截距

## 基准标尺(RMSPE,约数)
| 层 | RMSPE |
|---|---|
| naive(下桶=本桶) | ~0.34 |
| 公开 nearest-neighbor baseline | ~0.29 |
| 奖牌区 | ~0.21-0.23 |
| 金牌 | ~0.205 |
| 冠军 | ~0.19-0.20 |

## 高手方案
- **1st "Nearest Neighbors"** (discussion/274970):核心=找历史相似时刻。最重要特征就是上一桶RV;用当前桶信息+同time_id跨股票状态,在训练集里找最像的历史time_id,用它们的下一桶RV当预测
- **3rd "life is volatile"**:先预测全市场平均vol(宏观时钟)再回推个股 — 市场时钟的模型化
- **7th**:开源代码 https://github.com/michaelpoluektov/orvp
- **Public 39th**:纯 10-fold LGBM+同time_id"leak"特征 → private 0.20018(LGBM路线足够深)
- 汇总帖: discussion/249523
- 共同规律:上一桶RV(聚集)+ 同time_id跨股票(时钟) >> 簿形状特征(增量)

## 理论加分点(面试)
- RV=对数价格路径的离散二次变差(quadratic variation),采样加密→收敛积分方差
- **微观结构噪声**:采样过密时 bid-ask bounce 贡献假跳,Σr² 不收敛反发散 → noise-robust 估计器(two-scales RV / realized kernels);比赛给~1s快照已避开最毒区间

## 阶段计划
| 阶段 | 内容 | 产出 | 目标RMSPE | 时间 |
|---|---|---|---|---|
| P0 | EDA:打开簿、画WAP路径、算第一个RV、验证波动聚集(本桶RV vs target 散点) | 01_eda.py + 图 | — | 1-2天 |
| P1 | naive baseline 全量 | 02_naive.py | ~0.34 | 0.5天 |
| P2 | 簿特征工程(RV系列/spread/imbalance/trade强度)+LGBM+时间CV | 03_features.py 04_lgbm.py | 0.23-0.25 | 5-7天 |
| P3 | 同time_id跨股票特征(市场时钟) | 05_marketclock.py | 0.21-0.22 | 3-4天 |
| P4 | (可选)NN/ensemble | 06_nn.py | 0.20-0.21 | +1周 |
| P5 | "我 vs top" REPORT + README 挂GitHub | REPORT.md | — | 2天 |

核心版本(P0-P3+P5)约 2.5-3 周(每天3-4h);最低可信版(P0-P2+简报)约 10 天。

## 纪律(从 bank/pharma workflow 带过来)
- 时间CV防泄漏:验证用 time_id 在训练之后(不能随机shuffle)
- 一切"谁更好"是口径函数:固定同一验证集再比
- 结果必附脚本路径

## 进度
- 2026-08-18 P0 ✅ (01_eda.py): stock_0 波动聚集 Pearson 0.841 亲手验证;三图 outputs/(book_evolution / wap_path / rv_scatter)。⚠️ 股票编号非连续(0~126缺15个,train=112只), 用 list_stock_ids() 勿用 range(112)
- 2026-08-18 P1 ✅ (02_naive.py): **全量 naive RMSPE 0.3414**(锚点≈0.34,WAP/RV管线口径认证通过);pooled Pearson 0.874;分股中位 0.323,最好 stock_77 0.217,**最差 stock_31 1.039(P2待查)**;各股RV中位数差6.7倍。缓存 cache/rv_now_full.parquet(428,932行)
- 2026-08-18 P2a ✅ (03_features.py): 19特征(全本桶可观测,无跨桶泄漏)48s;NaN仅78(rv_last60退化桶);cache/features_v1.parquet。踩坑:特征帧要自带time_id列
- 2026-08-18 P2b ✅ (04_lgbm.py): 时间切分80/20(分界time_id=25683) | naive同折0.3497 → **LGBM v1 0.2673(改善23.6%)** Pearson 0.910 best_iter=528;importance头部=spread/depth/n_snap/trade_orders(流动性代理股票层级)+rv_last300/rv_now(聚集)。踩坑:LGBM sklearn API自定义metric签名是(y_true,y_pred)。⚠️未到0.23-0.25目标,待办:多折+调参/stock_31查询/**P3市场时钟特征(最大杠杆)**
- 2026-08-18 P3 ✅ (05_marketclock.py): **裸证据:市场层级聚集 corr=0.923**(本桶全市场RV中位 vs 下桶全市场target均值;个股自身0.874,纯市场信号0.698);v1 0.2673→**v2 0.2586**(+8市场时钟特征;importance前4全是市场特征但增益有限——与流动性特征信息重叠);**stock_31体检:水平全正常(rv分位56%/spread分位78%)但转换不可测**(v1分股RMSPE 0.965,次差才0.363→波动不聚集的异质股,值得深挖)。features_v2.parquet(27特征)。
- 2026-08-18 P3+ ✅ (06_twostage.py): B0复现0.2588 | **B1(+stock_id类别)=0.2493**(漏加的异质性杠杆,单独-0.95pp) | S1市场模型 corr 0.9633/**市场级RMSPE仅0.127**(市场波动比个股波动好预测2倍,难的是个股残差R) | 两阶段0.2489≈B1(分解被B1的stock_id+市场特征隐性吃掉,诚实教训) | **几何blend=0.2458 现最优**。stock_31两阶段仍0.816(异质性铁证)。cache/twostage_val_pred.parquet。待办:多折+调参/更细簿特征/P4(NN/冠军NN法)
- 2026-08-18 P4a ✅ (10_multifold.py): 4扩张折认证 **B1=0.2519±0.0037**(单折0.2493属运气偏好,诚实数~0.252) alt(叶127/lr.03)=0.2511±0.0033略优;naive 0.3433±0.0080锚定。
- 2026-08-18 P4b ✗无增益 (08/09): features_v3(+16细特征:100s子窗RV矩阵/rv_trade/不平衡稳定性/enc_stock等47列)单模0.2517,blend最好0.2462 —— 与LGBM已提取信息冗余;enc_stock进重要性第4但总体不涨。⚠️踩坑:pred与target行序不一致→blend出0.59垃圾数字,已修+加断言。
- 2026-08-18 P4c ✗首版无增益 (11_nn.py): MLP+stock嵌入(MPS,2种子,logit空间平均)单模0.2780,blend(NN+B1+两阶段)0.2496<0.2458 —— NN太弱拖累几何blend;要加分需per-stock target归一+更强结构,推后。⚠️踩坑:exp下溢→log(p)炸nan。
- **现状: 最优=blend(B1+两阶段) 0.2458(单折) / B1 4折诚实 0.2519±0.0037**;奖牌区外沿(0.21-0.23)。剩余: P5报告/深NN/stock_31案例
- 2026-08-18 P4c-2 ✅✅ (12_nn_deep.py): 深NN三升级(标签=log target−enc_stock个股归一 / LayerNorm+256-128-64+emb16+AdamW余弦 / 4种子logit平均)→ 单模**0.2470**(首版0.2780)追平B1;**blend(NN2×2+B1+两阶段)=0.2389 新最优**(进奖牌区)。stock_31在NN下0.968,三模型家族一致认定异类。两阶段诊断→定向修复=教科书案例。
- 2026-08-18 ⚠️重大发现 (13_timeorder_check.py): **time_id编号≠时间顺序**(相邻编号lag1相关-0.013≈随机打乱-0.001,个股/市场双层验证;第7名README证实他靠价格向量哈密顿路径逆向重排才拿到0.20013)。三个含义: ①我们的"time split"实为按乱序id的分区(≈随机分区),数字仍诚实(无val标签泄漏)但叙事改口径 ②enc_stock只含train标签→对test式评估合法(=第7名的lagged label思路),但非部署现实 ③**下一个大杠杆=重建真实时间顺序→真时序特征(前序桶RV链)**,冲0.21-0.20的路径。
- 2026-08-18 开源对标: 第7名 repo(github.com/michaelpoluektov/orvp)=主模型0.20013(时间重排trick)+simple_model 0.22955(实用LGBM+BorutaSHAP完整代码)→"我vs top"的参照物。
- 2026-08-18 A 时间重排 ◐部分成功/模型中性 (14_reorder.py 15_lag_model.py): 首版贪心argmin撞车严重(互前驱对仅64/3830)→改**匈牙利一对一指派+环打断24**;重建证据:前驱距离=随机对的0.49倍、**真lag聚集 corr(rv_prev1,rv_now)=0.650(乱序-0.013)**、corr(rv_prev1,target)=0.592。但模型端:B1+lag 0.2518 vs B1同折0.2529(Δ-0.001),blend全家族不加好(0.2398>0.2391)→**重建质量不足以产生增量信号**(差距根源:第7名有tick size恢复真实价格+KNN+暴力SHP路径目标,我们只有归一化价格+指派)。留档改进:多快照价格匹配/路径目标局部搜索(Or-opt)。**0.2389维持最优**。合法性论证存档:lag特征=预测时点已实现的历史,非泄漏。
- 2026-08-18 A-v2 精细重做 ✗更差 (16_reorder_v2.py): 三招全负——①tick反推失败(min-diff抓到浮点噪声5.96e-8,恢复出$16万股价;应改差值直方图众数)②**5快照均值指纹反而伤**(0.650→0.525:薄簿的末5张快照跨数分钟,"桶末状态"被抹花——事件数据教训:假设规则采样的稳健化会适得其反,单张末快照才是真最新状态)③λ路径覆盖切碎链(只留1175/3830边,2655碎片链中位深0,选择性偏差)。**v1(单快照+全指派)保持最优0.650**。负价值:验证了7th每一步都不是随便选的。
- 2026-08-18 复现第7名 🏆 (17/18a/18b + orvp_fp_patched.py): clone repo(external/,gitignore)。pandas2适配=3处apply→transform。**特征管线复现:428,932×46特征44s**(WAP2/quadpower quarticity/6窗RV回归截距斜率残差/fix_offsets桶起点归零/跨股票time统计)。
  - **他的协议OOF=0.1990**(随机5折,5折0.195-0.207;他自己LB 0.2296 → 随机折高估=期内插值vs真外推,同bank教训)
  - **我们的split拆解: A2他的特征+他的1/y²加权=0.2242 | A1他的特征+我们的log协议=0.2434 | B2我们的特征+他的1/y²=0.2265 | B1我们全套=0.2493**
  - **★当日最大教训: 1/y²加权MSE精确对齐RMSPE,完胜log-target(单这一招-2.3pp)**——log只在误差小时≈相对误差,RV误差30-40%二阶项不可忽略;目标函数要精确对齐度量,不要近似对齐
  - 他特征优势~0.6pp(两协议一致);**HIS单模0.2242>我们全家族blend 0.2389**;HIS与我们blend反而变差(0.2261-0.2270,弱者稀释强者的几何blend,同首版NN教训)
  - **新最优=0.2242(HIS A2,我们的split)**;待办:NN换1/y²加权loss/blend(HIS+B2)/他的1000轮封顶未收敛/伪时间split稳健性检查
- 2026-08-19 三发收官 (19a/19b):
  - 弹药3 ✗ HIS放量3000轮=0.2272>1000轮0.2242(best_iter顶到2999仍差)——此折上过拟合,负结果
  - 弹药2 ✅ blend(HIS+B2):HIS1000+B2=**0.2207**(等强+特征多样性;HIS3版0.2215)
  - 弹药1 ✅ NN换1/y²精确对齐(比值空间r=target/est,est约掉→(r̂−r)/r≡RMSPE,标准化z+权重1/r²):0.2470→**0.2350**(-1.2pp,加权技巧对NN同样成立;成为有效ensemble成员)
  - **★最终最优 blend(HIS1000×2+B2+NNW)= 0.2193** (4way等权0.2195)
  - 全程: naive 0.3414→…→0.2389(自家blend)→0.2242(7th栈)→**0.2193(混合家族)**,误差砍36%
  - ⚠️诚实口径: 0.2193是分区CV(期内插值);按7th的OOF→LB退化比(0.199→0.2296,+15%)折算LB等效≈0.252——报告必须双口径
- **项目技术线收尾。下一步P5报告(双口径+方法论时间线+我vs top)。**
- 2026-08-19 真实结果定标 (20_true_eval.py): leave-regime-out压力测试(扣出top10%风暴时段,训练完全没见过):
  - 风暴hold-out: **HIS 0.2453(+2.1pp) | B2 0.2942(+6.8pp!) | blend 0.2608 | naive 0.5451**
  - 平静hold-out(对照): HIS 0.2864 / B2 0.2839(向平静外推两者都掉~6pp)
  - **诚实定标: 真实结果=区间 [0.22插值, ~0.26严苛外推], LB等效折算~0.25**;相对naive的误差削减(0.55→0.26)在未见regime下保持≈52%——绝对数会膨胀,相对优势稳健
  - 附带发现: HIS特征在风暴外推下远稳于我们(+2.1 vs +6.8pp)——WAP2/quadpower稳健估计的红利,报告里写
