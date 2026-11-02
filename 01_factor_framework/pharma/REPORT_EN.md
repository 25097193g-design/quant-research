# A Cross-Sectional Factor Strategy for China A-Share Pharma & Biotech

> Job-market showcase | Li Changhao | 2026-08
> Code & outputs: [projects/pharma_factor_workflow/](.) · script index in appendix · executable walkthrough in [pharma_factor_workflow.ipynb](pharma_factor_workflow.ipynb)

---

## Executive Summary

- **Strategy**: a 12-factor composite (**not equal-weighted across factors**) = 7 core factors (time-robust, directions locked) + 5 satellite factors (style, directions re-estimated monthly). Each month-end, factors are weighted by **trailing-12-month |ICIR|**; all ~207 pharma stocks are scored, and the strategy buys the **top 20 equal-weighted**, rebalanced monthly.
- **Performance** (rqalpha event-driven engine, T+1 execution, 2020-01-23~2026-05-15, benchmark CSI 300 +2.6% over the same window): **annualized 15.69%, total +142.8%, Sharpe 0.62, Sortino 0.85, alpha 13.0%, beta 0.71, information ratio 0.73, max drawdown −31.3%**. The in-house backtest (through 2026-06) returns 15.77% — real-world frictions ≈ 0.
- **Excess vs the benchmark trio**: money-market fund (1.75%) **+13.9pp**; CSI 300 (+2.56%) **+13.1pp**; SW Pharma Index (**−3.42%/yr**) **+19.1pp** — holding the pharma ETF lost money over this window.
- **Positive stock-selection alpha in 7 of 7 years** (vs pharma equal-weight): the losing years (2023, 2026H1) were sector beta, not selection failure. Caveats disclosed: 2020 alone contributes +35pp of excess; **2025 excess fell to ~zero (+0.4pp) — attributed in §4.4**.
- **Machine learning was tested systematically and rejected**: {XGBoost} × {monthly/quarterly} × {portfolio size} × {phase/start robustness} × {two engines} — no configuration beats the transparent composite on both engines.
- Stress test: from 2018-01 (including the 2018 bear market and the old style regime) the strategy survives at 15.37% annualized.

**One-page summary (by reader)**

- **For PMs**: 15.7% annualized (rqalpha T+1, 2020-01~2026-05); +13.1pp vs CSI 300, +19.1pp vs the pharma index; monthly, 20 equal-weighted names (5% each), 60-70% monthly turnover. **Capacity**: 2025 sector daily turnover ≈ RMB 52bn, median single-name daily turnover ≈ RMB 158m; with a one-day rebalance capped at 10% of a stock's daily turnover, **one-day capacity ≈ RMB 300m, three-day execution ≈ RMB 1bn** — not capacity-constrained. **Current status**: 2026H1 excess back positive (+0.7pp); dividend-factor IC recovered to +0.101 (strongest in seven years) — the strategy is in its normal operating range.
- **For quants**: 137 factors → two-stage screen (full-sample IC → 24; sub-period survival → 7) + a 5-factor style satellite on rolling signs; 12-month rolling ICIR weighting; XGBoost tested across the full matrix and rejected.
- **For risk**: max drawdown −31.3% (587 days, 2023-05~2024-12); beta 0.71 (a 2023-style sector bear costs −12%; selection does not hedge the sector); 2025 excess ≈ zero — currently the single metric to monitor (§4.4).

---

## 1. Question, Data, and Methodological Guardrails

Where does cross-sectional alpha in pharma & biotech (SW L1, PIT whitelist of **228 stocks**, 115 month-ends from 2017-01 to 2026-07) come from? Using 137 candidate factors (WorldQuant 101 + 29 fundamentals + 7 self-computed price-volume), this study answers three questions: which factors survive across regimes, how to construct the portfolio, and whether machine learning is necessary.

**Sample convention (unified across the report)**: the **full 115 months (2017-01~2026-07) are used for factor screening** (the first 36 months double as the rolling-window warm-up); the strategy validation window is **78 months (in-house: 2020-01~2026-06)**; the rqalpha engine window is **2020-01-23~2026-05-15** (limited by the market-data bundle's end date). Every number below belongs to one of these three windows.

**Data** (all local parquet):

| File | Content | Use |
|---|---|---|
| `mainboard_price.parquet` | Daily adjusted prices, 3,396 mainboard stocks incl. delisted | Forward returns, self-computed factors |
| `factors_monthend.parquet` | WQ101 + MQ29 monthly factors | Factor library |
| `universe_monthend_pit.parquet` | Monthly PIT industry membership + listing dates | Whitelist |

**Panel QC** ([00](00_panel_qc.py)): the whitelist already holds 184 stocks in 2017 and stabilizes at 210-219 (median 207) from 2019 — no small-cross-section problem; candidate factors have zero missing values inside the whitelist; whitelist inclusion lags listing by ~12-13 months (a deliberate new-stock exclusion). The data layer is clean; the interesting problems are methodological.

**Methodological guardrails** (enforced throughout, against four classic sources of inflated results):

1. **Label embargo**: with 3-month labels, the training window ends 3 months before the decision date (otherwise training labels contain test-window returns — measured inflation 2-3pp)
2. **Phase robustness**: every quarterly result is reported as the **median of 3 start phases** (a single-phase estimate can sit 4pp+ from the median); monthly has no phase dimension by construction; 3 start-month offsets additionally tested
3. **Same-window benchmarks, geometric annualization**: benchmarks on the strategy's own window, unified compounding (a full-sample benchmark vs a 2020-start strategy doubles the stated excess)
4. **Two-engine reconciliation**: in-house scans the matrix; rqalpha (real matching, price limits, T+1, commissions) validates finalists (otherwise a paper strategy can hide 3pp of unexecutable friction — the monthly-ML case in point)

---

## 2. Factor Selection: 137 → 24 → 7 Iron Survivors

### 2.1 Two-stage design

**Stage 1 (breadth)**: full-sample IC screen (|IC|≥0.03 & |ICIR|≥0.3, union of 1m/3m horizons) → 24 candidates.

**Stage 2 (time-robustness)**: sub-period survival testing ([10](10_subperiod_ic.py), 1m horizon) — two segments (2017-19 vs 2020-26) plus a three-way cut (2017-19 / 2020-22 / 2023-26). Survival requires: late-period significance, unchanged sign, stable magnitude (|z_diff|<2), sufficient early sample; the three-way test additionally requires same sign in all three and |IC|≥0.03 in the last two.

**Where the thresholds come from**: |IC|≥0.03 scales with cross-section size (the bank study on 41 stocks used 0.05; at 207 stocks the single-month IC standard error is smaller, so the bar comes down proportionally); |ICIR|≥0.3 follows the mean/volatility convention for monthly signals; |z_diff|<2 is the conventional 2σ. **Sensitivity disclosed**: tightening the late-period threshold to 0.05 keeps only alpha040 and dividend yield of the 7 survivors — the other five are deliberately "weak-but-stable" signals (IC between 0.03 and 0.05) that contribute through portfolio weighting, a design choice rather than a screening accident.

**Funnel: 137 → 24 screened → 12 two-segment → 9 three-segment → 7 iron survivors**. The final tier is the **intersection of the two judgments**: passing both the two-segment test (late significance + stable sign + stable magnitude) **and** the three-segment test (same sign in all three + |IC|≥0.03 in the last two) — of the 9 three-segment survivors, 2 (a006/a088) miss the two-segment bar on late ICIR; of the 12 two-segment survivors, 5 fail three segments; the intersection leaves 7.

### 2.2 Results: 7 survivors, 3 failure modes

**Factor definitions and direction rules** (formulas from this project's implementation code `wq_lib.py` / `frame_05`; H = high, V = volume, C = close, VWAP = volume-weighted average price; "long" = higher factor value is better, the sign is already inside the formula):

| Factor | Logic (sketch) | One-line meaning | Direction rule |
|---|---|---|---|
| alpha040 | −rank(std(H,10)) × corr(H,V,10) | Low high-price volatility **and** price-volume co-movement → long (calm + healthy tape) | Long·locked (IC positive in all segments) |
| alpha016 | −rank(cov(rank(H),rank(V),5)) | High prices on rising volume → short (overheating reversal) | Long·locked |
| alpha026 | −ts_max(corr(ts_rank(V),ts_rank(H),5),3) | Extreme price-volume synchronicity → short (co-moves don't persist) | Long·locked |
| alpha044 | −corr(H,rank(V),5) | High prices with expanding volume → short (short-term overheat) | Long·locked |
| alpha011 | (rank(max(VWAP−C,3))+rank(min(VWAP−C,3)))×rank(ΔV,3) | Extremes of price-vs-VWAP gap resonating with volume changes | Long·locked |
| alpha024 | In sideways regimes (100d MA momentum ≤5%) buy distance from the 100d low; else 3d reversal | Sideways-market oversold bounce | Long·locked |
| dividend_yield | Dividend yield | Cash-flow return | Long·locked |
| bm / pb / ps | Book-to-market / P/B / P/S | Cheapness (three measures of one value cluster) | **Rolling** (as of 2026-06: bm = +1, pb/ps = −1, the persistent post-2020 state; flips next month if the 12-month mean IC crosses zero) |
| VOLT20 / VOLT60 | 20/60-day return volatility | Low-volatility anomaly | **Rolling** (as of 2026-06: −1; same flip rule) |

**The iron survivors and their three-segment record** (IC = mean monthly cross-sectional rank correlation):

| Factor | IC 17-19 | IC 20-22 | IC 23-26 | Late IC | Late ICIR |
|---|---|---|---|---|---|
| alpha040 | +0.045 | +0.072 | +0.081 | 0.077 | 0.59 |
| alpha016 | +0.059 | +0.064 | +0.035 | 0.048 | 0.50 |
| **alpha026** | +0.040 | +0.037 | +0.056 | 0.047 | 0.46 |
| dividend_yield | +0.064 | +0.061 | +0.041 | 0.050 | 0.32 |
| alpha024 | +0.034 | +0.054 | +0.039 | 0.046 | 0.32 |
| alpha044 | +0.031 | +0.043 | +0.036 | 0.039 | 0.43 |
| alpha011 | +0.028 | +0.039 | +0.041 | 0.040 | 0.39 |

Only 10 of the 24 stage-1 candidates pass two-segment survival, 8 pass three-segment — **full-sample IC happily admits "one-regime stories"**. Three failure modes:

1. **Early decay** (surplus_cash +.054→+.018, ocf_to_debt, and 9 more): full-sample averages propped up by history, no recent signal
2. **Style reversal** (bm/pb/ps/VOLT20/60): one direction in 2017-19, the opposite from 2020 onward, strengthening — a regime switch, not noise
3. **Middle-segment dependence** (a013/015/027/050): alive on two segments, collapsing in 2023-26

### 2.3 A data-pipeline case: alpha026 missed and recovered

alpha026 was missed at stage 1 — not for lack of signal. Its formula involves division and the raw data contains 4,154 infinities; the upstream IC summary table was generated without inf-cleaning, and a single inf turns the cross-sectional std into NaN → the factor's IC was reported as NaN everywhere → the screen never saw it. Stage 2 of this study **bypasses the summary table and recomputes all 137 factors directly from the panel with proper cleaning**; alpha026 reappears, passes all three segments positive, and **enters the core pool**.

Lesson, now a pipeline rule: division-based formula factors require `inf→NaN` cleaning; whenever a cleaning rule changes, upstream summaries must be recomputed — otherwise the fix is incomplete.

### 2.4 Boundary test on outside candidates

The strongest outside candidate, ev_to_ebitda (two-segment survivor, cross-sectional correlation with existing factors only ±0.3 — **not redundant** — but three-segment IC decaying −.008/−.043/−.022), was tested for inclusion: **13-factor version 12.90% vs 12-factor 13.10% (−0.2pp, noise-level)** — not included; what it lacks is not independence but a still-strengthening signal. This marks the pool's boundary: adding factors beyond it yields no increment.

---

## 3. Portfolio Construction: Monthly, 12-Month Rolling, Top 20

### 3.1 Monthly decision process (past information only)

```
Month-end t:
  ① each factor: direction s = sign of trailing-12m mean IC (core factors use fixed directions),
                 weight w = |trailing-12m ICIR| / Σ|ICIR|   (ICIR = mean IC / std of IC)
  ② cross-sectional winsorize (1%/99%) + z-score → score = Σ w_f × s_f × z_f
  ③ rank all ~207 pharma stocks, buy top 20 equal-weighted (5% each)
  ④ hold one month; monthly name turnover 60-70%
```

**Two direction regimes**: the 7 iron survivors use **fixed directions** (same sign in all three segments — never re-estimated); the 5 style-reversal factors use **rolling directions** (re-estimated monthly from the trailing 12-month mean IC) — the only workable treatment for factors whose sign flipped in 2020. The rolling sign has **no smoothing**: it flips the moment the 12-month mean crosses zero; the practical "6-12 month lag" is the natural detection time of the 12-month window, not a designed delay. Weights and directions are re-estimated from the same window; there are no fitted parameters anywhere.

### 3.2 Three design choices and their evidence

| Choice | Evidence |
|---|---|
| **Monthly** | Factor evidence lives on 1-month IC; monthly has no phase dimension by construction (quarterly results depend on which month you start) |
| **12-month rolling window** | Same pool: 13.10% vs 11.81% for a 36-month window (+1.3pp); independently confirmed on the 2018+ long window |
| **Top 20** | Inverted-U (parens = rqalpha T+1, same window): Q5 13.10% (13.40%) / **Top20 15.77% (15.69%)** / Top10 13.66% (12.00%) / Top5 10.21% (8.89%) — the same conclusion on both engines at every point, with engine friction growing as concentration rises. Passes all robustness gates: 3 start-offsets spread 0.4pp; 2018+ long window 15.37% vs Q5's 11.35%. Mechanism = dropping the diluted back half of a wide quintile + ~20 stocks is the classic diversification saturation point (5% positions) |

### 3.3 Long-window stress test (from 2018-01, 102 months)

Includes the 2018 bear market and the old style regime (when value/low-vol pointed the other way): **15.37% / Sharpe 0.61 / drawdown −31.3%**. Rolling signs turned themselves around inside the old regime — the iron-core + adaptive-satellite design survives its most hostile window.

---

## 4. Performance, Benchmarks, and Yearly Decomposition

### 4.1 rqalpha T+1 validation (2020-01-23~2026-05-15)

| Metric | Value |
|---|---|
| Annualized return | **15.69%** (total +142.8%, NAV 2.43) |
| Sharpe / Sortino | 0.62 / 0.85 |
| alpha / beta (vs CSI 300) | **13.0% / 0.71** |
| Information ratio / geometric excess | **0.73 / +12.79%** |
| Max drawdown | −31.3% (2023-05~2024-12, 587 days) |
| In-house counterpart | 15.77% — **engine friction ≈ 0** (T+1, price limits, commissions, stamp duty all included) |

### 4.2 Benchmark trio and skewness quantification ([22](22_bench_and_xgb12f.py), geometric, same window)

| Benchmark | Annualized | Top20 excess | Question answered |
|---|---|---|---|
| Money-market fund | +1.75% | **+13.9pp** | Risk-free floor |
| CSI 300 | +2.56% | **+13.1pp** | Did you beat the broad market? |
| SW Pharma Index (000808) | **−3.42%** | **+19.1pp** | What passively holding pharma bought you |
| Pharma cap-weighted | −0.08% | +13.8pp | Large-cap-dominated sector return |
| Pharma equal-weighted (reference) | +4.84% | +10.9pp | No-skill selection expectation (toughest ruler) |
| Typical stock (cross-sectional median) | −8.60% | +24.3pp | The real experience of a pharma stock |

**Skewness quantified**: pharma equal-weight +4.84% vs typical stock −8.60% — **13.4pp/yr of the gap is pure right-tail (lottery stocks)**: a stock can only fall 100% but can rise several-fold, so means are inflated by winners. Equal-weight is therefore demoted to an academic reference; the trio above is the primary lens.

### 4.3 Yearly decomposition ([24c](24c_yearly_decomp.py))

| Year | Top20 | Sector·pharma EW | Sector·SW pharma index | All-A EW | Selection excess (vs pharma EW) |
|---|---|---|---|---|---|
| 2020 | +42.9% | +7.6% | +55.1% | +8.8% | **+35.3pp** |
| 2021 | +15.7% | +13.5% | −13.0% | +24.1% | +2.3pp |
| 2022 | +20.1% | +13.3% | −23.7% | +10.6% | +6.9pp |
| 2023 | −12.4% | −19.8% | −12.4% | −17.6% | **+7.4pp** |
| 2024 | +15.2% | +7.0% | −14.5% | +19.3% | +8.2pp |
| 2025 | +25.1% | +24.8% | +5.3% | +44.4% | **+0.4pp** ⚠️ |
| 2026H1 | −8.5% | −9.2% | −4.3% | −15.3% | +0.7pp |

The two sector columns make "what fell was beta" self-evident: 2023 sector −19.8%/−12.4% vs strategy −12.4%; 2026H1 sector −9.2%/−4.3% vs strategy −8.5%. They also expose how violently the two sector measures disagree year by year (2025: EW +24.8% vs index +5.3% — the equal-weight is inflated by the small-cap right tail, the index is not) — exactly why §4.2 promotes the index to the primary benchmark.

Three honest footnotes: (i) in losing years (2023, 2026H1) excess is still positive — **what fell was sector beta; selection never failed**; (ii) 2020 is an extreme excess year (+35pp, pandemic-rebound factor resonance) — unrepeatable tailwind; excluding it, the remaining years average ≈ +4.3pp/yr, still clearly positive and the right baseline for extrapolation; (iii) **2025's near-zero excess is attributed in §4.4**. (Bookkeeping: the return booked at month m covers m→m+1, so January's move lands in the prior year's row; **the 2026H1 row is the in-house monthly caliber** (prices through 2026-07-17), while the rqalpha window ends 2026-05-15 and excludes the June crash and July rebound — the two 2026 figures are not directly comparable.)

### 4.4 Attributing 2025's near-zero excess ([27](27_year2025_attribution.py))

Three-layer decomposition; the verdict is **a benchmark-inflation year plus localized factor fatigue — not engine failure**:

1. **Factor layer**: 9 of 12 factors kept significant 2025 ICs (alpha016 +0.057, alpha026 +0.072, bm +0.073, VOLT20 −0.086 — all normal). The anomalies: **dividend yield's IC turned negative (−0.007, its only negative year in seven)**, with alpha024 (+0.017) and alpha011 (+0.008) visibly weak — the composite score genuinely dulled, but it was not a collective collapse
2. **Benchmark layer (main driver)**: 2025 was a small-cap melt-up (all-A equal-weight +44.4%); pharma equal-weight +24.8% was **extremely inflated by the right tail** — pharma cross-sectional dispersion (0.101) was the highest since 2021. A 20-stock portfolio structurally cannot hold the full right tail; an equal-weight benchmark with total lottery-stock exposure is nearly unbeatable in a melt-up
3. **Structural check**: core7 alone ran −8.0pp below the sector in 2025, sat5 alone −5.3pp (both legs lagged), yet the combination returned to +0.4pp. Mechanism: **the combination is not a weighted average of the two legs' returns** — it is a different set of 20 names re-ranked on the joint 12-factor score (the two legs' top-20 lists only partially overlap); in 2025 the joint ranking found names neither leg could select, and the legs' negative-excess months were staggered. That is the point of the composite design — but also a reminder: +0.4pp is "barely level," not robust positive excess.

**Monitoring criterion**: in 2026H1 dividend IC is back to +0.101 (strongest in seven years) and the combined excess is positive again (+0.7pp). If excess stays thin in a year when the benchmark is *not* tail-inflated, that would be evidence of genuine engine decay; until then 2025 is classified as a benchmark-inflation year.

---

## 5. Why Not Machine Learning: The Full Matrix

The 12 factors are compressed into 9 cluster composites and fed to XGBoost (shallow, strongly regularized; hyperparameters untouched for this sector), under exactly the same guardrails:

| Configuration | In-house (median of 3 phases/offsets, range in parens) | rqalpha T+1 (ann. / max DD) | One-line verdict |
|---|---|---|---|
| **Transparent × Top20** | **15.77%** (monthly, no phase dimension) | **15.69% / −31.3%** | — (winner) |
| Transparent × Q5 | 13.10% | 13.40% / −30.3% | dilution |
| XGB quarterly × Q5 | 13.4% (range 1.8pp) | 13.94% (range 4.2pp) / −32.5% (deepest −32.7%) | phase-unstable + black box, and deeper drawdowns |
| XGB monthly × Q5 | 10.4% (range 0.2pp) | 7.62% / −28.1% | consistently bad: 1-month labels ≈ noise for trees |
| XGB monthly × Top20 | 12.19% | 8.88% / −30.9% | concentration doesn't rescue; friction −3.3pp |
| XGB 26 factors × Q5 | 12.0% (range 1.4pp) | 10.87% / −30.1% | more features dilute shallow trees |

The drawdown column matters: ML drawdowns (−28~−33%) are the same order as the transparent composite (−30~−31%) — **ML does not buy its return with shallower drawdowns**, and the best ML config earns only +0.5pp while carrying 4.2pp of phase spread.

The best ML configuration (quarterly, rqalpha 3-phase median 13.94%) beats the transparent composite by only +0.5pp, at the cost of phase risk, deeper drawdowns, and opacity. **Conclusion: on a ~207-stock cross-section with 78 monthly decisions, transparent ICIR weighting is the optimum of this problem** — ML's apparent gains are either noise or consumed by execution frictions.

---

## 6. From Research to Operations

Monthly, 20 names, 60-70% monthly turnover ≈ 12-14 order pairs per month, 5% positions — the lowest tier of automation complexity. The signal engine and the execution layer are cleanly separated; T+1, price limits, and real fees are already embedded in the rqalpha validation (friction ≈ 0; limit-up misses are diversified away). Live path: broker quant API (QMT/miniQMT class) fully automates the monthly rebalance — month-end signal → diff target vs holdings → order flow → API execution, failed orders retried next day.

---

## 7. Risks and Failure Conditions (Disclosed)

1. **The satellite is a style bet**: value/low-vol direction has held for 6 years since the 2020 reversal and is strengthening — but "held for 6 years" ≠ "will hold forever". On the next reversal, rolling signs follow with ~1 year of lag and excess suffers in between. The satellite is only 5 of 12 weights.
2. **2025 excess ≈ zero** (+0.4pp): attributed to a tail-inflated benchmark year plus dividend-factor fatigue (§4.4); the monitoring criterion is stated there.
3. Sample: 6.3 years / 78 monthly decisions, including one pandemic outlier year; excluding 2020, average excess ≈ +4.3pp.
4. beta 0.71: the portfolio carries sector beta; a 2023-style sector bear costs −12% — stock selection does not hedge the sector.
5. The factor pool was chosen by sub-period testing (selection degrees of freedom exist); mitigated by 3 phases / 3 start offsets / the 2018+ long window / two-engine reconciliation — but not reducible to zero.

---

## Appendix: Script & Output Index (by function)

**Full pipeline order**: `00 → 10 → 14 → 17 → 24 → 16 → 22 → 27` (QC → screening → pool matrix → lookback → concentration → rqalpha validation → benchmarks → attribution); outputs under `outputs/`.

**Data & QC**: `00_panel_qc.py` (counts / missingness / low-N sensitivity) · `common.py` (single source of truth: cleaning + backtest rules)

**Factor selection**: `10_subperiod_ic.py` (two-stage core: 137→24→7) ★ · `20_add_ev_to_ebitda.py` (boundary test)

**Portfolio & backtest**: `14_core_pool_backtest.py` (pool matrix) · `17_lookback_12m.py` (lookback × three starts) · `24/24b/24c` (concentration + robustness + yearly decomposition) ★

**ML comparison**: `12_embargo_check.py` (label-embargo audit) · `15_ml_newpool.py` · `19/21/23/25` (XGB full matrix / grid unification)

**Validation & outputs**: `16_rqalpha_redo.py` (T+1 final validation) · `22_bench_and_xgb12f.py` (benchmark suite) · `27_year2025_attribution.py` (2025 attribution) · `26_final_artifacts.py` (equity curve / concentration chart / return archive)

Every claim has a corresponding CSV/PNG under `outputs/`; the chain is fully reproducible.
