# A-Share Bank-Sector Factor Strategy — End-to-End Research (Final · Aug 2026)

**Core conclusion 1 (signal)**: after 2020, the only cross-period-stable alpha factor in the bank cross-section is the **dividend yield** — of 137 factors, it alone survived the 2020 break; everything else is auxiliary (diversification and risk control, no independent signal).

**Core conclusion 2 (source of returns)**: the strategy's excess return comes **entirely from stock selection** — roughly +6.0pp vs bank equal-weight (hand-written backtest) and +10.1pp vs the bank ETF (engine backtest). Timing contributes no return; it only cuts drawdown.

**Core conclusion 3 (engine validation)**: rqalpha realistic execution (4F · quarterly · T+1 · real commissions & slippage) returns **16.0% annualized**, alpha 0.146, beta 0.39.

| Convention | Value |
|---|---|
| Universe | SW "Banks" PIT whitelist (15 stocks in 2017 → 41 by 2026) |
| Data | Monthly factors 2017-01 ~ 2026-07 (rqalpha engine limited by bundle to 2026-05-15) |
| Backtest window | **Unified start Dec-2020** (rationale in §1) |
| Benchmarks | All-A equal-weight 9.9% (same window) · bank equal-weight 5.82% · bank ETF 5.90-5.96% · CSI 300 −0.98% (inside engine) |
| Costs | Hand-written: 15bp/period (conservative); engine: real per-trade commissions + slippage |

> **Validation philosophy**: this study went through nine rounds of corrections, each overturning part of the previous round (full post-mortems in §8). Every surviving conclusion passed a "survives across windows / across implementations" test. **Read §8 before trusting any number** — it is also the confidence boundary of this report.
>
> Note: "convention" (口径) = the complete set of settings behind a number (sample window, benchmark, costs, frequency, compounding). Used 30+ times in this sense.

---

## §1 Why the backtest starts in Dec-2020

> **Conclusion first**: **factor screening uses the full sample (from 2017); performance reporting starts Dec-2020. No data was deleted — only a segment unfit for the stage.** Three technical reasons below.

**Why the reporting window starts Dec-2020 (three interlocking facts)**:

1. Banks listed gradually: the whitelist held only ~15 names in 2017, **crossed 30 only in Jan-2020**, and reached 41 by end-sample. Dec-2020 is exactly where the 36-month rolling weights finish warming up.
2. The "whitelist hits 30" date (2020-01), the factor-strength break (§2), and the warm-up completion point (2020-12) coincide — **2020 is a regime line for bank factors**.
3. Before that, cross-sections are under 30 names and weights are not warmed up; showing early performance would mislead.

**Why low-N months are NOT dropped in screening**: each month's IC is unbiased; dropping months by cross-section size deletes the entire 2017-2019 regime (empirically it wrongly kills 5 qualified factors: beta60 / ev_to_ebitda / pe_ttm / pe_lyr / net margin). The right tool is a sub-period test (§2), not deletion.

![Panel QC](outputs/panel_qc_en.png)

---

## §2 After 2020, only one factor survives: dividend yield

**Sub-period IC test (script 10)**: all 137 factors are evaluated separately in the early (2017-2019) and late (2020-2026) windows with IC/ICIR; `z_diff` tests the strength change. Survivor = late-significant + same sign + |z_diff| < 2.

| Verdict | Factor | Evidence |
|---|---|---|
| ✅ **only cross-period survivor** | **dividend_yield** | early +0.161/0.49 → late +0.104/0.34, z = −0.9 (also survives at 3m horizon) |
| ⚠️ near miss | alpha026/088, pe_ttm/pe_lyr, beta60 | late IC ≥ 0.05 and stable, but late ICIR 0.26-0.28 just under 0.3 |
| ❌ one-era story | valuation / quality / size / momentum families | early ICIR 0.4-0.6 → late 0.15-0.28, z negative across the board |
| ❌ sign flip | DAVOL20 | z = −4.2 |

**Reading**: after 2020 the bank cross-section has exactly one real independent story — dividends; the other members of 4F/7F are auxiliaries (diversifiers + style tilts), not independent signals. The engine quantifies it: **the single dividend factor alone reaches 13.5-14.1% ≈ 85% of 4F**. This is also why ML has no food here (§7) — true signal dimension d ≈ 1.

**Engine-basis magnitude estimate (not precise attribution, and not a decomposition of the hand-written +6.0pp)**: single-dividend absolute return 13.51% vs bank ETF 5.90% ≈ +7.6pp; full 4F 16.00% vs single-dividend ≈ +2.5pp; the two gaps sum to ≈10.1pp, consistent with the engine-basis excess. ⚠️ portfolio returns are not linearly additive (rebalancing/compounding/factor covariance) — these gaps indicate magnitude only, not orthogonalized factor contributions; exact attribution needs Brinson or factor-return regressions.

**Macro hypothesis on the 2020 break (narrative, not evidence)**: after LPR reform, net interest margins entered a narrowing-and-diverging channel; combined with property-exposure repricing, "cheap / leveraged" banks turned from value traps into risk itself. Meanwhile an asset famine kept pricing the dividend. Post-hoc narrative to give the statistics an economic intuition; no event-study validation.

![Two-period IC](outputs/subperiod_scatter_en.png)

---

## §3 Timing is a brake (and a seat-belt), not an engine — gate head-to-head (11)

Design: A no-gate (always in Q5) / B per-stock gate (the original 04 implementation) / C sector-level gate / D pure sector timing / E sideways-market gate. Idle months hold cash at 2%/yr; the gate look-back is the true previous 3 months.

| Variant | Full-window ann. | Sharpe | Max DD | Exposure | Trigger-period ann.* |
|---|---|---|---|---|---|
| **A no-gate** | **11.80%** | 0.83 | −17.8% | 99% | 11.9% |
| B per-stock gate | 4.75% | 0.62 | −10.2% | 25% | 13.0% |
| C sector gate (<−1%) | 7.55% | **1.22** | **−4.3%** | 28% | **21.9%** |
| D pure sector timing (<−3%) | 7.23% | 0.71 | −9.1% | 21% | **27.4%** |
| E sideways gate | 0.04% | 0.00 | −20.7% | 18% | **−9.1% (falsified)** |

> \* **Trigger-period annualized = the return pace during actually-held months only**, not the full-window return. Full window = trigger return × exposure + 2% cash × idle share (e.g. C = 0.28×21.9% + 0.72×2% ≈ 7.6% ✓ matches 7.55%). **E's −9.1% is negative = held months lose money on average** (not "the strategy loses 9.1% overall") — the sideways-market hypothesis is falsified by that very number.

Three conclusions: (i) on total return every gate subtracts (A is highest) — the *conditional* statement "after a drawdown there is a rally" is true (trigger pace 21.9-27.4%) but triggers are too rare (21-28% of months) to compound; (ii) **the real alpha is selection: Q5 11.80% vs bank equal-weight 5.82% = +6.0pp** (same sector, same window — the only difference is which names you hold); (iii) "sideways-then-breakout" is falsified. **Timing is a brake on return and a seat-belt on risk**: C's 7.55% < A's 11.80%, but Sharpe 1.22 > 0.83 and drawdown −4.3% ≪ −17.8% — under a risk-adjusted or drawdown-controlled mandate (insurance/pension money) the ranking reverses.

---

## §4 The factor pipeline: 137 → 18 → 8 → 4

- **01 Screening**: small-universe threshold correction (cross-section N ≈ 36, single-month IC SE ≈ 1/√35 ≈ 0.17, so the industry's |IC|>0.02 rule is far too loose here) → |IC| ≥ 0.05 & |ICIR| ≥ 0.3 → 6 qualified at 1m, 14 at 3m; union 14 + 4 candidates = **18 factors analysed**.
- **02 Correlation**: hierarchical clustering (distance = 1 − |corr|) → **8 clusters**; strongest redundancies pe_ttm/pe_lyr 0.97, the market-cap pair 0.91; **counter-intuitive finding: for banks, "low beta" and "large cap" are the same bet** (5 factors in one cluster).
- **03 Health checks**: PCA effective dimension **18 → PR 7.0; 7F → 4.2** (quantitative proof that stacking factors = overfitting); WorldQuant alphas' monthly autocorrelation 0.03-0.22 → **not tradable at monthly frequency** (a turnover hard constraint); all 8 representative factors pass monotonicity (full-sample basis).

**Portfolio members (used throughout; +/- = inclusion direction, by full-sample 3m IC sign)**:

| Set | Members (direction, meaning) |
|---|---|
| **4F** | dividend_yield_ttm (+, high yield) · ev_to_ebitda_ttm (−, cheap) · beta60 (−, low beta) · MOM_12M_1M (+, high momentum) |
| **7F** | 4F + pe_ratio_ttm (−, low PE) · market_cap_3 (+, large) · net_profit_margin_ttm (+, profitable) |

> Note: MOM_12M_1M has **positive** IC in the bank cross-section (early +0.15 / late +0.05) — the opposite of the all-A "momentum is reversal" pattern. MOM and size weakened to insignificance after 2020 (§2 near-miss tier) but never flipped sign.

![Correlation heatmap](outputs/factor_corr_heatmap.png)

*Figure: correlation heatmap of the 18 factors, ordered by the clustering tree (Chinese title: "18 qualified bank factors, hierarchical-cluster ordered"); axis labels are factor names.*

---

## §5 De-redundancy & frequency: the death of static ICIR + the current matrix (04/05/09)

**04 (the decisive control)**: same backtest, only the factor set changes. Under static ICIR (full-sample weights = look-ahead) 7F wins at 16.19%; switch to rolling ICIR and 7F collapses while 4F holds — **redundant factors helped fake an illusion when fed future information**. The old 16.4% headline is sentenced to death (its composition ≈ 2.4pp start-point luck + ~5pp paper parking + 1.6pp implementation drift + the real selection core).

**05/09 current matrix** (unified Dec-2020 window, common implementation, same-window benchmark 9.86-9.89%):

| No filter | Monthly | Quarterly |
|---|---|---|
| 4F | 11.94% / 0.83 | 12.61% / 0.87 |
| 7F | 12.90% / 0.93 | **14.77% / 1.01** |
| PCA | 14.05% / 0.86 | 8.93% (the old "most stable" is dead) |

**09 phase check (hand-written)**: 4F dispersion **2.12pp (most stable)**, 7F 2.77pp, PCA 3.29pp (least stable) — no single-phase point estimate may be quoted alone; 4F is reported as the 10.5-12.6% range. The −1% drawdown filter loses in all six cells (same direction as §3).

**Yearly returns (4F · monthly · hand-written, from Dec-2020)**:

| Year | 4F | Bank EW | All-A EW | Reading |
|---|---|---|---|---|
| 2020* (Dec only) | +5.1% | −1.8% | −6.0% | |
| 2021 | −2.9% | −8.0% | **+24.1%** | risk-on year: defence shows (−2.9 ≫ −8.0) but absolute loss, far behind the market |
| 2022 | +1.3% | −3.8% | +10.6% | small positive in a choppy year |
| 2023 | **+22.6%** | −0.4% | −17.6% | **the selection-alpha year**: banks flat, strategy +22.6% on selection |
| 2024 | +31.9% | **+38.2%** | +19.3% | broad bank rally: strategy trails equal-weight by 6pp — selection drags in beta rallies |
| 2025 | +7.1% | +10.3% | **+44.4%** | big-market year: strategy behind |
| 2026* (H1) | +2.4% | −8.3% | −15.3% | bank drawdown: defence again |

**Shape conclusion**: relative strength concentrates in "bank-flat-but-internally-dispersed" years (2023). In broad bank rallies (2024) and risk-on markets (2021/2025) it cannot win — **that is the inherent shape of a selection strategy, not a flaw; but the capital must match the mandate** (buy the ETF for beta; use this when you want dispersion-year alpha).

![Monotonicity](outputs/factor_monotone.png)

*Figure: quintile mean forward returns for the 8 representative factors (Chinese title: "5-group monthly mean forward returns, oriented high = good"); factor names in Latin script.*

---

## §6 Engine validation: T+1, commissions, slippage, lots, suspensions (06/12/13/14)

**First, why the hand-written 11.80% and the engine 16.00% are not the same number** — the two measurement stacks differ item by item:

| Item | Hand-written | Engine |
|---|---|---|
| Signal | 4F rolling-ICIR Q5 | identical (bit-for-bit, self-checked) |
| Rebalance | monthly | **quarterly** |
| Costs | 15bp/period (conservative; true turnover needs ~5bp) | real per-trade commissions + slippage |
| Suspension / lots | not modelled | real order rejection (2021-12 Ningbo bank case), lot rounding |
| Execution | month-end close | T+1 (next trading day) |

**Step-by-step decomposition of 11.80% → 16.00%**:

| Step | Change | Contribution |
|---|---|---|
| 11.80% → 14.47% | same monthly signal; engine real costs/matching/suspensions (cost basis 15bp→real ~5bp ≈ +1.3pp; daily compounding/matching ≈ +1.4pp) | **+2.67pp** |
| 14.47% → 16.00% | monthly → quarterly (half the trading friction) | **+1.53pp** |
| **Total** | | **+4.20pp** |

The gap between the two excess numbers (+6.0 vs +10.1) is 100% measurement stack — the benchmarks themselves are equivalent (EW 5.82% ≈ ETF 5.90/5.96%).

**06 (4F · quarterly · T+1 · vs CSI 300 −0.98%)**: **16.0% annualized / Sharpe 0.75 (daily) / alpha 0.146 / beta 0.39 / IR 0.99 / Sortino 1.10**, max DD −17.3%. A same-day-close variant (15.0%) confirms insensitivity to execution timing.

**12 full matrix** (self-check: 4F·quarterly = 16.00% bit-identical to 06 → no drift. **Absolute returns**, net of engine commissions/slippage, suspensions handled; alpha/IR columns benchmark CSI 300):

| Variant | Ann. | Sharpe (daily) | alpha | beta | IR |
|---|---|---|---|---|---|
| Dividend-only · Q | 13.51% | 0.75 | 0.119 | 0.35 | 1.07 |
| 4F · Q | 16.00% | 0.75 | 0.146 | 0.39 | 0.99 |
| **7F · Q** | **17.79%** | **0.86** | **0.164** | 0.39 | **1.13** |
| Dividend-only · M | 14.12% | 0.79 | 0.126 | 0.36 | 1.13 |
| 4F · M | 14.47% | 0.68 | 0.130 | 0.38 | 0.89 |
| 7F · M | 15.44% | 0.77 | 0.139 | 0.35 | 1.02 |

Readings: **7F·Q is the engine champion and matches the hand-written direction (14.77>12.61 ↔ 17.79>16.00) — one of the few rankings that survive across implementations**; **single dividend ≈ 85% of 4F** (engine quantification of §2); under real commissions the monthly-quarterly gap narrows and single-dividend monthly actually beats quarterly. The engine also surfaced real frictions invisible to hand-written code (order rejection on a suspended stock).

**13 phase check (engine, start month shifted by one either side) — honest disclosure**:

| Config | phase0 (Nov-20) | **phase1 (Dec-20, current)** | phase2 (Jan-21) | Dispersion |
|---|---|---|---|---|
| 4F · Q | 11.98% | **16.00%** | 10.70% | **5.29pp** |
| 7F · Q | 13.85% | **17.79%** | 12.43% | **5.36pp** |

⚠️ **The current phase happens to be the best of the three** (the hand-written set shows the same: Dec-20 is its max too) — **16.0%/17.8% contain ~3-5pp of start-month luck**; honest quoting is the range: 4F engine 10.7-16.0% (median ~12.0%), 7F 12.4-17.8% (median ~13.9%).

**Monthly vs quarterly stability (14, start-shift test)**: shifting the monthly start moves annualized by only **2.4pp** (4F: 12.07/14.47/12.65; 7F: 12.98/15.44/14.02) — half the quarterly dispersion (5.3pp), because monthly rebalancing has **no calendar lottery** (every month is a rebalance); what remains is the window-edge effect (Nov-Dec 2020 happened to be a bank rally, so the current start is again the max). **On medians monthly ≥ quarterly** (4F 12.65 vs 11.98; 7F 14.02 vs 13.85) with a higher floor, at the cost of 1-3pp deeper drawdown. One line: **monthly is the steadier choice; quarterly is the gamblers' choice — its 16.0%/17.8% holds only on the lucky phase.**

**Investable benchmark (rqdatac)**: bank ETF 512800 returns **5.90%** annualized over the identical window (2020-12~2026-05-15; 5.96% extended to 2026-07) — indistinguishable from the self-built equal-weight 5.82%: **two ways of "holding all banks" are equivalent**, yet the ETF draws down −27.7%. **Strategy vs investable bank beta: +10.1pp/yr with a shallower drawdown** (hand-written basis ≈ +5.9pp) — the most down-to-earth answer to "is selection worth anything".

**Capacity (rough, conservative)**: 2024 median daily turnover of the head banks is RMB 0.9-2.1bn; with 8 equal-weight names, single-day build, 5% participation → **≈ RMB 0.3-0.4bn**; a 3-day build roughly triples it and the actual 34% monthly turnover dilutes impact further.

![Equity curve](outputs/netvalue_4F_T1_en.png)

---

## §7 ML head-to-head: differences are unresolvable (07/08)

Four versions of the conclusion, each overturning the previous: "ML loses across the board" (an inf-bug artifact) → "XGB leads" (2020-01 window) → "Lasso best" (2020-12 window) → **final: on 20-23 quarters, differences between models are unresolvable**. Moving the window by just 4 quarters sent XGBoost from first (11.8%) to last (8.3%) and Lasso from mid to first (12.7%) — **the variance of model choice exceeds the true difference between models**. The companion "raw-18 drawdown doubles / clustering regularizes trees" died with it. Mechanism (§2): true signal dimension d ≈ 1 — **intuition: with a single real signal, linear weighting already sits at the bias-variance optimum; complexity buys variance, not bias reduction**. The only stable row: the ICIR-weighted baseline (≥9.2% in both windows).

---

## §8 Nine rounds of correction (the core asset of this study)

**Three classes of lessons first** (all nine rounds attributed):

| Lesson class | Rounds | One-line lesson |
|---|---|---|
| **Data / indexing** | #1 inf, #5 date index, #6 end-check, #9 gate look-back | data hygiene precedes modelling; when changing window definitions, every "look back from current month" index must explicitly use the global position |
| **Window / convention** | #2 backtest drift, #3 benchmark window, #4 Lasso window, #6 T+1, #7/#8 window unification | every "which is best" ranking is a function of conventions; excess must be same-window and frequency-labelled |
| **Over-reading conclusions** | #4 "Lasso clearly worse", #7 ML ranking, #8 "PCA most stable" | on small samples a ranking is noise; a conclusion may only leave the building if it survives across windows and implementations |

**Full archive (appendix)**:

| # | Correction | Dead conclusion |
|---|---|---|
| 1 | inf→NaN cleaning (alpha026 carried 82 infs) | "PCA 16.4%", "PCA phase least stable 5.9pp" |
| 2 | common.py single source of truth | ~1pp drift between 05 and 09 |
| 3 | benchmark same-window as strategy | "+6pp excess" (actually +2~3pp) |
| 4 | month-grouped CV folds + common-months basis | "Lasso clearly worse, 8.9%" |
| 5 | backtest date index + two latent bugs | equity-curve time axis wrong |
| 6 | rqalpha T+1 + end-date check | "bundle ends 2025-05" (doc error); execution-timing doubts (dual-variant check: insensitive) |
| 7 | 07/08 window alignment | "XGB leads", "raw-18 drawdown doubles" |
| 8 | 05/09/11 window unification | **"PCA quarterly most stable, 0.65pp" (→ 8.93% bottom)** |
| 9 | gate look-back index bug (caught by the author's "feels off") | "sideways gate Sharpe 1.46" (→ falsified); "sector gate worthless" (→ actually a real risk filter) |

---

## §9 External numbers & boundaries

**Quote list**:

1. **rqalpha 4F · quarterly (T+1) current phase 16.0% / alpha 0.146 / beta 0.39 / IR 0.99** (vs CSI 300 −0.98%); **engine 3-phase range 10.7-16.0% (median ~12.0%; current phase happens to be the max — start-month luck disclosed)**; 7F · Q engine 17.79% / alpha 0.164 as a footnote (direction consistent with hand-written; its range 12.4-17.8%).
2. **Selection excess**: hand-written basis ≈ +5.9pp (11.80% vs bank EW 5.82% / ETF 5.96%, benchmarks equivalent); engine basis +10.1pp (16.0% vs ETF 5.90%, DD −17.3% vs −27.7%).
3. Sector drawdown gate — a brake on return, a seat-belt on risk (Sharpe 1.22 / DD −4.3%, suited to risk-averse mandates).
4. **The only cross-period-stable factor after 2020 is the dividend yield** (the rest are auxiliaries: diversification + style tilt; single-dividend engine 13.5-14.1%).

**Boundaries (volunteered)**: single window Dec-2020 ~ May/Jul-2026 (a bank-favourable era, includes sector tailwind); 41-stock small cross-section (~50 effective independent observations); the all-A equal-weight benchmark is inflated by the small-cap tailwind (so the stated excess is a floor); 4F phase range 10.5-12.6% (no single points); signal dimension d ≈ 1 → ML has no room (proven); capacity ≈ RMB 0.3-0.4bn (conservative, §6).

> One line: **not a complicated method — nine pits dug out one by one, a post-mortem for every dead conclusion, and in the end knowing exactly what this strategy earns (selection + dividends) and what it does not (timing, factor stacking, ML).**