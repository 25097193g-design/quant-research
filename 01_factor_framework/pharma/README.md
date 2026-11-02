# Pharma & Biotech — 137-Factor Pipeline and a Rejected ML Branch

The larger half of the factor framework: a 228-name cross-section where de-duplication *costs* alpha and machine learning *loses* to a transparent composite.

> 中文版：[README.zh.md](README.zh.md) · **Full report in English: [REPORT_EN.md](REPORT_EN.md)**

---

## Strategy and result

A **12-factor composite** — 7 core factors that survived sub-period screening, plus 5 style satellites whose signs are re-estimated monthly — weighted by trailing-12-month |ICIR|. Monthly rebalance, long the top 20 equal-weighted.

| Metric | Value |
|---|---|
| Annualised (rqalpha, T+1) | **15.69%** |
| Total return | +142.8% (NAV 2.43) |
| Sharpe | 0.62 |
| Sortino | 0.85 |
| Information ratio | 0.73 |
| Alpha | 13.0% |
| Beta | 0.71 |
| Max drawdown | −31.3% |

Window 2020-01-23 – 2026-05-15. Benchmark CSI 300 +2.6%; the SW Pharma index actually **lost −3.42%/yr** over the window, so excess vs the sector is +19.1pp.

**Selection alpha was positive in 7 of 7 years.** The two losing years (2023, 2026H1) were sector beta, not selection failure — the −31.3% drawdown is the sector, and the report says so.

---

## The ML result (the part worth showing)

XGBoost was not tried once and dismissed. It was tested across the full matrix:

**{XGBoost} × {monthly / quarterly} × {portfolio size} × {start phase} × {two engines}**

**No configuration beats the transparent composite on both engines.** The matrix is in `REPORT_EN.md` §5; the scripts are `19_xgb_newpool_rqalpha.py`, `21_xgb_26f.py`, `22_bench_and_xgb12f.py`, `25_ml_top20_monthly.py`.

The same test in the bank universe was *inconclusive* rather than negative — the model ranking shuffles with the window, so at 20–23 quarterly observations the difference is not resolvable. Reporting the two sectors differently is the point: one is a rejection, the other is "we cannot tell".

---

## Pipeline

```
137 candidate factors
  → 24   full-sample IC initial screen
  →  7   sub-period survival (the "iron survival" set)
  +  5   style satellites on rolling signs
  = 12   factor composite, trailing-12m |ICIR| weights
  →      top-20 equal weight, monthly, rqalpha T+1
```

| Stage | Script |
|---|---|
| Panel QC | `00_panel_qc.py` |
| Two-stage screen | `10_subperiod_ic.py` ★ |
| Pool matrix (core + satellite) | `14_core_pool_backtest.py` |
| Look-back window (12m beats 36m) | `17_lookback_12m.py` |
| Concentration inverted-U → Top20 | `24_topN_concentration.py`, `24b_top20_robust.py` ★ |
| Final rqalpha validation | `16_rqalpha_redo.py` |
| Label-embargo audit | `12_embargo_check.py` |
| Final artefacts | `26_final_artifacts.py` |

★ = start here if you only read two scripts.

---

## Capacity (stated in the report)

2025 sector daily turnover ≈ RMB 52bn; median single-name daily turnover ≈ RMB 158m. Capping any rebalance at 10% of a stock's daily turnover gives **≈ RMB 300m one-day capacity, ≈ RMB 1bn over three days**. Not capacity-constrained at this size.

---

## Caveats the report states up front

1. 2020 alone contributes +35pp of the excess return — the early window is doing a lot of work.
2. **2025 excess fell to ≈ +0.4pp** and is flagged as the single metric to monitor.
3. Start-phase sensitivity: three offsets spread 0.4pp on the Top-20 version, but the group comparison is less stable.
4. The backtest uses an event-driven engine with T+1 and costs; the in-house engine agrees to within 0.08pp (15.77% vs 15.69%), which is what "frictions ≈ 0" means here.

---

## Data

`rqdatac` education edition, SW L1 pharma & biotech, PIT whitelist of 228 names, 115 month-ends (2017-01 – 2026-07). The first 36 months serve as rolling-window warm-up; the validation window is 78 months. Licensed data is not redistributable.
