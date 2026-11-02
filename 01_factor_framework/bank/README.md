# Bank Sector — Factor Workflow, De-duplication and Validation

The companion to [`../pharma/`](../pharma/). Same framework, opposite conclusion. The bank universe is small (41 names), so almost every standard screening rule breaks — the interesting work is in the corrections.

> 中文版：[README.zh.md](README.zh.md) · Full write-up (Chinese): [REPORT.md](REPORT.md)

---

## Strategy and result

Value + quality + low-β + momentum, ICIR-weighted cross-sectionally, long the top quintile (**~8 banks**), quarterly rebalance, equal weight, **T+1 execution**.

| Metric | Value |
|---|---|
| Annualised | **16.0%** |
| Sharpe (daily) | **0.75** |
| Information ratio | **0.99** |
| Alpha | 14.6% |
| Beta | 0.39 |
| Max drawdown | 17.3% |
| Benchmark (CSI 300, same window) | −0.98% |
| Investable comparison (bank ETF 512800) | 5.90% / −27.7% DD |

Window: 2020-12 – 2026-05, rqalpha event-driven engine with costs.

**Reported honestly:** the engine's three start phases span **10.7%–16.0%**, and the current phase happens to be the highest — roughly 3–5pp of the point estimate is start-month luck. The `-1%` oversold filter version *underperforms* the equal-weight market on a same-window basis.

---

## Why a 41-stock universe needs different rules

Three corrections that a generic A-share pipeline would get wrong:

1. **Threshold inflation.** With N ≈ 36 per cross-section, the standard error of a monthly IC is ≈ 1/√35 ≈ 0.17. The industry default `|IC| > 0.02` is far too loose here; this workflow uses `|mean IC| ≥ 0.05` **and** `|ICIR| ≥ 0.3` (qualified), or `≥ 0.04 / ≥ 0.25` (candidate, used to expose redundancy).
2. **Long-horizon ICIR is inflated.** The forward-return labels overlap 11/12 months at h=12, so the IC series is autocorrelated and its std is compressed — `MOM_12M_1M` jumps from ICIR 0.22 at 1m to 1.45 at 12m partly from this artefact alone. Robustness therefore uses **1m/3m only**; IC *means* are comparable across horizons, ICIR *levels* are not.
3. **Don't drop low-N months.** The PIT whitelist only reaches 30 names in 2020-01; dropping low-N months would delete the entire 2017-2019 regime and flipped five 1m-qualified factors. The workflow keeps the months and runs sub-period sensitivity instead.

---

## Four traps found in the data (the real output of this project)

1. **`inf` contamination.** `alpha026` carried 82 `inf` values that silently poisoned 22 months — dropping loss months, inflating PCA returns, and manufacturing fake phase sensitivity. After the fix, PCA phase dispersion fell from 5.9pp to 0.65pp.
2. **Static-ICIR overfitting.** A 7-factor set showed 16.4% on a short window; under walk-forward it collapsed to 6–7%. The 16.4% was a look-ahead artefact of estimating weights on the full sample.
3. **Benchmark window mismatch.** "Equal-weight market 7%" was a 2017–2026 full-sample figure while the strategy starts in 2020; the same-window benchmark is 9.9–12.0%, which shrinks the excess return from ~10pp to ~1–2.4pp.
4. **Low-β softness in rallies.** In 2025-H2 the bank index rose +11% / +8.8% while the 4-factor strategy lost −4.5% / −3.0%.

A single `common.py` is the source of truth for panel construction; before it, two scripts that were supposed to compute the same PCR differed by 1pp.

---

## Files

| File | What it is |
|---|---|
| `CONCISE.md` | One-page summary including the four traps (Chinese) |
| `REPORT.md` | Full write-up (Chinese) |
| `PRESENTATION_EN.md` | English slide-style summary |
| `00_panel_qc.py` | Panel QC: name counts per cross-section, low-N sensitivity |
| `01_factor_select.py` | IC/ICIR screening with the small-universe correction |
| `02_factor_correlation.py` → `04_dedup_vs_full.py` | Correlation, hierarchical clustering, de-dup test |
| `06_rqalpha_backtest.py` | Exchange-level backtest |
| `09_quarterly_phase.py` | The three-phase robustness check |
| `11_sector_gate.py`, `12_rqalpha_variants.py` | Variants tested and rejected |
| `figures/equity_curve.png` | Net value curve |

---

## Data

`rqdatac` (Ricequant education edition): 137 candidate factors = WorldQuant 101 + 29 Ricequant fundamentals + 7 self-computed price-volume; SW L1 industry classification; PIT whitelist. Licensed data is not redistributable.
