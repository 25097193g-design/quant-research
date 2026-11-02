# 01 · Cross-Sector Factor Research Framework

One pipeline, applied to two sectors that behave in opposite ways. The framework is the deliverable; the two strategies are tests of it.

> **Pipeline**: 137 candidate factors → IC screening → hierarchical-cluster de-duplication → multi-model comparison → **rqalpha exchange-level backtest (T+1, with costs)**

---

## Results

| | Bank | Pharma & biotech |
|---|---|---|
| Universe | SW Bank, PIT whitelist, 41 names | SW Pharma, PIT, 228 names |
| Factors kept | 18 → 8 clusters | 137 → 24 → 7 (12-factor composite) |
| Backtest window | 2020-12 – 2026-05 | 2020-01 – 2026-05 |
| Annualised | **16.0%** | **15.69%** |
| Sharpe | **0.75** | 0.62 |
| Information ratio | **0.99** | 0.73 |
| Beta | 0.39 | 0.71 |
| Max drawdown | 17.3% | −31.3% |
| Benchmark (CSI 300, same window) | −0.98% | +2.6% |

Both are reported net of costs, from the rqalpha event-driven engine. The bank strategy also reports its **10.7–16.0% start-phase range** rather than the best phase — the point estimate carries ~3pp of start-month luck.

---

## The cross-sector contrast (the actual finding)

The same pipeline inverts when the cross-section changes:

| Question | Bank (41 names) | Pharma (228 names) |
|---|---|---|
| Does de-duplication help? | Harmless — factors are already near-orthogonal | **Costs alpha** — correlated factors carry independent signal |
| Does ML add value? | Indistinguishable from baseline | Loses to the transparent composite; **rejected after a full matrix test** (frequency × portfolio size × start phase × engine) |
| What breaks first? | Low-β softness in strong rallies | Sector beta in a cycle downturn (the −31.3% drawdown) |

This is the contribution: a quantitative characterisation of **how factor structure degrades as the cross-section widens and industry attributes change**.

---

## Research discipline (three traps found and fixed)

1. **Inf contamination** — `alpha026` had 82 `inf` values that silently poisoned 22 months of backtest (dropping loss months and manufacturing fake phase sensitivity).
2. **Static ICIR overfitting** — a 7-factor set showed 16.4% on a short window; walk-forward collapsed it to 6–7%.
3. **Benchmark mismatch** — "equal-weight market 7%" was a full-sample number while the strategy started in 2020; the same-window benchmark is 9.9–12.0%, so excess return is far smaller than it first appeared.

Fixing these cost returns and gained credibility. `common.py` is the single source of truth for panel construction, which removed a 1pp drift between two scripts that were supposed to compute the same PCR.

---

## Where to start reading

**Pharma**
- `REPORT_EN.md` — full write-up (executive summary, method, ML rejection matrix, capacity analysis)
- `01_factor_select.py` → `05_freq_sets_matrix.py` → `06_rqalpha_backtest.py` — the core pipeline
- `19_xgb_newpool_rqalpha.py` — the ML test that lost
- `figures/equity_curve.png`

**Bank**
- `CONCISE.md` — one-page summary including the four traps
- `REPORT.md` — full write-up
- `03_factor_relationships.py` → `06_rqalpha_backtest.py` — core pipeline
- `figures/equity_curve.png`

---

## Data

A-share prices, fundamentals and industry classification via `rqdatac` (Ricequant education edition), with `akshare` for cross-checks. The licensed data is not redistributable; scripts assume it is available locally, and `00_panel_qc.py` documents every column the pipeline expects.
