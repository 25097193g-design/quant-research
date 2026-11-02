# Optiver Realized Volatility — Competition Reproduction

Kaggle *Optiver Realized Volatility Prediction* (2021), reproduced end to end: order-book microstructure → feature engineering → LightGBM / NN ensemble.

> 中文版（详细日志）：[README.zh.md](README.zh.md)

---

## The task

Given the order book (L2 bid/ask prices and sizes) and trade flow for one stock in the **current 10-minute bucket**, predict the **realized volatility of the next 10 minutes**. Scored by RMSPE — a *relative* error, which is why everyone works in log space.

Why a market maker asks this: a dealer earns the spread on two-sided quotes while carrying inventory risk proportional to volatility. Better vol forecasts → tighter quotes, better sizing, cheaper hedging. It is the core of the business, not an academic exercise.

## Benchmark ladder (RMSPE)

| Level | RMSPE |
|---|---|
| Naive (next bucket = current bucket) | ~0.34 |
| Public nearest-neighbour baseline | ~0.29 |
| Medal zone | ~0.21–0.23 |
| Gold | ~0.205 |
| Winner | ~0.19–0.20 |

## Where this reproduction got to

| Step | RMSPE | What moved it |
|---|---|---|
| Naive baseline | 0.3414 | — (matches the published ≈0.34 anchor, so the WAP/RV pipeline is correct) |
| v1 LightGBM | 0.2673 | book + trade features |
| v2 + market clock | 0.2586 | cross-sectional volatility features |
| + `stock_id` categorical | 0.2493 | per-stock heterogeneity |
| Two-stage (market model → residual) | 0.2489 | market-level RMSPE is only 0.127 — market vol is twice as predictable as single-name vol |
| Geometric blend | 0.2458 | ensemble |
| NN with 1/y²-weighted MSE | **0.2350** | see below |

**0.235 lands inside the medal zone** (~0.21–0.23 is the threshold band).

## Three findings worth the space

1. **The market clock.** Same `time_id` across stocks = the same 10-minute market interval. Cross-sectional volatility co-movement has **corr = 0.923** — higher than a stock's own persistence (0.874). A pure market signal alone scores 0.698. A-share sector co-movement is the same phenomenon at a lower frequency.
2. **1/y² weighting beats log-target.** RMSPE is a *ratio* error; `log` only approximates it when errors are small. Weighting MSE by **1/y²** aligns the loss exactly with the metric — worth **−2.3pp on its own**, more than most feature engineering. Alignment with the evaluation metric matters more than the loss's mathematical convenience.
3. **Heterogeneity is real.** `stock_31` has normal levels (RV at the 56th percentile, spread at the 78th) but its next-bucket transition is not predictable — per-stock RMSPE 0.965 versus 0.363 for the second worst. Some names simply do not exhibit volatility clustering.

## Files

| File | What it does |
|---|---|
| `common.py` | WAP, log-return, RV computation, data loading |
| `01_eda.py`, `21_data_review.py` | Structure checks (3,830 time_ids, ~240 snapshots per bucket) |
| `02_naive.py` | Naive baseline — always compute this first |
| `03_features.py`, `08_features_fine.py` | Book / trade / microstructure features |
| `05_marketclock.py` | Cross-sectional volatility features |
| `06_twostage.py` | Market model → per-stock residual |
| `10_multifold.py` | Multi-fold validation |
| `11_nn.py`, `12_nn_deep.py`, `19b_nn_weighted.py` | Neural nets, incl. the 1/y²-weighted variant |
| `20_true_eval.py` | Held-out evaluation protocol |
| `build_book_explainer.py` | Generates the order-book explainer visuals in `outputs/` |

## Data

Kaggle competition data (`~/Downloads/optiver-realized-volatility-prediction`, 2.5 GB, 112 anonymous stocks) — **not redistributable**, so it is not in this repository. `common.py` documents the expected layout; the scripts assume the same directory structure as the competition download.
