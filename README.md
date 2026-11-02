# Quant Research Portfolio

**Li Changhao** — Physics BSc + MSc Business Analytics (PolyU, Hong Kong). Seeking quantitative research / data-science roles.

I build factor research end to end — A-share data engineering → IC screening → cluster de-duplication → rqalpha exchange-level backtests — and I put comparable effort into proving **when a method fails**. Every project here ends with a stated boundary of validity, not just a number.

---

## Selected results

| # | Project | Method | Headline | Folder |
|---|---|---|---|---|
| 01 | **Cross-sector factor framework** (bank + pharma) | 137 factors → IC screen → cluster de-dup → rqalpha T+1 | Bank **16.0% / Sharpe 0.75 / IR 0.99 / β 0.39**; pharma **15.69% / Sharpe 0.62 / IR 0.73**; XGBoost tested across the full matrix and **rejected** | [`01_factor_framework/`](01_factor_framework/) |
| 02 | **Disclosure tone: dictionary vs FinBERT** | 150-sentence stratified hand audit; 8,745 10-K MD&A; measurement-error test | FinBERT **F1 0.667 / κ 0.60** vs dictionary **0.412 / 0.219**; dictionary precision 0.27 at recall 1.00 | [`02_disclosure_tone/`](02_disclosure_tone/) |
| 03 | **Tariff counterfactual** (working paper) | SCM (cvxpy) / CausalImpact (BSTS) / event study / mediation | Counterfactual gap ranges **−6.73% to +7.72%** across donors and methods → the contribution is the sensitivity boundary | [`03_tariff_paper/`](03_tariff_paper/) |
| 04 | **Optiver realized volatility** | LightGBM + NN, market clock, loss alignment | naive RMSPE **0.3414 → 0.235** (medal zone ≈ 0.21–0.23) | [`04_optiver_rv/`](04_optiver_rv/) |
| 05 | **Falsification studies** (A-share) | Capital flow / ETF rotation / quality score | Four popular signals, all rejected on point-in-time-corrected data | [`05_falsification/`](05_falsification/) |
| 06 | **Equity research** (18 names) | One five-step framework, uniformly applied | 18 companies × 5 sectors × HK/A/US, each with a stated valuation range and a falsifiable trigger | [`06_stock_research/`](06_stock_research/) |

---

## Repository map

```
01_factor_framework/   bank + pharma factor strategies (full pipeline, reports)
02_disclosure_tone/    text measurement: LM dictionary vs TF-IDF vs FinBERT
03_tariff_paper/       causal-inference working paper (LaTeX + notebooks)
04_optiver_rv/         Kaggle competition reproduction: order book → volatility
05_falsification/      four negative results, documented
06_stock_research/     18 one-page theses, five-step framework
```

Each folder has its own README with: the question, the data, the method, the result, and what would make the result wrong.

---

## What connects the six

Three things show up in every project, and they are the reason to read them together:

1. **A stated failure boundary.** The bank/pharma framework inverts across sectors — de-duplication is harmless in banks but costs alpha in pharma. The tariff paper reports a range, not a point estimate, because donor choice moves the answer by 14 percentage points. The disclosure-tone study is a bounded null (±20 bps/SD via TOST), not "no result".

2. **Cost and leakage treated as first-class.** Every backtest is exchange-level (rqalpha, T+1) with costs; every factor screen is point-in-time; every cross-validation has an embargo where overlap exists.

3. **Rejected ideas are kept.** `05_falsification/` exists because four plausible signals did not survive testing. Deleting them would have made the portfolio look better and the research worse.

---

## Reproducibility

Raw data is **not** included (size and licensing). Each folder's README states exactly which data is needed and where to get it:

- A-share: `rqdatac` / `rqsdk` (Ricequant) — education edition, plus `akshare` for fundamentals
- US filings: SEC EDGAR full-text (`AnnualReports1618.csv`); CRSP/COMPUSTAT fundamentals for the regression part are licensed and not redistributable
- Orders book: Kaggle Optiver competition data
- Market data: yfinance / 同花顺

Scripts are ordered and numbered; each folder's README names the entry point.

---

## Contact

- Email: 25097193g@connect.polyu.hk
- GitHub: [25097193g-design](https://github.com/25097193g-design)
- Location: Hong Kong
