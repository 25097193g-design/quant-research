# 05 · Falsification Studies

Four signals that looked plausible, were tested properly, and **did not survive**. This folder exists because deleting it would have made the portfolio look better and the research worse.

Every study here is written as: what the signal was, how it was tested, what killed it, and what that implies for the next attempt.

---

## 1. Block-trade factor — rejected

**Idea.** Large block trades at a discount signal informed selling; follow the direction.

**What killed it.** After point-in-time correction the long-short portfolio's **Sharpe is 0.34** — indistinguishable from noise once costs are applied. The raw signal was partly a look-ahead artefact of how block-trade data is timestamped.

**Write-up.** `block_trade/BLOCK_TRADE_REPORT.md` (the full investigation, including a data-source detour after a vendor rate-limit).

---

## 2. Capital-flow signals — rejected

Northbound flows, margin financing, main-force net inflow: IC ≈ 0 on point-in-time data, in every specification tried.

**Files.** `block_trade/margin_ic.csv`, `block_trade/fund_flow_ic.py`, `block_trade/block_trade_ic.csv`

---

## 3. ETF sector rotation — rejected

Momentum rotation across sector ETFs, tested across the parameter grid.

| | Annualised | Sharpe | Max DD |
|---|---|---|---|
| Rotation strategy | 3.38% | 0.153 | −44.2% |
| Buy-and-hold benchmark | 3.20% | 0.153 | −42.6% |

**The strategy and its benchmark have the same Sharpe to three decimals.** After round-trip costs the reversal variant has an IR of ≈0.05. There is no rotation alpha in this universe at this frequency.

**Files.** `etf_rotation/rotation_backtest.py`, `rotation_stats.csv`, `rotation_scan.csv`

---

## 4. Quality score — a filter, not alpha

An 8-dimension fundamental score applied to the entire A-share universe (4,885 stocks, v1→v4 iterations with multi-year trend and growth dimensions).

**Finding.** Over 2020–24 the score has **negative** predictive power for forward returns. It works as a *screening* layer — removing red-flag names — but it does not generate alpha, and using it as a ranking signal loses money.

That is why the multi-factor strategies in `01_factor_framework/` use quality as a *pool filter* and take their alpha from price-volume factors, not from the score itself.

**Files.** `quality_score/README.md`, `quality_score/` reports (scoring report v1, v4 falsification review, ring-repair report v2)

---

## Why keep this

A negative result with a clean design is evidence about the **method**, not just about the signal. Each of the four above closes off a class of ideas:

- block trades → microstructure data timestamps are not what they appear
- capital flows → the public aggregate series is already arbitraged
- ETF rotation → sector momentum at monthly frequency is inside the cost band
- quality scores → fundamental quality is priced, or at least not mispriced in the direction of a naive ranking

The falsification discipline is also the reason the other five projects state their failure boundaries explicitly.
