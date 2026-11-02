# 03 · Counterfactual Estimation of Trump's Tariffs on the S&P 500

**Working paper.** What would the S&P 500 have done without the tariff announcements — and how much does the answer depend on how you ask?

---

## The finding

The counterfactual gap is **not a number, it is a range**:

| Method | Donor pool | Short window | Long window |
|---|---|---|---|
| Synthetic control (SCM, hand-written in cvxpy) | Sector ETFs | **+2.26%** | **+7.72%** |
| CausalImpact (BSTS state space) | Overseas indices | **−6.73%** (p = 0.80) | **+7.00%** (p = 0.082) |
| Event study (beta-controlled CAR) | — | XLK flips from worst absolute loss to **+5.19% abnormal** | — |

The two counterfactual methods disagree **in sign** on the short window. That disagreement is the contribution: financial-event SCM suffers from *donor contamination* — every candidate donor is itself exposed to the same global shock — and the paper maps the sensitivity boundary rather than claiming a point estimate.

**Mechanism.** Causal mediation puts **56% of the 2025 Liberation-Day decline through the VIX channel** (ACME −6.85%, bootstrap-significant). The market did not reprice cash flows so much as reprice risk.

**Placebo / robustness.** Short-window SCM gap fails the permutation placebo (p = 54.55%), while leave-one-out is stable; a 2018-vs-2025 comparison separates a *gradual* shock (S&P +4.3% short-window, market under-pricing escalation) from a *shock* (−12.1%, V-shaped).

---

## Files

| File | What it is |
|---|---|
| `paper.pdf` / `paper.tex` | The paper (LaTeX source + compiled) |
| `论文_正式版.md` | Chinese working draft |
| `paper_lib.py` | SCM (cvxpy), CausalImpact wrappers, event-study utilities |
| `notebooks/03_scm_sector_donor.ipynb` | SCM construction and gaps |
| `notebooks/04_causal_impact.ipynb` | BSTS counterfactual |
| `notebooks/05_event_study_car.ipynb` | Beta-controlled CAR |
| `notebooks/06_placebo_loo.ipynb` | Permutation placebo + leave-one-out |
| `generate_15_donor_confounding.py` | The donor-contamination experiment |
| `positive_control.py` | Positive control (smoking/California replication) |
| `NOTES.md` | Working notes and data provenance |

---

## Methods actually implemented

- **Synthetic control** — hand-written in `cvxpy` (no `pysyncon` dependency), with predictor weights and a concave penalty on donor concentration
- **CausalImpact** — statsmodels BSTS state-space, with the standard posterior tail-probability readout
- **Event study** — market-model CAR with beta estimated pre-event
- **GARCH** — volatility-regime conditioning
- **Causal mediation** — ACME / ADE decomposition with bootstrap CI

---

## Limitations (stated in the paper)

1. Donor contamination is structural, not fixable by a better donor — this is the endemic problem of SCM applied to global financial events.
2. Two rounds of tariffs (2018-19, 2025) differ in anticipation and in the policy path that followed; pooling them would be wrong.
3. The 2025 event is recent enough that long-window estimates rest on a short post-period.

---

## Status

Target: SSRN + arXiv (q-fin). Feedback on the identification strategy is welcome.
