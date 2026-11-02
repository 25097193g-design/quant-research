#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Positive control for the diagnostic protocol (paper Section 6).

Runs the same four diagnostics used on the 2018/2025 tariff events on the
classic clean-control case of Abadie, Diamond & Hainmueller (2010):
California Proposition 99 (treated from 1989; 38 donor states; cigsale
panel 1970-2000).

    (1) donor-treatment homogeneity: pre-period first-difference correlation
        between treated and each donor (tariff case: max 0.91, XLK);
    (2) permutation placebos: do they have power? (tariff case: p=100%);
    (3) weight concentration: top-1/top-3 shares (tariff case: overseas
        pool top-1 = 0.76 Canada);
    (4) binding-tightness sweep: ridge-penalized counterfactual over
        alpha in {0.01, 0.1, 1, 10, 100}, event-date anchored
        (tariff case: gap sweeps monotonically from ~0 to the raw return).

Data: smoking_california.csv (long panel; the numeric `state` codes in this
mirror are NOT FIPS -- California is code 3, identified by the built-in
`california` boolean column), cached alongside this script.

Usage:  python positive_control.py            (run with the rqsdk env python)
"""
import os
import numpy as np
import pandas as pd
from pysyncon import Dataprep, Synth

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "smoking_california.csv")

PRE_END, POST_END = 1988, 2000
_raw = pd.read_csv(CSV)
TREATED = int(_raw.loc[_raw["california"] == True, "state"].unique()[0])  # = 3
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def load_wide():
    df = pd.read_csv(CSV)
    wide = df.pivot(index="year", columns="state", values="cigsale").sort_index()
    donors = [c for c in wide.columns if c != TREATED]
    return wide, donors


def homogeneity(wide, donors):
    pre = wide.loc[:PRE_END]
    d_treated = pre[TREATED].diff().dropna()
    corrs = {s: d_treated.corr(pre[s].diff().dropna()) for s in donors}
    vals = pd.Series(corrs).dropna()
    return vals.max(), vals.idxmax(), float(vals.mean())


def fit_scm(wide, treated_id):
    """Convex SCM with the ADH predictor set; returns (weights, gaps series).
    Gap series computed directly as treated - donors @ w on the raw panel."""
    df = pd.read_csv(CSV)
    donors = [s for s in df["state"].unique() if s != treated_id]
    dp = Dataprep(
        foo=df,
        predictors=["age15to24", "retprice"],
        predictors_op="mean",
        dependent="cigsale",
        unit_variable="state",
        time_variable="year",
        treatment_identifier=treated_id,
        controls_identifier=donors,
        time_predictors_prior=range(1970, PRE_END + 1),
        time_optimize_ssr=range(1970, PRE_END + 1),
        special_predictors=[
            ("cigsale", [1975], "mean"),
            ("cigsale", [1980], "mean"),
            ("cigsale", [1988], "mean"),
        ],
    )
    synth = Synth()
    synth.fit(dataprep=dp)
    w = synth.weights().astype(float)
    gap = wide[treated_id] - wide[w.index].values @ w.values
    return w, gap


def placebos(wide, cache_path=None):
    """Permutation placebo: each state in turn treated from 1989.
    Returns (terminal gap Series, full gap-path DataFrame). Weights come
    from the same convex SCM as for the treated unit; gap paths are cached
    to CSV so re-runs are instant."""
    if cache_path is not None and os.path.exists(cache_path):
        paths = pd.read_csv(cache_path, index_col=0)
        paths.columns = paths.columns.astype(int)
        return paths.loc[POST_END].astype(float), paths
    rng = range(1970, PRE_END + 1)
    df = pd.read_csv(CSV)
    paths = {}
    for s in df["state"].unique():
        donors = [x for x in df["state"].unique() if x != s]
        dp = Dataprep(
            foo=df,
            predictors=["age15to24", "retprice"],
            predictors_op="mean",
            dependent="cigsale",
            unit_variable="state",
            time_variable="year",
            treatment_identifier=s,
            controls_identifier=donors,
            time_predictors_prior=rng,
            time_optimize_ssr=rng,
            special_predictors=[
                ("cigsale", [1975], "mean"),
                ("cigsale", [1980], "mean"),
                ("cigsale", [1988], "mean"),
            ],
        )
        syn = Synth()
        syn.fit(dataprep=dp)
        w = syn.weights().astype(float)
        gap = wide[s] - wide[w.index].values @ w.values
        paths[s] = gap
    paths = pd.DataFrame(paths)
    if cache_path is not None:
        paths.to_csv(cache_path)
    return paths.loc[POST_END].astype(float), paths


def ascm_sweep(wide, donors):
    """Ridge counterfactual over alpha, anchored at the event year (gap=0 in
    1988). Predictors standardized with pre-period moments, as in the paper."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    pre = wide.loc[:PRE_END]
    post = wide.loc[PRE_END:POST_END]
    sc = StandardScaler().fit(pre)
    Zpre = sc.transform(pre)
    Zpost = sc.transform(post)
    y_pre_std = (pre[TREATED].values - sc.mean_[list(wide.columns).index(TREATED)]) / np.sqrt(
        sc.var_[list(wide.columns).index(TREATED)]
    )
    res = {}
    for a in ALPHAS:
        r = Ridge(alpha=a).fit(Zpre[:, [wide.columns.get_loc(s) for s in donors]],
                               y_pre_std)
        pred_std = r.predict(
            Zpost[:, [wide.columns.get_loc(s) for s in donors]])
        # de-standardize
        mu, sd = sc.mean_[wide.columns.get_loc(TREATED)], np.sqrt(
            sc.var_[wide.columns.get_loc(TREATED)])
        pred = pred_std * sd + mu
        # anchor at event year: shift so predicted == treated at PRE_END
        pred = pred + (wide.loc[PRE_END, TREATED] - pred[0])
        gap = wide.loc[PRE_END:POST_END, TREATED].values - pred
        res[a] = float(gap[-1])
    return res


def main():
    wide, donors = load_wide()

    print("=" * 64)
    print("(1) DONOR HOMOGENEITY  (pre-period first-difference corr)")
    mx, argmx, mean_ = homogeneity(wide, donors)
    print(f"    max corr with treated = {mx:.2f} (state {argmx}); mean = {mean_:.2f}")
    print("    [tariff case: max 0.91 with XLK; sector donors are slices]")

    print("(2) WEIGHT CONCENTRATION")
    w, gaps = fit_scm(wide, TREATED)
    w = w / w.sum()
    top = w.sort_values(ascending=False)
    print(f"    nonzero donors: {(w > 1e-4).sum()} / {len(w)}")
    print(f"    top-1 = {top.iloc[0]:.2f} (state {top.index[0]}); "
          f"top-3 = {top.iloc[:3].sum():.2f}")
    print("    [tariff overseas case: top-1 = 0.76 Canada]")
    sc_gap_2000 = float(gaps.loc[POST_END])
    print(f"    convex SCM gap at {POST_END}: {sc_gap_2000:+.1f} packs "
          "(published ADH estimate: about -18 to -25)")

    print("(3) PLACEBO POWER")
    cache = os.path.join(HERE, "positive_control_placebo_paths.csv")
    pg, paths = placebos(wide, cache_path=cache)
    treated_gap = pg[TREATED]
    p_perm = float((pg.abs() >= abs(treated_gap)).mean())
    rank = int((pg.abs() > abs(treated_gap)).sum()) + 1
    print("    [terminal-year |gap| statistic]")
    print(f"    treated (California) gap {treated_gap:+.1f}; rank {rank} of "
          f"{len(pg)} by |gap| (1 = most extreme)")
    print(f"    permutation p = {p_perm:.3f}   [tariff case: p = 1.000]")
    print(f"    placebo |gap| range: [{pg.drop(TREATED).abs().min():.1f}, "
          f"{pg.drop(TREATED).abs().max():.1f}]")
    # ADH-style post/pre MSPE-ratio statistic (Abadie et al. 2010, fig. 5)
    pre = paths.loc[:PRE_END]
    post = paths.loc[PRE_END + 1:]
    ratio = np.sqrt((post ** 2).mean()) / np.sqrt((pre ** 2).mean())
    p_ratio = float((ratio >= ratio[TREATED]).mean())
    rank_ratio = int((ratio > ratio[TREATED]).sum()) + 1
    print("    [post/pre RMSPE-ratio statistic, ADH-style]")
    print(f"    California ratio {ratio[TREATED]:.1f}; rank {rank_ratio} of "
          f"{len(ratio)} (1 = most extreme)")
    print(f"    permutation p = {p_ratio:.3f}")

    print("(4) BINDING-TIGHTNESS SWEEP  (gap at 2000, anchored 1988)")
    sweep = ascm_sweep(wide, donors)
    raw = wide.loc[POST_END, TREATED] - wide.loc[PRE_END, TREATED]
    print(f"    convex SCM: {sc_gap_2000:+.1f}")
    for a, g in sweep.items():
        print(f"    ridge alpha={a:<6}: {g:+.1f}")
    print(f"    raw change {PRE_END}->{POST_END}: {raw:+.1f}")
    print("    [tariff case: sweeps monotonically from ~0 to the raw return]")

    print("=" * 64)
    rows = {
        "max_pre_corr": mx, "perm_p": p_perm, "perm_rank": rank,
        "perm_p_mspe": p_ratio, "perm_rank_mspe": rank_ratio,
        "top1_weight": float(top.iloc[0]), "nonzero_donors": int((w > 1e-4).sum()),
        "scm_gap_2000": sc_gap_2000, "raw_change": float(raw),
        **{f"ridge_a{a}": g for a, g in sweep.items()},
    }
    pd.Series(rows).to_csv(os.path.join(HERE, "outputs_positive_control.csv")
                           if os.path.isdir(os.path.join(HERE, "outputs"))
                           else os.path.join(HERE, "positive_control_results.csv"),
                           header=False)
    print("saved: positive_control_results.csv")


if __name__ == "__main__":
    main()
