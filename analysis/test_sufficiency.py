#!/usr/bin/env python3
"""Test whether R_t is a sufficient conditioning set for the closed-loop decision.

H0: after conditioning on C and R_t, a forecast of R at t+Delta carries no
information about the outcome. Rejecting H0 means R_t is insufficient, which
is a reportable result in its own right.

Runs on Campaign B trial JSONs, which already contain risk_state_history at
~30 Hz and per-sample controller state.
"""
import os
import sys
import json
import glob
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Ensure package root is importable
pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

FEATS = ["r_min", "r_width", "r_ttc", "r_dens", "r_clear", "r_curve"]


def build_trial_windows(trial_json, deltas, horizon_s=4.0, max_delta=6.0, step=5):
    """One row per decision instant: R_t, R_{t+delta} for each delta, and outcome."""
    try:
        d = json.load(open(trial_json))
    except Exception:
        return []
    rs = d.get("risk_state_history", [])
    if len(rs) < 10:
        return []
    t = np.array([r.get("timestamp", 0.0) for r in rs], dtype=float)
    t -= t[0]
    F = {k: np.array([r.get(k, np.nan) for r in rs], dtype=float) for k in FEATS}

    is_collided = int(d.get("is_collided", False) or d.get("status") == "COLLISION")
    hit_times = [e.get("t", t[-1]) for e in d.get("collision_links", [])]
    if is_collided and not hit_times:
        hit_times = [t[-1]]

    rows = []
    for i in range(0, len(t), step):
        t_curr = t[i]
        row = {
            "trial": d.get("trial_id", os.path.basename(trial_json)),
            "t_curr": t_curr,
            "y": int(any(t_curr <= h <= t_curr + horizon_s for h in hit_times)) if hit_times else 0,
        }
        for k in FEATS:
            row[f"{k}_t"] = F[k][i]

        for delta in deltas:
            t_fut = t_curr + delta
            is_clamped = int(t_fut > t[-1])
            j = int(np.searchsorted(t, t_fut))
            j = min(j, len(t) - 1)
            for k in FEATS:
                row[f"{k}_fut_{delta}"] = F[k][j]
            row[f"clamp_{delta}"] = is_clamped

        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", required=True, help="glob for Campaign B trial JSONs")
    ap.add_argument("--deltas", default="1.0,2.0,4.0,6.0")
    args = ap.parse_args()

    deltas = [float(x) for x in args.deltas.split(",") if float(x) > 0]
    max_delta = max(deltas)

    all_rows = []
    for f in glob.glob(args.trials):
        all_rows += build_trial_windows(f, deltas, horizon_s=4.0, max_delta=max_delta, step=5)

    if not all_rows:
        print("No valid trial windows extracted.")
        return

    df = pd.DataFrame(all_rows).dropna()
    n_events = int(df["y"].sum())
    n_trials = df["trial"].nunique()
    print(f"Corrected window set: n={len(df)}  events={n_events} ({100.0*n_events/len(df):.2f}%)  trials={n_trials}\n")

    if n_events == 0:
        print("Warning: 0 collision events found in extracted trial windows.")
        return

    cols_t = [f"{k}_t" for k in FEATS]
    print(f"{'Delta':>6} {'Wald chi2':>10} {'p (clustered)':>15}  verdict")

    best_delta, max_wald = None, -1.0
    for delta in deltas:
        cols_f = [f"{k}_fut_{delta}" for k in FEATS] + [f"clamp_{delta}"]
        try:
            m_r = sm.Logit(df["y"], sm.add_constant(df[cols_t])).fit(
                disp=0, cov_type="cluster", cov_kwds={"groups": df["trial"]}
            )
            m_f = sm.Logit(df["y"], sm.add_constant(df[cols_t + cols_f])).fit(
                disp=0, cov_type="cluster", cov_kwds={"groups": df["trial"]}
            )

            r_matrix = np.zeros((len(cols_f), len(m_f.params)))
            for idx_c, col_name in enumerate(cols_f):
                param_idx = list(m_f.params.index).index(col_name)
                r_matrix[idx_c, param_idx] = 1.0

            wald_res = m_f.wald_test(r_matrix)
            wald_statistic = float(np.squeeze(wald_res.statistic))
            p_val = float(np.squeeze(wald_res.pvalue))

            if wald_statistic > max_wald:
                max_wald = wald_statistic
                best_delta = delta

            verdict = "R_t INSUFFICIENT" if p_val < 0.01 else "no evidence against R_t"
            print(f"{delta:6.1f} {wald_statistic:10.1f} {p_val:15.3e}   {verdict}")
        except Exception as e:
            print(f"{delta:6.1f}   fit failed: {e}")

    if best_delta is not None:
        print(f"\nDelta* (max Wald) = {best_delta:.1f} s")


if __name__ == "__main__":
    main()
