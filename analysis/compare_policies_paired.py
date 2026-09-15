#!/usr/bin/env python3
"""Paired comparison of policies under a shared doubly-robust score matrix.

Two things this adds to evaluate_ope.py:

1. PAIRED differences. Every policy is scored on the same rows with the same
   Gamma, so V(pi_a) - V(pi_b) is a within-row difference. Its standard error
   is far smaller than se(V_a) + se(V_b), and comparing the two marginal
   confidence intervals -- which is what a side-by-side table invites -- is the
   wrong test. Overlapping marginal intervals routinely hide a significant
   paired difference.

2. OUTCOME DECOMPOSITION. V_DR is a scalar utility, so it cannot support the
   Pareto claim. This reports each policy's implied collision probability,
   stall probability and carry fraction separately, which is what Figure A and
   the task-metric table actually need.

Standard errors are clustered by episode where an episode column exists. With
several probe trials per episode, row-level errors are too narrow, and the
tree-vs-static comparison is exactly the one a reviewer will scrutinise.

Usage:
    python3 analysis/compare_policies_paired.py \
        --data campaign_a.csv --models models/causal_tuner_models.pkl \
        --alpha 0.20 --reference "Static Best-Fixed (Tuck)"
"""
import os
import sys
import json
import pickle
import argparse
import numpy as np

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from online_causal_tuner.train_causal_models import (  # noqa: E402
    load_dataset_csv, extract_features, build_dr_scores,
    apply_policy_tree, POLICY_ACTION_SET, FOOTPRINT_KEY, STALL_EPS_M)

# Candidate column names for the episode identifier, in preference order.
EPISODE_KEYS = ("pose_pool_index", "episode_id", "episode", "trial_id", "run_id", "batch_id")


def find_episode_groups(kept_rows, n):
    """Cluster unit for the standard errors. Falls back to row-level, loudly."""
    for key in EPISODE_KEYS:
        if kept_rows and key in kept_rows[0]:
            g = np.array([str(r.get(key, "")) for r in kept_rows])
            if len(np.unique(g)) > 1 and len(np.unique(g)) < n:
                print(f"Clustering standard errors by '{key}' "
                      f"({len(np.unique(g))} clusters over {n} rows).")
                return g
    print("WARNING: no episode column found; standard errors are row-level and "
          "will be too narrow if probes cluster by episode.")
    return np.arange(n).astype(str)


def cluster_mean_se(v, groups):
    """Mean of v with a cluster-robust standard error."""
    uniq = np.unique(groups)
    G = len(uniq)
    mean = float(v.mean())
    sums = np.array([(v[groups == g] - mean).sum() for g in uniq])
    n = len(v)
    var = (G / max(G - 1, 1)) * (sums ** 2).sum() / (n ** 2)
    return mean, float(np.sqrt(max(var, 0.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--alpha", type=float, default=0.20)
    ap.add_argument("--reference", default="Static Best-Fixed (Tuck)",
                    help="Policy every other policy is compared against")
    ap.add_argument("--out", default="policy_comparison.json")
    args = ap.parse_args()

    rows = load_dataset_csv(args.data)
    X, y_safe, y_prog, groups_region, risk_cols, param_cols, feature_cols, kept_rows = \
        extract_features(rows)
    X_mat = np.asarray(X, dtype=float)
    n = X_mat.shape[0]
    print(f"Rows kept for evaluation: {n}")

    art = pickle.load(open(args.models, "rb"))
    w = art.get("objective_weights", {"risk_lambda": 10.0, "stall_mu": 2.0, "payload_omega": 0.5})
    RISK_LAMBDA, STALL_MU, PAYLOAD_OMEGA = w["risk_lambda"], w["stall_mu"], w["payload_omega"]

    fp_idx = feature_cols.index(FOOTPRINT_KEY)
    carry_obs = X_mat[:, fp_idx]
    stall_obs = (y_prog <= STALL_EPS_M).astype(float)
    U_obs = ((1.0 + PAYLOAD_OMEGA * carry_obs) * np.asarray(y_prog, float)
             - RISK_LAMBDA * np.asarray(y_safe, float)
             - STALL_MU * stall_obs)

    safety, stall, speed = art["safety_model"], art["stall_model"], art["speed_model"]
    j_max = float(art["progress_support"]["max"])

    def _components(Xa):
        p_c = safety.predict_proba(Xa)[:, 1]
        p_s = stall.predict_proba(Xa)[:, 1]
        e_sp = np.clip(speed.predict(Xa), 0.0, j_max)
        return p_c, p_s, (1.0 - p_s) * e_sp

    def _mu(Xa):
        carry = Xa[:, fp_idx]
        p_c, p_s, prog = _components(Xa)
        return ((1.0 + PAYLOAD_OMEGA * carry) * prog
                - RISK_LAMBDA * p_c
                - STALL_MU * p_s)

    action_cols = list(POLICY_ACTION_SET.keys())
    Gamma, cells, mu_mat = build_dr_scores(X_mat, kept_rows, U_obs, _mu,
                                           action_cols, POLICY_ACTION_SET, feature_cols)
    idx_of = {c: j for j, c in enumerate(cells)}
    col_idx = [feature_cols.index(c) for c in action_cols]
    fp_pos = action_cols.index("param__local_costmap__footprint")

    # Per-action outcome components, for the decomposition.
    A = len(cells)
    P_coll = np.zeros((n, A))
    P_stall = np.zeros((n, A))
    for j, cell in enumerate(cells):
        Xa = X_mat.copy()
        for k, ci in enumerate(col_idx):
            Xa[:, ci] = cell[k]
        P_coll[:, j], P_stall[:, j], _ = _components(Xa)

    policies = {}

    def _const(name, cell):
        if cell in idx_of:
            policies[name] = np.full(n, idx_of[cell], dtype=int)
        else:
            snapped = tuple(
                min(POLICY_ACTION_SET[c], key=lambda L: abs(cell[i] - L))
                for i, c in enumerate(action_cols)
            )
            if snapped in idx_of:
                policies[name] = np.full(n, idx_of[snapped], dtype=int)
            else:
                print(f"WARNING: {name} snaps to {cell}, which is not in the action set.")

    _const("Nav2 Default", (1.0, 0.60, 90.0))
    _const("Static Best-Fixed (Tuck)", (0.0, 0.30, 90.0))
    _const("Static Best-Fixed (Carry)", (1.0, 0.30, 90.0))
    _const("CURE (Tucked)", (0.0, 0.45, 90.0))
    _const("CURE (Carry)", (1.0, 0.45, 90.0))

    seen = {}
    for name, acts in policies.items():
        key = int(acts[0])
        seen.setdefault(key, []).append(name)
    for key, names in seen.items():
        if len(names) > 1:
            print(f"NOTE: these baselines are the SAME policy under the coarse "
                  f"action set {cells[key]}: {names}")

    if art.get("policy_tree") is not None and art.get("policy_leaf_action"):
        risk_idx = [feature_cols.index(c) for c in art["policy_risk_cols"]]
        policies["Ours (policy tree)"] = apply_policy_tree(
            art["policy_tree"], art["policy_leaf_action"], X_mat[:, risk_idx])

    ens = (art.get("safety_ensemble"), art.get("stall_ensemble"), art.get("speed_ensemble"))
    if all(ens):
        s_ens, st_ens, sp_ens = ens
        U_pess = np.zeros((n, A))
        for j, cell in enumerate(cells):
            Xa = X_mat.copy()
            for k, ci in enumerate(col_idx):
                Xa[:, ci] = cell[k]
            P = np.vstack([m.predict_proba(Xa)[:, 1] for m in s_ens])
            S = np.vstack([m.predict_proba(Xa)[:, 1] for m in st_ens])
            E = np.vstack([np.clip(m.predict(Xa), 0.0, j_max) for m in sp_ens])
            J = (1.0 - S) * E
            is_carry = 1.0 if float(cell[fp_pos]) == 1.0 else 0.0
            U_pess[:, j] = ((1.0 + PAYLOAD_OMEGA * is_carry) * np.quantile(J, args.alpha, axis=0)
                            - RISK_LAMBDA * np.quantile(P, 1.0 - args.alpha, axis=0)
                            - STALL_MU * np.quantile(S, 1.0 - args.alpha, axis=0))
        policies[f"Ours (pessimistic, a={args.alpha})"] = U_pess.argmax(axis=1)

    mu_only = np.zeros((n, A))
    for j, cell in enumerate(cells):
        Xa = X_mat.copy()
        for k, ci in enumerate(col_idx):
            Xa[:, ci] = cell[k]
        mu_only[:, j] = _mu(Xa)
    policies["mu-hat greedy (not a ceiling)"] = mu_only.argmax(axis=1)

    ep = find_episode_groups(kept_rows, n)
    ridx = np.arange(n)

    print(f"\n{'policy':<34} {'V_DR':>8} {'SE':>7}  {'P(coll)':>8} {'P(stall)':>9} {'carry':>6}")
    out = {}
    for name, acts in policies.items():
        acts = np.asarray(acts, dtype=int)
        v = Gamma[ridx, acts]
        mean, se = cluster_mean_se(v, ep)
        pc = float(P_coll[ridx, acts].mean())
        ps = float(P_stall[ridx, acts].mean())
        carry = float(np.mean([cells[a][fp_pos] for a in acts]))
        out[name] = {"v_dr": mean, "se": se, "ci": [mean - 1.96 * se, mean + 1.96 * se],
                     "p_collision": pc, "p_stall": ps, "carry_fraction": carry}
        print(f"{name:<34} {mean:8.4f} {se:7.4f}  {pc:8.4f} {ps:9.4f} {carry:6.2f}")

    if args.reference not in policies:
        print(f"\nReference '{args.reference}' not available; skipping paired tests.")
    else:
        ref = np.asarray(policies[args.reference], dtype=int)
        v_ref = Gamma[ridx, ref]
        print(f"\nPaired difference vs {args.reference} "
              f"(same rows, same Gamma, clustered SE):")
        print(f"{'policy':<34} {'diff':>8} {'SE':>7} {'z':>7} {'p':>10}")
        for name, acts in policies.items():
            if name == args.reference:
                continue
            d = Gamma[ridx, np.asarray(acts, dtype=int)] - v_ref
            md, sd = cluster_mean_se(d, ep)
            z = md / sd if sd > 0 else 0.0
            from scipy import stats
            p = 2 * stats.norm.sf(abs(z))
            out[name]["paired_vs_ref"] = {"diff": md, "se": sd, "z": z, "p": p}
            print(f"{name:<34} {md:8.4f} {sd:7.4f} {z:7.2f} {p:10.3e}")

    out["_meta"] = {"alpha": args.alpha, "n": int(n), "n_actions": A,
                    "reference": args.reference,
                    "n_clusters": int(len(np.unique(ep)))}
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
