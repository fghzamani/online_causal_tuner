#!/usr/bin/env python3
"""Doubly-robust off-policy evaluation of every strategy on Campaign A.

Campaign A used randomized configuration assignment, so the propensity e(c) is
known by design. This makes the DR estimator unbiased regardless of whether the
outcome model is correct -- an unusually strong position for policy evaluation.

Gives a quantitative comparison of all strategies, with confidence intervals,
over ~13,110 trials, without running a single simulation.
"""
import os
import sys
import argparse
import json
import pickle
import numpy as np

# Ensure package root is importable
pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from online_causal_tuner.train_causal_models import (
    load_dataset_csv, extract_features, POLICY_ACTION_SET)

RISK_LAMBDA = 10.0


def dr_value(Gamma, policy_actions, cluster_ids=None, n_boot=500, seed=42):
    """V_DR(pi) and cluster-robust standard error by episode/cluster."""
    v = Gamma[np.arange(len(policy_actions)), policy_actions]
    v_mean = float(v.mean())
    if cluster_ids is None:
        se = float(v.std(ddof=1) / np.sqrt(len(v)))
    else:
        rng = np.random.default_rng(seed)
        unique_clusters, cluster_indices = np.unique(cluster_ids, return_inverse=True)
        n_clusters = len(unique_clusters)
        if n_clusters <= 1:
            se = 0.0
        else:
            cluster_sums = np.bincount(cluster_indices, weights=v)
            cluster_counts = np.bincount(cluster_indices)
            boot_clust_samples = rng.choice(n_clusters, size=(n_boot, n_clusters), replace=True)
            boot_sums = cluster_sums[boot_clust_samples].sum(axis=1)
            boot_counts = cluster_counts[boot_clust_samples].sum(axis=1)
            boot_means = boot_sums / np.maximum(boot_counts, 1)
            se = float(np.std(boot_means, ddof=1))
    return v_mean, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Campaign A rct_results.csv")
    ap.add_argument("--models", required=True, help="causal_tuner_models.pkl")
    ap.add_argument("--alphas", default="0.05,0.10,0.20,0.35,0.50", help="Comma-separated alphas for sweep")
    ap.add_argument("--out", default="ope_results.json")
    args = ap.parse_args()

    alpha_list = [float(x) for x in args.alphas.split(",") if float(x) >= 0]
    rows = load_dataset_csv(args.data)
    X, y_safe, y_prog, groups, risk_cols, param_cols, feature_cols, kept_rows = extract_features(rows)
    X_mat = np.array(X, dtype=float)
    U_obs = np.array(y_prog, float) - RISK_LAMBDA * np.array(y_safe, float)
    cluster_ids = np.array([r.get("trial_id", r.get("episode_id", r.get("probe_id", i))) for i, r in enumerate(kept_rows)])

    art = pickle.load(open(args.models, "rb"))
    safety, stall_model, speed_model = art["safety_model"], art["stall_model"], art["speed_model"]
    progress_support = art["progress_support"]

    def _mu(Xa):
        p_st = stall_model.predict_proba(Xa)[:, 1]
        e_sp = np.clip(speed_model.predict(Xa), 0.0, progress_support["max"])
        expected_progress = (1.0 - p_st) * e_sp
        p_sf = safety.predict_proba(Xa)[:, 1]
        return expected_progress - RISK_LAMBDA * p_sf

    from online_causal_tuner.train_causal_models import build_dr_scores, apply_policy_tree
    action_cols = list(POLICY_ACTION_SET.keys())
    Gamma, cells, mu_mat = build_dr_scores(X_mat, kept_rows, U_obs, _mu,
                                           action_cols, POLICY_ACTION_SET, feature_cols)
    idx_of = {c: j for j, c in enumerate(cells)}
    n = len(kept_rows)

    def _const(cell):
        return np.full(n, idx_of[cell], dtype=int)

    # Baselines expressed as policies.
    policies = {
        "Nav2 Default":            _const((1.0, 0.60, 1.0)),
        "Static Best-Fixed (Tuck)": _const((0.0, 0.30, 1.0)),
        "Static Best-Fixed (Carry)": _const((1.0, 0.30, 1.0)),
        "CURE (Tucked)":           _const((0.0, 0.30, 1.0)),
        "CURE (Carry)":            _const((1.0, 0.30, 1.0)),
        "Oracle (R-measurable, mu-hat)": mu_mat.argmax(axis=1),
    }

    # Reconstruct pessimistic policy for given alphas
    if "safety_ensemble" in art and len(art["safety_ensemble"]) > 0:
        col_idx = [feature_cols.index(c) for c in action_cols]
        max_prog_support = float(progress_support["max"])
        
        # Precompute stacked ensemble predictions across candidates
        P_all = []
        P_stall_all = []
        E_speed_all = []
        for j, cell in enumerate(cells):
            Xa = X_mat.copy()
            for k, ci in enumerate(col_idx):
                Xa[:, ci] = cell[k]
            P_all.append(np.vstack([m.predict_proba(Xa)[:, 1] for m in art["safety_ensemble"]]))
            P_stall_all.append(np.vstack([m.predict_proba(Xa)[:, 1] for m in art["stall_ensemble"]]))
            E_speed_all.append(np.vstack([np.clip(m.predict(Xa), 0.0, max_prog_support) for m in art["speed_ensemble"]]))

        for a_val in alpha_list:
            u_pess = np.zeros((n, len(cells)))
            for j, cell in enumerate(cells):
                P = P_all[j]
                P_stall = P_stall_all[j]
                E_speed = E_speed_all[j]
                J = (1.0 - P_stall) * E_speed

                p_ucb = np.quantile(P, 1.0 - a_val, axis=0)
                p_stall_ucb = np.quantile(P_stall, 1.0 - a_val, axis=0)
                j_lcb = np.quantile(J, a_val, axis=0)

                is_carry = 1.0 if float(cell[action_cols.index("param__local_costmap__footprint")]) == 1.0 else 0.0
                u_pess[:, j] = (1.0 + 0.5 * is_carry) * j_lcb - RISK_LAMBDA * p_ucb - 1.0 * p_stall_ucb

            policies[f"Ours (pessimistic argmax, alpha={a_val:.2f})"] = u_pess.argmax(axis=1)

    if "policy_tree" in art:
        risk_idx = [feature_cols.index(c) for c in art["policy_risk_cols"]]
        if "policy_leaf_action" in art:
            policies["Ours (policy tree)"] = apply_policy_tree(
                art["policy_tree"], art["policy_leaf_action"], X_mat[:, risk_idx]
            )
        else:
            policies["Ours (policy tree)"] = art["policy_tree"].predict(X_mat[:, risk_idx])

    out = {"_meta": {"alphas": alpha_list, "n": n, "n_actions": len(cells), "n_clusters": len(np.unique(cluster_ids))}}
    print(f"{'policy':<38} {'V_DR':>9} {'SE (cluster)':>12}   95% CI")
    for name, acts in policies.items():
        v, se = dr_value(Gamma, np.asarray(acts, dtype=int), cluster_ids=cluster_ids)
        out[name] = {"v_dr": v, "se": se, "ci": [v - 1.96 * se, v + 1.96 * se]}
        print(f"{name:<38} {v:9.4f} {se:12.4f}   [{v-1.96*se:.4f}, {v+1.96*se:.4f}]")

    json.dump(out, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
