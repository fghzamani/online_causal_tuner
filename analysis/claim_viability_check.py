#!/usr/bin/env python3
"""Is the 'adaptive beats fixed' claim viable? Four checks in one run.

This is a DECISION INSTRUMENT, not a results script. Run it once and read the
verdict at the bottom before deciding how to spend the remaining days.

  CHECK 1  Best constant policy as the reference, not a chosen one.
           Beating "Static Best-Fixed" is not the claim; beating the best
           fixed configuration in the action set is.

  CHECK 2  Clustering unit. run_id gives 9 clusters, too few for the
           cluster-robust sandwich. Reports every available unit so you can
           see how much the choice moves the answer.

  CHECK 3  Cross-fitted value. The tree's leaves are fitted on the same Gamma
           rows they are scored on; constants are not. Refit per fold.

  CHECK 4  Stratified effect. Most probes sit in open corridors where no
           configuration matters, which dilutes a real effect in constricted
           contexts toward zero. Splits by r_min tertile.

Usage:
    python3 analysis/claim_viability_check.py \
        --data campaign_a.csv --models models/causal_tuner_models.pkl --folds 5
"""
import os
import sys
import json
import pickle
import argparse
import itertools
import numpy as np
from scipy import stats

pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from online_causal_tuner.train_causal_models import (  # noqa: E402
    load_dataset_csv, extract_features, build_dr_scores, fit_policy_tree,
    apply_policy_tree, POLICY_ACTION_SET, FOOTPRINT_KEY, STALL_EPS_M)

CLUSTER_CANDIDATES = ("pose_pool_index", "episode_id", "episode", "trial_id", "pose_pair_id",
                      "run_id", "batch_id")


def cluster_stats(v, groups):
    """Mean, cluster-robust SE, and the cluster count G.

    At small G the sandwich is downward biased and the reference distribution
    is t(G-1), not normal. G is returned so the caller can say so.
    """
    uniq = np.unique(groups)
    G, n = len(uniq), len(v)
    mean = float(v.mean())
    sums = np.array([(v[groups == g] - mean).sum() for g in uniq])
    var = (G / max(G - 1, 1)) * (sums ** 2).sum() / (n ** 2)
    return mean, float(np.sqrt(max(var, 0.0))), G


def report(label, d, groups, indent=""):
    m, se, G = cluster_stats(d, groups)
    t = m / se if se > 0 else 0.0
    p_t = 2 * stats.t.sf(abs(t), max(G - 1, 1))
    star = "*" if p_t < 0.05 else " "
    print(f"{indent}{label:<40} {m:+8.4f}  SE {se:.4f}  t {t:6.2f}  "
          f"G={G:5d}  p(t) {p_t:8.4f} {star}")
    return m, se, G, p_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--out", default="claim_viability.json")
    args = ap.parse_args()

    rows = load_dataset_csv(args.data)
    X, y_safe, y_prog, _, risk_cols, param_cols, feature_cols, kept = extract_features(rows)
    X_mat = np.asarray(X, float)
    n = X_mat.shape[0]

    art = pickle.load(open(args.models, "rb"))
    w = art.get("objective_weights")
    if w is None:
        raise SystemExit("Artifact lacks objective_weights; retrain after the C1 fix.")
    LAM, MU, OMG = w["risk_lambda"], w["stall_mu"], w["payload_omega"]

    safety, stall, speed = art["safety_model"], art["stall_model"], art["speed_model"]
    j_max = float(art["progress_support"]["max"])
    fp_idx = feature_cols.index(FOOTPRINT_KEY)

    carry_obs = X_mat[:, fp_idx]
    stall_obs = (np.asarray(y_prog, float) <= STALL_EPS_M).astype(float)
    U_obs = ((1.0 + OMG * carry_obs) * np.asarray(y_prog, float)
             - LAM * np.asarray(y_safe, float) - MU * stall_obs)

    def _mu(Xa):
        c = Xa[:, fp_idx]
        p_st = stall.predict_proba(Xa)[:, 1]
        e_sp = np.clip(speed.predict(Xa), 0.0, j_max)
        p_sf = safety.predict_proba(Xa)[:, 1]
        return (1.0 + OMG * c) * (1.0 - p_st) * e_sp - LAM * p_sf - MU * p_st

    action_cols = list(POLICY_ACTION_SET.keys())
    Gamma, cells, mu_mat = build_dr_scores(X_mat, kept, U_obs, _mu,
                                           action_cols, POLICY_ACTION_SET, feature_cols)
    risk_idx = [feature_cols.index(c) for c in art["policy_risk_cols"]]
    R_mat = X_mat[:, risk_idx]
    ridx = np.arange(n)
    print(f"n = {n} rows, {len(cells)} action cells, "
          f"objective lambda={LAM} mu={MU} omega={OMG}\n")

    # ---------------- CHECK 2: clustering unit ----------------
    print("=" * 92)
    print("CHECK 2 - clustering unit")
    print("=" * 92)
    units = {}
    for key in CLUSTER_CANDIDATES:
        if kept and key in kept[0]:
            g = np.array([str(r.get(key, "")) for r in kept])
            if 1 < len(np.unique(g)) < n:
                units[key] = g
                print(f"  {key:<16} {len(np.unique(g)):6d} clusters")
    if not units:
        print("  none found; using row-level (understates correlation)")
        units["row"] = np.arange(n).astype(str)
    primary = min(units, key=lambda k: -len(np.unique(units[k])))
    ep = units[primary]
    print(f"\n  Primary unit for the tests below: '{primary}' "
          f"({len(np.unique(ep))} clusters)")
    if len(np.unique(ep)) < 30:
        print("  WARNING: fewer than 30 clusters. The sandwich is downward biased "
              "and t(G-1) is only a partial correction. Use a wild cluster "
              "bootstrap before quoting any p-value.")

    # ---------------- CHECK 1: best constant policy ----------------
    print("\n" + "=" * 92)
    print("CHECK 1 - best CONSTANT policy (the honest reference)")
    print("=" * 92)
    const_vals = []
    for j, cell in enumerate(cells):
        m, se, G = cluster_stats(Gamma[:, j], ep)
        const_vals.append((m, se, j, cell))
    const_vals.sort(reverse=True)
    print(f"  {'rank':<5} {'action cell':<28} {'V_DR':>9} {'SE':>8}")
    for r, (m, se, j, cell) in enumerate(const_vals[:5], 1):
        print(f"  {r:<5} {str(cell):<28} {m:9.4f} {se:8.4f}")
    best_m, best_se, best_j, best_cell = const_vals[0]
    print(f"\n  BEST CONSTANT = {best_cell}  V_DR = {best_m:.4f}")
    ref_vec = Gamma[:, best_j]

    # ---------------- CHECK 3: cross-fitted vs in-sample ----------------
    print("\n" + "=" * 92)
    print("CHECK 3 - learned policy vs BEST CONSTANT, in-sample and cross-fitted")
    print("=" * 92)
    in_acts = apply_policy_tree(art["policy_tree"], art["policy_leaf_action"], R_mat)
    v_in = Gamma[ridx, in_acts]

    uniq_ep = np.unique(ep)
    rng = np.random.RandomState(42)
    rng.shuffle(uniq_ep)
    fold_of = {e: i % args.folds for i, e in enumerate(uniq_ep)}
    fold = np.array([fold_of[e] for e in ep])

    v_cf = np.zeros(n)
    carry_cf = np.zeros(n)
    for k in range(args.folds):
        tr, te = fold != k, fold == k
        if te.sum() == 0 or tr.sum() < 100:
            continue
        tk, lk, _ = fit_policy_tree(R_mat[tr], Gamma[tr], mu_mat[tr], max_depth=args.max_depth)
        ak = apply_policy_tree(tk, lk, R_mat[te])
        v_cf[te] = Gamma[np.where(te)[0], ak]
        carry_cf[te] = [cells[a][action_cols.index(FOOTPRINT_KEY)] for a in ak]

    print()
    m_in, *_ = report("tree - best constant  [IN-SAMPLE]", v_in - ref_vec, ep, "  ")
    m_cf, se_cf, G_cf, p_cf = report("tree - best constant  [CROSS-FITTED]",
                                     v_cf - ref_vec, ep, "  ")
    print(f"\n  Optimism (in-sample minus cross-fitted): {m_in - m_cf:+.4f}")
    print(f"  Cross-fitted carry fraction: {carry_cf.mean():.3f}")

    mu_all = np.zeros((n, len(cells)))
    col_idx = [feature_cols.index(c) for c in action_cols]
    for j, cell in enumerate(cells):
        Xa = X_mat.copy()
        for k2, ci in enumerate(col_idx):
            Xa[:, ci] = cell[k2]
        mu_all[:, j] = _mu(Xa)
    greedy = mu_all.argmax(axis=1)
    print()
    report("mu-hat greedy - best constant", Gamma[ridx, greedy] - ref_vec, ep, "  ")
    report("tree - mu-hat greedy  [CROSS-FITTED]", v_cf - Gamma[ridx, greedy], ep, "  ")

    # ---------------- CHECK 4: stratified by context ----------------
    print("\n" + "=" * 92)
    print("CHECK 4 - effect by context stratum (r_min tertile)")
    print("=" * 92)
    print("  Most probes sit in open corridors where no configuration matters.")
    print("  Pooling those with constricted contexts dilutes a real effect.\n")
    rmin_col = feature_cols.index("risk__r_min")
    rmin = X_mat[:, rmin_col]
    q1, q2 = np.quantile(rmin, [1 / 3, 2 / 3])
    strata = [("constricted (r_min low)", rmin <= q1),
              ("moderate", (rmin > q1) & (rmin <= q2)),
              ("open (r_min high)", rmin > q2)]
    strat_out = {}
    for name, m in strata:
        print(f"  {name}  (n={int(m.sum())}, r_min <= {q1:.2f} / <= {q2:.2f})")
        bj = int(np.argmax([Gamma[m, j].mean() for j in range(len(cells))]))
        d = v_cf[m] - Gamma[np.where(m)[0], bj]
        mm, ss, GG, pp = report("tree - best constant in stratum",
                                d, ep[m], "      ")
        strat_out[name] = {"n": int(m.sum()), "diff": mm, "se": ss,
                           "G": GG, "p_t": pp, "best_cell": str(cells[bj])}
        print()

    # ---------------- verdict ----------------
    print("=" * 92)
    print("VERDICT")
    print("=" * 92)
    alive_pooled = p_cf < 0.05 and m_cf > 0
    alive_strat = any(v["p_t"] < 0.05 and v["diff"] > 0 for v in strat_out.values())

    if alive_pooled:
        print("  POOLED claim is alive: the cross-fitted advantage over the best")
        print("  constant policy excludes zero. Report it with the clustering unit")
        print("  and the cross-fitting stated explicitly.")
    elif alive_strat:
        s = max((v for v in strat_out.values() if v["p_t"] < 0.05),
                key=lambda v: v["diff"])
        print("  POOLED claim does NOT clear, but a STRATIFIED claim does.")
        print(f"  Strongest stratum: diff {s['diff']:+.4f}, p(t) {s['p_t']:.4f}.")
        print("  This is the stronger paper anyway: 'adaptation helps where context")
        print("  varies, and not elsewhere' IS the context-dependence claim, stated")
        print("  as an effect rather than a horse race. Lead with the stratified")
        print("  table and report the pooled number honestly alongside it.")
    else:
        print("  Neither pooled nor stratified clears at this sample and action set.")
        print("  Do NOT spend more days here. Switch to the mission-constraint")
        print("  framing (paper_reframe_decision.md) and report this comparison as")
        print("  a stated null. A volunteered null costs a paragraph; a discovered")
        print("  one costs the paper.")

    json.dump({"n": int(n), "primary_cluster": primary,
               "n_clusters": int(len(np.unique(ep))),
               "best_constant": str(best_cell), "v_best_constant": best_m,
               "diff_in_sample": m_in, "diff_crossfit": m_cf,
               "se_crossfit": se_cf, "p_crossfit_t": p_cf,
               "optimism": m_in - m_cf,
               "carry_fraction_crossfit": float(carry_cf.mean()),
               "strata": strat_out}, open(args.out, "w"), indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
