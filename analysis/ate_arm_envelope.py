#!/usr/bin/env python3
"""ATE of arm envelope on feasibility, safety and progress.

Campaign A randomized C_arm ~50/50 independently of R. Under randomization the
UNADJUSTED difference in means is an unbiased estimate of the average treatment
effect. No outcome model, no propensity model, no cross-fitting, no omega, no
policy learning. This is the cleanest result in the project and it is a
two-hour computation.

Usage:
    python3 analysis/ate_arm_envelope.py --data campaign_a.csv \
        --cluster episode_id --out results/ate_arm.json
"""
import csv, json, argparse
import numpy as np
from scipy import stats


def to_float(v, default=np.nan):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return default


def cluster_se(v, groups):
    """Cluster-robust SE of a mean. Returns (se, G)."""
    uniq = np.unique(groups)
    G, n = len(uniq), len(v)
    m = v.mean()
    sums = np.array([(v[groups == g] - m).sum() for g in uniq])
    var = (G / max(G - 1, 1)) * (sums ** 2).sum() / (n ** 2)
    return float(np.sqrt(max(var, 0.0))), G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cluster", default=None,
                    help="Column to cluster on (episode_id, run_id, ...). "
                         "Omit for unclustered.")
    ap.add_argument("--out", default="ate_arm.json")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.data, encoding="utf-8", errors="ignore"))
            if str(r.get("treatment_valid", "")).strip() in ("1", "true", "True")]
    print(f"Valid rows: {len(rows)}")

    carry = np.array([str(r.get("param__local_costmap__footprint", "")).strip()
                      in ("carry", "1.0", "1") for r in rows])
    prog = np.array([to_float(r.get("probe_progress_m")) for r in rows])
    coll = np.array([to_float(r.get("y_h")) for r in rows])
    ok = ~(np.isnan(prog) | np.isnan(coll))
    rows = [r for r, k in zip(rows, ok) if k]
    carry, prog, coll = carry[ok], prog[ok], coll[ok]
    stall = (prog <= 0.001).astype(float)
    coll = (coll > 0.5).astype(float)

    print(f"Usable rows: {len(rows)}  carry fraction: {carry.mean():.4f}")
    if abs(carry.mean() - 0.5) > 0.05:
        print("  WARNING: assignment is not ~50/50. Check the randomizer before "
              "treating the raw difference as the ATE.")

    if args.cluster and args.cluster in rows[0]:
        g = np.array([str(r.get(args.cluster, "")) for r in rows])
        print(f"Clustering on '{args.cluster}': {len(np.unique(g))} clusters")
    else:
        if args.cluster:
            print(f"  Column '{args.cluster}' not found; using unclustered SEs.")
        g = np.arange(len(rows)).astype(str)

    out = {}
    print(f"\n{'outcome':<12} {'carry':>9} {'tucked':>9} {'ATE':>10} "
          f"{'95% CI':>24} {'G':>6}")
    for name, y in (("stall", stall), ("collision", coll), ("progress_m", prog)):
        a, b = y[carry], y[~carry]
        ate = a.mean() - b.mean()
        se_a, G = cluster_se(a, g[carry])
        se_b, _ = cluster_se(b, g[~carry])
        se = np.sqrt(se_a ** 2 + se_b ** 2)
        lo, hi = ate - 1.96 * se, ate + 1.96 * se
        rel = (a.mean() / b.mean()) if b.mean() > 0 else np.nan
        out[name] = {"carry": float(a.mean()), "tucked": float(b.mean()),
                     "ate": float(ate), "se": float(se), "ci": [float(lo), float(hi)],
                     "ratio": float(rel), "n_carry": int(carry.sum()),
                     "n_tucked": int((~carry).sum()), "G": int(G)}
        print(f"{name:<12} {a.mean():9.4f} {b.mean():9.4f} {ate:+10.4f} "
              f"[{lo:+.4f}, {hi:+.4f}] {G:6d}")
        if not np.isnan(rel):
            print(f"{'':<12} ratio carry/tucked = {rel:.2f}x")

    infl = np.array([to_float(r.get(
        "param__local_costmap__inflation_layer.inflation_radius")) for r in rows])
    print("\nInteraction: effect of inflation on collision, by envelope")
    print("(logistic slope per +0.1 m inflation, with cluster-robust SE)")
    try:
        import statsmodels.api as sm
        for lab, m in (("tucked", ~carry), ("carry", carry)):
            Xi = sm.add_constant(infl[m] * 10.0)
            fit = sm.Logit(coll[m], Xi).fit(disp=0, cov_type="cluster",
                                            cov_kwds={"groups": g[m]})
            b1 = fit.params[1]
            ci = fit.conf_int()[1]
            print(f"  {lab:<8} log-odds {b1:+.4f}  OR {np.exp(b1):.3f} "
                  f"[{np.exp(ci[0]):.3f}, {np.exp(ci[1]):.3f}]  n={int(m.sum())}")
            out[f"inflation_effect_{lab}"] = {
                "log_odds": float(b1), "or": float(np.exp(b1)),
                "or_ci": [float(np.exp(ci[0])), float(np.exp(ci[1]))],
                "n": int(m.sum())}
    except ImportError:
        print("  statsmodels not installed; skipping interaction.")

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
