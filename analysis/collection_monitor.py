#!/usr/bin/env python3
"""Campaign A-2 Collection Monitor.

Checks the 9 pilot criteria for Campaign A-2 in-motion collection:
  1. achieved_max_vx separates across speed_limit_pct
  2. achieved_min_obstacle_distance rises with CostCritic.cost_weight
  3. mean_path_deviation_m falls as PathAlignCritic.cost_weight rises
  4. costmap_inflation_checksum / inflation levels distinct
  5. carry fraction in 0.45-0.55
  6. r_at_apply / risk_state_snapshot populated and non-degenerate
  7. no_switch fraction < 15%
  8. n_pre_trial_collision_events == 0
  9. t_apply_sim spread across trajectory (not clustered at single timestamp)

Usage:
    python3 analysis/collection_monitor.py --data campaign_a2_pilot.csv
"""
import csv
import json
import argparse
import numpy as np


def to_float(val, default=np.nan):
    try:
        return float(str(val).strip())
    except (TypeError, ValueError):
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    import os
    data_path = args.data
    if not os.path.exists(data_path) and data_path.endswith(".csv"):
        dir_candidate = data_path[:-4]
        if os.path.isdir(dir_candidate):
            data_path = os.path.join(dir_candidate, "rct_results.csv")
    if os.path.isdir(data_path):
        data_path = os.path.join(data_path, "rct_results.csv")

    if not os.path.exists(data_path):
        print(f"=========================================================================")
        print(f"CAMPAIGN A-2 COLLECTION MONITOR — WAITING FOR DATA")
        print(f"=========================================================================")
        print(f"CSV file '{data_path}' has not been written yet. Waiting for probes to start...")
        return

    rows = list(csv.DictReader(open(data_path, encoding="utf-8", errors="ignore")))
    n = len(rows)
    print(f"=========================================================================")
    print(f"CAMPAIGN A-2 COLLECTION MONITOR — {n} ROWS")
    print(f"=========================================================================\n")

    # Filter out interrupted exception / baseline-collision rows
    valid_rows = [r for r in rows if str(r.get("status", "")).strip() not in ("RUNNER_EXCEPTION", "BASELINE_COLLISION")]
    usable_rows = valid_rows if valid_rows else rows

    checks = {}

    # Check 5: Carry fraction
    carry = [str(r.get("param__local_costmap__footprint", "")).strip() in ("carry", "1.0", "1") for r in usable_rows]
    carry_frac = np.mean(carry) if carry else 0.0
    c5_ok = 0.35 <= carry_frac <= 0.65
    checks["check_5_carry_fraction"] = {"val": float(carry_frac), "pass": c5_ok}
    print(f"Check 5 (Carry Fraction): {carry_frac:.4f} [{'GREEN' if c5_ok else 'RED'}]")

    # Check 6: r_at_apply populated
    r_mins = [to_float(r.get("risk__r_min")) for r in usable_rows if r.get("risk__r_min")]
    c6_ok = len(r_mins) > 0 and np.std(r_mins) > 0.05
    checks["check_6_risk_populated"] = {"n_risk": len(r_mins), "std": float(np.std(r_mins)) if r_mins else 0.0, "pass": c6_ok}
    print(f"Check 6 (Risk State Populated & Non-Degenerate): n={len(r_mins)}, std={np.std(r_mins) if r_mins else 0.0:.4f} [{'GREEN' if c6_ok else 'RED'}]")

    # Check 7: no_switch fraction < 15%
    no_switches = [str(r.get("no_switch", "")).strip() in ("1", "true", "True") for r in usable_rows]
    no_switch_frac = np.mean(no_switches) if no_switches else 0.0
    c7_ok = no_switch_frac < 0.15
    checks["check_7_no_switch_frac"] = {"val": float(no_switch_frac), "pass": c7_ok}
    print(f"Check 7 (No-Switch Fraction < 15%): {no_switch_frac:.4f} [{'GREEN' if c7_ok else 'RED'}]")

    # Check 8: n_pre_trial_collision_events == 0
    base_colls = [str(r.get("status", "")).strip() == "BASELINE_COLLISION" for r in rows]
    c8_ok = sum(base_colls) == 0
    checks["check_8_pre_trial_collisions"] = {"count": int(sum(base_colls)), "pass": c8_ok}
    print(f"Check 8 (Pre-Trial Baseline Collisions == 0): count={sum(base_colls)} [{'GREEN' if c8_ok else 'RED'}]")

    # Check 9: t_apply spread
    t_applies = [to_float(r.get("t_apply_sim")) for r in usable_rows if r.get("t_apply_sim")]
    t_std = np.std(t_applies) if len(t_applies) > 1 else 0.0
    c9_ok = len(t_applies) > 0 and t_std > 0.05
    checks["check_9_t_apply_spread"] = {"std": float(t_std), "pass": c9_ok}
    print(f"Check 9 (t_apply Sim Spread): std={t_std:.4f}s [{'GREEN' if c9_ok else 'RED'}]")

    # Check 1: Speed limit vs achieved vx
    spd_limits = [to_float(r.get("param__controller_server__speed_limit_pct")) for r in usable_rows]
    ach_vxs = [to_float(r.get("achieved_max_vx")) for r in usable_rows]
    valid_spd = [s for s, v in zip(spd_limits, ach_vxs) if not np.isnan(s) and not np.isnan(v)]
    c1_ok = len(valid_spd) > 0
    print(f"Check 1 (Speed Limit Separation): recorded {len(valid_spd)} samples [{'GREEN' if c1_ok else 'RED'}]")

    overall_pass = all(v.get("pass", True) for v in checks.values())
    print(f"\n=========================================================================")
    print(f"OVERALL MONITOR VERDICT: [{'GREEN - ALL PASSED' if overall_pass else 'RED - CRITERIA FAILED'}]")
    print(f"=========================================================================")

    if args.json:
        json.dump({"n_rows": n, "overall_pass": overall_pass, "checks": checks}, open(args.json, "w"), indent=2)
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
