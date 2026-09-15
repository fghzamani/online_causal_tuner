#!/usr/bin/env python3
"""
CURE & MOBO Offline Optimization Engine (P2.1).

Ref: Hossen, Kharade, O'Kane, Schmerl, Garlan, Jamshidi,
     "CURE: Automated Simulation-Based Parameter Auto-Tuning for Mobile Robots in Complex Environments"
     IEEE Transactions on Robotics (T-RO), vol. 41, pp. 2825–2842, 2025.

Performs CURE's causal reduction: computes Average Causal Effect (ACE) and t-statistics
for each software parameter to identify causally relevant dimensions.
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def run_cure_offline(data_path: str, output_dir: str = "/home/forough/phd_projects/online_tuner/baselines/cure"):
    print("=" * 80)
    print(" RUNNING OFFLINE CURE & MOBO CAUSAL REDUCTION & OPTIMIZATION")
    print("=" * 80)

    if not os.path.exists(data_path):
        data_path = "/home/forough/phd_projects/online_tuner/rct_data_campaign_a/rct_results.csv"

    if not os.path.exists(data_path):
        print(f"Error: Dataset {data_path} not found.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} total rows from {data_path}.")

    os.makedirs(output_dir, exist_ok=True)

    target_col = "y_h" if "y_h" in df.columns else "collision"
    df["is_collision"] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
    df["progress"] = pd.to_numeric(df["probe_progress_m"], errors="coerce").fillna(0.0)

    fp_col = "param__local_costmap__footprint" if "param__local_costmap__footprint" in df.columns else [c for c in df.columns if "footprint" in c][0]
    df["arm_is_carry"] = df[fp_col].apply(lambda v: 1.0 if "carry" in str(v).lower() or "0.698" in str(v) or str(v).strip() in ("1.0", "1") else 0.0)

    v_col = "param__controller_server__FollowPath.vx_max" if "param__controller_server__FollowPath.vx_max" in df.columns else [c for c in df.columns if "vx_max" in c][0]
    w_col = "param__controller_server__FollowPath.wz_max" if "param__controller_server__FollowPath.wz_max" in df.columns else [c for c in df.columns if "wz_max" in c][0]
    c_col = "param__controller_server__FollowPath.CostCritic.cost_weight" if "param__controller_server__FollowPath.CostCritic.cost_weight" in df.columns else [c for c in df.columns if "cost_weight" in c][0]
    i_col = "param__local_costmap__inflation_layer.inflation_radius" if "param__local_costmap__inflation_layer.inflation_radius" in df.columns else [c for c in df.columns if "inflation" in c][0]

    param_cols = [v_col, w_col, c_col, i_col]

    for arm_label, arm_val in [("tucked", 0.0), ("carry", 1.0)]:
        print(f"\n--- Processing Arm Envelope: {arm_label.upper()} ---")
        sub = df[df["arm_is_carry"] == arm_val].copy()

        if len(sub) == 0:
            print(f"Warning: No rows found for arm {arm_label}.")
            continue

        sub = sub.dropna(subset=param_cols + ["progress"]).copy()
        if len(sub) == 0:
            print(f"Warning: No valid non-NaN rows found for arm {arm_label}.")
            continue

        # Compute ACE and t-statistics
        X = sub[param_cols].values
        y_prog = sub["progress"].values

        reg = LinearRegression().fit(X, y_prog)
        n, k = X.shape
        resid = y_prog - reg.predict(X)
        dof = max(1, n - k - 1)
        sigma2 = float(resid @ resid) / dof
        Xc = np.column_stack([np.ones(n), X])
        try:
            cov = sigma2 * np.linalg.pinv(Xc.T @ Xc)
            se = np.sqrt(np.diag(cov))[1:]
        except np.linalg.LinAlgError:
            se = np.full(k, np.inf)

        aces = dict(zip(param_cols, reg.coef_))
        tstats = {c: (reg.coef_[i] / se[i] if se[i] > 0 else 0.0) for i, c in enumerate(param_cols)}

        print("Average Causal Effects (ACE) on Progress:")
        for col in param_cols:
            print(f"  - {col.split('.')[-1]:<25}: ACE = {aces[col]:+.4f}  t = {tstats[col]:+.2f}")

        # CURE's causal reduction: keep only causally relevant dimensions (|t| >= 2.0)
        causal_cols = [c for c in param_cols if abs(tstats[c]) >= 2.0]
        if not causal_cols:
            causal_cols = [max(tstats, key=lambda c: abs(tstats[c]))]
        print(f"Causally relevant subset: {[c.split('.')[-1] for c in causal_cols]}")

        safe_sub = sub[sub["is_collision"] == 0]
        if len(safe_sub) == 0:
            safe_sub = sub

        def _shrunk_best(frame, cols, n_bins=4, top_k=5):
            """Best configuration region via binned cell median."""
            if len(frame) < top_k:
                return frame.loc[frame["progress"].idxmax()]
            g = frame.copy()
            keys = []
            for c in cols:
                kb = f"__bin_{c}"
                g[kb] = pd.qcut(g[c], q=min(n_bins, g[c].nunique()), labels=False, duplicates="drop")
                keys.append(kb)
            agg = g.groupby(keys)["progress"].agg(["mean", "count"])
            agg = agg[agg["count"] >= 3]
            if agg.empty:
                return frame.loc[frame["progress"].idxmax()]
            best_key = agg["mean"].idxmax()
            if not isinstance(best_key, tuple):
                best_key = (best_key,)
            mask = np.ones(len(g), dtype=bool)
            for kb, v in zip(keys, best_key):
                mask &= (g[kb] == v).values
            cell = g[mask].nlargest(top_k, "progress")
            return cell.median(numeric_only=True)

        def _to_cfg(row):
            return {
                "controller_server": {
                    "FollowPath.vx_max": round(float(row[v_col]), 3),
                    "FollowPath.wz_max": round(float(row[w_col]), 3),
                    "FollowPath.CostCritic.cost_weight": round(float(row[c_col]), 3),
                },
                "local_costmap": {
                    "inflation_layer.inflation_radius": round(float(row[i_col]), 3),
                },
            }

        # MOBO: search full space
        mobo_config = _to_cfg(_shrunk_best(safe_sub, param_cols))

        # CURE: search causally relevant dimensions only; pin non-causal dimensions to median
        cure_row = _shrunk_best(safe_sub, causal_cols).copy()
        for c in param_cols:
            if c not in causal_cols:
                cure_row[c] = float(safe_sub[c].median())
        cure_config = _to_cfg(cure_row)

        mobo_yaml_path = os.path.join(output_dir, f"mobo_config_{arm_label}.yaml")
        cure_yaml_path = os.path.join(output_dir, f"cure_config_{arm_label}.yaml")

        with open(mobo_yaml_path, "w") as f:
            yaml.dump(mobo_config, f, default_flow_style=False)

        with open(cure_yaml_path, "w") as f:
            yaml.dump(cure_config, f, default_flow_style=False)

        print(f"Saved {mobo_yaml_path} ✓")
        print(f"Saved {cure_yaml_path} ✓")

    print("\n" + "=" * 80)
    print(" OFFLINE CURE & MOBO CONFIGURATION EXTRACTION COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    data_file = "/home/forough/phd_projects/online_tuner/rct_data_campaign_a/rct_results.csv"
    run_cure_offline(data_file)
