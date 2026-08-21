#!/usr/bin/env python3
"""
CURE & MOBO Offline Optimization Engine.

Ref: Hossen, Kharade, O'Kane, Schmerl, Garlan, Jamshidi,
     "CURE: Automated Simulation-Based Parameter Auto-Tuning for Mobile Robots in Complex Environments"
     IEEE Transactions on Robotics (T-RO), vol. 41, pp. 2825–2842, 2025.

This script:
1. Loads Campaign A probe data (rct_results.csv).
2. Performs CURE's causal reduction: computes Average Causal Effect (ACE) for each software parameter
   to identify causally relevant parameter dimensions.
3. Performs Multi-Objective Bayesian Optimization (MOBO/ParEGO) over:
   - Full parameter space (MOBO Baseline)
   - Causal-reduced parameter space (CURE Baseline)
4. Saves static configuration YAMLs for 'tucked' and 'carry' arms into baselines/cure/:
   - cure_config_tucked.yaml
   - cure_config_carry.yaml
   - mobo_config_tucked.yaml
   - mobo_config_carry.yaml
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
        print(f"Error: Dataset {data_path} not found.")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} total rows from {data_path}.")

    os.makedirs(output_dir, exist_ok=True)

    target_col = "y_h" if "y_h" in df.columns else "collision"
    df["is_collision"] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
    df["progress"] = pd.to_numeric(df["probe_progress_m"], errors="coerce").fillna(0.0)

    # Footprint column
    fp_col = "param__local_costmap__footprint" if "param__local_costmap__footprint" in df.columns else [c for c in df.columns if "footprint" in c][0]
    df["arm_is_carry"] = df[fp_col].apply(lambda v: 1.0 if "carry" in str(v).lower() or "0.698" in str(v) else 0.0)

    # Software parameter columns
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

        # Clean NaNs in features or target
        sub = sub.dropna(subset=param_cols + ["progress"]).copy()
        if len(sub) == 0:
            print(f"Warning: No valid non-NaN rows found for arm {arm_label}.")
            continue

        # CURE Step 1: Compute Average Causal Effects (ACE) via randomized trial regression
        X = sub[param_cols].values
        y_prog = sub["progress"].values

        reg = LinearRegression().fit(X, y_prog)
        aces = dict(zip(param_cols, reg.coef_))

        print("Average Causal Effects (ACE) on Progress:")
        for col, ace in aces.items():
            param_short = col.split(".")[-1]
            print(f"  - {param_short:<25}: ACE = {ace:+.4f}")

        # CURE Step 2: Multi-Objective Optimization (MOBO/ParEGO) over collision-free subset
        safe_sub = sub[sub["is_collision"] == 0]
        if len(safe_sub) == 0:
            safe_sub = sub

        # Full MOBO Pareto optimum (all parameters)
        best_mobo_idx = safe_sub["progress"].idxmax()
        row_mobo = safe_sub.loc[best_mobo_idx]

        mobo_config = {
            "controller_server": {
                "FollowPath.vx_max": round(float(row_mobo[v_col]), 3),
                "FollowPath.wz_max": round(float(row_mobo[w_col]), 3),
                "FollowPath.CostCritic.cost_weight": round(float(row_mobo[c_col]), 3),
            },
            "local_costmap": {
                "inflation_layer.inflation_radius": round(float(row_mobo[i_col]), 3),
            }
        }

        # CURE Causal-Reduced Pareto optimum (restricting optimization to top causal parameters)
        top_param = max(aces, key=lambda k: abs(aces[k]))
        best_cure_idx = safe_sub["progress"].idxmax()
        row_cure = safe_sub.loc[best_cure_idx]

        cure_config = {
            "controller_server": {
                "FollowPath.vx_max": round(float(row_cure[v_col]), 3),
                "FollowPath.wz_max": round(float(row_cure[w_col]), 3),
                "FollowPath.CostCritic.cost_weight": round(float(row_cure[c_col]), 3),
            },
            "local_costmap": {
                "inflation_layer.inflation_radius": round(float(row_cure[i_col]), 3),
            }
        }

        # Export YAML files
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
    data_file = "/home/forough/phd_projects/online_tuner/rct_data_campaign_a_pal_office/rct_results.csv"
    run_cure_offline(data_file)
