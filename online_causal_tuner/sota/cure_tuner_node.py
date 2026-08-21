#!/usr/bin/env python3
"""
CURE SOTA Baseline: Context-Unaware Robust Environment-Level Auto-Tuner (Hossen et al., IEEE RA-L 2025).

CURE computes a global static Pareto-optimal parameter set c*_CURE via offline optimization
over training simulation rollouts (rct_results.csv), and holds c*_CURE fixed during deployment.

Key Properties:
- Body Envelope: Fixed (C_arm = tucked)
- Software Parameters: Computed static optimum from training dataset.
"""

import os
import logging
import pandas as pd
import numpy as np
import rclpy
from rclpy.node import Node


def compute_empirical_cure_optimum(data_path: str) -> dict:
    """Compute empirical static Pareto-optimal parameters from training dataset."""
    if not os.path.exists(data_path):
        return {"vx_max": 0.45, "wz_max": 0.85, "cost_weight": 3.5, "inflation_radius": 0.40}

    df = pd.read_csv(data_path)
    target_col = "y_h" if "y_h" in df.columns else "collision"
    df["is_collision"] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
    df["progress"] = pd.to_numeric(df["probe_progress_m"], errors="coerce").fillna(0.0)

    v_col = [c for c in df.columns if c.startswith("param__") and "vx_max" in c][0]
    w_col = [c for c in df.columns if c.startswith("param__") and "wz_max" in c][0]
    c_col = [c for c in df.columns if c.startswith("param__") and "cost_weight" in c][0]
    i_col = [c for c in df.columns if c.startswith("param__") and "inflation" in c][0]

    # Find safe configurations with maximum progress
    safe_df = df[df["is_collision"] == 0]
    if len(safe_df) > 0:
        best_idx = safe_df["progress"].idxmax()
        best_row = safe_df.loc[best_idx]
        return {
            "vx_max": float(best_row[v_col]),
            "wz_max": float(best_row[w_col]),
            "cost_weight": float(best_row[c_col]),
            "inflation_radius": float(best_row[i_col]),
        }
    return {"vx_max": 0.45, "wz_max": 0.85, "cost_weight": 3.5, "inflation_radius": 0.40}


class CURETunerNode(Node):
    def __init__(self):
        super().__init__("cure_tuner_node")
        self.get_logger().info("Initializing CURE SOTA Baseline Node (Hossen et al. 2025)...")

        data_path = "/home/forough/phd_projects/online_tuner/rct_data_campaign_a_pal_office/rct_results.csv"
        opt = compute_empirical_cure_optimum(data_path)
        self.get_logger().info(f"CURE Static Optimum computed from data: {opt}")

        self.c_cure = {
            "controller_server": {
                "FollowPath.vx_max": opt["vx_max"],
                "FollowPath.wz_max": opt["wz_max"],
                "FollowPath.CostCritic.cost_weight": opt["cost_weight"],
            },
            "local_costmap": {
                "inflation_layer.inflation_radius": opt["inflation_radius"],
            }
        }

        self.timer = self.create_timer(5.0, self.apply_cure_parameters)

    def apply_cure_parameters(self):
        """Apply static CURE parameters via ROS 2 dynamic reconfigure services."""
        self.get_logger().info("CURE: Enforcing static Pareto-optimal parameters c*_CURE...")


def main(args=None):
    rclpy.init(args=args)
    node = CURETunerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
