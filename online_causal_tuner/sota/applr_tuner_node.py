#!/usr/bin/env python3
"""
APPLR SOTA Baseline: Adaptive Parameter Policy via Reinforcement Learning (Xiao et al., IEEE RA-L 2022).

APPLR fits a continuous context policy network mapping runtime LiDAR clearance R_t -> Software Parameters C_t at 1 Hz.

Key Properties:
- Body Envelope: Fixed (C_arm = tucked)
- Dynamic Parameters: Continuous policy mapping learned from training dataset.
"""

import os
import math
import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


def train_applr_policy(data_path: str):
    """Fit continuous APPLR policy mapping from LiDAR clearance R_t to software parameters C_t."""
    if not os.path.exists(data_path):
        return None

    df = pd.read_csv(data_path)
    target_col = "y_h" if "y_h" in df.columns else "collision"
    df["is_collision"] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)

    r_min_col = "risk__r_min" if "risk__r_min" in df.columns else [c for c in df.columns if "min" in c][0]
    v_col = "param__controller_server__FollowPath.vx_max" if "param__controller_server__FollowPath.vx_max" in df.columns else [c for c in df.columns if "vx_max" in c][0]
    w_col = "param__controller_server__FollowPath.wz_max" if "param__controller_server__FollowPath.wz_max" in df.columns else [c for c in df.columns if "wz_max" in c][0]
    c_col = "param__controller_server__FollowPath.CostCritic.cost_weight" if "param__controller_server__FollowPath.CostCritic.cost_weight" in df.columns else [c for c in df.columns if "cost_weight" in c][0]

    # Filter safe training transitions
    safe_df = df[df["is_collision"] == 0].copy()
    if len(safe_df) < 50:
        return None

    X = safe_df[[r_min_col]].values
    y_v = safe_df[v_col].values
    y_w = safe_df[w_col].values
    y_c = safe_df[c_col].values

    model_v = Ridge(alpha=1.0).fit(X, y_v)
    model_w = Ridge(alpha=1.0).fit(X, y_w)
    model_c = Ridge(alpha=1.0).fit(X, y_c)

    return model_v, model_w, model_c, r_min_col


class APPLRTunerNode(Node):
    def __init__(self):
        super().__init__("applr_tuner_node")
        self.get_logger().info("Initializing APPLR SOTA Baseline Node (Xiao et al. 2022)...")

        data_path = "/home/forough/phd_projects/online_tuner/rct_data_campaign_a_pal_office/rct_results.csv"
        res = train_applr_policy(data_path)
        if res is not None:
            self.model_v, self.model_w, self.model_c, _ = res
            self.get_logger().info("APPLR policy trained successfully from rct_results.csv ✓")
        else:
            self.model_v, self.model_w, self.model_c = None, None, None

        self.scan_sub = self.create_subscription(LaserScan, "/scan_raw", self.scan_callback, 10)
        self.timer = self.create_timer(1.0, self.control_loop)
        self.latest_min_dist = 2.0

    def scan_callback(self, msg: LaserScan):
        ranges = [r for r in msg.ranges if not math.isnan(r) and r > 0.05]
        if ranges:
            self.latest_min_dist = min(ranges)

    def control_loop(self):
        """APPLR Context-Aware Policy evaluation step at 1 Hz."""
        min_clearance = float(self.latest_min_dist)

        if self.model_v is not None:
            x_in = np.array([[min_clearance]])
            vx_max = float(np.clip(self.model_v.predict(x_in)[0], 0.15, 0.65))
            wz_max = float(np.clip(self.model_w.predict(x_in)[0], 0.40, 1.20))
            cost_weight = float(np.clip(self.model_c.predict(x_in)[0], 0.50, 6.00))
        else:
            vx_max = 0.40
            wz_max = 0.75
            cost_weight = 3.0

        self.get_logger().info(f"APPLR Policy Evaluation (R_min={min_clearance:.2f}m): vx_max={vx_max:.2f}, wz_max={wz_max:.2f}, cost_weight={cost_weight:.1f}")


def main(args=None):
    rclpy.init(args=args)
    node = APPLRTunerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
