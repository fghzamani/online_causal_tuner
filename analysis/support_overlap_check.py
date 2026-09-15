#!/usr/bin/env python3
"""
Day 3: Support-Overlap Check & Figure 2 Generator.

Compares recorded environment risk state vectors R_t against Campaign A's
training envelope (artifact["support"]) to verify contextual support overlap
and generate Figure 2 for the paper.

Usage:
  # Live ROS 2 mode (subscribes to /risk_state for 300 seconds):
  python3 support_overlap_check.py --duration 300 --output-dir ./campaign-b-data

  # Offline CSV mode:
  python3 support_overlap_check.py --data-csv ./recorded_risk_states.csv

  # Dry-run test mode:
  python3 support_overlap_check.py --dummy-test
"""

import os
import sys
import time
import pickle
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False

import threading

sys.path.insert(0, "/home/forough/phd_projects/online_tuner/src/rct_data_collector")
from rct_collector.trial_runner import TrialRunner
from rct_collector.scripts.pose_sampler import PoseSampler

sys.path.insert(0, "/home/forough/phd_projects/online_tuner/src/online_causal_tuner")
import online_causal_tuner.train_causal_models as tcm
sys.modules["__main__"].CausalInteractionTransformer = tcm.CausalInteractionTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("support_overlap_check")

RISK_FEATURE_NAMES = [
    "risk__r_min",
    "risk__r_width",
    "risk__r_ttc",
    "risk__r_dens",
    "risk__r_clear",
    "risk__r_curve",
]


class RiskStateRecorder(Node):
    """ROS 2 Node to subscribe and record /risk_state messages."""
    def __init__(self, max_samples=5000):
        super().__init__("risk_state_recorder")
        self.max_samples = max_samples
        self.samples = []
        self.sub = self.create_subscription(
            Float64MultiArray, "/risk_state", self._callback, 10
        )
        self.get_logger().info("RiskStateRecorder subscribed to /risk_state...")

    def _callback(self, msg: Float64MultiArray):
        if len(self.samples) < self.max_samples:
            if len(msg.data) >= len(RISK_FEATURE_NAMES):
                row = list(msg.data[: len(RISK_FEATURE_NAMES)])
                self.samples.append(row)
                if len(self.samples) % 200 == 0:
                    self.get_logger().info(f"Recorded {len(self.samples)}/{self.max_samples} risk state samples...")


def generate_dummy_samples(num_samples=1000):
    """Generate realistic dummy risk state samples for testing script logic."""
    rng = np.random.default_rng(42)
    r_min = rng.uniform(0.35, 3.5, num_samples)
    r_width = r_min * rng.uniform(1.2, 2.5, num_samples)
    r_ttc = r_min * rng.uniform(0.8, 4.0, num_samples)
    r_dens = rng.uniform(0.001, 0.05, num_samples)
    r_clear = rng.uniform(0.5, 3.5, num_samples)
    r_curve = rng.uniform(0.0, 0.5, num_samples)
    return np.column_stack([r_min, r_width, r_ttc, r_dens, r_clear, r_curve])


def analyze_support_overlap(samples_mat, artifact, output_dir="./campaign-b-data"):
    """Compute support coverage metrics and plot Figure 2."""
    os.makedirs(output_dir, exist_ok=True)
    support = artifact.get("support", {})

    print("\n" + "=" * 75)
    print("=== ICRA 2027 DAY 3: SUPPORT-OVERLAP DIAGNOSTIC ANALYSIS ===")
    print("=" * 75)

    num_samples = samples_mat.shape[0]
    inside_mask = np.ones(num_samples, dtype=bool)

    metrics_rows = []

    for idx, col in enumerate(RISK_FEATURE_NAMES):
        vals = samples_mat[:, idx]
        v_min, v_max = float(np.min(vals)), float(np.max(vals))
        v_mean, v_std = float(np.mean(vals)), float(np.std(vals))

        supp_info = support.get(col, {"min": 0.0, "max": 10.0})
        s_min, s_max = float(supp_info["min"]), float(supp_info["max"])

        # Check dimension bound compliance
        dim_inside = (vals >= s_min) & (vals <= s_max)
        inside_mask = inside_mask & dim_inside
        coverage_pct = float(np.mean(dim_inside) * 100.0)

        metrics_rows.append({
            "feature": col,
            "recorded_min": round(v_min, 3),
            "recorded_max": round(v_max, 3),
            "recorded_mean": round(v_mean, 3),
            "support_min": round(s_min, 3),
            "support_max": round(s_max, 3),
            "dim_coverage_pct": round(coverage_pct, 2),
        })

    df_metrics = pd.DataFrame(metrics_rows)
    overall_coverage_pct = float(np.mean(inside_mask) * 100.0)

    print("\n--- Feature-wise Support Coverage Metrics ---")
    print(df_metrics.to_string(index=False))
    print("-" * 75)
    print(f"OVERALL CONTEXT ENVELOPE COVERAGE: {overall_coverage_pct:.2f}% of samples strictly inside Campaign A bounds.")
    print("=" * 75 + "\n")

    # Save metrics CSV
    csv_path = os.path.join(output_dir, "support_overlap_metrics.csv")
    df_metrics.to_csv(csv_path, index=False)
    logger.info(f"Saved support overlap metrics CSV to {csv_path}")

    # Generate Figure 2 Plot
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Figure 2: Risk Context Distribution Overlap vs. Campaign A Training Support Envelope", fontsize=14, fontweight="bold")

    feature_titles = {
        "risk__r_min": "Obstacle Distance r_min (m)",
        "risk__r_width": "Passage Width r_width (m)",
        "risk__r_ttc": "Time-to-Collision r_ttc (s)",
        "risk__r_dens": "Obstacle Density r_dens",
        "risk__r_clear": "Heading Clearance r_clear (m)",
        "risk__r_curve": "Path Curvature r_curve (rad/m)",
    }

    for idx, col in enumerate(RISK_FEATURE_NAMES):
        ax = axes[idx // 3, idx % 3]
        vals = samples_mat[:, idx]
        supp_info = support.get(col, {"min": float(vals.min()), "max": float(vals.max())})
        s_min, s_max = float(supp_info["min"]), float(supp_info["max"])

        ax.hist(vals, bins=30, color="#1f77b4", alpha=0.7, edgecolor="black", label="Recorded Target Map")
        ax.axvline(s_min, color="red", linestyle="--", linewidth=2.0, label="Training Support Min")
        ax.axvline(s_max, color="red", linestyle="--", linewidth=2.0, label="Training Support Max")

        ax.set_title(feature_titles.get(col, col), fontsize=11, fontweight="bold")
        ax.set_xlabel("Value")
        ax.set_ylabel("Sample Count")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = os.path.join(output_dir, "figure2_support_overlap.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved Figure 2 plot to {fig_path} ✓")

    return overall_coverage_pct


def main():
    parser = argparse.ArgumentParser(description="Day 3 Support-Overlap Check & Figure 2 Generator")
    parser.add_argument("--model-path", default="./src/online_causal_tuner/models/causal_tuner_models.pkl")
    parser.add_argument("--map-yaml", default="", help="Path to map YAML file for active TrialRunner navigation")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of navigation episodes to drive")
    parser.add_argument("--min-goal-dist", type=float, default=3.0, help="Min Euclidean distance (m) between start and goal poses")
    parser.add_argument("--data-csv", default="", help="Path to pre-recorded risk states CSV")
    parser.add_argument("--output-dir", default="./campaign-b-data")
    parser.add_argument("--duration", type=float, default=600.0, help="Max duration (sec) to record live ROS 2 risk states")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--dummy-test", action="store_true", help="Run test mode with generated dummy samples")
    args = parser.parse_args()

    abs_model_path = os.path.abspath(args.model_path)
    if not os.path.exists(abs_model_path):
        logger.error(f"Model file not found at {abs_model_path}!")
        return

    with open(abs_model_path, "rb") as f:
        artifact = pickle.load(f)
    logger.info(f"Loaded artifact v{artifact.get('artifact_version', 1)} from {abs_model_path}")

    if args.dummy_test:
        logger.info("Running script in --dummy-test mode...")
        samples_mat = generate_dummy_samples(num_samples=1000)
    elif args.data_csv:
        logger.info(f"Loading recorded risk state CSV from {args.data_csv}...")
        df_in = pd.read_csv(args.data_csv)
        samples_mat = df_in[RISK_FEATURE_NAMES].to_numpy(dtype=np.float64)
    elif HAS_ROS2:
        logger.info("Initializing ROS 2 for active navigation and risk state recording...")
        recorded_samples = []

        if args.map_yaml:
            abs_map_path = os.path.abspath(args.map_yaml)
            if not os.path.exists(abs_map_path):
                logger.error(f"Map YAML file NOT found at: '{args.map_yaml}' (abs: '{abs_map_path}')!")
                logger.error("Please provide a valid map YAML path (e.g. './src/gazebo_simulation/maps/map.yaml').")
                return

            logger.info(f"Starting active navigation driving using TrialRunner (map: {abs_map_path}, episodes: {args.num_episodes}, min_dist: {args.min_goal_dist}m)...")
            runner = TrialRunner(
                output_dir=args.output_dir,
                map_yaml_path=abs_map_path,
                collect_risk_features=True,
                timeout_sec=240.0
            )
            pose_sampler = PoseSampler(
                map_yaml_path=abs_map_path,
                min_goal_distance=args.min_goal_dist,
                max_goal_distance=15.0,
                seed=42
            )
            pose_sampler.load_map()

            for ep_id in range(1, args.num_episodes + 1):
                if len(recorded_samples) >= args.max_samples:
                    break
                logger.info(f"--- Support Check Episode {ep_id}/{args.num_episodes} ---")
                start_pose, goal_pose = pose_sampler.sample_start_goal()

                runner._teleport_robot(start_pose)
                runner._publish_initial_pose(start_pose)
                time.sleep(1.0)

                res = runner.run_trial(
                    trial_id=f"support_ep{ep_id}",
                    start_pose=start_pose,
                    goal_pose=goal_pose,
                    params={"controller_server": {"FollowPath.vx_max": 0.55}}
                )

                # Extract recorded risk states natively from trial path
                if hasattr(res, "controller_path"):
                    for entry in res.controller_path:
                        rs = entry.get("risk_state")
                        if isinstance(rs, dict):
                            row = [
                                float(rs.get("r_min", 0.0)),
                                float(rs.get("r_width", 0.0)),
                                float(rs.get("r_ttc", 0.0)),
                                float(rs.get("r_dens", 0.0)),
                                float(rs.get("r_clear", 0.0)),
                                float(rs.get("r_curve", 0.0)),
                            ]
                            recorded_samples.append(row)

                logger.info(f"Episode {ep_id} finished ({res.status}). Total risk samples recorded: {len(recorded_samples)}")
        else:
            logger.info(f"No --map-yaml provided; passive subscriber mode...")
            rclpy.init()
            recorder = RiskStateRecorder(max_samples=args.max_samples)
            spin_thread = threading.Thread(target=rclpy.spin, args=(recorder,), daemon=True)
            spin_thread.start()
            time.sleep(args.duration)
            recorded_samples = recorder.samples
            recorder.destroy_node()
            rclpy.shutdown()

        samples_mat = np.array(recorded_samples, dtype=np.float64)
    else:
        logger.error("No ROS 2 environment found and no --data-csv or --dummy-test passed!")
        return

    if samples_mat.shape[0] == 0:
        logger.error("No samples recorded or loaded. Exiting.")
        return

    analyze_support_overlap(samples_mat, artifact, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
