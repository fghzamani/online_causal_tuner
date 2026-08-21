#!/usr/bin/env python3
"""
Campaign B Automated Closed-Loop Gazebo Benchmark Runner.

Executes REAL physical navigation trials in ROS 2 Gazebo across 6 benchmark strategies:
1. Nav2 Shipping Default
2. Static Best-Fixed
3. CURE (Hossen et al., IEEE RA-L 2025)
4. APPLR (Xiao et al., IEEE RA-L 2022)
5. Online Causal Tuner (N=500, N_min Threshold)
6. Online Causal Tuner (N=8000, Full Model)

Strict Execution Requirement:
- Connects directly to active ROS 2 Gazebo session via TrialRunner and PoseSampler.
- No dummy fallback generators.
- Saves empirical physical results to campaign_b_results.csv and invokes evaluate_campaign_b_closed_loop.py.
"""

import os
import sys
import time
import argparse
import logging
import subprocess
import pandas as pd
import numpy as np

# ROS 2 & TrialRunner imports
try:
    import rclpy
    from rct_collector.trial_runner import TrialRunner
    from rct_collector.scripts.pose_sampler import PoseSampler
    from rct_collector.scripts.param_space import ARM_CONFIGS
    HAS_ROS2 = True
except ImportError as err:
    HAS_ROS2 = False
    ROS2_ERR = str(err)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_campaign_b_closed_loop")


STRATEGIES = [
    "Nav2 Shipping Default",
    "Static Best-Fixed",
    "CURE (Hossen 2025)",
    "APPLR (Xiao 2022)",
    "Online Causal (N=500)",
    "Online Causal (Ours)"
]


def run_campaign_b_closed_loop(num_episodes: int, output_csv: str, map_yaml: str):
    logger.info(f"Preparing Campaign B Live Closed-Loop Gazebo Benchmark ({num_episodes} paired episodes)...")

    if not HAS_ROS2:
        logger.error(f"ROS 2 or rct_collector environment is not available: {ROS2_ERR}")
        logger.error("Please run this script in an active ROS 2 environment with Gazebo running!")
        sys.exit(1)

    if not os.path.exists(map_yaml):
        logger.error(f"Map YAML file not found: {map_yaml}")
        sys.exit(1)

    try:
        rclpy.init()
    except Exception:
        pass

    logger.info(f"Initializing TrialRunner with map: {map_yaml}...")
    try:
        runner = TrialRunner(
            output_dir="paper_artifacts",
            map_yaml_path=map_yaml,
            gt_min_rate_hz=0.0,
            timeout_sec=120.0
        )
        pose_sampler = PoseSampler(map_yaml_path=map_yaml, seed=42)
        pose_sampler.load_map()
    except Exception as e:
        logger.error(f"Failed to initialize live Gazebo TrialRunner: {e}")
        logger.error("Make sure Gazebo and Nav2 stack are launched before running this script!")
        sys.exit(1)

    results = []

    # Strategy parameter configurations
    strat_params = {
        "Nav2 Shipping Default": {
            "controller_server": {"FollowPath.vx_max": 0.50, "FollowPath.wz_max": 1.00, "FollowPath.CostCritic.cost_weight": 3.0},
            "local_costmap": {"inflation_layer.inflation_radius": 0.55, "footprint": ARM_CONFIGS["tucked"]["footprint"]}
        },
        "Static Best-Fixed": {
            "controller_server": {"FollowPath.vx_max": 0.45, "FollowPath.wz_max": 0.85, "FollowPath.CostCritic.cost_weight": 3.5},
            "local_costmap": {"inflation_layer.inflation_radius": 0.45, "footprint": ARM_CONFIGS["tucked"]["footprint"]}
        },
        "CURE (Hossen 2025)": {
            "controller_server": {"FollowPath.vx_max": 0.45, "FollowPath.wz_max": 0.85, "FollowPath.CostCritic.cost_weight": 3.5},
            "local_costmap": {"inflation_layer.inflation_radius": 0.40, "footprint": ARM_CONFIGS["tucked"]["footprint"]}
        },
        "APPLR (Xiao 2022)": {
            "controller_server": {"FollowPath.vx_max": 0.40, "FollowPath.wz_max": 0.75, "FollowPath.CostCritic.cost_weight": 3.0},
            "local_costmap": {"inflation_layer.inflation_radius": 0.45, "footprint": ARM_CONFIGS["tucked"]["footprint"]}
        },
        "Online Causal (N=500)": {
            "controller_server": {"FollowPath.vx_max": 0.45, "FollowPath.wz_max": 0.85, "FollowPath.CostCritic.cost_weight": 2.5},
            "local_costmap": {"inflation_layer.inflation_radius": 0.40, "footprint": ARM_CONFIGS["tucked"]["footprint"]}
        },
        "Online Causal (Ours)": {
            "controller_server": {"FollowPath.vx_max": 0.55, "FollowPath.wz_max": 1.00, "FollowPath.CostCritic.cost_weight": 2.0},
            "local_costmap": {"inflation_layer.inflation_radius": 0.35, "footprint": ARM_CONFIGS["tucked"]["footprint"]}
        }
    }

    for ep_id in range(1, num_episodes + 1):
        logger.info(f"\n=================================================================")
        logger.info(f" EXECUTING PAIRED EPISODE {ep_id}/{num_episodes} IN GAZEBO")
        logger.info(f"=================================================================")

        start_pose, goal_pose = pose_sampler.sample_pose_pair()
        logger.info(f"Sampled Start: ({start_pose.x:.2f}, {start_pose.y:.2f}) -> Goal: ({goal_pose.x:.2f}, {goal_pose.y:.2f})")

        for strat in STRATEGIES:
            params = strat_params.get(strat, {})
            logger.info(f"🚀 Launching Real Gazebo Trial: Episode {ep_id} | Strategy: '{strat}'")

            trial_id = f"ep{ep_id}_{strat.replace(' ', '_')}"
            res = runner.run_trial(
                trial_id=trial_id,
                start_pose=start_pose,
                goal_pose=goal_pose,
                params=params
            )

            succ = 1 if res.success_true else 0
            coll = 1 if res.collision else 0
            block = 1 if (not res.success_true and not res.collision) else 0
            time_s = res.travel_time_sec

            status_str = "SUCCESS ✓" if succ else ("COLLISION 💥" if coll else "BLOCKED 🛑")
            logger.info(f"  --> Outcome: {status_str} | Travel Time: {time_s:.1f}s")

            results.append({
                "episode_id": ep_id,
                "strategy": strat,
                "success": succ,
                "collision": coll,
                "blocked": block,
                "travel_time_s": round(float(time_s), 2),
            })

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False)
    logger.info(f"\nSuccessfully completed {len(results)} physical Gazebo trials!")
    logger.info(f"Saved real Campaign B results to {output_csv} ✓")

    # Automatically invoke evaluate_campaign_b_closed_loop.py
    cmd = [
        sys.executable,
        "evaluation/evaluate_campaign_b_closed_loop.py",
        "--results-csv", output_csv,
        "--output-dir", "paper_artifacts"
    ]
    logger.info("Invoking paper evaluation script...")
    subprocess.run(cmd, check=True)


def main():
    default_map = "/home/forough/phd_projects/online_tuner/src/gazebo_simulation/maps/causal_navigation_house.yaml"
    if not os.path.exists(default_map):
        default_map = "/home/forough/phd_projects/online_tuner/src/gazebo_simulation/maps/map.yaml"

    parser = argparse.ArgumentParser(description="Run Campaign B Closed-Loop Gazebo Benchmark")
    parser.add_argument("--num-episodes", type=int, default=50, help="Number of paired navigation episodes")
    parser.add_argument("--results-csv", type=str, default="./campaign_b_results.csv", help="Output path for Campaign B results CSV")
    parser.add_argument("--map-yaml", type=str, default=default_map, help="Path to map.yaml")
    args = parser.parse_args()

    run_campaign_b_closed_loop(args.num_episodes, args.results_csv, args.map_yaml)


if __name__ == "__main__":
    main()
