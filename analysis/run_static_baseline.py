#!/usr/bin/env python3
"""
Static Baseline Trial Runner for Campaign B.

Applies exported static baseline YAML configurations (e.g. CURE, MOBO, Nav2 Default)
and runs physical navigation trials in Gazebo using rct_collector.trial_runner.

Usage:
  docker exec ros2-tiago-dev python3 /home/forough/phd_projects/online_tuner/analysis/run_static_baseline.py \
      --strategy CURE \
      --config baselines/cure/cure_config_tucked.yaml \
      --num-episodes 50
"""

import os
import sys
import yaml
import argparse
import logging
import pandas as pd
import numpy as np

import rclpy
from rct_collector.trial_runner import TrialRunner
from rct_collector.scripts.pose_sampler import PoseSampler
from rct_collector.scripts.param_space import ARM_CONFIGS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_static_baseline")


def run_static_baseline(strategy_name: str, config_yaml: str, num_episodes: int, map_yaml: str, output_csv: str):
    logger.info(f"Preparing Static Baseline Evaluation for '{strategy_name}' ({num_episodes} episodes)...")

    # Load baseline parameters from YAML if provided
    params = {}
    if os.path.exists(config_yaml):
        with open(config_yaml, "r") as f:
            params = yaml.safe_load(f)
        logger.info(f"Loaded static parameters from {config_yaml}: {params}")
    else:
        logger.info(f"No YAML provided for {strategy_name}; using default parameters.")

    if not rclpy.ok():
        rclpy.init()

    output_dir = "campaign-b-data"
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Connecting TrialRunner to active ROS 2 Gazebo session (map: {map_yaml})...")
    runner = TrialRunner(
        output_dir=output_dir,
        map_yaml_path=map_yaml,
        gt_min_rate_hz=0.0,
        timeout_sec=120.0
    )

    pose_sampler = PoseSampler(map_yaml_path=map_yaml, seed=42)
    pose_sampler.load_map()

    results = []

    for ep_id in range(1, num_episodes + 1):
        logger.info(f"\n--- Episode {ep_id}/{num_episodes} | Strategy: '{strategy_name}' ---")
        start_pose, goal_pose = pose_sampler.sample_pose_pair()

        trial_id = f"ep{ep_id}_{strategy_name.replace(' ', '_')}"
        res = runner.run_trial(
            trial_id=trial_id,
            start_pose=start_pose,
            goal_pose=goal_pose,
            params=params
        )

        succ = 1 if res.success_true else 0
        coll = 1 if res.collision else 0
        block = 1 if (not res.success_true and not res.collision) else 0
        time_s = res.travel_time_sec or 0.0

        status_str = "SUCCESS ✓" if succ else ("COLLISION 💥" if coll else "BLOCKED 🛑")
        logger.info(f"Outcome: {status_str} | Travel Time: {time_s:.1f}s | Path Length: {res.path_length_m:.2f}m")

        results.append({
            "episode_id": ep_id,
            "strategy": strategy_name,
            "success": succ,
            "collision": coll,
            "blocked": block,
            "travel_time_s": round(float(time_s), 2),
            "status": res.status,
            "failure_reason": res.failure_reason
        })

    df_out = pd.DataFrame(results)
    
    # Append or write CSV
    if os.path.exists(output_csv):
        df_old = pd.read_csv(output_csv)
        df_out = pd.concat([df_old, df_out], ignore_index=True)

    df_out.to_csv(output_csv, index=False)
    logger.info(f"Successfully recorded results for strategy '{strategy_name}' to {output_csv} ✓")


def main():
    default_map = "/home/forough/phd_projects/online_tuner/src/gazebo_simulation/maps/causal_navigation_house.yaml"
    if not os.path.exists(default_map):
        default_map = "/home/forough/phd_projects/online_tuner/src/gazebo_simulation/maps/map.yaml"

    parser = argparse.ArgumentParser(description="Run Static Baseline Navigation Trials")
    parser.add_argument("--strategy", type=str, required=True, help="Strategy name (e.g. 'CURE (Hossen 2025)', 'MOBO Auto-Tuning')")
    parser.add_argument("--config", type=str, default="", help="Path to static baseline YAML config")
    parser.add_argument("--num-episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--map-yaml", type=str, default=default_map, help="Path to map.yaml")
    parser.add_argument("--results-csv", type=str, default="campaign-b-data/campaign_b_results.csv", help="Output CSV path")
    args = parser.parse_args()

    run_static_baseline(args.strategy, args.config, args.num_episodes, args.map_yaml, args.results_csv)


if __name__ == "__main__":
    main()
