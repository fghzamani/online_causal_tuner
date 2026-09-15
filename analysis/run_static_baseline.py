#!/usr/bin/env python3
"""
Static Baseline Trial Runner for Campaign B.

Applies exported static baseline YAML configurations (e.g. CURE, MOBO, Nav2 Default)
and runs physical navigation trials in Gazebo using rct_collector.trial_runner.
"""

import os
import sys
import subprocess
import time
import yaml
import argparse
import logging
import threading
import pandas as pd
import numpy as np

# Ensure rct_collector package is in path
sys.path.insert(0, "/home/forough/phd_projects/online_tuner/src/rct_data_collector")

import glob
import re
import json
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rct_collector.trial_runner import TrialRunner
from rct_collector.scripts.pose_sampler import PoseSampler
from rct_collector.scripts.param_space import ARM_CONFIGS, ARM_JOINT_NAMES
from rct_collector.scripts.param_applier import ParamApplier
from rct_collector.environment_risk_node import OptimizedRiskStateNode

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_static_baseline")


def move_arm_to_pose(runner: TrialRunner, pose_label: str, global_envelope: str = "carry") -> bool:
    """Move TIAGo's arm physically to joint targets for pose_label and set local/global costmap footprints."""
    if pose_label not in ARM_CONFIGS:
        logger.error(f"Arm config '{pose_label}' is not defined in ARM_CONFIGS!")
        return False

    cfg = ARM_CONFIGS[pose_label]
    joints = cfg.get("joints")
    footprint = cfg.get("footprint")
    if not joints:
        logger.error(f"No joint targets defined for '{pose_label}'!")
        return False

    logger.info(f"Moving physical arm to '{pose_label}' pose...")
    names = ", ".join(ARM_JOINT_NAMES)
    pos = ", ".join(str(float(v)) for v in joints)
    t = 5
    goal = (
        f"{{trajectory: {{joint_names: [{names}], "
        f"points: [{{positions: [{pos}], time_from_start: {{sec: {t}}}}}]}}}}"
    )
    cmd = [
        "bash", "-c",
        f"source /opt/ros/humble/setup.bash && ros2 action send_goal "
        f"/arm_controller/follow_joint_trajectory "
        f"control_msgs/action/FollowJointTrajectory '{goal}'"
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=30.0)
        time.sleep(5.0)
    except subprocess.TimeoutExpired:
        logger.error("Arm motion action timed out!")
        return False

    if footprint:
        global_fp = ARM_CONFIGS[global_envelope]["footprint"]
        for node, fp in (("/local_costmap/local_costmap", footprint),
                         ("/global_costmap/global_costmap", global_fp)):
            cmd = ["bash", "-c",
                   f"source /opt/ros/humble/setup.bash && ros2 param set {node} footprint '{fp}'"]
            success = False
            for attempt in range(2):
                try:
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=10.0)
                    if r.returncode == 0:
                        success = True
                        break
                    else:
                        logger.warning(f"Attempt {attempt+1}: Param set returned non-zero for {node}: {r.stderr}")
                except subprocess.TimeoutExpired:
                    logger.warning(f"Attempt {attempt+1}: Param set timed out for {node} after 10.0s")
                time.sleep(0.5)
            if not success:
                logger.error(f"FAILED to set {node} footprint parameter!")
                return False
        logger.info(f"Footprints set: local='{pose_label}', global='{global_envelope}' ✓")

    return True


def apply_all_params(params: dict):
    """Apply static configuration parameters to active Nav2 nodes via ParamApplier."""
    logger.info("Applying parameters to Nav2 nodes via ROS 2 parameter services...")
    param_node = Node("baseline_param_applier")
    applier = ParamApplier(param_node)

    try:
        for node_name, node_params in params.items():
            for param_name, value in node_params.items():
                if node_name == "local_costmap":
                    target_nodes = ["local_costmap/local_costmap"]
                elif node_name == "global_costmap":
                    target_nodes = ["global_costmap/global_costmap"]
                else:
                    target_nodes = [node_name]

                param_type = "continuous"
                if param_name == "FollowPath.time_steps":
                    param_type = "discrete"
                elif param_name == "footprint":
                    param_type = "footprint"
                    value = ARM_CONFIGS.get(value, {}).get("footprint", value)
                elif param_name == "inflation_layer.inflation_radius":
                    param_type = "continuous"
                    target_nodes = ["local_costmap/local_costmap", "global_costmap/global_costmap"]

                for rn in target_nodes:
                    logger.info(f"Setting parameter {rn}.{param_name} = {value}...")
                    out = applier.set_and_verify(rn, param_name, value, param_type)
                    if not out.ok:
                        logger.error(f"Failed to apply {rn}.{param_name}: {out.outcome} ({out.detail})")
                    else:
                        logger.info(f"Successfully set {rn}.{param_name} to {value} ✓")
    finally:
        param_node.destroy_node()


def completed_episodes(output_dir: str, strategy: str, output_csv: str = "") -> set:
    """Episode ids already completed on disk or CSV for this strategy, so a restart resumes cleanly."""
    done = set()
    strat_clean = strategy.lower().replace(" ", "").replace("_", "").replace("-", "").replace("(", "").replace(")", "")

    search_dirs = [output_dir, os.path.join(output_dir, "trials"), os.path.join(output_dir, "test_trials", "trials")]
    json_files = []
    for d in search_dirs:
        if os.path.exists(d):
            json_files.extend(glob.glob(os.path.join(d, "*.json")))
            json_files.extend(glob.glob(os.path.join(d, "**", "*.json"), recursive=True))

    for fpath in set(json_files):
        fname = os.path.basename(fpath)
        fname_clean = fname.lower().replace(" ", "").replace("_", "").replace("-", "").replace("(", "").replace(")", "")

        if strat_clean in fname_clean or fname_clean.startswith("trial_ep") or fname_clean.startswith("ep"):
            m = re.search(r"ep(\d+)", fname, re.IGNORECASE)
            if m:
                done.add(int(m.group(1)))
            else:
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                    ep = data.get("episode_id") or data.get("ep_id")
                    if ep is not None:
                        done.add(int(ep))
                except Exception:
                    pass

    csv_paths = [output_csv] if output_csv else []
    csv_paths.extend([
        os.path.join(output_dir, "results.csv"),
        os.path.join(output_dir, f"{strategy.lower().replace(' ', '_')}_results.csv"),
    ])
    for cp in csv_paths:
        if cp and os.path.exists(cp):
            try:
                df = pd.read_csv(cp)
                if "episode_id" in df.columns:
                    if "strategy" in df.columns:
                        df_sub = df[df["strategy"].astype(str).str.lower().str.replace(" ", "").str.replace("_", "").str.replace("-", "").str.replace("(", "").str.replace(")", "") == strat_clean]
                        done.update(df_sub["episode_id"].astype(int).tolist())
                    else:
                        done.update(df["episode_id"].astype(int).tolist())
            except Exception:
                pass

    return done


def run_static_baseline(strategy_name: str, config_yaml: str, num_episodes: int, map_yaml: str, output_csv: str, arm_pose: str = "", timeout_sec: float = 240.0):
    logger.info(f"Preparing Static Baseline Evaluation for '{strategy_name}' ({num_episodes} episodes)...")

    params = {}
    if os.path.exists(config_yaml):
        with open(config_yaml, "r") as f:
            params = yaml.safe_load(f)
        logger.info(f"Loaded static parameters from {config_yaml}: {params}")
    else:
        logger.info(f"No YAML provided for {strategy_name}; using default parameters.")

    if arm_pose:
        params.setdefault("local_costmap", {})["footprint"] = arm_pose
        params.setdefault("global_costmap", {})["footprint"] = arm_pose

    if not output_csv:
        strat_clean = strategy_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        pose_suffix = f"_{arm_pose}" if arm_pose else ""
        output_csv = os.path.join("campaign-b-data", f"{strat_clean}{pose_suffix}_results.csv")

    output_dir = os.path.dirname(output_csv) or "campaign-b-data"
    os.makedirs(output_dir, exist_ok=True)

    if not rclpy.ok():
        rclpy.init()

    # Instantiate passive OptimizedRiskStateNode in background for risk feature logging
    risk_node = OptimizedRiskStateNode()
    executor = MultiThreadedExecutor()
    executor.add_node(risk_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    logger.info(f"Connecting TrialRunner to active ROS 2 Gazebo session (map: {map_yaml})...")
    runner = TrialRunner(
        output_dir=output_dir,
        map_yaml_path=map_yaml,
        gt_min_rate_hz=0.0,
        collect_risk_features=True,
        timeout_sec=timeout_sec
    )

    try:
        if arm_pose:
            move_arm_to_pose(runner, arm_pose, global_envelope=arm_pose)

        apply_all_params(params)

        pose_sampler = PoseSampler(map_yaml_path=map_yaml, seed=42)
        pose_sampler.load_map()

        done_episodes = completed_episodes(output_dir, strategy_name, output_csv)
        if done_episodes:
            logger.info(f"Resuming run: {len(done_episodes)} episodes already completed on disk for '{strategy_name}'.")

        results = []

        for ep_id in range(1, num_episodes + 1):
            if ep_id in done_episodes:
                logger.info(f"Skipping episode {ep_id}/{num_episodes} (already completed on disk/CSV)")
                continue
            logger.info(f"\n--- Episode {ep_id}/{num_episodes} | Strategy: '{strategy_name}' ---")
            start_pose, goal_pose = pose_sampler.sample_start_goal()

            trial_id = f"ep{ep_id}_{strategy_name.replace(' ', '_')}"
            res = runner.run_trial(
                trial_id=trial_id,
                start_pose=start_pose,
                goal_pose=goal_pose,
                params=params
            )

            st = str(res.status)
            reached = 1 if st == "SUCCESS" else 0
            coll = 1 if st == "COLLISION" else 0
            plan_fail = 1 if st == "PLANNING_FAILED" else 0
            stuck = 1 if st in ("STUCK", "FAILED", "TIMEOUT") else 0
            pose_ok = 1 if res.success_true else 0
            time_s = res.travel_time_sec or 0.0

            logger.info(
                f"Outcome: {st} | PoseOK: {bool(pose_ok)} | "
                f"Travel Time: {time_s:.1f}s | Path Length: {res.path_length_m:.2f}m")

            row_dict = {
                "episode_id": ep_id,
                "strategy": strategy_name,
                "arm_pose": arm_pose,
                "reached": reached,
                "pose_ok": pose_ok,
                "collision": coll,
                "plan_fail": plan_fail,
                "stuck": stuck,
                "travel_time_s": round(float(time_s), 2),
                "status": res.status,
                "failure_reason": res.failure_reason
            }
            for n_name, n_params in params.items():
                for p_name, p_val in n_params.items():
                    row_dict[f"param__{n_name}__{p_name}"] = str(p_val)

            results.append(row_dict)

        df_out = pd.DataFrame(results)
        if os.path.exists(output_csv):
            df_old = pd.read_csv(output_csv)
            df_out = pd.concat([df_old, df_out], ignore_index=True)

        df_out.to_csv(output_csv, index=False)
        logger.info(f"Successfully recorded results for strategy '{strategy_name}' to {output_csv} ✓")
    finally:
        runner.shutdown()
        executor.shutdown()


def main():
    default_map = "/home/forough/phd_projects/online_tuner/src/gazebo_simulation/maps/causal_benchmark.yaml"

    parser = argparse.ArgumentParser(description="Run Static Baseline Navigation Trials")
    parser.add_argument("--strategy", type=str, required=True, help="Strategy name (e.g. 'CURE (Hossen 2025)', 'MOBO Auto-Tuning')")
    parser.add_argument("--config", type=str, default="", help="Path to static baseline YAML config")
    parser.add_argument("--num-episodes", type=int, default=30, help="Number of episodes")
    parser.add_argument("--map-yaml", type=str, default=default_map, help="Path to map.yaml")
    parser.add_argument("--results-csv", type=str, default="", help="Output CSV path")
    parser.add_argument("--arm-pose", type=str, default="", choices=["", "tucked", "carry"], help="Set arm pose for static baseline")
    parser.add_argument("--timeout-sec", type=float, default=240.0, help="Maximum trial duration in seconds (default: 240.0)")
    args = parser.parse_args()

    run_static_baseline(args.strategy, args.config, args.num_episodes, args.map_yaml, args.results_csv, args.arm_pose, args.timeout_sec)


if __name__ == "__main__":
    main()
