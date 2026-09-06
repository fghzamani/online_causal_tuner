#!/usr/bin/env python3
"""
Campaign B Master Benchmark Runner.

Executes closed-loop evaluation trials across the benchmark episodes
(from pose_pool_seed42_causal_benchmark.json) in the updated causal_benchmark.world
environment (which includes the unmapped dynamic obstacle box_middle_east at x=1.01, y=3.55).

Supports running a single strategy or all strategy arms in a loop:
  1. "Nav2 Default"
  2. "Static Best-Fixed (Carry)"
  3. "Static Best-Fixed (Tucked)"
  4. "MOBO (Tucked)"
  5. "MOBO (Carry)"
  6. "CURE (Tucked)"
  7. "CURE (Carry)"
  8. "Envelope-Only" (Ablation Baseline)
  9. "Online Causal (Ours)"

Results accumulate into:
  campaign-b-data/campaign_b_evaluation_results.csv
"""

import os
import sys
import time
import math
import json
import glob
import re
import argparse
import logging
import threading
import subprocess
import yaml
import pandas as pd

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

sys.path.insert(0, "/home/forough/phd_projects/online_tuner/src")
sys.path.insert(0, "/home/forough/phd_projects/online_tuner/src/rct_data_collector")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from online_causal_tuner.online_tuner_node import (
    OnlineCausalTunerNode,
    ARM_CONFIGS,
    ARM_JOINT_NAMES,
)
from rct_collector.trial_runner import TrialRunner
from rct_collector.scripts.pose_sampler import PoseSampler
from rct_collector.environment_risk_node import OptimizedRiskStateNode

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_campaign_b_benchmark")

# ── Baseline Parameter Maps ──────────────────────────────────────────────────
NAV2_DEFAULT_CONFIG = {
    "controller_server": {
        "FollowPath.vx_max": 0.55,
        "FollowPath.wz_max": 1.0,
        "FollowPath.CostCritic.cost_weight": 3.81,
        "FollowPath.time_steps": 56,
    },
    "local_costmap": {
        "inflation_layer.inflation_radius": 0.55,
    },
}

STATIC_BEST_FIXED_CONFIG = {
    "controller_server": {
        "FollowPath.vx_max": 0.45,
        "FollowPath.wz_max": 1.0,
        "FollowPath.CostCritic.cost_weight": 3.0,
        "FollowPath.time_steps": 50,
    },
    "local_costmap": {
        "inflation_layer.inflation_radius": 0.30,
    },
}

ALL_STRATEGIES_MAP = {
    "Nav2 Default": {
        "config_yaml": "",
        "arm_pose": "carry",
    },
    "Static Best-Fixed (Carry)": {
        "config_yaml": "",
        "arm_pose": "carry",
    },
    "Static Best-Fixed (Tucked)": {
        "config_yaml": "",
        "arm_pose": "tucked",
    },
    "MOBO (Carry)": {
        "config_yaml": "/home/forough/phd_projects/online_tuner/baselines/cure/mobo_config_carry.yaml",
        "arm_pose": "carry",
    },
    "MOBO (Tucked)": {
        "config_yaml": "/home/forough/phd_projects/online_tuner/baselines/cure/mobo_config_tucked.yaml",
        "arm_pose": "tucked",
    },
    "CURE (Carry)": {
        "config_yaml": "/home/forough/phd_projects/online_tuner/baselines/cure/cure_config_carry.yaml",
        "arm_pose": "carry",
    },
    "CURE (Tucked)": {
        "config_yaml": "/home/forough/phd_projects/online_tuner/baselines/cure/cure_config_tucked.yaml",
        "arm_pose": "tucked",
    },
    "Envelope-Only": {
        "config_yaml": "",
        "arm_pose": "carry",
    },
    "Online Causal (Ours)": {
        "config_yaml": "",
        "arm_pose": "carry",
    },
}


def move_arm_to_pose(runner: TrialRunner, pose_label: str, global_envelope: str = "tucked") -> bool:
    """Move physical arm asynchronously via ROS 2 action server and set local/global costmap footprints."""
    cfg = ARM_CONFIGS.get(pose_label)
    if not cfg:
        logger.error(f"Unknown arm pose label '{pose_label}'")
        return False

    footprint = cfg.get("footprint")
    joints = cfg.get("joints")

    # 1. Trigger physical arm joint trajectory
    if joints:
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
            time.sleep(4.0)
        except subprocess.TimeoutExpired:
            logger.error("Arm motion action timed out!")

    # P0.3: Set local footprint = pose_label, global footprint = global_envelope per strategy protocol
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


def completed_episodes(output_dir: str, strategy_tag: str) -> set:
    """P0.5: Episode ids already on disk for this strategy, so a restart resumes cleanly."""
    done = set()
    for f in glob.glob(os.path.join(output_dir, f"trial_*_{strategy_tag}.json")):
        m = re.search(r"_ep(\d+)_", os.path.basename(f))
        if m:
            done.add(int(m.group(1)))
    return done


def load_pose_pool(pose_pool_json: str, map_yaml: str, seed: int = 42) -> list:
    """Load pre-generated seed=42 pose pool or fallback to PoseSampler."""
    if os.path.exists(pose_pool_json):
        with open(pose_pool_json) as f:
            poses = json.load(f)
        logger.info(f"Loaded {len(poses)} paired seed={seed} poses from {pose_pool_json} ✓")
        return poses

    logger.warning(f"Pose pool JSON not found at {pose_pool_json}; generating on the fly...")
    sampler = PoseSampler(map_yaml_path=map_yaml, obstacle_clearance_m=0.45, seed=seed)
    sampler.load_map()
    poses = []
    for ep in range(1, 51):
        s, g = sampler.sample_start_goal()
        poses.append({"episode_id": ep, "start_pose": s, "goal_pose": g})
    return poses


def run_benchmark_strategy(
    strategy: str,
    num_episodes: int,
    map_yaml: str,
    pose_pool_json: str,
    master_csv: str,
    model_path: str,
    arm_pose: str = "carry",
    config_yaml: str = "",
    timeout_sec: float = 240.0
):
    """Run benchmark navigation trials for a specific strategy and append to master evaluation CSV."""
    logger.info(f"\n==========================================================================")
    logger.info(f" RUNNING CAMPAIGN B BENCHMARK: Strategy = '{strategy}' ({num_episodes} episodes)")
    logger.info(f"==========================================================================")

    output_dir = os.path.dirname(master_csv) or "campaign-b-data"
    os.makedirs(output_dir, exist_ok=True)
    pose_pool = load_pose_pool(pose_pool_json, map_yaml)

    if not rclpy.ok():
        rclpy.init()

    # Determine baseline configuration params
    init_params = {}
    if config_yaml and os.path.exists(config_yaml):
        with open(config_yaml) as f:
            init_params = yaml.safe_load(f)
        logger.info(f"Loaded custom strategy configuration from {config_yaml} ✓")
    elif strategy == "Nav2 Default":
        init_params = NAV2_DEFAULT_CONFIG.copy()
    elif "Static Best-Fixed" in strategy:
        init_params = STATIC_BEST_FIXED_CONFIG.copy()

    # P1.1: Always ensure OptimizedRiskStateNode is running as a passive observer for all strategy arms
    check_node = Node("benchmark_node_checker")
    active_nodes = check_node.get_node_names()
    check_node.destroy_node()

    nodes_to_add = []
    risk_node = None
    if not any("risk_state" in n for n in active_nodes):
        logger.info("Instantiating OptimizedRiskStateNode in background thread...")
        risk_node = OptimizedRiskStateNode()
        nodes_to_add.append(risk_node)

    tuner_node = None
    if strategy in ("Online Causal (Ours)", "Envelope-Only"):
        if not any("online_causal_tuner" in n for n in active_nodes):
            abs_model_path = os.path.abspath(model_path)
            logger.info(f"Instantiating OnlineCausalTunerNode with models from {abs_model_path}...")
            tuner_node = OnlineCausalTunerNode()
            tuner_node.model_path = abs_model_path
            tuner_node.selection_mode = "enumerate"
            if strategy == "Envelope-Only":
                tuner_node.envelope_only = True
                logger.info("Configured OnlineCausalTunerNode for ENVELOPE-ONLY ablation mode ✓")
            tuner_node._load_models()
            nodes_to_add.append(tuner_node)

    executor = None
    if nodes_to_add:
        executor = MultiThreadedExecutor()
        for n in nodes_to_add:
            executor.add_node(n)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

    # P1.1: Set collect_risk_features=True for all strategy arms
    runner = TrialRunner(
        output_dir=output_dir,
        map_yaml_path=map_yaml,
        gt_min_rate_hz=0.0,
        collect_risk_features=True,
        timeout_sec=timeout_sec
    )
    runner.tuner_node = tuner_node

    # P0.5: Resume support: check completed episodes on disk
    strategy_tag = strategy.replace(' ', '_')
    done_episodes = completed_episodes(output_dir, strategy_tag)
    if done_episodes:
        logger.info(f"Resuming run: {len(done_episodes)} episodes already on disk for '{strategy}'.")

    # Global costmap envelope policy per strategy
    global_env = "tucked" if strategy in ("Online Causal (Ours)", "Envelope-Only") else ("carry" if "Carry" in strategy else "tucked")

    results = []

    try:
        for ep_info in pose_pool[:num_episodes]:
            ep_id = ep_info["episode_id"]
            start_pose = ep_info["start_pose"]
            goal_pose = ep_info["goal_pose"]

            if ep_id in done_episodes:
                logger.info(f"Skipping episode {ep_id}/{num_episodes} (already completed on disk)")
                continue

            logger.info(f"\n--- Episode {ep_id}/{num_episodes} | Strategy: '{strategy}' ---")
            logger.info(f"Start: ({start_pose['x']:.2f}, {start_pose['y']:.2f}) -> Goal: ({goal_pose['x']:.2f}, {goal_pose['y']:.2f})")

            # 1. Teleport robot to start pose FIRST
            runner._teleport_robot(start_pose)
            runner._publish_initial_pose(start_pose)
            time.sleep(1.0)

            # 2. Set initial physical arm pose & costmap footprints (local + global)
            current_arm = arm_pose if arm_pose else ("carry" if "Carry" in strategy else "tucked")
            move_arm_to_pose(runner, current_arm, global_envelope=global_env)
            if tuner_node:
                tuner_node.current_arm_label = current_arm
                tuner_node.reset_trial_metrics()
                tuner_node.set_goal(goal_pose)

            # 3. Run trial navigation
            trial_id = f"ep{ep_id}_{strategy_tag}"
            res = runner.run_trial(
                trial_id=trial_id,
                start_pose=start_pose,
                goal_pose=goal_pose,
                params=init_params
            )

            # P1.2: Write decision log and parameter set failures into trial JSON
            if tuner_node is not None:
                trial_json = os.path.join(output_dir, f"trial_{res.trial_id}.json")
                if os.path.exists(trial_json):
                    try:
                        with open(trial_json, "r") as f:
                            blob = json.load(f)
                        blob["tuner_decision_log"] = tuner_node.decision_log
                        blob["n_param_set_failures"] = tuner_node.n_param_set_failures
                        blob["param_set_failures"] = tuner_node.param_set_failures
                        blob["global_envelope"] = global_env
                        blob["global_inflation_radius"] = 0.55
                        blob["n_stale_ticks"] = tuner_node.n_stale_ticks
                        blob["n_tick_overruns"] = tuner_node.n_tick_overruns
                        blob["max_tick_ms"] = round(float(tuner_node.max_tick_ms), 1)
                        blob["n_arm_retries"] = tuner_node.n_arm_retries
                        with open(trial_json, "w") as f:
                            json.dump(blob, f, indent=2)
                    except Exception as e:
                        logger.error(f"Error writing tuner decision log to trial JSON: {e}")

            # Outcomes partition on `status`, which is mutually exclusive and
            # sums to 100%. `pose_ok` is a separate QUALITY metric, not a
            # partition member: a trial can be reached=1, pose_ok=0.
            st = str(res.status)
            reached = 1 if st == "SUCCESS" else 0
            coll = 1 if st == "COLLISION" else 0
            plan_fail = 1 if st == "PLANNING_FAILED" else 0
            stuck = 1 if st in ("STUCK", "FAILED", "TIMEOUT") else 0
            pose_ok = 1 if res.success_true else 0
            time_s = res.travel_time_sec or 0.0
            path_len = res.path_length_m or 0.0

            if tuner_node:
                carry_frac = tuner_node.get_carry_fraction()
                carry_dist = path_len * carry_frac
                n_cfg_sw = tuner_node.n_config_switches
                n_arm_sw = tuner_node.n_arm_switches
                out_of_supp_frac = tuner_node.get_out_of_support_fraction()
            else:
                carry_frac = float("nan")
                carry_dist = float("nan")
                n_cfg_sw = 0
                n_arm_sw = 0
                out_of_supp_frac = 0.0

            logger.info(
                f"Outcome: {st} | PoseOK: {bool(pose_ok)} | "
                f"Travel Time: {time_s:.1f}s | Path Length: {path_len:.2f}m")

            row_dict = {
                "episode_id": ep_id,
                "strategy": strategy,
                "arm_pose": current_arm,
                "reached": reached,
                "pose_ok": pose_ok,
                "collision": coll,
                "plan_fail": plan_fail,
                "stuck": stuck,
                "travel_time_s": round(float(time_s), 2),
                "path_length_m": round(float(path_len), 2),
                "carry_distance_m": round(float(carry_dist), 2) if not math.isnan(carry_dist) else float("nan"),
                "carry_fraction": round(float(carry_frac), 4) if not math.isnan(carry_frac) else float("nan"),
                "n_config_switches": n_cfg_sw,
                "n_arm_switches": n_arm_sw,
                "out_of_support_fraction": round(float(out_of_supp_frac), 4),
                "status": res.status,
                "failure_reason": res.failure_reason
            }
            results.append(row_dict)

        # Consolidate results into master CSV file
        if results:
            df_new = pd.DataFrame(results)

            if os.path.exists(master_csv):
                df_existing = pd.read_csv(master_csv)
                df_existing = df_existing[df_existing["strategy"] != strategy]
                df_final = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_final = df_new

            df_final.to_csv(master_csv, index=False)
            logger.info(f"\nSuccessfully updated master evaluation CSV: {master_csv} ✓")

    finally:
        runner.shutdown()
        if executor:
            executor.shutdown()
        if tuner_node:
            tuner_node.destroy_node()


def main():
    default_map = "/home/forough/phd_projects/online_tuner/src/gazebo_simulation/maps/causal_benchmark.yaml"
    default_pool = "/home/forough/phd_projects/online_tuner/src/gazebo_simulation/maps/pose_pool_seed42_causal_benchmark.json"
    default_master_csv = "/home/forough/phd_projects/online_tuner/src/online_causal_tuner/evaluation_results/campaign_b_evaluation_results.csv"
    default_model = "/home/forough/phd_projects/online_tuner/src/online_causal_tuner/models/causal_tuner_models.pkl"

    parser = argparse.ArgumentParser(description="Campaign B Master Benchmark Runner")
    parser.add_argument("--strategy", type=str, default="all", choices=[
        "all",
        "Nav2 Default",
        "Static Best-Fixed (Carry)",
        "Static Best-Fixed (Tucked)",
        "MOBO (Tucked)",
        "MOBO (Carry)",
        "CURE (Tucked)",
        "CURE (Carry)",
        "Envelope-Only",
        "Online Causal (Ours)"
    ], help="Strategy name to run, or 'all' to loop through all strategies")
    parser.add_argument("--num-episodes", type=int, default=30, help="Number of episodes per strategy (default: 30)")
    parser.add_argument("--map-yaml", type=str, default=default_map, help="Path to map.yaml")
    parser.add_argument("--pose-pool", type=str, default=default_pool, help="Path to seed42 pose pool JSON")
    parser.add_argument("--master-csv", type=str, default=default_master_csv, help="Master output CSV file")
    parser.add_argument("--model-path", type=str, default=default_model, help="Path to causal_tuner_models.pkl")
    parser.add_argument("--config-yaml", type=str, default="", help="Optional YAML config path override")
    parser.add_argument("--arm-pose", type=str, default="carry", choices=["carry", "tucked"], help="Initial arm pose")
    parser.add_argument("--timeout-sec", type=float, default=240.0, help="Trial timeout in seconds")
    args = parser.parse_args()

    if args.strategy == "all":
        logger.info(f"Looping through ALL {len(ALL_STRATEGIES_MAP)} benchmark strategy arms ({args.num_episodes} episodes each)...")
        for strat_name, strat_info in ALL_STRATEGIES_MAP.items():
            cfg = args.config_yaml if args.config_yaml else strat_info.get("config_yaml", "")
            arm = args.arm_pose if args.arm_pose != "carry" else strat_info.get("arm_pose", "carry")
            run_benchmark_strategy(
                strategy=strat_name,
                num_episodes=args.num_episodes,
                map_yaml=args.map_yaml,
                pose_pool_json=args.pose_pool,
                master_csv=args.master_csv,
                model_path=args.model_path,
                arm_pose=arm,
                config_yaml=cfg,
                timeout_sec=args.timeout_sec
            )

        logger.info("\nAll strategy evaluations completed! Evaluating master results...")
        try:
            from online_causal_tuner.analysis.evaluate_campaign_b_closed_loop import evaluate_campaign_b_closed_loop
        except ImportError:
            from evaluate_campaign_b_closed_loop import evaluate_campaign_b_closed_loop
        paper_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper_artifacts")
        evaluate_campaign_b_closed_loop(args.master_csv, paper_dir)
    else:
        run_benchmark_strategy(
            strategy=args.strategy,
            num_episodes=args.num_episodes,
            map_yaml=args.map_yaml,
            pose_pool_json=args.pose_pool,
            master_csv=args.master_csv,
            model_path=args.model_path,
            arm_pose=args.arm_pose,
            config_yaml=args.config_yaml,
            timeout_sec=args.timeout_sec
        )


if __name__ == "__main__":
    main()
