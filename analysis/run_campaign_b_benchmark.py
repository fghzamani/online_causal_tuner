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
from typing import Optional, List, Dict

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

CURE_TUCKED_YAML = "/home/forough/phd_projects/online_tuner/baselines/cure/cure_config_tucked.yaml"
CURE_CARRY_YAML = "/home/forough/phd_projects/online_tuner/baselines/cure/cure_config_carry.yaml"

ALL_STRATEGIES_MAP = {
    "Nav2 Default": {
        "config_yaml": "",
        "arm_pose": "carry",
        "global_envelope": "carry",
        "adaptive": False,
        "init_params": NAV2_DEFAULT_CONFIG,
    },
    "CURE (Carry)": {
        "config_yaml": CURE_CARRY_YAML if os.path.exists(CURE_CARRY_YAML) else "",
        "arm_pose": "carry",
        "global_envelope": "carry",
        "adaptive": False,
    },
    "CURE (Tucked)": {
        "config_yaml": CURE_TUCKED_YAML if os.path.exists(CURE_TUCKED_YAML) else "",
        "arm_pose": "tucked",
        "global_envelope": "tucked",
        "adaptive": False,
    },
    # "Static Best-Fixed (Carry)": {
    #     "config_yaml": "",
    #     "arm_pose": "carry",
    #     "global_envelope": "carry",
    #     "adaptive": False,
    #     "init_params": STATIC_BEST_FIXED_CONFIG,
    # },
    # "Static Best-Fixed (Tucked)": {
    #     "config_yaml": "",
    #     "arm_pose": "tucked",
    #     "global_envelope": "tucked",
    #     "adaptive": False,
    #     "init_params": STATIC_BEST_FIXED_CONFIG,
    # },
    "Retract-Always": {
        "config_yaml": "",
        "arm_pose": "carry",
        "global_envelope": "tucked",
        "adaptive": False,
        "init_params": NAV2_DEFAULT_CONFIG,
    },
    "Envelope-Only": {
        "config_yaml": "",
        "arm_pose": "carry",
        "global_envelope": "tucked",
        "adaptive": True,
        "init_params": NAV2_DEFAULT_CONFIG,
    },
    "Online Causal (Ours)": {
        "config_yaml": "",
        "arm_pose": "carry",
        "global_envelope": "tucked",
        "adaptive": True,
        "init_params": NAV2_DEFAULT_CONFIG,
    },
}


ARM_JOINT_TOLERANCE_RAD = 0.15


def read_verified_arm_label(runner, timeout_s: float = 2.0):
    """Arm label confirmed from /joint_states, or None if in transit."""
    import rclpy
    from sensor_msgs.msg import JointState
    positions = {}

    def _cb(msg):
        for n, p in zip(msg.name, msg.position):
            positions[n] = p

    node = None
    if hasattr(runner, "_recorder") and runner._recorder is not None:
        node = runner._recorder
    elif hasattr(runner, "node") and runner.node is not None:
        node = runner.node
    elif hasattr(runner, "tuner_node") and runner.tuner_node is not None:
        node = runner.tuner_node
    else:
        return None

    sub = node.create_subscription(JointState, "/joint_states", _cb, 10)
    t0 = time.time()
    while time.time() - t0 < timeout_s and not positions:
        time.sleep(0.05)
    node.destroy_subscription(sub)
    if not positions:
        return None
    for label, cfg in ARM_CONFIGS.items():
        err = max(abs(positions.get(n, 1e3) - t)
                  for n, t in zip(ARM_JOINT_NAMES, cfg["joints"]))
        if err <= ARM_JOINT_TOLERANCE_RAD:
            return label
    return None


def read_back_params(runner=None) -> dict:
    """Read back live Nav2 controller & costmap parameters from running ROS nodes."""
    target_params = {
        "controller_server": [
            "FollowPath.vx_max",
            "FollowPath.vx_std",
            "FollowPath.wz_max",
            "FollowPath.CostCritic.cost_weight",
            "FollowPath.ConstraintCritic.cost_weight",
            "FollowPath.PathAlignCritic.cost_weight",
        ],
        "local_costmap": [
            "inflation_layer.inflation_radius",
        ]
    }
    readback = {}
    for node_name, pnames in target_params.items():
        node_dict = {}
        ros_node_path = "/local_costmap/local_costmap" if node_name == "local_costmap" else f"/{node_name}"
        for pname in pnames:
            try:
                out = subprocess.check_output(
                    ["ros2", "param", "get", ros_node_path, pname],
                    stderr=subprocess.DEVNULL, timeout=2.0
                ).decode("utf-8")
                if "value is:" in out:
                    val_str = out.split("value is:")[-1].strip()
                    try:
                        node_dict[pname] = float(val_str)
                    except ValueError:
                        node_dict[pname] = val_str
            except Exception:
                pass
        if node_dict:
            readback[node_name] = node_dict
    return readback


def get_ros_setup_prefix() -> str:
    cmds = []
    if os.path.exists("/opt/ros/humble/setup.bash"):
        cmds.append("source /opt/ros/humble/setup.bash")
    for ws_path in ["/home/forough/phd_projects/online_tuner/install/setup.bash",
                   "/home/forough/phd_projects/online_tuner/src/install/setup.bash"]:
        if os.path.exists(ws_path):
            cmds.append(f"source {ws_path}")
            break
    return " && ".join(cmds) + " && " if cmds else ""


def move_arm_to_pose(runner: TrialRunner, pose_label: str, global_envelope: Optional[str] = "tucked", max_retries: int = 3) -> bool:
    """Move physical arm via ROS 2 action server and verify /joint_states reaches pose_label."""
    cfg = ARM_CONFIGS.get(pose_label)
    if not cfg:
        logger.error(f"Unknown arm pose label '{pose_label}'")
        return False

    footprint = cfg.get("footprint")
    joints = cfg.get("joints")

    src_prefix = get_ros_setup_prefix()

    # 1. Check if arm is already in requested pose
    current = read_verified_arm_label(runner, timeout_s=1.0)
    if current == pose_label:
        logger.info(f"Arm already verified in '{pose_label}' pose ✓")
    else:
        # 2. Trigger physical arm joint trajectory (or play_motion2) and verify arrival
        for attempt in range(1, max_retries + 1):
            logger.info(f"Commanding arm -> '{pose_label}' pose (attempt {attempt}/{max_retries})...")
            
            # If tucked, try play_motion2 'home' motion first
            if pose_label == "tucked" and attempt == 1:
                pm_cmd = [
                    "bash", "-c",
                    f"{src_prefix}ros2 action send_goal "
                    f"/play_motion2 play_motion2_msgs/action/PlayMotion2 "
                    f"\"{{motion_name: home, skip_planning: false}}\""
                ]
                try:
                    subprocess.run(pm_cmd, capture_output=True, text=True, timeout=12.0)
                except subprocess.TimeoutExpired:
                    logger.warning("play_motion2 'home' motion timed out, falling back to joint trajectory")

            # FollowJointTrajectory goal fallback / primary
            if joints:
                names = ", ".join(ARM_JOINT_NAMES)
                pos = ", ".join(str(float(v)) for v in joints)
                t = 3
                goal = (
                    f"{{trajectory: {{joint_names: [{names}], "
                    f"points: [{{positions: [{pos}], time_from_start: {{sec: {t}}}}}]}}}}"
                )
                cmd = [
                    "bash", "-c",
                    f"{src_prefix}ros2 action send_goal "
                    f"/arm_controller/follow_joint_trajectory "
                    f"control_msgs/action/FollowJointTrajectory '{goal}'"
                ]
                try:
                    subprocess.run(cmd, capture_output=True, text=True, timeout=15.0)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Arm motion action timed out (attempt {attempt}/{max_retries})")

            # Poll joint_states verification until arm arrives
            t_verify_end = time.time() + 4.0
            verified_label = None
            while time.time() < t_verify_end:
                verified_label = read_verified_arm_label(runner, timeout_s=0.2)
                if verified_label == pose_label:
                    break
                time.sleep(0.1)

            if verified_label == pose_label:
                logger.info(f"Arm verified in '{pose_label}' pose on attempt {attempt} ✓")
                break
            else:
                logger.warning(f"Arm not yet verified in '{pose_label}' pose after attempt {attempt}. Retrying...")

    # P0.3: Set local footprint = pose_label, global footprint = global_envelope per strategy protocol
    if footprint:
        targets = [("/local_costmap/local_costmap", footprint)]
        if global_envelope is not None and global_envelope in ARM_CONFIGS:
            targets.append(("/global_costmap/global_costmap", ARM_CONFIGS[global_envelope]["footprint"]))

        for node, fp in targets:
            cmd = ["bash", "-c",
                   f"{src_prefix}ros2 param set {node} footprint '{fp}'"]
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
        global_label = global_envelope if global_envelope is not None else "untouched (None)"
        logger.info(f"Footprints set: local='{pose_label}', global='{global_label}' ✓")

    return True


def completed_episodes(output_dir: str, strategy: str, output_csv: str = "") -> set:
    """Episode ids already completed on disk for this strategy, so restarts resume cleanly."""
    done = set()
    strat_clean = strategy.lower().replace(" ", "").replace("_", "").replace("-", "").replace("(", "").replace(")", "")

    # 1. Search trial JSON files in output_dir and subdirectories
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

    # 2. Check existing CSV files
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
    strategy: str = "Online Causal (Ours)",
    num_episodes: int = 30,
    map_yaml: str = "",
    pose_pool_json: str = "",
    master_csv: str = "",
    model_path: str = "",
    arm_pose: str = "carry",
    config_yaml: str = "",
    timeout_sec: float = 240.0,
    trials_dir: Optional[str] = None,
    overwrite: bool = False
):
    """Run benchmark navigation trials for a specific strategy and append to master evaluation CSV."""
    logger.info(f"\n==========================================================================")
    logger.info(f" RUNNING CAMPAIGN B BENCHMARK: Strategy = '{strategy}' ({num_episodes} episodes)")
    logger.info(f"==========================================================================")

    output_dir = trials_dir if trials_dir else (os.path.dirname(master_csv) or "campaign-b-data")
    os.makedirs(output_dir, exist_ok=True)

    strategy_tag = strategy.replace(' ', '_')
    done_episodes = set() if overwrite else completed_episodes(output_dir, strategy, master_csv)
    if len(done_episodes) >= num_episodes:
        logger.info(f"Strategy '{strategy}' already has all {len(done_episodes)}/{num_episodes} episodes completed on disk/CSV. Skipping strategy entirely ✓")
        return

    pose_pool = load_pose_pool(pose_pool_json, map_yaml)

    if not rclpy.ok():
        rclpy.init()

    # Determine baseline configuration params
    strat_info = ALL_STRATEGIES_MAP.get(strategy, {})
    config_yaml = config_yaml or strat_info.get("config_yaml", "")

    init_params = {}
    if config_yaml and os.path.exists(config_yaml):
        with open(config_yaml) as f:
            init_params = yaml.safe_load(f)
        logger.info(f"Loaded custom strategy configuration from {config_yaml} ✓")
    elif strat_info.get("init_params"):
        init_params = dict(strat_info["init_params"])
        logger.info(f"Using mapped baseline initial configuration for '{strategy}' ✓")
    else:
        raise ValueError(
            f"strategy '{strategy}' has no configuration; a baseline with "
            f"undefined parameters is not a baseline")

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
    done_episodes = set() if overwrite else completed_episodes(output_dir, strategy, master_csv)
    if done_episodes:
        logger.info(f"Resuming run: {len(done_episodes)} episodes already on disk for '{strategy}'.")

    # Global costmap envelope policy per strategy
    strat_info = ALL_STRATEGIES_MAP.get(strategy, {})
    global_env = strat_info.get("global_envelope", "tucked")

    results = []

    try:
        for ep_info in pose_pool[:num_episodes]:
            ep_id = int(ep_info["episode_id"])
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

            # Reset dynamic obstacle to standby pose for the new trial
            try:
                subprocess.run(
                    ["ros2", "service", "call", "/reset_dynamic_obstacle", "std_srvs/srv/Trigger", "{}"],
                    capture_output=True, text=True, timeout=2.0
                )
            except Exception:
                pass
            time.sleep(1.0)

            # 2. Set initial physical arm pose & costmap footprints (local + global)
            current_arm = strat_info.get("arm_pose", "carry")
            if strategy == "Retract-Always":
                # Retract-Always trial setup sequence:
                # 1. Teleport to start, arm commanded to carry (post-grasp state)
                move_arm_to_pose(runner, "carry", global_envelope=global_env)
                # 2. Command tucked, wait for read_verified_arm_label(...) == "tucked"
                move_arm_to_pose(runner, "tucked", global_envelope="tucked")
                for _ in range(10):
                    lbl = read_verified_arm_label(runner, timeout_s=0.5)
                    if lbl == "tucked":
                        break
                current_arm = "tucked"
            else:
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

            # P1.2 & §3.1: Write decision log & verified arm label into trial JSON
            out_of_supp_frac_mission = float("nan")
            _dl = None
            if tuner_node is not None:
                # The node keeps ticking after run_trial returns, so the
                # tail of the log is the robot parked at the goal. Those
                # are not mission decisions and must not enter any rate.
                _dl = tuner_node.decision_log
                if _dl:
                    _t_end = _dl[0]["t"] + float(res.travel_time_sec)
                    _dl = [x for x in _dl if x["t"] <= _t_end]
                out_of_supp_frac_mission = (
                    sum(bool(x.get("out_of_support")) for x in _dl) / len(_dl)
                    if _dl else float("nan")
                )

            # Final arm state at mission completion
            if _dl and len(_dl) > 0 and "arm_verified" in _dl[-1]:
                final_arm = _dl[-1]["arm_verified"]
            else:
                final_arm = read_verified_arm_label(runner)

            is_collided = getattr(res, "is_collided", False) or bool(getattr(res, "collision_links", []))
            mission_succ = int(str(res.status) == "SUCCESS" and final_arm == "carry" and not is_collided)

            trial_json = os.path.join(output_dir, f"trial_{res.trial_id}.json")
            if os.path.exists(trial_json):
                try:
                    with open(trial_json, "r") as f:
                        blob = json.load(f)
                    blob["final_arm_label"] = final_arm
                    blob["mission_success"] = mission_succ
                    blob["global_envelope"] = global_env if global_env is not None else "nav2_default"
                    if tuner_node is not None:
                        blob["selection_mode"] = "enumerate"
                        blob["tuner_decision_log"] = _dl
                        blob["n_post_mission_ticks_dropped"] = (
                            len(tuner_node.decision_log) - len(_dl) if _dl is not None else 0)
                        blob["out_of_support_fraction_mission"] = out_of_supp_frac_mission
                        blob["n_param_set_failures"] = tuner_node.n_param_set_failures
                        blob["param_set_failures"] = tuner_node.param_set_failures
                        blob["global_inflation_radius"] = 0.55
                        blob["n_stale_ticks"] = tuner_node.n_stale_ticks
                        blob["n_tick_overruns"] = tuner_node.n_tick_overruns
                        blob["max_tick_ms"] = round(float(tuner_node.max_tick_ms), 1)
                        blob["n_arm_retries"] = tuner_node.n_arm_retries
                        blob["envelope_constants"] = getattr(tuner_node, "envelope_provenance", {})
                    blob["applied_baseline_config"] = read_back_params(runner)
                    with open(trial_json, "w") as f:
                        json.dump(blob, f, indent=2)
                except Exception as e:
                    logger.error(f"Error updating trial JSON: {e}")

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
                carry_dist = getattr(tuner_node, "carry_distance_m", 0.0)
                if carry_dist <= 0.0:
                    carry_dist = path_len * carry_frac
                n_cfg_sw = tuner_node.n_config_switches
                n_arm_sw = tuner_node.n_arm_switches
                out_of_supp_frac = out_of_supp_frac_mission if not math.isnan(out_of_supp_frac_mission) else tuner_node.get_out_of_support_fraction()
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
    default_map = "/home/forough/phd_projects/online_tuner/src/gazebo_simulation/maps/causal_benchmark_3.yaml"
    default_pool = "/home/forough/phd_projects/online_tuner/src/online_causal_tuner/evaluation_results/pose_pool_30_trials_world3.json"
    default_master_csv = "/home/forough/phd_projects/online_tuner/src/online_causal_tuner/evaluation_results/causal_benchmark_3/campaign_b_world3_evaluation_results.csv"
    default_model = "/home/forough/phd_projects/online_tuner/src/online_causal_tuner/models/causal_tuner_models.pkl"

    parser = argparse.ArgumentParser(description="Campaign B Master Benchmark Runner")
    parser.add_argument("--strategy", type=str, default="all", choices=[
        "all",
        "Nav2 Default",
        "CURE (Carry)",
        "CURE (Tucked)",
        "Static Best-Fixed (Carry)",
        "Static Best-Fixed (Tucked)",
        "Retract-Always",
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
    parser.add_argument("--test", action="store_true", help="Run in test mode: outputs to test CSV and test_trials/ directory without skipping")
    parser.add_argument("--overwrite", action="store_true", help="Force re-running episodes even if completed JSONs exist on disk")
    args = parser.parse_args()

    master_csv = args.master_csv
    trials_dir = None
    overwrite = args.overwrite

    if args.test:
        if args.master_csv == default_master_csv:
            eval_base_dir = os.path.dirname(default_master_csv)
            master_csv = os.path.join(eval_base_dir, "campaign_b_world3_test_evaluation_results.csv")
            trials_dir = os.path.join(eval_base_dir, "test_trials")
        else:
            master_csv = args.master_csv
            trials_dir = os.path.dirname(args.master_csv)
        logger.info(f"[TEST MODE ACTIVE] Saving test results to: {master_csv}")
        logger.info(f"[TEST MODE ACTIVE] Saving test trial JSONs to: {trials_dir}")

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
                master_csv=master_csv,
                model_path=args.model_path,
                arm_pose=arm,
                config_yaml=cfg,
                timeout_sec=args.timeout_sec,
                trials_dir=trials_dir,
                overwrite=overwrite
            )

        logger.info("\nAll strategy evaluations completed! Evaluating master results...")
        try:
            from online_causal_tuner.analysis.evaluate_campaign_b_closed_loop import evaluate_campaign_b_closed_loop
        except ImportError:
            from evaluate_campaign_b_closed_loop import evaluate_campaign_b_closed_loop
        paper_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper_artifacts")
        evaluate_campaign_b_closed_loop(master_csv, paper_dir)
    else:
        strat_info = ALL_STRATEGIES_MAP.get(args.strategy, {})
        cfg = args.config_yaml if args.config_yaml else strat_info.get("config_yaml", "")
        arm = args.arm_pose if args.arm_pose != "carry" else strat_info.get("arm_pose", "carry")
        run_benchmark_strategy(
            strategy=args.strategy,
            num_episodes=args.num_episodes,
            map_yaml=args.map_yaml,
            pose_pool_json=args.pose_pool,
            master_csv=master_csv,
            model_path=args.model_path,
            arm_pose=arm,
            config_yaml=cfg,
            timeout_sec=args.timeout_sec,
            trials_dir=trials_dir,
            overwrite=overwrite
        )


if __name__ == "__main__":
    main()
