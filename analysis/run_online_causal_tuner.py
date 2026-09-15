#!/usr/bin/env python3
"""
Run Online Causal Tuner (Ours) 50 Navigation Trials in Gazebo.

Uses PoseSampler with seed=42 to guarantee the exact same 50 start/goal pose pairs
used by the CURE baseline, ensuring paired evaluation.
"""

import os
import sys
import time
import math
import glob
import re
import json
import yaml
import argparse
import logging
import threading
import subprocess
import pandas as pd

import rclpy
from rclpy.executors import MultiThreadedExecutor

sys.path.insert(0, "/home/forough/phd_projects/online_tuner/src/rct_data_collector")
sys.path.insert(0, "/home/forough/phd_projects/online_tuner/src/online_causal_tuner")

from rct_collector.trial_runner import TrialRunner
from rct_collector.scripts.pose_sampler import PoseSampler
from rct_collector.scripts.param_space import ARM_CONFIGS, ARM_JOINT_NAMES
from online_causal_tuner.online_tuner_node import OnlineCausalTunerNode

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_online_causal_tuner")


def move_arm_to_pose(runner: TrialRunner, pose_label: str, global_envelope: str = "tucked") -> bool:
    """Move TIAGo's arm physically to joint targets for pose_label and set local/global costmap footprints."""
    if pose_label not in ARM_CONFIGS:
        logger.error(f"Arm config '{pose_label}' is not defined in ARM_CONFIGS!")
        return False

    cfg = ARM_CONFIGS[pose_label]
    joints = cfg.get("joints")
    footprint = cfg.get("footprint")

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


def completed_episodes(output_dir: str, strategy: str, output_csv: str = "") -> set:
    """P0.5: Episode ids already on disk or CSV for this strategy, so a restart resumes cleanly."""
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


def run_online_causal_tuner(num_episodes: int, map_yaml: str, output_csv: str, model_path: str,
                            arm_pose: str = "carry", timeout_sec: float = 240.0,
                            clearance: float = 0.45, goal_clearance: float = 0.70,
                            pose_pool_json: str = ""):
    strategy_name = "Online Causal (Ours)"
    strategy_tag = "Online_Causal"
    logger.info(f"Preparing Online Causal Tuner Evaluation for '{strategy_name}' ({num_episodes} episodes)...")

    if not output_csv:
        output_csv = os.path.join("campaign-b-data", f"online_causal_{arm_pose}_results.csv" if arm_pose else "online_causal_results.csv")

    output_dir = os.path.dirname(output_csv) or "campaign-b-data"
    os.makedirs(output_dir, exist_ok=True)

    if not rclpy.ok():
        rclpy.init()

    # P0.5: Resume support
    done_episodes = completed_episodes(output_dir, strategy_name, output_csv)
    if done_episodes:
        logger.info(f"Resuming run: {len(done_episodes)} episodes already on disk for '{strategy_name}'.")

    # Instantiate OnlineCausalTunerNode and load models explicitly from absolute path
    abs_model_path = os.path.abspath(model_path)
    logger.info(f"Initializing OnlineCausalTunerNode with model_path={abs_model_path}...")
    tuner_node = OnlineCausalTunerNode()
    tuner_node.model_path = abs_model_path
    tuner_node._load_models()

    from rct_collector.environment_risk_node import OptimizedRiskStateNode
    risk_node = OptimizedRiskStateNode()

    executor = MultiThreadedExecutor()
    executor.add_node(risk_node)
    executor.add_node(tuner_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    logger.info("OptimizedRiskStateNode & Online Causal Tuner active in background thread ✓")

    # P1.1: Enable risk feature collection for full logging
    runner = TrialRunner(
        output_dir=output_dir,
        map_yaml_path=map_yaml,
        gt_min_rate_hz=0.0,
        collect_risk_features=True,
        timeout_sec=timeout_sec
    )
    runner.tuner_node = tuner_node

    try:
        pose_pool = []
        if pose_pool_json and os.path.exists(pose_pool_json):
            with open(pose_pool_json) as fp:
                pose_pool = json.load(fp)
            logger.info(f"Loaded {len(pose_pool)} paired poses from {pose_pool_json} ✓")

        pose_sampler = None
        if not pose_pool:
            logger.info(f"Initializing PoseSampler with clearance={clearance}m, goal_clearance={goal_clearance}m...")
            pose_sampler = PoseSampler(map_yaml_path=map_yaml, obstacle_clearance_m=clearance, seed=42)
            if hasattr(pose_sampler, "goal_clearance_m"):
                pose_sampler.goal_clearance_m = goal_clearance
            else:
                logger.warning(
                    "PoseSampler has no separate goal clearance; using %.2f m for both.",
                    max(clearance, goal_clearance))
                pose_sampler.obstacle_clearance_m = max(clearance, goal_clearance)
            pose_sampler.load_map()

        results = []

        for ep_id in range(1, num_episodes + 1):
            if pose_pool and ep_id <= len(pose_pool):
                ep_info = pose_pool[ep_id - 1]
                start_pose = ep_info["start_pose"]
                goal_pose = ep_info["goal_pose"]
            else:
                start_pose, goal_pose = pose_sampler.sample_start_goal()

            if ep_id in done_episodes:
                logger.info(f"Skipping episode {ep_id}/{num_episodes} (already completed on disk)")
                continue

            logger.info(f"\n--- Episode {ep_id}/{num_episodes} | Strategy: '{strategy_name}' ---")
            logger.info(f"Relocating robot to start pose: ({start_pose['x']:.2f}, {start_pose['y']:.2f})...")
            runner._teleport_robot(start_pose)
            runner._publish_initial_pose(start_pose)
            time.sleep(1.0)

            if arm_pose:
                move_arm_to_pose(runner, arm_pose, global_envelope="tucked")
                tuner_node.current_arm_label = arm_pose
                tuner_node.reset_trial_metrics()
            tuner_node.set_goal(goal_pose)

            trial_id = f"ep{ep_id}_{strategy_tag}"
            init_params = {
                "local_costmap": {"footprint": arm_pose},
                "controller_server": {"FollowPath.vx_max": 0.55, "FollowPath.wz_max": 1.0}
            }

            res = runner.run_trial(
                trial_id=trial_id,
                start_pose=start_pose,
                goal_pose=goal_pose,
                params=init_params
            )

            # P1.2: Write decision log and parameter set failures into trial JSON
            out_of_supp_frac_mission = float("nan")
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
                trial_json = os.path.join(output_dir, f"trial_{res.trial_id}.json")
                if os.path.exists(trial_json):
                    try:
                        with open(trial_json, "r") as f:
                            blob = json.load(f)
                        blob["selection_mode"] = "enumerate"
                        blob["tuner_decision_log"] = _dl
                        blob["n_post_mission_ticks_dropped"] = (
                            len(tuner_node.decision_log) - len(_dl))
                        blob["out_of_support_fraction_mission"] = out_of_supp_frac_mission
                        blob["n_param_set_failures"] = tuner_node.n_param_set_failures
                        blob["param_set_failures"] = tuner_node.param_set_failures
                        blob["global_envelope"] = "tucked"
                        blob["global_inflation_radius"] = 0.55
                        blob["n_stale_ticks"] = tuner_node.n_stale_ticks
                        blob["n_tick_overruns"] = tuner_node.n_tick_overruns
                        blob["max_tick_ms"] = round(float(tuner_node.max_tick_ms), 1)
                        blob["n_arm_retries"] = tuner_node.n_arm_retries
                        blob["envelope_constants"] = getattr(tuner_node, "envelope_provenance", {})
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

            carry_frac = tuner_node.get_carry_fraction()
            carry_dist = path_len * carry_frac
            n_cfg_sw = tuner_node.n_config_switches
            n_arm_sw = tuner_node.n_arm_switches
            out_of_supp_frac = out_of_supp_frac_mission if not math.isnan(out_of_supp_frac_mission) else tuner_node.get_out_of_support_fraction()

            logger.info(
                f"Outcome: {st} | PoseOK: {bool(pose_ok)} | "
                f"Travel Time: {time_s:.1f}s | Path Length: {path_len:.2f}m")

            row_dict = {
                "episode_id": ep_id,
                "strategy": strategy_name,
                "arm_pose": arm_pose,
                "reached": reached,
                "pose_ok": pose_ok,
                "collision": coll,
                "plan_fail": plan_fail,
                "stuck": stuck,
                "time_to_goal": round(float(time_s), 2),
                "path_length_m": round(float(path_len), 2),
                "carry_distance_m": round(float(carry_dist), 2),
                "carry_fraction": round(float(carry_frac), 4),
                "n_config_switches": n_cfg_sw,
                "n_arm_switches": n_arm_sw,
                "out_of_support_fraction": round(float(out_of_supp_frac), 4),
                "status": res.status,
                "failure_reason": res.failure_reason
            }
            results.append(row_dict)

        if results:
            df_out = pd.DataFrame(results)
            if os.path.exists(output_csv):
                df_old = pd.read_csv(output_csv)
                df_out = pd.concat([df_old, df_out], ignore_index=True)
            df_out.to_csv(output_csv, index=False)
            logger.info(f"Successfully recorded results for '{strategy_name}' to {output_csv} ✓")

    finally:
        runner.shutdown()
        executor.shutdown()
        tuner_node.destroy_node()


def main():
    default_map = "/home/forough/phd_projects/online_tuner/src/gazebo_simulation/maps/causal_benchmark.yaml"
    default_model = "/home/forough/phd_projects/online_tuner/src/online_causal_tuner/models/causal_tuner_models.pkl"

    parser = argparse.ArgumentParser(description="Run Online Causal Tuner Navigation Trials")
    parser.add_argument("--num-episodes", type=int, default=30, help="Number of episodes")
    parser.add_argument("--map-yaml", type=str, default=default_map, help="Path to map.yaml")
    parser.add_argument("--model-path", type=str, default=default_model, help="Path to causal_tuner_models.pkl")
    parser.add_argument("--results-csv", type=str, default="", help="Output CSV path")
    parser.add_argument("--arm-pose", type=str, default="carry", choices=["", "tucked", "carry"], help="Set arm pose")
    parser.add_argument("--timeout-sec", type=float, default=240.0, help="Maximum trial duration in seconds (default: 240.0)")
    parser.add_argument("--clearance", type=float, default=0.45, help="Obstacle clearance (m) for PoseSampler start poses (default: 0.45)")
    parser.add_argument("--goal-clearance", type=float, default=0.70, help="Obstacle clearance (m) for GOAL poses (default: 0.70)")
    default_pool = "/home/forough/phd_projects/online_tuner/src/gazebo_simulation/maps/pose_pool_seed42_causal_benchmark.json"
    parser.add_argument("--pose-pool", type=str, default=default_pool, help="Path to pre-generated pose pool JSON file")
    args = parser.parse_args()

    run_online_causal_tuner(args.num_episodes, args.map_yaml, args.results_csv, args.model_path, args.arm_pose, timeout_sec=args.timeout_sec, clearance=args.clearance, goal_clearance=args.goal_clearance, pose_pool_json=args.pose_pool)


if __name__ == "__main__":
    main()
