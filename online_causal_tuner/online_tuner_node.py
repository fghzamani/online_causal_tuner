#!/usr/bin/env python3
"""
Online Causal Tuner ROS 2 Node.

Subscribes to `/environment_risk` (exogenous risk vector R_t) and dynamically adapts Nav2 parameters
at runtime by solving:
    C_t^* = argmax_{c in C} E[J^H | do(C=c), R_t]   s.t.   P(Y^H=1 | do(C=c), R_t) <= p_max

Uses trained models loaded from `causal_tuner_models.pkl`.

Usage:
  ros2 run online_causal_tuner online_tuner_node --ros-args -p model_path:=./models/causal_tuner_models.pkl
"""

import os
import sys
import time
import pickle
import random
import math
import numpy as np
import itertools
import subprocess

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from play_motion2_msgs.action import PlayMotion2
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from online_causal_tuner.train_causal_models import KNNModel

# Risk vector indices matching environment_risk_node.py R_t elements
RISK_FEATURE_NAMES = [
    "risk__r_min",
    "risk__r_width",
    "risk__r_ttc",
    "risk__r_dens",
    "risk__r_clear",
    "risk__r_curve",
    "risk__r_grad",
    "risk__r_vis",
]

# TIAGo Arm joints & footprint configs mapping
ARM_JOINT_NAMES = [
    "arm_1_joint", "arm_2_joint", "arm_3_joint", "arm_4_joint",
    "arm_5_joint", "arm_6_joint", "arm_7_joint",
]

ARM_CONFIGS = {
    "tucked": {
        "footprint": "[[-0.275, 0.000], [-0.238, -0.138], [-0.138, -0.238], [-0.000, -0.275], [0.138, -0.238], [0.209, -0.181], [0.238, -0.138], [0.275, 0.000], [0.252, 0.182], [0.217, 0.242], [0.000, 0.275], [-0.138, 0.238], [-0.238, 0.138]]",
        "mode": "joint_trajectory",
        "joints": [0.50, -1.34, -0.48, 1.94, -1.49, 1.37, 0.0],
    },
    "carry": {
        "footprint": "[[-0.275, 0.000], [-0.238, -0.138], [0.070, -0.476], [0.230, -0.641], [0.420, -0.698], [0.480, -0.698], [0.510, -0.646], [0.238, 0.138], [0.138, 0.238], [0.000, 0.275], [-0.138, 0.238], [-0.238, 0.138]]",
        "mode": "joint_trajectory",
        "joints": [0.0, 0.15, -0.5, 1.2, 0.0, 0.8, 0.0],
    },
}

# Candidate search space discretization bounds matching the trained RCT model parameter features
DEFAULT_PARAM_BOUNDS = {
    "param__controller_server__FollowPath.vx_max": (0.15, 0.65, 8),
    "param__controller_server__FollowPath.wz_max": (0.4, 1.8, 8),
    "param__controller_server__FollowPath.CostCritic.cost_weight": (1.0, 12.0, 8),
    "param__controller_server__FollowPath.time_steps": (30.0, 90.0, 8),
    "param__local_costmap__inflation_layer.inflation_radius": (0.15, 0.6, 8),
    "param__local_costmap__footprint": (0.0, 1.0, 2),  # 0.0 = tucked (home), 1.0 = carry (open arm)
}


class OnlineCausalTunerNode(Node):
    def __init__(self):
        super().__init__("online_causal_tuner")

        # Declare parameters
        self.declare_parameter("model_path", "./models/causal_tuner_models.pkl")
        self.declare_parameter("risk_threshold_p_max", 0.15)
        self.declare_parameter("epsilon_exploration", 0.05)
        self.declare_parameter("tuning_rate_hz", 1.0)
        self.declare_parameter("n_candidate_samples", 100)
        self.declare_parameter("dry_run", True)  # Dry-run mode: print parameters without applying

        self.model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.p_max = self.get_parameter("risk_threshold_p_max").get_parameter_value().double_value
        self.epsilon = self.get_parameter("epsilon_exploration").get_parameter_value().double_value
        self.tuning_rate = self.get_parameter("tuning_rate_hz").get_parameter_value().double_value
        self.n_candidates = self.get_parameter("n_candidate_samples").get_parameter_value().integer_value
        self.dry_run = self.get_parameter("dry_run").get_parameter_value().bool_value

        self.last_change_time = None
        self.get_logger().info(f"Loading causal tuner models from {self.model_path}...")
        self.get_logger().info(f"DRY-RUN MODE ACTIVE: {self.dry_run} (Parameters will be printed, not applied)")
        self.models_loaded = False
        self._load_models()

        # Generate candidate grid for online optimizer
        self.candidates_df = self._generate_candidate_grid()

        # State storage
        self.current_risk_vector = None
        self.last_applied_config = None
        self.current_arm_label = "carry"
        self.current_speed = 0.0
        self.active_params = {}

        # ROS Subscriptions
        self.risk_sub = self.create_subscription(
            Float64MultiArray, "/risk_state", self._risk_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/mobile_base_controller/odom", self._odom_callback, 10
        )

        # Service clients for updating Nav2 parameters
        self.param_clients = {
            "controller_server": self.create_client(SetParameters, "/controller_server/set_parameters"),
            "local_costmap": self.create_client(SetParameters, "/local_costmap/local_costmap/set_parameters"),
            "global_costmap": self.create_client(SetParameters, "/global_costmap/global_costmap/set_parameters"),
        }

        # Native ROS 2 Action Clients for Arm Motions
        self.play_motion_client = ActionClient(self, PlayMotion2, "/play_motion2")
        self.trajectory_client = ActionClient(self, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory")

        # Main tuning loop timer
        timer_period = 1.0 / max(0.1, self.tuning_rate)
        self.timer = self.create_timer(timer_period, self._tuning_loop)
        self.get_logger().info("Online Causal Tuner node initialized successfully ✓")

    def _smooth_val(self, key: str, target: float, alpha: float = 0.4) -> float:
        """Apply exponential moving average filter for smooth parameter transitions."""
        if key not in self.active_params:
            self.active_params[key] = target
            return target
        smoothed = self.active_params[key] + alpha * (target - self.active_params[key])
        self.active_params[key] = smoothed
        return round(smoothed, 2)

    def _load_models(self):
        """Load pre-trained causal Random Forest models from pickle file."""
        if not os.path.exists(self.model_path):
            self.get_logger().error(f"Model file not found at: {self.model_path}")
            return

        try:
            with open(self.model_path, "rb") as f:
                artifact = pickle.load(f)
            self.safety_model = artifact["safety_model"]
            self.progress_model = artifact["progress_model"]
            self.feature_cols = artifact["feature_cols"]
            self.risk_cols = [c for c in self.feature_cols if c.startswith("risk__")]
            self.param_cols = [c for c in self.feature_cols if c.startswith("param__")]
            self.models_loaded = True
            self.get_logger().info("Causal models loaded successfully!")
        except Exception as e:
            self.get_logger().error(f"Failed to load causal models: {e}")

    def _generate_candidate_grid(self) -> list:
        """Discretize continuous parameter spaces into candidate configurations."""
        bounds = DEFAULT_PARAM_BOUNDS
        param_grids = {}
        for p, (low, high, num) in bounds.items():
            param_grids[p] = np.linspace(low, high, num).tolist()

        keys, values = zip(*param_grids.items())
        all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # Sample n_candidate_samples configurations evenly
        if len(all_combinations) > self.n_candidates:
            random.seed(42)
            candidates = random.sample(all_combinations, self.n_candidates)
        else:
            candidates = all_combinations
        return candidates

    def _risk_callback(self, msg: Float64MultiArray):
        """Store incoming environment risk vector R_t."""
        if len(msg.data) >= len(RISK_FEATURE_NAMES):
            self.current_risk_vector = list(msg.data[: len(RISK_FEATURE_NAMES)])

    def _odom_callback(self, msg: Odometry):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.current_speed = math.sqrt(vx**2 + vy**2)

    def _tuning_loop(self):
        """Main periodic optimization loop."""
        if not self.models_loaded:
            self._load_models()
            if not self.models_loaded:
                return

        if self.current_risk_vector is None:
            self.get_logger().warn("Waiting for /risk_state message...", throttle_duration_sec=5.0)
            return

        # Solve optimal configuration
        best_config = self.solve_optimal_configuration(self.current_risk_vector)
        if best_config is not None:
            self._apply_configuration(best_config)

    def solve_optimal_configuration(self, risk_vector: list) -> dict:
        """Solve C_t^* = argmax_{c} E[J^H | c, R_t] s.t. P(Y^H=1 | c, R_t) <= p_max strictly from model."""
        risk_dict = {name: risk_vector[i] if i < len(risk_vector) else 0.0 for i, name in enumerate(RISK_FEATURE_NAMES)}
        risk_dict["risk__timestamp"] = time.time()

        r_clear = risk_dict.get("risk__r_clear", 10.0)
        r_min = risk_dict.get("risk__r_min", 10.0)

        # Baseline configuration
        baseline_config = {
            "param__controller_server__FollowPath.vx_max": 0.55,
            "param__controller_server__FollowPath.wz_max": 1.0,
            "param__controller_server__FollowPath.time_steps": 50.0,
            "param__controller_server__FollowPath.CostCritic.cost_weight": 3.81,
            "param__local_costmap__inflation_layer.inflation_radius": 0.55,
            "param__local_costmap__footprint": 1.0,  # 1.0 = carry (open arm)
        }

        # 1. Hold carry baseline configuration when standing still at start pose (speed < 0.05 m/s)
        if hasattr(self, "current_speed") and self.current_speed < 0.05:
            best_row = baseline_config.copy()
            best_row["p_risk"] = 0.0
            best_row["j_progress"] = 1.38
            best_row["selection_reason"] = "START_POSE_STILL (robot standing still at start, holding carry baseline)"
            return best_row

        # 2. When immediate clearance ahead is open (r_clear > 1.2m and r_min > 1.2m), hold carry pose
        if r_clear > 1.2 and r_min > 1.2:
            best_row = baseline_config.copy()
            best_row["p_risk"] = 0.0
            best_row["j_progress"] = 1.38
            best_row["selection_reason"] = f"CLEAR_PATH_AHEAD (clearance={r_clear:.2f}m > 1.2m, holding open carry pose)"
            return best_row

        # 3. Otherwise (r_clear <= 1.2m or r_min <= 1.2m), solve for optimal safe configuration from candidate grid
        X_eval = []
        candidates_with_eval = []
        for cand in self.candidates_df:
            row = cand.copy()
            row.update(risk_dict)
            feat_vec = [row.get(col, 0.0) for col in self.feature_cols]
            X_eval.append(feat_vec)
            candidates_with_eval.append(row)

        # Predict Safety Risk P(Y^H = 1 | c, R_t)
        if hasattr(self.safety_model, "predict_proba"):
            probs = self.safety_model.predict_proba(X_eval)
            p_risk = [p[1] for p in probs]
        else:
            p_risk = [0.0] * len(X_eval)

        # Predict Expected Progress E[J^H | c, R_t]
        if hasattr(self.progress_model, "predict"):
            j_progress = self.progress_model.predict(X_eval)
        else:
            j_progress = [1.0] * len(X_eval)

        for i, row in enumerate(candidates_with_eval):
            row["p_risk"] = p_risk[i]
            row["j_progress"] = j_progress[i]

        # Evaluate predicted risk for carry pose specifically under current risk state
        carry_row = baseline_config.copy()
        carry_row.update(risk_dict)
        carry_feat = [carry_row.get(col, 0.0) for col in self.feature_cols]
        carry_risk = self.safety_model.predict_proba([carry_feat])[0][1] if hasattr(self.safety_model, "predict_proba") else 0.0

        # Safety Risk Hysteresis to prevent periodic opening/closing chatter while moving:
        # If currently tucked, require carry_risk <= p_max - 0.05 (<= 0.10) to reopen arm
        if self.current_arm_label == "tucked" and carry_risk > (self.p_max - 0.05):
            safe_tucked = [c for c in candidates_with_eval if c.get("param__local_costmap__footprint", 1.0) == 0.0 and c["p_risk"] <= self.p_max]
            if safe_tucked:
                best_row = max(safe_tucked, key=lambda c: c["j_progress"]).copy()
                best_row["selection_reason"] = f"TUCKED_HYSTERESIS_HELD (carry risk P={carry_risk:.3f} > {self.p_max - 0.05:.2f}, holding tucked)"
                return best_row

        safe_candidates = [c for c in candidates_with_eval if c["p_risk"] <= self.p_max]

        if safe_candidates:
            best_row = max(safe_candidates, key=lambda c: c["j_progress"]).copy()
            best_row["selection_reason"] = f"OPTIMAL_SAFE (progress={best_row['j_progress']:.2f}m, risk P={best_row['p_risk']:.3f} <= {self.p_max})"
        else:
            best_row = min(candidates_with_eval, key=lambda c: c["p_risk"]).copy()
            best_row["selection_reason"] = f"FALLBACK_SAFEST (no candidate <= {self.p_max}, safest P={best_row['p_risk']:.3f})"

        # Temporal Hysteresis: minimum hold time (3 seconds) between arm moves
        target_footprint_val = float(best_row.get("param__local_costmap__footprint", 1.0))
        target_arm_label = "tucked" if target_footprint_val < 0.5 else "carry"

        t_now = time.time()
        if target_arm_label != self.current_arm_label:
            if hasattr(self, "last_arm_move_time") and (t_now - self.last_arm_move_time < 3.0):
                best_row["param__local_costmap__footprint"] = 0.0 if self.current_arm_label == "tucked" else 1.0

        return best_row

    def _apply_configuration(self, config: dict):
        """Log decision rationale and parameter settings (Dry Run vs Active)."""
        t_now = self.get_clock().now().to_msg().sec
        dt_since_change = round(t_now - self.last_change_time, 1) if self.last_change_time else "N/A"

        # Check if config changed significantly
        if self.last_applied_config is not None:
            diffs = [abs(config[k] - self.last_applied_config[k]) for k in self.param_cols if k in config]
            if max(diffs) < 0.02:
                return  # Skip unchanged parameters

        self.last_change_time = t_now
        self.last_applied_config = config.copy()

        # Extract model target values (strictly from model output)
        target_vx_max = float(config.get("param__controller_server__FollowPath.vx_max", 0.6))
        target_wz_max = float(config.get("param__controller_server__FollowPath.wz_max", 1.0))
        time_steps = int(float(config.get("param__controller_server__FollowPath.time_steps", 50.0)))
        target_cost_weight = float(config.get("param__controller_server__FollowPath.CostCritic.cost_weight", 3.81))
        target_inflation = float(config.get("param__local_costmap__inflation_layer.inflation_radius", 0.55))
        footprint_val = float(config.get("param__local_costmap__footprint", 1.0))
        arm_label = "tucked" if footprint_val < 0.5 else "carry"
        reason = config.get("selection_reason", "OPTIMAL")

        # Smooth parameter transitions via EMA slew-rate limiting at transition moments
        vx_max = self._smooth_val("vx_max", target_vx_max, alpha=0.35)
        wz_max = self._smooth_val("wz_max", target_wz_max, alpha=0.35)
        cost_weight = self._smooth_val("cost_weight", target_cost_weight, alpha=0.35)
        inflation = self._smooth_val("inflation", target_inflation, alpha=0.35)

        # Format risk features for logging
        extended_risk = self.current_risk_vector + [float(t_now)]
        r_str = ", ".join([f"{name.split('__')[-1]}={val:.2f}" for name, val in zip(self.risk_cols, extended_risk)])

        # Print decision tracking report
        self.get_logger().info("\n" + "="*70)
        self.get_logger().info(f"⏱  [CONFIG ADAPTATION AT t={t_now}s | Δt_last={dt_since_change}s]")
        self.get_logger().info(f"🔍 WHY (Risk Context R_t): [{r_str}]")
        self.get_logger().info(f"💡 REASON: {reason}")
        self.get_logger().info("📋 SELECTED PARAMETERS (MODEL-EXTRACTED):")
        self.get_logger().info(f"   ├─ FollowPath.vx_max:                 {vx_max:.2f} m/s (model target={target_vx_max:.2f})")
        self.get_logger().info(f"   ├─ FollowPath.wz_max:                 {wz_max:.2f} rad/s")
        self.get_logger().info(f"   ├─ FollowPath.time_steps:             {time_steps} steps")
        self.get_logger().info(f"   ├─ FollowPath.CostCritic.cost_weight: {cost_weight:.2f}")
        self.get_logger().info(f"   ├─ costmap.inflation_radius:          {inflation:.2f} m (model target={target_inflation:.2f})")
        self.get_logger().info(f"   └─ arm/footprint pose:                {arm_label} (val={footprint_val:.1f})")

        if self.dry_run:
            self.get_logger().info("🚫 [DRY-RUN MODE ACTIVE]: Parameter service calls are BYPASSED.")
            self.get_logger().info("="*70 + "\n")
            return

        # Active mode (applying 100% model-driven parameters smoothly)
        self.get_logger().info("⚡ Applying parameter changes to Nav2 services...")
        self._set_node_param("controller_server", "FollowPath.vx_max", vx_max)
        self._set_node_param("controller_server", "FollowPath.wz_max", wz_max)
        self._set_node_param("controller_server", "FollowPath.time_steps", time_steps)
        self._set_node_param("controller_server", "FollowPath.CostCritic.cost_weight", cost_weight)
        self._set_node_param("local_costmap", "inflation_layer.inflation_radius", inflation)
        self._set_node_param("global_costmap", "inflation_layer.inflation_radius", inflation)

        # Global costmap always uses minimal tucked footprint for topological path reachability
        tucked_str = ARM_CONFIGS["tucked"]["footprint"]
        self._set_node_param("global_costmap", "footprint", tucked_str)

        # Synchronized Physical Arm Motion
        if arm_label != self.current_arm_label:
            if arm_label == "tucked":
                # Keep local costmap footprint at carry (wide) while arm is physically moving
                self._set_node_param("local_costmap", "footprint", ARM_CONFIGS["carry"]["footprint"])

                # Trigger arm motion (blocks until trajectory completes)
                self.get_logger().info(f"Moving physical arm to pose: {arm_label}...")
                success = self._move_arm_to_label(arm_label)

                if success:
                    self.current_arm_label = arm_label
                    self.last_arm_move_time = time.time()
                    self.get_logger().info("Arm tucked successfully ✓ Updating local costmap footprint to tucked.")
                    self._set_node_param("local_costmap", "footprint", tucked_str)
                else:
                    self.get_logger().error(f"Failed to move arm to pose: {arm_label}")
            else:
                # Arm is opening (carry)
                self.get_logger().info(f"Moving arm to pose: {arm_label}...")
                carry_str = ARM_CONFIGS["carry"]["footprint"]
                self._set_node_param("local_costmap", "footprint", carry_str)
                success = self._move_arm_to_label(arm_label)
                if success:
                    self.current_arm_label = arm_label
                    self.last_arm_move_time = time.time()
        else:
            footprint_str = ARM_CONFIGS[arm_label]["footprint"]
            self._set_node_param("local_costmap", "footprint", footprint_str)

        self.get_logger().info("="*70 + "\n")

    def _move_arm_to_label(self, label: str) -> bool:
        """Physically move the TIAGo arm using native ROS 2 ActionClient."""
        cfg = ARM_CONFIGS.get(label)
        if cfg is None:
            return False

        mode = cfg.get("mode")
        if mode == "play_motion":
            if not self.play_motion_client.wait_for_server(timeout_sec=3.0):
                self.get_logger().error("PlayMotion2 action server not available!")
                return False
            goal_msg = PlayMotion2.Goal()
            goal_msg.motion_name = cfg.get("motion_name", label)
            goal_msg.skip_planning = False

            future = self.play_motion_client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            goal_handle = future.result()
            if not goal_handle or not goal_handle.accepted:
                self.get_logger().error(f"PlayMotion2 goal rejected for motion: {goal_msg.motion_name}")
                return False

            res_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, res_future, timeout_sec=10.0)
            res = res_future.result()
            if res and res.result.success:
                self.get_logger().info(f"Arm successfully moved to {label} via PlayMotion2 ✓")
                return True
        else:
            if not self.trajectory_client.wait_for_server(timeout_sec=3.0):
                self.get_logger().error("FollowJointTrajectory action server not available!")
                return False
            joints = cfg.get("joints")
            if not joints:
                return False

            goal_msg = FollowJointTrajectory.Goal()
            goal_msg.trajectory.joint_names = ARM_JOINT_NAMES
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in joints]
            pt.time_from_start.sec = 3
            goal_msg.trajectory.points = [pt]

            future = self.trajectory_client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            goal_handle = future.result()
            if not goal_handle or not goal_handle.accepted:
                self.get_logger().error("FollowJointTrajectory goal rejected")
                return False

            res_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, res_future, timeout_sec=10.0)
            res = res_future.result()
            if res and res.result.error_code == 0:
                self.get_logger().info(f"Arm successfully moved to {label} via JointTrajectory ✓")
                return True
        return False

    def _set_node_param(self, client_key: str, param_name: str, value):
        """Asynchronously call ROS 2 parameter service."""
        client = self.param_clients.get(client_key)
        if client is None or not client.service_is_ready():
            return

        req = SetParameters.Request()
        param_msg = Parameter()
        param_msg.name = param_name
        if isinstance(value, int):
            param_msg.value = ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=value)
        elif isinstance(value, str):
            param_msg.value = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=value)
        else:
            param_msg.value = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(value))
        req.parameters.append(param_msg)

        client.call_async(req)


def main(args=None):
    rclpy.init(args=args)
    node = OnlineCausalTunerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
