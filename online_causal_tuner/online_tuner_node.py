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
import subprocess

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
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
        "mode": "play_motion",
        "motion_name": "home",
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

        # Main tuning loop timer
        timer_period = 1.0 / max(0.1, self.tuning_rate)
        self.timer = self.create_timer(timer_period, self._tuning_loop)
        self.get_logger().info("Online Causal Tuner node initialized successfully ✓")

    def _load_models(self):
        """Load trained safety & progress model pickle."""
        if not os.path.exists(self.model_path):
            self.get_logger().error(f"Model file not found at {self.model_path}. Node waiting...")
            return

        try:
            with open(self.model_path, "rb") as f:
                artifact = pickle.load(f)
            self.safety_model = artifact["safety_model"]
            self.progress_model = artifact["progress_model"]
            self.feature_scaler = artifact.get("feature_scaler")
            self.risk_cols = artifact["risk_cols"]
            self.param_cols = artifact["param_cols"]
            self.feature_cols = artifact["feature_cols"]
            self.safety_type = artifact.get("safety_model_type", "rf")
            self.models_loaded = True
            self.get_logger().info("Causal models loaded successfully!")
        except Exception as e:
            self.get_logger().error(f"Failed to load models: {e}")

    def _generate_candidate_grid(self) -> list:
        """Sample discrete candidate configs for runtime evaluation."""
        rng = random.Random(42)
        records = []
        for _ in range(self.n_candidates):
            row = {}
            for param, (low, high, _) in DEFAULT_PARAM_BOUNDS.items():
                row[param] = float(rng.uniform(low, high))
            records.append(row)
        return records

    def _risk_callback(self, msg: Float64MultiArray):
        """Store incoming environment risk vector R_t."""
        if len(msg.data) >= len(RISK_FEATURE_NAMES):
            self.current_risk_vector = list(msg.data[: len(RISK_FEATURE_NAMES)])

    def _odom_callback(self, msg: Odometry):
        pass

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
        """Solve C_t^* = argmax_{c} E[J^H | c, R_t] s.t. P(Y^H=1 | c, R_t) <= p_max."""
        risk_dict = {name: risk_vector[i] if i < len(risk_vector) else 0.0 for i, name in enumerate(RISK_FEATURE_NAMES)}
        risk_dict["risk__timestamp"] = time.time()

        # 1. Evaluate baseline configuration first to avoid jitter/tuning in free space
        baseline_config = {
            "param__controller_server__FollowPath.vx_max": 0.55,
            "param__controller_server__FollowPath.wz_max": 1.0,
            "param__controller_server__FollowPath.time_steps": 50.0,
            "param__controller_server__FollowPath.CostCritic.cost_weight": 3.81,
            "param__local_costmap__inflation_layer.inflation_radius": 0.55,
            "param__local_costmap__footprint": 1.0,  # 1.0 = carry (open arm)
        }

        baseline_row = baseline_config.copy()
        baseline_row.update(risk_dict)
        baseline_feat = [baseline_row.get(col, 0.0) for col in self.feature_cols]

        if hasattr(self.safety_model, "predict_proba"):
            baseline_p = self.safety_model.predict_proba([baseline_feat])[0][1]
        else:
            baseline_p = 0.0

        # 1. Check spatial risk context for open/free space
        r_clear = risk_dict.get("risk__r_clear", 10.0)
        r_min = risk_dict.get("risk__r_min", 10.0)

        # In open/free space (r_clear > 1.5m and r_min > 1.5m), maintain baseline carry pose to avoid jitter
        if r_clear > 1.5 and r_min > 1.5:
            best_row = baseline_config.copy()
            best_row["p_risk"] = baseline_p
            best_row["j_progress"] = 1.38
            best_row["selection_reason"] = f"OPEN_SPACE_BASELINE (clearance={r_clear:.2f}m, staying at carry baseline)"
            return best_row

        # 2. Otherwise, solve for optimal safe configuration from the candidate grid
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

        # Epsilon-greedy exploration (only during active tuning)
        if random.random() < self.epsilon:
            best_row = random.choice(candidates_with_eval).copy()
            best_row["selection_reason"] = f"EXPLORATION (eps={self.epsilon})"
            return best_row

        safe_candidates = [c for c in candidates_with_eval if c["p_risk"] <= self.p_max]

        if safe_candidates:
            best_row = max(safe_candidates, key=lambda c: c["j_progress"]).copy()
            best_row["selection_reason"] = f"OPTIMAL_SAFE (progress={best_row['j_progress']:.2f}m, risk P={best_row['p_risk']:.3f} <= {self.p_max})"
            return best_row
        else:
            best_row = min(candidates_with_eval, key=lambda c: c["p_risk"]).copy()
            best_row["selection_reason"] = f"FALLBACK_SAFEST (no candidate <= {self.p_max}, safest P={best_row['p_risk']:.3f})"
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

        # Extract values
        vx_max = float(config.get("param__controller_server__FollowPath.vx_max", 0.6))
        wz_max = float(config.get("param__controller_server__FollowPath.wz_max", 1.0))
        time_steps = int(float(config.get("param__controller_server__FollowPath.time_steps", 50.0)))
        cost_weight = float(config.get("param__controller_server__FollowPath.CostCritic.cost_weight", 3.81))
        inflation = float(config.get("param__local_costmap__inflation_layer.inflation_radius", 0.55))
        footprint_val = float(config.get("param__local_costmap__footprint", 1.0))
        arm_label = "tucked" if footprint_val < 0.5 else "carry"
        reason = config.get("selection_reason", "OPTIMAL")

        # Format risk features for logging
        extended_risk = self.current_risk_vector + [float(t_now)]
        r_str = ", ".join([f"{name.split('__')[-1]}={val:.2f}" for name, val in zip(self.risk_cols, extended_risk)])

        # Print decision tracking report
        self.get_logger().info("\n" + "="*70)
        self.get_logger().info(f"⏱  [CONFIG ADAPTATION AT t={t_now}s | Δt_last={dt_since_change}s]")
        self.get_logger().info(f"🔍 WHY (Risk Context R_t): [{r_str}]")
        self.get_logger().info(f"💡 REASON: {reason}")
        self.get_logger().info("📋 SELECTED PARAMETERS:")
        self.get_logger().info(f"   ├─ FollowPath.vx_max:                 {vx_max:.2f} m/s")
        self.get_logger().info(f"   ├─ FollowPath.wz_max:                 {wz_max:.2f} rad/s")
        self.get_logger().info(f"   ├─ FollowPath.time_steps:             {time_steps} steps")
        self.get_logger().info(f"   ├─ FollowPath.CostCritic.cost_weight: {cost_weight:.2f}")
        self.get_logger().info(f"   ├─ costmap.inflation_radius:          {inflation:.2f} m")
        self.get_logger().info(f"   └─ arm/footprint pose:                {arm_label} (val={footprint_val:.1f})")

        if self.dry_run:
            self.get_logger().info("🚫 [DRY-RUN MODE ACTIVE]: Parameter service calls are BYPASSED.")
            self.get_logger().info("="*70 + "\n")
            return

        # Active mode (when dry_run is False)
        self.get_logger().info("⚡ Applying parameter changes to Nav2 services...")
        self._set_node_param("controller_server", "FollowPath.vx_max", vx_max)
        self._set_node_param("controller_server", "FollowPath.wz_max", wz_max)
        self._set_node_param("controller_server", "FollowPath.time_steps", time_steps)
        self._set_node_param("controller_server", "FollowPath.CostCritic.cost_weight", cost_weight)
        self._set_node_param("local_costmap", "inflation_layer.inflation_radius", inflation)
        self._set_node_param("global_costmap", "inflation_layer.inflation_radius", inflation)

        # Apply footprint preset: local costmap uses active pose, global costmap uses minimal tucked pose
        footprint_str = ARM_CONFIGS[arm_label]["footprint"]
        tucked_str = ARM_CONFIGS["tucked"]["footprint"]
        self.get_logger().info(f"Changing local costmap footprint to: {arm_label}")
        self._set_node_param("local_costmap", "footprint", footprint_str)
        self._set_node_param("global_costmap", "footprint", tucked_str)

        # Move physical arm if changed
        if arm_label != self.current_arm_label:
            self.get_logger().info(f"Moving arm to pose: {arm_label}...")
            success = self._move_arm_to_label(arm_label)
            if success:
                self.current_arm_label = arm_label
            else:
                self.get_logger().error(f"Failed to move arm to pose: {arm_label}")

        self.get_logger().info("="*70 + "\n")

    def _move_arm_to_label(self, label: str) -> bool:
        """Physically move the TIAGo arm to the configuration matching the footprint."""
        cfg = ARM_CONFIGS.get(label)
        if cfg is None:
            return False

        import json
        mode = cfg.get("mode")
        if mode == "play_motion":
            motion = cfg.get("motion_name", label)
            goal_dict = {"motion_name": motion, "skip_planning": False}
            goal = json.dumps(goal_dict)
            cmd = ["ros2", "action", "send_goal", "/play_motion2", "play_motion2_msgs/action/PlayMotion2", goal]
        else:
            joints = cfg.get("joints")
            if not joints:
                return False
            goal_dict = {
                "trajectory": {
                    "joint_names": ARM_JOINT_NAMES,
                    "points": [{
                        "positions": [float(v) for v in joints],
                        "time_from_start": {"sec": 4}
                    }]
                }
            }
            goal = json.dumps(goal_dict)
            cmd = ["ros2", "action", "send_goal", "/arm_controller/follow_joint_trajectory",
                   "control_msgs/action/FollowJointTrajectory", goal]

        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
            if r.returncode == 0 and "SUCCEEDED" in r.stdout:
                self.get_logger().info(f"Arm successfully moved to {label} ✓")
                return True
        except Exception as e:
            self.get_logger().error(f"Exception moving arm to {label}: {e}")
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
