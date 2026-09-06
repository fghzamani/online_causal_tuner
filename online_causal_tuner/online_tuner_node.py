#!/usr/bin/env python3
"""
Online Causal Tuner ROS 2 Node.

Subscribes to `/risk_state` (exogenous risk vector R_t) and dynamically adapts Nav2 parameters
at runtime by solving:
    C_t^* = argmax_{c in A(R_t)} U(c, R_t)   s.t.   P(collision | do(C=c), R_t) <= p_max

Uses trained 3-Model Hurdle causal models loaded from `causal_tuner_models.pkl`.
"""

import os
import sys
import argparse
import time
import math
import pickle
import random
import json
import numpy as np
import pandas as pd
import itertools
import subprocess

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from tf2_ros import Buffer, TransformListener
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PolygonStamped, Point32
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from nav2_msgs.msg import SpeedLimit


from online_causal_tuner.train_causal_models import CausalInteractionTransformer

# Register custom transformer class with __main__ so pickle resolves smoothly
_main_mod = sys.modules.get("__main__")
if _main_mod is not None:
    setattr(_main_mod, "CausalInteractionTransformer", CausalInteractionTransformer)

# Index-to-name map matching environment_risk_node.py RiskIndex enum exactly
RISK_VECTOR_INDEX_TO_NAME = {
    0: "risk__r_min",
    1: "risk__r_width",
    2: "risk__r_ttc",
    3: "risk__r_dens",
    4: "risk__r_clear",
    5: "risk__r_curve",
    6: "risk__r_grad",
    7: "risk__r_vis",
}

RISK_FEATURE_NAMES = list(RISK_VECTOR_INDEX_TO_NAME.values())

# TIAGo Arm joints & footprint configs mapping
ARM_JOINT_NAMES = [
    "arm_1_joint", "arm_2_joint", "arm_3_joint", "arm_4_joint",
    "arm_5_joint", "arm_6_joint", "arm_7_joint",
]

ARM_CONFIGS = {
    "tucked": {
        "footprint": "[[-0.275, 0.000], [-0.238, -0.138], [-0.138, -0.238], [-0.000, -0.275], [0.138, -0.238], [0.209, -0.181], [0.238, -0.138], [0.275, 0.000], [0.252, 0.182], [0.217, 0.242], [0.000, 0.275], [-0.138, 0.238], [-0.238, 0.138]]",
        "joints": [0.50, -1.34, -0.48, 1.94, -1.49, 1.37, 0.0],
    },
    "carry": {
        "footprint": "[[-0.275, 0.000], [-0.238, -0.138], [0.070, -0.476], [0.230, -0.641], [0.420, -0.698], [0.480, -0.698], [0.510, -0.646], [0.238, 0.138], [0.138, 0.238], [0.000, 0.275], [-0.138, 0.238], [-0.238, 0.138]]",
        "joints": [0.0, 0.15, -0.5, 1.2, 0.0, 0.8, 0.0],
    },
}

# Gate B1: Footprint Geometry Constants (measured from physical polygons)
FOOTPRINT_GEOMETRY = {
    0.0: {"label": "tucked", "r_circ": 0.325, "width": 0.550},
    1.0: {"label": "carry",  "r_circ": 0.847, "width": 0.973},
}

# P0.2a: Envelope-Only ablation fixed software knobs and joint tolerance
ENVELOPE_ONLY_SOFTWARE = {
    "param__controller_server__speed_limit_pct": 100.0,
    "param__controller_server__FollowPath.vx_std": 0.35,
    "param__controller_server__FollowPath.ConstraintCritic.cost_weight": 2.0,
    "param__controller_server__FollowPath.CostCritic.cost_weight": 3.0,
    "param__controller_server__FollowPath.PathAlignCritic.cost_weight": 15.0,
    "param__local_costmap__inflation_layer.inflation_radius": 0.30,
}

ARM_JOINT_TOLERANCE_RAD = 0.15

# Fixed terminal-phase speed budget. Nav2's percentage SpeedLimit scales wz as
# well as vx, so 30% would leave wz_eff = 0.57 rad/s. 70% gives 1.33 rad/s.
TERMINAL_SPEED_LIMIT_PCT = 70.0

FOOTPRINT_KEY = "param__local_costmap__footprint"
INFLATION_KEY = "param__local_costmap__inflation_layer.inflation_radius"
SPEED_LIMIT_KEY = "param__controller_server__speed_limit_pct"
VX_STD_KEY = "param__controller_server__FollowPath.vx_std"
CONSTRAINT_KEY = "param__controller_server__FollowPath.ConstraintCritic.cost_weight"
COST_WEIGHT_KEY = "param__controller_server__FollowPath.CostCritic.cost_weight"
PATH_ALIGN_KEY = "param__controller_server__FollowPath.PathAlignCritic.cost_weight"

MAX_VX_LIMIT = 0.7
MIN_SPEED_LIMIT = 30.0

# Gate B5: Optimized Candidate Grid Space (2304 grid points)
PARAM_CANDIDATE_GRID = {
    "param__controller_server__speed_limit_pct": [30.0, 50.0, 70.0, 90.0],
    "param__controller_server__FollowPath.vx_std": [0.15, 0.35],
    "param__controller_server__FollowPath.ConstraintCritic.cost_weight": [0.5, 3.0, 6.0],
    "param__controller_server__FollowPath.CostCritic.cost_weight": [1.0, 5.0, 10.0, 20.0],
    "param__controller_server__FollowPath.PathAlignCritic.cost_weight": [4.0, 15.0, 30.0],
    "param__local_costmap__inflation_layer.inflation_radius": [0.15, 0.30, 0.45, 0.60],
    "param__local_costmap__footprint": [0.0, 1.0],  # 0.0 = tucked, 1.0 = carry
}


class OnlineCausalTunerNode(Node):
    _instance = None

    def __init__(self):
        super().__init__("online_causal_tuner")
        OnlineCausalTunerNode._instance = self

        # Gate A1: Declare ROS parameters matching single YAML source of truth
        self.declare_parameter("model_path", "/home/forough/phd_projects/online_tuner/src/online_causal_tuner/models/causal_tuner_models.pkl")
        self.declare_parameter("risk_threshold_p_max", 0.10)
        self.declare_parameter("risk_penalty_lambda", 10.0)
        self.declare_parameter("stall_penalty_mu", 2.0)
        self.declare_parameter("payload_value_omega", 0.5)
        self.declare_parameter("utility_deadband_frac", 0.025)
        self.declare_parameter("arm_switch_cost_frac", 0.055)
        self.declare_parameter("clearance_margin_m", 0.10)
        self.declare_parameter("decel_limit_mps2", 0.60)
        self.declare_parameter("inflation_floor_m", 0.15)
        self.declare_parameter("tuning_rate_hz", 5.0)
        self.declare_parameter("dry_run", False)
        self.declare_parameter("envelope_only", False)
        self.declare_parameter("pessimism_alpha", 0.20)
        self.declare_parameter("anticipation_delta_s", 4.0)
        self.declare_parameter("selection_mode", "enumerate")
        self.declare_parameter("terminal_radius_m", 1.0)
        self.declare_parameter("arm_switch_dwell_s", 8.0)
        self.declare_parameter("arm_switch_persist_ticks", 5)

        self.model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.p_max = self.get_parameter("risk_threshold_p_max").get_parameter_value().double_value
        self.risk_lambda = self.get_parameter("risk_penalty_lambda").get_parameter_value().double_value
        self.stall_mu = self.get_parameter("stall_penalty_mu").get_parameter_value().double_value
        self.payload_omega = self.get_parameter("payload_value_omega").get_parameter_value().double_value
        self.deadband_frac = self.get_parameter("utility_deadband_frac").get_parameter_value().double_value
        self.arm_switch_frac = self.get_parameter("arm_switch_cost_frac").get_parameter_value().double_value
        self.clearance_margin_m = self.get_parameter("clearance_margin_m").get_parameter_value().double_value
        self.decel_limit_mps2 = self.get_parameter("decel_limit_mps2").get_parameter_value().double_value
        self.inflation_floor_m = self.get_parameter("inflation_floor_m").get_parameter_value().double_value
        self.tuning_rate = self.get_parameter("tuning_rate_hz").get_parameter_value().double_value
        self.dry_run = self.get_parameter("dry_run").get_parameter_value().bool_value
        self.envelope_only = self.get_parameter("envelope_only").get_parameter_value().bool_value
        self.alpha = self.get_parameter("pessimism_alpha").get_parameter_value().double_value
        self.anticipation_delta = self.get_parameter("anticipation_delta_s").get_parameter_value().double_value
        self.selection_mode = self.get_parameter("selection_mode").get_parameter_value().string_value
        self.terminal_radius = self.get_parameter("terminal_radius_m").get_parameter_value().double_value
        self.arm_switch_dwell = self.get_parameter("arm_switch_dwell_s").get_parameter_value().double_value
        self.arm_persist_ticks = self.get_parameter("arm_switch_persist_ticks").get_parameter_value().integer_value

        from collections import deque
        self.risk_history = deque(maxlen=90)   # ~3 s at 30 Hz
        self.last_oos_fraction = float("nan")

        # Gate A2: Transportability Out-of-Support Instrument counters
        self.n_ticks = 0
        self.n_out_of_support = 0
        self.out_of_support_features = {}

        # Gate A3: Switch rate counters
        self.n_config_switches = 0
        self.n_arm_switches = 0
        self.carry_samples = 0
        self.total_samples = 0

        # P0.2b & P1.2: Joint states verification & decision log
        self.joint_positions = {}
        self.applied_envelope_label = None
        self.verified_arm_label = None
        self.n_param_set_failures = 0
        self.param_set_failures = {}
        self.decision_log = []

        self.last_change_time = None
        self.models_loaded = False
        self._load_models()

        # Persistent service clients for async parameter updates
        self.param_clients = {
            "controller_server": self.create_client(SetParameters, "/controller_server/set_parameters"),
            "local_costmap": self.create_client(SetParameters, "/local_costmap/local_costmap/set_parameters"),
            "global_costmap": self.create_client(SetParameters, "/global_costmap/global_costmap/set_parameters"),
        }

        # Native ROS 2 Action Client for physical arm joint trajectory execution
        self.arm_action_client = ActionClient(self, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory")

        # Generate full candidate configuration space
        self.candidates_df = self._generate_candidate_grid()

        self.current_risk_vector = None
        self.last_applied_config = None
        self.current_arm_label = "carry"
        self.target_arm_label = "carry"
        self.arm_state_initialized = False
        self.arm_transition_until_time = 0.0
        self.current_speed = 0.0

        # Sensor callbacks must never be blocked by the tuning loop. Without
        # explicit groups every callback lands in the node's default
        # MutuallyExclusiveCallbackGroup, which a MultiThreadedExecutor still
        # serializes -- a slow tick then starves /risk_state and the selector
        # runs on a stale context.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.goal_xy = None
        self.in_terminal_phase = False
        self.n_terminal_ticks = 0
        self.n_projected_ticks = 0
        self.last_arm_switch_time = -1e9
        self.pending_arm_label = None
        self.pending_arm_count = 0

        self.sensor_cb_group = ReentrantCallbackGroup()
        self.tuner_cb_group = MutuallyExclusiveCallbackGroup()

        self.risk_sub = self.create_subscription(
            Float64MultiArray, "/risk_state", self._risk_callback, 10,
            callback_group=self.sensor_cb_group,
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/mobile_base_controller/odom", self._odom_callback, 10,
            callback_group=self.sensor_cb_group,
        )
        self.joint_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_state_callback, 10,
            callback_group=self.sensor_cb_group,
        )

        # Footprint & Speed Limit Publishers for immediate RViz visualization sync
        self.local_footprint_pub = self.create_publisher(PolygonStamped, "/local_costmap/published_footprint", 10)
        self.footprint_pub = self.create_publisher(PolygonStamped, "/footprint", 10)
        self.speed_limit_pub = self.create_publisher(SpeedLimit, "/speed_limit", 10)
        self.current_speed_limit_pct = None

        # Main tuning loop timer
        self.tick_period = 1.0 / max(0.1, self.tuning_rate)
        self.n_tick_overruns = 0
        self.n_stale_ticks = 0
        self.max_tick_ms = 0.0
        self.n_arm_retries = 0
        self.timer = self.create_timer(
            self.tick_period, self._tuning_loop, callback_group=self.tuner_cb_group
        )
        self.get_logger().info("Online Causal Tuner node initialized successfully (Hurdle Causal Formulation v2).")

    def _joint_state_callback(self, msg: JointState):
        """Store physical joint positions for verified arm state checking."""
        for name, pos in zip(msg.name, msg.position):
            self.joint_positions[name] = pos

    def _verified_arm_label(self):
        """Arm label confirmed by /joint_states, or None if in transit/unknown."""
        if not self.joint_positions:
            return None
        for label, cfg in ARM_CONFIGS.items():
            err = max(abs(self.joint_positions.get(n, 1e3) - t)
                      for n, t in zip(ARM_JOINT_NAMES, cfg["joints"]))
            if err <= ARM_JOINT_TOLERANCE_RAD:
                return label
        return None

    def _envelope_label(self, target_label: str) -> str:
        """Costmap envelope: the larger of the physical and target arm states.

        The costmap may only shrink once the arm is verified tucked, and must
        grow before the arm is commanded to carry.
        """
        physical = self._verified_arm_label()
        if physical is None:
            return "carry"          # unknown arm state -> assume the larger envelope
        if physical == target_label:
            return target_label
        return "carry"

    def _publish_footprint_polygon(self, footprint_str: str):
        """Publish PolygonStamped for RViz visualization."""
        try:
            pts = json.loads(footprint_str)
            poly_msg = PolygonStamped()
            poly_msg.header.stamp = self.get_clock().now().to_msg()
            poly_msg.header.frame_id = "base_link"
            for pt in pts:
                p = Point32()
                p.x = float(pt[0])
                p.y = float(pt[1])
                p.z = 0.0
                poly_msg.polygon.points.append(p)
            self.local_footprint_pub.publish(poly_msg)
            self.footprint_pub.publish(poly_msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing footprint polygon: {e}")

    def _load_models(self):
        """Load pre-trained 3-model Hurdle causal models from pickle artifact."""
        if not os.path.exists(self.model_path):
            self.get_logger().error(f"Model file not found at: {self.model_path}")
            return

        try:
            with open(self.model_path, "rb") as f:
                artifact = pickle.load(f)
            
            version = artifact.get("artifact_version", 1)
            assert version in (2, 3), f"Artifact version mismatch: expected 2 or 3, got {version}. Retrain models!"

            self.safety_model = artifact["safety_model"]
            self.stall_model = artifact["stall_model"]
            self.speed_model = artifact["speed_model"]
            self.feature_cols = artifact["feature_cols"]
            self.support_ranges = artifact.get("support", {})
            self.support = artifact.get("support_model", None)

            # Load ensembles
            self.safety_ensemble = artifact.get("safety_ensemble", [])
            self.stall_ensemble = artifact.get("stall_ensemble", [])
            self.speed_ensemble = artifact.get("speed_ensemble", [])
            self.use_bounds = (len(self.safety_ensemble) > 0 and len(self.stall_ensemble) > 0 and len(self.speed_ensemble) > 0)
            # Collapse each bootstrap member (scaler + linear estimator) into a
            # single affine map, so scoring the ensemble is one matmul per model
            # rather than 50 pipeline calls. The interaction transformer is
            # deterministic and shared, so it is applied once per tick instead.
            self.ens_affine = {}
            if self.use_bounds:
                for name, members, is_logit in (
                    ("safety", self.safety_ensemble, True),
                    ("stall", self.stall_ensemble, True),
                    ("speed", self.speed_ensemble, False),
                ):
                    W, b = [], []
                    for m in members:
                        est = m.named_steps["estimator"]
                        w = np.ravel(est.coef_).astype(float)
                        c = float(np.ravel(est.intercept_)[0])
                        sc = m.named_steps.get("scaler")
                        if sc is not None:
                            s = np.asarray(sc.scale_, dtype=float)
                            mu = np.asarray(sc.mean_, dtype=float)
                            w = w / s
                            c = c - float(w @ mu)
                        W.append(w)
                        b.append(c)
                    self.ens_affine[name] = (np.vstack(W), np.asarray(b), is_logit)
                shapes = {k: v[0].shape for k, v in self.ens_affine.items()}
                self.get_logger().info(f"Ensemble collapsed to affine maps: {shapes}")
            if not self.use_bounds:
                self.get_logger().warn(
                    "No bootstrap ensemble in model artifact; falling back to point "
                    "estimates. Selection will be optimistically biased."
                )

            # Load policy tree
            self.policy_tree = artifact.get("policy_tree", None)
            self.policy_leaf_action = {
                int(k): int(v) for k, v in artifact.get("policy_leaf_action", {}).items()
            }
            if self.policy_tree is not None and not self.policy_leaf_action:
                self.get_logger().error(
                    "policy_tree present without policy_leaf_action -- artifact "
                    "predates the leaf-assignment fix. Retrain before using "
                    "selection_mode=policy_tree."
                )
                self.policy_tree = None

            self.policy_action_cells = artifact.get("policy_action_cells", [])
            self.policy_action_cols = artifact.get("policy_action_cols", [])
            self.policy_risk_cols = artifact.get("policy_risk_cols", [])

            if getattr(self, "policy_action_cols", None):
                uncontrolled = [k for k in PARAM_CANDIDATE_GRID
                                if k not in self.policy_action_cols]
                self.get_logger().info(
                    f"Distilled policy controls {len(self.policy_action_cols)} of "
                    f"{len(PARAM_CANDIDATE_GRID)} knobs; held at grid median: {uncontrolled}")

            w = artifact.get("objective_weights")
            if w is None:
                self.get_logger().error(
                    "Artifact has no objective_weights; it predates the objective "
                    "fix. The policy tree was trained on a different utility than "
                    "this node optimizes. Retrain before using selection_mode=policy_tree.")
                self.policy_tree = None
            else:
                mapping = {"risk_lambda": "risk_lambda", "stall_mu": "stall_mu", "payload_omega": "payload_omega"}
                mismatch = {k: (v, getattr(self, mapping[k]))
                            for k, v in w.items()
                            if abs(v - getattr(self, mapping[k])) > 1e-9}
                if mismatch:
                    self.get_logger().error(
                        f"Objective weight mismatch (artifact vs node): {mismatch}. "
                        "The learned policy optimizes a different objective than this "
                        "node. Refusing to load. Fix the ROS parameters or retrain.")
                    self.policy_tree = None
                    self.models_loaded = False
                    return

            # Gate A4: Strict artifact loading for progress support and scaled hysteresis
            self.progress_support = artifact["progress_support"]
            j_max = float(self.progress_support["max"])
            self.deadband_margin = self.deadband_frac * j_max
            self.arm_switch_cost = self.arm_switch_frac * j_max

            self.risk_cols = [c for c in self.feature_cols if c.startswith("risk__")]
            self.param_cols = [c for c in self.feature_cols if c.startswith("param__")]
            self.models_loaded = True
            self.get_logger().info(f"3-Model Hurdle Causal Artifact v{version} loaded ✓ (j_max={j_max:.3f}m, deadband={self.deadband_margin:.3f}m, use_bounds={self.use_bounds})")
        except Exception as e:
            self.get_logger().error(f"Failed to load causal models: {e}")

    def _generate_candidate_grid(self) -> list:
        """Construct full structured candidate configuration grid across model parameter space."""
        keys, values = zip(*PARAM_CANDIDATE_GRID.items())
        all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        self.get_logger().info(f"Generated {len(all_combinations)} candidate configurations across model parameter space.")
        return all_combinations

    def _admissible(self, cand: dict, r_min: float, r_width: float) -> bool:
        """Gate B1: Feasibility envelope A(R). Verifiable robot geometry and kinematics only."""
        geom = FOOTPRINT_GEOMETRY[float(cand[FOOTPRINT_KEY])]
        width = geom["width"]
        half = geom["width"] / 2.0
        delta = self.clearance_margin_m

        # 1. Pose must physically fit through passage constriction
        if r_width < width + 2.0 * delta:
            return False

        # 2. Inflation must not close free lateral gap or swallow robot footprint
        lateral = (r_width - width) / 2.0 - delta
        radial = r_min - half - delta
        max_allowed_inf = max(self.inflation_floor_m, min(lateral, radial))
        if float(cand[INFLATION_KEY]) > max_allowed_inf + 1e-6:
            return False

        # 3. Kinematic stopping velocity ceiling: v_max <= sqrt(2 * a * stopping)
        stopping = max(0.0, r_min - half - delta)
        v_ceiling = max(MIN_SPEED_LIMIT * MAX_VX_LIMIT / 100.0, math.sqrt(2.0 * self.decel_limit_mps2 * stopping))
        cmd_speed = float(cand[SPEED_LIMIT_KEY]) * MAX_VX_LIMIT / 100.0
        if cmd_speed > v_ceiling + 1e-9:
            return False

        return True

    def _risk_callback(self, msg: Float64MultiArray):
        """Store incoming environment risk vector R_t (8 context features)."""
        data_list = list(msg.data)
        if len(data_list) >= len(RISK_FEATURE_NAMES):
            self.current_risk_vector = data_list[:len(RISK_FEATURE_NAMES)]
        else:
            self.current_risk_vector = data_list
        if hasattr(self, "risk_history"):
            self.risk_history.append((self._sim_time_sec(), data_list))

    def _odom_callback(self, msg: Odometry):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.current_speed = math.sqrt(vx**2 + vy**2)

    def _sim_time_sec(self) -> float:
        """Return current ROS simulation time in seconds."""
        return self.get_clock().now().nanoseconds / 1e9

    def _forecast_risk(self, delta_s: float):
        """Linear extrapolation of R over a short horizon, from the recent buffer.

        Deliberately simple. This is a nuisance component: its job is to carry
        the information the sufficiency test showed exists, not to be a good
        model in its own right. Falls back to R_t when the buffer is short or
        the robot is nearly stationary (extrapolating from noise is worse than
        not extrapolating).
        """
        if delta_s <= 0.0 or len(self.risk_history) < 10:
            return self.current_risk_vector
        if abs(self.current_speed) < 0.05:
            return self.current_risk_vector

        t = np.array([h[0] for h in self.risk_history], dtype=float)
        Y = np.array([h[1] for h in self.risk_history], dtype=float)
        t = t - t[-1]
        keep = t >= -1.5                      # fit on the last 1.5 s only
        t, Y = t[keep], Y[keep]
        if len(t) < 5:
            return self.current_risk_vector

        A = np.vstack([t, np.ones_like(t)]).T
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)   # (2, n_features)
        pred = coef[0] * delta_s + coef[1]

        # Clip each feature to the range seen in the buffer plus a small margin,
        # so a noisy slope cannot extrapolate to an implausible context.
        lo, hi = Y.min(axis=0), Y.max(axis=0)
        span = np.maximum(hi - lo, 1e-3)
        pred_clipped = np.clip(pred, lo - 0.5 * span, hi + 0.5 * span)
        
        return list(pred_clipped[:len(self.current_risk_vector)])

    def _evaluate_utility(self, expected_progress: float, p_collision: float, p_stall: float, is_carry: float) -> float:
        """Compute multiplicative hurdle utility U(c, R_t)."""
        return (
            (1.0 + self.payload_omega * is_carry) * expected_progress
            - (self.risk_lambda * p_collision)
            - (self.stall_mu * p_stall)
        )

    def _prepare_feature_vectors(self, candidates: list, risk_dict: dict):
        """Format candidate dictionary rows and extract feature matrix X_eval."""
        X_eval = []
        candidates_with_eval = []
        for cand in candidates:
            row = cand.copy()
            row.update(risk_dict)
            try:
                feat_vec = [row[col] for col in self.feature_cols]
            except KeyError as exc:
                self.get_logger().error(f"Missing feature column {exc}; skipping tick")
                return None, None
            X_eval.append(feat_vec)
            candidates_with_eval.append(row)
        return candidates_with_eval, np.asarray(X_eval, dtype=np.float64)

    def _transform_only(self, X_arr):
        """Interaction-expanded features, shared by every ensemble member."""
        return np.asarray(
            self.safety_model.named_steps["interaction"].transform(X_arr), dtype=float)

    def _evaluate_positivity(self, X_arr: np.ndarray, candidates_with_eval: list):
        """Evaluate k-NN distance out-of-support status for candidate configurations."""
        if self.support is not None:
            Z = self.support["scaler"].transform(X_arr)
            d_k, _ = self.support["nn"].kneighbors(Z, n_neighbors=self.support["k"])
            d_k = d_k[:, -1]
            oos = d_k > self.support["d_q99"]
            for i, row in enumerate(candidates_with_eval):
                row["support_dist"] = float(d_k[i])
                row["out_of_support"] = bool(oos[i])
            self.last_oos_fraction = float(oos.mean())
        else:
            for row in candidates_with_eval:
                row["support_dist"] = float("nan")
                row["out_of_support"] = False
            self.last_oos_fraction = float("nan")

    def _risk_row_for_policy(self, risk_vector):
        d = {n: (risk_vector[i] if i < len(risk_vector) else 0.0)
             for i, n in enumerate(RISK_FEATURE_NAMES)}
        missing = [c for c in self.policy_risk_cols if c not in d]
        if missing:
            raise KeyError(f"Policy expects risk features not on /risk_state: {missing}")
        return np.array([[d[c] for c in self.policy_risk_cols]], dtype=float)

    def _solve_by_policy_tree(self, risk_vector):
        """O(depth) selection. The tree encodes the DR-optimal action per context."""
        r = self._risk_row_for_policy(risk_vector)
        leaf = int(self.policy_tree.apply(r)[0])
        action_idx = self.policy_leaf_action.get(leaf)
        if action_idx is None:
            self.get_logger().error(
                f"Leaf {leaf} missing from policy_leaf_action; falling back to "
                "bounded argmax for this tick."
            )
            return None
        cell = self.policy_action_cells[int(action_idx)]
        row = dict(zip(self.policy_action_cols, cell))
        # Knobs the distilled policy does not control are held at the grid
        # median, so every emitted configuration is a member of C.
        for key, levels in PARAM_CANDIDATE_GRID.items():
            if key not in row:
                row[key] = float(np.median(levels))
        row["selection_reason"] = f"POLICY_TREE (leaf={leaf})"

        # Evaluate model predictions for this single row to populate log metrics
        risk_dict = {name: risk_vector[i] if i < len(risk_vector) else 0.0 for i, name in enumerate(RISK_FEATURE_NAMES)}
        r_min = float(risk_dict.get("risk__r_min", 3.0))
        r_width = float(risk_dict.get("risk__r_width", 5.0))
        if not self._admissible(row, r_min, r_width):
            row = self._project_to_admissible(row, r_min, r_width)

        candidates_with_eval, X_arr = self._prepare_feature_vectors([row], risk_dict)
        if candidates_with_eval is None:
            return row

        self._evaluate_positivity(X_arr, candidates_with_eval)
        eval_row = candidates_with_eval[0]
        
        p_c = float(self.safety_model.predict_proba(X_arr)[0, 1])
        p_st = float(self.stall_model.predict_proba(X_arr)[0, 1])
        max_prog_support = float(self.progress_support["max"])
        e_sp = float(np.clip(self.speed_model.predict(X_arr)[0], 0.0, max_prog_support))
        expected_progress = (1.0 - p_st) * e_sp
        
        is_carry = 1.0 if float(eval_row.get(FOOTPRINT_KEY, 0.0)) == 1.0 else 0.0
        utility_raw = self._evaluate_utility(expected_progress, p_c, p_st, is_carry)
        
        eval_row["p_risk"] = p_c
        eval_row["p_risk_ucb"] = p_c
        eval_row["p_risk_sd"] = 0.0
        eval_row["p_stall"] = p_st
        eval_row["p_stall_ucb"] = p_st
        eval_row["e_speed"] = e_sp
        eval_row["j_progress"] = expected_progress
        eval_row["utility_raw"] = utility_raw
        eval_row["utility"] = utility_raw
        eval_row["utility_point"] = utility_raw

        # Enforce arm switch hysteresis to suppress unnecessary arm motion on minor risk noise
        ref_arm = getattr(self, "target_arm_label", self.current_arm_label)
        cand_arm = "carry" if float(row.get(FOOTPRINT_KEY, 0.0)) == 1.0 else "tucked"
        if ref_arm and cand_arm != ref_arm:
            curr_fp_val = 1.0 if ref_arm == "carry" else 0.0
            row_curr = row.copy()
            row_curr[FOOTPRINT_KEY] = curr_fp_val
            _, X_curr = self._prepare_feature_vectors([row_curr], risk_dict)
            if X_curr is not None:
                pc_curr = float(self.safety_model.predict_proba(X_curr)[0, 1])
                pst_curr = float(self.stall_model.predict_proba(X_curr)[0, 1])
                esp_curr = float(np.clip(self.speed_model.predict(X_curr)[0], 0.0, max_prog_support))
                prog_curr = (1.0 - pst_curr) * esp_curr
                u_curr = self._evaluate_utility(prog_curr, pc_curr, pst_curr, curr_fp_val)
                if (utility_raw - u_curr) < self.arm_switch_cost:
                    eval_row[FOOTPRINT_KEY] = curr_fp_val
                    eval_row["selection_reason"] += f" (retained {ref_arm} via hysteresis)"
            
        return eval_row

    def set_goal(self, goal_pose):
        """Called by the runner at trial start. Enables terminal-phase detection."""
        self.goal_xy = (float(goal_pose["x"]), float(goal_pose["y"]))
        self.in_terminal_phase = False
        self.get_logger().info(f"Goal set for terminal-phase gating: {self.goal_xy}")

    def _distance_to_goal(self):
        """Euclidean distance base_link -> goal in the map frame, or None."""
        if self.goal_xy is None:
            return None
        try:
            tf = self.tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05))
        except Exception:
            return None
        dx = tf.transform.translation.x - self.goal_xy[0]
        dy = tf.transform.translation.y - self.goal_xy[1]
        return math.hypot(dx, dy)

    def _terminal_config(self):
        """Fixed configuration for the goal-alignment phase.

        The effect models were fitted on fixed-horizon TRAVERSAL probes with a
        progress outcome. Terminal alignment optimizes pose convergence, not
        progress, so the learned utility does not apply. Freeze at the smallest
        envelope, the inflation floor (maximum free space around the goal cell)
        and enough speed budget that Nav2's percentage speed limit does not
        throttle angular velocity. Identical for every strategy.
        """
        return {
            SPEED_LIMIT_KEY: TERMINAL_SPEED_LIMIT_PCT,
            VX_STD_KEY: 0.15,
            CONSTRAINT_KEY: 2.0,
            COST_WEIGHT_KEY: 1.0,
            PATH_ALIGN_KEY: 4.0,
            INFLATION_KEY: self.inflation_floor_m,
            FOOTPRINT_KEY: 0.0,
            "selection_reason": "TERMINAL_PHASE (fixed, not tuned)",
        }

    def _project_to_admissible(self, row: dict, r_min: float, r_width: float) -> dict:
        """Clamp a policy action onto A(R) instead of re-solving.

        A(R) is a hard geometric constraint, not a preference. Projection keeps
        the learned arm decision and tightens only the knobs the constraint
        binds, in O(1). Re-solving called the 2304-candidate scorer, which at
        ~150 ms/tick starved the /risk_state callback near obstacles.
        """
        out = dict(row)
        geom = FOOTPRINT_GEOMETRY[float(out[FOOTPRINT_KEY])]
        d = self.clearance_margin_m

        if r_width < geom["width"] + 2.0 * d:
            out[FOOTPRINT_KEY] = 0.0
            geom = FOOTPRINT_GEOMETRY[0.0]
            out["selection_reason"] += " [proj:envelope->tucked]"
        half = geom["width"] / 2.0

        lateral = (r_width - geom["width"]) / 2.0 - d
        radial = r_min - half - d
        max_inf = max(self.inflation_floor_m, min(lateral, radial))
        ok_inf = [v for v in PARAM_CANDIDATE_GRID[INFLATION_KEY] if v <= max_inf + 1e-6]
        tgt_inf = max(ok_inf) if ok_inf else self.inflation_floor_m
        if float(out[INFLATION_KEY]) > tgt_inf:
            out[INFLATION_KEY] = tgt_inf
            out["selection_reason"] += f" [proj:inf->{tgt_inf:.2f}]"

        stopping = max(0.0, r_min - half - d)
        v_ceiling = max(MIN_SPEED_LIMIT * MAX_VX_LIMIT / 100.0,
                        math.sqrt(2.0 * self.decel_limit_mps2 * stopping))
        pct_ceiling = 100.0 * v_ceiling / MAX_VX_LIMIT
        ok_spd = [v for v in PARAM_CANDIDATE_GRID[SPEED_LIMIT_KEY] if v <= pct_ceiling + 1e-6]
        tgt_spd = max(ok_spd) if ok_spd else MIN_SPEED_LIMIT
        if float(out[SPEED_LIMIT_KEY]) > tgt_spd:
            out[SPEED_LIMIT_KEY] = tgt_spd
            out["selection_reason"] += f" [proj:spd->{tgt_spd:.0f}]"

        self.n_projected_ticks += 1
        return out

    def _publish_speed_limit(self, pct: float):
        """Publish ROS 2 SpeedLimit message to global /speed_limit topic."""
        if pct is None:
            return
        msg = SpeedLimit()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.percentage = True
        msg.speed_limit = float(pct)
        self.speed_limit_pub.publish(msg)

    def _tuning_loop(self):
        t0 = time.perf_counter()
        self._tuning_loop_impl()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.max_tick_ms = max(self.max_tick_ms, dt_ms)
        if dt_ms > self.tick_period * 1000.0:
            self.n_tick_overruns += 1
            self.get_logger().warn(
                f"Tuning tick {dt_ms:.0f} ms > {self.tick_period*1000:.0f} ms budget "
                f"(overruns={self.n_tick_overruns}).", throttle_duration_sec=5.0)

    def _tuning_loop_impl(self):
        """Main periodic optimization loop."""
        if not self.models_loaded:
            self._load_models()
            if not self.models_loaded:
                return

        # Continuously maintain speed limit on global /speed_limit topic
        if self.current_speed_limit_pct is not None:
            self._publish_speed_limit(self.current_speed_limit_pct)

        if self.current_risk_vector is None:
            self.get_logger().warn("Waiting for /risk_state message...", throttle_duration_sec=5.0)
            return

        if self.risk_history:
            age = self._sim_time_sec() - self.risk_history[-1][0]
            if age > 0.5:
                self.n_stale_ticks += 1
                self.get_logger().error(
                    f"Risk vector {age:.2f}s stale; skipping tick.",
                    throttle_duration_sec=2.0)
                return

        if self.current_speed < 0.02 and self.last_applied_config is None:
            self.get_logger().info("Idle robot / stationary; tuning loop gated.", throttle_duration_sec=10.0)
            return

        d_goal = self._distance_to_goal()
        # Schmitt trigger: enter at terminal_radius, leave only past 1.5x it, so
        # backing off during alignment does not restart tuning.
        if d_goal is None:
            entering = False
        elif self.in_terminal_phase:
            entering = d_goal <= self.terminal_radius * 1.5
        else:
            entering = d_goal <= self.terminal_radius

        if entering != self.in_terminal_phase:
            self.in_terminal_phase = entering
            self.get_logger().info(
                f"{'ENTER' if entering else 'EXIT'} terminal phase "
                f"(d_goal={d_goal:.2f} m, radius={self.terminal_radius:.2f} m)")
        if self.in_terminal_phase:
            self.n_terminal_ticks += 1

        r_eval = self._forecast_risk(self.anticipation_delta)
        # Store for apply configuration decision log
        self.last_forecast_risk = r_eval

        if self.selection_mode == "policy_tree" and getattr(self, "policy_tree", None) is not None:
            best_config = self._solve_by_policy_tree(r_eval)
        else:
            best_config = self.solve_optimal_configuration(r_eval)

        if best_config is not None:
            self._apply_configuration(best_config)

    def solve_optimal_configuration(self, risk_vector: list) -> dict:
        """
        Solve C_t^* = argmax_{c in A(R)} U(c, R_t) subject to P_collision <= p_max.
        """
        t_solve_start = time.perf_counter()
        sim_time = self._sim_time_sec()
        risk_dict = {name: risk_vector[i] if i < len(risk_vector) else 0.0 for i, name in enumerate(RISK_FEATURE_NAMES)}

        r_min = float(risk_dict.get("risk__r_min", 3.0))
        r_width = float(risk_dict.get("risk__r_width", 5.0))

        self.n_ticks += 1

        # Gate B1: Filter candidates via geometric and kinematic feasibility envelope A(R)
        feasible_candidates = [c for c in self.candidates_df if self._admissible(c, r_min, r_width)]
        if not feasible_candidates:
            feasible_candidates = [c for c in self.candidates_df if float(c[FOOTPRINT_KEY]) == 0.0 and float(c[SPEED_LIMIT_KEY]) == MIN_SPEED_LIMIT]

        # P1.3: Envelope-Only Ablation Mode (holds software knobs fixed, varies envelope gating)
        if self.envelope_only:
            t_select_start = time.perf_counter()

            def _software_distance(c):
                return sum(abs(float(c[k]) - float(v)) for k, v in ENVELOPE_ONLY_SOFTWARE.items())

            best_row = min(feasible_candidates, key=lambda c: (_software_distance(c), -float(c[FOOTPRINT_KEY]))).copy()
            best_row["selection_reason"] = "ENVELOPE_ONLY_ABLATION"
            for k in ("p_risk", "p_risk_ucb", "p_risk_sd", "p_stall", "p_stall_ucb", "utility_raw", "utility_effective", "utility_point", "t_infer_ms"):
                best_row[k] = 0.0
            best_row["e_speed"] = float(best_row.get(SPEED_LIMIT_KEY, MIN_SPEED_LIMIT)) * MAX_VX_LIMIT / 100.0
            best_row["j_progress"] = best_row["e_speed"]
            best_row["j_progress_lcb"] = best_row["e_speed"]
            best_row["out_of_support"] = False
            best_row["t_select_ms"] = (time.perf_counter() - t_select_start) * 1000.0
            self.last_oos_fraction = 0.0
            return best_row

        # Gate A4 & B2: Model inference and multiplicative Hurdle Utility scoring
        t_infer_start = time.perf_counter()
        candidates_with_eval, X_arr = self._prepare_feature_vectors(feasible_candidates, risk_dict)
        if candidates_with_eval is None:
            return None

        # Positivity diagnostics
        self._evaluate_positivity(X_arr, candidates_with_eval)

        max_prog_support = float(self.progress_support["max"])

        # Point predictions first
        if hasattr(self.safety_model, "predict_proba"):
            p_point_all = np.asarray(self.safety_model.predict_proba(X_arr))[:, 1]
        else:
            p_point_all = np.zeros(len(X_arr))

        if hasattr(self.stall_model, "predict_proba"):
            p_stall_point_all = np.asarray(self.stall_model.predict_proba(X_arr))[:, 1]
        else:
            p_stall_point_all = np.zeros(len(X_arr))

        if hasattr(self.speed_model, "predict"):
            e_speed_point_all = np.asarray(np.clip(self.speed_model.predict(X_arr), 0.0, max_prog_support))
        else:
            e_speed_point_all = np.ones(len(X_arr))

        j_point_all = (1.0 - p_stall_point_all) * e_speed_point_all

        if self.use_bounds:
            if len(candidates_with_eval) > 100:
                u_pts = [
                    self._evaluate_utility(j_point_all[k], p_point_all[k], p_stall_point_all[k], 1.0 if float(row.get(FOOTPRINT_KEY, 0.0)) == 1.0 else 0.0)
                    for k, row in enumerate(candidates_with_eval)
                ]
                top_indices = np.argsort(u_pts)[::-1][:100]
                X_eval_sub = X_arr[top_indices]
                cands_sub = [candidates_with_eval[k] for k in top_indices]
            else:
                top_indices = np.arange(len(candidates_with_eval))
                X_eval_sub = X_arr
                cands_sub = candidates_with_eval

            candidates_with_eval = cands_sub
            X_arr = X_eval_sub

            # B bootstrap predictions -> (B, n_candidates), one matmul per model
            Z = self._transform_only(X_arr)

            Ws, bs, _ = self.ens_affine["safety"]
            P = 1.0 / (1.0 + np.exp(-np.clip(Ws @ Z.T + bs[:, None], -30.0, 30.0)))

            Ws, bs, _ = self.ens_affine["stall"]
            P_stall = 1.0 / (1.0 + np.exp(-np.clip(Ws @ Z.T + bs[:, None], -30.0, 30.0)))

            Ws, bs, _ = self.ens_affine["speed"]
            E_speed = np.clip(Ws @ Z.T + bs[:, None], 0.0, max_prog_support)
            J = (1.0 - P_stall) * E_speed

            p_point = P.mean(axis=0)
            p_stall_point = P_stall.mean(axis=0)
            e_speed_point = E_speed.mean(axis=0)
            j_point = J.mean(axis=0)

            p_ucb = np.quantile(P, 1.0 - self.alpha, axis=0)
            p_stall_ucb = np.quantile(P_stall, 1.0 - self.alpha, axis=0)
            j_lcb = np.quantile(J, self.alpha, axis=0)
            p_sd = P.std(axis=0)
        else:
            p_point = p_point_all
            p_stall_point = p_stall_point_all
            e_speed_point = e_speed_point_all
            j_point = j_point_all
            p_ucb = p_point
            p_stall_ucb = p_stall_point
            j_lcb = j_point
            p_sd = np.zeros_like(p_point)

            j_point = (1.0 - p_stall_point) * e_speed_point
            
            p_ucb = p_point
            p_stall_ucb = p_stall_point
            j_lcb = j_point
            p_sd = np.zeros_like(p_point)

        # Gate B2: Compute expected and pessimistic utilities
        for i, row in enumerate(candidates_with_eval):
            pc = p_point[i]
            pc_ucb = p_ucb[i]
            pst = p_stall_point[i]
            pst_ucb = p_stall_ucb[i]
            sp = e_speed_point[i]
            jp = j_point[i]
            jp_lcb = j_lcb[i]
            
            is_carry = 1.0 if float(row.get(FOOTPRINT_KEY, 0.0)) == 1.0 else 0.0
            
            utility_raw = self._evaluate_utility(jp_lcb, pc_ucb, pst_ucb, is_carry)
            utility_point = self._evaluate_utility(jp, pc, pst, is_carry)
            
            row["p_risk"] = pc
            row["p_risk_ucb"] = pc_ucb
            row["p_risk_sd"] = float(p_sd[i])
            row["p_stall"] = pst
            row["p_stall_ucb"] = pst_ucb
            row["e_speed"] = sp
            row["j_progress"] = jp
            row["j_progress_lcb"] = jp_lcb
            row["utility_raw"] = utility_raw
            row["utility"] = utility_raw
            row["utility_point"] = utility_point

        t_infer_ms = (time.perf_counter() - t_infer_start) * 1000.0
        t_select_start = time.perf_counter()

        # Hard safety backstop filter: P_collision <= p_max
        safe_candidates = [c for c in candidates_with_eval if c["p_risk_ucb"] <= self.p_max]

        if safe_candidates:
            ref_arm_label = getattr(self, "target_arm_label", self.current_arm_label)
            curr_arm_val = 1.0 if ref_arm_label == "carry" else 0.0
            n_knobs = float(len(self.param_cols))
            
            for c in safe_candidates:
                u_eff = c["utility_raw"]
                cand_arm = 1.0 if float(c.get(FOOTPRINT_KEY, 0.0)) == 1.0 else 0.0
                
                if self.last_applied_config is not None:
                    if cand_arm != curr_arm_val:
                        u_eff -= self.arm_switch_cost
                        
                    n_changed = sum(
                        1 for k in self.param_cols
                        if abs(float(c.get(k, 0)) - float(self.last_applied_config.get(k, 0))) >= 1e-4
                    )
                    if n_changed > 0:
                        u_eff -= self.deadband_margin * (float(n_changed) / n_knobs)
                        
                c["utility_effective"] = u_eff

            best_row = max(
                safe_candidates,
                key=lambda c: (
                    c.get("utility_effective", c["utility_raw"]),
                    -float(c.get(SPEED_LIMIT_KEY, 95.0))
                )
            ).copy()
            best_row["selection_reason"] = (
                f"HURDLE_UTILITY_OPTIMAL (U_pess={best_row['utility_raw']:.3f}, U_point={best_row.get('utility_point', 0.0):.3f}, E[J_lcb]={best_row['j_progress_lcb']:.2f}m, "
                f"P_coll_ucb={best_row['p_risk_ucb']:.4f}, P_stall_ucb={best_row['p_stall_ucb']:.3f})"
            )
        else:
            best_row = min(candidates_with_eval, key=lambda c: (c["p_risk_ucb"], c["p_stall_ucb"], -c["utility_raw"])).copy()
            best_row["selection_reason"] = f"FALLBACK_SAFEST (no candidate <= {self.p_max:.2f}, safest P_ucb={best_row['p_risk_ucb']:.4f})"

        best_row["t_infer_ms"] = t_infer_ms
        best_row["t_select_ms"] = (time.perf_counter() - t_select_start) * 1000.0
        return best_row

    def _set_param_async(self, node_key: str, param_name: str, value):
        """P0.4: Set a Nav2 parameter and record whether the service accepted it."""
        if self.dry_run:
            self.get_logger().info(f"[DRY RUN] {node_key}.{param_name} = {value}")
            return

        client = self.param_clients.get(node_key)
        if client is None:
            return
        if not client.service_is_ready() and not client.wait_for_service(timeout_sec=0.2):
            self._record_param_failure(node_key, param_name, "service_not_ready")
            return

        req = SetParameters.Request()
        p = Parameter()
        p.name = param_name
        if isinstance(value, bool):
            p.value = ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=value)
        elif isinstance(value, int):
            p.value = ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=int(value))
        elif isinstance(value, str):
            p.value = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=value)
        else:
            p.value = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(value))
        req.parameters.append(p)

        def _done(fut):
            try:
                resp = fut.result()
                res = resp.results[0] if resp.results else None
                if res is None or not res.successful:
                    reason = res.reason if res is not None else "empty_response"
                    self._record_param_failure(node_key, param_name, reason)
            except Exception as exc:
                self._record_param_failure(node_key, param_name, repr(exc))

        client.call_async(req).add_done_callback(_done)

    def _record_param_failure(self, node_key: str, param_name: str, reason: str):
        key = f"{node_key}.{param_name}"
        self.n_param_set_failures += 1
        self.param_set_failures[key] = self.param_set_failures.get(key, 0) + 1
        self.get_logger().error(f"Parameter set REJECTED: {key} ({reason})")

    def _move_arm_to_label_async(self, label: str) -> bool:
        """Command the physical arm. Returns False if the goal could not be sent."""
        if self.dry_run:
            self.get_logger().info(f"[DRY RUN] Would move physical arm to '{label}' pose")
            return True

        cfg = ARM_CONFIGS.get(label)
        if not cfg or not cfg.get("joints"):
            return False

        if not self.arm_action_client.wait_for_server(timeout_sec=0.5):
            self.get_logger().error(
                "Arm action server '/arm_controller/follow_joint_trajectory' "
                "unavailable; arm command NOT sent.")
            return False

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = list(ARM_JOINT_NAMES)
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in cfg["joints"]]
        pt.time_from_start.sec = 4
        pt.time_from_start.nanosec = 0
        goal_msg.trajectory.points = [pt]

        sim_now = self._sim_time_sec()
        self.get_logger().info(
            f"Sending arm trajectory goal -> '{label}' (sim_time={sim_now:.2f}s)")

        def _goal_response_callback(future):
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error(
                    f"Arm goal '{label}' REJECTED; will retry next tick.")
                self.arm_transition_until_time = 0.0
            else:
                self.current_arm_label = label

        self.arm_action_client.send_goal_async(goal_msg).add_done_callback(
            _goal_response_callback)
        return True

    def _apply_configuration(self, target_config: dict):
        """P0.2d: Apply target parameters and verified costmap footprints asynchronously."""
        speed_limit_pct = float(target_config["param__controller_server__speed_limit_pct"])
        vx_std = float(target_config["param__controller_server__FollowPath.vx_std"])
        constraint_weight = float(target_config["param__controller_server__FollowPath.ConstraintCritic.cost_weight"])
        cost_weight = float(target_config["param__controller_server__FollowPath.CostCritic.cost_weight"])
        path_align_weight = float(target_config["param__controller_server__FollowPath.PathAlignCritic.cost_weight"])
        inflation = float(target_config["param__local_costmap__inflation_layer.inflation_radius"])
        footprint_val = float(target_config["param__local_costmap__footprint"])
        arm_label = "carry" if footprint_val >= 0.5 else "tucked"

        sim_now = self._sim_time_sec()
        if sim_now < self.arm_transition_until_time:
            speed_limit_pct = min(speed_limit_pct, 30.0)

        # P0.2d: Footprint tracks physical envelope verified from /joint_states
        envelope_label = self._envelope_label(arm_label)
        target_footprint = ARM_CONFIGS[envelope_label]["footprint"]

        # P0.2e: Track verified physical arm state samples
        self.total_samples += 1
        self.verified_arm_label = self._verified_arm_label()
        if self.verified_arm_label == "carry":
            self.carry_samples += 1

        last_cfg = self.last_applied_config or {}

        any_param_changed = (
            (speed_limit_pct != last_cfg.get("param__controller_server__speed_limit_pct")) or
            (vx_std != last_cfg.get("param__controller_server__FollowPath.vx_std")) or
            (constraint_weight != last_cfg.get("param__controller_server__FollowPath.ConstraintCritic.cost_weight")) or
            (cost_weight != last_cfg.get("param__controller_server__FollowPath.CostCritic.cost_weight")) or
            (path_align_weight != last_cfg.get("param__controller_server__FollowPath.PathAlignCritic.cost_weight")) or
            (inflation != last_cfg.get("param__local_costmap__inflation_layer.inflation_radius")) or
            (envelope_label != self.applied_envelope_label)
        )

        if any_param_changed:
            self.n_config_switches += 1

        self.current_speed_limit_pct = speed_limit_pct
        self._publish_speed_limit(speed_limit_pct)

        if vx_std != last_cfg.get("param__controller_server__FollowPath.vx_std"):
            self._set_param_async("controller_server", "FollowPath.vx_std", vx_std)

        if constraint_weight != last_cfg.get("param__controller_server__FollowPath.ConstraintCritic.cost_weight"):
            self._set_param_async("controller_server", "FollowPath.ConstraintCritic.cost_weight", constraint_weight)

        if cost_weight != last_cfg.get("param__controller_server__FollowPath.CostCritic.cost_weight"):
            self._set_param_async("controller_server", "FollowPath.CostCritic.cost_weight", cost_weight)

        if path_align_weight != last_cfg.get("param__controller_server__FollowPath.PathAlignCritic.cost_weight"):
            self._set_param_async("controller_server", "FollowPath.PathAlignCritic.cost_weight", path_align_weight)

        if inflation != last_cfg.get("param__local_costmap__inflation_layer.inflation_radius"):
            self._set_param_async("local_costmap", "inflation_layer.inflation_radius", inflation)

        # P0.2d: Local costmap footprint tracks verified physical envelope; global footprint managed per-strategy
        if envelope_label != self.applied_envelope_label:
            self._set_param_async("local_costmap", "footprint", target_footprint)
            self._publish_footprint_polygon(target_footprint)
            self.applied_envelope_label = envelope_label

        # Command the arm when the decision changes, OR when a previously
        # commanded target was never physically reached. Without the second
        # condition a dropped goal or a stale cross-episode target leaves the
        # arm extended for the rest of the run with no retry.
        # Arm chatter guard. r_width crossing a single tree split (6.05 m) flips
        # the leaf between a tucked and a carry action, and while rotating in place
        # that boundary is crossed repeatedly. Require the new label to persist and
        # a minimum dwell since the last switch before committing 4 s of arm motion.
        if arm_label != self.target_arm_label:
            if arm_label == self.pending_arm_label:
                self.pending_arm_count += 1
            else:
                self.pending_arm_label = arm_label
                self.pending_arm_count = 1
        else:
            self.pending_arm_label = None
            self.pending_arm_count = 0

        persisted = self.pending_arm_count >= self.arm_persist_ticks
        dwell_ok = (sim_now - self.last_arm_switch_time) >= self.arm_switch_dwell

        transition_done = sim_now >= self.arm_transition_until_time
        target_changed = (arm_label != self.target_arm_label) and persisted and dwell_ok
        target_unmet = (
            transition_done
            and self.verified_arm_label is not None
            and self.verified_arm_label != self.target_arm_label
        )
        if transition_done and (target_changed or target_unmet):
            if target_changed:
                self.n_arm_switches += 1
                self.last_arm_switch_time = sim_now
                self.pending_arm_label = None
                self.pending_arm_count = 0
                self.get_logger().info(
                    f"Decision: arm '{self.target_arm_label}' -> '{arm_label}'")
            else:
                self.n_arm_retries += 1
                self.get_logger().warn(
                    f"Arm target '{self.target_arm_label}' not reached "
                    f"(verified '{self.verified_arm_label}'); re-issuing "
                    f"(retry {self.n_arm_retries}).")
            self.target_arm_label = arm_label
            self.arm_transition_until_time = sim_now + 4.0
            if not self._move_arm_to_label_async(arm_label):
                self.arm_transition_until_time = 0.0

        # P Record tick trace in decision_log
        self.decision_log.append({
            "t": sim_now,
            "risk": list(self.current_risk_vector or []),
            "risk_forecast": list(getattr(self, "last_forecast_risk", [])),
            "speed_limit_pct": speed_limit_pct,
            "vx_std": vx_std,
            "constraint_weight": constraint_weight,
            "cost_weight": cost_weight,
            "path_align_weight": path_align_weight,
            "inflation": inflation,
            "arm_target": arm_label,
            "arm_verified": self.verified_arm_label,
            "envelope_applied": self.applied_envelope_label,
            "p_risk": float(target_config.get("p_risk", float("nan"))),
            "p_risk_ucb": float(target_config.get("p_risk_ucb", float("nan"))),
            "p_stall": float(target_config.get("p_stall", float("nan"))),
            "p_stall_ucb": float(target_config.get("p_stall_ucb", float("nan"))),
            "utility": float(target_config.get("utility_raw", float("nan"))),
            "utility_point": float(target_config.get("utility_point", float("nan"))),
            "reason": target_config.get("selection_reason", ""),
            "out_of_support": bool(target_config.get("out_of_support", False)),
            "terminal_phase": bool(self.in_terminal_phase),
            "d_goal": self._distance_to_goal(),
        })

        target_config["_applied_speed_limit_pct"] = speed_limit_pct
        self.last_applied_config = target_config.copy()

    def reset_trial_metrics(self):
        """Reset per-episode sample metrics AND arm bookkeeping.

        The runner re-poses the arm between episodes. A target_arm_label carried
        over from the previous episode makes `arm_label != target_arm_label`
        false for the whole episode, so the arm is never commanded again and the
        tuner degenerates to a fixed configuration. Re-anchor on the physically
        verified joint state, falling back to whatever the runner set.
        """
        self.total_samples = 0
        self.carry_samples = 0
        self.n_arm_switches = 0
        self.n_arm_retries = 0
        self.n_config_switches = 0
        self.n_ticks = 0
        self.n_stale_ticks = 0
        self.n_tick_overruns = 0
        self.max_tick_ms = 0.0
        self.n_out_of_support = 0
        self.out_of_support_features = {}
        self.decision_log = []
        self.applied_envelope_label = None
        self.n_param_set_failures = 0
        self.param_set_failures = {}

        verified = self._verified_arm_label()
        if verified is not None:
            self.current_arm_label = verified
        self.target_arm_label = self.current_arm_label
        self.verified_arm_label = verified
        self.arm_transition_until_time = 0.0
        self.last_applied_config = None
        self.last_arm_switch_time = -1e9
        self.pending_arm_label = None
        self.pending_arm_count = 0
        self.in_terminal_phase = False
        self.n_terminal_ticks = 0
        self.n_projected_ticks = 0

    def get_carry_fraction(self) -> float:
        """Return proportion of tuning samples spent in carry pose during trial."""
        if self.total_samples == 0:
            return 1.0 if self.current_arm_label == "carry" else 0.0
        return float(self.carry_samples) / float(self.total_samples)

    def get_out_of_support_fraction(self) -> float:
        """Gate A2: Return proportion of decision ticks outside training context support."""
        if self.n_ticks == 0:
            return 0.0
        return float(self.n_out_of_support) / float(self.n_ticks)

    @property
    def carry_fraction(self) -> float:
        return self.get_carry_fraction()

    @property
    def out_of_support_fraction(self) -> float:
        return self.get_out_of_support_fraction()


def main(args=None):
    rclpy.init(args=args)
    node = OnlineCausalTunerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
