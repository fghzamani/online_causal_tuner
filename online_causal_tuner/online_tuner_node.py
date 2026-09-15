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
import warnings

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
from geometry_msgs.msg import PolygonStamped, Point32, PoseStamped
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

# Gate B5: Optimized Candidate Grid Space (9600 grid points)
PARAM_CANDIDATE_GRID = {
    "param__controller_server__speed_limit_pct": [30.0, 50.0, 70.0, 90.0, 95.0],
    "param__controller_server__FollowPath.vx_std": [0.15, 0.35, 0.40],
    "param__controller_server__FollowPath.ConstraintCritic.cost_weight": [0.5, 6.0],
    "param__controller_server__FollowPath.CostCritic.cost_weight": [1.0, 3.81, 5.0, 12.0, 20.0],
    "param__controller_server__FollowPath.PathAlignCritic.cost_weight": [4.0, 15.0, 30.0, 32.0],
    "param__local_costmap__inflation_layer.inflation_radius": [0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    "param__local_costmap__footprint": [0.0, 1.0],  # 0.0 = tucked, 1.0 = carry
}


class OnlineCausalTunerNode(Node):
    _instance = None

    def __init__(self):
        super().__init__("online_causal_tuner")
        OnlineCausalTunerNode._instance = self

        # Gate A1: Declare ROS parameters matching single YAML source of truth
        self.declare_parameter("model_path", "/home/forough/phd_projects/online_tuner/src/online_causal_tuner/models/causal_tuner_models.pkl")
        default_env_json = "/home/forough/phd_projects/online_tuner/src/online_causal_tuner/models/envelope_constants.json"
        if not os.path.exists(default_env_json):
            pkg_env_json = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "envelope_constants.json")
            if os.path.exists(pkg_env_json):
                default_env_json = pkg_env_json
        self.declare_parameter("envelope_constants_path", default_env_json if os.path.exists(default_env_json) else "")
        self.declare_parameter("risk_threshold_p_max", 0.20)
        self.declare_parameter("risk_penalty_lambda", 10.0)
        self.declare_parameter("stall_penalty_mu", 2.0)
        self.declare_parameter("payload_value_omega", 0.8333333333333334)
        self.declare_parameter("utility_deadband_frac", 0.025)
        self.declare_parameter("arm_switch_cost_frac", 0.25)
        self.declare_parameter("clearance_margin_m", 0.10)
        self.declare_parameter("decel_limit_mps2", 1.50)
        self.declare_parameter("lateral_tracking_tau_s", 0.35)
        self.declare_parameter("inflation_floor_m", 0.45)
        self.declare_parameter("tuning_rate_hz", 5.0)
        self.declare_parameter("dry_run", False)
        self.declare_parameter("envelope_only", False)
        self.declare_parameter("pessimism_alpha", 0.20)
        self.declare_parameter("anticipation_delta_s", 4.0)
        self.declare_parameter("terminal_radius_m", 1.0)
        self.declare_parameter("placement_radius_m", 3.0)
        self.declare_parameter("arm_switch_dwell_s", 2.0)
        self.declare_parameter("arm_switch_persist_ticks", 5)
        self.declare_parameter("envelope_hysteresis_m", 0.10)

        self.model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.p_max = self.get_parameter("risk_threshold_p_max").get_parameter_value().double_value
        self.risk_lambda = self.get_parameter("risk_penalty_lambda").get_parameter_value().double_value
        self.stall_mu = self.get_parameter("stall_penalty_mu").get_parameter_value().double_value
        self.payload_omega = self.get_parameter("payload_value_omega").get_parameter_value().double_value
        self.deadband_frac = self.get_parameter("utility_deadband_frac").get_parameter_value().double_value
        self.arm_switch_frac = self.get_parameter("arm_switch_cost_frac").get_parameter_value().double_value
        self.clearance_margin_m = self.get_parameter("clearance_margin_m").get_parameter_value().double_value
        self.decel_limit_mps2 = self.get_parameter("decel_limit_mps2").get_parameter_value().double_value
        self.lateral_tracking_tau_s = self.get_parameter("lateral_tracking_tau_s").get_parameter_value().double_value
        self.inflation_floor_m = self.get_parameter("inflation_floor_m").get_parameter_value().double_value
        self.tuning_rate = self.get_parameter("tuning_rate_hz").get_parameter_value().double_value
        self.dry_run = self.get_parameter("dry_run").get_parameter_value().bool_value
        self.envelope_only = self.get_parameter("envelope_only").get_parameter_value().bool_value
        self.alpha = self.get_parameter("pessimism_alpha").get_parameter_value().double_value
        self.anticipation_delta = self.get_parameter("anticipation_delta_s").get_parameter_value().double_value
        self.terminal_radius = self.get_parameter("terminal_radius_m").get_parameter_value().double_value
        self.placement_radius = self.get_parameter("placement_radius_m").get_parameter_value().double_value
        self.in_placement_phase = False
        self.arm_switch_dwell = self.get_parameter("arm_switch_dwell_s").get_parameter_value().double_value
        self.arm_persist_ticks = self.get_parameter("arm_switch_persist_ticks").get_parameter_value().integer_value
        self.envelope_hysteresis_m = self.get_parameter("envelope_hysteresis_m").get_parameter_value().double_value

        self.envelope_constants_path = self.get_parameter("envelope_constants_path").get_parameter_value().string_value
        self._load_envelope_constants()

        self.carry_distance_m = 0.0
        self.total_distance_m = 0.0
        self.last_odom_pose = None
        self.pending_inflation = None
        self.pending_inflation_count = 0

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
        # What the policy *chose* before A(R) clamped it. Distinct from
        # target_arm_label, which records what was actually applied. A
        # geometry-forced retraction must not become the hysteresis reference.
        self.preferred_arm_label = "carry"
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
        self.goal_sub = self.create_subscription(
            PoseStamped, "/goal_pose", self._goal_pose_callback, 10,
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

    ENVELOPE_CONSTANT_NAMES = (
        "clearance_margin_m",
        "decel_limit_mps2",
        "inflation_floor_m",
        "envelope_hysteresis_m",
        "lateral_tracking_tau_s",
    )

    def _load_envelope_constants(self):
        """Replace hand-set envelope constants with calibrated estimates.

        A(R) must not contain hand-selected numbers. Every constant either
        comes from calibrate_envelope.py or is explicitly declared as an
        override in the launch parameters, and which of the two applied is
        recorded per constant so it can be reported.
        """
        self.envelope_provenance = {
            n: {"value": float(getattr(self, n)), "source": "hardcoded_default"}
            for n in self.ENVELOPE_CONSTANT_NAMES
        }

        if not self.envelope_constants_path:
            self.get_logger().error(
                "envelope_constants_path is unset: A(R) is running on "
                "hand-selected constants. Run analysis/calibrate_envelope.py.")
            return

        try:
            with open(self.envelope_constants_path) as f:
                artifact = json.load(f)
        except Exception as exc:
            raise RuntimeError(
                f"envelope constants not loadable from "
                f"{self.envelope_constants_path}: {exc}") from exc

        consts = artifact.get("constants", {})
        for name in self.ENVELOPE_CONSTANT_NAMES:
            entry = consts.get(name)
            if entry is None or entry.get("value") is None:
                self.get_logger().warn(
                    f"{name} not in the calibration artifact; keeping "
                    f"{getattr(self, name)} and recording it as uncalibrated.")
                continue
            setattr(self, name, float(entry["value"]))
            self.envelope_provenance[name] = {
                "value": float(entry["value"]),
                "source": "calibrated",
                "estimator": entry.get("estimator"),
                "quantile": entry.get("quantile"),
                "n_samples": entry.get("n_samples"),
                "ci95": entry.get("ci95"),
            }

        self.envelope_provenance["_artifact"] = artifact.get("provenance", {})
        for name, p in self.envelope_provenance.items():
            if name.startswith("_"):
                continue
            self.get_logger().info(
                f"envelope constant {name} = {p['value']:.4f} ({p['source']})")

    def _goal_pose_callback(self, msg: PoseStamped):
        """Auto-set goal when published on /goal_pose (e.g. from RViz or Nav2)."""
        goal_dict = {"x": msg.pose.position.x, "y": msg.pose.position.y}
        self.set_goal(goal_dict)

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
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    artifact = pickle.load(f)
            
            version = artifact.get("artifact_version", 1)
            assert version in (2, 3), f"Artifact version mismatch: expected 2 or 3, got {version}. Retrain models!"

            self.safety_model = artifact["safety_model"]
            self.stall_model = artifact["stall_model"]
            self.speed_model = artifact["speed_model"]
            self.feature_cols = artifact["feature_cols"]
            self.support_ranges = artifact.get("support", {})
            self.support = artifact.get("support_model", None)

            # Ensure scikit-learn version cross-compatibility for unpickled LogisticRegression models
            def _fix_sklearn_compat(m):
                if m is not None and hasattr(m, "named_steps"):
                    est = m.named_steps.get("estimator")
                    if est is not None:
                        if not hasattr(est, "multi_class"):
                            setattr(est, "multi_class", "auto")

            _fix_sklearn_compat(self.safety_model)
            _fix_sklearn_compat(self.stall_model)
            _fix_sklearn_compat(self.speed_model)

            # Load ensembles
            self.safety_ensemble = artifact.get("safety_ensemble", [])
            self.stall_ensemble = artifact.get("stall_ensemble", [])
            self.speed_ensemble = artifact.get("speed_ensemble", [])
            for ens in (self.safety_ensemble, self.stall_ensemble, self.speed_ensemble):
                for m in ens:
                    _fix_sklearn_compat(m)
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

            w = artifact.get("objective_weights")
            if w is None:
                self.get_logger().error(
                    "Artifact has no objective_weights; it predates the objective "
                    "fix. The models were fitted against a different utility than "
                    "this node optimizes. Retrain before running.")
                self.models_loaded = False
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

        # 1. The envelope must fit the constriction. Asymmetric by design:
        #    r_width is min(measured, forecast) and therefore noisy, so a
        #    single threshold makes carry flicker in and out of the feasible
        #    set and the arm chatters. Adopting carry requires extra clearance;
        #    retaining it does not.
        #    Carry mode expands the physical footprint laterally and forward.
        #    Adopting carry requires corridor width (r_width >= 1.35m) AND obstacle clearance (r_min >= 1.20m).
        #    Retaining carry requires corridor width (r_width >= 1.25m) AND obstacle clearance (r_min >= 0.80m).
        #    Prevents adopting or keeping carry in temporary small openings inside dense clutter or door frames.
        is_carry_cand = (float(cand[FOOTPRINT_KEY]) == 1.0)
        is_currently_carry = (getattr(self, "target_arm_label", "tucked") == "carry")

        if is_carry_cand:
            if not is_currently_carry:
                need_width = width + 2.0 * delta + self.envelope_hysteresis_m
                if r_width < need_width or r_min < 1.20:
                    return False
            else:
                need_width = width + 2.0 * delta + 0.15
                if r_width < need_width or r_min < 1.20:
                    return False
        else:
            need_width = width + 2.0 * delta
            if r_width < need_width:
                return False

        # 2. Candidate inflation ceiling and floor (non-colliding)
        lateral = (r_width - width) / 2.0 - delta
        max_allowed_inf = max(self.inflation_floor_m, lateral)
        if r_min < 1.2:
            max_allowed_inf = min(max_allowed_inf, 0.60)
        if float(cand[INFLATION_KEY]) > max_allowed_inf + 1e-6:
            return False

        if r_min < 0.80 and float(cand[INFLATION_KEY]) < self.inflation_floor_m - 1e-6:
            return False

        # 3. Speed ceiling. Two independent limits.
        #    (a) longitudinal: stop before the obstacle ahead.
        #    (b) lateral: hold tracking error inside the free lateral margin.
        #        A constriction fails by clipping a wall while turning, not by
        #        failing to stop, so the stopping rule alone does not cover it.
        # 3. Speed ceiling. Two independent limits.
        #    (a) longitudinal: stop before the obstacle ahead (using front bumper clearance r_front = 0.27m).
        #    (b) lateral: hold tracking error inside the free lateral margin.
        r_front = 0.27
        stopping = max(0.0, r_min - r_front - delta)
        decel_phys = max(self.decel_limit_mps2, 1.20)
        v_stop = math.sqrt(2.0 * decel_phys * stopping)

        margin = max(0.0, (r_width - width) / 2.0 - delta)
        v_lat = margin / self.lateral_tracking_tau_s

        v_ceiling = max(MIN_SPEED_LIMIT * MAX_VX_LIMIT / 100.0, min(v_stop, v_lat))
        cmd_speed = float(cand[SPEED_LIMIT_KEY]) * MAX_VX_LIMIT / 100.0
        if cmd_speed > v_ceiling + 1e-9:
            return False

        # 4. Require path adherence and costmap repulsion in constrictions (margin < 0.30m or r_min < 0.80m)
        if (margin < 0.30 or r_min < 0.80) and float(cand[PATH_ALIGN_KEY]) < 15.0 - 1e-6:
            return False
        if (margin < 0.30 or r_min < 0.80) and float(cand[COST_WEIGHT_KEY]) < 5.0 - 1e-6:
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
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        if getattr(self, "last_odom_pose", None) is not None:
            dist = math.hypot(px - self.last_odom_pose[0], py - self.last_odom_pose[1])
            if dist < 1.0:  # filter potential resets/jumps
                self.total_distance_m += dist
                if getattr(self, "verified_arm_label", None) == "carry":
                    self.carry_distance_m += dist
        self.last_odom_pose = (px, py)

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

        # Clip each feature to physical domain bounds, so linear extrapolation
        # can predict upcoming constrictions without buffer clipping floors
        # blocking the anticipation lookahead.
        min_bounds = np.zeros_like(pred)
        max_bounds = np.full(len(pred), 10.0)
        bounds_spec = [5.0, 10.0, 10.0, 1.0, 5.0, 5.0, 10.0, 10.0, 10.0]
        for i in range(min(len(pred), len(bounds_spec))):
            max_bounds[i] = bounds_spec[i]
        pred_clipped = np.clip(pred, min_bounds, max_bounds)
        
        return list(pred_clipped[:len(self.current_risk_vector)])

    def _evaluate_utility(self, expected_progress: float, p_collision: float, p_stall: float, is_carry: float) -> float:
        """Compute Hurdle utility U(c, R_t) with payload transport task reward."""
        j_max = float(self.progress_support.get("max", 3.135)) if hasattr(self, "progress_support") and isinstance(self.progress_support, dict) else 3.135
        payload_reward = self.payload_omega * is_carry * j_max
        return (
            expected_progress
            + payload_reward
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
        X_arr = np.nan_to_num(np.asarray(X_eval, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return candidates_with_eval, X_arr

    def _transform_only(self, X_arr):
        """Interaction-expanded features, shared by every ensemble member."""
        X_clean = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        return np.asarray(
            self.safety_model.named_steps["interaction"].transform(X_clean), dtype=float)

    def _evaluate_positivity(self, X_arr: np.ndarray, candidates_with_eval: list):
        """Evaluate k-NN distance out-of-support status for candidate configurations."""
        if self.support is not None:
            X_clean = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
            Z = self.support["scaler"].transform(X_clean)
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

    def _terminal_config(self, arm_val: float = 0.0):
        """Fixed configuration for the goal-alignment phase.

        The software knobs are frozen and identical for every strategy.
        The arm is not frozen to 0: the mission requires placement which
        requires the extended envelope, so the arm state is carried in from
        the placement phase if admissible.
        """
        return {
            SPEED_LIMIT_KEY: TERMINAL_SPEED_LIMIT_PCT,
            VX_STD_KEY: 0.15,
            CONSTRAINT_KEY: 2.0,
            COST_WEIGHT_KEY: 1.0,
            PATH_ALIGN_KEY: 4.0,
            INFLATION_KEY: self.inflation_floor_m,
            FOOTPRINT_KEY: arm_val,
            "selection_reason": "TERMINAL_PHASE (software fixed, arm from placement phase)",
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

        # Capture the policy's own arm choice before the constraint clamps it.
        self.preferred_arm_label = (
            "carry" if float(out[FOOTPRINT_KEY]) == 1.0 else "tucked")

        if r_width < geom["width"] + 2.0 * d:
            out[FOOTPRINT_KEY] = 0.0
            geom = FOOTPRINT_GEOMETRY[0.0]
            out["selection_reason"] += " [proj:envelope->tucked]"
        half = geom["width"] / 2.0

        max_inf = max(self.inflation_floor_m,
                      (r_width - geom["width"]) / 2.0 - d)
        if r_min < 1.2:
            max_inf = min(max_inf, 0.60)
        ok_inf = [v for v in PARAM_CANDIDATE_GRID[INFLATION_KEY] if v <= max_inf + 1e-6]
        tgt_inf = max(ok_inf) if ok_inf else self.inflation_floor_m
        if float(out[INFLATION_KEY]) > tgt_inf:
            out[INFLATION_KEY] = tgt_inf
            out["selection_reason"] += f" [proj:inf->{tgt_inf:.2f}]"

        stopping = max(0.0, r_min - half - d)
        v_stop = math.sqrt(2.0 * self.decel_limit_mps2 * stopping)
        margin = max(0.0, (r_width - geom["width"]) / 2.0 - d)
        v_lat = margin / self.lateral_tracking_tau_s
        v_ceiling = max(MIN_SPEED_LIMIT * MAX_VX_LIMIT / 100.0, min(v_stop, v_lat))
        pct_ceiling = 100.0 * v_ceiling / MAX_VX_LIMIT
        ok_spd = [v for v in PARAM_CANDIDATE_GRID[SPEED_LIMIT_KEY] if v <= pct_ceiling + 1e-6]
        tgt_spd = max(ok_spd) if ok_spd else MIN_SPEED_LIMIT
        if float(out[SPEED_LIMIT_KEY]) > tgt_spd:
            out[SPEED_LIMIT_KEY] = tgt_spd
            out["selection_reason"] += f" [proj:spd->{tgt_spd:.0f}]"

        if margin < 0.25 and float(out[PATH_ALIGN_KEY]) < 15.0:
            out[PATH_ALIGN_KEY] = 15.0
            out["selection_reason"] += " [proj:align->15]"

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
        try:
            self._tuning_loop_impl()
        except Exception as e:
            self.get_logger().error(f"Error in tuning loop tick: {e}", throttle_duration_sec=2.0)
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

        if self.current_speed < 0.02 and self.last_applied_config is not None:
            self.get_logger().info("Idle robot / stationary; tuning loop gated.", throttle_duration_sec=10.0)
            return

        d_goal = self._distance_to_goal()

        # Placement phase: the mission ends with the object on the surface, so
        # the arm must be extended and verified before the base arrives.
        entering_place = (d_goal is not None and d_goal <= self.placement_radius)
        if entering_place != self.in_placement_phase:
            self.in_placement_phase = entering_place
            self.get_logger().info(
                f"{'ENTER' if entering_place else 'EXIT'} placement phase "
                f"(d_goal={d_goal:.2f} m)")

        # Schmitt trigger: enter at terminal_radius, leave only past 1.5x it
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

        r_min = getattr(self, "_last_r_min", 3.0)
        r_width = getattr(self, "_last_r_width", 5.0)

        if self.in_terminal_phase:
            self.n_terminal_ticks += 1
            # The terminal phase freezes the software knobs; it does not decide
            # the arm. Whatever the tuned policy last selected is held through
            # goal alignment, so the final metre is not a hidden intervention.
            held = 1.0 if getattr(self, "target_arm_label", "tucked") == "carry" else 0.0
            best_config = self._terminal_config(held)
        else:
            r_eval = self._forecast_risk(self.anticipation_delta)
            # Store for apply configuration decision log
            self.last_forecast_risk = r_eval

            best_config = self.solve_optimal_configuration(r_eval, self.current_risk_vector)

        if best_config is not None:
            # Counted here rather than inside solve_optimal_configuration() so
            # both selection modes are covered; the policy-tree path never
            # calls the enumerating solver.
            self.n_ticks += 1
            if bool(best_config.get("out_of_support", False)):
                self.n_out_of_support += 1
            self._apply_configuration(best_config)

    def solve_optimal_configuration(self, risk_vector: list,
                                    risk_measured: list = None) -> dict:
        """
        Solve C_t^* = argmax_{c in A(R)} U(c, R_t) subject to P_collision <= p_max.

        `risk_vector` is the anticipated state and feeds the effect models.
        `risk_measured` is the state as sensed now and, together with the
        forecast, gates geometric feasibility. Feasibility is a claim about
        the present, so a forecast may only tighten it, never relax it.
        """
        t_solve_start = time.perf_counter()
        sim_time = self._sim_time_sec()
        risk_dict = {name: risk_vector[i] if i < len(risk_vector) else 0.0 for i, name in enumerate(RISK_FEATURE_NAMES)}

        meas = risk_measured if risk_measured is not None else risk_vector
        meas_dict = {name: meas[i] if i < len(meas) else 0.0
                     for i, name in enumerate(RISK_FEATURE_NAMES)}

        # Geometric feasibility uses the worse of measured and anticipated
        # clearance. A 4 s linear extrapolation fitted on 1.5 s of history can
        # predict a 6 m corridor while the robot sits in a 0.9 m gap; using it
        # to decide whether the footprint fits puts the extended arm into door
        # frames and raises the inflation ceiling at exactly the wrong moment.
        # Taking the minimum keeps the anticipation benefit -- the forecast can
        # only make the tuner retract earlier -- without ever permitting an
        # envelope the robot does not currently fit.
        r_min = min(float(risk_dict.get("risk__r_min", 3.0)),
                    float(meas_dict.get("risk__r_min", 3.0)))
        r_width = min(float(risk_dict.get("risk__r_width", 5.0)),
                      float(meas_dict.get("risk__r_width", 5.0)))

        # r_width reads below r_min in 50% of decisions, which is
        # geometrically impossible, and sits at ~0.31 m on open floor. A
        # passage is at least 2*r_min wide, so this is a valid lower bound
        # that never over-claims clearance. The raw value is retained for
        # the carry gate, which has no margin to spare.
        self._r_width_raw = r_width
        r_width = max(r_width, 1.50, 2.0 * r_min)

        self._last_r_min = r_min
        self._last_r_width = r_width



        # §3.2 Log unconstrained arm preference (n_carry_total, n_carry_feasible)
        self.n_carry_total = sum(1 for c in self.candidates_df if float(c[FOOTPRINT_KEY]) == 1.0)
        self.n_carry_feasible = sum(1 for c in self.candidates_df if float(c[FOOTPRINT_KEY]) == 1.0 and self._admissible(c, r_min, r_width))

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
            self.preferred_arm_label = (
                "carry" if float(best_row.get(FOOTPRINT_KEY, 0.0)) == 1.0 else "tucked")
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
            e_speed_point_all = np.asarray(np.clip(self.speed_model.predict(X_arr) * self.anticipation_delta, 0.0, max_prog_support))
        else:
            e_speed_point_all = np.ones(len(X_arr)) * self.anticipation_delta

        j_point_all = e_speed_point_all
        if self.use_bounds:
            # Score all feasible candidates using fast vectorized affine matrix multiplications
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
            E_speed = np.clip((Ws @ Z.T + bs[:, None]) * self.anticipation_delta, 0.0, max_prog_support)
            J = E_speed

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
            # Same reasoning as the policy-tree path: switch cost is charged
            # against the policy's own previous choice, not against a
            # geometry-forced retraction.
            ref_arm_label = (getattr(self, "preferred_arm_label", None)
                             or getattr(self, "target_arm_label",
                                        self.current_arm_label))
            curr_arm_val = 1.0 if ref_arm_label == "carry" else 0.0
            n_knobs = float(len(self.param_cols))
            last_cfg = self.last_applied_config
            
            for c in safe_candidates:
                u_eff = c["utility_raw"]
                cand_arm = 1.0 if float(c.get(FOOTPRINT_KEY, 0.0)) == 1.0 else 0.0
                
                if cand_arm != curr_arm_val:
                    if not (cand_arm == 1.0 and ref_arm_label == "tucked"):
                        u_eff -= self.arm_switch_cost

                if last_cfg is not None and self.deadband_margin > 0.0:
                    n_changed = 0
                    for k in (INFLATION_KEY, COST_WEIGHT_KEY, VX_STD_KEY, CONSTRAINT_KEY, PATH_ALIGN_KEY):
                        last_v = last_cfg.get(f"param__{k}")
                        if last_v is not None and abs(float(c[k]) - float(last_v)) > 1e-4:
                            n_changed += 1
                    last_spd = last_cfg.get(f"param__{SPEED_LIMIT_KEY}")
                    if last_spd is not None and float(c[SPEED_LIMIT_KEY]) < float(last_spd) - 5.0:
                        n_changed += 1
                    if n_changed > 0:
                        u_eff -= self.deadband_margin * (float(n_changed) / n_knobs)
                        
                c["utility_effective"] = u_eff

            _u_best = max(c.get("utility_effective", c["utility_raw"]) for c in safe_candidates)
            top_candidates = [
                c for c in safe_candidates 
                if (_u_best - c.get("utility_effective", c["utility_raw"])) <= self.deadband_margin
            ]
            # Hysteresis retention: If last applied config is safe and within top_candidates,
            # retain it to prevent parameter thrashing and keep MPPI trajectory generation smooth!
            matching_last = []
            if last_cfg is not None:
                matching_last = [
                    c for c in top_candidates
                    if all(
                        abs(float(c[k]) - float(last_cfg.get(f"param__{k}", c[k]))) < 1e-4
                        for k in (INFLATION_KEY, COST_WEIGHT_KEY, VX_STD_KEY, CONSTRAINT_KEY, PATH_ALIGN_KEY, SPEED_LIMIT_KEY, FOOTPRINT_KEY)
                        if f"param__{k}" in last_cfg
                    )
                ]

            if matching_last:
                best_row = max(matching_last, key=lambda c: c.get("utility_effective", c["utility_raw"])).copy()
            else:
                best_row = max(
                    top_candidates,
                    key=lambda c: (
                        float(c.get(SPEED_LIMIT_KEY, 0.0)),
                        -float(c.get(INFLATION_KEY, 1.0)),
                        -float(c.get(COST_WEIGHT_KEY, 100.0)),
                        c.get("utility_effective", c["utility_raw"]),
                        float(c.get(FOOTPRINT_KEY, 0.0))
                    )
                ).copy()
            best_row["n_tied_at_optimum"] = len(top_candidates)
            best_row["n_safe_candidates"] = len(safe_candidates)
            best_row["selection_reason"] = (
                f"HURDLE_UTILITY_OPTIMAL (U_pess={best_row['utility_raw']:.3f}, U_point={best_row.get('utility_point', 0.0):.3f}, E[J_lcb]={best_row['j_progress_lcb']:.2f}m, "
                f"P_coll_ucb={best_row['p_risk_ucb']:.4f}, P_stall_ucb={best_row['p_stall_ucb']:.3f})"
            )
        else:
            best_row = min(candidates_with_eval, key=lambda c: (c["p_risk_ucb"], c["p_stall_ucb"], float(c.get(COST_WEIGHT_KEY, 1.0)), -c["utility_raw"])).copy()
            best_row["selection_reason"] = f"FALLBACK_SAFEST (no candidate <= {self.p_max:.2f}, safest P_ucb={best_row['p_risk_ucb']:.4f})"
            best_row["n_tied_at_optimum"] = 1
            best_row["n_safe_candidates"] = 0

        # Per-level best utility, so the decision surface is recoverable from
        # the logs instead of being inferred from the winner alone.
        def _best_by(key):
            out = {}
            for c in safe_candidates if safe_candidates else feasible_candidates:
                k = round(float(c[key]), 3)
                u = c.get("utility_effective", c["utility_raw"])
                if k not in out or u > out[k]:
                    out[k] = round(float(u), 5)
            return out

        best_row["u_by_speed"] = _best_by(SPEED_LIMIT_KEY)
        best_row["u_by_inflation"] = _best_by(INFLATION_KEY)
        best_row["u_by_costw"] = _best_by(COST_WEIGHT_KEY)
        best_row["u_by_vxstd"] = _best_by(VX_STD_KEY)
        best_row["u_by_constraint"] = _best_by(CONSTRAINT_KEY)
        best_row["u_by_pathalign"] = _best_by(PATH_ALIGN_KEY)

        best_row["inf_ceiling"] = round(float(best_row[INFLATION_KEY]), 3)

        # The enumerate path filters by _admissible() before scoring, so the
        # winner is always feasible and its arm state is the policy's own
        # choice. Record it here: the switch cost must be charged against a
        # real previous decision, not against a value frozen at construction.
        self.preferred_arm_label = (
            "carry" if float(best_row.get(FOOTPRINT_KEY, 0.0)) == 1.0 else "tucked")

        best_row["t_infer_ms"] = t_infer_ms
        best_row["t_select_ms"] = (time.perf_counter() - t_select_start) * 1000.0
        return best_row

    def _set_param_async(self, node_key: str, param_name: str, value) -> bool:
        """P0.4: Set a Nav2 parameter and record whether the service accepted it."""
        if self.dry_run:
            self.get_logger().info(f"[DRY RUN] {node_key}.{param_name} = {value}")
            return True

        client = self.param_clients.get(node_key)
        if client is None:
            return False
        if not client.service_is_ready() and not client.wait_for_service(timeout_sec=0.2):
            self._record_param_failure(node_key, param_name, "service_not_ready")
            return False

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

        t_call_start = time.perf_counter()

        def _done(fut):
            dt_ms = (time.perf_counter() - t_call_start) * 1000.0
            if not hasattr(self, "param_latencies_ms"):
                self.param_latencies_ms = []
            self.param_latencies_ms.append(dt_ms)
            try:
                resp = fut.result()
                res = resp.results[0] if resp.results else None
                if res is None or not res.successful:
                    reason = res.reason if res is not None else "empty_response"
                    self._record_param_failure(node_key, param_name, reason)
            except Exception as exc:
                self._record_param_failure(node_key, param_name, repr(exc))

        client.call_async(req).add_done_callback(_done)
        return True

    def _record_param_failure(self, node_key: str, param_name: str, reason: str):
        key = f"{node_key}.{param_name}"
        self.n_param_set_failures += 1
        self.param_set_failures[key] = self.param_set_failures.get(key, 0) + 1
        if reason == "service_not_ready":
            self.get_logger().warn(f"Parameter set PENDING (service not ready, will retry): {key}")
        else:
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
        pt.time_from_start.sec = 2
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

        # P0.2d: Local costmap footprint immediately tracks target configuration chosen by tuner
        target_footprint = ARM_CONFIGS[arm_label]["footprint"]

        # P0.2e: Track verified physical arm state samples
        self.total_samples += 1
        self.verified_arm_label = self._verified_arm_label()
        if self.verified_arm_label == "carry":
            self.carry_samples += 1

        d_g = self._distance_to_goal()
        d_g_str = f"{d_g:.2f}m" if d_g is not None else "N/A"
        reason = target_config.get("selection_reason", "OPTIMAL")
        self.get_logger().info(
            f"[TUNER TICK #{self.n_ticks}] t={sim_now:.1f}s | d_goal={d_g_str} | "
            f"Arm: target='{arm_label}' (verified='{self.verified_arm_label}') | "
            f"Speed: {speed_limit_pct:.0f}% | Inflation: {inflation:.2f}m | "
            f"Weights: [cw={constraint_weight}, cost={cost_weight}, align={path_align_weight}] | "
            f"Reason: {reason}"
        )

        last_cfg = self.last_applied_config or {}

        any_param_changed = (
            (last_cfg.get("param__controller_server__speed_limit_pct") is None or abs(speed_limit_pct - float(last_cfg.get("param__controller_server__speed_limit_pct"))) >= 5.0) or
            (last_cfg.get("param__controller_server__FollowPath.vx_std") is None or abs(vx_std - float(last_cfg.get("param__controller_server__FollowPath.vx_std"))) >= 0.05) or
            (last_cfg.get("param__controller_server__FollowPath.ConstraintCritic.cost_weight") is None or abs(constraint_weight - float(last_cfg.get("param__controller_server__FollowPath.ConstraintCritic.cost_weight"))) >= 0.5) or
            (last_cfg.get("param__controller_server__FollowPath.CostCritic.cost_weight") is None or abs(cost_weight - float(last_cfg.get("param__controller_server__FollowPath.CostCritic.cost_weight"))) >= 0.5) or
            (last_cfg.get("param__controller_server__FollowPath.PathAlignCritic.cost_weight") is None or abs(path_align_weight - float(last_cfg.get("param__controller_server__FollowPath.PathAlignCritic.cost_weight"))) >= 1.0) or
            (last_cfg.get("param__local_costmap__inflation_layer.inflation_radius") is None or abs(inflation - float(last_cfg.get("param__local_costmap__inflation_layer.inflation_radius"))) >= 0.10) or
            (arm_label != self.applied_envelope_label)
        )

        if any_param_changed:
            self.n_config_switches += 1

        if self.last_applied_config is None:
            self.last_applied_config = {}

        # Clamp speed limit during active arm transition to prevent moving fast while arm is swinging
        if sim_now < getattr(self, "arm_transition_until_time", 0.0):
            speed_limit_pct = min(speed_limit_pct, 15.0)

        last_spd = last_cfg.get("param__controller_server__speed_limit_pct")
        if last_spd is None or abs(speed_limit_pct - float(last_spd)) >= 5.0:
            self.current_speed_limit_pct = speed_limit_pct
            self._publish_speed_limit(speed_limit_pct)
            self.last_applied_config["param__controller_server__speed_limit_pct"] = speed_limit_pct

        last_vx = last_cfg.get("param__controller_server__FollowPath.vx_std")
        if last_vx is None or abs(vx_std - float(last_vx)) >= 0.05:
            if self._set_param_async("controller_server", "FollowPath.vx_std", vx_std):
                self.last_applied_config["param__controller_server__FollowPath.vx_std"] = vx_std

        last_cw = last_cfg.get("param__controller_server__FollowPath.ConstraintCritic.cost_weight")
        if last_cw is None or abs(constraint_weight - float(last_cw)) >= 0.5:
            if self._set_param_async("controller_server", "FollowPath.ConstraintCritic.cost_weight", constraint_weight):
                self.last_applied_config["param__controller_server__FollowPath.ConstraintCritic.cost_weight"] = constraint_weight

        last_cost = last_cfg.get("param__controller_server__FollowPath.CostCritic.cost_weight")
        if last_cost is None or abs(cost_weight - float(last_cost)) >= 0.5:
            if self._set_param_async("controller_server", "FollowPath.CostCritic.cost_weight", cost_weight):
                self.last_applied_config["param__controller_server__FollowPath.CostCritic.cost_weight"] = cost_weight

        last_align = last_cfg.get("param__controller_server__FollowPath.PathAlignCritic.cost_weight")
        if last_align is None or abs(path_align_weight - float(last_align)) >= 1.0:
            ok1 = self._set_param_async("controller_server", "FollowPath.PathAlignCritic.cost_weight", path_align_weight)
            ok2 = self._set_param_async("controller_server", "FollowPath.PathFollowCritic.cost_weight", path_align_weight)
            if ok1 and ok2:
                self.last_applied_config["param__controller_server__FollowPath.PathAlignCritic.cost_weight"] = path_align_weight
                self.last_applied_config["param__controller_server__FollowPath.PathFollowCritic.cost_weight"] = path_align_weight

        last_inf = last_cfg.get("param__local_costmap__inflation_layer.inflation_radius")
        if last_inf is None or abs(inflation - float(last_inf)) >= 0.10:
            if self._set_param_async("local_costmap", "inflation_layer.inflation_radius", inflation):
                self.last_applied_config["param__local_costmap__inflation_layer.inflation_radius"] = inflation

        # The costmap must track the VERIFIED physical envelope, never
        # the commanded one. Shrinking it at command time leaves 4 s in
        # which Nav2 plans against a 0.55 m footprint around a robot
        # whose arm is still swinging. _envelope_label() returns the
        # larger of physical and target, and "carry" when the physical
        # state is unknown.
        env_label = self._envelope_label(arm_label)
        if env_label != self.applied_envelope_label:
            if self._set_param_async("local_costmap", "footprint", ARM_CONFIGS[env_label]["footprint"]):
                # self._set_param_async("global_costmap", "footprint", ARM_CONFIGS[env_label]["footprint"])
                self._publish_footprint_polygon(ARM_CONFIGS[env_label]["footprint"])
                self.applied_envelope_label = env_label

        # Command the arm when the decision changes, OR when a previously
        # commanded target was never physically reached.
        if arm_label != self.target_arm_label:
            if arm_label == self.pending_arm_label:
                self.pending_arm_count += 1
            else:
                self.pending_arm_label = arm_label
                self.pending_arm_count = 1
        else:
            self.pending_arm_label = None
            self.pending_arm_count = 0

        # Safety-critical arm retraction (carry -> tucked) MUST command immediately (1st tick)
        # and ignore dwell_ok so the robot does not collide while waiting for dwell timer.
        if arm_label == "tucked" and self.target_arm_label == "carry":
            persisted = True
            dwell_ok = True
        elif arm_label == "carry" and self.target_arm_label == "tucked":
            persisted = self.pending_arm_count >= 2
            dwell_ok = (sim_now - self.last_arm_switch_time) >= self.arm_switch_dwell
        else:
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
            self.arm_transition_until_time = sim_now + 2.0
            if not self._move_arm_to_label_async(arm_label):
                self.arm_transition_until_time = 0.0

        # P1.2b Record tick trace in decision_log with full Table I intervention space
        self.decision_log.append({
            # --- timing and context (measured BEFORE this config was applied)
            "t": sim_now,
            "risk": list(self.current_risk_vector or []),
            "risk_measured": list(self.current_risk_vector or []),
            "risk_forecast": list(getattr(self, "last_forecast_risk", [])),

            # --- the seven parameters of Table I, by their Table I names
            "speed_limit_pct":  float(speed_limit_pct),
            "vx_std":           float(vx_std),
            "ConstraintCritic": float(constraint_weight),
            "CostCritic":       float(cost_weight),
            "PathAlignCritic":  float(path_align_weight),
            "inflation_radius": float(inflation),
            "arm_target":       arm_label,

            # --- footprint verification chain (P0.2)
            "arm_verified":      self.verified_arm_label,
            "envelope_applied":  self.applied_envelope_label,

            # --- held fixed; logged so the claim is checkable, not asserted
            "wz_max":     1.0,
            "time_steps": 56,

            # --- selector internals
            "p_risk":         float(target_config.get("p_risk", float("nan"))),
            "p_stall":        float(target_config.get("p_stall", float("nan"))),
            "utility":        float(target_config.get("utility_raw", float("nan"))),
            "reason":         target_config.get("selection_reason", ""),
            "out_of_support": bool(target_config.get("out_of_support", False)),

            # --- alias keys for backward compatibility with existing analysis tools
            "constraint_weight": float(constraint_weight),
            "cost_weight":       float(cost_weight),
            "path_align_weight":  float(path_align_weight),
            "inflation":         float(inflation),
            "arm_preferred":     getattr(self, "preferred_arm_label", None),
            "p_risk_ucb":        float(target_config.get("p_risk_ucb", float("nan"))),
            "p_stall_ucb":       float(target_config.get("p_stall_ucb", float("nan"))),
            "utility_point":     float(target_config.get("utility_point", float("nan"))),
            "terminal_phase":    bool(self.in_terminal_phase),
            "d_goal":            self._distance_to_goal(),
            "u_by_speed":        target_config.get("u_by_speed"),
            "u_by_inflation":    target_config.get("u_by_inflation"),
            "u_by_costw":        target_config.get("u_by_costw"),
            "inf_ceiling":       target_config.get("inf_ceiling"),
            "n_tied_at_optimum": target_config.get("n_tied_at_optimum"),
            "n_safe_candidates": target_config.get("n_safe_candidates"),
            "n_carry_feasible":  getattr(self, "n_carry_feasible", 0),
            "n_carry_total":     getattr(self, "n_carry_total", 0),
        })

        target_config["_applied_speed_limit_pct"] = speed_limit_pct

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
        self.carry_distance_m = 0.0
        self.total_distance_m = 0.0
        self.last_odom_pose = None
        self.pending_inflation = None
        self.pending_inflation_count = 0
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
        self.in_placement_phase = False
        self.in_terminal_phase = False
        self.n_param_set_failures = 0
        self.param_set_failures = {}

        verified = self._verified_arm_label()
        if verified is not None:
            self.current_arm_label = verified
        self.target_arm_label = self.current_arm_label
        self.preferred_arm_label = "carry"
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
