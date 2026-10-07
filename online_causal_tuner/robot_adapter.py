"""Convert Mirte measurements and configurations to Campaign A units."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class RobotAdapter:
    """Apply the explicitly configured transfer scaling for a physical robot."""

    LENGTH_FEATURES = {"risk__r_min", "risk__r_width", "risk__r_clear"}

    def __init__(self, profile_path: str | Path):
        self.profile_path = str(profile_path)
        with open(profile_path, encoding="utf-8") as profile_file:
            profile = yaml.safe_load(profile_file) or {}

        try:
            reference = profile["reference"]
            robot = profile["robot"]
            features = profile["features"]
            self.model_width = float(reference["width_m"])
            self.robot_width = float(robot["width_m"])
            self.footprint = str(robot["footprint"])
            self.feature_order = list(features["order"])
            self.feature_support = {
                name: (float(bounds[0]), float(bounds[1]))
                for name, bounds in features["support"].items()
            }
            self.p_max = float(profile["risk_threshold_p_max"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid robot profile {self.profile_path}: missing or invalid field"
            ) from exc

        if self.model_width <= 0.0 or self.robot_width <= 0.0:
            raise ValueError("reference.width_m and robot.width_m must be positive")
        if len(self.feature_order) != len(set(self.feature_order)):
            raise ValueError("features.order must not contain duplicate names")
        self.length_scale = self.robot_width / self.model_width

    def features_to_model(self, features: list[float]) -> tuple[list[float], bool]:
        """Scale length features and clip every supported feature to its fit range."""
        if len(features) != len(self.feature_order):
            raise ValueError(
                f"Expected {len(self.feature_order)} risk features, got {len(features)}"
            )

        model_features = []
        out_of_support = False
        for name, value in zip(self.feature_order, features):
            model_value = float(value)
            if name in self.LENGTH_FEATURES:
                model_value /= self.length_scale
            if name in self.feature_support:
                lower, upper = self.feature_support[name]
                out_of_support |= model_value < lower or model_value > upper
                model_value = min(max(model_value, lower), upper)
            model_features.append(model_value)
        return model_features, out_of_support

    def config_to_robot(self, model_config: dict[str, Any]) -> dict[str, Any]:
        """Convert the current tuner action space to Mirte parameter values."""
        robot_config = dict(model_config)
        inflation_key = "param__local_costmap__inflation_layer.inflation_radius"
        if inflation_key in robot_config:
            robot_config[inflation_key] = float(robot_config[inflation_key]) * self.length_scale
        return robot_config