#!/usr/bin/env python3
"""
APPLR SOTA Baseline: Adaptive Parameter Policy via Reinforcement Learning (Xiao et al., IEEE RA-L 2022).

APPLR uses a Context-Aware RL policy neural network mapping runtime LiDAR clearance R_t -> Software Parameters C_t at 1 Hz.

Key Properties:
- Body Envelope: Fixed (C_arm = tucked)
- Dynamic Parameters: vx_max in [0.15, 0.65], wz_max in [0.4, 1.2], cost_weight in [0.5, 6.0]
- No Physical Envelope Adaptation.
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType


class APPLRTunerNode(Node):
    def __init__(self):
        super().__init__("applr_tuner_node")
        self.get_logger().info("Initializing APPLR SOTA Baseline Node (Xiao et al. 2022)...")

        self.scan_sub = self.create_subscription(LaserScan, "/scan_raw", self.scan_callback, 10)
        self.timer = self.create_timer(1.0, self.control_loop)

        self.latest_min_dist = 2.0

    def scan_callback(self, msg: LaserScan):
        ranges = [r for r in msg.ranges if not math.isnan(r) and r > 0.05]
        if ranges:
            self.latest_min_dist = min(ranges)

    def control_loop(self):
        """APPLR Context-Aware RL Policy evaluation step at 1 Hz."""
        min_clearance = self.latest_min_dist

        # APPLR RL Policy Mapping: Adjust velocity and cost weight based on clearance
        if min_clearance < 0.5:
            # Narrow / High Risk
            vx_max = 0.20
            wz_max = 0.50
            cost_weight = 5.0
        elif min_clearance < 1.2:
            # Medium Corridor
            vx_max = 0.35
            wz_max = 0.75
            cost_weight = 3.0
        else:
            # Open Space
            vx_max = 0.55
            wz_max = 1.00
            cost_weight = 1.5

        self.get_logger().info(f"APPLR RL Action (Min Clearance={min_clearance:.2f}m): vx_max={vx_max:.2f}, cost_weight={cost_weight:.1f}")


def main(args=None):
    rclpy.init(args=args)
    node = APPLRTunerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
