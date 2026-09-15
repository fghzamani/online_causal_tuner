#!/usr/bin/env python3
"""
ROS 2 Dynamic Obstacle Controller Node for causal_benchmark_3_dynamic.
Implements a Direction-Agnostic One-Shot Dynamic Obstacle Policy:
1. Standby: Obstacle waits off to the side at x = 3.0 m (y = 3.50 m).
2. Trigger: Triggers symmetrically when robot is within 1.0 to 3.2 m Y-distance of y = 3.50 m
   (works identically for Northbound trials y: -8.5 -> +8.5 and Southbound trials y: +8.5 -> -8.5).
3. Hold: Obstacle slides to x = -4.20 m at 0.70 m/s and HOLDS in place for the rest of the trial.
4. Reset: Teleports obstacle back to standby on /initialpose, /reset_dynamic_obstacle service, or robot teleportation.
"""

import sys
import time
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_srvs.srv import Trigger
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import SetEntityState, SetModelState


class DynamicObstacleController(Node):
    def __init__(self):
        super().__init__('dynamic_obstacle_controller')

        self.declare_parameter('mode', 'one_shot') # 'one_shot' (default) or 'oscillating'
        self.declare_parameter('obstacle_name', 'w3_dynamic_obstacle_1')
        self.declare_parameter('robot_name', 'tiago')
        self.declare_parameter('trigger_dist_min', 1.0)
        self.declare_parameter('trigger_dist_max', 3.2)
        self.declare_parameter('standby_x', 3.0)
        self.declare_parameter('blocked_x', -4.2)
        self.declare_parameter('fixed_y', 3.50)
        self.declare_parameter('fixed_z', 0.5)
        self.declare_parameter('speed_mps', 0.70)
        self.declare_parameter('update_rate_hz', 20.0)

        self.mode = str(self.get_parameter('mode').value).lower()
        self.obstacle_name = self.get_parameter('obstacle_name').value
        self.robot_name = self.get_parameter('robot_name').value
        self.trigger_dist_min = self.get_parameter('trigger_dist_min').value
        self.trigger_dist_max = self.get_parameter('trigger_dist_max').value
        self.standby_x = self.get_parameter('standby_x').value
        self.blocked_x = self.get_parameter('blocked_x').value
        self.fixed_y = self.get_parameter('fixed_y').value
        self.fixed_z = self.get_parameter('fixed_z').value
        self.speed_mps = self.get_parameter('speed_mps').value
        self.update_rate_hz = self.get_parameter('update_rate_hz').value

        self.robot_x = 0.0
        self.robot_y = -9.5
        self.last_robot_pose = None
        self.has_robot_pose = False
        self.pose_source = "none"

        self.curr_x = self.standby_x
        self.has_triggered = False
        self.movement_complete = False
        self.oscillate_direction = -1.0  # -1.0 towards blocked_x, +1.0 towards standby_x
        self.last_log_time = 0.0

        # Gazebo service clients & topics
        self.gazebo_clients = [
            self.create_client(SetEntityState, '/set_entity_state'),
            self.create_client(SetEntityState, '/gazebo/set_entity_state'),
            self.create_client(SetModelState, '/set_model_state'),
            self.create_client(SetModelState, '/gazebo/set_model_state'),
        ]
        self.set_model_pub = self.create_publisher(ModelState, '/gazebo/set_model_state', 10)

        # Service for explicit reset from benchmark script
        self.reset_srv = self.create_service(Trigger, '/reset_dynamic_obstacle', self._reset_service_cb)

        # Subscriptions: ground-truth /gazebo/model_states, /amcl_pose fallback, and /initialpose
        self.model_states_sub = self.create_subscription(
            ModelStates,
            '/gazebo/model_states',
            self._model_states_cb,
            10
        )
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self._amcl_cb,
            10
        )
        self.initial_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self._initialpose_cb,
            10
        )

        dt = 1.0 / self.update_rate_hz
        self.timer = self.create_timer(dt, self._timer_cb)

        self.get_logger().info(
            f"DynamicObstacleController initialized (Mode='{self.mode}'). "
            f"Obstacle '{self.obstacle_name}' standby at x={self.standby_x:.2f}m (y={self.fixed_y:.2f}m). "
            f"Target blocked_x={self.blocked_x:.2f}m at speed={self.speed_mps:.2f}m/s."
        )

    def _reset_service_cb(self, request, response):
        self._do_reset("service call /reset_dynamic_obstacle")
        response.success = True
        response.message = "Dynamic obstacle reset to standby pose."
        return response

    def _initialpose_cb(self, msg: PoseWithCovarianceStamped):
        self._do_reset("received /initialpose topic")

    def _do_reset(self, reason: str):
        self.get_logger().info(f"RESET TRIGGERED ({reason}). Teleporting obstacle back to standby x={self.standby_x:.2f}m.")
        self.curr_x = self.standby_x
        self.has_triggered = False
        self.movement_complete = False
        self.oscillate_direction = -1.0
        self._send_obstacle_pose(self.standby_x, vx=0.0)

    def _update_robot_pose(self, rx: float, ry: float, source: str):
        if self.last_robot_pose is not None:
            jump = math.hypot(rx - self.last_robot_pose[0], ry - self.last_robot_pose[1])
            if jump > 4.0:
                self._do_reset(f"robot teleported ({jump:.2f}m jump)")

        self.robot_x = rx
        self.robot_y = ry
        self.last_robot_pose = (rx, ry)
        self.has_robot_pose = True
        self.pose_source = source

    def _model_states_cb(self, msg: ModelStates):
        if self.robot_name in msg.name:
            idx = msg.name.index(self.robot_name)
            rx = msg.pose[idx].position.x
            ry = msg.pose[idx].position.y
            self._update_robot_pose(rx, ry, "gazebo_model_states")

    def _amcl_cb(self, msg: PoseWithCovarianceStamped):
        if self.pose_source != "gazebo_model_states":
            rx = msg.pose.pose.position.x
            ry = msg.pose.pose.position.y
            self._update_robot_pose(rx, ry, "amcl_pose")

    def _timer_cb(self):
        dt = 1.0 / self.update_rate_hz
        now = time.time()

        if not self.has_robot_pose:
            return

        dist_y = abs(self.robot_y - self.fixed_y)

        # MODE 2: Continuous Oscillating Back and Forth
        if self.mode in ["oscillating", "continuous", "back_and_forth"]:
            step = self.speed_mps * dt * self.oscillate_direction
            self.curr_x += step

            if self.oscillate_direction < 0 and self.curr_x <= self.blocked_x:
                self.curr_x = self.blocked_x
                self.oscillate_direction = 1.0
            elif self.oscillate_direction > 0 and self.curr_x >= self.standby_x:
                self.curr_x = self.standby_x
                self.oscillate_direction = -1.0

            self._send_obstacle_pose(self.curr_x, vx=step / dt)

            if now - self.last_log_time >= 1.0:
                self.get_logger().info(
                    f"[OSCILLATING] Obstacle x={self.curr_x:.2f}m (dir={self.oscillate_direction:+.1f}) | Robot y={self.robot_y:.2f}m"
                )
                self.last_log_time = now
            return

        # MODE 1 (DEFAULT): One-Shot Trigger & Hold
        if not self.has_triggered:
            if self.trigger_dist_min <= dist_y <= self.trigger_dist_max:
                self.has_triggered = True
                self.get_logger().info(
                    f"[DIRECTION-AGNOSTIC TRIGGER] Robot reached y={self.robot_y:.2f}m (dist_y={dist_y:.2f}m to obstacle line y={self.fixed_y:.2f}m). "
                    f"Obstacle moving from x={self.standby_x:.2f}m to x={self.blocked_x:.2f}m at {self.speed_mps:.2f}m/s!"
                )

        # Move obstacle once if triggered and hold position
        if self.has_triggered and not self.movement_complete:
            dx = self.blocked_x - self.curr_x
            step = math.copysign(self.speed_mps * dt, dx)

            if abs(dx) <= abs(step):
                self.curr_x = self.blocked_x
                self.movement_complete = True
                self._send_obstacle_pose(self.curr_x, vx=0.0)
                self.get_logger().info(
                    f"[MOVEMENT COMPLETE & HELD] Obstacle reached target x={self.blocked_x:.2f}m (y={self.fixed_y}m). Holding position!"
                )
            else:
                self.curr_x += step
                self._send_obstacle_pose(self.curr_x, vx=step / dt)

                if now - self.last_log_time >= 0.5:
                    self.get_logger().info(
                        f"[SLIDING] Obstacle at x={self.curr_x:.2f}m -> target x={self.blocked_x:.2f}m | Robot y={self.robot_y:.2f}m (dist_y={dist_y:.2f}m)"
                    )
                    self.last_log_time = now

    def _send_obstacle_pose(self, x_pos: float, vx: float = 0.0):
        ms = ModelState()
        ms.model_name = self.obstacle_name
        ms.pose.position.x = float(x_pos)
        ms.pose.position.y = float(self.fixed_y)
        ms.pose.position.z = float(self.fixed_z)
        ms.pose.orientation.w = 1.0
        ms.twist.linear.x = float(vx)
        ms.reference_frame = 'world'

        # Topic Publisher
        self.set_model_pub.publish(ms)

        # Service Calls
        req_entity = SetEntityState.Request()
        req_entity.state.name = self.obstacle_name
        req_entity.state.pose = ms.pose
        req_entity.state.twist = ms.twist
        req_entity.state.reference_frame = 'world'

        req_model = SetModelState.Request()
        req_model.model_state = ms

        for client in self.gazebo_clients:
            if client.service_is_ready():
                if client.srv_type == SetEntityState:
                    client.call_async(req_entity)
                elif client.srv_type == SetModelState:
                    client.call_async(req_model)


def main(args=None):
    rclpy.init(args=args)
    node = DynamicObstacleController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
