#!/usr/bin/env python3
"""speed_limit_probe2.py — why does /speed_limit work when held but not in the
collector?

Established by v1: vx_max = 0.7, a 30 % limit held at 5 Hz gives p95 |v_x| =
0.210 m/s, exactly 0.30 * 0.7. Delivery works.

The collector calls publish_speed_limit() ONCE at apply time and then issues six
SetParameters calls ~30 ms later. This script reproduces that sequence and
compares it against the held case.

Fixes over v1:
  * alternates between two goals, so the robot is always driving during a window
  * waits until it is actually translating instead of a fixed warmup
  * reports the forward-sample count so an empty window is obvious
  * the verdict guards against NaN

Run with NOTHING else publishing /speed_limit:
    ros2 node list | grep -i tuner
    ros2 topic info /speed_limit          # Publisher count must be 0

    python3 speed_limit_probe2.py --ax 10.0 --ay 0.0 --bx 1.0 --by 0.0

Pick two poses with a long, straight, obstacle-free run between them.
"""
import argparse
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav2_msgs.msg import SpeedLimit
from rcl_interfaces.srv import SetParameters, GetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

MOVING = 0.02          # m/s; below this the sample is not forward motion


class Probe(Node):
    def __init__(self, args):
        super().__init__("speed_limit_probe2")
        self.args = args
        self.samples = []
        self.recording = False
        self.last_vx = 0.0
        self.create_subscription(Twist, args.cmd_vel_topic, self._cmd_cb, 10)
        self.sl_pub = self.create_publisher(SpeedLimit, "/speed_limit", 10)
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.hold_value = None                      # None = do not republish
        self.create_timer(0.2, self._republish)
        self.set_cli = self.create_client(SetParameters,
                                          "/controller_server/set_parameters")
        self.get_cli = self.create_client(GetParameters,
                                          "/controller_server/get_parameters")
        self._goal_toggle = 0

    def _cmd_cb(self, msg: Twist):
        self.last_vx = msg.linear.x
        if self.recording:
            self.samples.append((time.time(), msg.linear.x))

    def _republish(self):
        if self.hold_value is not None:
            self.publish_limit(self.hold_value)

    def publish_limit(self, pct):
        m = SpeedLimit()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "map"
        m.percentage = True
        m.speed_limit = float(pct)
        self.sl_pub.publish(m)

    def get_param(self, name):
        if not self.get_cli.wait_for_service(timeout_sec=3.0):
            return None
        req = GetParameters.Request()
        req.names = [name]
        fut = self.get_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=3.0)
        r = fut.result()
        return r.values[0].double_value if (r and r.values) else None

    def set_param(self, name, value):
        req = SetParameters.Request()
        p = Parameter()
        p.name = name
        p.value = ParameterValue(type=ParameterType.PARAMETER_DOUBLE,
                                 double_value=float(value))
        req.parameters = [p]
        fut = self.set_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=3.0)
        r = fut.result()
        return bool(r and r.results and r.results[0].successful)

    def send_next_goal(self):
        """Alternate between the two goals so the robot always has somewhere to go."""
        a = self.args
        x, y = ((a.ax, a.ay) if self._goal_toggle == 0 else (a.bx, a.by))
        self._goal_toggle ^= 1
        g = PoseStamped()
        g.header.frame_id = "map"
        g.header.stamp = self.get_clock().now().to_msg()
        g.pose.position.x, g.pose.position.y = float(x), float(y)
        yaw = math.atan2(a.ay - a.by, a.ax - a.bx) if self._goal_toggle else \
              math.atan2(a.by - a.ay, a.bx - a.ax)
        g.pose.orientation.z = math.sin(yaw / 2.0)
        g.pose.orientation.w = math.cos(yaw / 2.0)
        self.goal_pub.publish(g)
        return x, y

    def spin_for(self, sec):
        t0 = time.time()
        while rclpy.ok() and time.time() - t0 < sec:
            rclpy.spin_once(self, timeout_sec=0.02)

    def wait_until_moving(self, timeout=25.0):
        t0 = time.time()
        while rclpy.ok() and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.last_vx > 0.10:
                return True
        return False

    @staticmethod
    def p95(sel):
        v = np.array([abs(vx) for _, vx in sel if vx > MOVING])
        return (float(np.percentile(v, 95)) if len(v) >= 5 else float("nan")), len(v)

    def condition(self, label, pct, window, hold, set_param_at=None):
        """One measurement window.

        hold=True  -> republish the limit at 5 Hz for the whole window
        hold=False -> publish exactly once at window start (collector behaviour)
        """
        self.hold_value = None
        self.samples = []
        self.recording = False

        gx, gy = self.send_next_goal()
        if not self.wait_until_moving():
            print(f"  {label:<38} SKIPPED — robot never moved toward ({gx}, {gy})")
            return float("nan"), float("nan")
        self.spin_for(1.5)                    # let it reach cruise

        if pct is not None:
            self.publish_limit(pct)
            if hold:
                self.hold_value = pct
        self.recording = True

        if set_param_at is None:
            self.spin_for(window)
            t_split = None
        else:
            self.spin_for(set_param_at)
            t_split = time.time()
            ok = self.set_param("FollowPath.vx_std", self.args.vx_std_probe)
            print(f"    SetParameters(FollowPath.vx_std={self.args.vx_std_probe})"
                  f" -> {'ok' if ok else 'FAILED'}")
            self.spin_for(window - set_param_at)

        self.recording = False
        self.hold_value = None

        if t_split is None:
            v, n = self.p95(self.samples)
            print(f"  {label:<38} p95|vx| = {v:.3f} m/s   (n_fwd={n})")
            return v, float("nan")
        b, nb = self.p95([s for s in self.samples if s[0] < t_split])
        a, na = self.p95([s for s in self.samples if s[0] >= t_split])
        print(f"  {label:<38} before={b:.3f}  after={a:.3f} m/s "
              f"(n_fwd={nb}/{na})")
        return b, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ax", type=float, required=True)
    ap.add_argument("--ay", type=float, required=True)
    ap.add_argument("--bx", type=float, required=True)
    ap.add_argument("--by", type=float, required=True)
    ap.add_argument("--window-sec", type=float, default=8.0)
    ap.add_argument("--vx-std-probe", type=float, default=0.25)
    ap.add_argument("--cmd-vel-topic", default="/cmd_vel")
    args = ap.parse_args()

    rclpy.init()
    n = Probe(args)
    n.spin_for(1.0)
    vx_max = n.get_param("FollowPath.vx_max")

    print("=" * 72)
    print(f"FollowPath.vx_max = {vx_max}")
    if vx_max:
        print(f"expected under 30 % = {0.30*vx_max:.3f} m/s | "
              f"under 95 % = {0.95*vx_max:.3f} m/s")
    print("=" * 72)

    print("\nbaseline")
    free, _ = n.condition("A  no limit", None, args.window_sec, hold=False)

    print("\nheld at 5 Hz (probe behaviour)")
    h30, _ = n.condition("B  30 % held", 30.0, args.window_sec, hold=True)
    h95, _ = n.condition("C  95 % held", 95.0, args.window_sec, hold=True)
    hb, ha = n.condition("D  30 % held + param set", 30.0, args.window_sec,
                         hold=True, set_param_at=args.window_sec / 2.0)

    print("\npublished ONCE (collector behaviour)")
    s30, _ = n.condition("E  30 % once", 30.0, args.window_sec, hold=False)
    sb, sa = n.condition("F  30 % once + param set", 30.0, args.window_sec,
                         hold=False, set_param_at=args.window_sec / 2.0)

    def ok(x):
        return isinstance(x, float) and not math.isnan(x)

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    cap = 0.30 * vx_max if vx_max else float("nan")
    tol = 0.06

    if not ok(s30):
        print("  E produced no forward samples — rerun with a longer straight run.")
    elif abs(s30 - cap) <= tol:
        print("  A single publish DOES cap the controller. The collection-time")
        print("  failure is not the publish pattern. Next suspects: another")
        print("  /speed_limit publisher during collection, or the limit being")
        print("  reset when the probe's goToPose starts a new FollowPath.")
    else:
        print(f"  A single publish does NOT cap ({s30:.3f} vs expected {cap:.3f}),")
        print("  while the held publish does. The collector's one-shot publish is")
        print("  being lost or overwritten. Fix: hold the speed limit for the")
        print("  duration of the horizon rather than publishing once at apply.")

    if ok(sb) and ok(sa) and sa > sb + tol:
        print("  F also shows the limit surviving until a SetParameters call and")
        print("  breaking after it — publish the speed limit LAST in _apply_params.")
    if ok(hb) and ok(ha) and ha > hb + tol:
        print("  D shows a param set defeating even a held limit — the reset is in")
        print("  MPPI's dynamic-parameter callback.")

    print(f"\n  A free={free:.3f}  B held30={h30:.3f}  C held95={h95:.3f}")
    print(f"  D held30 {hb:.3f} -> {ha:.3f}   E once30={s30:.3f}   "
          f"F once30 {sb:.3f} -> {sa:.3f}")

    n.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()