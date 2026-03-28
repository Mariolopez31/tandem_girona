#!/usr/bin/env python3
import math
from collections import deque
from typing import Optional

import rclpy
from rclpy.node import Node

from std_msgs.msg import Empty
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf_transformations import euler_from_quaternion

from interface.srv import Relocalize, IsValid


def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


RED = "\033[31m"
GREEN = "\033[32m"
BLUE = "\033[34m"
YELLOW = "\033[33m"
RESET = "\033[0m"


class ArucoToRelocalize(Node):
    def __init__(self):
        super().__init__("aruco_to_relocalize")

        # Parameters
        self.declare_parameter("aruco_pose_topic", "/blueboat/navigator/aruco_pose")
        self.declare_parameter("trigger_topic", "/blueboat/navigator/aruco_relocalize_trigger")
        self.declare_parameter("relocalize_service", "/localizer/relocalize")
        self.declare_parameter("relocalize_check_service", "/localizer/relocalize_check")

        self.declare_parameter("pcd_path", "")
        self.declare_parameter("auto_trigger_on_first_pose", False)
        self.declare_parameter("check_success_after_call", True)
        self.declare_parameter("check_delay_sec", 1.0)

        # 2D assumptions by default
        self.declare_parameter("fixed_z", 0.0)
        self.declare_parameter("fixed_roll", 0.0)
        self.declare_parameter("fixed_pitch", 0.0)

        # Optional offsets in case frames are not exactly aligned
        self.declare_parameter("x_offset", 0.0)
        self.declare_parameter("y_offset", 0.0)
        self.declare_parameter("yaw_offset", 0.0)

        # Safety filters
        self.declare_parameter("max_pose_age_sec", 2.0)
        self.declare_parameter("require_pose_before_trigger", True)

        # Robust trigger window
        self.declare_parameter("trigger_window_sec", 10.0)
        self.declare_parameter("min_trigger_samples", 10)
        self.declare_parameter("trigger_max_xy_deviation", 0.40)
        self.declare_parameter("trigger_max_yaw_deviation_deg", 12.0)
        self.declare_parameter("trigger_pose_buffer_size", 50)

        # Logging
        self.declare_parameter("log_every_n_pose_msgs", 0)

        self.aruco_pose_topic = self.get_parameter("aruco_pose_topic").get_parameter_value().string_value
        self.trigger_topic = self.get_parameter("trigger_topic").get_parameter_value().string_value
        self.relocalize_service_name = self.get_parameter("relocalize_service").get_parameter_value().string_value
        self.relocalize_check_service_name = self.get_parameter("relocalize_check_service").get_parameter_value().string_value

        self.pcd_path = self.get_parameter("pcd_path").get_parameter_value().string_value
        self.auto_trigger_on_first_pose = self.get_parameter("auto_trigger_on_first_pose").get_parameter_value().bool_value
        self.check_success_after_call = self.get_parameter("check_success_after_call").get_parameter_value().bool_value
        self.check_delay_sec = self.get_parameter("check_delay_sec").get_parameter_value().double_value

        self.fixed_z = self.get_parameter("fixed_z").get_parameter_value().double_value
        self.fixed_roll = self.get_parameter("fixed_roll").get_parameter_value().double_value
        self.fixed_pitch = self.get_parameter("fixed_pitch").get_parameter_value().double_value

        self.x_offset = self.get_parameter("x_offset").get_parameter_value().double_value
        self.y_offset = self.get_parameter("y_offset").get_parameter_value().double_value
        self.yaw_offset = self.get_parameter("yaw_offset").get_parameter_value().double_value

        self.max_pose_age_sec = self.get_parameter("max_pose_age_sec").get_parameter_value().double_value
        self.require_pose_before_trigger = self.get_parameter("require_pose_before_trigger").get_parameter_value().bool_value

        self.trigger_window_sec = self.get_parameter("trigger_window_sec").get_parameter_value().double_value
        self.min_trigger_samples = max(1, self.get_parameter("min_trigger_samples").get_parameter_value().integer_value)
        self.trigger_max_xy_deviation = self.get_parameter("trigger_max_xy_deviation").get_parameter_value().double_value
        self.trigger_max_yaw_deviation_deg = self.get_parameter("trigger_max_yaw_deviation_deg").get_parameter_value().double_value
        self.trigger_pose_buffer_size = max(1, self.get_parameter("trigger_pose_buffer_size").get_parameter_value().integer_value)

        self.log_every_n_pose_msgs = max(
            0,
            self.get_parameter("log_every_n_pose_msgs").get_parameter_value().integer_value
        )

        self.last_pose_msg: Optional[PoseWithCovarianceStamped] = None
        self.last_x: Optional[float] = None
        self.last_y: Optional[float] = None
        self.last_yaw: Optional[float] = None
        self.pose_history = deque(maxlen=self.trigger_pose_buffer_size)
        self.has_called_once = False
        self.pending_check = False
        self.pose_msg_counter = 0

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.aruco_pose_topic,
            self.pose_cb,
            10,
        )

        self.trigger_sub = self.create_subscription(
            Empty,
            self.trigger_topic,
            self.trigger_cb,
            10,
        )

        self.relocalize_client = self.create_client(Relocalize, self.relocalize_service_name)
        self.relocalize_check_client = self.create_client(IsValid, self.relocalize_check_service_name)

        self.check_timer = None

        self.get_logger().info("ArucoToRelocalize started")
        self.get_logger().info(f"  aruco_pose_topic: {self.aruco_pose_topic}")
        self.get_logger().info(f"  trigger_topic: {self.trigger_topic}")
        self.get_logger().info(f"  relocalize_service: {self.relocalize_service_name}")
        self.get_logger().info(f"  relocalize_check_service: {self.relocalize_check_service_name}")
        self.get_logger().info(f"  pcd_path: {self.pcd_path}")
        self.get_logger().info(f"  auto_trigger_on_first_pose: {self.auto_trigger_on_first_pose}")
        self.get_logger().info(f"  trigger_window_sec: {self.trigger_window_sec}")
        self.get_logger().info(f"  min_trigger_samples: {self.min_trigger_samples}")
        self.get_logger().info(f"  trigger_max_xy_deviation: {self.trigger_max_xy_deviation}")
        self.get_logger().info(f"  trigger_max_yaw_deviation_deg: {self.trigger_max_yaw_deviation_deg}")
        self.get_logger().info(f"  trigger_pose_buffer_size: {self.trigger_pose_buffer_size}")
        self.get_logger().info(f"  log_every_n_pose_msgs: {self.log_every_n_pose_msgs}")

    def pose_cb(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        x = float(p.x + self.x_offset)
        y = float(p.y + self.y_offset)
        yaw = wrap_angle(float(yaw + self.yaw_offset))

        self.last_pose_msg = msg
        self.last_x = x
        self.last_y = y
        self.last_yaw = yaw
        self.pose_history.append({
            "time": rclpy.time.Time.from_msg(msg.header.stamp),
            "x": x,
            "y": y,
            "yaw": yaw,
        })

        self.pose_msg_counter += 1
        if self.log_every_n_pose_msgs > 0 and self.pose_msg_counter % self.log_every_n_pose_msgs == 0:
            self.get_logger().info(
                f"Latest FILTERED ArUco pose: "
                f"x={x:.3f} y={y:.3f} yaw={yaw:.3f} rad ({math.degrees(yaw):.1f} deg)"
            )

        if self.auto_trigger_on_first_pose and not self.has_called_once:
            self.get_logger().info("Auto trigger on first pose is enabled. Calling relocalize.")
            self.call_relocalize()

    def trigger_cb(self, _msg: Empty):
        self.get_logger().info("Received relocalize trigger")
        self.call_relocalize()

    def pose_is_recent_enough(self) -> bool:
        if self.last_pose_msg is None:
            return False

        stamp = self.last_pose_msg.header.stamp
        pose_time = rclpy.time.Time.from_msg(stamp)
        now = self.get_clock().now()
        age = (now - pose_time).nanoseconds / 1e9

        if age > self.max_pose_age_sec:
            self.get_logger().warn(
                f"Latest ArUco pose is too old: {age:.3f}s > {self.max_pose_age_sec:.3f}s"
            )
            return False

        return True

    def compute_trigger_pose(self) -> Optional[tuple[float, float, float]]:
        now = self.get_clock().now()
        recent = []
        for item in self.pose_history:
            age = (now - item["time"]).nanoseconds / 1e9
            if age <= self.trigger_window_sec:
                recent.append(item)

        if len(recent) < self.min_trigger_samples:
            self.get_logger().warn(
                f"Not enough recent ArUco poses for robust trigger: "
                f"{len(recent)} < {self.min_trigger_samples} in {self.trigger_window_sec:.2f}s"
            )
            return None

        xs = [p["x"] for p in recent]
        ys = [p["y"] for p in recent]
        yaws = [p["yaw"] for p in recent]

        med_x = sorted(xs)[len(xs) // 2]
        med_y = sorted(ys)[len(ys) // 2]
        ref_yaw = math.atan2(sum(math.sin(a) for a in yaws), sum(math.cos(a) for a in yaws))

        max_yaw_dev = math.radians(self.trigger_max_yaw_deviation_deg)
        inliers = []
        outliers = []
        for item in recent:
            dx = item["x"] - med_x
            dy = item["y"] - med_y
            xy_dev = math.hypot(dx, dy)
            yaw_dev = abs(wrap_angle(item["yaw"] - ref_yaw))
            sample = {
                **item,
                "xy_dev": xy_dev,
                "yaw_dev": yaw_dev,
            }
            if xy_dev <= self.trigger_max_xy_deviation and yaw_dev <= max_yaw_dev:
                inliers.append(sample)
            else:
                outliers.append(sample)

        self.get_logger().info(
            f"{YELLOW}Relocalization trigger analysis: {len(recent)} recent samples in {self.trigger_window_sec:.2f}s{RESET}"
        )
        for idx, item in enumerate(inliers, start=1):
            self.get_logger().info(
                f"{GREEN}[ACCEPT {idx:02d}]{RESET} "
                f"x={item['x']:.3f} y={item['y']:.3f} yaw={item['yaw']:.3f} "
                f"dxy={item['xy_dev']:.3f} dyaw={math.degrees(item['yaw_dev']):.2f}deg"
            )
        for idx, item in enumerate(outliers, start=1):
            self.get_logger().warn(
                f"{RED}[REJECT {idx:02d}]{RESET} "
                f"x={item['x']:.3f} y={item['y']:.3f} yaw={item['yaw']:.3f} "
                f"dxy={item['xy_dev']:.3f} dyaw={math.degrees(item['yaw_dev']):.2f}deg"
            )

        if len(inliers) < self.min_trigger_samples:
            self.get_logger().warn(
                f"Too many trigger-pose outliers: kept {len(inliers)} / {len(recent)} samples"
            )
            return None

        inlier_xs = sorted(item["x"] for item in inliers)
        inlier_ys = sorted(item["y"] for item in inliers)
        inlier_yaws = [item["yaw"] for item in inliers]

        out_x = inlier_xs[len(inlier_xs) // 2]
        out_y = inlier_ys[len(inlier_ys) // 2]
        out_yaw = math.atan2(
            sum(math.sin(a) for a in inlier_yaws),
            sum(math.cos(a) for a in inlier_yaws),
        )

        self.get_logger().info(
            f"{BLUE}[ROBUST MEAN]{RESET} "
            f"x={out_x:.3f} y={out_y:.3f} yaw={out_yaw:.3f} "
            f"from {len(inliers)}/{len(recent)} samples"
        )
        return float(out_x), float(out_y), float(out_yaw)

    def call_relocalize(self):
        if not self.pcd_path:
            self.get_logger().error("pcd_path parameter is empty")
            return

        if self.require_pose_before_trigger:
            if self.last_pose_msg is None or self.last_x is None or self.last_y is None or self.last_yaw is None:
                self.get_logger().warn("No ArUco pose received yet, cannot relocalize")
                return

            if not self.pose_is_recent_enough():
                return

        robust_pose = self.compute_trigger_pose()
        if robust_pose is None:
            return
        robust_x, robust_y, robust_yaw = robust_pose

        if not self.relocalize_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f"Service {self.relocalize_service_name} not available")
            return

        req = Relocalize.Request()
        req.pcd_path = self.pcd_path
        req.x = float(robust_x)
        req.y = float(robust_y)
        req.z = float(self.fixed_z)
        req.yaw = float(robust_yaw)
        req.roll = float(self.fixed_roll)
        req.pitch = float(self.fixed_pitch)

        self.get_logger().info(
            f"Calling relocalize with "
            f"pcd='{req.pcd_path}', x={req.x:.3f}, y={req.y:.3f}, z={req.z:.3f}, "
            f"yaw={req.yaw:.3f}, roll={req.roll:.3f}, pitch={req.pitch:.3f}"
        )

        future = self.relocalize_client.call_async(req)
        future.add_done_callback(self.relocalize_done_cb)

        self.has_called_once = True

    def relocalize_done_cb(self, future):
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"Relocalize service call failed: {e}")
            return

        self.get_logger().info(
            f"Relocalize response: success={response.success}, message='{response.message}'"
        )

        if response.success and self.check_success_after_call:
            self.schedule_check()

    def schedule_check(self):
        if self.pending_check:
            return

        self.pending_check = True
        self.get_logger().info(f"Scheduling relocalize_check in {self.check_delay_sec:.2f}s")

        self.check_timer = self.create_timer(self.check_delay_sec, self.check_once_cb)

    def check_once_cb(self):
        if self.check_timer is not None:
            self.check_timer.cancel()
            self.check_timer = None

        self.pending_check = False

        if not self.relocalize_check_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f"Service {self.relocalize_check_service_name} not available")
            return

        req = IsValid.Request()
        req.code = 0

        future = self.relocalize_check_client.call_async(req)
        future.add_done_callback(self.relocalize_check_done_cb)

    def relocalize_check_done_cb(self, future):
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"Relocalize check service call failed: {e}")
            return

        self.get_logger().info(f"Relocalize check: valid={response.valid}")


def main():
    rclpy.init()
    node = ArucoToRelocalize()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
