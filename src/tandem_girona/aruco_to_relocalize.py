#!/usr/bin/env python3
import math
from typing import Optional

import rclpy
from rclpy.node import Node

from std_msgs.msg import Empty
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf_transformations import euler_from_quaternion

from interface.srv import Relocalize, IsValid


def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


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

        # Logging
        self.declare_parameter("log_every_n_pose_msgs", 5)

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

        self.log_every_n_pose_msgs = max(
            1,
            self.get_parameter("log_every_n_pose_msgs").get_parameter_value().integer_value
        )

        self.last_pose_msg: Optional[PoseWithCovarianceStamped] = None
        self.last_x: Optional[float] = None
        self.last_y: Optional[float] = None
        self.last_yaw: Optional[float] = None
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

        self.pose_msg_counter += 1
        if self.pose_msg_counter % self.log_every_n_pose_msgs == 0:
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

        if not self.relocalize_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f"Service {self.relocalize_service_name} not available")
            return

        req = Relocalize.Request()
        req.pcd_path = self.pcd_path
        req.x = float(self.last_x)
        req.y = float(self.last_y)
        req.z = float(self.fixed_z)
        req.yaw = float(self.last_yaw)
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