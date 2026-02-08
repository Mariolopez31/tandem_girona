#!/usr/bin/env python3
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data

import tf2_ros
import tf2_geometry_msgs

from aruco_opencv_msgs.msg import ArucoDetection
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped

from tf_transformations import quaternion_from_euler, euler_from_quaternion


# ============================
# Fixed ArUco map (cirtesu_base_link)
# ============================

CIRTESU_MESH_PATH = "package://blueboat_stonefish/meshes/cirtesu.dae"
CIRTESU_MESH_SCALE = [1.0, 1.0, 1.0]
CIRTESU_MESH_POS = [0.0, 0.0, 0.2]
CIRTESU_MESH_YAW = 0.0

ARUCO_YAW_OFFSET = -np.pi / 2.0

ARUCO_MAP = {
    0: (-1.35,  3.0, 5.0),
    1: (-4.35, -3.0, 5.0),
    2: ( 1.65,  0.0, 5.0),
    3: ( 1.65,  3.0, 5.0),
    4: (-1.35,  0.0, 5.0),
    5: (-4.35,  3.0, 5.0),
    6: (-4.35,  0.0, 5.0),
    7: (-1.35, -3.0, 5.0),
    8: ( 1.65, -3.0, 5.0),
}


class ArucoMapLocalization(Node):
    def __init__(self):
        super().__init__("aruco_map_localization")

        self.world_frame = "world_ned"
        self.base_frame = "blueboat/base_link"
        self.camera_frame = "blueboat_camera_link"

        # Topics 
        self.aruco_topic = "/blueboat/down_camera/aruco_detections"
        self.marker_topic = "/blueboat/aruco_map_markers"
        self.pose_topic = "/blueboat/navigator/aruco_pose"

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publishers
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 1)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, self.pose_topic, 1)

        # Subscriber (QoS sensor)
        self.create_subscription(
            ArucoDetection,
            self.aruco_topic,
            self.aruco_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info("Aruco map localization node started (ROS2)")

    def aruco_callback(self, msg: ArucoDetection):
        visible_ids = [m.marker_id for m in msg.markers]

        # -------------------------------------------------
        # 1) Publish fixed map MarkerArray
        # -------------------------------------------------
        marker_array = MarkerArray()

        mesh_marker = Marker()
        mesh_marker.header.frame_id = "cirtesu_base_link"
        mesh_marker.header.stamp = self.get_clock().now().to_msg()
        mesh_marker.ns = "cirtesu_mesh"
        mesh_marker.id = 1000
        mesh_marker.type = Marker.MESH_RESOURCE
        mesh_marker.action = Marker.ADD
        mesh_marker.mesh_resource = CIRTESU_MESH_PATH
        mesh_marker.mesh_use_embedded_materials = True

        mesh_marker.scale.x = float(CIRTESU_MESH_SCALE[0])
        mesh_marker.scale.y = float(CIRTESU_MESH_SCALE[1])
        mesh_marker.scale.z = float(CIRTESU_MESH_SCALE[2])

        mesh_marker.pose.position.x = float(CIRTESU_MESH_POS[0])
        mesh_marker.pose.position.y = float(CIRTESU_MESH_POS[1])
        mesh_marker.pose.position.z = float(CIRTESU_MESH_POS[2])

        qx, qy, qz, qw = quaternion_from_euler(np.pi, 0.0, CIRTESU_MESH_YAW)
        mesh_marker.pose.orientation.x = float(qx)
        mesh_marker.pose.orientation.y = float(qy)
        mesh_marker.pose.orientation.z = float(qz)
        mesh_marker.pose.orientation.w = float(qw)

        mesh_marker.color.a = 1.0
        marker_array.markers.append(mesh_marker)

        for mid, (mx, my, mz) in ARUCO_MAP.items():
            m = Marker()
            m.header.frame_id = "cirtesu_base_link"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "aruco_map"
            m.id = int(mid)
            m.type = Marker.CUBE
            m.action = Marker.ADD

            m.pose.position.x = float(mx)
            m.pose.position.y = float(my)
            m.pose.position.z = float(mz)
            m.pose.orientation.w = 1.0

            m.scale.x = 0.30
            m.scale.y = 0.30
            m.scale.z = 0.08

            # GREEN default
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = 0.8

            # visible -> YELLOW
            if mid in visible_ids:
                m.color.r = 1.0
                m.color.g = 1.0
                m.color.b = 0.0

            marker_array.markers.append(m)

        self.marker_pub.publish(marker_array)

        # -------------------------------------------------
        # 2) Robot localization
        # -------------------------------------------------
        if len(msg.markers) == 0:
            return

        try:
            tf_base_cam = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                rclpy.time.Time(),  # latest
                timeout=Duration(seconds=0.2),
            )
        except Exception:
            self.get_logger().warn("TF base->camera not available")
            return

        estimates, weights = [], []
        yaw_estimates, yaw_weights = [], []

        for marker in msg.markers:
            mid = marker.marker_id
            if mid not in ARUCO_MAP:
                continue

            cam_marker = PoseStamped()
            cam_marker.header = msg.header
            cam_marker.pose = marker.pose

            try:
                base_marker = tf2_geometry_msgs.do_transform_pose(cam_marker, tf_base_cam)
            except Exception:
                continue

            wx, wy, wz = ARUCO_MAP[mid]

            rx = wx - base_marker.pose.position.x
            ry = wy - base_marker.pose.position.y
            rz = wz - base_marker.pose.position.z

            dx = base_marker.pose.position.x
            dy = base_marker.pose.position.y
            dz = base_marker.pose.position.z
            dist = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            weight = 1.0 / (dist * dist + 0.25)

            estimates.append([rx, ry, rz])
            weights.append(weight)

            q = base_marker.pose.orientation
            _, _, yaw_marker = euler_from_quaternion([q.x, q.y, q.z, q.w])

            yaw_robot = yaw_marker + ARUCO_YAW_OFFSET
            yaw_robot = float(np.arctan2(np.sin(yaw_robot), np.cos(yaw_robot)))

            yaw_estimates.append(yaw_robot)
            yaw_weights.append(weight)

        if not estimates:
            return

        est_np = np.array(estimates, dtype=float)
        w_np = np.array(weights, dtype=float)
        mean_pos = np.sum(est_np * w_np[:, None], axis=0) / np.sum(w_np)

        yaw_np = np.array(yaw_estimates, dtype=float)
        wy_np = np.array(yaw_weights, dtype=float)
        sin_sum = np.sum(np.sin(yaw_np) * wy_np)
        cos_sum = np.sum(np.cos(yaw_np) * wy_np)
        mean_yaw = float(np.arctan2(sin_sum, cos_sum))

        out = PoseWithCovarianceStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "cirtesu_base_link"

        out.pose.pose.position.x = float(mean_pos[0])
        out.pose.pose.position.y = float(mean_pos[1])
        out.pose.pose.position.z = float(mean_pos[2])

        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, -mean_yaw)
        out.pose.pose.orientation.x = float(qx)
        out.pose.pose.orientation.y = float(qy)
        out.pose.pose.orientation.z = float(qz)
        out.pose.pose.orientation.w = float(qw)

        # Covariance 
        total_weight = float(np.sum(w_np))
        sigma_xy = 0.01 / np.sqrt(max(total_weight, 1e-6))
        sigma_z = 0.005 / np.sqrt(max(total_weight, 1e-6))
        sigma_yaw = 0.1 / max(len(yaw_estimates), 1)

        cov = [0.0] * 36
        cov[0] = float(sigma_xy)
        cov[7] = float(sigma_xy)
        cov[14] = float(sigma_z)
        cov[21] = 0.01
        cov[28] = 0.01
        cov[35] = float(sigma_yaw)

        out.pose.covariance = cov
        self.pose_pub.publish(out)


def main():
    rclpy.init()
    node = ArucoMapLocalization()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
