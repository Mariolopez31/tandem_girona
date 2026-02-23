#!/usr/bin/env python3
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data

import tf2_ros

from aruco_opencv_msgs.msg import ArucoDetection
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped

from tf_transformations import quaternion_from_euler, euler_from_quaternion, quaternion_matrix, quaternion_multiply


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


def transform_pose(p: Pose, t) -> Pose:
    """
    Apply TransformStamped t (target <- source) to Pose p expressed in source frame.
    Returns Pose expressed in target frame.
    """
    # translation
    tr = t.transform.translation
    # rotation
    qr = t.transform.rotation

    T = quaternion_matrix([qr.x, qr.y, qr.z, qr.w])
    T[0, 3] = tr.x
    T[1, 3] = tr.y
    T[2, 3] = tr.z

    v = np.array([p.position.x, p.position.y, p.position.z, 1.0], dtype=float)
    v2 = T @ v

    out = Pose()
    out.position.x = float(v2[0])
    out.position.y = float(v2[1])
    out.position.z = float(v2[2])

    qi = [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
    qT = [qr.x, qr.y, qr.z, qr.w]
    qo = quaternion_multiply(qT, qi)

    out.orientation.x = float(qo[0])
    out.orientation.y = float(qo[1])
    out.orientation.z = float(qo[2])
    out.orientation.w = float(qo[3])
    return out


class ArucoMapLocalization(Node):
    def __init__(self):
        super().__init__("down_camera_localization")

        self.base_frame = "blueboat/base_link"

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

        # Subscriber
        self.create_subscription(
            ArucoDetection,
            self.aruco_topic,
            self.aruco_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info("Down camera ArUco localization started")

    def aruco_callback(self, msg: ArucoDetection):
        visible_ids = [m.marker_id for m in msg.markers]
        self.get_logger().info(
            f"det frame={msg.header.frame_id} n={len(msg.markers)} ids={visible_ids}"
        )

        # 1) Publish fixed map markers
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
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.0, 0.8
            if mid in visible_ids:
                m.color.r, m.color.g, m.color.b = 1.0, 1.0, 0.0
            marker_array.markers.append(m)

        self.marker_pub.publish(marker_array)

        # 2) Localization
        if not msg.markers:
            return

        cam_frame = msg.header.frame_id
        self.get_logger().info(f"lookup TF {self.base_frame} <- {cam_frame}")

        try:
            tf_base_cam = self.tf_buffer.lookup_transform(
                self.base_frame,
                cam_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2),
            )
        except Exception as e:
            self.get_logger().warn(f"TF base<-cam missing: {e}")
            return

        estimates, weights = [], []
        yaw_estimates, yaw_weights = [], []

        for mk in msg.markers:
            mid = int(mk.marker_id)
            if mid not in ARUCO_MAP:
                continue

            try:
                base_pose = transform_pose(mk.pose, tf_base_cam)  # Pose in base_link
            except Exception as e:
                self.get_logger().warn(f"transform_pose failed (id={mid}): {e}")
                continue

            wx, wy, wz = ARUCO_MAP[mid]

            rx = wx - base_pose.position.x
            ry = wy - base_pose.position.y
            rz = wz - base_pose.position.z

            dx = base_pose.position.x
            dy = base_pose.position.y
            dz = base_pose.position.z
            dist = float(np.sqrt(dx*dx + dy*dy + dz*dz))
            w = 1.0 / (dist*dist + 0.25)

            estimates.append([rx, ry, rz])
            weights.append(w)

            q = base_pose.orientation
            _, _, yaw_marker = euler_from_quaternion([q.x, q.y, q.z, q.w])

            yaw_robot = yaw_marker + ARUCO_YAW_OFFSET
            yaw_robot = float(np.arctan2(np.sin(yaw_robot), np.cos(yaw_robot)))

            yaw_estimates.append(yaw_robot)
            yaw_weights.append(w)

        if not estimates:
            self.get_logger().warn("No estimates -> not publishing")
            return

        est_np = np.array(estimates, dtype=float)
        w_np = np.array(weights, dtype=float)
        mean_pos = np.sum(est_np * w_np[:, None], axis=0) / np.sum(w_np)

        yaw_np = np.array(yaw_estimates, dtype=float)
        wy_np = np.array(yaw_weights, dtype=float)
        mean_yaw = float(np.arctan2(np.sum(np.sin(yaw_np)*wy_np), np.sum(np.cos(yaw_np)*wy_np)))

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

        total_w = float(np.sum(w_np))
        sigma_xy  = 0.20               # 20 cm
        sigma_z   = 10.0               # si estás en 2D, “no me fío” del z
        sigma_yaw = np.deg2rad(10.0)   # 10 grados

        cov = [0.0] * 36
        cov[0]  = sigma_xy**2
        cov[7]  = sigma_xy**2
        cov[14] = sigma_z**2
        cov[35] = sigma_yaw**2

        out.pose.covariance = cov


        self.get_logger().info(
            f"PUBLISH aruco_pose x={mean_pos[0]:.2f} y={mean_pos[1]:.2f} z={mean_pos[2]:.2f} yaw={mean_yaw:.2f}"
        )
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
