#!/usr/bin/env python3
import math
from collections import deque
from typing import List, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data

import tf2_ros

from aruco_opencv_msgs.msg import ArucoDetection
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray

from tf_transformations import (
    quaternion_from_euler,
    euler_from_quaternion,
    quaternion_matrix,
    quaternion_multiply,
)


# ============================================================
# Fixed map of ArUcos in cirtesu_base_link
# ============================================================

FRAME_MAP = "cirtesu_base_link"
BASE_FRAME = "blueboat/base_link"

CIRTESU_MESH_PATH = "package://blueboat_stonefish_core/meshes/cirtesu.dae"
CIRTESU_MESH_SCALE = [1.0, 1.0, 1.0]
CIRTESU_MESH_POS = [0.0, 0.0, 0.2]
CIRTESU_MESH_YAW = 0.0

# Ajuste detector / convención.
ARUCO_YAW_OFFSET = -np.pi / 2.0

ARUCO_MAP = {
    0: {"x": -1.35, "y":  3.0, "z": 5.0, "yaw": 0.0},
    1: {"x": -4.35, "y": -3.0, "z": 5.0, "yaw": 0.0},
    2: {"x":  1.65, "y":  0.0, "z": 5.0, "yaw": 0.0},
    3: {"x":  1.65, "y":  3.0, "z": 5.0, "yaw": 0.0},
    4: {"x": -1.35, "y":  0.0, "z": 5.0, "yaw": 0.0},
    5: {"x": -4.35, "y":  3.0, "z": 5.0, "yaw": 0.0},
    6: {"x": -4.35, "y":  0.0, "z": 5.0, "yaw": 0.0},
    7: {"x": -1.35, "y": -3.0, "z": 5.0, "yaw": 0.0},
    8: {"x":  1.65, "y": -3.0, "z": 5.0, "yaw": 0.0},
}


def wrap_angle(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def angle_diff(a: float, b: float) -> float:
    return wrap_angle(a - b)


def weighted_circular_mean(angles: np.ndarray, weights: np.ndarray) -> float:
    s = np.sum(np.sin(angles) * weights)
    c = np.sum(np.cos(angles) * weights)
    return float(np.arctan2(s, c))


def circular_mean(angles: np.ndarray) -> float:
    s = np.mean(np.sin(angles))
    c = np.mean(np.cos(angles))
    return float(np.arctan2(s, c))


def transform_pose(p: Pose, t) -> Pose:
    """
    Apply TransformStamped t (target <- source) to Pose p expressed in source frame.
    Returns Pose expressed in target frame.
    """
    tr = t.transform.translation
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


class DownCameraLocalization(Node):
    def __init__(self):
        super().__init__("down_camera_localization_matrix")

        self.base_frame = BASE_FRAME

        self.aruco_topic = "/blueboat/down_camera/aruco_detections"
        self.marker_topic = "/blueboat/aruco_map_markers"
        self.pose_topic = "/blueboat/navigator/aruco_pose"

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 1)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, self.pose_topic, 1)
        self.pose_viz_topic = "/blueboat/navigator/aruco_pose_viz"
        self.pose_viz_pub = self.create_publisher(PoseWithCovarianceStamped, self.pose_viz_topic, 1)

        self.create_subscription(
            ArucoDetection,
            self.aruco_topic,
            self.aruco_callback,
            qos_profile_sensor_data,
        )

        # ======================================================
        # Filtro temporal / publicación estable
        # ======================================================
        self.pose_buffer = deque(maxlen=20)   # últimas estimaciones instantáneas
        self.last_good_pose = None            # (x, y, yaw)
        self.last_good_ros_time = None        # rclpy.time.Time
        self.last_detect_stamp_msg = None     # builtin_interfaces/Time del detector

        self.publish_rate_hz = 2.0            # publica más lento que la cámara
        self.publish_period = 1.0 / self.publish_rate_hz
        self.hold_timeout_sec = 1.0           # mantiene la última buena si pierdes unos frames
        self.min_recent_samples = 5           # mínimo de muestras recientes para publicar pose nueva
        self.min_markers_for_estimate = 3     # mínimo de markers visibles para aceptar una estimación instantánea

        self.publish_timer = self.create_timer(self.publish_period, self.publish_filtered_pose)

        self._log_counter = 0   

        self.get_logger().info("Down camera ArUco localization started")
        self.get_logger().info(
            f"Temporal filter: buffer={self.pose_buffer.maxlen}, "
            f"publish_rate={self.publish_rate_hz:.1f} Hz, "
            f"hold_timeout={self.hold_timeout_sec:.2f}s, "
            f"min_recent_samples={self.min_recent_samples}, "
            f"min_markers_for_estimate={self.min_markers_for_estimate}"
        )

    def publish_fixed_markers(self, stamp, visible_ids: List[int]):
        marker_array = MarkerArray()

        mesh_marker = Marker()
        mesh_marker.header.frame_id = FRAME_MAP
        mesh_marker.header.stamp = stamp
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

        for mid, data in ARUCO_MAP.items():
            m = Marker()
            m.header.frame_id = FRAME_MAP
            m.header.stamp = stamp
            m.ns = "aruco_map"
            m.id = int(mid)
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = float(data["x"])
            m.pose.position.y = float(data["y"])
            m.pose.position.z = float(data["z"])

            qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, float(data["yaw"]))
            m.pose.orientation.x = float(qx)
            m.pose.orientation.y = float(qy)
            m.pose.orientation.z = float(qz)
            m.pose.orientation.w = float(qw)

            m.scale.x = 0.30
            m.scale.y = 0.30
            m.scale.z = 0.08
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.0, 0.8

            if mid in visible_ids:
                m.color.r, m.color.g, m.color.b = 1.0, 1.0, 0.0

            marker_array.markers.append(m)

        self.marker_pub.publish(marker_array)

    def estimate_from_one_marker(self, base_pose: Pose, marker_world: dict) -> Tuple[np.ndarray, float, float]:
        """
        base_pose = pose of the marker expressed in base_link.
                    marker position/orientation as seen from the robot base.
        marker_world = known marker pose in world/map.

        Returns:
            p_w_b: np.array([x, y, z]) robot/base position in world
            yaw_w_b: robot yaw in world
            weight: confidence weight
        """
        mx_b = float(base_pose.position.x)
        my_b = float(base_pose.position.y)
        mz_b = float(base_pose.position.z)

        q = base_pose.orientation
        _, _, yaw_b_m = euler_from_quaternion([q.x, q.y, q.z, q.w])

        yaw_w_m = float(marker_world["yaw"])
        yaw_w_b = wrap_angle(yaw_w_m - yaw_b_m + ARUCO_YAW_OFFSET)

        R_w_b = np.array([
            [math.cos(yaw_w_b), -math.sin(yaw_w_b)],
            [math.sin(yaw_w_b),  math.cos(yaw_w_b)],
        ], dtype=float)

        p_b_m = np.array([mx_b, my_b], dtype=float)
        p_w_m = np.array([marker_world["x"], marker_world["y"]], dtype=float)

        p_w_b_xy = p_w_m - (R_w_b @ p_b_m)
        z_w_b = float(marker_world["z"] - mz_b)

        dist_xy = float(np.hypot(mx_b, my_b))
        weight = 1.0 / (dist_xy * dist_xy + 0.10)

        p_w_b = np.array([p_w_b_xy[0], p_w_b_xy[1], z_w_b], dtype=float)
        return p_w_b, yaw_w_b, weight

    def robust_fuse_estimates(self, positions: List[np.ndarray], yaws: List[float], weights: List[float]):
        pos_np = np.array(positions, dtype=float)
        yaw_np = np.array(yaws, dtype=float)
        w_np = np.array(weights, dtype=float)

        if len(pos_np) == 1:
            return pos_np[0], yaw_np[0], w_np

        mean_pos = np.sum(pos_np * w_np[:, None], axis=0) / np.sum(w_np)
        mean_yaw = weighted_circular_mean(yaw_np, w_np)

        pos_err = np.linalg.norm(pos_np[:, :2] - mean_pos[:2], axis=1)
        yaw_err = np.abs(np.array([angle_diff(a, mean_yaw) for a in yaw_np]))

        keep = (pos_err < 0.75) & (yaw_err < np.deg2rad(25.0))

        if np.sum(keep) >= 1:
            pos_np = pos_np[keep]
            yaw_np = yaw_np[keep]
            w_np = w_np[keep]

        mean_pos = np.sum(pos_np * w_np[:, None], axis=0) / np.sum(w_np)
        mean_yaw = weighted_circular_mean(yaw_np, w_np)

        return mean_pos, mean_yaw, w_np

    def aruco_callback(self, msg: ArucoDetection):
        visible_ids = [int(m.marker_id) for m in msg.markers]
        stamp = msg.header.stamp

        self.publish_fixed_markers(stamp, visible_ids)

        if not msg.markers:
            return

        cam_frame = msg.header.frame_id
        if not cam_frame:
            self.get_logger().warn("ArucoDetection header.frame_id is empty")
            return

        try:
            tf_base_cam = self.tf_buffer.lookup_transform(
                self.base_frame,
                cam_frame,
                stamp,
                timeout=Duration(seconds=0.2),
            )
        except Exception as e:
            self.get_logger().warn(f"TF {self.base_frame} <- {cam_frame} missing: {e}")
            return

        positions = []
        yaws = []
        weights = []

        for mk in msg.markers:
            mid = int(mk.marker_id)
            if mid not in ARUCO_MAP:
                self.get_logger().warn(f"Marker id {mid} not in ARUCO_MAP")
                continue

            try:
                base_pose = transform_pose(mk.pose, tf_base_cam)
            except Exception as e:
                self.get_logger().warn(f"transform_pose failed for id={mid}: {e}")
                continue

            try:
                p_w_b, yaw_w_b, weight = self.estimate_from_one_marker(base_pose, ARUCO_MAP[mid])
            except Exception as e:
                self.get_logger().warn(f"estimate_from_one_marker failed for id={mid}: {e}")
                continue

            positions.append(p_w_b)
            yaws.append(yaw_w_b)
            weights.append(weight)

        if not positions:
            return

        if len(positions) < self.min_markers_for_estimate:
            self._log_counter += 1
            if self._log_counter % 10 == 0:
                self.get_logger().warn(
                    f"Skipping instantaneous estimate: only {len(positions)} marker estimates"
                )
            return

        mean_pos, mean_yaw, kept_weights = self.robust_fuse_estimates(positions, yaws, weights)

        corrected_yaw = wrap_angle(mean_yaw)

        # Guardamos estimación instantánea en buffer, no publicamos aquí
        now_ros = self.get_clock().now()
        self.last_detect_stamp_msg = stamp

        self.pose_buffer.append({
            "time": now_ros,
            "x": float(mean_pos[0]),
            "y": float(mean_pos[1]),
            "yaw": float(corrected_yaw),
            "markers": int(len(positions)),
            "sum_w": float(np.sum(kept_weights)),
        })

        self._log_counter += 1

    def publish_filtered_pose(self):
        now = self.get_clock().now()

        recent = []
        for item in self.pose_buffer:
            age = (now - item["time"]).nanoseconds / 1e9
            if age <= self.hold_timeout_sec:
                recent.append(item)

        publish_mode = None

        if len(recent) >= self.min_recent_samples:
            xs = np.array([p["x"] for p in recent], dtype=float)
            ys = np.array([p["y"] for p in recent], dtype=float)
            yaws = np.array([p["yaw"] for p in recent], dtype=float)
            weights = np.array([p["sum_w"] for p in recent], dtype=float)

            # x,y por mediana para aguantar saltos discretos
            pub_x = float(np.median(xs))
            pub_y = float(np.median(ys))

            # yaw por media circular ponderada
            pub_yaw = weighted_circular_mean(yaws, weights)

            self.last_good_pose = (pub_x, pub_y, pub_yaw)
            self.last_good_ros_time = now
            publish_mode = "fresh"
        elif self.last_good_pose is not None and self.last_good_ros_time is not None:
            age = (now - self.last_good_ros_time).nanoseconds / 1e9
            if age <= self.hold_timeout_sec:
                pub_x, pub_y, pub_yaw = self.last_good_pose
                publish_mode = "hold"
            else:
                return
        else:
            return

        out = PoseWithCovarianceStamped()
        out.header.stamp = self.last_detect_stamp_msg if self.last_detect_stamp_msg is not None else now.to_msg()
        out.header.frame_id = FRAME_MAP

        out.pose.pose.position.x = float(pub_x)
        out.pose.pose.position.y = float(pub_y)
        out.pose.pose.position.z = 0.0

        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, float(pub_yaw))
        out.pose.pose.orientation.x = float(qx)
        out.pose.pose.orientation.y = float(qy)
        out.pose.pose.orientation.z = float(qz)
        out.pose.pose.orientation.w = float(qw)

        # Covarianza simple, algo conservadora
        cov = [0.0] * 36
        cov[0] = 0.15 ** 2
        cov[7] = 0.15 ** 2
        cov[14] = 10.0 ** 2
        cov[21] = 999.0
        cov[28] = 999.0
        cov[35] = np.deg2rad(8.0) ** 2
        out.pose.covariance = cov

        self.pose_pub.publish(out)

        out_viz = PoseWithCovarianceStamped()
        out_viz.header = out.header
        out_viz.pose.pose.position.x = out.pose.pose.position.x
        out_viz.pose.pose.position.y = out.pose.pose.position.y
        out_viz.pose.pose.position.z = out.pose.pose.position.z

        viz_yaw = wrap_angle(pub_yaw + np.pi)
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, float(viz_yaw))
        out_viz.pose.pose.orientation.x = float(qx)
        out_viz.pose.pose.orientation.y = float(qy)
        out_viz.pose.pose.orientation.z = float(qz)
        out_viz.pose.pose.orientation.w = float(qw)
        out_viz.pose.covariance = list(out.pose.covariance)

        self.pose_viz_pub.publish(out_viz)



def main():
    rclpy.init()
    node = DownCameraLocalization()
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