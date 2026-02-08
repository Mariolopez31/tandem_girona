# tandem_girona (ROS 2)

This branch (**ros2**) contains the **ROS 2 port** of the original `tandem_girona` repository.  
It is adapted for the **BlueBoat** robot model (Stonefish-style simulation stack) and the CIRTESU environment.

## Goal

- Port the original stack to **ROS 2**.
- Provide a **localization pipeline** based on:
  - **Absolute pose** from ArUco (downward-facing camera).
  - **Odometry** from simulation (Stonefish).
  - **Fusion** using `robot_localization` (EKF).

## Main changes vs ROS 1 version

- Nodes and launch files migrated to **ROS 2**.
- Updated **frames** and naming for BlueBoat:
  - `base_link`: `blueboat/base_link`
  - `odom`: `world_ned`
  - `map`: `cirtesu_base_link` (global frame of the CIRTESU environment)
- EKF configured to fuse:
  - `pose0` from ArUco: `/blueboat/navigator/aruco_pose`
  - `odom0` from Stonefish: `/blueboat/navigator/odometry`
- Publishes a **fixed ArUco map** (RViz markers) and estimates robot pose from detections.

## Frames

This stack assumes the following conceptual frame setup:

- `world_ned` → NED world frame (simulator/world origin)
- `cirtesu_base_link` → global frame of the CIRTESU installation/environment
- `blueboat/base_link` → BlueBoat robot base frame

The launch file publishes a static transform:

- `world_ned` → `cirtesu_base_link` (with yaw ≈ π)

> Note: In this repository, `cirtesu_base_link` is treated as the global map/world frame of the environment.

## EKF (robot_localization)

Key configuration (excerpt):

- `map_frame: cirtesu_base_link`
- `odom_frame: world_ned`
- `base_link_frame: blueboat/base_link`
- `world_frame: world_ned`
- `publish_tf: false` (avoid publishing TF from EKF to prevent duplicate/conflicting TFs)
- Inputs:
  - **pose0 (ArUco absolute pose)**: `/blueboat/navigator/aruco_pose`
  - **odom0 (Stonefish odometry)**: `/blueboat/navigator/odometry`

The intent is to use ArUco as an absolute correction (x, y, yaw) while odometry provides continuous motion tracking.

## Launch

Main launch (high level):

- Publishes static TF `world_ned -> cirtesu_base_link`
- Starts the downward camera localization node
- Starts `ekf_node` using `ekf_aruco.yaml`

Typical file: `launch/<...>.launch.py`

## ArUco map localization (downward camera)

ROS 2 node that:

1. Publishes a `MarkerArray` with:
   - A CIRTESU **mesh marker**.
   - **Fixed ArUco markers** defined in `ARUCO_MAP` (in `cirtesu_base_link`).
   - Highlights visible markers in yellow based on detections.

2. Estimates robot pose using:
   - TF `base_link <- camera_link`
   - ArUco detections (`aruco_opencv_msgs/ArucoDetection`)
   - Weighted averaging (weight ~ 1/(dist² + c)) for position and yaw.

### Topics

- Sub:
  - `/blueboat/down_camera/aruco_detections`
- Pub:
  - `/blueboat/aruco_map_markers` (RViz)
  - `/blueboat/navigator/aruco_pose` (`PoseWithCovarianceStamped`)

### Parameters / constants to review

- `ARUCO_MAP`: fixed marker positions (in `cirtesu_base_link`)
- `ARUCO_YAW_OFFSET`: yaw offset to match camera/detector conventions
- `camera_frame`: `blueboat_camera_link`
- `base_frame`: `blueboat/base_link`

## Quick usage

From your ROS 2 workspace:

```bash
colcon build --symlink-install
source install/setup.bash
