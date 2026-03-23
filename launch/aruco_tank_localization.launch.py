from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('tandem_girona')
    ekf_yaml = os.path.join(pkg_share, 'config', 'ekf_aruco.yaml')

    sim = {'use_sim_time': False}

    return LaunchDescription([
        # Downward camera localization
        Node(
            package='tandem_girona',
            executable='down_camera_localization_matrix',
            name='down_camera_localization_matrix',
            output='screen',
            parameters=[sim],
        ),

        # Bridge: ArUco pose -> /localizer/relocalize service
        Node(
            package='tandem_girona',
            executable='aruco_to_relocalize',
            name='aruco_to_relocalize',
            output='screen',
            parameters=[{
                'aruco_pose_topic': '/blueboat/navigator/aruco_pose',
                'trigger_topic': '/blueboat/navigator/aruco_relocalize_trigger',
                'relocalize_service': '/localizer/relocalize',
                'relocalize_check_service': '/localizer/relocalize_check',

                'pcd_path': '/home/mariolopez31/cirtesu_ws/src/fast_lio/PCD/sim_cirtesu.pcd',

                'auto_trigger_on_first_pose': False,
                'check_success_after_call': True,
                'check_delay_sec': 1.0,

                # 2D
                'fixed_z': 0.0,
                'fixed_roll': 0.0,
                'fixed_pitch': 0.0,

                'x_offset': 0.0,
                'y_offset': 0.0,
                'yaw_offset': 0.0,

                'max_pose_age_sec': 2.0,
                'require_pose_before_trigger': True,
            }],
        ),

        # EKF si luego lo quieres activar
        # Node(
        #     package='robot_localization',
        #     executable='ekf_node',
        #     name='ekf_aruco',
        #     output='screen',
        #     parameters=[sim, ekf_yaml],
        #     remappings=[("odometry/filtered", "odometry/aruco_filtered")],
        # ),
    ])