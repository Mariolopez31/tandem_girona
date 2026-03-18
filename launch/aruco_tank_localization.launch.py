from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('tandem_girona')
    ekf_yaml = os.path.join(pkg_share, 'config', 'ekf_aruco.yaml')
    
    sim = {'use_sim_time': False}

    return LaunchDescription([
        # your world_frame -> cirtesu_base_link
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='nedtocirtesu',
            arguments=[
                '--x', '0', '--y', '0', '--z', '0',
                '--yaw', '3.1416', '--pitch', '0', '--roll', '3.1416',
                '--frame-id', 'map',
                '--child-frame-id', 'cirtesu_base_link',
            ],
            output='screen',
        ),

        # Downward camera localization
        Node(
            package='tandem_girona',
            executable='down_camera_localization_matrix',
            name='down_camera_localization_matrix',
            output='screen',
            parameters=[sim],
        ),

        # EKF
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_aruco',
            output='screen',
            parameters=[sim, ekf_yaml],
            remappings=[("odometry/filtered", "odometry/aruco_filtered")],
        ),
    ])
