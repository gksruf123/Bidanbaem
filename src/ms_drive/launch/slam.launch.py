from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('ms_drive')

    slam_params = os.path.join(pkg_share, 'config', 'slam_toolbox_params.yaml')
    ekf_params = os.path.join(pkg_share, 'config', 'ekf.yaml')

    return LaunchDescription([
        # 1️⃣ SLAM Toolbox
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[slam_params]
        ),

        # 2️⃣ EKF localization (fuse wheel + laser)
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[ekf_params]
        ),

        # 3️⃣ RViz visualization
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', os.path.join(pkg_share, 'config', 'slam_nav.rviz')],
            condition=None  # optional if you have rviz config
        ),
    ])
