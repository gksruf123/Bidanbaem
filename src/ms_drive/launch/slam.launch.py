from launch import LaunchDescription, LaunchService
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os
from launch.actions import TimerAction


def generate_launch_description():
    
    compiled = os.environ['need_compile']
    if compiled == 'True':
        peripherals_package_path = get_package_share_directory('peripherals')
        controller_package_path = get_package_share_directory('controller')
        package_share_directory = get_package_share_directory('ms_drive')
        navigation_package_path = get_package_share_directory('nav2_bringup')
        bringup_package_path = get_package_share_directory('bringup')
    else:
        peripherals_package_path = '/home/ubuntu/ros2_ws/src/peripherals'
        controller_package_path = '/home/ubuntu/ros2_ws/src/driver/controller'
        package_share_directory = '/home/ubuntu/ros2_ws/src/ms_drive'
        navigation_package_path = '/home/ubuntu/ros2_ws/src/nav2_bringup'
        bringup_package_path = '/home/ubuntu/ros2_ws/src/bringup'

    package_share_directory = get_package_share_directory('ms_drive')

    slam_params = os.path.join(package_share_directory, 'config', 'slam_toolbox_params.yaml')
    ekf_params = os.path.join(package_share_directory, 'config', 'ekf.yaml')

    return LaunchDescription([
        # 1️⃣ SLAM Toolbox
        Node(
            package='ms_drive',
            executable='lidar_fixed',
            name='lider_fixed',
            output='screen',
        ),
                
        TimerAction(
            period=8.0,  # or 5.0 if your LiDAR spins slowly
            actions=[
                Node(
                    package='slam_toolbox',
                    executable='async_slam_toolbox_node',
                    name='slam_toolbox',
                    output='screen',
                    parameters=[slam_params],
                )
            ]
        ),
        # Node(
        #     package='slam_toolbox',
        #     executable='sync_slam_toolbox_node',
        #     name='slam_toolbox',
        #     output='screen',
        #     parameters=[{
        #         'use_sim_time': False,
        #         'base_frame': 'base_link',
        #         'odom_frame': 'odom',
        #         'map_frame': 'map',
        #         'scan_topic': '/scan_raw',
        #         'mode': 'localization',  # Change to 'localization' for pure odometry
        #         'do_loop_closing': True,
        #         'resolution': 0.05,
        #         'publish_map_transform': True,
        #     }]
        # ),

        # 2️⃣ EKF localization (fuse wheel + laser)
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[ekf_params]
        ),

        # 3️⃣ RViz visualization
        # Node(
        #     package='rviz2',
        #     executable='rviz2',
        #     name='rviz2',
        #     output='screen',
        #     arguments=['-d', os.path.join(package_share_directory, 'config', 'slam_nav.rviz')],
        #     condition=None  # optional if you have rviz config
        # ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(bringup_package_path, 'launch', 'bringup.launch.py')
            )
        ),
    ])

if __name__ == '__main__':
    # 创建一个LaunchDescription对象
    ld = generate_launch_description()

    ls = LaunchService()
    ls.include_launch_description(ld)
    ls.run()

