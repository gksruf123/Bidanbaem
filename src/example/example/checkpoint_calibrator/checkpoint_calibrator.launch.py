import os
from ament_index_python.packages import get_package_share_directory

from launch_ros.actions import Node
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, OpaqueFunction

def launch_setup(context):
    compiled = os.environ.get('need_compile', 'False')
    if compiled == 'True':
        peripherals_package_path = get_package_share_directory('peripherals')
        package_share_directory = get_package_share_directory('yolov5_ros2')
    else:
        peripherals_package_path = '/home/ubuntu/ros2_ws/src/peripherals'
        package_share_directory = '/home/ubuntu/ros2_ws/src/yolov5_ros2'

    depth_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(peripherals_package_path, 'launch/depth_camera.launch.py')),
    )

    yolov5_node = Node(
        package='yolov5_ros2',
        executable='yolo_detect',
        output='screen',
        parameters=[{
            'classes': ['go', 'right', 'park', 'red', 'green'],
            'device': 'cpu',
            'model': 'best',
            'image_topic': '/ascamera/camera_publisher/rgb0/image',
            'camera_info_topic': '/camera/camera_info',
            'camera_info_file': f"{package_share_directory}/config/camera_info.yaml",
            'pub_result_img': True,
        }]
    )

    calibrator_node = Node(
        package='example',
        executable='checkpoint_calibrator',
        output='screen',
        parameters=[{
            'rgb_topic': '/ascamera/camera_publisher/rgb0/image',
            'depth_topic': '/ascamera/camera_publisher/depth0/image_raw',
            'classes': ['go', 'right', 'park', 'red', 'green'],
            'hfov_deg': 69.0,
            'samples': 30,
        }]
    )

    return [depth_camera_launch, yolov5_node, calibrator_node]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch_setup)
    ])
