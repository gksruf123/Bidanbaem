#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, PoseStamped
from tf2_ros import TransformException, Buffer, TransformListener
import tf2_geometry_msgs
from cv_bridge import CvBridge
import numpy as np

class DepthPixelToOdom(Node):
    def __init__(self):
        super().__init__('depth_pixel_to_odom')
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.bridge = CvBridge()

        # Your camera intrinsics
        self.fx = 576.5324096679688
        self.fy = 576.1083374023438  
        self.cx = 332.5771484375
        self.cy = 232.551513671875

        # Subscribe to depth image
        self.depth_image_sub = self.create_subscription(
            Image,
            '/ascamera/camera_publisher/depth0/image_raw',
            self.depth_image_callback,
            10)

        self.target_pixel = (320, 240)
        self.point_pub = self.create_publisher(PoseStamped, '/transformed_point', 10)
        
        self.get_logger().info("Depth processor ready - waiting for odometry TF...")

    def depth_image_callback(self, msg):
        try:
            if msg.encoding == '16UC1':
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
                depth_scale = 0.001
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
                depth_scale = 1.0
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        u, v = self.target_pixel
        height, width = cv_image.shape
        
        if u >= width or v >= height or u < 0 or v < 0:
            return

        raw_depth = cv_image[v, u]
        if raw_depth == 0:
            return

        depth_meters = raw_depth * depth_scale
        if np.isnan(depth_meters) or np.isinf(depth_meters) or depth_meters <= 0.0:
            return

        # Project to 3D in camera frame
        point_in_optical_frame = PointStamped()
        point_in_optical_frame.header.stamp = msg.header.stamp
        point_in_optical_frame.header.frame_id = "depth_cam"

        Z = depth_meters
        point_in_optical_frame.point.x = (u - self.cx) * Z / self.fx
        point_in_optical_frame.point.y = (v - self.cy) * Z / self.fy  
        point_in_optical_frame.point.z = Z

        # Transform to odom frame
        try:
            transform = self.tf_buffer.lookup_transform(
                'odom',
                'depth_cam', 
                rclpy.time.Time(),  # Latest available transform
                timeout=rclpy.duration.Duration(seconds=0)
            )

            point_in_odom_frame = tf2_geometry_msgs.do_transform_point(
                point_in_optical_frame, transform)

            self.get_logger().info(
                f"Pixel {self.target_pixel} | Depth: {raw_depth}mm -> "
                f"Odom: X: {point_in_odom_frame.point.x:.3f}, "
                f"Y: {point_in_odom_frame.point.y:.3f}, "
                f"Z: {point_in_odom_frame.point.z:.3f}"
            )

            # Publish for visualization
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "odom"
            pose_msg.pose.position = point_in_odom_frame.point
            pose_msg.pose.orientation.w = 1.0
            self.point_pub.publish(pose_msg)

        except TransformException as e:
            self.get_logger().warn(f'TF transform not ready: {e}')

def main():
    rclpy.init()
    node = DepthPixelToOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()