#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, PoseStamped, Quaternion, Point
from tf2_ros import TransformException, Buffer, TransformListener
import tf2_geometry_msgs
from cv_bridge import CvBridge
import numpy as np
import math

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
        
        self.get_logger().info("Depth processor - handling optical frame correctly")

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

        # Method 1: Try using camera's optical frame (if it exists)
        optical_frame_success = self.try_optical_frame(u, v, depth_meters, msg.header.stamp)
        
        if not optical_frame_success:
            # Method 2: Manual transformation from optical to link frame, then TF to odom
            self.manual_transformation(u, v, depth_meters, msg.header.stamp)

    def try_optical_frame(self, u, v, depth_meters, stamp):
        """Try to use the camera's optical frame directly"""
        optical_frame_id = "depth_cam_optical_frame"  # Common naming convention
        
        # Project to 3D in camera optical frame
        Z_cam = depth_meters  # Forward (depth)
        X_cam = (u - self.cx) * Z_cam / self.fx  # Right
        Y_cam = (v - self.cy) * Z_cam / self.fy  # Down

        point_in_optical = PointStamped()
        point_in_optical.header.stamp = stamp
        point_in_optical.header.frame_id = optical_frame_id
        point_in_optical.point.x = X_cam
        point_in_optical.point.y = Y_cam  
        point_in_optical.point.z = Z_cam

        try:
            transform = self.tf_buffer.lookup_transform(
                'odom',
                optical_frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )

            point_in_odom = tf2_geometry_msgs.do_transform_point(point_in_optical, transform)
            
            self.get_logger().info(
                f"Optical frame SUCCESS - Odom: X: {point_in_odom.point.x:.3f}, "
                f"Y: {point_in_odom.point.y:.3f}, Z: {point_in_odom.point.z:.3f}"
            )
            
            self.publish_point(point_in_odom.point)
            return True
            
        except TransformException as e:
            self.get_logger().warn(f'Optical frame transform failed: {e}')
            return False

    
    def manual_transformation(self, u, v, depth_meters, stamp):
        """Manually handle optical to link frame transformation, then use TF"""
        # Step 1: Project to camera optical frame
        Z_optical = depth_meters  # Forward in optical frame
        X_optical = (u - self.cx) * Z_optical / self.fx  # Right in optical frame  
        Y_optical = (v - self.cy) * Z_optical / self.fy  # Down in optical frame

        # Step 2: Convert from optical frame to camera link frame
        # Optical frame: X=right, Y=down, Z=forward
        # Link frame: X=forward, Y=left, Z=up
        X_link = Z_optical   # Optical Z (forward) -> Link X (forward)
        Y_link = -X_optical  # Optical X (right) -> Link Y (left) 
        Z_link = -Y_optical  # Optical Y (down) -> Link Z (up)

        self.get_logger().info(
            f"Manual transform - Optical: ({X_optical:.3f}, {Y_optical:.3f}, {Z_optical:.3f}) -> "
            f"Link: ({X_link:.3f}, {Y_link:.3f}, {Z_link:.3f})"
        )

        # Step 3: Create point in camera link frame and transform to odom
        point_in_link = PointStamped()
        point_in_link.header.stamp = stamp
        point_in_link.header.frame_id = "depth_cam"  # Camera link frame
        point_in_link.point.x = X_link
        point_in_link.point.y = Y_link
        point_in_link.point.z = Z_link

        try:
            transform = self.tf_buffer.lookup_transform(
                'odom',
                'depth_cam',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )

            point_in_odom = tf2_geometry_msgs.do_transform_point(point_in_link, transform)
            
            self.get_logger().info(
                f"Manual SUCCESS - Odom: X: {point_in_odom.point.x:.3f}, "
                f"Y: {point_in_odom.point.y:.3f}, Z: {point_in_odom.point.z:.3f}"
            )
            
            self.publish_point(point_in_odom.point)
            
        except TransformException as e:
            self.get_logger().warn(f'Link frame transform failed: {e}')

    def publish_point(self, point):
        """Publish the transformed point for visualization"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "odom"
        pose_msg.pose.position = point
        pose_msg.pose.orientation.w = 1.0
        self.point_pub.publish(pose_msg)

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