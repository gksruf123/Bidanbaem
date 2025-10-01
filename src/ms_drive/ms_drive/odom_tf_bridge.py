#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

class OdometryToTF(Node):
    def __init__(self):
        super().__init__('odometry_to_tf')
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        self.get_logger().info("Converting /odom messages to TF transforms with CURRENT time")

    def odom_callback(self, msg):
        t = TransformStamped()
        
        # Use CURRENT time instead of bag time to avoid "data from the past" errors
        t.header.stamp = self.get_clock().now().to_msg()  # CHANGED: Current time
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        
        # Copy position from odometry
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        
        # Copy orientation from odometry
        t.transform.rotation = msg.pose.pose.orientation
        
        self.tf_broadcaster.sendTransform(t)
        
        # Log for debugging
        self.get_logger().info(
            f"Published TF: X: {t.transform.translation.x:.2f}, Y: {t.transform.translation.y:.2f}", 
            throttle_duration_sec=1.0
        )

def main():
    rclpy.init()
    node = OdometryToTF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()