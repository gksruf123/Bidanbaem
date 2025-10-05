#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PoseWithCovariance, Twist, TwistWithCovariance
from util import *
import math

class CorrectedOdomNode(Node):
    def __init__(self):
        super().__init__('corrected_odom_node')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.odom_pub = self.create_publisher(Odometry, '/odom_corrected', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz

    def timer_callback(self):
        try:
            map_to_odom = self.tf_buffer.lookup_transform('map', 'odom', rclpy.time.Time())
            odom_to_base = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time())

            # Wheel odometry translation
            wheel_pos = [odom_to_base.transform.translation.x,
                         odom_to_base.transform.translation.y,
                         odom_to_base.transform.translation.z]

            # SLAM correction translation
            slam_trans = [map_to_odom.transform.translation.x,
                          map_to_odom.transform.translation.y,
                          map_to_odom.transform.translation.z]

            # Quaternions
            q_map_odom = [map_to_odom.transform.rotation.x,
                          map_to_odom.transform.rotation.y,
                          map_to_odom.transform.rotation.z,
                          map_to_odom.transform.rotation.w]
            q_odom_base = [odom_to_base.transform.rotation.x,
                           odom_to_base.transform.rotation.y,
                           odom_to_base.transform.rotation.z,
                           odom_to_base.transform.rotation.w]

            # Rotate wheel odom translation by map->odom rotation
            corrected_trans = rotate_vector_by_quaternion(wheel_pos, q_map_odom)
            # Add translation of map->odom
            corrected_trans = [corrected_trans[0] + slam_trans[0],
                               corrected_trans[1] + slam_trans[1],
                               corrected_trans[2] + slam_trans[2]]

            # Corrected rotation
            q_corrected = quaternion_multiply(q_map_odom, q_odom_base)

            # Publish Odometry message
            odom_msg = Odometry()
            odom_msg.header.stamp = self.get_clock().now().to_msg()
            odom_msg.header.frame_id = 'map'
            odom_msg.child_frame_id = 'base_link'
            odom_msg.pose.pose.position.x = corrected_trans[0]
            odom_msg.pose.pose.position.y = corrected_trans[1]
            odom_msg.pose.pose.position.z = corrected_trans[2]
            odom_msg.pose.pose.orientation.x = q_corrected[0]
            odom_msg.pose.pose.orientation.y = q_corrected[1]
            odom_msg.pose.pose.orientation.z = q_corrected[2]
            odom_msg.pose.pose.orientation.w = q_corrected[3]
            odom_msg.twist.twist = Twist()  # zero twist for now
            self.odom_pub.publish(odom_msg)

            # --- Logging ---
            self.get_logger().info(
                f"[Wheel Odom] x={wheel_pos[0]:.2f}, y={wheel_pos[1]:.2f} | "
                f"[SLAM Corr] x={slam_trans[0]:.2f}, y={slam_trans[1]:.2f} | "
                f"[Corrected] x={corrected_trans[0]:.2f}, y={corrected_trans[1]:.2f}"
            )

        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            
def main(args=None):
    rclpy.init(args=args)
    node = CorrectedOdomNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()