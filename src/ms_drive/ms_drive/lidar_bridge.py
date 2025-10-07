import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

class ScanRelay(Node):
    def __init__(self):
        super().__init__('scan_relay')
        self.sub = self.create_subscription(
            LaserScan, '/scan_raw',
            self.cb,
            10)  # reliable (default)
        self.pub = self.create_publisher(
            LaserScan, '/scan_best_effort',
            rclpy.qos.qos_profile_sensor_data)  # best effort

    def cb(self, msg):
        self.pub.publish(msg)

rclpy.init()
rclpy.spin(ScanRelay())

