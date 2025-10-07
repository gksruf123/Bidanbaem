from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import rclpy

class ScanFixer(Node):
    def __init__(self):
        super().__init__('scan_fixer')
        self.sub = self.create_subscription(LaserScan, '/scan_raw', self.cb, 5)
        self.pub = self.create_publisher(LaserScan, '/scan_fixed', 1)
        self.expected_len = 503
        self.ready = False

    def cb(self, msg):
        n = len(msg.ranges)
        # Wait until we see a "stable" scan
        if n != self.expected_len:
            if n > self.expected_len:
                msg.ranges = msg.ranges[:self.expected_len]
            else:
                msg.ranges.extend([msg.ranges[-1]] * (self.expected_len - n))

        # Only start publishing after first stable scan
        if not self.ready:
            self.ready = True
            self.get_logger().info(f"Publishing first stabilized scan with {len(msg.ranges)} ranges")

        if self.ready:
            self.pub.publish(msg)

def main():
    rclpy.init()
    node = ScanFixer()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
