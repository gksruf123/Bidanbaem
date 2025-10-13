import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class SmoothCmdVel(Node):
    def __init__(self):
        super().__init__('smooth_cmd_vel')

        # ✅ 파라미터 (필요하면 launch에서 바꿀 수 있음)
        self.declare_parameter('rate_hz', 30.0)
        self.declare_parameter('max_accel_linear', 2.4 * 1.0)   # m/s²
        self.declare_parameter('max_accel_angular', 4.5 * 2.5)  # rad/s²


        # ✅ 파라미터 로드
        self.rate_hz = self.get_parameter('rate_hz').get_parameter_value().double_value
        self.max_accel_linear = self.get_parameter('max_accel_linear').get_parameter_value().double_value
        self.max_accel_angular = self.get_parameter('max_accel_angular').get_parameter_value().double_value

        # ✅ 현재/목표 Twist
        self.current_twist = Twist()
        self.target_twist = Twist()

        # ✅ Subscriber / Publisher
        self.create_subscription(Twist, '/cmd_vel_input', self.cmd_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/controller/cmd_vel', 10)

        # ✅ 주기적 타이머 (velocity smoothing loop)
        self.create_timer(1 / self.rate_hz, self.update_velocity)

        self.get_logger().info("✅ [smooth_cmd_vel] Node started (rate: %.1f Hz)" % self.rate_hz)

    def cmd_callback(self, msg: Twist):
        """ 목표 속도 갱신 """
        self.target_twist = msg

    def update_velocity(self):
        """ 현재 속도를 목표 속도로 부드럽게 보간 """
        dt = 1.0 / self.rate_hz

        # ---- 선속도 (linear.x) ----
        diff_linear = self.target_twist.linear.x - self.current_twist.linear.x
        # if diff_linear < 0:
        #     diff_linear *= 1.5
        max_delta_lin = self.max_accel_linear * dt
        diff_linear = max(-max_delta_lin, min(max_delta_lin, diff_linear))
        self.current_twist.linear.x += diff_linear

        # ---- 각속도 (angular.z) ----
        diff_angular = self.target_twist.angular.z - self.current_twist.angular.z
        max_delta_ang = self.max_accel_angular * dt
        diff_angular = max(-max_delta_ang, min(max_delta_ang, diff_angular))
        self.current_twist.angular.z += diff_angular

        self.current_twist.linear.y = self.target_twist.linear.y

        # ---- 퍼블리시 ----
        self.cmd_pub.publish(self.current_twist)

def main(args=None):
    rclpy.init(args=args)
    node = SmoothCmdVel()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 [smooth_cmd_vel] Node stopped by user.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
