from functools import partial
import math
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import Image
from interfaces.msg import ObjectsInfo
from geometry_msgs.msg import Twist, PoseStamped, Quaternion, Point
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data, QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import queue

class MSMDriver(Node):
    def __init__(self, name='ms_driver'):
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.name = name
        self.is_running = True

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.rcbg = ReentrantCallbackGroup()
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, qos_profile=qos, callback_group=self.rcbg)
        self.logic_sub = self.create_subscription(Point, '/ms_logic', self.logic_cb, qos_profile=qos, callback_group=self.rcbg)

        self.cmd_vel_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)

        self.max_linear_speed = 0.5
        self.max_strafe_speed = 0.3
        self.max_angular_speed = 3.0

        self.max_linear_accel = 0.8
        self.max_strafe_accel = self.max_linear_accel / 0.8
        self.max_angular_accel = 1.0

        self.cvx = 0.0
        self.cvy = 0.0
        self.caz = 0.0
        self.current_theta = 0.0  # Added missing variable

        self.tvx = 0.0
        self.tvy = 0.0
        self.taz = 0.0

        self.target = queue.Queue(1)
        self.lt = self.get_clock().now()
        self.init_timer = self.create_timer(0.01, self.init_process)  # Use different variable name
        self.tick_timer = self.create_timer(0.02, self.tick_speed)  # 50hz tick

    def enqueue(self, q:queue.Queue, item):
        try:
            q.put_nowait(item)
        except queue.Full:
            _ = q.get_nowait()
            q.put_nowait(item)
    
    def logic_cb(self, msg_logic): 
        partial(self.enqueue, self.target, msg_logic)()

    def normalize_angle(self, angle):  # Fixed: moved outside odom_cb
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def odom_cb(self, msg_odom:Odometry):
        def dist_sq(x1,y1,x2,y2):
            return (x1-x2)**2+(y1-y2)**2
            
        def yaw_from_quaternion(q):
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            return math.atan2(siny_cosp, cosy_cosp)
            
        pose = msg_odom.pose.pose
        pos = pose.position
        ori = pose.orientation
        
        if self.target.empty():  # Fixed: check if queue is empty
            self.stop()
            return
            
        try:
            target = self.target.get_nowait()  # Use get_nowait to avoid blocking
            self.target.task_done()
        except queue.Empty:
            self.stop()
            return
            
        cmd_vel = Twist()
        dx = target.x - pos.x
        dy = target.y - pos.y
        dist_squared = dx*dx + dy*dy  # Use squared distance for comparison
        self.current_theta = yaw_from_quaternion(ori)

        if dist_squared > (0.01**2):  # if dist > 10mm
            # Calculate direction vector and normalize
            distance = math.sqrt(dist_squared)
            direction_x = dx / distance
            direction_y = dy / distance
            
            # Scale by max speed
            speed = min(self.max_linear_speed, distance * 1.0)  # Simple P-control
            cmd_vel.linear.x = direction_x * speed
            cmd_vel.linear.y = direction_y * speed
            
            # Limit strafe speed
            if abs(cmd_vel.linear.y) > self.max_strafe_speed:
                cmd_vel.linear.y = math.copysign(self.max_strafe_speed, cmd_vel.linear.y)
        
        desired_angle = math.atan2(dy, dx)
        angle_delta = self.normalize_angle(desired_angle - self.current_theta)
        
        # Angular control with speed limiting
        angular_cmd = 1.2 * angle_delta
        cmd_vel.angular.z = max(-self.max_angular_speed, min(self.max_angular_speed, angular_cmd))

        self.set_target_speed(tw=cmd_vel)

    def limit(self, cur, targ, max_accel, dt):
        delta = targ - cur
        max_delta = max_accel * dt
        
        # Allow faster deceleration
        if delta < 0: 
            max_delta *= 2

        if abs(delta) <= max_delta:
            return targ
        else:
            return cur + math.copysign(max_delta, delta)  # Fixed: add to current value

    def tick_speed(self):
        ct = self.get_clock().now()
        dt = (ct - self.lt).nanoseconds / 1e9
        if dt <= 0:
            return

        vel = Twist()
        vel.linear.x = self.limit(self.cvx, self.tvx, self.max_linear_accel, dt)
        vel.linear.y = self.limit(self.cvy, self.tvy, self.max_strafe_accel, dt)
        vel.angular.z = self.limit(self.caz, self.taz, self.max_angular_accel, dt)  # Fixed: max_angular_accel

        # Update current speeds
        self.cvx = vel.linear.x
        self.cvy = vel.linear.y  
        self.caz = vel.angular.z

        self.cmd_vel_pub.publish(vel)
        self.lt = ct

    def set_target_speed(self, x=0.0, y=0.0, z=0.0, tw:Twist=None):
        if tw is not None:
            x = tw.linear.x
            y = tw.linear.y
            z = tw.angular.z
            
        # Apply speed limits
        self.tvx = max(-self.max_linear_speed, min(self.max_linear_speed, x))
        self.tvy = max(-self.max_strafe_speed, min(self.max_strafe_speed, y))
        self.taz = max(-self.max_angular_speed, min(self.max_angular_speed, z))

    def set_current_speed(self, x=0.0, y=0.0, z=0.0, tw:Twist=None):
        if tw is not None:
            x = tw.linear.x
            y = tw.linear.y
            z = tw.angular.z
        self.cvx = x
        self.cvy = y
        self.caz = z

    def stop(self):
        self.set_target_speed()
        
    def init_process(self):
        self.init_timer.cancel()  # Cancel the init timer
        self.panic_stop()

    def panic_stop(self):
        if not rclpy.ok():
            return
        t = Twist()
        self.set_target_speed()
        self.set_current_speed(tw=t)
        self.cmd_vel_pub.publish(t)

def main():
    rclpy.init()  # Moved rclpy.init here
    node = MSMDriver()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt received')
    finally:
        # Safe shutdown
        node.panic_stop()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()