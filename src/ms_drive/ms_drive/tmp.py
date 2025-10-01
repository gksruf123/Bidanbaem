def simple_move_to_target(self, target_x, target_y, current_odom):
    """Simple proportional controller to move towards target"""
    current_x = current_odom.pose.pose.position.x
    current_y = current_odom.pose.pose.position.y
    current_yaw = yaw_from_quaternion(current_odom.pose.pose.orientation)
    
    # Calculate angle to target
    target_yaw = math.atan2(target_y - current_y, target_x - current_x)
    yaw_error = target_yaw - current_yaw
    
    # Normalize angle error
    while yaw_error > math.pi:
        yaw_error -= 2 * math.pi
    while yaw_error < -math.pi:
        yaw_error += 2 * math.pi
    
    # Calculate distance to target
    distance = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
    
    # Simple proportional control
    cmd_vel = Twist()
    
    if distance > 0.1:  # Only move if > 10cm away
        if abs(yaw_error) > 0.2:  # Rotate first if misaligned
            cmd_vel.angular.z = 0.5 * yaw_error
        else:
            cmd_vel.linear.x = 0.2 * min(distance, 1.0)  # Limit speed
    else:
        # Target reached
        self.get_logger().info("Target reached!")
    
    self.wheel_pub.publish(cmd_vel)


#!/usr/bin/env python3
import json
import math
import threading
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from interfaces.msg import ObjectInfo, ObjectsInfo
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
from message_filters import Subscriber, ApproximateTimeSynchronizer
from geometry_msgs.msg import Twist, PoseStamped, Quaternion, Point
from nav_msgs.msg import Odometry
from rclpy.time import Time
import numpy as np
import queue

### Check camera transform ### 
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped

class PixelToOdomTransformer:
    def __init__(self, node):
        self.node = node
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, node)
        
        # Camera intrinsics
        self.fx = 576.5324096679688
        self.fy = 576.1083374023438  
        self.cx = 332.5771484375
        self.cy = 232.551513671875

    def pixel_to_camera_frame(self, u, v, depth_mm):
        """Convert pixel + depth to 3D point in camera optical frame"""
        depth_m = depth_mm * 0.001  # Convert mm to meters
        
        X = (u - self.cx) * depth_m / self.fx
        Y = (v - self.cy) * depth_m / self.fy
        Z = depth_m
        
        return X, Y, Z

    def camera_to_odom_frame(self, x_cam, y_cam, z_cam, timestamp=None):
        """Transform point from camera optical frame to odom frame"""
        point_in_camera = PointStamped()
        point_in_camera.header.stamp = timestamp or self.node.get_clock().now().to_msg()
        point_in_camera.header.frame_id = "depth_cam"  # Using your frame name
        point_in_camera.point.x = x_cam
        point_in_camera.point.y = y_cam
        point_in_camera.point.z = z_cam

        try:
            # Try to get transform - use latest available if timestamp causes issues
            transform = self.tf_buffer.lookup_transform(
                "odom",
                "depth_cam",
                rclpy.time.Time(),  # Use latest transform
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            point_in_odom = tf2_geometry_msgs.do_transform_point(point_in_camera, transform)
            return point_in_odom.point
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            self.node.get_logger().warn(f"TF transform failed: {e}")
            return None

    def pixel_to_odom(self, u, v, depth_mm, timestamp=None):
        """Complete pipeline: pixel + depth -> odom coordinates"""
        # Step 1: Pixel to camera frame
        x_cam, y_cam, z_cam = self.pixel_to_camera_frame(u, v, depth_mm)
        
        # Step 2: Camera frame to odom frame
        odom_point = self.camera_to_odom_frame(x_cam, y_cam, z_cam, timestamp)
        
        return odom_point

def yaw_to_quaternion(yaw):
    # yaw in radians
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw/2.0)
    q.w = math.cos(yaw/2.0)
    return q

def yaw_from_quaternion(q):
    # q is geometry_msgs.msg.Quaternion
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def yaw_from_quaternion_deg(q):
    # q is geometry_msgs.msg.Quaternion
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(yaw)

class Navigation(Node):
    def __init__(self, name='MS_Self_Drive'):
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.name = name
        self.is_running = True
        self.image_queue = queue.Queue(1)
        self.msg_queue = queue.Queue(1)
        self.cv_bridge = CvBridge()

        # Initialize pixel to odom transformer
        self.pixel_transformer = PixelToOdomTransformer(self)

        self.odom_sub = Subscriber(self, Odometry, '/odom')
        self.rgb_sub = Subscriber(self, Image, '/ascamera/camera_publisher/rgb0/image')
        self.depth_sub = Subscriber(self, Image, '/ascamera/camera_publisher/depth0/image_raw')

        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.odom_sub],
            queue_size=6,
            slop=0.15
        )
        self.ts.registerCallback(self.direct)

        with open('LAB-cal.json', 'r') as f:
            d = json.load(f)
            self.lm = d['l_min']
            self.lM = d['l_max']
            self.am = d['a_min']
            self.aM = d['a_max']
            self.bm = d['b_min']
            self.bM = d['b_max']

        self.frame_count = 0
        self.last_print = time.time()
        self.qc = 0
        self.status = 'stop'
        self.last_target_time = 0
        self.target_timeout = 2.0  # seconds
        self.current_target = None
        self.navigation_enabled = True

        # Mecanum wheel control parameters
        self.max_linear_speed = 0.3  # m/s
        self.max_angular_speed = 0.8  # rad/s
        self.max_strafe_speed = 0.3   # m/s for sideways movement
        self.position_tolerance = 0.15  # meters
        self.angle_tolerance = 0.15    # radians
        
        # PID gains for mecanum control
        self.kp_linear = 0.8    # Proportional gain for linear movement
        self.kp_angular = 1.2   # Proportional gain for rotation
        self.kp_strafe = 0.6    # Proportional gain for strafing

        self.wheel_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.timer = self.create_timer(0.0, self.init_process)
        
        # Publisher for visualization
        self.target_point_pub = self.create_publisher(PoseStamped, '/target_point_odom', 10)

        self.get_logger().info("Mecanum wheel navigation ready!")

    def mecanum_move_to_target(self, target_x, target_y, current_odom):
        """
        Mecanum wheel control - can move in any direction simultaneously
        Returns True if target is reached, False otherwise
        """
        if target_x is None or target_y is None:
            return False

        current_pos = current_odom.pose.pose.position
        current_ori = current_odom.pose.pose.orientation
        
        current_x = current_pos.x
        current_y = current_pos.y
        current_yaw = yaw_from_quaternion(current_ori)
        
        # Calculate errors in robot frame
        dx_world = target_x - current_x
        dy_world = target_y - current_y
        
        # Transform world errors to robot frame
        dx_robot = dx_world * math.cos(current_yaw) + dy_world * math.sin(current_yaw)
        dy_robot = -dx_world * math.sin(current_yaw) + dy_world * math.cos(current_yaw)
        
        distance = math.sqrt(dx_robot**2 + dy_robot**2)
        
        cmd_vel = Twist()
        
        # Check if target is reached
        if distance < self.position_tolerance:
            self.get_logger().info("Target reached!")
            self.wheel_pub.publish(cmd_vel)
            return True
        
        # Mecanum control logic - move directly toward target
        # Calculate desired velocities in robot frame
        vx = self.kp_linear * dx_robot
        vy = self.kp_strafe * dy_robot
        
        # Limit speeds
        vx = max(min(vx, self.max_linear_speed), -self.max_linear_speed)
        vy = max(min(vy, self.max_strafe_speed), -self.max_strafe_speed)
        
        # For mecanum, we can use both vx and vy simultaneously
        cmd_vel.linear.x = vx
        cmd_vel.linear.y = vy
        
        # Add small angular correction if needed (optional)
        target_yaw = math.atan2(dy_robot, dx_robot)
        yaw_error = target_yaw
        if abs(yaw_error) > self.angle_tolerance:
            cmd_vel.angular.z = self.kp_angular * yaw_error
            cmd_vel.angular.z = max(min(cmd_vel.angular.z, self.max_angular_speed), -self.max_angular_speed)
        
        self.wheel_pub.publish(cmd_vel)
        
        self.get_logger().info(
            f"Moving: vx:{vx:.2f}, vy:{vy:.2f}, wz:{cmd_vel.angular.z:.2f}, "
            f"dist:{distance:.2f}m"
        )
        return False

    def mecanum_move_to_target_simple(self, target_x, target_y, current_odom):
        """
        Simplified version - just forward/backward and strafing without rotation
        Good for lane following
        """
        if target_x is None or target_y is None:
            return False

        current_pos = current_odom.pose.pose.position
        current_ori = current_odom.pose.pose.orientation
        
        current_x = current_pos.x
        current_y = current_pos.y
        current_yaw = yaw_from_quaternion(current_ori)
        
        # Calculate errors in robot frame
        dx_world = target_x - current_x
        dy_world = target_y - current_y
        
        # Transform to robot frame
        dx_robot = dx_world * math.cos(current_yaw) + dy_world * math.sin(current_yaw)
        dy_robot = -dx_world * math.sin(current_yaw) + dy_world * math.cos(current_yaw)
        
        distance = math.sqrt(dx_robot**2 + dy_robot**2)
        
        cmd_vel = Twist()
        
        if distance < self.position_tolerance:
            self.get_logger().info("Target reached!")
            self.wheel_pub.publish(cmd_vel)
            return True
        
        # Simple proportional control - prioritize forward movement
        if abs(dx_robot) > 0.1:  # If we need significant forward movement
            vx = self.kp_linear * dx_robot
            vx = max(min(vx, self.max_linear_speed), -self.max_linear_speed)
            cmd_vel.linear.x = vx
        
        # Add strafing to correct lateral position
        if abs(dy_robot) > 0.05:  # If we need lateral correction
            vy = self.kp_strafe * dy_robot
            vy = max(min(vy, self.max_strafe_speed), -self.max_strafe_speed)
            cmd_vel.linear.y = vy
        
        self.wheel_pub.publish(cmd_vel)
        
        self.get_logger().info(
            f"Simple move: vx:{cmd_vel.linear.x:.2f}, vy:{cmd_vel.linear.y:.2f}, "
            f"dist:{distance:.2f}m"
        )
        return False

    def stop_movement(self):
        """Stop the robot"""
        cmd_vel = Twist()
        self.wheel_pub.publish(cmd_vel)
        self.get_logger().info("Movement stopped")

    def print_odom(self, odom_m):
        stamp = odom_m.header.stamp.nanosec
        pos = odom_m.pose.pose.position
        ori = odom_m.pose.pose.orientation
        twist_l = odom_m.twist.twist.linear
        twist_a = odom_m.twist.twist.angular
        print(f'timestamp: {stamp}')
        print(f'pos: (x:{pos.x:.3f}, y:{pos.y:.3f})')
        print(f'ang: ({yaw_from_quaternion_deg(ori):.2f})')
        print(f'twist_l: (x:{twist_l.x:.3f}, y:{twist_l.y:.2f})')
        print(f'twist_a: (z:{twist_a.z:.2f})')

    def direct(self, rgb_m, dep_m, odom_m:Odometry):
        self.print_odom(odom_m)
        self.proc(rgb_m, dep_m, odom_m)

    def ff_mask(self, mask, point):
        _mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8)
        cv2.floodFill(mask, _mask, point, newVal=255, 
                      flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8))
        return _mask[1:-1, 1:-1]

    def proc(self, rgb_m, dep_m, odom_m:Odometry = None):
        image_bgr = self.cv_bridge.imgmsg_to_cv2(rgb_m, 'bgr8')
        image_dep = self.cv_bridge.imgmsg_to_cv2(dep_m, '16UC1')

        oh, ow = image_bgr.shape[:2]
        scale = 1/2
        work_size = (int(ow*scale), int(oh*scale))
        image_bgr = cv2.resize(image_bgr, work_size)
        image_dep = cv2.resize(image_dep, work_size)

        h, w = image_bgr.shape[:2]
        image_dep = cv2.inpaint(image_dep, (image_dep == 0).astype('uint8'), 2, cv2.INPAINT_TELEA)

        lower = np.array([self.lm, self.am, self.bm])
        upper = np.array([self.lM, self.aM, self.bM])
        lab_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        mask_lab = cv2.inRange(lab_img, lower, upper)

        # Lane detection (your existing code)
        seedpoint_l = None
        _x = w//2 - 1
        _y = h-1
        while _x > 0:
            if mask_lab[h-1, _x]:
                seedpoint_l = (_x, h-1)
                break
            _x -= 1
        if not seedpoint_l:
            while _y > 0:
                if mask_lab[_y, _x]:
                    seedpoint_l = (_x, _y)
                    break
                _y -= 1

        seedpoint_r = None
        _x = w//2
        _y = h-1
        while _x < w - 1:
            if mask_lab[h-1, _x]:
                seedpoint_r = (_x, h-1)
                break
            _x += 1
        if not seedpoint_r:
            while _y > 0:
                if mask_lab[_y, _x]:
                    seedpoint_r = (_x, _y)
                    break
                _y -= 1

        if seedpoint_l:
            ff_left = self.ff_mask(mask_lab, seedpoint_l)
        if seedpoint_r:
            ff_right = self.ff_mask(mask_lab, seedpoint_r)

        target_point = None
        target_depth = 0
        
        # Find target point between lanes
        if seedpoint_l and seedpoint_r:
            _step = 5
            _y = h//4
            while _y < h-1:
                ll = np.where(ff_left[_y, :] > 0)[0]
                lr = np.where(ff_right[_y, :] > 0)[0]
                if len(ll) and len(lr):
                    lx = ll[-1]
                    rx = lr[0]
                    if rx - lx < 100:
                        target_point = None
                        break
                    else:
                        target_point = ((lx+rx)//2, _y)
                        target_depth = image_dep[_y, (lx+rx)//2]
                    cv2.line(mask_lab, (lx, _y), (rx, _y), 128, 2)
                    break
                _y += _step

        # Convert mask to BGR for visualization
        out = cv2.cvtColor(mask_lab, cv2.COLOR_GRAY2BGR)
        
        # Process target point with pixel-to-odom transformation
        current_time = time.time()
        target_odom_x = None
        target_odom_y = None
        
        if target_point and target_depth > 0:
            u, v = target_point
            
            # Scale coordinates back to original image size
            u_original = int(u / scale)
            v_original = int(v / scale)
            
            # Transform to odom frame
            odom_point = self.pixel_transformer.pixel_to_odom(u_original, v_original, target_depth, rgb_m.header.stamp)
            
            if odom_point:
                target_odom_x = odom_point.x
                target_odom_y = odom_point.y
                
                # Check if this is a new target
                is_new_target = (
                    self.current_target is None or
                    current_time - self.last_target_time > self.target_timeout or
                    math.sqrt((odom_point.x - self.current_target[0])**2 + 
                             (odom_point.y - self.current_target[1])**2) > 0.3
                )
                
                if is_new_target and self.navigation_enabled:
                    self.current_target = (odom_point.x, odom_point.y)
                    self.last_target_time = current_time
                
                # Use mecanum control to move to target
                if self.navigation_enabled and self.current_target:
                    target_reached = self.mecanum_move_to_target_simple(
                        self.current_target[0], self.current_target[1], odom_m
                    )
                    
                    if target_reached:
                        self.current_target = None
                
                # Display info
                cv2.putText(out, f'Odom: ({odom_point.x:.2f}, {odom_point.y:.2f})', 
                           (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(out, 'MECANUM NAV', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Publish for visualization
                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = "odom"
                pose_msg.pose.position = odom_point
                pose_msg.pose.orientation.w = 1.0
                self.target_point_pub.publish(pose_msg)

            cv2.circle(out, target_point, 5, (0, 0, 255), -1)
            cv2.putText(out, f'd:{target_depth}', target_point,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        else:
            # No target detected - stop
            if self.navigation_enabled:
                self.stop_movement()
                cv2.putText(out, 'NO TARGET', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display
        out = cv2.resize(out, (640, 480), interpolation=cv2.INTER_NEAREST_EXACT)
        pos = odom_m.pose.pose.position
        ori = odom_m.pose.pose.orientation
        
        cv2.putText(out, f'pos: (x:{pos.x:.3f}, y:{pos.y:.3f})', (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
        cv2.putText(out, f'ang: ({yaw_from_quaternion_deg(ori):.2f})', (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
        
        cv2.imshow('lab mask', out)
        cv2.waitKey(1)

    def init_process(self):
        self.timer.cancel() 
        self.stop_movement()

def main():
    cv2.namedWindow('lab mask')
    cv2.moveWindow('lab mask', 500, 0)
    rclpy.init(args=None)
    node = Navigation()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_movement()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()