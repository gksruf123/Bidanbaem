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

    def pixel_to_odom_direct(self, u, v, depth_mm, robot_odom):
        """
        Direct transformation from pixel to odom coordinates without TF
        Uses camera intrinsics + extrinsics + robot odometry
        """
        depth_m = depth_mm * 0.001  # Convert mm to meters
        
        # Step 1: Project pixel to 3D point in camera optical frame
        # Camera optical frame: X=right, Y=down, Z=forward
        Z_cam_optical = depth_m
        X_cam_optical = (u - self.cx) * Z_cam_optical / self.fx
        Y_cam_optical = (v - self.cy) * Z_cam_optical / self.fy
        
        # Step 2: Transform from camera optical frame to camera link frame
        # Optical: X=right, Y=down, Z=forward → Link: X=forward, Y=left, Z=up
        X_cam_link = Z_cam_optical    # Forward
        Y_cam_link = -X_cam_optical   # Left  
        Z_cam_link = -Y_cam_optical   # Up
        
        # Step 3: Apply camera mounting transformation
        # This accounts for camera position/orientation relative to robot base
        point_camera = np.array([X_cam_link, Y_cam_link, Z_cam_link])
        
        # Apply camera rotation (pitch - looking slightly down)
        cos_p = math.cos(self.camera_pitch)
        sin_p = math.sin(self.camera_pitch)
        point_rotated = np.array([
            point_camera[0],
            cos_p * point_camera[1] - sin_p * point_camera[2],
            sin_p * point_camera[1] + cos_p * point_camera[2]
        ])
        
        # Add camera position offset
        point_base = point_rotated + np.array([self.camera_x, self.camera_y, self.camera_z])
        
        # Step 4: Transform from robot base frame to odom frame
        robot_pos = robot_odom.pose.pose.position
        robot_ori = robot_odom.pose.pose.orientation
        
        # Get robot yaw from quaternion
        robot_yaw = self.quaternion_to_yaw(robot_ori)
        
        # Transform point from robot frame to world frame
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        
        # Rotate point to world frame and add robot position
        point_world = np.array([
            robot_pos.x + cos_yaw * point_base[0] - sin_yaw * point_base[1],
            robot_pos.y + sin_yaw * point_base[0] + cos_yaw * point_base[1],
            robot_pos.z + point_base[2]
        ])
        
        # Create Point message
        odom_point = Point()
        odom_point.x = point_world[0]
        odom_point.y = point_world[1] 
        odom_point.z = point_world[2]
        
        return odom_point

def yaw_to_quaternion(yaw):
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw/2.0)
    q.w = math.cos(yaw/2.0)
    return q

def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def yaw_from_quaternion_deg(q):
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
        self.status = 'lane_following'
        
        # Navigation control parameters
        self.current_target = None
        self.last_target_point = None  # Store the last detected target point
        self.road_direction = None
        self.last_target_time = 0
        self.target_timeout = 2.0
        self.last_target_persist_time = 5.0  # How long to keep using last target (seconds)
        
        # Control parameters for mecanum with road alignment
        self.max_linear_speed = 0.5
        self.max_strafe_speed = 0.5
        self.max_angular_speed = 0.2
        self.position_tolerance = 0.15
        self.lateral_tolerance = 0.08
        self.heading_tolerance = 0.5
        
        # PID gains
        self.kp_forward = 0.6
        self.kp_strafe = 0.8
        self.kp_heading = 1.0

        self.wheel_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.timer = self.create_timer(0.0, self.init_process)

        self.get_logger().info("Lane following with persistent target tracking started")

    def estimate_road_direction(self, ff_left, ff_right, image_shape, current_odom, image_dep):
        """
        Estimate road direction by analyzing lane points at different distances
        """
        h, w = image_shape
        road_points = []
        
        # Sample lane points at different distances (rows in the image)
        sample_rows = [h//4, h//3, h//2, 2*h//3, 3*h//4]
        
        for row in sample_rows:
            # Get left and right lane points at this row
            left_points = np.where(ff_left[row, :] > 0)[0]
            right_points = np.where(ff_right[row, :] > 0)[0]
            
            if len(left_points) > 0 and len(right_points) > 0:
                # Calculate lane center at this row
                lane_center = (left_points[-1] + right_points[0]) // 2
                
                # Convert to odom coordinates
                depth = image_dep[row, lane_center]
                if depth > 0:
                    odom_point = self.pixel_transformer.pixel_to_odom_direct(
                        lane_center, row, depth, current_odom
                    )
                    if odom_point:
                        road_points.append((odom_point.x, odom_point.y))
        
        # Calculate road direction from points
        if len(road_points) >= 2:
            # Use linear regression to find road direction
            x_coords = [p[0] for p in road_points]
            y_coords = [p[1] for p in road_points]
            
            if len(set(x_coords)) > 1:  # Ensure we have variation in x
                # Simple linear fit to get direction
                A = np.vstack([x_coords, np.ones(len(x_coords))]).T
                slope, _ = np.linalg.lstsq(A, y_coords, rcond=None)[0]
                road_direction = math.atan(slope)
                
                self.get_logger().info(f"Estimated road direction: {math.degrees(road_direction):.1f}°")
                return road_direction
        
        return None

    def follow_lane_with_alignment(self, target_x, target_y, road_direction, current_odom):
        """
        Follow lane while aligning robot heading with road direction
        """
        if target_x is None or target_y is None or road_direction is None:
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
        
        lateral_error = dy_robot
        forward_error = dx_robot
        
        # Calculate heading error (difference between current and road direction)
        heading_error = road_direction - current_yaw
        
        # Normalize heading error to [-pi, pi]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
        
        cmd_vel = Twist()
        
        # Check if target is reached
        if (abs(lateral_error) < self.lateral_tolerance and 
            abs(forward_error) < self.position_tolerance and
            abs(heading_error) < self.heading_tolerance):
            self.get_logger().info("Perfectly aligned with road!")
            self.wheel_pub.publish(cmd_vel)
            return True
        
        # Control strategy: Prioritize road alignment, then lateral position
        if abs(heading_error) > self.heading_tolerance:
            # First, align with road direction
            angular_speed = self.kp_heading * heading_error
            angular_speed = max(min(angular_speed, self.max_angular_speed), -self.max_angular_speed)
            cmd_vel.angular.z = angular_speed
            
            self.get_logger().info(
                f"Aligning to road: heading error {math.degrees(heading_error):.1f}°, "
                f"angular: {cmd_vel.angular.z:.2f} rad/s"
            )
        
        else:
            # Once aligned with road, focus on lateral position
            if abs(lateral_error) > self.lateral_tolerance:
                strafe_speed = self.kp_strafe * lateral_error
                strafe_speed = max(min(strafe_speed, self.max_strafe_speed), -self.max_strafe_speed)
                cmd_vel.linear.y = strafe_speed
            
            # Small forward movement to maintain progress
            if abs(forward_error) > 0.05:
                forward_speed = self.kp_forward * min(forward_error, 0.5)  # Limit forward speed
                forward_speed = max(min(forward_speed, self.max_linear_speed), 0)  # Only forward
                cmd_vel.linear.x = forward_speed
            
            # Small heading correction to maintain road alignment
            if abs(heading_error) > 0.05:
                angular_correction = 0.5 * self.kp_heading * heading_error
                cmd_vel.angular.z = angular_correction
            
            self.get_logger().info(
                f"Following lane: lateral error {lateral_error:.3f}m, "
                f"forward: {cmd_vel.linear.x:.2f}m/s, "
                f"strafe: {cmd_vel.linear.y:.2f}m/s"
            )
        
        self.wheel_pub.publish(cmd_vel)
        return False
    
    def goto(self, target_x, target_y, odom:Odometry):
        if target_x is None or target_y is None:
            return False
        
        current_pos = odom.pose.pose.position
        current_ori = odom.pose.pose.orientation

        current_x = current_pos.x
        current_y = current_pos.y

        current_yaw = yaw_from_quaternion(current_ori)

        dx_world = target_x - current_x
        dy_world = target_y - current_y

        dx_robot = dx_world * math.cos(current_yaw) + dy_world * math.sin(current_yaw)
        dy_robot = -dy_world * math.sin(current_yaw) + dy_world * math.cos(current_yaw)

        cmd_vel = Twist()

        cmd_vel.linear.x = max(min(dx_robot, self.max_linear_speed), -dx_robot)
        strafe_speed = self.kp_strafe * dy_robot
        cmd_vel.linear.y = max(min(strafe_speed, self.max_strafe_speed), -self.max_strafe_speed)

        self.wheel_pub.publish(cmd_vel)
        return False

    def should_use_last_target(self, current_time):
        """Check if we should use the last target point"""
        if self.last_target_point is None:
            return False
        
        time_since_last_target = current_time - self.last_target_time
        return time_since_last_target < self.last_target_persist_time

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
        # self.print_odom(odom_m)
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
        scale = 1/4
        work_size = (int(ow*scale), int(oh*scale))
        image_bgr = cv2.resize(image_bgr, work_size)
        image_dep = cv2.resize(image_dep, work_size)

        h, w = image_bgr.shape[:2]
        image_dep = cv2.inpaint(image_dep, (image_dep == 0).astype('uint8'), 2, cv2.INPAINT_TELEA)

        lower = np.array([self.lm, self.am, self.bm])
        upper = np.array([self.lM, self.aM, self.bM])
        lab_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        mask_lab = cv2.inRange(lab_img, lower, upper)

        # Remove small noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(mask_lab, cv2.MORPH_OPEN, kernel_open)
        
        # Fill small holes
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

        # Lane detection
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

        ff_left = None
        ff_right = None
        ff_right_far = None
        
        if seedpoint_l:
            ff_left = self.ff_mask(mask_lab.copy(), seedpoint_l)
        if seedpoint_r:
            ff_right = self.ff_mask(mask_lab.copy(), seedpoint_r)

        target_point = None
        target_depth = 0
        new_target_found = False
        # Convert mask to BGR for visualization
        out = cv2.cvtColor(mask_lab, cv2.COLOR_GRAY2BGR)
        
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
                    if rx in ll:
                        break
                    if rx - lx < 100:
                        target_point = None
                        break
                    else:
                        target_point = ((lx+rx)//2, _y)
                        target_depth = image_dep[_y, (lx+rx)//2]
                        new_target_found = True
                    cv2.line(mask_lab, (lx, _y), (rx, _y), (128,128,128), 2)
                    break
                _y += _step
            _step = 2
            _y = 0
            points = []
            while _y < h-1:
                ll = np.where(ff_left[_y, :] > 0)[0]
                lr = np.where(ff_right[_y, :] > 0)[0]
                if len(ll) and len(lr):
                    lx = ll[-1]
                    rx = lr[0]
                    if rx in ll:
                        break
                    points.append(((lx+rx)//2, _y))
                _y += _step
            
            for _x, _y in points:
                cv2.circle(out, (_x, _y), 1, (0,0,255), 1)
        
        
        out = cv2.resize(out, (640, 480), interpolation=cv2.INTER_NEAREST_EXACT)
        cv2.line(out, (332, 232), (332, 480 - 1), (0,255,0), 1)
        cv2.circle(out, (332,232), 1, (255,0,0), 1) # true center
        cv2.imshow('lab mask', out)
        cv2.waitKey(1)
        return

        current_time = time.time()
        
        # Determine which target to use
        if new_target_found and target_point and target_depth > 0:
            # New target found - use it and update last target
            u, v = target_point
            u_original = int(u / scale)
            v_original = int(v / scale)
            
            odom_point = self.pixel_transformer.pixel_to_odom_direct(u_original, v_original, target_depth, odom_m)
            
            if odom_point:
                self.current_target = (odom_point.x, odom_point.y)
                self.last_target_point = (odom_point.x, odom_point.y)  # Store for future use
                self.last_target_time = current_time
                target_source = "NEW TARGET"
                
        elif self.should_use_last_target(current_time) and self.last_target_point:
            # Use last target point (no new target found but within timeout)
            self.current_target = self.last_target_point
            target_source = "LAST TARGET"
            odom_point = Point()
            odom_point.x = self.current_target[0]
            odom_point.y = self.current_target[1]
            odom_point.z = 0.0
            
        else:
            # No target available
            self.stop_movement()
            self.current_target = None
            cv2.putText(out, 'NO TARGET', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Resize and display
            out = cv2.resize(out, (640, 480), interpolation=cv2.INTER_NEAREST_EXACT)
            cv2.imshow('lab mask', out)
            cv2.waitKey(1)
            return

        # Continue with navigation using current_target
        if self.current_target:
            # Estimate road direction
            if ff_left is not None and ff_right is not None:
                self.road_direction = self.estimate_road_direction(ff_left, ff_right, (h, w), odom_m)
            
            # Follow lane with road alignment
            status_text = 'CUSTOM'
            self.goto(self.current_target[0], self.current_target[1], odom_m)
            # if self.road_direction is not None:
            #     target_reached = self.follow_lane_with_alignment(
            #         self.current_target[0], self.current_target[1], 
            #         self.road_direction, odom_m
            #     )
            #     status_text = 'PERFECTLY ALIGNED' if target_reached else 'ALIGNING WITH ROAD'
            # else:
            #     # Fallback to simple lane following
            #     self.simple_lane_following(self.current_target[0], self.current_target[1], odom_m)
            #     status_text = 'SIMPLE LANE FOLLOWING'
            
            # Display status and target source
            cv2.putText(out, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(out, f'Target: {target_source}', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Display road direction if available
            if self.road_direction is not None:
                cv2.putText(out, f'Road dir: {math.degrees(self.road_direction):.1f}°', 
                           (10, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.putText(out, f'Odom: ({self.current_target[0]:.2f}, {self.current_target[1]:.2f})', 
                       (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Publish target point for visualization
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "odom"
            pose_msg.pose.position.x = self.current_target[0]
            pose_msg.pose.position.y = self.current_target[1]
            pose_msg.pose.position.z = 0.0
            pose_msg.pose.orientation.w = 1.0
            self.target_point_pub.publish(pose_msg)

            # Visualize the target point if it's a new detection
            if new_target_found and target_point:
                cv2.circle(out, target_point, 5, (0, 0, 255), -1)
                cv2.putText(out, f'd:{target_depth}', target_point,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display current heading and time since last target
        current_yaw = yaw_from_quaternion(odom_m.pose.pose.orientation)
        time_since_target = current_time - self.last_target_time
        cv2.putText(out, f'Robot heading: {math.degrees(current_yaw):.1f}°', 
                   (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(out, f'odom pos: {odom_m.pose.pose.position.x}, {odom_m.pose.pose.position.y}',
                    (10, h-120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1 )
        cv2.putText(out, f'Time since target: {time_since_target:.1f}s', 
                   (10, h-80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Resize and display
        out = cv2.resize(out, (640, 480), interpolation=cv2.INTER_NEAREST_EXACT)
        pos = odom_m.pose.pose.position
        ori = odom_m.pose.pose.orientation
        
        cv2.putText(out, f'pos: (x:{pos.x:.3f}, y:{pos.y:.3f})', (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
        cv2.putText(out, f'ang: ({yaw_from_quaternion_deg(ori):.2f})', (20, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
        
        cv2.imshow('lab mask', out)
        cv2.waitKey(1)

    def init_process(self):
        self.timer.cancel() 
        self.stop_movement()

def main():
    cv2.namedWindow('lab mask')
    cv2.moveWindow('lab mask', 0, 0)
    cv2.namedWindow('img')
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
