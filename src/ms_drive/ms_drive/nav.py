#!/usr/bin/env python3
from collections import defaultdict
from enum import IntEnum, auto
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
from ms_drive.util import *
from std_srvs.srv import SetBool, Trigger


### Check camera transform ### 
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped, TransformStamped


class Status(IntEnum):
    init = auto()
    """scanning for green"""
    stopped = auto()
    find_target = auto()
    turning = auto()
    """always right turn"""
    moving = auto()
    arrived = auto()
    scanning = auto()

def default_detect():
    return {'count':0, 'box': [0.0]*4}

class SlowLogger:
    def __init__(self, node:Node):
        self.logger = node.get_logger()
        self.last_msg = ''
        self.last_msg_time = time.time()
        self.interval = 1.0
    
    def log(self, msg, force=False):
        ct = time.time()
        """will skip for interval if prev msg is same"""
        if msg == self.last_msg and self.last_msg_time + self.interval > ct:
            return
        self.logger.info(msg)
        self.last_msg_time = ct
        self.last_msg = msg
    


class AngleSnapper:
    def __init__(self, initial_angle_rad: float):
        """Store the reference angle."""
        self.initial_angle = initial_angle_rad

    def snap(self, angle_rad: float) -> float:
        """Snap the given angle to the nearest 90° increment (π/2) from the initial angle."""
        delta = angle_rad - self.initial_angle
        snapped_delta = round(delta / (math.pi / 2)) * (math.pi / 2)
        return self.initial_angle + snapped_delta


class PixelToOdomTransformer:
    def __init__(self, node):
        self.node = node
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, node)
        
        # Camera intrinsics from camera_info which shouldn't change so just bake it in.
        self.fx = 576.5324096679688
        self.fy = 576.1083374023438  
        self.cx = 332.5771484375
        self.cy = 232.551513671875

        self.camera_pitch = math.radians(12) # not sure if necessary
        self.camera_x = 0
        self.camera_y = 0
        self.camera_z = 0

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
        robot_yaw = yaw_from_quaternion(robot_ori)
        
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


class MapOdomWrapper:
    def __init__(self, node):
        self.node = node
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, node)
    
    def get_odom_msg(self):
        try:
            # Lookup the latest transform from SLAM map frame to base_link
            t: TransformStamped = self.tf_buffer.lookup_transform(
                'map',      # target frame
                'base_link',# source frame
                rclpy.time.Time()
            )
            odom_msg = Odometry()
            odom_msg.header.stamp = self.node.get_clock().now().to_msg()
            odom_msg.header.frame_id = 'map'
            odom_msg.child_frame_id = 'base_link'
            
            # Fill position
            odom_msg.pose.pose.position.x = t.transform.translation.x
            odom_msg.pose.pose.position.y = t.transform.translation.y
            odom_msg.pose.pose.position.z = t.transform.translation.z
            
            # Fill orientation
            odom_msg.pose.pose.orientation = t.transform.rotation # perhaps just use odom orientation as that was stable already.

            # You can optionally zero velocities or compute from tf if needed
            odom_msg.twist.twist.linear.x = 0.0
            odom_msg.twist.twist.linear.y = 0.0
            odom_msg.twist.twist.linear.z = 0.0
            odom_msg.twist.twist.angular.x = 0.0
            odom_msg.twist.twist.angular.y = 0.0
            odom_msg.twist.twist.angular.z = 0.0

            return odom_msg
        except Exception as e:
            return None


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

        # Odom mapper
        self.odom_mapper = MapOdomWrapper(self)

        # Slow logger
        self.slogger = SlowLogger(self)

        # Subscriptions
        # self.odom_sub = Subscriber(self, Odometry, '/odom')
        self.rgb_sub = Subscriber(self, Image, '/ascamera/camera_publisher/rgb0/image')
        self.depth_sub = Subscriber(self, Image, '/ascamera/camera_publisher/depth0/image_raw')

        # Tie the RGB and depth together 
        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub
            #  , self.odom_sub
            ],
            queue_size=6,
            slop=0.15
        )
        self.ts.registerCallback(self.direct)


        # Load the LAB settings - needs more perm. Solution or just could bake it in.
        with open('LAB-cal.json', 'r') as f:
            d = json.load(f)
            self.lm = d['l_min']
            self.lM = d['l_max']
            self.am = d['a_min']
            self.aM = d['a_max']
            self.bm = d['b_min']
            self.bM = d['b_max']

        self.status = Status.init
        
        # Navigation control parameters
        self.proc_scale = 1/4 # 1/4 seems to work fine, use 1/2 if problems occur
        self.current_target = None
        self.last_target_point = None  # Store the last detected target point
        
        # Control parameters for mecanum wheel. 
        self.max_linear_speed = 0.8 # m/s
        self.max_strafe_speed = self.max_linear_speed * 0.7
        self.max_angular_speed = 1.0

        self.max_linear_accel = 4.0 # m/s**2 
        self.max_strafe_accel = 3.5
        self.max_angular_accel = 9.0 # rad/s**2

        self.position_tolerance = 0.1
        self.lateral_tolerance = 0.1
        self.heading_tolerance = 0.025
        self.target_angle = None

        self.turn_speed = self.max_linear_speed * 0.8
        self.turn_radius = 0.25

        self.cur_lx = 0.0
        self.cur_ly = 0.0
        self.cur_az = 0.0
        
        # yolo settings
        self.yolo_active = None
        self.yolo_min_count = 2
        self.yolo_min_conf = 0.4
        """minimum count to act"""
        # yolov5 service clients
        self.yolov5_start_client = self.create_client(Trigger, '/yolov5/start')
        self.yolov5_stop_client = self.create_client(Trigger, '/yolov5/stop')

        # publishes, services. Direct driving for now.
        # self.create_service(Trigger, '/ms_driver/arrived', self.arrive_srv_cb)
        self.wheel_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.logic_pub = self.create_publisher(Point, '/ms_logic', 1)
        """Point, Linear Z 0.0 means target, alignment otherwise"""
        self.init_timer = self.create_timer(0.0, self.init_process)
        self.drive_loop = self.create_timer(0.05, self.target_loop) # Lidar slam is updated at max 10hz, 20hz wheel management sounds ok. 
        self.last_drive_tick = time.time()

        self.detects = defaultdict(default_detect)
        """[name]: {box, count}, box is latest detection"""

        self.get_logger().info("MS_drive nav node started")

    def init_process(self):
        self.init_timer.cancel() 
        self.stop_movement()
        self.yolov5_start_client.wait_for_service() ### disable for debug
        self.yolov5_stop_client.wait_for_service()
        self.activate_yolo()
        """activate yolo on init"""

    def goto(self, target_x, target_y, odom:Odometry, is_cw = False):
        # print('goto', is_cw, self.target_angle)
        if target_x is None or target_y is None:
            return False
        current_pos = odom.pose.pose.position
        current_ori = odom.pose.pose.orientation

        current_x = current_pos.x
        current_y = current_pos.y
        current_yaw = yaw_from_quaternion(current_ori)

        # World-frame vector to target
        dx_world = target_x - current_x
        dy_world = target_y - current_y

        # Transform to robot frame
        dx_robot = dx_world * math.cos(current_yaw) + dy_world * math.sin(current_yaw)
        dy_robot = -dx_world * math.sin(current_yaw) + dy_world * math.cos(current_yaw)

        # Compute distance along robot's forward direction
        forward_distance = dx_robot
        if is_cw:
            forward_distance -= 0.15

        # Only move forward if we haven't reached the stopping distance
        forward_speed = 0.0
        if forward_distance > 0.05:  # small threshold
            # forward_speed = self.kp_forward * min(forward_distance, 0.5)
            # forward_speed = min(forward_speed, self.max_linear_speed)
            forward_speed = self.max_linear_speed

        # Strafe speed
        strafe_speed = dy_robot
        strafe_speed = max(min(strafe_speed, self.max_strafe_speed), -self.max_strafe_speed)

        # Heading correction toward target angle
        heading_error = self.target_angle - current_yaw
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi

        angular_correction = 0.0
        if abs(heading_error) > 0.025:
            angular_correction = self.kp_heading * heading_error
            angular_correction = max(min(angular_correction, self.max_angular_speed), -self.max_angular_speed)

        # Publish velocities
        cmd_vel = Twist()
        cmd_vel.linear.x = forward_speed
        cmd_vel.linear.y = strafe_speed
        cmd_vel.angular.z = angular_correction
        self.wheel_pub.publish(cmd_vel)

        # Return True if we reached stopping distance
        return forward_distance <= 0.05

    def target_loop(self):
        def lim(cur, targ, max_a, dt):
            delta = targ - cur
            max_delta = max_a * dt

            if delta < 0: # faster decel.
                max_delta *= 2

            if abs(delta) <= max_delta:
                return targ
            else:
                return cur + math.copysign(max_delta, delta)
            
        # print(time.time())
        if self.status != Status.moving or not self.current_target:
            return
        if not (odom := self.odom_mapper.get_odom_msg()):
            self.slogger.log('target_loop failed to get odom')
            return
        target_x, target_y = self.current_target
        # print('goto', is_cw, self.target_angle)
        if target_x is None or target_y is None:
            return False

        # ct = time.time()
        # dt = ct - self.last_drive_tick
        dt = 0.05 #1 / 20 # ehh should be good enough. 

        current_pos = odom.pose.pose.position
        current_ori = odom.pose.pose.orientation

        current_x = current_pos.x
        current_y = current_pos.y
        current_yaw = yaw_from_quaternion(current_ori)

        self.slogger.log(f'target loop target: {target_x:.1f}, {target_y:.1f}, cp: {current_pos.x:.1f}, {current_pos.y:.1f}, yaw: {current_yaw:.1f}')

        # World-frame vector to target
        dx_world = target_x - current_x
        dy_world = target_y - current_y

        # Transform to robot frame
        dx_robot = dx_world * math.cos(current_yaw) + dy_world * math.sin(current_yaw)
        dy_robot = -dx_world * math.sin(current_yaw) + dy_world * math.cos(current_yaw)

        # Compute distance along robot's forward direction
        forward_distance = dx_robot - 0.2 
        forward_speed = 0.0
        if forward_distance > self.position_tolerance:
            forward_speed = math.copysign(min(self.max_linear_speed, dx_robot / 0.4), forward_distance)

        strafe_speed = 0.0
        if abs(dy_robot) > self.lateral_tolerance:
            strafe_speed = math.copysign(min(self.max_strafe_speed, dy_robot / 0.75), dy_robot)

        self.cur_lx = lim(self.cur_lx, forward_speed, self.max_linear_accel, dt)
        self.cur_ly = lim(self.cur_ly, strafe_speed, self.max_strafe_speed, dt)

        # Heading correction toward target angle
        heading_error = self.target_angle - current_yaw
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi

        angular_speed = 0.0
        if abs(heading_error) > self.heading_tolerance:
            angular_speed = math.copysign(self.max_angular_speed, heading_error)
        self.cur_az = lim(self.cur_az, angular_speed, self.max_angular_accel, dt)


        # Publish velocities
        cmd_vel = Twist()
        cmd_vel.linear.x = self.cur_lx
        cmd_vel.linear.y = self.cur_ly
        cmd_vel.angular.z = self.cur_az
        self.wheel_pub.publish(cmd_vel)

        if forward_speed == 0.0 and strafe_speed == 0.0 and angular_speed == 0.0: # not moving for some reason
            if forward_distance < self.position_tolerance: # arrived, start scanning
                self.current_target = None
                self.status = Status.scanning
            else:
                self.status = Status.stopped


    def stop_movement(self):
        """Stop the robot"""
        cmd_vel = Twist()
        self.wheel_pub.publish(cmd_vel)
        self.get_logger().info("Movement stopped")

    def arrive_srv_cb(self, req, resp):
        """Unused for now."""
        self.get_logger().info('driver has arrived')
        self.status = Status.arrived
        resp.success = True
        resp.message = 'OK'
        return resp

    def send_request(self, client, msg):
        future = client.call_async(msg)
    
        # Spin until future is complete
        rclpy.spin_until_future_complete(self, future)
        
        return future.result()  # safe to get result now

    def activate_yolo(self):
        self.get_logger().info('activating yolo')
        self.yolo_active = True
        self.send_request(self.yolov5_start_client, Trigger.Request())
        self.yolo5_sub = self.create_subscription(ObjectsInfo, '/yolov5_ros2/object_detect', self.yolo_cb, 2)

    def deactivate_yolo(self):
        self.get_logger().info('deactivating yolo')
        self.yolo_active = False
        self.yolo5_sub.destroy()
        self.send_request(self.yolov5_stop_client, Trigger.Request())
        self.detects.clear()

    def yolo_cb(self, msg:ObjectsInfo):
        self.get_logger().info('yolo cb')
        objects = msg.objects
        if not self.status == Status.scanning:
            self.get_logger().warn(f'yolo callback when status is not scanning, {self.status}')
            return
        if not objects: # can this even happen? idk.
            return
        for obj in objects:
            obj:ObjectInfo
            self.get_logger().info(f'{name} detected, count: {self.detects[name]['count']}, conf: {obj.score:.2f}')
            if obj.score < self.yolo_min_conf:
                continue
            name = obj.class_name
            points = obj.box
            count = self.detects[name]['count'] + 1
            self.detects[name] = {'box':points, 'count':count}

        # print(self.detects)
        # if self.detects['right']['count'] > 3:
        #     self.get_logger().info('deactivate yolo')
        #     self.deactivate_yolo()

    def direct(self, rgb_m, dep_m, odom_m:Odometry):
        """was ment to use queue to separate cb to logic but logic seems light enough. 
        Just do proc directly. 
        """
        # self.print_odom(odom_m)
        self.proc(rgb_m, dep_m, odom_m)

    def ff_mask(self, mask, point):
        _mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8)
        cv2.floodFill(mask, _mask, point, newVal=255, 
                      flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8))
        return _mask[1:-1, 1:-1]
    
    def lane_detection(self, mask):
        h, w = mask.shape[:2]
        # Lane detection
        seedpoint_l = None
        _x = w//2 - 1
        _y = h-1
        while _x > 0:
            if mask[h-1, _x]:
                seedpoint_l = (_x, h-1)
                break
            _x -= 1
        if not seedpoint_l:
            while _y > 0:
                if mask[_y, _x]:
                    seedpoint_l = (_x, _y)
                    break
                _y -= 1

        seedpoint_r = None
        _x = w//2
        _y = h-1
        while _x < w - 1:
            if mask[h-1, _x]:
                seedpoint_r = (_x, h-1)
                break
            _x += 1
        if not seedpoint_r:
            while _y > 0:
                if mask[_y, _x]:
                    seedpoint_r = (_x, _y)
                    break
                _y -= 1

        ff_left = None
        ff_right = None
        ff_right_far = None
        if seedpoint_l:
            ff_left = self.ff_mask(mask, seedpoint_l) # since FF_MASK_ONLY, perhaps just give mask instead of copying over.
        if seedpoint_r:
            if ff_left and ff_left[seedpoint_r[1], seedpoint_r[0]] > 0: #seedpoint_r is in ff_left, nullify it.
                seedpoint_r = None
            else:
                ff_right = self.ff_mask(mask, seedpoint_r)

        seedpoint_rf = None
        if seedpoint_r:
            _x = w - 1
            _y = 0
            while _y < h - 1:
                if mask[_y, _x] and not ff_right[_y, _x]:
                    seedpoint_rf = (_x, _y)
                    break
                _y += 1

        if seedpoint_rf:
            ff_right_far = self.ff_mask(mask, seedpoint_rf)
        
        return seedpoint_l, seedpoint_r, seedpoint_rf, ff_left, ff_right, ff_right_far

    def turn_right(self, odom:Odometry):
        current_pos = odom.pose.pose.position
        current_ori = odom.pose.pose.orientation
        cmd_vel = Twist()

        cmd_vel.linear.x = self.turn_speed
        
        cmd_vel.angular.z = -self.turn_speed / self.turn_radius
        # return
        self.wheel_pub.publish(cmd_vel)
        return True

    def proc(self, rgb_m, dep_m):
        """main logic, tied to 15(or 20)fps of the cameras"""
        if not self.target_angle: # set the initial angle as target_angle. 
            self.target_angle = yaw_from_quaternion(odom_m.pose.pose.orientation)
            self.angle_snapper = AngleSnapper(self.target_angle)

        if self.status == Status.init: # Init state, wait for green count
            if self.detects['green']['count'] >= self.yolo_min_count: # Start! Deactivate yolo and proceed to drive logic
                self.deactivate_yolo()
                self.status = Status.find_target
            else:
                return # Wait for firm greens

        elif self.status == Status.scanning: # need to scan yolo
            if not self.yolo_active:
                self.activate_yolo()
                return
            go = self.detects['go']['count']
            gr = self.detects['green']['count']
            rt = self.detects['right']['count']
            # perhaps check red? 
            # need some sort of failsafe. 
            if go >= self.yolo_min_count or gr >= self.yolo_min_count: # time to move
                self.status = Status.find_target 
            elif rt >= self.yolo_min_count: # turn right
                self.status = Status.turning
            else:
                return

        if self.status != Status.find_target and self.status != Status.moving: # only process when finding target or moving
            return


        image_bgr = self.cv_bridge.imgmsg_to_cv2(rgb_m, 'bgr8')
        image_dep = self.cv_bridge.imgmsg_to_cv2(dep_m, '16UC1')

        if not (odom_m := self.odom_mapper.get_odom_msg()):
            self.slogger.log('proc: self.odom_mapper.get_odom_msg returned None')
            return

        self.target_angle = self.angle_snapper.snap(yaw_from_quaternion(odom_m.pose.pose.orientation))

        oh, ow = image_bgr.shape[:2]
        work_size = (int(ow*self.proc_scale), int(oh*self.proc_scale))
        image_bgr = cv2.resize(image_bgr, work_size)
        image_dep = cv2.resize(image_dep, work_size)

        h, w = image_bgr.shape[:2]
        image_dep = cv2.inpaint(image_dep, (image_dep == 0).astype('uint8'), 2, cv2.INPAINT_TELEA) #inpaint is costly, so the smaller the scale the better. 

        # LAB filtering, perhaps add or merge other filterings?
        lower = np.array([self.lm, self.am, self.bm])
        upper = np.array([self.lM, self.aM, self.bM])
        lab_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        mask_lab = cv2.inRange(lab_img, lower, upper) 
        
        # Fill small holes
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask_lab, cv2.MORPH_CLOSE, kernel_close)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        # # Remove small noise
        # kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)

        seedpoint_l, seedpoint_r, seedpoint_rf, ff_left, ff_right, ff_right_far = self.lane_detection(mask)

        ### for visualization
        # Convert mask to BGR and resize it for visualization
        out = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        out = cv2.resize(out, (640, 480))
        # if ff_left is not None:
        #     cv2.imshow('ll', ff_left)
        # if ff_right is not None:
        #     cv2.imshow('rl', ff_right)
        # if ff_right_far is not None:
        #     cv2.imshow('rl_f', ff_right_far)

        target_point = None

        if seedpoint_l and seedpoint_r:
            _step = 5   
            _y = 0
            while _y < h-1:
                ll = np.where(ff_left[_y, :] > 0)[0]
                lr = np.where(ff_right[_y, :] > 0)[0]
                if len(ll) and len(lr):
                    lx = ll[-1]
                    rx = lr[0] # gotta be careful with cam angles, might cause problems
                    if rx in ll:
                        break
                    else:
                        target_point = ((lx+rx)//2, _y)
                        # target_depth = image_dep[_y, (lx+rx)//2]

                    cv2.line(out, (lx, _y), (rx, _y), (128,128,128), 2)
                    break
                _y += _step

            # perhaps get more points and average the _x as single point might look at wrong edges. 
            # _step = 2
            # _y = 0
            # points = []
            # while _y < h-1:
            #     ll = np.where(ff_left[_y, :] > 0)[0]
            #     lr = np.where(ff_right[_y, :] > 0)[0]
            #     if len(ll) and len(lr):
            #         lx = ll[-1]
            #         rx = lr[0]
            #         if rx in ll:
            #             break
            #         points.append(((lx+rx)//2, _y))
            #     _y += _step
            
            # for _x, _y in points:
            #     cv2.circle(out, (_x, _y), 1, (0,0,255), 1)
        elif seedpoint_l and not seedpoint_r:
            self.turn_right()

        if target_point: # found target
            # need_update = False
            # if 
            u, v = target_point
            u_orig = int(u / self.proc_scale)
            v_orig = int(v / self.proc_scale)
            odom_point = self.pixel_transformer.pixel_to_odom_direct(u_orig, v_orig, image_dep[v, u], odom_m)

            if odom_point: # transform successful
                self.status = Status.moving
                self.current_target = (odom_point.x, odom_point.y)
                self.slogger.log(f'new target: {odom_point.x:.1f},{odom_point.y:.1f} cp:{odom_m.pose.pose.position.x}, {odom_m.pose.pose.position.y}')
        
        cv2.imshow('lab mask', out)
        cv2.waitKey(1)


def main():
    cv2.namedWindow('lab mask')
    cv2.moveWindow('lab mask', 0, 0)
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
