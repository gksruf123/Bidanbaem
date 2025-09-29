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
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import numpy as np
import queue

#MAP_X = 3000
#MAP_Y = 2800
#ROAD_W = 450
#CROSSWALK_H = 110

import math
from geometry_msgs.msg import Quaternion

def yaw_to_quaternion(yaw):
    # yaw in radians
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw/2.0)
    q.w = math.cos(yaw/2.0)
    return q

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

        # self.client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        # self.client.wait_for_server()
 
        self.odom_sub = Subscriber(self, Odometry, '/odom')
        self.rgb_sub = Subscriber(self, Image, '/ascamera/camera_publisher/rgb0/image')
        self.depth_sub = Subscriber(self, Image, '/ascamera/camera_publisher/depth0/image_raw')

        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.odom_sub],
            # queue_size = 6,
            queue_size = 6,
            slop=0.15
        )
        # self.ts.registerCallback(self.enqueue)
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


        self.wheel_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.timer = self.create_timer(0.0, self.init_process) # run init_process asap
        
        # threading.Thread(target=self.worker, daemon=True).start()  

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
        # print('dir')
        self.print_odom(odom_m)
        self.proc(rgb_m, dep_m, odom_m)

    def ff_mask(self, mask, point):
        _mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8)
        cv2.floodFill(mask, _mask, point, newVal=255, 
                      flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8))
        return _mask[1:-1, 1:-1]


    def proc(self, rgb_m, dep_m, odom_m:Odometry = None):
        image_bgr = self.cv_bridge.imgmsg_to_cv2(rgb_m, 'bgr8')
        # cv2.imshow('orig', image_bgr)
        image_dep = self.cv_bridge.imgmsg_to_cv2(dep_m, '16UC1')

        oh, ow = image_bgr.shape[:2]
        #resize down for speed, 1/2 or 1/4
        scale = 1/2
        work_size = (int(ow*scale), int(oh*scale))
        image_bgr = cv2.resize(image_bgr, work_size)
        image_dep = cv2.resize(image_dep, work_size)

        h, w = image_bgr.shape[:2]
        #fill in empty holes in depth imgage
        image_dep = cv2.inpaint(image_dep, (image_dep == 0).astype('uint8'), 2, cv2.INPAINT_TELEA)

        lower = np.array([self.lm, self.am, self.bm])
        upper = np.array([self.lM, self.aM, self.bM])
        lab_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        mask_lab = cv2.inRange(lab_img, lower, upper)
        # cv2.imshow('mask', mask_lab)

        ### erode is too strong for our image
        #try and remove small specs
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # # mask_erode = cv2.erode(mask_lab, np.ones((3, 3), np.uint8), iterations=1) 
        # mask_erode = cv2.erode(mask_lab, kernel, iterations=1)

        # # cv2.imshow('test', image_rgb)
        # merged = cv2.resize(merged, (640*2, 480), interpolation=cv2.INTER_NEAREST_EXACT)
        # cv2.imshow('erode', merged)

        #find left lane
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

        #find right lane
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
            # out = cv2.resize(ff_left, (640, 480), interpolation=cv2.INTER_NEAREST_EXACT)
        #     cv2.imshow('ff left', out)
        #     cv2.moveWindow('ff left', 0, 500)

        if seedpoint_r:
            ff_right = self.ff_mask(mask_lab, seedpoint_r)
        #     out = cv2.resize(ff_right, (640, 480), interpolation=cv2.INTER_NEAREST_EXACT)
        #     cv2.imshow('ff right', out)
        #     cv2.moveWindow('ff right', 1000, 500)

        # out = cv2.resize(mask_lab, (640, 480), interpolation=cv2.INTER_NEAREST_EXACT)
        # cv2.imshow('lab mask', out)
        # cv2.moveWindow('lab mask', 500, 0)

        ###ff_left check shape?
        ###if ff_right exists, set target 
        ###did we cross crosswalk?

        target_point = None
        ###detect crosswalk
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
                    cv2.line(mask_lab, (lx, _y), (rx, _y), 128, 2)
                    break

                _y += _step

        out = cv2.cvtColor(mask_lab, cv2.COLOR_GRAY2BGR)
        if target_point:
            cv2.circle(out, target_point, 1, (0,0,255), 1)
            cv2.putText(out, f'd:{image_dep[target_point[1]+3,target_point[0]]}', target_point,
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255,255,255), 1)
            
            _y = target_point[1]
            if target_point[1] > h//2:
                while _y > 0:
                    if mask_lab[_y, target_point[0]]:
                        self.status = 'stop'
                        print('stop')
                        self.wheel_pub.publish(Twist())
                    _y -= 1
            #H64.4°×V51.7°
            # W, H = w, h
            # u, v = target_point[0], target_point[1]
            # Z = image_dep[v, u]
            # theta_x = (u - W/2) / (W/2) * (64.4 / 2)
            # theta_y = (v - H/2) / (H/2) * (51.7 / 2)

            # # Convert degrees to radians
            # theta_x = math.radians(theta_x)
            # theta_y = math.radians(theta_y)

            # # Camera frame coordinates
            # X_cam = Z * math.tan(theta_x)  # left/right
            # Y_cam = Z * math.tan(theta_y)  # up/down
            # Z_cam = Z            

            # cv2.putText(out, f'{X_cam:.2f}, {Y_cam:.2f}, {Z_cam:.2f}', (0, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
        out = cv2.resize(out, (640, 480), interpolation=cv2.INTER_NEAREST_EXACT)
        stamp = odom_m.header.stamp.nanosec
        pos = odom_m.pose.pose.position
        ori = odom_m.pose.pose.orientation
        twist_l = odom_m.twist.twist.linear
        twist_a = odom_m.twist.twist.angular
        cv2.putText(out, f'timestamp: {stamp}', (20, 20), 
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 255, 255))
        cv2.putText(out, f'pos: (x:{pos.x:.3f}, y:{pos.y:.3f})', (20, 40), 
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 255, 255))
        
        cv2.putText(out, f'ang: ({yaw_from_quaternion_deg(ori):.2f})', (20, 60), 
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 255, 255))
        # cv2.putText(out, f'ori: (x:{ori.x:.2f}, y:{ori.y:.2f}, z:{ori.z:.2f})', (20, 60), 
        #             cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 255, 255))
        cv2.putText(out, f'twist_l: (x:{twist_l.x:.3f}, y:{twist_l.y:.2f})', (20, 80), 
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 255, 255))
        cv2.putText(out, f'twist_a: (z:{twist_a.z:.2f})', (20, 100), 
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 255, 255))
        cv2.imshow('lab mask', out)

        

        cv2.waitKey(1)

    def send_goal(self, node, x, y, yaw=0.0):
        print(f'goal: {x},{y},{yaw}')
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.headerb.frame_id = 'odom'  # must match Nav2 global frame
        goal_msg.pose.header.stamp = node.get_clock().now().to_msg()
        goal_msg.pose.pose.position = Point(x=x, y=y, z=0.0)
        goal_msg.pose.pose.orientation = yaw_to_quaternion(yaw)  # helper function
        self.client.send_goal_async(goal_msg)

        
    def enqueue(self, rgb, depth):
        item = (rgb, depth)        
        try:
            self.msg_queue.put_nowait(item)
        except queue.Full:
            _ = self.msg_queue.get_nowait()
            self.msg_queue.put_nowait(item)
            
    def worker(self):
        while True:
            item = self.msg_queue.get()
            if item is None:
                break
            rgb, depth = item
            self.proc(rgb, depth)
            self.msg_queue.task_done()

    def init_process(self):
        self.timer.cancel() 

        self.stop_movement() # reset wheel

    def stop_movement(self):
        self.wheel_pub.publish(Twist())

    def image_callback(self):
        pass
    def object_detect_callback(self):
        pass

def main():
    cv2.namedWindow('lab mask')
    cv2.moveWindow('lab mask', 500, 0)
    rclpy.init(args=None)
    node = Navigation()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()