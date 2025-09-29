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
import numpy as np
import queue

class CVTest(Node):
    def __init__(self):
        super().__init__('subscribe')

        #QOS
        self.cv_bridge = CvBridge()
        # self.rgb_sub = Subscriber(self, Image, '/ascamera/camera_publisher/rgb0/image', qos_profile=qos)
        # self.depth_sub = Subscriber(self, Image, '/ascamera/camera_publisher/depth0/image_raw', qos_profile=qos)

        self.rgb_sub = Subscriber(self, Image, '/ascamera/camera_publisher/rgb0/image')
        self.depth_sub = Subscriber(self, Image, '/ascamera/camera_publisher/depth0/image_raw')

        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            # queue_size = 6,
            queue_size = 5,
            slop=0.1
        )
        self.ts.registerCallback(self.enqueue)

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
        self.msg_queue = queue.Queue(1)
        self.qc = 0

        threading.Thread(target=self.worker, daemon=True).start()

    def enqueue(self, rgb, depth):
        now = time.time()
        if now - self.last_print >= 1.0:
            print(f'frames recv: {self.qc}')
            print(f'frames proc: {self.frame_count}')
            self.qc = 0
            self.frame_count = 0
            self.last_print = now

        self.qc += 1
        item = (rgb, depth)
        try:
            self.msg_queue.put_nowait(item)
        except queue.Full:
            _ = self.msg_queue.get_nowait()
            self.msg_queue.put_nowait(item)

    def worker(self):
        print('worker start')
        while True:
            item = self.msg_queue.get()
            if item is None:
                break
            rgb, depth = item
            self.proc(rgb, depth)
            self.msg_queue.task_done()
        print('worker finished')

    def proc(self, rgb_msg, depth_msg):
        self.frame_count += 1
        # return

        # RGB, Depth images
        rgb_img = self.cv_bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        depth_img = self.cv_bridge.imgmsg_to_cv2(depth_msg, '16UC1')

        ### half the size for speed
        ### even 160,120 looks acceptable, could possibly raise camera fps from 15 to 30 w/o problem. 
        ### perhaps check only half the frame vertically or something, set ROI.
        rgb_img = cv2.resize(rgb_img, (160, 120))
        depth_img = cv2.resize(depth_img, (160, 120))

        h, w = depth_img.shape[:2]
        x, y = w//2, h//2

        # fill in depth cam blanks
        depth_img = cv2.inpaint(depth_img, (depth_img == 0).astype('uint8'), 2, cv2.INPAINT_TELEA)

        # constant depth filter TODO: make variable filter using 2d Lidar to detect wall distance
        depth_mask = cv2.inRange(depth_img, 0,99999)# 1600)
        # cv2.imshow('depth mask', depth_mask)
        
        # Depth cam visualization
        depth_vis = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = cv2.applyColorMap(depth_vis.astype('uint8'), cv2.COLORMAP_JET)
        depth_vis = cv2.resize(depth_vis, (rgb_img.shape[1], rgb_img.shape[0]))
        depth_filtered = cv2.bitwise_and(rgb_img, rgb_img, mask=depth_mask)

        # Apply LAB filter to RGB image to filter out yellow
        lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2LAB)

        lower = np.array([self.lm, self.am, self.bm])
        upper = np.array([self.lM, self.aM, self.bM])
        lab_mask = cv2.inRange(lab_img, lower, upper)
        lab_and_depth_masked = cv2.bitwise_and(depth_filtered, depth_filtered, mask=lab_mask)
        lab_and_depth_mask = cv2.bitwise_and(lab_mask, depth_mask)
        cv2.erode(lab_and_depth_mask, np.ones((5, 5), np.uint8), iterations=5) # 
        # cv2.imshow('ldor', lab_and_depth_mask)

        # floodfill the road - segmenting 
        ffm = np.zeros((h+2, w+2), np.uint8)
        rldm = lab_and_depth_mask.copy()

        seedpoint = None
        for offset in range(w //2 + 1):
            x_right = x + offset
            if x_right < w and lab_and_depth_mask[h-1, x_right] > 0:
                seedpoint = (x_right, h-1)
                break

            x_left = x - offset
            if x_left >= 0 and lab_and_depth_mask[h-1, x_left] > 0:
                seedpoint = (x_left, h-1)
                break

        if seedpoint is not None:
            cv2.floodFill(rldm, ffm, seedpoint, 255) 
            
        # cv2.imshow('ff',rldm)
        ffm = ffm[1:-1, 1:-1]
        ff_mask = ffm.astype(np.uint8) * 255 
        ff_img = cv2.bitwise_and(lab_and_depth_masked, lab_and_depth_masked, mask=ff_mask)

        # inpaint the road - fill holes 
        _h, _w = ff_mask.shape[:2]
        mask_flood = ff_mask.copy()
        # need open edges to not fill area surrounded at the edge.
        mask_flood[0,:] = 0;  mask_flood[:, 0] = 0; mask_flood[:,-1] = 0 # don't open bottom row
        mask_tmp = np.zeros((_h+2, _w+2), np.uint8)

        # Flood fill background from top-left corner
        cv2.floodFill(mask_flood, mask_tmp, (0,0), 255)

        # Invert floodfilled image
        mask_flood_inv = cv2.bitwise_not(mask_flood)

        # Combine with original to fill holes
        filled_mask = ff_mask | mask_flood_inv
        # cv2.imshow('masks', np.hstack((mask_flood_inv, ff_mask)))
        road_img = cv2.bitwise_and(rgb_img, rgb_img, mask=filled_mask)

        # for _y in range(20, h - 20, 20):
        #     depth = depth_img[_y, x]
        #     dist = math.sqrt(depth**2 - 245**2)
        #     cv2.putText(rgb_img, f'dep:{depth},dst:{dist:.1f}', (x, _y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        #     cv2.putText(rgb_img, f'dep:{depth},dst:{dist:.1f}', (x, _y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        #     cv2.rectangle(rgb_img, (x, _y), (x, _y), (0, 0, 255), 3)

        merged = np.hstack((rgb_img, depth_filtered, depth_vis))
        # cv2.imshow('cam view', merged)
        # merged2 = np.hstack((lab_and_depth_masked, ff_img, road_img))
        # cv2.imshow('2nd view', merged2)

        ###tmp
        if seedpoint:
            cv2.rectangle(filled_mask, seedpoint, seedpoint, 128, 3)
        ###


        # try to map the road
        left_road_edge = None
        step = -1 if filled_mask[-1, w//2 - 1] else 1
        _x = w//2 - 1
        while (1 <= _x < w - 1):
            if filled_mask[-1, _x] == (255 if step > 0 else 0):
                break
            _x += step

        _y = h - 1
        while (0 < _y):
            if filled_mask[_y, _x] == (255 if step > 0 else 0):
                break
            _y -= 1
        cv2.rectangle(filled_mask, (_x, _y), (_x, _y), 0, 3)
        cv2.rectangle(filled_mask, (_x, _y), (_x, _y), 255, 1)
        
        print(_x, _y)
        # out = cv2.cvtColor(filled_mask, cv2.COLOR_GRAY2BGR)

        # for y in range(0, h, 5):  # step through rows
        #     row = filled_mask[y, :]
        #     xs = np.where(row > 0)[0]  # road pixels
        #     if len(xs) > 0:
        #         x_left, x_right = xs[0], xs[-1]
        #         x_center = (x_left + x_right) // 2
        #         cv2.circle(out, (x_center, y), 2, (0, 0, 255), -1)
        #         cv2.circle(out, (x_left, y), 2, (255, 0, 0), -1)
        #         cv2.circle(out, (x_right, y), 2, (0, 255, 0), -1)
        
        # out = cv2.resize(out, (640, 480), interpolation=cv2.INTER_NEAREST_EXACT)
        # cv2.imshow("centerline", out)

        start_y = h - 1       # start from bottom
        step_y = -1           # scan upward
        center_x = w // 2

        left_points = []
        right_points = []
        center_points = []

        # Initialize previous edges (for fallback)
        prev_x_left = 0
        prev_x_right = w-1

        # Scan upward from bottom
        out = cv2.cvtColor(filled_mask, cv2.COLOR_GRAY2BGR)
        for y in range(start_y, 50, step_y):
            row = filled_mask[y, :]
            # if row[0] == 255 or row[-1] == 255:
            #     continue
            if np.all(row== 255):
                continue  # skip this row
            # Find all contiguous 255 segments
            segments = []
            start = None
            for x in range(w):
                if row[x] == 255 and start is None:
                    start = x
                elif row[x] == 0 and start is not None:
                    segments.append((start, x-1))
                    start = None
            if start is not None:
                segments.append((start, w-1))

            # Pick the widest segment
            if len(segments) == 0:
                # fallback to previous row
                x_left, x_right = prev_x_left, prev_x_right
            else:
                x_left, x_right = max(segments, key=lambda s: s[1]-s[0])
                prev_x_left, prev_x_right = x_left, x_right

            if row[0] != 255:
                left_points.append((x_left, y))
                cv2.rectangle(out, (x_left, y), (x_left, y), (255, 255, 0))
            if row[-1] != 255:
                right_points.append((x_right, y))
                cv2.rectangle(out, (x_right, y), (x_right, y), (255, 255, 0))
            center_points.append(((x_left + x_right)//2, y))

        # Convert to numpy arrays
        left_points = np.array(left_points)
        right_points = np.array(right_points)
        center_points = np.array(center_points)

        # Extrapolate lines to top
        def extrapolate_line(points, y_end, image_width):
            if len(points) < 2:
                return points
            y_vals = points[:,1]
            x_vals = points[:,0]
            coeffs = np.polyfit(y_vals, x_vals, 1)  # x = f(y)
            y_fit = np.linspace(y_vals[0], y_end, 200)
            x_fit = np.polyval(coeffs, y_fit)
            # x_fit = np.clip(x_fit, 0, image_width-1)
            return np.vstack((x_fit, y_fit)).T.astype(np.int32)

        y_top = 0
        left_line = extrapolate_line(left_points, y_top, w)
        right_line = extrapolate_line(right_points, y_top, w)
        center_line = extrapolate_line(center_points, y_top, w)

        # Draw lines on color image
        # cv2.polylines(out, [left_line.reshape(-1,1,2)], False, (0,255,0), 2)     # left edge
        # cv2.polylines(out, [right_line.reshape(-1,1,2)], False, (0,0,255), 2)    # right edge
        # cv2.polylines(out, [center_line.reshape(-1,1,2)], False, (255,0,0), 2)   # center line

        # out = cv2.resize(out, (640, 480), interpolation=cv2.INTER_NEAREST_EXACT)
        # cv2.imshow("road edges and centerline", out)
            
        # final_mask_view = cv2.resize(filled_mask, (640, 480), interpolation=cv2.INTER_NEAREST_EXACT)
        # cv2.imshow('final mask', final_mask_view)
        # cv2.waitKey(1)
        

    def rgb_callback(self, msg: Image):
        image = self.cv_bridge.imgmsg_to_cv2(msg, 'rgb8')

    def depth_callback(self, msg: Image):
        image = self.cv_bridge.imgmsg_to_cv2(msg, '16UC1')
        depth_vis = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = depth_vis.astype('uint8')
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        depth_vis = cv2.medianBlur(depth_vis, 5)
        depth_vis = cv2.medianBlur(depth_vis, 5)
        depth_vis = cv2.medianBlur(depth_vis, 5)
        cv2.imshow('depth', depth_vis)
        cv2.waitKey(1)




def main(args=None):
    cv2.namedWindow('final mask')
    cv2.resizeWindow('final mask', 640, 480)
    cv2.moveWindow('final mask', 0, 900)
    
    rclpy.init(args=args)
    node = CVTest()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
