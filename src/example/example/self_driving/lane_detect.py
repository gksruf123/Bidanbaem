#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/03/11
# @author:aiden
# lane detection for autonomous driving
import os
import cv2
import math
import queue
import threading
import numpy as np
import sdk.common as common
from cv_bridge import CvBridge
import numpy as np

bridge = CvBridge()

username = os.getenv("USER")
if username == "ubuntu":
    config_path = "/home/ubuntu/software/lab_tool/lab_config.yaml"
else:
    config_path = "/home/intel/ros2_ws/lab_config.yaml"
lab_data = common.get_yaml_data(config_path)

# lab_data = common.get_yaml_data("/home/ubuntu/software/lab_tool/lab_config.yaml")

class LaneDetector(object):
    def __init__(self, color):
        # lane color
        self.target_color = color
        # (x1,y1,x2,y2) in 320x240 bin coords
        self.last_line = None 
        # ROI for lane detection
        if os.environ['DEPTH_CAMERA_TYPE'] == 'ascamera':
            self.rois = ((338, 360, 0, 320, 0.7), (292, 315, 0, 320, 0.2), (248, 270, 0, 320, 0.1))
        else:
            self.rois = ((450, 480, 0, 320, 0.7), (390, 480, 0, 320, 0.2), (330, 480, 0, 320, 0.1))
        self.weight_sum = 1.0

        self.prev_x = None
        self.prev_ang = None
        self.smooth_x = None
        self.smooth_ang = None  

        self.y_ratio = 0.75

    # 클래스 메서드로 추가: ROI 로컬 좌표의 선분이 y=y_t에서 가지는 x 위치
    def _x_at_y(self, x1, y1, x2, y2, y_t):
        dy = (y2 - y1)
        dx = (x2 - x1)
        if abs(dy) < 1e-6:
            return 0.5*(x1 + x2)
        # x = x1 + (y_t - y1) * dx/dy
        return x1 + (y_t - y1) * (dx / (dy + 1e-6))

    def set_roi(self, roi):
        self.rois = roi

    @staticmethod
    def get_area_max_contour(contours, threshold=100):
        '''
        obtain the contour corresponding to the maximum area
        :param contours:
        :param threshold:
        :return:
        '''
        contour_area = zip(contours, tuple(map(lambda c: math.fabs(cv2.contourArea(c)), contours)))
        contour_area = tuple(filter(lambda c_a: c_a[1] > threshold, contour_area))
        if len(contour_area) > 0:
            max_c_a = max(contour_area, key=lambda c_a: c_a[1])
            return max_c_a
        return None
    
    def add_horizontal_line(self, image):
        #   |____  --->   |————   ---> ——
        h, w = image.shape[:2]
        roi_w_min = int(w/2)
        roi_w_max = w
        roi_h_min = 0
        roi_h_max = h
        roi = image[roi_h_min:roi_h_max, roi_w_min:roi_w_max]  # crop the right half
        flip_binary = cv2.flip(roi, 0)  # flip upside down
        max_y = cv2.minMaxLoc(flip_binary)[-1][1]  # extract the coordinates of the top-left point with a value of 255

        return h - max_y

    def add_vertical_line_far(self, image):
        h, w = image.shape[:2]
        roi_w_min = int(w/8)
        roi_w_max = int(w/2)
        roi_h_min = 0
        roi_h_max = h
        roi = image[roi_h_min:roi_h_max, roi_w_min:roi_w_max]
        flip_binary = cv2.flip(roi, -1)  # flip the image horizontally and vertically
        #cv2.imshow('1', flip_binary)
        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(ret)
        # minVal：the minimum value
        # maxVal：the maximum value
        # minLoc：the location of the minimum value
        # maxLoc：the location of the maximum value
        # the order of traversal is: first rows, then columns, with rows from left to right and columns from top to bottom
        (x_0, y_0) = cv2.minMaxLoc(flip_binary)[-1]  # extract the coordinates of the top-left point with a value of 255
        y_center = y_0 + 55
        roi = flip_binary[y_center:, :]
        (x_1, y_1) = cv2.minMaxLoc(roi)[-1]
        down_p = (roi_w_max - x_1, roi_h_max - (y_1 + y_center))
        
        y_center = y_0 + 65
        roi = flip_binary[y_center:, :]
        (x_2, y_2) = cv2.minMaxLoc(roi)[-1]
        up_p = (roi_w_max - x_2, roi_h_max - (y_2 + y_center))

        up_point = (0, 0)
        down_point = (0, 0)
        if up_p[1] - down_p[1] != 0 and up_p[0] - down_p[0] != 0:
            up_point = (int(-down_p[1]/((up_p[1] - down_p[1])/(up_p[0] - down_p[0])) + down_p[0]), 0)
            down_point = (int((h - down_p[1])/((up_p[1] - down_p[1])/(up_p[0] - down_p[0])) + down_p[0]), h)

        return up_point, down_point

    def add_vertical_line_near(self, image):
        # ——|         |——        |
        #   |   --->  |     --->
        h, w = image.shape[:2]
        roi_w_min = 0
        roi_w_max = int(w/2)
        roi_h_min = int(h/2)
        roi_h_max = h
        roi = image[roi_h_min:roi_h_max, roi_w_min:roi_w_max]
        flip_binary = cv2.flip(roi, -1)  # flip the image horizontally and vertically
        #cv2.imshow('1', flip_binary)
        (x_0, y_0) = cv2.minMaxLoc(flip_binary)[-1]  # extract the coordinates of the top-left point with a value of 255
        down_p = (roi_w_max - x_0, roi_h_max - y_0)

        (x_1, y_1) = cv2.minMaxLoc(roi)[-1]
        y_center = int((roi_h_max - roi_h_min - y_1 + y_0)/2)
        roi = flip_binary[y_center:, :] 
        (x, y) = cv2.minMaxLoc(roi)[-1]
        up_p = (roi_w_max - x, roi_h_max - (y + y_center))

        up_point = (0, 0)
        down_point = (0, 0)
        if up_p[1] - down_p[1] != 0 and up_p[0] - down_p[0] != 0:
            up_point = (int(-down_p[1]/((up_p[1] - down_p[1])/(up_p[0] - down_p[0])) + down_p[0]), 0)
            down_point = down_p

        return up_point, down_point, y_center

    def get_binary(self, image):
        resized = cv2.resize(image, (320, 240))
        img_lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        img_blur = cv2.GaussianBlur(img_lab, (5, 5), 3)
        mask = cv2.inRange(
            img_blur,
            tuple(lab_data['lab']['Stereo'][self.target_color]['min']),
            tuple(lab_data['lab']['Stereo'][self.target_color]['max'])
        )
        eroded  = cv2.erode(mask,  cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))

        h, w = dilated.shape[:2]
        out = dilated.copy()

        # 3사분면 ROI (로컬 좌표계)
        roi_left = (h//2, h, 0, w//2)  # (y0:y1, x0:x1)
        blob_left = dilated[roi_left[0]:roi_left[1], roi_left[2]:roi_left[3]]

        # 허프 파라미터(안정화용)
        lines_left = cv2.HoughLinesP(
            blob_left, 1, np.pi/180,
            threshold=100,
            minLineLength=int(0.5*(roi_left[1]-roi_left[0])),
            maxLineGap=8
        )

        roi_hough = np.zeros_like(blob_left)
        self.last_line = None

        if lines_left is not None:
            # 후보 필터링: 각도/바닥 근접
            cand = []
            y0, y1, x0, x1 = roi_left
            roi_h = y1 - y0

            # 전역 y(=화면 75%)를 ROI 로컬로 변환해서 동일 기준으로 사용
            y_target_global = int(h * self.y_ratio)
            y_target_local  = np.clip(y_target_global - y0, 0, roi_h - 1)
            bottom_thresh   = int(0.8*roi_h)

            for L in lines_left:
                x1r, y1r, x2r, y2r = L[0]
                ang    = np.degrees(np.arctan2((y2r - y1r), (x2r - x1r)))
                length = np.hypot(x2r - x1r, y2r - y1r)

                if not (20 <= abs(ang) <= 80):
                    continue
                if max(y1r, y2r) < bottom_thresh:
                    continue

                x_mid = self._x_at_y(x1r, y1r, x2r, y2r, y_target_local)
                cand.append((x1r, y1r, x2r, y2r, ang, length, x_mid))

            # 스코어: 길이 우선 + 지난 프레임 근접성
            best = None
            best_score = 1e18
            prev_ang = self.prev_ang if self.prev_ang is not None else None
            prev_x   = self.prev_x   if self.prev_x   is not None else None

            for (x1r, y1r, x2r, y2r, ang, length, x_mid) in cand:
                score = -1.0*length
                if prev_ang is not None: score += 0.7*abs(ang - prev_ang)
                if prev_x  is not None: score += 0.3*abs(x_mid - prev_x)
                if score < best_score:
                    best_score = score
                    best = (x1r, y1r, x2r, y2r, ang, x_mid)

            if best is not None:
                x1r, y1r, x2r, y2r, ang, x_mid = best

                # ROI 로컬에 선만 그려서 3사분면 교체
                cv2.line(roi_hough, (x1r, y1r), (x2r, y2r), 255, 1)

                # 전체 좌표로 저장(후속 계산용)
                self.last_line = (x1r + x0, y1r + y0, x2r + x0, y2r + y0)

                # EMA 스무딩(후보 선택 안정화용)
                alpha = 0.3
                self.smooth_ang = ang   if self.smooth_ang is None else (1-alpha)*self.smooth_ang + alpha*ang
                self.smooth_x   = x_mid if self.smooth_x   is None else (1-alpha)*self.smooth_x   + alpha*x_mid
                self.prev_ang, self.prev_x = float(self.smooth_ang), float(self.smooth_x)

                out[roi_left[0]:roi_left[1], roi_left[2]:roi_left[3]] = roi_hough

        # (선택) thinning
        try:
            thinned = cv2.ximgproc.thinning(out, thinningType=cv2.ximgproc.THINNING_GUOHALL)
        except Exception:
            thinned = out

        return thinned


    def __call__(self, image, result_image):
        h, w = image.shape[:2]
        roi_right = (h//2, h, w//2, w)
        blob_right = image[roi_right[0]:roi_right[1], roi_right[2]:roi_right[3]]

        lane_x, lane_angle = None, None

        if self.last_line is not None:
            x1, y1, x2, y2 = self.last_line

            # 각도: 스무딩 값이 있으면 사용, 없으면 계산
            lane_angle = float(self.smooth_ang) if self.smooth_ang is not None \
                        else float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

            # 최종 lane_x는 항상 "전역 y"에서 라인으로 딱 한 번만 계산
            y_target = int(h * self.y_ratio)
            dx = (x2 - x1); dy = (y2 - y1)
            lane_x = 0.5*(x1+x2) if abs(dy)<1e-6 else x1 + (y_target - y1) * (dx/(dy + 1e-6))
            lane_x = float(np.clip(lane_x, 0, w - 1))

        # 우측 차선 존재 판정(원하면 행-점유 방식으로 교체 가능)
        # count_pix = cv2.countNonZero(blob_right)
        # threshold = int((h - h//2) * 0.35)
        # has_right = count_pix > threshold
        # 권장 대안:
        rows_touched = int(np.count_nonzero(blob_right.any(axis=1)))
        has_right = rows_touched >= int((roi_right[1]-roi_right[0]) * 0.10)

        if lane_x is not None:
            scale_x = result_image.shape[1] / float(w)
            lane_x_scaled = lane_x * scale_x
            status = "GO_STRAIGHT" if has_right else "STOP_LINE"
            return result_image, status, lane_angle, lane_x_scaled
        else:
            return result_image, None, None, None


image_queue = queue.Queue(2)
def image_callback(ros_image):
    cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
    bgr_image = np.array(cv_image, dtype=np.uint8)
    if image_queue.full():
        # if the queue is full, remove the oldest image
        image_queue.get()
        # put the image into the queue
    image_queue.put(bgr_image)

def main():
    running = True

    while running:
        try:
            image = image_queue.get(block=True, timeout=1)
        except queue.Empty:
            if not running:
                break
            else:
                continue
        binary_image = lane_detect.get_binary(image)
        cv2.imshow('binary', binary_image)
        img = image.copy()

        '''
        up, down = lane_detect.add_vertical_line_far(binary_image)
        #up, down, center = lane_detect.add_vertical_line_near(binary_image)
        cv2.line(img, up, down, (255, 255, 255), 10)
        '''
        cv2.imshow('image', img)
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:  # press Q or Esc to quit
            break

    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    import rclpy
    from sensor_msgs.msg import Image
    rclpy.init()
    node = rclpy.create_node('lane_detect')
    lane_detect = LaneDetector('yellow')
    node.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image', image_callback, 1)
    threading.Thread(target=main, daemon=True).start()
    rclpy.spin(node)




