#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/03/11
# @author:aiden
# Lane and crosswalk detection module
import os
import cv2
import queue
import threading
import numpy as np
import sdk.common as common
from cv_bridge import CvBridge

bridge = CvBridge()

# lab_config.yaml 파일 경로 설정 및 데이터 로딩
username = os.getenv("USER")
if username == "ubuntu":
    config_path = "/home/ubuntu/software/lab_tool/lab_config.yaml"
else:
    config_path = "/home/intel/ros2_ws/lab_config.yaml"
lab_data = common.get_yaml_data(config_path)

class LaneDetector(object):
    def __init__(self, color_to_detect="yellow"):
        self.lane_color_name = color_to_detect
        
        # 이미지 처리 크기
        self.img_width = 320
        self.img_height = 240

        # 1. 좌측 차선 감지를 위한 ROI (Region of Interest)
        self.lane_roi_y_start = int(self.img_height * 0.5)
        self.lane_roi_y_end = self.img_height
        self.lane_roi_x_start = 0
        self.lane_roi_x_end = int(self.img_width * 0.4)

        # 2. 횡단보도 감지를 위한 ROI
        self.crosswalk_roi_y_start = int(self.img_height * 0.6)
        self.crosswalk_roi_y_end = int(self.img_height * 1.0)
        self.crosswalk_roi_x_start = int(self.img_width * 0.1)
        self.crosswalk_roi_x_end = int(self.img_width * 0.85)

        # 3. 횡단보도 판단 기준: ROI 면적 대비 흰색 픽셀의 '비율'
        self.stop_line_ratio_threshold = 0.25

        # 횡단보도 ROI의 전체 면적을 미리 계산
        roi_width = self.crosswalk_roi_x_end - self.crosswalk_roi_x_start
        roi_height = self.crosswalk_roi_y_end - self.crosswalk_roi_y_start
        self.crosswalk_roi_area = roi_width * roi_height


    def _get_binary_mask(self, img_lab, color_name):
        """LAB 이미지와 색상 이름으로 이진화 마스크를 생성합니다."""
        try:
            color_range = lab_data['lab']['Stereo'][color_name]
            mask = cv2.inRange(img_lab, tuple(color_range['min']), tuple(color_range['max']))
            return mask
        except KeyError:
            return np.zeros((self.img_height, self.img_width), dtype=np.uint8)

    def get_binary(self, original_image):
        """
        [수정]
        이미지를 처리하여 (1)시각화 결과물과 (2)흰색 마스크, (3)노란색 마스크를 반환합니다.
        """
        resized = cv2.resize(original_image, (self.img_width, self.img_height))
        result_image = resized.copy()
        img_lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        img_blur = cv2.GaussianBlur(img_lab, (5, 5), 3)
        mask_white = self._get_binary_mask(img_blur, 'white')
        mask_yellow = self._get_binary_mask(img_blur, self.lane_color_name)

        # 횡단보도 감지 및 시각화
        crosswalk_roi_mask = mask_white[self.crosswalk_roi_y_start:self.crosswalk_roi_y_end, 
                                        self.crosswalk_roi_x_start:self.crosswalk_roi_x_end]
        white_pixel_count = cv2.countNonZero(crosswalk_roi_mask)
        current_ratio = 0.0
        if self.crosswalk_roi_area > 0:
            current_ratio = white_pixel_count / self.crosswalk_roi_area
        ratio_text = f"White Ratio: {current_ratio:.2f}"
        cv2.putText(result_image, ratio_text, (self.crosswalk_roi_x_start, self.crosswalk_roi_y_start - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.rectangle(result_image, (self.crosswalk_roi_x_start, self.crosswalk_roi_y_start), (self.crosswalk_roi_x_end, self.crosswalk_roi_y_end), (0, 255, 0), 2)
        if current_ratio > self.stop_line_ratio_threshold:
            cv2.rectangle(result_image, (self.crosswalk_roi_x_start, self.crosswalk_roi_y_start), (self.crosswalk_roi_x_end, self.crosswalk_roi_y_end), (0, 0, 255), 2)
            cv2.putText(result_image, "STOP LINE", (self.crosswalk_roi_x_start, self.crosswalk_roi_y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 좌측 차선 감지 및 시각화
        roi_mask_left = np.zeros_like(mask_yellow)
        cv2.rectangle(roi_mask_left, (self.lane_roi_x_start, self.lane_roi_y_start), (self.lane_roi_x_end, self.lane_roi_y_end), 255, -1)
        yellow_lane_roi = cv2.bitwise_and(mask_yellow, mask_yellow, mask=roi_mask_left)
        contours, _ = cv2.findContours(yellow_lane_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.rectangle(result_image, (self.lane_roi_x_start, self.lane_roi_y_start), (self.lane_roi_x_end, self.lane_roi_y_end), (255, 0, 0), 2)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 50]
        if valid_contours:
            largest_contour = max(valid_contours, key=cv2.contourArea)
            cv2.drawContours(result_image, [largest_contour], -1, (255, 255, 255), 3)
            x, y, w, h = cv2.boundingRect(largest_contour)
            lane_x = x + w // 2
            cv2.line(result_image, (lane_x, self.lane_roi_y_start), (lane_x, self.lane_roi_y_end), (255, 0, 255), 2)

        # [수정] 반환 값 변경
        return result_image, mask_white, mask_yellow

    def __call__(self, mask_white, mask_yellow):
        # 1. 항상 좌측 차선을 먼저 찾아 lane_x를 확보합니다.
        lane_x = -1  # 기본값을 -1로 설정
        
        roi_mask_left = np.zeros_like(mask_yellow)
        cv2.rectangle(roi_mask_left, (self.lane_roi_x_start, self.lane_roi_y_start), (self.lane_roi_x_end, self.lane_roi_y_end), 255, -1)
        yellow_lane_roi = cv2.bitwise_and(mask_yellow, mask_yellow, mask=roi_mask_left)
        contours, _ = cv2.findContours(yellow_lane_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = [c for c in contours if cv2.contourArea(c) > 50]
        if valid_contours:
            largest_contour = max(valid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            lane_x = x + w // 2
            # 여기서 바로 return하지 않습니다.

        # 2. 횡단보도 여부를 확인합니다.
        is_stop_line = False
        crosswalk_roi = mask_white[self.crosswalk_roi_y_start:self.crosswalk_roi_y_end, 
                                self.crosswalk_roi_x_start:self.crosswalk_roi_x_end]
        if self.crosswalk_roi_area > 0 and \
        (cv2.countNonZero(crosswalk_roi) / self.crosswalk_roi_area) > self.stop_line_ratio_threshold:
            is_stop_line = True

        # 3. 결과를 조합하여 최종 반환값을 결정합니다.
        if is_stop_line:
            # 횡단보도가 감지되면, 위에서 계산한 lane_x와 함께 반환합니다.
            return "STOP_LINE", lane_x

        if lane_x != -1:
            # 횡단보도는 없지만 차선이 감지되면, GO_STRAIGHT를 반환합니다.
            return "GO_STRAIGHT", lane_x

        # 둘 다 감지되지 않으면 None을 반환합니다.
        return None, -1

# --- 아래 코드는 이 파일을 단독으로 실행하여 테스트할 때 사용됩니다. ---
image_queue = queue.Queue(2)
def image_callback(ros_image):
    cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
    if image_queue.full():
        image_queue.get()
    image_queue.put(np.array(cv_image, dtype=np.uint8))

def main():
    lane_detect = LaneDetector("yellow")
    running = True
    while running:
        try:
            image = image_queue.get(block=True, timeout=1)
        except queue.Empty:
            if not running: break
            else: continue
        
        # [수정] 테스트 코드도 변경된 반환 값에 맞춰 수정
        visual_image, _, _ = lane_detect.get_binary(image)
        
        cv2.imshow('Test Visualization', visual_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    import rclpy
    from sensor_msgs.msg import Image
    rclpy.init()
    node = rclpy.create_node('lane_detect_test')
    node.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image', image_callback, 1)
    threading.Thread(target=main, daemon=True).start()
    rclpy.spin(node)