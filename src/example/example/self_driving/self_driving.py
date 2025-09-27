#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/03/28 (edited for landmark-PID localization)
# author: aiden (+chatgpt)

import os
import cv2
import math
import time
import rclpy
import threading
import numpy as np
import sdk.pid as pid
import sdk.fps as fps
from rclpy.node import Node
import sdk.common as common
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from interfaces.msg import ObjectsInfo
from std_srvs.srv import SetBool, Trigger
from sdk.common import colors, plot_one_box
from example.self_driving import lane_detect
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from ros_robot_controller_msgs.msg import SetPWMServoState
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# ==================== [체크포인트 기대 관측값] ====================
#  각 코너/횡단보도 앞에서 "어떤 랜드마크가 몇 도, 몇 m"에 보여야 하는지.
#  - bearing_deg: 카메라 정면을 0°, 왼쪽 +, 오른쪽 - (HFOV 기준)
#  - distance_m: 깊이에서 측정되는 중심 거리(근사)
#  - classes: 해당 위치에서 사용할 랜드마크 후보(여러 개 넣으면, 먼저 보이는 걸 사용)
#
#  !!! TODO: 트랙에서 1회 계측해 숫자만 고치면 끝 !!!
CHECKPOINT_EXPECTED = {
    # 시작 후 첫 횡단보도 앞
    "POINT_1": {"classes": ["crosswalk"],
                "bearing_deg": -5.82, "distance_m": 1.719,
                "bearing_tol_deg": 6.0, "dist_tol_m": 0.08},

    "POINT_2": {"classes": ["crosswalk"],
                "bearing_deg": -22.91, "distance_m": 0.952,
                "bearing_tol_deg": 6.0, "dist_tol_m": 0.05},

    "POINT_3": {"classes": ["crosswalk"],
                "bearing_deg": -5.82, "distance_m": 1.719,
                "bearing_tol_deg": 6.0, "dist_tol_m": 0.08},

    "POINT_4": {"classes": ["crosswalk"],
                "bearing_deg": -22.91, "distance_m": 0.952,
                "bearing_tol_deg": 6.0, "dist_tol_m": 0.05},

    "POINT_5": {"classes": ["crosswalk"],
                "bearing_deg": 28.57, "distance_m": 1.629,
                "bearing_tol_deg": 6.0, "dist_tol_m": 0.08},

    "POINT_6": {"classes": ["crosswalk"],
                "bearing_deg": -6.85, "distance_m": 0.480,
                "bearing_tol_deg": 6.0, "dist_tol_m": 0.05},
}

class SelfDrivingNode(Node):
    def __init__(self, name):
        super().__init__(name, allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)
        self.name = name
        self.is_running = True

        # === PID: 헤딩/거리 별도로 운용 ===
        #  - heading: 각도 오차 → 각속도 명령
        #  - dist   : 거리 오차 → 선속도 명령
        self.pid_heading = pid.PID(0.9, 0.0, 0.10)  # 튠 포인트: P 0.6~1.2, D 0.05~0.15
        self.pid_dist    = pid.PID(0.8, 0.0, 0.05)  # 튠 포인트: P 0.6~1.0, D 0.03~0.08

        self.param_init()

        self.latest_image = None
        self.latest_depth = None      # 깊이 프레임 저장
        self.img_lock = threading.RLock()
        self.depth_lock = threading.RLock()

        self.fps = fps.FPS()
        self.classes = ['go', 'right', 'park', 'red', 'green', 'crosswalk']
        self.display = True
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.colors = common.Colors()
        self.machine_type = os.environ.get('MACHINE_TYPE')
        self.lane_detect = lane_detect.LaneDetector("yellow")

        # 카메라 파라미터(수정 가능)
        self.hfov_deg = 69.0   # 카메라 수평 화각(대부분 60~78°). TODO: 실제 값으로 교체 권장.
        # 선언(기본값)
        self.declare_parameter("depth_topic", "/ascamera/camera_publisher/depth0/image_raw")
        # 조회
        self.depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value


        # QoS: 센서 스트림은 BEST_EFFORT, depth=1
        self.sensor_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.servo_state_pub = self.create_publisher(SetPWMServoState, 'ros_robot_controller/pwm_servo/set_state', 1)
        self.result_publisher = self.create_publisher(Image, '~/image_result', 1)

        # Services (servers)
        self.create_service(Trigger, '~/enter', self.enter_srv_callback)   # enter the game
        self.create_service(Trigger, '~/exit', self.exit_srv_callback)     # exit the game
        self.create_service(SetBool, '~/set_running', self.set_running_srv_callback)

        # Service clients
        timer_cb_group = ReentrantCallbackGroup()
        self.client = self.create_client(Trigger, '/yolov5_ros2/init_finish')
        self.start_yolov5_client = self.create_client(Trigger, '/yolov5/start', callback_group=timer_cb_group)
        self.stop_yolov5_client  = self.create_client(Trigger, '/yolov5/stop',  callback_group=timer_cb_group)

        for cli, name_ in [(self.client, '/yolov5_ros2/init_finish'),
                           (self.start_yolov5_client, '/yolov5/start'),
                           (self.stop_yolov5_client, '/yolov5/stop')]:
            if not self._wait_service(cli, 5.0):
                self.get_logger().warn(f"Service {name_} not available within timeout; continuing anyway.")

        # 타이머는 1회성
        self.timer = self.create_timer(0.05, self.init_process, callback_group=timer_cb_group)

        # 정지-정렬용 PID (차선)
        self.pid_lane_yaw = pid.PID(0.7, 0.0, 0.10)   # 픽셀->각도 보정용
        self.pid_lane_lat = pid.PID(0.6, 0.0, 0.05)   # 측면(strafe) 보정용(메카넘)
            # 튠 팁: 회전이 느리면 pid_lane_yaw.Kp += 0.1, 흔들리면 Kd += 0.02.

    # ------------------------- 초기화/파라미터 -------------------------
    def param_init(self):
        self.start = False
        self.enter = False
        self.stop_flag = False           # (사용X) 이름 충돌 방지
        self.rotating = False
        self.objects_info = []
        self.object_sub = None
        self.image_sub = None
        self.depth_sub = None

        # 속도 설정
        self.normal_speed = 0.30
        self.slow_down_speed = 0.12

        # ========== 주행 플랜(횡단보도 4회 정지 + 코너 직후 보정 포함) ==========
        #  DRIVE(dist, speed)
        #  TURN_RIGHT(deg)
        #  STOP_PERCEIVE_LIGHT(frames, timeout)
        #  STOP_LOCALIZE(checkpoint_key)  ← 랜드마크로 위치 보정
        self.plan = [
            ("STOP_LOCALIZE", "POINT_1"),
            # ("STOP_ALIGN_LANE",),                # ← 추가
            ("DRIVE", 2.6, 1.0),
            ("TURN_RIGHT", 90),
            ("STOP_LOCALIZE", "POINT_2"),
            # ("STOP_ALIGN_LANE",),
            ("DRIVE", 2.6, 1.0),
            ("TURN_RIGHT", 90),
            ("STOP_LOCALIZE", "POINT_3"),
            # ("STOP_ALIGN_LANE",),
            ("DRIVE", 2.6, 1.0),
            ("TURN_RIGHT", 90),
            ("STOP_LOCALIZE", "POINT_4"),
            # ("STOP_ALIGN_LANE",),
            ("DRIVE", 1.4, 1.0),
            ("TURN_RIGHT", 90),
            ("STOP_LOCALIZE", "POINT_5"),
            # ("STOP_ALIGN_LANE",),
            ("DRIVE", 1.7, 1.0),
            ("STOP_LOCALIZE", "POINT_6"),
            # ("STOP_ALIGN_LANE",),
            # ("DRIVE", 1.0, 1.5),
            # ("DRIVE", 1.0, 1.5),
            # ("STOP_LOCALIZE", "XWALK_4"),
            # ("STOP_ALIGN_LANE",),
            # ("DRIVE", 0.3, 1.5),
            # ("STOP_LOCALIZE", "CORNER_4"),
            # ("STOP_ALIGN_LANE",),
            # ("DRIVE", 1.7, 1.5),
            ("PARK", 0.40, 0.80),
        ]
        self.step_idx = 0
        self.in_action = False

    def _wait_service(self, client, sec=5.0):
        t0 = time.time()
        while not client.wait_for_service(timeout_sec=0.2):
            if time.time() - t0 > sec:
                return False
        return True

    def init_process(self):
        try: self.timer.cancel()
        except Exception: pass

        self._hard_stop()

        res = self.send_request(self.start_yolov5_client, Trigger.Request(), timeout_sec=5.0)
        if not res or not getattr(res, 'success', False):
            self.get_logger().warn("Failed to start YOLOv5 service")

        time.sleep(0.2)

        # 자동 시작
        self.display = True
        self.enter_srv_callback(Trigger.Request(), Trigger.Response())
        request = SetBool.Request()
        request.data = True
        self.set_running_srv_callback(request, SetBool.Response())

        # 쓰레드 시작
        threading.Thread(target=self.main, daemon=True).start()
        threading.Thread(target=self.run_course, daemon=True).start()
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def get_node_state(self, request, response):
        response.success = True
        return response

    def send_request(self, client, msg, timeout_sec=5.0):
        future = client.call_async(msg)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if not future.done():
            return None
        return future.result()

    # ------------------------- 서비스 콜백 -------------------------
    def enter_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "self driving enter")
        with self.lock:
            self.start = False
            self.image_sub = self.create_subscription(
                Image, '/ascamera/camera_publisher/rgb0/image', self.image_callback, self.sensor_qos
            )
            self.depth_sub = self.create_subscription(
                Image, self.depth_topic, self.depth_callback, self.sensor_qos
            )
            self.object_sub = self.create_subscription(
                ObjectsInfo, '/yolov5_ros2/object_detect', self.get_object_callback, 1
            )
            self._hard_stop()
            self.enter = True
        response.success = True
        response.message = "enter"
        return response

    def exit_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "self driving exit")
        with self.lock:
            try:
                if self.image_sub is not None:
                    self.destroy_subscription(self.image_sub); self.image_sub = None
                if self.depth_sub is not None:
                    self.destroy_subscription(self.depth_sub); self.depth_sub = None
                if self.object_sub is not None:
                    self.destroy_subscription(self.object_sub); self.object_sub = None
            except Exception as e:
                self.get_logger().info('\033[1;32m%s\033[0m' % str(e))
            self._hard_stop()
        self.param_init()
        response.success = True
        response.message = "exit"
        return response

    def set_running_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "set_running")
        with self.lock:
            self.start = request.data
            if not self.start:
                self._hard_stop()
        response.success = True
        response.message = "set_running"
        return response

    def shutdown(self, signum=None, frame=None):
        self.is_running = False

    # ------------------------- 콜백/유틸 -------------------------
    def _hard_stop(self):
        self.mecanum_pub.publish(Twist())
        time.sleep(0.02)
        self.mecanum_pub.publish(Twist())

    def image_callback(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        with self.img_lock:
            self.latest_image = rgb_image

    def depth_callback(self, ros_image):
        # 16UC1 또는 32FC1(미터) 기준. 장치에 따라 스케일이 다를 수 있음.
        depth = self.bridge.imgmsg_to_cv2(ros_image)
        with self.depth_lock:
            self.latest_depth = depth

    def get_object_callback(self, msg):
        self.objects_info = msg.objects

    # Quaternion -> yaw(rad)
    def _yaw_from_quat(self, q):
        siny_cosp = 2.0 * (q.w*q.z + q.x*q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def _lane_center_x(self, img):
        """
        하부 ROI에서 노란 차선의 무게중심 x(px)를 반환.
        lane_detect 객체가 mask를 제공하면 사용, 없으면 HSV fallback.
        """
        if img is None:
            return None
        # 1) lane_detect가 mask 반환하는 경우
        mask = None
        try:
            if hasattr(self.lane_detect, "get_mask"):
                mask = self.lane_detect.get_mask(img)  # (H,W) uint8 0/255 가정
        except Exception:
            mask = None

        # 2) fallback: HSV로 노란색 마스크
        if mask is None:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            # 조명/카메라마다 다름. 시작값: 노란색 대역
            lower = np.array([15, 80, 80], dtype=np.uint8)
            upper = np.array([40, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)

        h, w = mask.shape[:2]
        roi = mask[int(h*0.55):, :]  # 하부 45%만 사용
        m = cv2.moments(roi, binaryImage=True)
        if m['m00'] < 1000:  # 픽셀 수 너무 적으면 실패
            return None
        cx = int(m['m10']/m['m00'])
        # ROI 좌표 → 원본 이미지 좌표의 x만 필요하므로 그대로 반환
        return cx, w

    # ==================== 픽셀→방위각, 깊이 추출 ====================
    def _bbox_center_bearing_deg(self, bbox, img_w):
        # bbox: [x1,y1,x2,y2]
        x1, _, x2, _ = bbox
        cx = 0.5*(x1 + x2)
        # 정면 0°, 왼쪽 +, 오른쪽 -
        hfov = math.radians(self.hfov_deg)
        # -hfov/2 .. +hfov/2 로 정규화
        bearing = ( (cx - img_w/2.0) / (img_w/2.0) ) * (hfov/2.0)
        return math.degrees(-bearing)  # 오른쪽이 음수 되도록 부호 반전

    def _depth_at_bbox_center(self, bbox, depth_img):
        if depth_img is None:
            return None
        x1, y1, x2, y2 = map(int, bbox)
        h, w = depth_img.shape[:2]
        x1 = max(0, min(w-1, x1)); x2 = max(0, min(w-1, x2))
        y1 = max(0, min(h-1, y1)); y2 = max(0, min(h-1, y2))
        cx = int(0.5*(x1+x2)); cy = int(0.5*(y1+y2))
        # 주변 5x5 median으로 노이즈 완화
        xs = slice(max(0,cx-2), min(w,cx+3))
        ys = slice(max(0,cy-2), min(h,cy+3))
        patch = depth_img[ys, xs]
        if patch.size == 0:
            return None
        d = float(np.median(patch))
        # 센서 타입별로 단위가 mm일 수 있음 → 0.001 곱해 m 단위로 보정
        if d > 5.0:   # 5m 넘는 값이 수시로 나오면 mm 스케일일 확률 높음
            d *= 0.001
        return d if (0.05 < d < 10.0) else None

    # ==================== 제자리 회전 ====================
    def rotate_right_90_in_place(self, yaw_rate=1.0):
        self.rotating = True
        twist = Twist()
        twist.angular.z = -abs(yaw_rate)
        t0 = time.time()
        duration = (math.pi/2)/abs(yaw_rate)
        while time.time() - t0 < duration and self.is_running:
            self.mecanum_pub.publish(twist); time.sleep(0.02)
        self._hard_stop(); self.rotating = False

    def rotate_right_90_feedback(self, source="odom", timeout=5.0):
        yaw_now = None
        self.rotating = True

        def odom_cb(msg):
            nonlocal yaw_now
            yaw_now = self._yaw_from_quat(msg.pose.pose.orientation)

        def imu_cb(msg):
            nonlocal yaw_now
            yaw_now = self._yaw_from_quat(msg.orientation)

        sub = self.create_subscription(Odometry, "/odom", odom_cb, self.sensor_qos) if source=="odom" \
              else self.create_subscription(Imu, "/imu", imu_cb, 1)

        t0 = time.time()
        while yaw_now is None and (time.time() - t0) < 1.0:
            rclpy.spin_once(self, timeout_sec=0.05)
        if yaw_now is None:
            try: self.destroy_subscription(sub)
            except: pass
            self.get_logger().warn("Yaw not available; fallback to open-loop 90°.")
            return self.rotate_right_90_in_place(1.0)

        yaw_start = yaw_now
        target_delta = -math.pi/2
        self.pid_heading.reset()
        max_rate = 1.2; min_rate = 0.2

        twist = Twist()
        t0 = time.time()
        while (time.time()-t0) < timeout and self.is_running:
            rclpy.spin_once(self, timeout_sec=0.01)
            if yaw_now is None: continue
            delta = ((yaw_now - yaw_start + math.pi)%(2*math.pi))-math.pi
            err = target_delta - delta
            w_cmd = self.pid_heading.step(err, dt=0.01)
            w_cmd = max(-max_rate, min(max_rate, w_cmd))
            if abs(w_cmd) < min_rate: w_cmd = -min_rate if err<0 else min_rate
            twist.angular.z = w_cmd
            self.mecanum_pub.publish(twist)
            if abs(err) < math.radians(1.2): break
            time.sleep(0.01)

        self._hard_stop()
        try: self.destroy_subscription(sub)
        except: pass
        self.rotating = False

    # ==================== 주행 프리미티브 (PID) ====================
    def drive_distance_straight(self, target_dist=1.5, speed=0.25,
                                use_heading_feedback=True, max_w=0.35,
                                timeout=20.0, source="odom"):
        start = None; pos = None; yaw_now = None; yaw_ref = None

        def odom_cb(msg):
            nonlocal pos, start, yaw_now
            x = msg.pose.pose.position.x; y = msg.pose.pose.position.y
            if start is None: start = (x, y)
            pos = (x, y); yaw_now = self._yaw_from_quat(msg.pose.pose.orientation)

        def imu_cb(msg):
            nonlocal yaw_now
            yaw_now = self._yaw_from_quat(msg.orientation)

        sub = self.create_subscription(Odometry, "/odom", odom_cb, self.sensor_qos) if source=="odom" \
              else self.create_subscription(Imu, "/imu", imu_cb, 1)

        twist = Twist(); v = speed; t0 = time.time()
        self.pid_heading.reset()

        while rclpy.ok() and (time.time() - t0) < timeout:
            if source=="odom" and pos is not None and start is not None:
                dist = math.hypot(pos[0]-start[0], pos[1]-start[1])
                if dist >= target_dist: break

            if self.rotating:
                self._hard_stop(); time.sleep(0.01); continue

            w_cmd = 0.0
            if use_heading_feedback:
                if yaw_ref is None and yaw_now is not None: yaw_ref = yaw_now
                if yaw_ref is not None and yaw_now is not None:
                    err = ((yaw_ref - yaw_now + math.pi)%(2*math.pi))-math.pi
                    w_cmd = self.pid_heading.step(err, dt=0.01)
                    w_cmd = max(-max_w, min(max_w, w_cmd))

            twist.linear.x = v; twist.angular.z = w_cmd
            self.mecanum_pub.publish(twist)

            rclpy.spin_once(self, timeout_sec=0.01); time.sleep(0.01)

        self._hard_stop()
        try: self.destroy_subscription(sub)
        except: pass

    def strafe_right_distance(self, target_dist=0.4, speed=0.2,
                              use_heading_feedback=True, timeout=8.0, source="odom"):
        start = None; pos = None; yaw_now = None; yaw_ref = None

        def odom_cb(msg):
            nonlocal pos, start, yaw_now
            x = msg.pose.pose.position.x; y = msg.pose.pose.position.y
            if start is None: start = (x, y)
            pos = (x, y); yaw_now = self._yaw_from_quat(msg.pose.pose.orientation)

        def imu_cb(msg):
            nonlocal yaw_now
            yaw_now = self._yaw_from_quat(msg.orientation)

        sub = self.create_subscription(Odometry, "/odom", odom_cb, self.sensor_qos) if source=="odom" \
              else self.create_subscription(Imu, "/imu", imu_cb, 1)

        twist = Twist(); v_y = -abs(speed); t0 = time.time()
        self.pid_heading.reset(); max_w = 0.35

        while rclpy.ok() and (time.time()-t0) < timeout:
            if self.rotating:
                self._hard_stop(); time.sleep(0.01); continue

            if use_heading_feedback and yaw_ref is None and yaw_now is not None:
                yaw_ref = yaw_now

            w_cmd = 0.0
            if use_heading_feedback and (yaw_ref is not None) and (yaw_now is not None):
                err = ((yaw_ref - yaw_now + math.pi)%(2*math.pi))-math.pi
                w_cmd = self.pid_heading.step(err, dt=0.01)
                w_cmd = max(-max_w, min(max_w, w_cmd))

            # 우측 누적 이동량 판정
            done = False
            if source=="odom" and (pos is not None) and (start is not None) and (yaw_ref is not None):
                dx = pos[0]-start[0]; dy = pos[1]-start[1]
                lateral_left = (-math.sin(yaw_ref))*dx + (math.cos(yaw_ref))*dy
                lateral_right = -lateral_left
                if lateral_right >= target_dist: done = True

            twist.linear.x = 0.0; twist.linear.y = v_y; twist.angular.z = w_cmd
            self.mecanum_pub.publish(twist)

            if done: break
            rclpy.spin_once(self, timeout_sec=0.01); time.sleep(0.01)

        self._hard_stop()
        try: self.destroy_subscription(sub)
        except: pass

    # ==================== 신호등 인식(정지 중) ====================
    def perceive_trafficlight_at_stop(self, frames=3, timeout=1.0):
        self._hard_stop()
        t0 = time.time(); green_cnt = red_cnt = 0
        while (time.time()-t0) < timeout:
            objs = list(self.objects_info) if self.objects_info else []
            for o in objs:
                if o.class_name == 'green': green_cnt += 1
                elif o.class_name == 'red': red_cnt += 1
            if green_cnt >= frames or red_cnt >= frames: break
            time.sleep(0.05)
        if red_cnt >= frames: return "red"
        if green_cnt >= frames: return "green"
        return "unknown"

    # ==================== 랜드마크 기반 위치 보정 ====================
    def stop_and_localize(self, key, max_rotate_rate=0.8, max_drive=0.25, timeout=6.0):
        """
        1) 정지
        2) 기대 관측(CHECKPOINT_EXPECTED[key]) 불러옴
        3) RGB+Depth로 랜드마크 '하나' 측정(bearing, distance)
        4) PID로 각도→거리 순서로 보정
        """
        expect = CHECKPOINT_EXPECTED.get(key, None)
        if expect is None:
            self.get_logger().warn(f"[LOCALIZE] No expectation for key={key}; skip")
            return

        self._hard_stop()
        self.pid_heading.reset(); self.pid_dist.reset()
        t_dead = time.time() + timeout

        # 3-a) 감지 루틴
        def sense_once():
            with self.img_lock:
                img = None if self.latest_image is None else self.latest_image.copy()
            with self.depth_lock:
                depth = None if self.latest_depth is None else self.latest_depth.copy()
            if img is None or self.objects_info is None:
                return None
            img_h, img_w = img.shape[:2]
            # 원하는 클래스 우선순위 순회
            for cls in expect["classes"]:
                for o in self.objects_info:
                    if o.class_name != cls: continue
                    bbox = o.box  # [x1,y1,x2,y2]
                    bearing = self._bbox_center_bearing_deg(bbox, img_w)
                    dist = self._depth_at_bbox_center(bbox, depth)
                    if dist is None: continue
                    return {"bearing_deg": bearing, "distance_m": dist, "class": cls}
            return None

        meas = None
        while meas is None and time.time() < t_dead:
            meas = sense_once()
            time.sleep(0.05)
        if meas is None:
            self.get_logger().warn(f"[LOCALIZE] landmark not found for {key}")
            return

        # 4) 보정: (a) 각도 → (b) 거리
        # --- (a) 각도 ---
        target_bearing = expect["bearing_deg"]
        bearing_tol = expect.get("bearing_tol_deg", 6.0)
        while time.time() < t_dead:
            err_deg = target_bearing - meas["bearing_deg"]
            if abs(err_deg) <= bearing_tol: break
            w = self.pid_heading.step(math.radians(err_deg), dt=0.02)
            w = max(-max_rotate_rate, min(max_rotate_rate, w))
            twist = Twist(); twist.angular.z = w
            self.mecanum_pub.publish(twist)
            time.sleep(0.02)
            meas = sense_once() or meas  # 업데이트 시도
        self._hard_stop()

        # --- (b) 거리 ---
        target_dist = expect["distance_m"]
        dist_tol = expect.get("dist_tol_m", 0.12)
        while time.time() < t_dead:
            err = target_dist - meas["distance_m"]
            if abs(err) <= dist_tol: break
            v = self.pid_dist.step(err, dt=0.02)
            v = max(-max_drive, min(max_drive, v))
            twist = Twist(); twist.linear.x = v
            self.mecanum_pub.publish(twist)
            time.sleep(0.02)
            meas = sense_once() or meas
        self._hard_stop()
        self.get_logger().info(f"[LOCALIZE] {key} ok (cls={meas['class']})")

    def stop_and_align_to_lane(self, target_x_px=None, tol_px=10, timeout=3.0):
        """
        정지 상태에서만 수행. 차선 중심 cx(px)를 target_x_px로 보정.
        1) 각도 먼저 맞추고(픽셀 오차→방위각), 2) (메카넘) 측면 이동으로 잔여 오차 제거.
        Ackermann이면 2) 대신 아주 작은 전/후-조향 펄스로 근사 가능.
        """
        self._hard_stop()
        t_dead = time.time() + timeout
        self.pid_lane_yaw.reset(); self.pid_lane_lat.reset()

        # 한 장 촬영해 목표 픽셀 기본값(center) 설정
        with self.img_lock:
            img0 = None if self.latest_image is None else self.latest_image.copy()
        if img0 is None:
            self.get_logger().warn("[ALIGN] no image")
            return
        cx0_w = self._lane_center_x(img0)
        if cx0_w is None:
            self.get_logger().warn("[ALIGN] lane not visible")
            return
        _, w = cx0_w
        if target_x_px is None:
            target_x_px = w // 2  # 기본: 화면 중앙에 차선 중심을 오도록

        # 도(deg)/rad 변환 상수
        hfov = math.radians(self.hfov_deg)

        # --- (1) 회전으로 방향 먼저 정렬 ---
        while time.time() < t_dead:
            with self.img_lock:
                img = None if self.latest_image is None else self.latest_image.copy()
            got = self._lane_center_x(img) if img is not None else None
            if got is None:
                break
            cx, w = got
            err_px = (target_x_px - cx)

            if abs(err_px) <= max(tol_px, 6):
                break

            # 픽셀 → 방위각(rad): 이미지 반폭 대비 비율 × (HFOV/2)
            bearing = (err_px / (w/2.0)) * (hfov/2.0)
            w_cmd = self.pid_lane_yaw.step(bearing, dt=0.02)
            w_cmd = float(np.clip(w_cmd, -0.9, 0.9))

            twist = Twist()
            twist.angular.z = w_cmd
            self.mecanum_pub.publish(twist)
            time.sleep(0.02)

        self._hard_stop()

        # --- (2) 측면 이동으로 잔여 오차 제거 (메카넘 전용) ---
        if self.machine_type == 'MentorPi_Mecanum':
            # 각도는 유지(heading PID로 angular.z 작은 피드백), Δθ를 횡속도 입력으로 사용
            while time.time() < t_dead:
                meas = sense_once() or meas
                err_rad = math.radians(expect["bearing_deg"] - meas["bearing_deg"])
                if abs(math.degrees(err_rad)) < 1.0:  # 1도 이내면 종료
                    break
                # 좌우 스케일(너무 크면 출렁이니 작은 게 안전)
                v_y = np.clip(0.6 * err_rad, -0.20, 0.20)  # 라디안 비례, 튠 포인트: 0.4~0.8
                # 헤딩 유지 소량 피드백(필요시): w_cmd = self.pid_heading.step(0.0 - 0.0, dt=0.02)
                twist = Twist()
                twist.linear.y = v_y   # Δθ>0(왼쪽에 있음) → +y로 이동(하드웨어 부호 맞춰 조정)
                self.mecanum_pub.publish(twist)
                time.sleep(0.02)
            self._hard_stop()

        else:
            # Ackermann: 아주 작은 전/후 펄스 + 조향(= angular.z)로 근사
            # 필요시 여기 보간 로직을 추가해도 됨.
            pass

        self.get_logger().info("[ALIGN] lane ok")

    # ==================== 플랜 실행기 ====================
    def run_course(self):
        while not self.start and rclpy.ok():
            time.sleep(0.05)

        # 프레임 준비
        t0 = time.time()
        while (self.latest_image is None) and (time.time()-t0 < 2.0) and rclpy.ok():
            time.sleep(0.02)
        if self.latest_image is None:
            self.get_logger().warn("[GATE] No camera frame; delaying plan start by 1s")
            time.sleep(1.0)

        # 오돔 준비(선택)
        got_odom = [False]
        def _odom_once(msg): got_odom[0] = True
        sub_tmp = self.create_subscription(Odometry, "/odom", _odom_once, self.sensor_qos)
        t0 = time.time()
        while (not got_odom[0]) and (time.time()-t0 < 2.0) and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
        try: self.destroy_subscription(sub_tmp)
        except: pass

        self.in_action = True
        while rclpy.ok() and self.step_idx < len(self.plan):
            step = self.plan[self.step_idx]
            kind = step[0]

            if kind == "DRIVE":
                _, dist, speed = step
                self.get_logger().info(f"[PLAN] DRIVE {dist}m @ {speed}m/s")
                self.drive_distance_straight(dist, speed, True, 0.25, 15.0, "odom")
                self.step_idx += 1

            elif kind == "STOP_PERCEIVE_LIGHT":
                opts = step[1] if len(step) > 1 else {"frames":5,"timeout":1.0}
                frames = opts.get("frames", 5); timeout = opts.get("timeout", 1.0)
                self.get_logger().info("[PLAN] STOP & PERCEIVE (traffic light)")
                result = self.perceive_trafficlight_at_stop(frames, timeout)
                self.get_logger().info(f"[PERCEIVE] {result}")
                # 빨간불이면 최대 5초 대기
                wait_deadline = time.time() + 5.0
                while result == "red" and time.time() < wait_deadline:
                    time.sleep(0.2)
                    result = self.perceive_trafficlight_at_stop(frames=3, timeout=1.0)
                    self.get_logger().info(f"[PERCEIVE-WAIT] {result}")
                self.step_idx += 1

            elif kind == "STOP_LOCALIZE":
                _, key = step
                self.get_logger().info(f"[PLAN] STOP & LOCALIZE @{key}")
                self.stop_and_localize(key)
                self.step_idx += 1

            elif kind == "TURN_RIGHT":
                _, deg = step
                self.get_logger().info(f"[PLAN] TURN_RIGHT {deg}deg")
                if self.machine_type == 'MentorPi_Mecanum':
                    self.rotate_right_90_feedback(source="odom")
                else:
                    self.get_logger().warn("Ackermann: in-place turn not supported; open-loop fallback.")
                    self.rotate_right_90_in_place(0.8)
                self._hard_stop(); time.sleep(0.1)
                self.step_idx += 1

            elif kind == "PARK":
                _, dist, speed = step
                self.get_logger().info(f"[PLAN] PARK right {dist}m @ {speed}m/s")
                if self.machine_type == 'MentorPi_Mecanum':
                    self.strafe_right_distance(dist, speed, True, 8.0, "odom")
                else:
                    self.get_logger().warn("Ackermann: lateral PARK not supported.")
                self._hard_stop(); time.sleep(0.1)
                self.step_idx += 1
            
            elif kind == "STOP_ALIGN_LANE":
                self.get_logger().info("[PLAN] STOP & ALIGN to lane")
                # 필요하면 포인트별 목표 픽셀 지정도 가능: stop_and_align_to_lane(target_x_px=...)
                self.stop_and_align_to_lane(target_x_px=None, tol_px=10, timeout=2.5)
                self.step_idx += 1

            else:
                self.get_logger().warn(f"[PLAN] Unknown step: {step}")
                self.step_idx += 1

        self.in_action = False
        self.get_logger().info("[PLAN] Completed")

    # ------------------------- 표시 루프 -------------------------
    def main(self):
        while self.is_running:
            time_start = time.time()
            if self.rotating:
                self._hard_stop(); time.sleep(0.01); continue

            with self.img_lock:
                frame = None if self.latest_image is None else self.latest_image.copy()
            if frame is None:
                time.sleep(0.01); continue

            result_image = frame.copy()

            # (표시용) 객체 박스
            if self.display and self.objects_info:
                for i in self.objects_info:
                    box = i.box; class_name = i.class_name; cls_conf = i.score
                    cls_id = self.classes.index(class_name) if class_name in self.classes else 0
                    color = colors(cls_id, True)
                    plot_one_box(box, result_image, color=color,
                                 label="{}:{:.2f}".format(class_name, cls_conf))

            # FPS & 퍼블리시
            bgr_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            if self.display:
                self.fps.update(); bgr_image = self.fps.show_fps(bgr_image)
            self.result_publisher.publish(self.bridge.cv2_to_imgmsg(bgr_image, "bgr8"))

            # 루프 주기
            time_d = 0.03 - (time.time() - time_start)
            if time_d > 0: time.sleep(time_d)

        self._hard_stop()

# ------------------------- 엔트리 -------------------------
def main():
    rclpy.init()
    node = SelfDrivingNode('self_driving')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
