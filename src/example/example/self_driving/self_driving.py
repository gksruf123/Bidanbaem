#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/03/28
# @author:aiden
# autonomous driving

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

class SelfDrivingNode(Node):
    def __init__(self, name):
        # rclpy.init()  # <-- main()에서만 호출
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.name = name
        self.is_running = True
        self.pid = pid.PID(0.4, 0.0, 0.05)
        self.param_init()

        self.latest_image = None
        self.img_lock = threading.RLock()

        self.fps = fps.FPS()
        self.classes = ['go', 'right', 'park', 'red', 'green', 'crosswalk']
        self.display = True
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.colors = common.Colors()
        self.machine_type = os.environ.get('MACHINE_TYPE')
        self.lane_detect = lane_detect.LaneDetector("yellow")

        # QoS: 센서 스트림은 BEST_EFFORT, depth=1 권장
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

        # Service clients (with timeout wait)
        timer_cb_group = ReentrantCallbackGroup()
        self.client = self.create_client(Trigger, '/yolov5_ros2/init_finish')
        self.start_yolov5_client = self.create_client(Trigger, '/yolov5/start', callback_group=timer_cb_group)
        self.stop_yolov5_client  = self.create_client(Trigger, '/yolov5/stop',  callback_group=timer_cb_group)

        for cli, name_ in [(self.client, '/yolov5_ros2/init_finish'),
                           (self.start_yolov5_client, '/yolov5/start'),
                           (self.stop_yolov5_client, '/yolov5/stop')]:
            if not self._wait_service(cli, 5.0):
                self.get_logger().warn(f"Service {name_} not available within timeout; continuing anyway.")

        # 타이머는 0.0 금지 → 짧게 한 번 돌릴 용도로 0.05s
        self.timer = self.create_timer(0.05, self.init_process, callback_group=timer_cb_group)

    # ------------------------- 초기화/파라미터 -------------------------
    def param_init(self):
        self.start = False
        self.enter = False
        self.stop = False                 # 신호등/정지 처리 상태
        self.rotating = False             # 제자리 회전 중인지
        self.objects_info = []
        self.object_sub = None
        self.image_sub = None

        # 속도 설정
        self.normal_speed = 0.3
        self.slow_down_speed = 0.1

        # -------- 하드코딩 주행 플랜 --------
        self.plan = [
            ("DRIVE", 1.0, 0.8),
            ("STOP&PERCEIVE", {"frames": 7, "timeout": 1.0}),
            ("DRIVE", 1.6, 0.8),
            ("TURN_RIGHT", 90),
            ("DRIVE", 0.6, 0.8),
            ("STOP&PERCEIVE", {"frames": 7, "timeout": 1.0}),
            ("DRIVE", 2.0, 0.8),
            ("TURN_RIGHT", 90),
            ("DRIVE", 1.6, 0.8),
            ("STOP&PERCEIVE", {"frames": 7, "timeout": 1.0}),
            ("DRIVE", 1.0, 0.8),
            ("TURN_RIGHT", 90),
            ("DRIVE", 1.0, 0.8),
            ("STOP&PERCEIVE", {"frames": 7, "timeout": 1.0}),
            ("DRIVE", 0.3, 0.8),
            ("TURN_RIGHT", 90),
            ("DRIVE", 1.9, 0.8),
             ("PARK", 0.40, 0.80),   # (kind, right_distance_m, speed_mps)
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
        # 타이머는 1회성
        try:
            self.timer.cancel()
        except Exception:
            pass

        # 초기 정지
        self._hard_stop()

        # 객체 인식 노드 시작 (정지 때만 실제 결과를 사용)
        res = self.send_request(self.start_yolov5_client, Trigger.Request(), timeout_sec=5.0)
        if not res or not getattr(res, 'success', False):
            self.get_logger().warn("Failed to start YOLOv5 service")

        time.sleep(0.2)

        # 자동 시작: 서버 콜백 직접 호출(내부 상태 세팅 목적)
        self.display = True
        self.enter_srv_callback(Trigger.Request(), Trigger.Response())
        request = SetBool.Request()
        request.data = True
        self.set_running_srv_callback(request, SetBool.Response())

        # 쓰레드 시작: 표시 루프 + 플랜 실행기
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
            # 구독 핸들을 멤버에 저장(해제 위해) + QoS 적용
            self.image_sub = self.create_subscription(
                Image, '/ascamera/camera_publisher/rgb0/image', self.image_callback, self.sensor_qos
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
                    self.destroy_subscription(self.image_sub)
                    self.image_sub = None
                if self.object_sub is not None:
                    self.destroy_subscription(self.object_sub)
                    self.object_sub = None
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
        # 최신 프레임 1장만 보관
        with self.img_lock:
            self.latest_image = rgb_image

    def get_object_callback(self, msg):
        # 객체 인식 결과 최신 상태를 유지(정지 시에만 활용)
        self.objects_info = msg.objects

    # Quaternion -> yaw(rad)
    def _yaw_from_quat(self, q):
        siny_cosp = 2.0 * (q.w*q.z + q.x*q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    # ------------------------- 제자리 회전 -------------------------
    def rotate_right_90_in_place(self, yaw_rate=1.3):
        """오픈 루프: 제자리에서 오른쪽 90°(간단/캘리브 필요)"""
        self.rotating = True
        twist = Twist()
        twist.linear.x = twist.linear.y = twist.linear.z = 0.0
        twist.angular.x = twist.angular.y = 0.0
        twist.angular.z = -abs(yaw_rate)  # 오른쪽은 음수

        angle = math.pi / 2
        duration = angle / abs(yaw_rate)

        t0 = time.time()
        while time.time() - t0 < duration and self.is_running:
            self.mecanum_pub.publish(twist)
            time.sleep(0.02)

        self._hard_stop()
        self.rotating = False

    def rotate_right_90_feedback(self, source="odom", max_rate=1.3, min_rate=0.2, timeout=5.0):
        """피드백 루프: /odom 또는 /imu yaw로 정확히 -90° 도달까지 회전"""
        yaw_now = None
        self.rotating = True

        def odom_cb(msg):
            nonlocal yaw_now
            yaw_now = self._yaw_from_quat(msg.pose.pose.orientation)

        def imu_cb(msg):
            nonlocal yaw_now
            yaw_now = self._yaw_from_quat(msg.orientation)

        if source == "odom":
            sub = self.create_subscription(Odometry, "/odom", odom_cb, self.sensor_qos)
        else:
            sub = self.create_subscription(Imu, "/imu", imu_cb, 1)

        t0 = time.time()
        while yaw_now is None and (time.time() - t0) < 1.0:
            rclpy.spin_once(self, timeout_sec=0.05)

        if yaw_now is None:
            self.get_logger().warn("Yaw not available; fallback to open-loop 90°.")
            try: self.destroy_subscription(sub)
            except Exception: pass
            self.rotate_right_90_in_place(1.3)
            self.rotating = False
            return

        yaw_start = yaw_now
        target_delta = -math.pi / 2
        Kp = 1.2

        twist = Twist()
        twist.linear.x = twist.linear.y = 0.0

        t0 = time.time()
        while (time.time() - t0) < timeout and self.is_running:
            rclpy.spin_once(self, timeout_sec=0.01)
            if yaw_now is None:
                continue

            # -pi..pi wrap
            delta = ((yaw_now - yaw_start + math.pi) % (2*math.pi)) - math.pi
            err = target_delta - delta

            w_cmd = Kp * err
            w_cmd = max(-max_rate, min(max_rate, w_cmd))
            if abs(w_cmd) < min_rate:
                w_cmd = -min_rate if err < 0 else min_rate

            twist.angular.z = w_cmd
            self.mecanum_pub.publish(twist)

            if abs(err) < math.radians(1.5):
                break

            time.sleep(0.01)

        self._hard_stop()
        try: self.destroy_subscription(sub)
        except Exception: pass
        self.rotating = False

    # ------------------------- 주행 프리미티브 -------------------------
    def drive_distance_straight(self, target_dist=1.5, speed=0.25,
                                use_heading_feedback=True, max_w=0.35,
                                timeout=20.0, source="odom"):
        """
        차선 무시, '딱 직진'.
        - use_heading_feedback=True: 출발 yaw를 유지하며 직진(권장)
        - use_heading_feedback=False: 각속도 0 고정(오픈루프)
        """
        start = None
        pos = None
        yaw_now = None
        yaw_ref = None

        def odom_cb(msg):
            nonlocal pos, start, yaw_now
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            if start is None:
                start = (x, y)
            pos = (x, y)
            yaw_now = self._yaw_from_quat(msg.pose.pose.orientation)

        def imu_cb(msg):
            nonlocal yaw_now
            yaw_now = self._yaw_from_quat(msg.orientation)

        # 구독
        if source == "odom":
            sub = self.create_subscription(Odometry, "/odom", odom_cb, self.sensor_qos)
        else:
            sub = self.create_subscription(Imu, "/imu", imu_cb, 1)

        twist = Twist()
        v = speed
        t0 = time.time()

        # 헤딩 유지 제어 게인
        Kp_heading = 1.3  # 필요시 0.6~1.5 사이에서 튠
        while rclpy.ok() and (time.time() - t0) < timeout:
            # 거리 종료
            if source == "odom" and pos is not None and start is not None:
                dist = math.hypot(pos[0] - start[0], pos[1] - start[1])
                if dist >= target_dist:
                    break

            # 회전/정지 중이면 대기
            if self.rotating or self.stop:
                self._hard_stop()
                time.sleep(0.01)
                continue

            # 각속도 결정
            w_cmd = 0.0
            if use_heading_feedback:
                # 기준각 캡처
                if yaw_ref is None and yaw_now is not None:
                    yaw_ref = yaw_now
                if yaw_ref is not None and yaw_now is not None:
                    # err = yaw_ref - yaw_now (wrap to -pi..pi)
                    err = ((yaw_ref - yaw_now + math.pi) % (2 * math.pi)) - math.pi
                    w_cmd = max(-max_w, min(max_w, Kp_heading * err))
                else:
                    # 아직 yaw 못 받았으면 일단 각속도 0으로 직진
                    w_cmd = 0.0
            else:
                # 완전 오픈루프 직진
                w_cmd = 0.0

            twist.linear.x = v
            twist.angular.z = w_cmd
            self.mecanum_pub.publish(twist)

            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.01)

        self._hard_stop()
        try:
            self.destroy_subscription(sub)
        except Exception:
            pass

    # parking action
    def strafe_right_distance(self, target_dist=0.4, speed=0.2,
                            use_heading_feedback=True, timeout=8.0, source="odom"):
        """
        메카넘 우측 평행 이동: base_link 기준 +y가 좌측이므로 우측은 linear.y 음수.
        오돔을 이용해 '우측' 성분 이동거리를 누적해 target_dist에 도달하면 정지.
        """
        start = None
        pos = None
        yaw_now = None
        yaw_ref = None

        def odom_cb(msg):
            nonlocal pos, start, yaw_now
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            if start is None:
                start = (x, y)
            pos = (x, y)
            yaw_now = self._yaw_from_quat(msg.pose.pose.orientation)

        def imu_cb(msg):
            nonlocal yaw_now
            yaw_now = self._yaw_from_quat(msg.orientation)

        # 구독
        if source == "odom":
            sub = self.create_subscription(Odometry, "/odom", odom_cb, self.sensor_qos)
        else:
            sub = self.create_subscription(Imu, "/imu", imu_cb, 1)

        twist = Twist()
        v_y = -abs(speed)  # 우측 이동은 음수
        t0 = time.time()

        # 헤딩 유지 게인(직진 함수와 동일 톤)
        Kp_heading = 1.2
        max_w = 0.35

        while rclpy.ok() and (time.time() - t0) < timeout:
            if self.rotating or self.stop:
                self._hard_stop()
                time.sleep(0.01)
                continue

            # 기준 yaw 캡처
            if use_heading_feedback and (yaw_ref is None) and (yaw_now is not None):
                yaw_ref = yaw_now

            # 헤딩 에러 보정(회전 억제)
            w_cmd = 0.0
            if use_heading_feedback and (yaw_ref is not None) and (yaw_now is not None):
                err = ((yaw_ref - yaw_now + math.pi) % (2 * math.pi)) - math.pi
                w_cmd = max(-max_w, min(max_w, Kp_heading * err))

            # 목표 도달 판정(오돔에서 우측 성분만 투영)
            done = False
            if source == "odom" and (pos is not None) and (start is not None) and (yaw_ref is not None):
                dx = pos[0] - start[0]
                dy = pos[1] - start[1]
                # base_link 기준: 좌측(+y) 단위벡터 = [-sin(yaw_ref), cos(yaw_ref)]
                lateral_left = (-math.sin(yaw_ref)) * dx + (math.cos(yaw_ref)) * dy
                lateral_right = -lateral_left  # 우측을 양수로 정의
                if lateral_right >= target_dist:
                    done = True

            twist.linear.x = 0.0
            twist.linear.y = v_y
            twist.angular.z = w_cmd
            self.mecanum_pub.publish(twist)

            if done:
                break

            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.01)

        self._hard_stop()
        try:
            self.destroy_subscription(sub)
        except Exception:
            pass


    def perceive_trafficlight_at_stop(self, frames=3, timeout=1.0):
        """
        정지 상태에서만 인식. frames개 이상 일관되게 보이면 채택.
        반환: 'red' | 'green' | 'crosswalk' | 'unknown'
        """
        self._hard_stop()
        t0 = time.time()
        green_cnt = red_cnt = 0
        # crosswalk_cnt = 0

        while (time.time()-t0) < timeout:
            objs = list(self.objects_info) if self.objects_info else []
            for o in objs:
                if o.class_name == 'green':
                    green_cnt += 1
                elif o.class_name == 'red':
                    red_cnt += 1
                # elif o.class_name == 'crosswalk':
                #     crosswalk_cnt += 1
            if green_cnt >= frames or red_cnt >= frames: # 필요하면 or crosswalk_cnt 여기에 추가
                break
            time.sleep(0.05)

        if red_cnt >= frames:
            return "red"
        if green_cnt >= frames:
            return "green"
        # if crosswalk_cnt >= frames:
        #     return "crosswalk"
        return "unknown"

    # ------------------------- 플랜 실행기 -------------------------
    def run_course(self):
        """
        하드코딩된 self.plan을 순차 실행:
        DRIVE -> STOP&PERCEIVE -> TURN_RIGHT ...
        """
        # 1) start 신호 대기
        while not self.start and rclpy.ok():
            time.sleep(0.05)

        # 2) 카메라 프레임 준비 대기 (최대 2초)
        t0 = time.time()
        while (self.latest_image is None) and (time.time() - t0 < 2.0) and rclpy.ok():
            time.sleep(0.02)
        if self.latest_image is None:
            self.get_logger().warn("[GATE] No camera frame; delaying plan start by 1s")
            time.sleep(1.0)

        # 3) (선택) 오돔 준비 대기 (최대 2초)
        got_odom = [False]
        def _odom_once(msg):
            got_odom[0] = True
        sub_tmp = self.create_subscription(Odometry, "/odom", _odom_once, self.sensor_qos)
        t0 = time.time()
        while (not got_odom[0]) and (time.time() - t0 < 2.0) and rclpy.ok():
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
                self.drive_distance_straight(target_dist=dist, speed=speed,
                             use_heading_feedback=True,  # False면 각속도 0 고정
                             max_w=0.25, timeout=15.0, source="odom")
                self.step_idx += 1

            elif kind == "STOP&PERCEIVE":
                opts = step[1] if len(step) > 1 else {}
                frames = opts.get("frames", 5)
                timeout = opts.get("timeout", 1.0)
                self.get_logger().info("[PLAN] STOP & PERCEIVE")
                result = self.perceive_trafficlight_at_stop(frames=frames, timeout=timeout)
                self.get_logger().info(f"[PERCEIVE] {result}")

                # 예시 정책: 빨간불이면 최대 5초 동안 초록 기다림
                wait_deadline = time.time() + 5.0
                while result == "red" and time.time() < wait_deadline:
                    time.sleep(0.2)
                    result = self.perceive_trafficlight_at_stop(frames=3, timeout=1.0)
                    self.get_logger().info(f"[PERCEIVE-WAIT] {result}")

                self.step_idx += 1

            elif kind == "TURN_RIGHT":
                _, deg = step
                self.get_logger().info(f"[PLAN] TURN_RIGHT {deg}deg")
                if self.machine_type == 'MentorPi_Mecanum':
                    self.rotate_right_90_feedback(source="odom")
                else:
                    self.get_logger().warn("Ackermann: in-place turn not supported; using open-loop fallback.")
                    self.rotate_right_90_in_place(0.8)
                self._hard_stop()
                time.sleep(0.1)
                self.step_idx += 1

            elif kind == "PARK":
                _, dist, speed = step
                self.get_logger().info(f"[PLAN] PARK right {dist}m @ {speed}m/s")
                # 메카넘만 지원 (Ackermann이면 경고 후 스킵)
                if self.machine_type == 'MentorPi_Mecanum':
                    self.strafe_right_distance(target_dist=dist, speed=speed,
                                            use_heading_feedback=True, timeout=8.0, source="odom")
                else:
                    self.get_logger().warn("Ackermann: lateral PARK not supported.")
                self._hard_stop()
                time.sleep(0.1)
                self.step_idx += 1

            else:
                self.get_logger().warn(f"[PLAN] Unknown step: {step}")
                self.step_idx += 1

        self.in_action = False
        self.get_logger().info("[PLAN] Completed")

    # ------------------------- 표시 루프(제어 없음) -------------------------
    def main(self):
        while self.is_running:
            time_start = time.time()

            # 회전 중에는 정지 유지
            if self.rotating:
                self._hard_stop()
                time.sleep(0.01)
                continue

            # 최신 프레임 읽기
            with self.img_lock:
                frame = None if self.latest_image is None else self.latest_image.copy()

            if frame is None:
                time.sleep(0.01)
                continue

            result_image = frame.copy()

            # (표시용) 객체 박스 오버레이
            if self.display and self.objects_info:
                for i in self.objects_info:
                    box = i.box
                    class_name = i.class_name
                    cls_conf = i.score
                    cls_id = self.classes.index(class_name) if class_name in self.classes else 0
                    color = colors(cls_id, True)
                    plot_one_box(
                        box,
                        result_image,
                        color=color,
                        label="{}:{:.2f}".format(class_name, cls_conf),
                    )

            # FPS 오버레이 & 퍼블리시
            bgr_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            if self.display:
                self.fps.update()
                bgr_image = self.fps.show_fps(bgr_image)
            self.result_publisher.publish(self.bridge.cv2_to_imgmsg(bgr_image, "bgr8"))

            # 루프 주기
            time_d = 0.03 - (time.time() - time_start)
            if time_d > 0:
                time.sleep(time_d)

        self._hard_stop()
        # rclpy.shutdown()  # <-- main()에서 처리

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
