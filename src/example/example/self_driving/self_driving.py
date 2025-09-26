#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/03/28
# @author:aiden
# autonomous driving

import os
import cv2
import math
import time
# import queue
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

class SelfDrivingNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.name = name
        self.is_running = True
        self.pid = pid.PID(0.4, 0.0, 0.05)
        self.param_init()

        self.latest_image = None
        self.img_lock = threading.RLock()

        self.fps = fps.FPS()
        # self.image_queue = queue.Queue(maxsize=2)
        self.classes = ['go', 'right', 'park', 'red', 'green', 'crosswalk']
        self.display = True
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.colors = common.Colors()
        self.machine_type = os.environ.get('MACHINE_TYPE')
        self.lane_detect = lane_detect.LaneDetector("yellow")

        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.servo_state_pub = self.create_publisher(SetPWMServoState, 'ros_robot_controller/pwm_servo/set_state', 1)
        self.result_publisher = self.create_publisher(Image, '~/image_result', 1)

        self.create_service(Trigger, '~/enter', self.enter_srv_callback)   # enter the game
        self.create_service(Trigger, '~/exit', self.exit_srv_callback)     # exit the game
        self.create_service(SetBool, '~/set_running', self.set_running_srv_callback)

        timer_cb_group = ReentrantCallbackGroup()
        self.client = self.create_client(Trigger, '/yolov5_ros2/init_finish')
        self.client.wait_for_service()
        self.start_yolov5_client = self.create_client(Trigger, '/yolov5/start', callback_group=timer_cb_group)
        self.start_yolov5_client.wait_for_service()
        self.stop_yolov5_client = self.create_client(Trigger, '/yolov5/stop', callback_group=timer_cb_group)
        self.stop_yolov5_client.wait_for_service()

        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)

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
        # 액션: ("DRIVE", 거리[m], 속도[m/s]), ("STOP&PERCEIVE",), ("TURN_RIGHT", 각도[deg])
        # 코스에 맞게 아래 값들만 바꿔서 사용하세요.
        self.plan = [
            ("DRIVE", 2.6, 0.3),
            ("TURN_RIGHT", 90),
            ("DRIVE", 2.6, 0.3),
            ("TURN_RIGHT", 90),
            ("DRIVE", 2.6, 0.3),
            ("TURN_RIGHT", 90),
            ("DRIVE", 2.6, 0.3),
        ]
        self.step_idx = 0
        self.in_action = False

    def init_process(self):
        self.timer.cancel()

        self.mecanum_pub.publish(Twist())
        # 객체 인식 노드 시작 (정지 때만 실제 결과를 사용)
        self.send_request(self.start_yolov5_client, Trigger.Request())
        time.sleep(0.5)

        # 자동 시작
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

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    # ------------------------- 서비스 콜백 -------------------------
    def enter_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "self driving enter")
        with self.lock:
            self.start = False
            # 구독 핸들을 멤버에 저장(해제 위해)
            self.image_sub = self.create_subscription(
                Image, '/ascamera/camera_publisher/rgb0/image', self.image_callback, 1
            )
            self.object_sub = self.create_subscription(
                ObjectsInfo, '/yolov5_ros2/object_detect', self.get_object_callback, 1
            )
            self.mecanum_pub.publish(Twist())
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
            self.mecanum_pub.publish(Twist())
        self.param_init()
        response.success = True
        response.message = "exit"
        return response

    def set_running_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "set_running")
        with self.lock:
            self.start = request.data
            if not self.start:
                self.mecanum_pub.publish(Twist())
        response.success = True
        response.message = "set_running"
        return response

    def shutdown(self, signum, frame):
        self.is_running = False

    # ------------------------- 콜백/유틸 -------------------------
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
    def rotate_right_90_in_place(self, yaw_rate=0.8):
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

        self.mecanum_pub.publish(Twist())
        self.rotating = False

    def rotate_right_90_feedback(self, source="odom", max_rate=1.0, min_rate=0.2, timeout=5.0):
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
            sub = self.create_subscription(Odometry, "/odom", odom_cb, 1)
        else:
            sub = self.create_subscription(Imu, "/imu", imu_cb, 1)

        t0 = time.time()
        while yaw_now is None and (time.time() - t0) < 1.0:
            rclpy.spin_once(self, timeout_sec=0.05)

        if yaw_now is None:
            self.get_logger().warn("Yaw not available; fallback to open-loop 90°.")
            try: self.destroy_subscription(sub)
            except Exception: pass
            self.rotate_right_90_in_place(0.8)
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

            delta = ((yaw_now - yaw_start + math.pi) % (2*math.pi)) - math.pi
            err = target_delta - delta

            w_cmd = Kp * err
            w_cmd = max(-max_rate, min(max_rate, w_cmd))
            if abs(w_cmd) < min_rate:
                w_cmd = -min_rate if err < 0 else min_rate

            twist.angular.z = w_cmd
            self.mecanum_pub.publish(twist)

            if abs(err) < math.radians(2.0):
                break

            time.sleep(0.01)

        self.mecanum_pub.publish(Twist())
        try: self.destroy_subscription(sub)
        except Exception: pass
        self.rotating = False

    # ------------------------- 주행 프리미티브 -------------------------
    def drive_distance_with_lane(self, target_dist=1.5, speed=0.25,
                                 lane_setpoint=None, max_w=0.25, timeout=15.0):
        """
        /odom 기반으로 target_dist(m) 전진.
        이동 중 yaw는 차선 PID(lane_x)로만 보정 → 차선 이탈 방지.
        주행 중 객체 인식 결과는 사용하지 않음(흔들림 회피).
        """
        start = None
        pos = None
        LOST_MAX = 8
        lost_cnt = 0

        def odom_cb(msg):
            nonlocal pos, start
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            if start is None:
                start = (x, y)
            pos = (x, y)

        sub = self.create_subscription(Odometry, "/odom", odom_cb, 5)

        twist = Twist()
        v = speed
        t0 = time.time()
        lane_setpoint = lane_setpoint if lane_setpoint is not None else 130

        base_kp, base_kd = 0.4, 0.05

        while rclpy.ok() and (time.time() - t0) < timeout:
            # 거리 종료
            if pos is not None and start is not None:
                dist = math.hypot(pos[0]-start[0], pos[1]-start[1])
                if dist >= target_dist:
                    break

            # 회전/정지 중이면 대기
            if self.rotating or self.stop:
                self.mecanum_pub.publish(Twist())
                time.sleep(0.02)
                continue

            # --- 최신 프레임 가져오기 ---
            img = None
            with self.img_lock:
                if self.latest_image is not None:
                    img = self.latest_image.copy()

            if img is not None:
                binary = self.lane_detect.get_binary(img)
                _, lane_angle, lane_x = self.lane_detect(binary, img.copy())

                if lane_x >= 0:
                    lost_cnt = 0
                    self.pid.SetPoint = lane_setpoint
                    self.pid.Kp = base_kp / (1.0 + v)    # 속도↑ → Kp↓
                    self.pid.Kd = base_kd * (1.0 + v)    # 속도↑ → Kd↑
                    self.pid.update(lane_x)

                    w = common.set_range(self.pid.output, -max_w, max_w)
                    twist.linear.x = v
                    twist.angular.z = w
                    self.mecanum_pub.publish(twist)
                else:
                    lost_cnt += 1
            else:
                lost_cnt += 1

            if lost_cnt > LOST_MAX:
                # 안전 정지
                self.mecanum_pub.publish(Twist())
                break

            time.sleep(0.02)

        self.mecanum_pub.publish(Twist())
        try: self.destroy_subscription(sub)
        except Exception: pass

    def perceive_trafficlight_at_stop(self, frames=5, timeout=3.0):
        """
        정지 상태에서만 인식. frames개 이상 일관되게 보이면 채택.
        반환: 'red' | 'green' | 'crosswalk' | 'unknown'  (모두 소문자)
        """
        self.mecanum_pub.publish(Twist())
        t0 = time.time()
        green_cnt = red_cnt = crosswalk_cnt = 0

        while (time.time()-t0) < timeout:
            objs = list(self.objects_info) if self.objects_info else []
            for o in objs:
                if o.class_name == 'green':
                    green_cnt += 1
                elif o.class_name == 'red':
                    red_cnt += 1
                elif o.class_name == 'crosswalk':
                    crosswalk_cnt += 1
            if green_cnt >= frames or red_cnt >= frames or crosswalk_cnt >= frames:
                break
            time.sleep(0.05)

        if red_cnt >= frames:
            return "red"
        if green_cnt >= frames:
            return "green"
        if crosswalk_cnt >= frames:
            return "crosswalk"
        return "unknown"

    # ------------------------- 플랜 실행기 -------------------------
    def run_course(self):
        """
        하드코딩된 self.plan을 순차 실행:
        DRIVE -> STOP&PERCEIVE -> TURN_RIGHT ...
        """
        # 주행 허가가 떨어질 때까지 대기
        while not self.start and rclpy.ok():
            time.sleep(0.05)

        self.in_action = True
        while rclpy.ok() and self.step_idx < len(self.plan):
            step = self.plan[self.step_idx]
            kind = step[0]

            if kind == "DRIVE":
                _, dist, speed = step
                self.get_logger().info(f"[PLAN] DRIVE {dist}m @ {speed}m/s")
                self.drive_distance_with_lane(target_dist=dist, speed=speed)
                self.step_idx += 1

            elif kind == "STOP&PERCEIVE":
                self.get_logger().info("[PLAN] STOP & PERCEIVE")
                result = self.perceive_trafficlight_at_stop(frames=5, timeout=3.0)
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
                # 메카넘이면 제자리 피드백 회전. 아커만이면 별도 로직 필요.
                if self.machine_type == 'MentorPi_Mecanum':
                    self.rotate_right_90_feedback(source="odom")
                else:
                    # Ackermann 차량의 경우 in-place 회전이 불가. 여기선 간단히 제자리 회전 시도 대신
                    # 기존 각속도 기반 회전(전/후진 조합) 로직을 작성해야 함.
                    self.rotate_right_90_in_place(0.8)
                self.mecanum_pub.publish(Twist())
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
                self.mecanum_pub.publish(Twist())
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


        self.mecanum_pub.publish(Twist())
        rclpy.shutdown()

# ------------------------- 엔트리 -------------------------
def main():
    node = SelfDrivingNode('self_driving')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()

if __name__ == "__main__":
    main()
