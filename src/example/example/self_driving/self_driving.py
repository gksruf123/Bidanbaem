#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/03/28
# @author:aiden
# autonomous driving
import os
import time
import queue
import rclpy
import threading
import numpy as np
import sdk.pid as pid
import sdk.fps as fps
from rclpy.node import Node
import sdk.common as common
# from app.common import Heart
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from interfaces.msg import ObjectsInfo
from std_srvs.srv import SetBool, Trigger
from sdk.common import colors, plot_one_box
from example.self_driving import lane_detect
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from nav_msgs.msg import Odometry
import math

def quat_to_yaw(x: float, y:float, z:float, w:float) -> float:
    """Quaternion -> yaw (rad)."""
    siny_cosp = 2.0 * (w*z + x*y)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    return math.atan2(siny_cosp, cosy_cosp)

class SelfDrivingNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.name = name
        self.is_running = True
        self.pid = pid.PID(0.05, 0.0, 0.05)
        self.param_init()

        self.fps = fps.FPS()
        self.image_queue = queue.Queue(maxsize=1)
        self.classes = ['go', 'right', 'park', 'red', 'green', 'crosswalk']
        self.display = True
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.colors = common.Colors()
        self.machine_type = os.environ.get('MACHINE_TYPE')
        self.lane_detect = lane_detect.LaneDetector("yellow")

        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.result_publisher = self.create_publisher(Image, '~/image_result', 1)
        self.binary_publisher = self.create_publisher(Image, '~/image_binary', 1)

        self.create_service(Trigger, '~/enter', self.enter_srv_callback) # enter the game
        self.create_service(Trigger, '~/exit', self.exit_srv_callback) # exit the game
        self.create_service(SetBool, '~/set_running', self.set_running_srv_callback)
        timer_cb_group = ReentrantCallbackGroup()
        self.client = self.create_client(Trigger, '/yolov5_ros2/init_finish')
        self.client.wait_for_service()
        self.start_yolov5_client = self.create_client(Trigger, '/yolov5/start', callback_group=timer_cb_group)
        self.start_yolov5_client.wait_for_service()
        self.stop_yolov5_client = self.create_client(Trigger, '/yolov5/stop', callback_group=timer_cb_group)
        self.stop_yolov5_client.wait_for_service()

        self.timer = self.create_timer(0.0, self.init_process, callback_group=timer_cb_group)

        self._yolo_is_on = False
        self._yolo_last_toggle = 0.0
        self._yolo_min_interval = 0.2   # 연속 토글 최소 간격(초) - 파이프라인 흔들림 방지
        self._yolo_timer = None         # enable-for 타이머 핸들


    def yolo_start(self, delay_s: float = 0.0):
        """YOLO 추론 시작(서비스 호출) - 구독/카메라 연결은 유지됨."""
        def _do_start():
            now = time.time()
            if now - self._yolo_last_toggle < self._yolo_min_interval:
                return
            self._yolo_last_toggle = now
            if not self._yolo_is_on:
                self.send_request(self.start_yolov5_client, Trigger.Request())
                self._yolo_is_on = True
                self.get_logger().info("[self_driving] YOLO: START")
        if delay_s > 0:
            t = threading.Timer(delay_s, _do_start)
            t.daemon = True
            t.start()
        else:
            _do_start()

    def yolo_stop(self, delay_s: float = 0.0):
        """YOLO 추론 정지(서비스 호출) - 프레임 파이프라인은 건드리지 않음."""
        def _do_stop():
            now = time.time()
            if now - self._yolo_last_toggle < self._yolo_min_interval:
                return
            self._yolo_last_toggle = now
            if self._yolo_is_on:
                self.send_request(self.stop_yolov5_client, Trigger.Request())
                self._yolo_is_on = False
                self.get_logger().info("[self_driving] YOLO: STOP")
        if delay_s > 0:
            t = threading.Timer(delay_s, _do_stop)
            t.daemon = True
            t.start()
        else:
            _do_stop()

    def yolo_stop_and_wait(self, timeout_sec: float = 2.0) -> bool:
        """YOLO 추론 정지를 동기적으로 요청하고, 서비스 응답이 올 때까지 기다립니다."""
        if not self._yolo_is_on:
            return True

        self.get_logger().info("Requesting YOLO stop and waiting for confirmation...")
        future = self.stop_yolov5_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)

        if future.done() and future.result() is not None and future.result().success:
            self.get_logger().info("YOLO stop confirmed.")
            self._yolo_is_on = False
            self._yolo_last_toggle = time.time()
            return True
        else:
            self.get_logger().warn("YOLO stop service call failed or timed out.")
            self._yolo_is_on = False
            return False

    def init_process(self):
        self.timer.cancel()

        self.mecanum_pub.publish(Twist())
        if not self.get_parameter('only_line_follow').value:
            self.yolo_start()

        time.sleep(1)

        if 1:
            self.display = True
            self.enter_srv_callback(Trigger.Request(), Trigger.Response())
            request = SetBool.Request(); request.data = True
            self.set_running_srv_callback(request, SetBool.Response())

        # ⛔️ threading.Thread(target=self.main, daemon=True).start() 삭제
        # ✅ 타이머를 사용해 main_tick을 주기적으로 실행
        timer_period = 0.05  # 20Hz
        self.main_timer = self.create_timer(timer_period, self.main_tick)

        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')


    def param_init(self):
        self.start = False
        self.enter = False
        self.crt_time = time.time()
        self.start_flag = True # ✅ main_tick에서 사용하기 위해 클래스 속성으로 변경

        self.park_x = -1
        self.turn_right = False

        self.normal_speed = 0.7
        self.slow_down_speed = 0.0

        self.traffic_signs_status = None

        self.object_sub = None
        self.image_sub = None
        self.objects_info = []

        self.depth_sub = None
        self.depth_image = None
        self.avoid_until = 0.0
        self.dmin_ema = None
        self.min_wall_speed = 0.0
        self.last_avoid_s = 0.0

        self.last_depart_time = -1e9
        self.stop_cooldown = 2.0

        self.last_objects_ts = time.time()
        self.objects_timeout = 0.0

        self.signal_waiting = False
        self.signal_window = 3.0
        self.signal_deadline = 0.0
        self.red_hold = False
        self.max_red_wait = 10.0

        self.turn_right_count = 0
        self.stop_flag = True
        self.additional_flag = 10

    def get_node_state(self, request, response):
        response.success = True
        return response

    def send_request(self, client, msg):
        client.call_async(msg)

    def enter_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "self driving enter")
        with self.lock:
            self.start = False
            self.image_sub = self.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image' , self.image_callback, 1)
            self.depth_sub = self.create_subscription(Image, '/ascamera/camera_publisher/depth0/image_raw', self.depth_callback, 1)
            self.object_sub = self.create_subscription(ObjectsInfo, '/yolov5_ros2/object_detect', self.get_object_callback, 1)
            self.mecanum_pub.publish(Twist())
            self.enter = True
        response.success = True
        response.message = "enter"
        return response

    def exit_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "self driving exit")
        with self.lock:
            try:
                if self.image_sub: self.destroy_subscription(self.image_sub)
                if self.object_sub: self.destroy_subscription(self.object_sub)
                if self.depth_sub: self.destroy_subscription(self.depth_sub)
                self.image_sub = self.object_sub = self.depth_sub = None
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

    def image_callback(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        if not self.image_queue.full():
            self.image_queue.put_nowait(cv_image)

    def depth_callback(self, ros_depth):
        depth = self.bridge.imgmsg_to_cv2(ros_depth, desired_encoding='passthrough')
        depth_m = depth.astype(np.float32) / 1000.0 if depth.dtype == np.uint16 else depth
        with self.lock:
            self.depth_image = depth_m

    def _drive_straight(self, lane_x, x_setpoint, twist):
        if lane_x == -1:
            twist.linear.x = self.normal_speed
            twist.angular.z = 0.5
            self.mecanum_pub.publish(twist)
            self.get_logger().info("Searching for lane (called from invalid state)")
            return

        pos_error = lane_x - x_setpoint
        self.pid.SetPoint = 0
        self.pid.update(pos_error)
        twist.linear.x = self.normal_speed
        twist.angular.z = common.set_range(self.pid.output, -0.3, 0.3)
        self.mecanum_pub.publish(twist)

    def _do_right_turn(self):
        twist = Twist()
        self.get_logger().info("Moving straight for 0.5s before turning.")
        t_end = time.time() + 0.5
        while time.time() < t_end and self.is_running:
            twist.linear.x = 0.7
            twist.angular.z = 0.0
            self.mecanum_pub.publish(twist)
            time.sleep(0.02)

        self.get_logger().info("Turning right for 0.9s.")
        t_end = time.time() + 0.9
        while time.time() < t_end and self.is_running:
            twist.linear.x = 0.0
            twist.angular.z = -2.0
            self.mecanum_pub.publish(twist)
            time.sleep(0.02)

        self.mecanum_pub.publish(Twist())

    def _enter_signal_wait(self):
        self.objects_info = []
        now = time.time()
        self.signal_waiting = True
        self.signal_deadline = now + self.signal_window
        self.red_hold = False
        self.yolo_start()

    def _tick_signal_wait(self) -> bool:
        now = time.time()
        classes = {o.class_name for o in self.objects_info} if self.objects_info else set()

        if 'red' in classes:
            self.get_logger().info("RED!!!")
            self.red_hold = True
            self.mecanum_pub.publish(Twist())
            return True

        if 'green' in classes:
            self.get_logger().info("GREEEEEEN!!!")
            self.yolo_stop_and_wait() # ✅ 동기 호출로 변경
            self.objects_info = []
            self.signal_waiting = False
            self.red_hold = False
            self.last_depart_time = time.time()
            return False

        if 'right' in classes and not self.red_hold:
            self.yolo_stop_and_wait() # ✅ 동기 호출로 변경
            self._do_right_turn()
            self.objects_info = []
            self.signal_waiting = False
            self.last_depart_time = time.time()

            twist = Twist()
            self.get_logger().info("Parking sequence started.")
            # 1. 주차장 앞까지 직진
            t_end = time.time() + 2.8
            while time.time() < t_end and self.is_running:
                twist.linear.x = 0.7
                twist.angular.z = 0.0
                self.mecanum_pub.publish(twist)
                time.sleep(0.02)

            # 2. 주차
            t_end = time.time() + 1.0
            while time.time() < t_end and self.is_running:
                twist.linear.x = 0.0
                twist.linear.y = -0.5
                twist.angular.z = 0.0
                self.mecanum_pub.publish(twist)
                time.sleep(0.02)

            self.mecanum_pub.publish(Twist())
            self.is_running = False
            return False

        if self.red_hold:
            self.get_logger().info("WAIT.....")
            if self.max_red_wait and now > (self.signal_deadline + self.max_red_wait):
                self.yolo_stop_and_wait() # ✅ 동기 호출로 변경
                self.objects_info = []
                self.signal_waiting = False
                self.red_hold = False
                self.last_depart_time = time.time()
                return False
            self.mecanum_pub.publish(Twist())
            return True

        if now > self.signal_deadline:
            self.get_logger().info("YOLO couldn't detect anything........")
            self.yolo_stop_and_wait() # ✅ 동기 호출로 변경
            self.objects_info = []
            self.signal_waiting = False
            self.last_depart_time = time.time()
            return False

        self.mecanum_pub.publish(Twist())
        return True

    def main_tick(self):
        # 2차 방어: YOLO가 꺼져있을 때 만약의 유령 데이터 제거
        if not self._yolo_is_on:
            if self.objects_info:
                self.objects_info = []
        try:
            image = self.image_queue.get(block=False)
        except queue.Empty:
            return

        if self.start_flag:
            # 초기 출발 신호 대기 로직은 일회성이므로 tick 안에서 관리
            classes = {o.class_name for o in self.objects_info} if self.objects_info else set()
            if 'green' in classes:
                self.get_logger().info("Initial GREEN signal detected! Starting driving.")
                if self.yolo_stop_and_wait():
                    self.objects_info = []
                    self.start_flag = False
            elif 'red' in classes:
                self.get_logger().info("Initial signal is RED. Waiting...")
            return # 본격적인 주행은 다음 tick부터

        if self.start:
            h, w = image.shape[:2]
            visual_image, mask_white, mask_yellow = self.lane_detect.get_binary(image.copy())
            twist = Twist()

            # 1) 벽 회피 로직
            wall_turning = False
            with self.lock:
                depth_m = self.depth_image.copy() if self.depth_image is not None else None

            if depth_m is not None:
                y0, y1, x0, x1 = int(0.10*h), int(0.50*h), int(0.40*w), int(0.70*w)
                roi = depth_m[y0:y1, x0:x1]
                valid = np.isfinite(roi) & (roi > 0.05)
                if valid.any():
                    d_min = np.percentile(roi[valid], 10)
                    alpha = 0.4
                    d_est = d_min if self.dmin_ema is None else (1 - alpha) * self.dmin_ema + alpha * d_min
                    self.dmin_ema = d_est

                    NEAR, FAR = 0.35, 0.55
                    strength = 0.0
                    if d_est < FAR:
                        strength = np.clip((FAR - d_est) / max(FAR - NEAR, 1e-6), 0.0, 1.0)

                    now = time.time()
                    if strength > 0.05:
                        self.avoid_until = max(self.avoid_until, now + 0.20)

                    if (strength > 0.0) or (now < self.avoid_until):
                        wall_turning = True
                        s = strength ** 1.5 if strength > 0.0 else max(self.last_avoid_s * 0.7, 0.15)
                        self.last_avoid_s = s if strength > 0.0 else self.last_avoid_s * 0.7

                        twist.angular.z = -1.4 - 0.6 * s
                        v_min, v_max = self.min_wall_speed, self.normal_speed
                        twist.linear.x = v_min + (v_max - v_min) * (1.0 - s)
                        self.mecanum_pub.publish(twist)
                        self.get_logger().info("There's a Wall! I'm turning right!")
                        self.turn_right_count += 1
                        if self.turn_right_count > 4:
                            self.turn_right_count = 0
                            self.stop_flag = True

            if wall_turning:
                return
            else:
                self.additional_flag += 1

            # 2) 차선 주행 및 신호 처리 로직
            status, lane_x = self.lane_detect(mask_white, mask_yellow)
            x_setpoint = int(self.lane_detect.img_width * 0.20)

            if status == "GO_STRAIGHT":
                self._drive_straight(lane_x, x_setpoint, twist)
            elif status == "STOP_LINE":
                now = time.time()
                if now - self.last_depart_time < self.stop_cooldown:
                    self._drive_straight(lane_x, x_setpoint, twist)
                    return

                if self.signal_waiting:
                    if not self._tick_signal_wait(): # False를 반환하면 대기 종료
                        self._drive_straight(lane_x, x_setpoint, twist)
                elif self.stop_flag and self.additional_flag > 2:
                    self.get_logger().info("start a detection!!!!")
                    self.stop_flag = False
                    self.additional_flag = 0
                    self.mecanum_pub.publish(Twist())
                    self._enter_signal_wait()
            elif status is None:
                twist.linear.x = self.normal_speed
                twist.angular.z = 0.5
                self.mecanum_pub.publish(twist)
                self.get_logger().info("there isn't lane_x")
            else:
                self.pid.clear()

    def get_object_callback(self, msg):
        # 1차 방어: YOLO가 꺼져있다고 생각하면, 들어오는 모든 메시지를 무시
        if not self._yolo_is_on:
            if self.objects_info:
                self.objects_info = []
            return
        self.objects_info = msg.objects or []

def main():
    node = SelfDrivingNode('self_driving')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()