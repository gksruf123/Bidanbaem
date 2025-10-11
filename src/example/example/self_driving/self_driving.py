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

    def init_process(self):
        self.timer.cancel()

        self.mecanum_pub.publish(Twist())
        if not self.get_parameter('only_line_follow').value:
            # 기본 전략 선택: (a) 기본 ON → 필요할 때 잠깐 OFF, (b) 기본 OFF → 필요할 때 잠깐 ON
            # a안 예시:
            self.yolo_start()     # ← 기존 self.send_request(self.start_yolov5_client, ...) 대체
            # b안을 원하면: self.yolo_stop()

        time.sleep(1)

        if 1:
            self.display = True
            self.enter_srv_callback(Trigger.Request(), Trigger.Response())
            request = SetBool.Request(); request.data = True
            self.set_running_srv_callback(request, SetBool.Response())

        threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')


    def param_init(self):
        self.start = False
        self.enter = False
        self.crt_time = time.time()

        self.park_x = -1  # obtain the x-pixel coordinate of a parking sign
        self.turn_right = False  # right turning sign

        self.normal_speed = 0.65  # normal driving speed
        self.slow_down_speed = 0.0  # slowing down speed

        self.traffic_signs_status = None  # record the state of the traffic lights

        self.object_sub = None
        self.image_sub = None
        self.objects_info = []

        self.depth_sub = None
        self.depth_image = None
        self.avoid_until = 0.0
        self.dmin_ema = None        # d_min 평활화용
        self.min_wall_speed = 0.0
        self.last_avoid_s = 0.0

        self.last_depart_time = -1e9     # 횡단보도 마지막에 멈췄던 시간 체크용. 첫 실행 때 횡단보도 무시를 방지하기 위해 초기값을 과거로 설정.
        self.stop_cooldown = 2.0    # 횡단보도 한번 멈추면 그 이후로 안 멈추는 시간

        self.last_objects_ts = time.time()
        self.objects_timeout = 0.0  # 초

        self.signal_waiting = False     # 정지선 신호 대기 모드
        self.signal_window = 3.0        # 정지선에서 yolo 켜고 기다릴 시간
        self.signal_deadline = 0.0
        self.red_hold = False           # 빨간불 봤을 때 초록불 나올 때까지 대기하는 플래그
        self.max_red_wait = 10.0        # 빨간불 대기 최대 (혹시 몰라서. 없어도 됨.)

        self.turn_right_count = 0       # 이것도 벽 마주친 횟수 세기 위한 용도.
        self.stop_flag = True           # 벽 마주치고 우회전 했을 때만 다시 횡단보도 인식하게 만들기 위함.
        self.additional_flag = 10     # 우회전한 다음에 직진했을 때만 다시 인식하게끔 추가 플래그.

    def get_node_state(self, request, response):
        response.success = True
        return response

    def send_request(self, client, msg):
        client.call_async(msg)

    def enter_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "self driving enter")
        with self.lock:
            self.start = False
            camera = 'depth_cam'#self.get_parameter('depth_camera_name').value
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
                if self.image_sub:
                    self.destroy_subscription(self.image_sub)
                    self.image_sub = None

                if self.object_sub:
                    self.destroy_subscription(self.object_sub)
                    self.object_sub = None

                if self.depth_sub:
                    self.destroy_subscription(self.depth_sub)
                    self.depth_sub = None
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

    def shutdown(self, signum, frame):  # press 'ctrl+c' to close the program
        self.is_running = False

    def image_callback(self, ros_image):  # callback target checking
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        if self.image_queue.full():
            try: self.image_queue.get_nowait()
            except queue.Empty: pass
        try:
            self.image_queue.put_nowait(rgb_image)
        except queue.Full:
            pass
    
    def depth_callback(self, ros_depth):
        depth = self.bridge.imgmsg_to_cv2(ros_depth, desired_encoding='passthrough')

        if depth.dtype == np.uint16:
            depth_m = depth.astype(np.float32) / 1000.0
        elif depth.dtype == np.float32:
            depth_m = depth
        else:
            return
        
        with self.lock:
            self.depth_image = depth_m
    
    # PID for GO_STRAIGHT
    def _drive_straight(self, lane_x, x_setpoint, twist):
        if lane_x == -1:
            # lane_x가 -1이면 '차선 탐색' 동작을 수행하고 즉시 함수를 종료합니다.
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
        # 우회전 표지판 봤을 때 실행되는 메서드
        twist = Twist()

        # 1. 0.5초 동안 직진
        self.get_logger().info("Moving straight for 0.5s before turning.")
        forward_time = 0.55
        t_end_forward = time.time() + forward_time
        while time.time() < t_end_forward and self.is_running:
            twist.linear.x = 0.7  # 설정된 기본 속도로 직진
            twist.angular.z = 0.0
            self.mecanum_pub.publish(twist)
            time.sleep(0.02)

        # 2. 0.8초 동안 제자리에서 우회전 (원래 코드)
        self.get_logger().info("Turning right for 0.8s.")
        turn_time = 0.9
        t_end_turn = time.time() + turn_time
        while time.time() < t_end_turn and self.is_running:
            twist.linear.x = 0.0
            twist.angular.z = -2.0
            self.mecanum_pub.publish(twist)
            time.sleep(0.02)

        # 3. 마지막에 완전히 정지
        self.mecanum_pub.publish(Twist())

    def _enter_signal_wait(self):
        # 횡단보도에서 처음 멈췄을 때 실행되는 메서드
        self.objects_info = []
        now = time.time()
        self.signal_waiting = True
        self.signal_deadline = now + self.signal_window
        self.red_hold = False
        self.yolo_start()

    def _tick_signal_wait(self) -> bool:
        """
        신호 대기 중 호출. True면 계속 대기, False면 대기 종료(라인 팔로우 복귀).
        """
        now = time.time()

        # 최신 한 프레임에서 본 욜로 객체 클래스들의 집합
        classes = {o.class_name for o in self.objects_info} if self.objects_info else set()

        # 1) 빨간불일 경우: red_hold 진입 (초록불 볼 때까지 정지)
        if 'red' in classes:
            self.get_logger().info("RED!!!")
            self.red_hold = True
            self.mecanum_pub.publish(Twist())
            return True

        # 2) 초록불일 경우 즉시 출발
        if 'green' in classes:
            self.get_logger().info("GREEEEEEN!!!")
            self.yolo_stop(delay_s=0.0)
            self.objects_info = []
            self.signal_waiting = False
            self.red_hold = False
            self.last_depart_time = time.time()
            return False
        
        # 3) 우회전 표지 (빨간불 없을 때만 유효)
        if 'right' in classes and not self.red_hold:
            self._do_right_turn()
            self.yolo_stop(delay_s=0.0)
            self.objects_info = []
            self.signal_waiting = False
            self.last_depart_time = time.time()

            # 오돔 빼고 걍 슈퍼 하드 코딩
            twist = Twist()

            # 1. 주차장 앞까지 직진
            forward_time = 2.8
            t_end_forward = time.time() + forward_time
            while time.time() < t_end_forward and self.is_running:
                twist.linear.x = 0.7  # 설정된 기본 속도로 직진
                twist.angular.z = 0.0
                self.mecanum_pub.publish(twist)
                time.sleep(0.02)    

            # 2. 주차
            forward_time = 1.0
            t_end_forward = time.time() + forward_time
            while time.time() < t_end_forward and self.is_running:
                twist.linear.x = 0.0
                twist.linear.y = -0.5  # 설정된 기본 속도로 직진
                twist.angular.z = 0.0
                self.mecanum_pub.publish(twist)
                time.sleep(0.02)                    
            
            # 정지 및 주행 플래그 False로 바꿈으로써 주행 종료
            self.mecanum_pub.publish(Twist())
            self.is_running = False

            return False        
        
        # 4) 빨간불이 보인 적이 있다면: 초록불 나올 때까지 정지 유지
        if self.red_hold:
            self.get_logger().info("WAIT.....")
            # 최대 대기 시간 = 타임아웃
            if self.max_red_wait and now > (self.signal_deadline + self.max_red_wait):
                self.yolo_stop(delay_s=0.0)
                self.objects_info = []
                self.signal_waiting = False
                self.red_hold = False
                self.last_depart_time = time.time()
                return False
            self.mecanum_pub.publish(Twist())
            return True
        
        # 5) 아무 것도 안 보일 경우 최소 대기 시간만큼만 기다렸다가 출발
        if now > self.signal_deadline:
            self.get_logger().info("YOLO couldn't detect anything........")
            self.yolo_stop(delay_s=0.0)
            self.objects_info = []
            self.signal_waiting = False
            self.last_depart_time = time.time()
            return False
        
        # 6) 인식 중에는 계속 정지 유지
        self.mecanum_pub.publish(Twist())
        return True

    def main(self):
        start_flag = True

        while self.is_running:
            # 내 노드가 YOLO가 꺼져있다고 생각하면, 들어오는 모든 메시지를 무시
            if not self._yolo_is_on:
                # 만약을 위해 여기서 한 번 더 비워주면 더 안전함
                if self.objects_info:
                    self.objects_info = []
            
            time_start = time.time()
            try:
                image = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                if not self.is_running:
                    break
                else:
                    continue

            # 처음에 '초록불' 신호를 무한정 기다리는 로직
            if start_flag:
                start_flag = False
                self.get_logger().info("Waiting for initial GREEN signal to start...")
                self.yolo_start() # 신호를 보기 위해 YOLO를 켭니다.

                while self.is_running: # 초록불을 볼 때까지 무한정 반복
                    # YOLO가 감지한 객체 목록을 확인합니다.
                    classes = {o.class_name for o in self.objects_info} if self.objects_info else set()

                    if 'green' in classes:
                        self.get_logger().info("Initial GREEN signal detected! Starting driving.")
                        # 출발 후에는 신호 감지가 필요 없으므로 1초 뒤에 YOLO를 끕니다.
                        self.yolo_stop(delay_s=0.0)
                        self.objects_info = []
                        break # 무한 반복을 탈출하고 본격적인 주행 시작

                    if 'red' in classes:
                        self.get_logger().info("Initial signal is RED. Waiting...")

                    time.sleep(0.1) # 0.1초마다 신호를 다시 확인
                
                # 첫 번째 루프는 여기서 끝내고 다음 루프부터 정상 주행 시작
                continue

            result_image = image.copy()
            if self.start:
                h, w = image.shape[:2]

                # obtain the binary image of the lane
                visual_image, mask_white, mask_yellow = self.lane_detect.get_binary(image.copy())

                twist = Twist()

                # 1) 앞에 벽이 있을 때 우회전
                wall_turning = False
                with self.lock:
                    depth_m = None if self.depth_image is None else self.depth_image.copy()

                if depth_m is not None:
                    y0, y1 = int(0.10*h), int(0.50*h)
                    x0, x1 = int(0.40*w), int(0.70*w)
                    roi = depth_m[y0:y1, x0:x1]

                    valid = np.isfinite(roi) & (roi > 0.05)
                    if valid.any():
                        d_min = np.percentile(roi[valid], 10)

                        # EMA
                        alpha = 0.4
                        d_est = d_min if self.dmin_ema is None else (1 - alpha) * self.dmin_ema + alpha * d_min
                        self.dmin_ema = d_est

                        NEAR, FAR = 0.35, 0.52
                        if d_est < FAR:
                            strength = (FAR - d_est) / max(FAR - NEAR, 1e-6)
                            strength = float(np.clip(strength, 0.0, 1.0))
                        else:
                            strength = 0.0

                        now = time.time()
                        if strength > 0.05:
                            self.avoid_until = max(self.avoid_until, now + 0.20)

                        # === 회피 여부 결정 ===
                        if (strength > 0.0) or (now < self.avoid_until):
                            wall_turning = True

                            # 회피 강도 s 계산 (가까울수록 ↑). 유지 구간에서는 마지막 s를 서서히 감쇠.
                            if strength > 0.0:
                                s = strength ** 1.5
                                self.last_avoid_s = s
                            else:
                                # 유지 타임 동안은 이전 강도를 서서히 줄이며(감쇠) 자연 복귀
                                self.last_avoid_s *= 0.7
                                s = max(self.last_avoid_s, 0.15)   # 최소한의 회피 유지(필요시 0.10~0.20 튜닝)

                            # === 가변 조향 ===
                            twist.angular.z = -1.4 - 0.6 * s      # -0.2 ~ -0.8 근처

                            # === 가변 선속도 ===
                            v_min = self.min_wall_speed           # 예: 0.05
                            v_max = self.normal_speed             # 예: 0.20
                            twist.linear.x = v_min + (v_max - v_min) * (1.0 - s)

                            self.mecanum_pub.publish(twist)

                            self.get_logger().info("There's a Wall! I'm turning right!")
                            
                            self.turn_right_count += 1
                            if self.turn_right_count > 4:
                                self.turn_right_count = 0
                                self.stop_flag = True   # 횡단보도 다시 인식하게 만들기.
                
                    if (not wall_turning) and (time.time() < self.avoid_until):
                        wall_turning = True
                        # 유지 구간 감쇠
                        self.last_avoid_s *= 0.7
                        s = max(self.last_avoid_s, 0.15)

                        twist = Twist()
                        twist.angular.z = -1.4 - 0.6 * s
                        v_min = self.min_wall_speed
                        v_max = self.normal_speed
                        twist.linear.x = v_min + (v_max - v_min) * (1.0 - s)
                        self.mecanum_pub.publish(twist) 

                if wall_turning:
                    continue
                else:
                    # pid 차선 유지 로직이 이 안에 들어감.
                    # 즉, 회피 기동 중일 때는 차선 유지 로직이 아예 실행조차 안 됨.                              
                    self.additional_flag += 1

                # line following processing
                    status, lane_x = self.lane_detect(mask_white, mask_yellow)
                    resized_w = self.lane_detect.img_width
                    x_setpoint = int(resized_w * 0.20) # 화면 중앙에서 왼쪽.

                    if status == "GO_STRAIGHT":
                        self._drive_straight(lane_x, x_setpoint, twist)

                    elif status == "STOP_LINE":
                        now = time.time()

                        # 2) 횡단보도 한번 인식한 후로 일정 시간 동안은 횡단보도 무시.
                        if now - self.last_depart_time < self.stop_cooldown:
                            self._drive_straight(lane_x, x_setpoint, twist)
                            continue

                        # 이미 객체 인식 중이면 계속 인식 유지
                        if self.signal_waiting:
                            if self._tick_signal_wait():
                                continue
                            # 인식 끝나면 라인팔로우 복귀
                            self.yolo_stop(delay_s=0.0)
                            self.objects_info = []                            
                            self._drive_straight(lane_x, x_setpoint, twist)
                            continue

                        # 3) 횡단보도 처음 마주치면 정지 후 객체 인식 (벽 앞에서 우회전했을 때만 다시)
                        if self.stop_flag and self.additional_flag > 2:
                            self.get_logger().info("start a detection!!!!")
                            self.stop_flag = False
                            self.additional_flag = 0
                            self.mecanum_pub.publish(Twist())
                            self._enter_signal_wait()
                            continue
                    
                    # 아무것도 안 보이면 천천히 왼쪽으로 돌면서 차선 찾기
                    elif status is None:
                        twist.linear.x = self.normal_speed
                        twist.angular.z = 0.5
                        self.mecanum_pub.publish(twist)
                        self.get_logger().info("there isn't lane_x")
                    
                    else:
                        self.pid.clear()

            else:
                time.sleep(0.01)
           
            target_period = 1.0 / 20.0   # 20fps → 0.05초
            time_d = target_period - (time.time() - time_start)
            if time_d > 0:
                time.sleep(time_d)

        self.mecanum_pub.publish(Twist())
        rclpy.shutdown()


    # Obtain the target detection result
    def get_object_callback(self, msg):
        # 내 노드가 YOLO가 꺼져있다고 생각하면, 들어오는 모든 메시지를 무시
        if not self._yolo_is_on:
            # 만약을 위해 여기서 한 번 더 비워주면 더 안전함
            if self.objects_info:
                self.objects_info = []
            return

        self.last_objects_ts = time.time()
        self.objects_info = msg.objects or []

def main():
    node = SelfDrivingNode('self_driving')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
 
if __name__ == "__main__":
    main()

    
