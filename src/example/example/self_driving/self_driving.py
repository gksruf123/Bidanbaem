#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/03/28
# @author:aiden
# autonomous driving
import os
import cv2
import math
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
from ros_robot_controller_msgs.msg import BuzzerState, SetPWMServoState, PWMServoState
from datetime import datetime

class SelfDrivingNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.name = name
        self.is_running = True
        self.pid = pid.PID(0.4, 0.0, 0.05)
        self.param_init()

        self.fps = fps.FPS()  
        self.image_queue = queue.Queue(maxsize=2)
        self.classes = ['go', 'right', 'park', 'red', 'green', 'crosswalk']
        self.display = True
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.colors = common.Colors()
        # signal.signal(signal.SIGINT, self.shutdown)
        self.machine_type = os.environ.get('MACHINE_TYPE')
        self.lane_detect = lane_detect.LaneDetector("yellow")

        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.servo_state_pub = self.create_publisher(SetPWMServoState, 'ros_robot_controller/pwm_servo/set_state', 1)
        self.result_publisher = self.create_publisher(Image, '~/image_result', 1)
        self.binary_publisher = self.create_publisher(Image, '~/image_binary', 1)

        self.create_service(Trigger, '~/enter', self.enter_srv_callback) # enter the game
        self.create_service(Trigger, '~/exit', self.exit_srv_callback) # exit the game
        self.create_service(SetBool, '~/set_running', self.set_running_srv_callback)
        # self.heart = Heart(self.name + '/heartbeat', 5, lambda _: self.exit_srv_callback(None))
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
        self._yolo_min_interval = 0.5   # 연속 토글 최소 간격(초) - 파이프라인 흔들림 방지
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

    def yolo_enable_for(self, seconds: float, start_delay: float = 0.0):
        """YOLO를 잠깐 켰다가(seconds 후 자동으로 끄기)."""
        # 이전 예약 끄기 취소
        if self._yolo_timer is not None:
            try:
                self._yolo_timer.cancel()
            except Exception:
                pass
            self._yolo_timer = None

        # 시작(필요시 지연)
        self.yolo_start(delay_s=start_delay)

        # seconds 후 자동 STOP
        def _auto_stop():
            self.yolo_stop()
            self._yolo_timer = None
        t = threading.Timer(seconds + start_delay, _auto_stop)
        t.daemon = True
        t.start()
        self._yolo_timer = t


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
        self.right = True
        self.crt_time = time.time()

        self.have_turn_right = False
        self.detect_turn_right = False
        self.detect_far_lane = False
        self.park_x = -1  # obtain the x-pixel coordinate of a parking sign

        self.start_turn_time_stamp = 0
        self.count_turn = 0
        self.start_turn = False  # start to turn

        self.count_right = 0
        self.count_right_miss = 0
        self.turn_right = False  # right turning sign

        self.last_park_detect = False
        self.count_park = 0  
        self.stop = False  # stopping sign
        self.start_park = False  # start parking sign

        self.count_crosswalk = 0
        self.crosswalk_distance = 0  # distance to the zebra crossing
        self.crosswalk_length = 0.1 + 0.3  # the length of zebra crossing and the robot

        self.start_slow_down = False  # slowing down sign
        self.normal_speed = 0.2  # normal driving speed
        self.slow_down_speed = 0.1  # slowing down speed

        self.traffic_signs_status = None  # record the state of the traffic lights
        self.red_loss_count = 0

        self.object_sub = None
        self.image_sub = None
        self.objects_info = []

        self.depth_sub = None
        self.depth_image = None
        self.depth_stamp = None
        self.avoid_until = 0.0
        self.dmin_ema = None        # d_min 평활화용
        self.min_wall_speed = 0.1
        self.last_avoid_s = 0.0

        self.last_stop_time = 0     # 횡단보도 마지막에 멈췄던 시간 체크용
        self.stop_cooldown = 3.0    # 횡단보도 한번 멈추면 그 이후로 안 멈추는 시간

        self.stop_duration = 1.0  # 원하는 정지 시간(초)
        self.stop_until = 0.0

        self.last_objects_ts = time.time()
        self.objects_timeout = 1.0  # 초
        

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
            # if the queue is full, remove the oldest image
            self.image_queue.get()
        # put the image into the queue
        self.image_queue.put(rgb_image)
    
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
            self.depth_stamp = (ros_depth.header.stamp.sec, ros_depth.header.stamp.nanosec)
    
    # parking processing
    def park_action(self):
        if self.machine_type == 'MentorPi_Mecanum': 
            twist = Twist()
            twist.linear.y = -0.2
            self.mecanum_pub.publish(twist)
            time.sleep(0.38/0.2)
        elif self.machine_type == 'MentorPi_Acker':
            twist = Twist()
            twist.linear.x = 0.15
            twist.angular.z = twist.linear.x*math.tan(-0.5061)/0.145
            self.mecanum_pub.publish(twist)
            time.sleep(3)

            twist = Twist()
            twist.linear.x = 0.15
            twist.angular.z = -twist.linear.x*math.tan(-0.5061)/0.145
            self.mecanum_pub.publish(twist)
            time.sleep(2)

            twist = Twist()
            twist.linear.x = -0.15
            twist.angular.z = twist.linear.x*math.tan(-0.5061)/0.145
            self.mecanum_pub.publish(twist)
            time.sleep(1.5)

        else:
            twist = Twist()
            twist.angular.z = -1
            self.mecanum_pub.publish(twist)
            time.sleep(1.5)
            self.mecanum_pub.publish(Twist())
            twist = Twist()
            twist.linear.x = 0.2
            self.mecanum_pub.publish(twist)
            time.sleep(0.65/0.2)
            self.mecanum_pub.publish(Twist())
            twist = Twist()
            twist.angular.z = 1
            self.mecanum_pub.publish(twist)
            time.sleep(1.5)
        self.mecanum_pub.publish(Twist())

    def main(self):
        first_frame_seen = False

        while self.is_running:
            time_start = time.time()
            try:
                image = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                if not self.is_running:
                    break
                else:
                    continue

            # --- 첫 프레임을 받은 직후 한 번만 안전지연 stop(옵션) ---
            if not first_frame_seen:
                first_frame_seen = True
                # 기본 ON으로 시작했지만 라인팔로우가 주가라면, 프레임 안정 후 YOLO OFF
                # 필요 없으면 이 줄 지워도 됨
                self.yolo_stop(delay_s=0.7)  # 0.5~1.0s 사이 튜닝 권장

            # 욜로 감지된 지 오래됐으면 기존의 욜로 객체 전부 초기화
            yolo_now = time.time()
            if yolo_now - self.last_objects_ts > self.objects_timeout:
                self.objects_info = []
                self.traffic_signs_status = None
                self.park_x = -1
                self.turn_right = False 

            result_image = image.copy()
            if self.start:
                h, w = image.shape[:2]

                # obtain the binary image of the lane
                binary_image = self.lane_detect.get_binary(image)

                twist = Twist()

                # 1) 앞에 벽이 있을 때 우회전
                wall_turning = False
                with self.lock:
                    depth_m = None if self.depth_image is None else self.depth_image.copy()

                if depth_m is not None:
                    y0, y1 = int(0.20*h), int(0.40*h)
                    x0, x1 = int(0.30*w), int(0.70*w)
                    roi = depth_m[y0:y1, x0:x1]

                    valid = np.isfinite(roi) & (roi > 0.05)
                    if valid.any():
                        d_min = np.percentile(roi[valid], 10)

                        # EMA
                        alpha = 0.4
                        d_est = d_min if self.dmin_ema is None else (1 - alpha) * self.dmin_ema + alpha * d_min
                        self.dmin_ema = d_est

                        NEAR, FAR = 0.10, 0.30
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
                            twist.angular.z = -0.5 - 0.6 * s      # -0.2 ~ -0.8 근처

                            # === 가변 선속도 ===
                            v_min = self.min_wall_speed           # 예: 0.05
                            v_max = self.normal_speed             # 예: 0.20
                            twist.linear.x = v_min + (v_max - v_min) * (1.0 - s)

                            self.mecanum_pub.publish(twist)
                
                    if (not wall_turning) and (time.time() < self.avoid_until):
                        wall_turning = True
                        # 유지 구간 감쇠
                        self.last_avoid_s *= 0.7
                        s = max(self.last_avoid_s, 0.15)

                        twist = Twist()
                        twist.angular.z = -0.2 - 0.6 * s
                        v_min = self.min_wall_speed
                        v_max = self.normal_speed
                        twist.linear.x = v_min + (v_max - v_min) * (1.0 - s)
                        self.mecanum_pub.publish(twist) 

                if wall_turning:
                    pass
                else:
                    # pid 차선 유지 로직이 이 안에 들어감.
                    # 즉, 회피 기동 중일 때는 차선 유지 로직이 아예 실행조차 안 됨.
                                        
                # # if detecting the zebra crossing, start to slow down
                # self.get_logger().info('\033[1;33m%s\033[0m' % self.crosswalk_distance)
                # if 70 < self.crosswalk_distance and not self.start_slow_down:  # The robot starts to slow down only when it is close enough to the zebra crossing
                #     # 
                #     self.count_crosswalk += 1
                #     if self.count_crosswalk == 3:  # judge multiple times to prevent false detection
                #         self.count_crosswalk = 0
                #         self.start_slow_down = True  # sign for slowing down
                #         self.count_slow_down = time.time()  # fixing time for slowing down
                # else:  # need to detect continuously, otherwise reset
                #     self.count_crosswalk = 0

                # # deceleration processing
                # if self.start_slow_down:
                #     if self.traffic_signs_status is not None:
                #         area = abs(self.traffic_signs_status.box[0] - self.traffic_signs_status.box[2]) * abs(self.traffic_signs_status.box[1] - self.traffic_signs_status.box[3])
                #         if self.traffic_signs_status.class_name == 'red' and area < 1000:  # If the robot detects a red traffic light, it will stop
                #             self.mecanum_pub.publish(Twist())
                #             self.stop = True
                #         elif self.traffic_signs_status.class_name == 'green':  # If the traffic light is green, the robot will slow down and pass through
                #             twist.linear.x = self.slow_down_speed
                #             self.stop = False
                #     if not self.stop:  # In other cases where the robot is not stopped, slow down the speed and calculate the time needed to pass through the crosswalk. The time needed is equal to the length of the crosswalk divided by the driving speed
                #         twist.linear.x = self.slow_down_speed
                #         if time.time() - self.count_slow_down > self.crosswalk_length / twist.linear.x:
                #             self.start_slow_down = False
                # else:
                #     twist.linear.x = self.normal_speed  # go straight with normal speed

                # # If the robot detects a stop sign and a crosswalk, it will slow down to ensure stable recognition
                # if 0 < self.park_x and 135 < self.crosswalk_distance:
                #     twist.linear.x = self.slow_down_speed
                #     if not self.start_park and 180 < self.crosswalk_distance:  # When the robot is close enough to the crosswalk, it will start parking
                #         self.count_park += 1  
                #         if self.count_park >= 15:  
                #             self.mecanum_pub.publish(Twist())  
                #             self.start_park = True
                #             self.stop = True
                #             threading.Thread(target=self.park_action).start()
                #     else:
                #         self.count_park = 0  

                # line following processing
                    result_image, status, lane_angle, lane_x = self.lane_detect(binary_image, image.copy())  # the coordinate of the line while the robot is in the middle of the lane
                    x_setpoint = int(w * 0.20) # 화면 중앙에서 살짝 왼쪽.
                    angle_setpoint = 80

                    if status == "GO_STRAIGHT":
                        pos_error = lane_x - x_setpoint
                        angle_error = lane_angle - angle_setpoint
                        total_error = 0.7*pos_error + 0.3*angle_error

                        self.pid.SetPoint = 0
                        self.pid.update(total_error)
                        twist.linear.x = self.normal_speed
                        twist.angular.z = common.set_range(self.pid.output, -0.2, 0.2)
                        self.get_logger().info(f"pos_error={pos_error:.2f}, angle_error={angle_error:.2f}, total={total_error:.2f}")
                        self.mecanum_pub.publish(twist)

                    elif status == "STOP_LINE":
                        now = time.time()

                        # 이미 정지 유지 중이면 계속 0속도 퍼블리시
                        if now < self.stop_until:
                            self.mecanum_pub.publish(Twist())
                            continue  # 또는 return

                        # 새로 정지를 시작할 조건(쿨다운 경과)이라면: 정지 타이머 설정
                        if now - self.last_stop_time > self.stop_cooldown:
                            self.last_stop_time = now
                            self.stop_until = now + self.stop_duration
                            self.mecanum_pub.publish(Twist())  # 정지 시작
                            continue  # 또는 return

                        # 그 외엔(오검출/쿨다운 미경과) 저속 크리핑 등 원하는 기본 동작
                        twist.linear.x = self.normal_speed
                        twist.angular.z = 0.0
                        self.mecanum_pub.publish(twist)
                    
                    elif status is None:
                        twist.linear.x = self.slow_down_speed
                        twist.angular.z = 0.0
                        self.mecanum_pub.publish(twist)
                    
                    else:
                        self.pid.clear()


                # if lane_x >= 0 and not self.stop:  
                #     if lane_x > 150:  
                #         self.count_turn += 1
                #         if self.count_turn > 5 and not self.start_turn:
                #             self.start_turn = True
                #             self.count_turn = 0
                #             self.start_turn_time_stamp = time.time()
                #         if self.machine_type != 'MentorPi_Acker':
                #             twist.angular.z = -0.45  # turning speed
                #         else:
                #             twist.angular.z = twist.linear.x * math.tan(-0.5061) / 0.145
                #     else:  # use PID algorithm to correct turns on a straight road
                #         self.count_turn = 0
                #         if time.time() - self.start_turn_time_stamp > 2 and self.start_turn:
                #             self.start_turn = False
                #         if not self.start_turn:
                #             self.pid.SetPoint = 130  # the coordinate of the line while the robot is in the middle of the lane
                #             self.pid.update(lane_x)
                #             if self.machine_type != 'MentorPi_Acker':
                #                 twist.angular.z = common.set_range(self.pid.output, -0.1, 0.1)
                #             else:
                #                 twist.angular.z = twist.linear.x * math.tan(common.set_range(self.pid.output, -0.1, 0.1)) / 0.145
                #         else:
                #             if self.machine_type == 'MentorPi_Acker':
                #                 twist.angular.z = 0.15 * math.tan(-0.5061) / 0.145
                #     self.mecanum_pub.publish(twist)  
                # else:
                #     self.pid.clear()

             
                if self.objects_info:
                    for i in self.objects_info:
                        box = i.box
                        class_name = i.class_name
                        cls_conf = i.score
                        cls_id = self.classes.index(class_name)
                        color = self.colors(cls_id, True)
                        plot_one_box(
                            box,
                            result_image,
                            color=color,
                            label="{}:{:.2f}".format(class_name, cls_conf),
                        )

            else:
                time.sleep(0.01)

            
            bgr_image = result_image
            if self.display:
                self.fps.update()
                bgr_image = self.fps.show_fps(bgr_image)

            
            self.result_publisher.publish(self.bridge.cv2_to_imgmsg(bgr_image, "bgr8"))
            self.binary_publisher.publish(self.bridge.cv2_to_imgmsg(binary_image, "mono8"))

           
            target_period = 1.0 / 20.0   # 20fps → 0.05초
            time_d = target_period - (time.time() - time_start)
            if time_d > 0:
                time.sleep(time_d)

        self.mecanum_pub.publish(Twist())
        rclpy.shutdown()


    # Obtain the target detection result
    def get_object_callback(self, msg):
        frame_per_sec = 1 / (time.time() - self.crt_time)
        self.crt_time = time.time()
        self.last_objects_ts = time.time()

        self.objects_info = msg.objects
        if self.objects_info == []:  # If it is not recognized, reset the variable
            self.traffic_signs_status = None
            # self.crosswalk_distance = 0
        else:
            # min_distance = 0
            for i in self.objects_info:
                class_name = i.class_name
                center = (int((i.box[0] + i.box[2])/2), int((i.box[1] + i.box[3])/2))
                
                # if class_name == 'crosswalk':  
                #     if center[1] > min_distance:  # Obtain recent y-axis pixel coordinate of the crosswalk
                #         min_distance = center[1]
                if class_name == 'right':  # obtain the right turning sign
                    self.count_right += 1
                    self.count_right_miss = 0
                    if self.count_right >= 5:  # If it is detected multiple times, take the right turning sign to true
                        self.turn_right = True
                        self.count_right = 0
                elif class_name == 'park':  # obtain the center coordinate of the parking sign
                    self.park_x = center[0]
                elif class_name == 'red' or class_name == 'green':  # obtain the status of the traffic light
                    self.traffic_signs_status = i
               
            self.get_logger().info(f'\033[1;31m{frame_per_sec}\033[0m')
            self.get_logger().info('\033[1;32m%s\033[0m' % class_name)
            # self.crosswalk_distance = min_distance

def main():
    node = SelfDrivingNode('self_driving')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
 
if __name__ == "__main__":
    main()

    
