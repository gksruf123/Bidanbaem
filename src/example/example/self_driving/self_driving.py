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
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


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

    def init_process(self):
        self.timer.cancel()

        self.mecanum_pub.publish(Twist())
        if not self.get_parameter('only_line_follow').value:
            self.send_request(self.start_yolov5_client, Trigger.Request())
        time.sleep(1)
        
        if 1:#self.get_parameter('start').value:
            self.display = True
            self.enter_srv_callback(Trigger.Request(), Trigger.Response())
            request = SetBool.Request()
            request.data = True
            self.set_running_srv_callback(request, SetBool.Response())

        #self.park_action() 
        threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

    def param_init(self):
        self.start = False
        self.enter = False
        self.right = True

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

        self.normal_speed = 0.5  # normal driving speed
        self.slow_down_speed = 0.1  # slowing down speed

        self.traffic_signs_status = None  # record the state of the traffic lights
        self.red_loss_count = 0

        self.object_sub = None
        self.image_sub = None
        self.objects_info = []

        # --- Y-기반 턴 트리거 튜닝 파라미터 ---
        self.y_turn_ratio = 0.75       # 화면 높이의 하단 25% 임계 (0.70~0.90에서 조정)
        self.band_density_th = 0.02    # 하단 밴드 픽셀 밀도 임계(1~5% 추천)
        self.turn_frames_req = 6       # 연속 프레임 수(>5와 동일 의미)
        self.turn_hold_sec = 2.0       # 턴 유지 시간(초)
        self.turn_linear_limit = 0.0  # 턴 중 최대 직진 속도(안정화)
        self.rotating = False

    def get_node_state(self, request, response):
        response.success = True
        return response

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def enter_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "self driving enter")
        with self.lock:
            self.start = False
            # 구독 핸들을 멤버에 저장해서 GC로 사라지지 않게 함
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

    def shutdown(self, signum, frame):  # press 'ctrl+c' to close the program
        self.is_running = False

    def image_callback(self, ros_image):  # callback target checking
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        if self.image_queue.full():
            # if the queue is full, remove the oldest image
            self.image_queue.get()
        # put the image into the queue
        self.image_queue.put(rgb_image)
    
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

    # --- Quaternion -> yaw(rad) 유틸 ---
    def _yaw_from_quat(self, q):
        # geometry_msgs/Quaternion -> yaw
        # 참고: yaw 범위는 [-pi, pi]
        siny_cosp = 2.0 * (q.w*q.z + q.x*q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    # --- 옵션 A: 오픈 루프(간단/캘리브 필요) ---
    def rotate_right_90_in_place(self, yaw_rate=0.8):
        """
        제자리에서 시계방향(오른쪽) 90도 회전.
        yaw_rate: rad/s. 실제 90°가 맞지 않으면 yaw_rate나 duration을 조정.
        """
        self.rotating = True   # <-- 회전 시작
        twist = Twist()
        twist.linear.x = twist.linear.y = twist.linear.z = 0.0
        twist.angular.x = twist.angular.y = 0.0
        twist.angular.z = -abs(yaw_rate)  # 보통 오른쪽(시계)은 음수

        angle = math.pi / 2  # 90°
        duration = angle / abs(yaw_rate)

        t0 = time.time()
        while time.time() - t0 < duration and self.is_running:
            self.mecanum_pub.publish(twist)
            time.sleep(0.02)

        self.mecanum_pub.publish(Twist())
        self.rotating = False  # <-- 회전 끝


    # --- 옵션 B: 피드백 루프(정확/권장) ---
    def rotate_right_90_feedback(self, source="odom", max_rate=1.0, min_rate=0.2, timeout=5.0):
        """
        /odom 또는 /imu의 yaw로 -90° 도달까지 제자리 회전.
        source: "odom" | "imu"
        max_rate/min_rate: 각속도 제한
        """
        yaw_now = None
        self.rotating = True   # <-- 회전 시작
        

        def odom_cb(msg):
            nonlocal yaw_now
            yaw_now = self._yaw_from_quat(msg.pose.pose.orientation)

        def imu_cb(msg):
            nonlocal yaw_now
            yaw_now = self._yaw_from_quat(msg.orientation)

        # 구독 시작
        if source == "odom":
            sub = self.create_subscription(Odometry, "/odom", odom_cb, 1)
        else:
            sub = self.create_subscription(Imu, "/imu", imu_cb, 1)

        # 초기 yaw 확보
        t0 = time.time()
        while yaw_now is None and (time.time() - t0) < 1.0:
            rclpy.spin_once(self, timeout_sec=0.05)

        if yaw_now is None:
            self.get_logger().warn("Yaw source not available; fallback to open-loop 90°.")
            try:
                self.destroy_subscription(sub)
            except Exception:
                pass
            return self.rotate_right_90_in_place(0.8)

        yaw_start = yaw_now
        target_delta = -math.pi / 2  # 오른쪽 90도
        Kp = 1.2  # 필요 시 튜닝

        twist = Twist()
        twist.linear.x = twist.linear.y = 0.0  # 제자리 회전

        t0 = time.time()
        while (time.time() - t0) < timeout and self.is_running:
            rclpy.spin_once(self, timeout_sec=0.01)
            if yaw_now is None:
                continue

            # 현재 진행된 각도(−pi~pi wrap)
            delta = ((yaw_now - yaw_start + math.pi) % (2*math.pi)) - math.pi
            err = target_delta - delta

            # P제어 각속도
            w_cmd = Kp * err
            # 제한 + 최소 속도 보장
            w_cmd = max(-max_rate, min(max_rate, w_cmd))
            if abs(w_cmd) < min_rate:
                w_cmd = -min_rate if err < 0 else min_rate

            twist.angular.z = w_cmd
            self.mecanum_pub.publish(twist)

            # 오차 임계 도달 시 종료
            if abs(err) < math.radians(2.0):  # ±2°
                break

            time.sleep(0.01)

        self.mecanum_pub.publish(Twist())

        # 필요 시 구독 해제
        try:
            self.destroy_subscription(sub)
        except Exception:
            pass

        self.rotating = False  # <-- 회전 끝

    def main(self):
        while self.is_running:
            if self.rotating:
                # 회전 중에는 다른 주행 명령을 막고 정지 유지
                self.mecanum_pub.publish(Twist())
                time.sleep(0.01)
                continue
            time_start = time.time()
            try:
                image = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                if not self.is_running:
                    break
                else:
                    continue

            result_image = image.copy()
            if self.start:
                h, w = image.shape[:2]

                # obtain the binary image of the lane
                binary_image = self.lane_detect.get_binary(image)

                twist = Twist()

                # if detecting the zebra crossing, start to slow down
                self.get_logger().info('\033[1;33m%s\033[0m' % self.crosswalk_distance)
                if 70 < self.crosswalk_distance and not self.start_slow_down:  # The robot starts to slow down only when it is close enough to the zebra crossing
                    self.count_crosswalk += 1
                    if self.count_crosswalk == 3:  # judge multiple times to prevent false detection
                        self.count_crosswalk = 0
                        self.start_slow_down = True  # sign for slowing down
                        self.count_slow_down = time.time()  # fixing time for slowing down
                else:  # need to detect continuously, otherwise reset
                    self.count_crosswalk = 0

                # deceleration processing
                if self.start_slow_down:
                    if self.traffic_signs_status is not None:
                        area = abs(self.traffic_signs_status.box[0] - self.traffic_signs_status.box[2]) * abs(self.traffic_signs_status.box[1] - self.traffic_signs_status.box[3])
                        if self.traffic_signs_status.class_name == 'red' and area < 1000:  # If the robot detects a red traffic light, it will stop
                            self.mecanum_pub.publish(Twist())
                            self.stop = True
                        elif self.traffic_signs_status.class_name == 'green':  # If the traffic light is green, the robot will slow down and pass through
                            twist.linear.x = self.slow_down_speed
                            self.stop = False
                    if not self.stop:  # In other cases where the robot is not stopped, slow down the speed and calculate the time needed to pass through the crosswalk. The time needed is equal to the length of the crosswalk divided by the driving speed
                        twist.linear.x = self.slow_down_speed
                        if time.time() - self.count_slow_down > self.crosswalk_length / twist.linear.x:
                            self.start_slow_down = False
                else:
                    twist.linear.x = self.normal_speed  # go straight with normal speed

                # If the robot detects a stop sign and a crosswalk, it will slow down to ensure stable recognition
                if 0 < self.park_x and 135 < self.crosswalk_distance:
                    twist.linear.x = self.slow_down_speed
                    if not self.start_park and 180 < self.crosswalk_distance:  # When the robot is close enough to the crosswalk, it will start parking
                        self.count_park += 1  
                        if self.count_park >= 15:  
                            self.mecanum_pub.publish(Twist())  
                            self.start_park = True
                            self.stop = True
                            threading.Thread(target=self.park_action).start()
                    else:
                        self.count_park = 0  

                # line following processing (Y-기반 턴 트리거)
                result_image, lane_angle, lane_x = self.lane_detect(binary_image, image.copy())

                if not self.stop:
                    # 1) 하단 근접도 계산: 이진 이미지에서 가장 아래쪽(y_max)과 하단 밴드 밀도
                    h, w = image.shape[:2]
                    ys, xs = np.where(binary_image > 0)
                    y_max = int(ys.max()) if ys.size > 0 else -1

                    y_turn_th = int(h * self.y_turn_ratio)   # 예: h*0.75
                    band = binary_image[y_turn_th:, :]
                    band_density = (np.count_nonzero(band) / band.size) if band.size > 0 else 0.0
                    band_dense_enough = band_density > self.band_density_th

                    # --- 턴 트리거: 노란 선이 화면 하단 임계선 아래로 내려오고, 하단 밴드가 충분히 채워졌을 때 ---
                    turn_trigger = (y_max >= y_turn_th) and band_dense_enough

                    if turn_trigger:
                        # 여러 프레임 연속일 때만 턴 모드 진입 (노이즈 방지)
                        self.count_turn += 1
                        if self.count_turn >= self.turn_frames_req and not self.start_turn:
                            self.start_turn = True
                            self.count_turn = 0
                            self.start_turn_time_stamp = time.time()

                            # --- 메카넘: 제자리 90° 우회전 '한 번' 수행 ---
                            if self.machine_type == 'MentorPi_Mecanum':
                                # 피드백(정확) 또는 오픈루프(간단) 중 택1
                                self.rotate_right_90_feedback(source="odom")  # 추천
                                # self.rotate_right_90_in_place(yaw_rate=0.8)

                                # 회전 후 정리/복귀
                                self.mecanum_pub.publish(Twist())
                                self.start_turn = False
                                # (선택) 잠깐 대기
                                time.sleep(0.1)
                                # 바로 return 하면 같은 프레임에서 추가 퍼블리시를 피할 수 있음
                                # return
                                continue

                            else:
                                # Ackermann 등은 기존 강한 회전 로직 유지
                                pass

                        # (Ackermann 등 기존 턴 동작 유지)
                        if self.machine_type != 'MentorPi_Acker':
                            # 메카넘이지만 위에서 90° 이미 했으면 아래 각속도는 굳이 안 줘도 됨
                            # 필요 없다면 주석 처리 가능
                            twist.linear.x = min(twist.linear.x if twist.linear.x > 0 else self.normal_speed,
                                                self.turn_linear_limit)
                            twist.angular.z = -0.45
                        else:
                            twist.linear.x = min(twist.linear.x if twist.linear.x > 0 else self.normal_speed,
                                                self.turn_linear_limit)
                            twist.angular.z = twist.linear.x * math.tan(-0.5061) / 0.145

                        self.mecanum_pub.publish(twist)

                    else:
                        # 트리거 해제 시 카운터 리셋 및 턴 유지 타임아웃 처리
                        self.count_turn = 0
                        if self.start_turn and (time.time() - self.start_turn_time_stamp > self.turn_hold_sec):
                            self.start_turn = False

                        # --- 턴 상황이 아니면 기존 PID 미세 조향 ---
                        if lane_x >= 0 and not self.start_turn:
                            self.pid.SetPoint = 130  # 화면/전처리에 맞게 재보정 가능
                            self.pid.update(lane_x)
                            if self.machine_type != 'MentorPi_Acker':
                                twist.angular.z = common.set_range(self.pid.output, -0.1, 0.1)
                            else:
                                twist.angular.z = twist.linear.x * math.tan(
                                    common.set_range(self.pid.output, -0.1, 0.1)
                                ) / 0.145
                            self.mecanum_pub.publish(twist)
                        else:
                            # Ackermann에서 턴 유지 시간 동안 약한 유지 회전(선택)
                            if self.start_turn and self.machine_type == 'MentorPi_Acker':
                                twist.linear.x = min(twist.linear.x if twist.linear.x > 0 else self.normal_speed,
                                                    self.turn_linear_limit)
                                twist.angular.z = 0.15 * math.tan(-0.5061) / 0.145
                                self.mecanum_pub.publish(twist)
                            else:
                                self.pid.clear()
                else:
                    self.pid.clear()

                if self.objects_info:
                    for i in self.objects_info:
                        box = i.box
                        class_name = i.class_name
                        cls_conf = i.score
                        cls_id = self.classes.index(class_name)
                        color = colors(cls_id, True)
                        plot_one_box(
                            box,
                            result_image,
                            color=color,
                            label="{}:{:.2f}".format(class_name, cls_conf),
                        )

            else:
                time.sleep(0.01)

            
            bgr_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            if self.display:
                self.fps.update()
                bgr_image = self.fps.show_fps(bgr_image)

            
            self.result_publisher.publish(self.bridge.cv2_to_imgmsg(bgr_image, "bgr8"))

           
            time_d = 0.03 - (time.time() - time_start)
            if time_d > 0:
                time.sleep(time_d)
        self.mecanum_pub.publish(Twist())
        rclpy.shutdown()


    # Obtain the target detection result
    def get_object_callback(self, msg):
        self.objects_info = msg.objects
        if self.objects_info == []:  # If it is not recognized, reset the variable
            self.traffic_signs_status = None
            self.crosswalk_distance = 0
        else:
            min_distance = 0
            for i in self.objects_info:
                class_name = i.class_name
                center = (int((i.box[0] + i.box[2])/2), int((i.box[1] + i.box[3])/2))
                
                if class_name == 'crosswalk':  
                    if center[1] > min_distance:  # Obtain recent y-axis pixel coordinate of the crosswalk
                        min_distance = center[1]
                elif class_name == 'right':  # obtain the right turning sign
                    self.count_right += 1
                    self.count_right_miss = 0
                    if self.count_right >= 5:  # If it is detected multiple times, take the right turning sign to true
                        self.turn_right = True
                        self.count_right = 0
                elif class_name == 'park':  # obtain the center coordinate of the parking sign
                    self.park_x = center[0]
                elif class_name == 'red' or class_name == 'green':  # obtain the status of the traffic light
                    self.traffic_signs_status = i
               

            self.get_logger().info('\033[1;32m%s\033[0m' % class_name)
            self.crosswalk_distance = min_distance

def main():
    node = SelfDrivingNode('self_driving')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
 
if __name__ == "__main__":
    main()

    
