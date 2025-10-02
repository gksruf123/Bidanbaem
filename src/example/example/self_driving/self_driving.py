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
        self.machine_type = os.environ.get('MACHINE_TYPE')
        self.lane_detect = lane_detect.LaneDetector("yellow")

        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.servo_state_pub = self.create_publisher(SetPWMServoState, 'ros_robot_controller/pwm_servo/set_state', 1)
        self.result_publisher = self.create_publisher(Image, '~/image_result', 1)

        self.create_service(Trigger, '~/enter', self.enter_srv_callback)
        self.create_service(Trigger, '~/exit', self.exit_srv_callback)
        self.create_service(SetBool, '~/set_running', self.set_running_srv_callback)

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
        
        if 1:
            self.display = True
            self.enter_srv_callback(Trigger.Request(), Trigger.Response())
            request = SetBool.Request()
            request.data = True
            self.set_running_srv_callback(request, SetBool.Response())

        threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.get_logger().info('\033[1;32m%s\033[0m' % 'start')

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


    def param_init(self):
        self.start = False
        self.enter = False
        self.right = True

        self.have_turn_right = False
        self.detect_turn_right = False
        self.detect_far_lane = False
        self.park_x = -1

        self.start_turn_time_stamp = 0
        self.count_turn = 0
        self.start_turn = False

        self.count_right = 0
        self.count_right_miss = 0
        self.turn_right = False

        self.last_park_detect = False
        self.count_park = 0  
        self.stop = False
        self.start_park = False

        self.count_crosswalk = 0
        self.crosswalk_distance = 0
        self.crosswalk_length = 0.4

        self.start_slow_down = False
        self.normal_speed = 0.1
        self.slow_down_speed = 0.1

        self.traffic_signs_status = None
        self.red_loss_count = 0

        self.object_sub = None
        self.image_sub = None
        self.objects_info = []

        # 🔽 신호등 출발 대기
        self.await_green_start = True   # 처음엔 green 신호 대기
        self.green_stable_frames = 0
        self.GREEN_REQUIRED = 5

        # 🔽 횡단보도 정차
        self.crosswalk_stop_pending = False
        self.crosswalk_stop_active = False
        self.crosswalk_last_time = 0.0
        self.crosswalk_stop_count = 0
        self.CROSSWALK_STOP_SEC = 3.0
        self.MAX_CROSSWALK_STOPS = 4

                # 주차 관련
        self.park_x = -1
        self.start_park = False      # 주차 시작 여부
        self.park_detected = False   # park 표지판을 봤는지
        self.park_detect_time = 0.0  # park 감지 시각
        self.PARK_DELAY = 2.0        # park 감지 후 몇 초 뒤에 주차 시작할지

        
    


    def get_node_state(self, request, response):
        response.success = True
        return response

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def enter_srv_callback(self, request, response):
        self.get_logger().info("self driving enter")
        with self.lock:
            self.start = False
            self.image_sub = self.create_subscription(
                Image,
                '/ascamera/camera_publisher/rgb0/image',
                self.image_callback, 1
            )
            self.object_sub = self.create_subscription(
                ObjectsInfo,
                '/yolov5_ros2/object_detect',
                self.get_object_callback, 1
            )
            self.mecanum_pub.publish(Twist())
            self.enter = True
        response.success = True
        response.message = "enter"
        return response

    def exit_srv_callback(self, request, response):
        self.get_logger().info("self driving exit")
        with self.lock:
            try:
                if self.image_sub is not None:
                    self.destroy_subscription(self.image_sub)   # ✅ 수정
                if self.object_sub is not None:
                    self.destroy_subscription(self.object_sub) # ✅ 수정
            except Exception as e:
                self.get_logger().info(str(e))
            self.mecanum_pub.publish(Twist())
        self.param_init()
        response.success = True
        response.message = "exit"
        return response

    def set_running_srv_callback(self, request, response):
        self.get_logger().info("set_running")
        with self.lock:
            self.start = request.data
            if not self.start:
                self.mecanum_pub.publish(Twist())
        response.success = True
        response.message = "set_running"
        return response

    def shutdown(self, signum, frame):
        self.is_running = False

    def image_callback(self, ros_image):
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        if self.image_queue.full():
            self.image_queue.get()
        self.image_queue.put(rgb_image)

    def main(self):
        while self.is_running:
            time_start = time.time()
            try:
                image = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            result_image = image.copy()

            if self.start:
                h, w = image.shape[:2]
                binary_image = self.lane_detect.get_binary(image)
                twist = Twist()

                # 1) 출발 전 green 신호 대기
                if self.await_green_start:
                    if (self.traffic_signs_status and 
                        self.traffic_signs_status.class_name == "green"):
                        self.green_stable_frames += 1
                        if self.green_stable_frames >= self.GREEN_REQUIRED:
                            self.await_green_start = False
                            self.start = True
                            self.get_logger().info("[START] Green light detected → 출발!")
                    else:
                        self.green_stable_frames = 0

                    self.mecanum_pub.publish(Twist())
                    continue

                # 2) 횡단보도 정차
                if self.crosswalk_stop_pending and not self.crosswalk_stop_active:
                    self.crosswalk_stop_active = True
                    self.crosswalk_stop_pending = False
                    self.crosswalk_last_time = time.time()
                    self.crosswalk_stop_count += 1
                    self.mecanum_pub.publish(Twist())
                    self.get_logger().info(f"[CROSSWALK] Stop #{self.crosswalk_stop_count}")

                if self.crosswalk_stop_active:
                    if time.time() - self.crosswalk_last_time < self.CROSSWALK_STOP_SEC:
                        self.mecanum_pub.publish(Twist())
                        continue
                    else:
                        self.crosswalk_stop_active = False
                        self.get_logger().info("[CROSSWALK] Resume driving")

                # 3) 라인트레이싱
                try:
                    result_image, lane_angle, lane_x = self.lane_detect(binary_image, image.copy())
                except TypeError:
                    M = cv2.moments(binary_image)
                    lane_x = -1
                    if M["m00"] > 0:
                        lane_x = int(M["m10"] / M["m00"])
                    lane_angle = 0

                if lane_x >= 0 and not self.stop:
                    center_x = w // 2
                    offset = lane_x - center_x
                    self.pid.SetPoint = center_x
                    self.pid.update(lane_x)

                    if self.machine_type != 'MentorPi_Acker':
                        twist.angular.z = common.set_range(self.pid.output, -0.3, 0.3)
                    else:
                        twist.angular.z = twist.linear.x * math.tan(
                            common.set_range(self.pid.output, -0.3, 0.3)
                        ) / 0.145

                    if abs(offset) > 40:
                        twist.linear.x = 0.3   # 코너 감속
                    else:
                        twist.linear.x = 0.8   # 직선 가속
                    self.mecanum_pub.publish(twist)
                else:
                    self.pid.clear()
                    self.mecanum_pub.publish(Twist())

                # 4) 주차 처리
                if self.park_detected and not self.start_park:
                    if time.time() - self.park_detect_time > self.PARK_DELAY:
                        self.start_park = True
                        self.get_logger().info("[PARK] Parking action start")

                        self.mecanum_pub.publish(Twist())  # 잠시 정지
                        time.sleep(0.5)

                        self.park_action()
                        self.get_logger().info("[PARK] Parking completed")

                        self.is_running = False
                        self.mecanum_pub.publish(Twist())
                        continue

            # FPS & 결과 이미지 퍼블리시
            bgr_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            if self.display:
                self.fps.update()
                bgr_image = self.fps.show_fps(bgr_image)

            self.result_publisher.publish(self.bridge.cv2_to_imgmsg(bgr_image, "bgr8"))

            # 루프 주기 맞추기
            time_d = 0.03 - (time.time() - time_start)
            if time_d > 0:
                time.sleep(time_d)

        self.mecanum_pub.publish(Twist())
        rclpy.shutdown()




    def get_object_callback(self, msg):
        self.objects_info = msg.objects
        if self.objects_info == []:
            self.traffic_signs_status = None
            self.crosswalk_distance = 0
        else:
            min_distance = 0
            for i in self.objects_info:
                class_name = i.class_name
                center = (int((i.box[0] + i.box[2])/2), int((i.box[1] + i.box[3])/2))
                if class_name == 'crosswalk':
                    if center[1] > min_distance:
                        min_distance = center[1]
                elif class_name == 'right':
                    self.count_right += 1
                    self.count_right_miss = 0
                    if self.count_right >= 5:
                        self.turn_right = True
                        self.count_right = 0
                elif class_name == 'park':
                    self.park_x = center[0]
                    if not self.park_detected:  # 처음 감지될 때만 기록
                        self.park_detected = True
                        self.park_detect_time = time.time()
                        self.get_logger().info("[PARK] Park sign detected, preparing to park...")

                elif class_name in ['red', 'green']:
                    self.traffic_signs_status = i

            self.get_logger().info(class_name)
            self.crosswalk_distance = min_distance

def main():
    node = SelfDrivingNode('self_driving')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
 
if __name__ == "__main__":
    main()
