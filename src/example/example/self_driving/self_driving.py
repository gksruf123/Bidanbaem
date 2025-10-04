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
from math import atan2, asin, cos, pi, sin
# from app.common import Heart
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
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
        self.is_start = False
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
        self.get_logger().info(f"\033[1;31mself.machine_type: {self.machine_type}\033[0m")
        self.lane_detect = lane_detect.LaneDetector("yellow")

        self.mecanum_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.servo_state_pub = self.create_publisher(SetPWMServoState, 'ros_robot_controller/pwm_servo/set_state', 1)
        self.result_publisher = self.create_publisher(Image, '~/image_result', 1)
        self.odom_subscriber = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)

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
        self.running = False
        self.start = False
        self.wait = True
        self.stop = False
        self.turn = False

        self.detected_cw = False
        self.detected_go = False
        self.detected_right = False
        self.detected_park = False
        self.traffic_signs_status = None

        self.cw_distance = 0
        self.right_distance = 0
        self.sign_distance = 0
        self.fence_distance = 0

        self.count_turn = 0
        self.start_turn = False  # start to turn
        self.start_count = 0
        self.start_dist = 0
        self.basis_start_point_x = 0
        self.basis_start_point_y = 0
        self.turn_count = 0
        self.basis_turn_point = 0

        self.go_linear_x = 1.0
        self.slow_go_linear_x = 0.5
        self.turn_angular_z = -1.0
        self.park_linear_y = -0.5

        self.turn_finish = True
        self.go_finish = True
        self.stop_time = time.time()

        self.object_sub = None
        self.image_sub = None
        self.objects_info = []

        self.start_turn_time_stamp = 0

    def call_start(self):
        req = Trigger.Request()
        future = self.start_yolov5_client.call_async(req)
        future.add_done_callback(self._on_start_response)
    
    def _on_start_response(self, future):
        try:
            result = future.result()
            if result:
                self.get_logger().info(f"[START] 응답: {result.message}")
            else:
                self.get_logger().warn("[START] 응답이 None입니다.")
        except Exception as e:
            self.get_logger().error(f"[START] 서비스 호출 실패: {e}")

    def call_stop(self):
        req = Trigger.Request()
        future = self.stop_yolov5_client.call_async(req)
        future.add_done_callback(self._on_stop_response)

    def _on_stop_response(self, future):
        try:
            result = future.result()
            if result:
                self.get_logger().info(f"[STOP] 응답: {result.message}")
            else:
                self.get_logger().warn("[STOP] 응답이 None입니다.")
        except Exception as e:
            self.get_logger().error(f"[STOP] 서비스 호출 실패: {e}")
    '''
    def call_start(self):
        req = Trigger.Request()
        future = self.start_yolov5_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info(f"Start response: {future.result().message}")
        else:
            self.get_logger().error("Failed to call /yolov5/start")

    def call_stop(self):
        req = Trigger.Request()
        future = self.stop_yolov5_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info(f"Stop response: {future.result().message}")
        else:
            self.get_logger().error("Failed to call /yolov5/stop")
    '''
    def get_node_state(self, request, response):
        response.success = True
        return response

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def quat_to_yaw(self, q):
        # q: geometry_msgs/Quaternion
        # 표준 변환 (Z축 회전만 고려)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return atan2(siny_cosp, cosy_cosp)  # 라디안
    
    def odom_callback(self, msg: Odometry):
        self.position_x = msg.pose.pose.position.x
        self.position_y = msg.pose.pose.position.y
        self.yaw = self.quat_to_yaw(msg.pose.pose.orientation)
        self.degree = self.yaw * 180.0 / pi

    def enter_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "self driving enter")
        with self.lock:
            self.running = False
            camera = 'depth_cam'#self.get_parameter('depth_camera_name').value
            self.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image' , self.image_callback, 1)
            self.create_subscription(ObjectsInfo, '/yolov5_ros2/object_detect', self.get_object_callback, 1)
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
                    self.image_sub.unregister()
                if self.object_sub is not None:
                    self.object_sub.unregister()
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
            self.running = request.data
            if not self.running:
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

    def main(self):
        while self.is_running:
            time_start = time.time()
            try:
                image = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                if not self.is_running:
                    break
                else:
                    continue

            result_image = image.copy()
            if self.running:
                h, w = image.shape[:2]

                # obtain the binary image of the lane
                binary_image = self.lane_detect.get_binary(image)

                twist = Twist()

                if self.is_start: # 맨 처음 'green' 감지
                    # line following processing
                    result_image, left_lane_x, right_lane_x, mid_lane_x, turn_right = self.lane_detect(binary_image, image.copy())  # the coordinate of the line while the robot is in the middle of the lane
                    self.get_logger().info(f"\033[1;32m\nleft_lane_x: {left_lane_x}\tright_lane_x: {right_lane_x}\tmid_lane_x: {mid_lane_x}\033[0m")

                    if self.go_finish and self.turn_finish:
                        if self.wait:
                            if time.time() - self.stop_time > 5.0:
                                if self.traffic_signs_status != 'red':
                                    self.wait = False
                                    self.start = True
                                    self.call_stop()
                                    self.go_finish = False
                        elif self.start:
                            self.start_count = 0
                            if self.detected_park:
                                self.park = True
                                self.start = False
                                self.go_finish = False
                            elif turn_right:
                                self.start = False
                                self.turn = True
                                self.turn_finish = False
                            # 조금 갔다가 우회전 하는 것 구현
                            # elif (self.right_distance != -1 and self.right_distance < 300):
                            else:
                                self.stop_time = time.time()
                                self.wait = True
                                self.call_start()
                                self.start = False
                        elif self.turn:
                            self.turn_count = 0
                            self.wait = True
                            self.call_start()
                            self.turn = False
                        elif self.stop:
                            self.stop = False

                    if self.wait:
                        self.get_logger().info("\033[1;31mstate: **wait**\033[0m")
                        # self.detected_cw = False
                        # self.detected_go = False
                        # self.detected_right = False
                        # self.detected_park = False
                        # self.traffic_signs_status = None
                        self.mecanum_pub.publish(Twist())
                        continue

                    if self.start: # odom을 추가하여
                        self.get_logger().info("\033[1;31mstate: **start**\033[0m")
                        twist.linear.x = self.slow_go_linear_x
                        if self.start_count == 0:
                            self.get_logger().info(f"\033[1;31m1. self.detected_cw: {self.detected_cw}\033[0m")
                            self.get_logger().info(f"\033[1;31m2. detect sign: {self.traffic_signs_status != None or self.detected_go == True or self.detected_right == True}\033[0m")
                            self.get_logger().info(f"\033[1;31m3. self.sign_distance > 400: {self.sign_distance > 400}\033[0m")
                            if self.detected_cw and (self.traffic_signs_status != None or self.detected_go == True or self.detected_right == True) and self.sign_distance > 400:
                                self.start_dist = self.cw_distance
                                self.get_logger().info(f"\033[1;31mcross_walk distance: {self.start_dist}\033[0m")
                            else:
                                self.start_dist = self.fence_distance
                                self.get_logger().info(f"\033[1;31mfence distance: {self.start_dist}\033[0m")

                            self.start_count += 1
                            self.basis_start_point_x, self.basis_start_point_y = self.position_x, self.position_y

                        self.get_logger().info(f"\033[1;31modom: {max(abs(self.position_x - self.basis_start_point_x), abs(self.position_y - self.basis_start_point_y)) * 1000}, dist: {self.start_dist}\033[0m")
                        if max(abs(self.position_x - self.basis_start_point_x), abs(self.position_y - self.basis_start_point_y)) * 1000 > self.start_dist - 100:   # odom(m)과 distance(mm)의 단위를 고려하지 않음
                            self.get_logger().info(f"\033[1;31m**go finish**\033[0m")
                            self.go_finish = True
                            # self.detected_cw = False
                            # self.detected_go = False
                            # self.detected_right = False
                            # self.detected_park = False
                            # self.traffic_signs_status = None
                            self.mecanum_pub.publish(Twist())
                            continue
                        if left_lane_x >= 0 and not self.stop:
                            self.get_logger().info(f"\033[1;31m**left_lane_x: {left_lane_x}**\033[0m")
                            if mid_lane_x == -1:
                                self.pid.SetPoint = 180  # the coordinate of the line while the robot is in the middle of the lane
                                self.pid.update(left_lane_x)
                            else:
                                self.pid.SetPoint = 230  # the coordinate of the line while the robot is in the middle of the lane
                                self.pid.update(mid_lane_x)
                            if self.machine_type != 'MentorPi_Acker':
                                self.get_logger().info(f"\033[1;31m**Adjust Line**\033[0m")
                                twist.angular.z = common.set_range(self.pid.output, -0.15, 0.15)
                            else:
                                twist.angular.z = twist.linear.x * math.tan(common.set_range(self.pid.output, -0.1, 0.1)) / 0.145

                    if self.turn:
                        self.get_logger().info("\033[1;31mstate: **turn**\033[0m")
                        twist.linear.x = 0.0
                        twist.angular.z = self.turn_angular_z
                        if self.turn_count == 0:
                            self.turn_count += 1
                            self.basis_turn_point = self.degree      # 현재 기준 시작 각도 지정

                        if abs(self.basis_turn_point - self.degree) > 85:
                            self.turn_finish = True
                            # self.detected_cw = False
                            # self.detected_go = False
                            # self.detected_right = False
                            # self.detected_park = False
                            # self.traffic_signs_status = None
                            self.mecanum_pub.publish(Twist())
                            continue


                    # self.detected_cw = False
                    # self.detected_go = False
                    # self.detected_right = False
                    # self.detected_park = False
                    # self.traffic_signs_status = None

                    self.get_logger().info(f"\033[1;32mtwist.linear.x: {twist.linear.x}\033[0m")
                    self.get_logger().info(f"\033[1;32mtwist.angular.z: {twist.angular.z}\033[0m")
                    self.mecanum_pub.publish(twist)
                    # self.mecanum_pub.publish(Twist())

                
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
                    self.mecanum_pub.publish(Twist())

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
            self.cw_distance = -1
            self.right_distance = -1
            self.sign_distance = -1

            self.detected_cw = False
            self.detected_go = False
            self.detected_right = False
            self.detected_park = False
        else:
            self.cw_distance = 10000
            for i in self.objects_info:
                class_name = i.class_name
                center = (int((i.box[0] + i.box[2])/2), int((i.box[1] + i.box[3])/2))
                obj_distance = i.distance
                self.fence_distance = i.fence_distance
                
                if class_name == 'crosswalk':
                    self.detected_cw = True
                    if obj_distance < self.cw_distance:  # Obtain recent y-axis pixel coordinate of the crosswalk
                        self.cw_distance = obj_distance
                elif class_name == 'go':  # obtain the go sign
                    self.detected_go = True
                    self.sign_distance = obj_distance
                elif class_name == 'right':  # obtain the right turning sign
                    self.detected_cw = True
                    self.right_distance = obj_distance
                    self.sign_distance = obj_distance
                elif class_name == 'park':  # obtain the center coordinate of the parking sign
                    self.detected_park = True
                elif class_name == 'red':
                    self.traffic_signs_status = 'red'
                    self.sign_distance = obj_distance
                elif class_name == 'green':  # obtain the status of the traffic light
                    self.traffic_signs_status = 'green'
                    self.sign_distance = obj_distance
                    self.is_start = True
                    # self.get_logger().info(f"\033[1;31m**detected {class_name}**\033[0m")

                # if class_name == 'crosswalk':
                #     self.get_logger().info(f"\033[1;31m{class_name}: {cw_distance}\033[0m")
                # else:
                #     self.get_logger().info(f"\033[1;32m{class_name}: {cw_distance}\033[0m")
               

                # self.get_logger().info('\033[1;32m%s\033[0m' % class_name)

def main():
    node = SelfDrivingNode('self_driving')
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
 
if __name__ == "__main__":
    main()

    
