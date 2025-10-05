from math import frexp
from traceback import print_tb
import torch
from yolov5 import YOLOv5
import rclpy
import yolov5_ros2.fps as fps
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import ParameterDescriptor
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose, Detection2D
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import yaml
#from sdk import common

from yolov5_ros2.cv_tool import px2xy
import os
from interfaces.msg import ObjectInfo, ObjectsInfo

from std_srvs.srv import Trigger

# Get the ROS distribution version and set the shared directory for YoloV5 configuration files.
ros_distribution = os.environ.get("ROS_DISTRO")
package_share_directory = get_package_share_directory('yolov5_ros2')

# Create a ROS 2 Node class YoloV5Ros2.
class YoloV5Ros2(Node):
    def __init__(self):
        super().__init__('yolov5_ros2')
        self.get_logger().info(f"Current ROS 2 distribution: {ros_distribution}")
        self.fps = fps.FPS()

        self.declare_parameter("device", "cpu", ParameterDescriptor(
            name="device", description="Compute device selection, default: cpu, options: cuda:0"))

        self.declare_parameter("model", "yolov5s", ParameterDescriptor(
            name="model", description="Default model selection: yolov5s"))

        self.declare_parameter("image_topic", "/ascamera/camera_publisher/rgb0/image", ParameterDescriptor(
            name="image_topic", description="Image topic, default: /ascamera/camera_publisher/rgb0/image"))

        self.declare_parameter("show_result", False, ParameterDescriptor(
            name="show_result", description="Whether to display detection results, default: False"))

        self.declare_parameter("pub_result_img", False, ParameterDescriptor(
            name="pub_result_img", description="Whether to publish detection result images, default: False"))

        self.create_service(Trigger, '/yolov5/start', self.start_srv_callback)
        self.create_service(Trigger, '/yolov5/stop', self.stop_srv_callback) 
        self.create_service(Trigger, '~/init_finish', self.get_node_state)

        # Load the model.
        model_path = package_share_directory + "/config/new_model.onnx"
        device = self.get_parameter('device').value
        self.yolov5 = YOLOv5(model_path=model_path, device=device)

        # Create publishers.
        self.yolo_result_pub = self.create_publisher(Detection2DArray, "yolo_result", 10)
        self.result_msg = Detection2DArray()
        self.object_pub = self.create_publisher(ObjectsInfo, '~/object_detect', 1)
        self.result_img_pub = self.create_publisher(Image, "result_img", 10)

        # Create an image subscriber with the updated topic.
        image_topic = self.get_parameter('image_topic').value
        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, 10)

        # Image format conversion (using cv_bridge).
        self.bridge = CvBridge()

        self.show_result = self.get_parameter('show_result').value
        self.pub_result_img = self.get_parameter('pub_result_img').value

        self.start = False

    def get_node_state(self, request, response):
        response.success = True
        return response

    def start_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "start yolov5 detect")
        self.start = True
        response.success = True
        response.message = "start"
        return response

    def stop_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "stop yolov5 detect")
        self.start = False
        response.success = True
        response.message = "stop"
        return response

    def image_callback(self, msg: Image):
        if hasattr(self, "start") and not self.start:
            return

        # 1) ROS Image → RGB ndarray
        image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        orig_h, orig_w = image.shape[:2]

        # 2) 모델 기대 크기로 리사이즈 (간단 버전: 비율 무시하고 정사각)
        im_in = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)

        # 3) 추론
        detect_result = self.yolov5.predict(im_in)

        # 4) 결과 파싱
        self.result_msg.detections.clear()

        # ▼ [P2] 입력 이미지의 헤더 그대로 승계 (frame_id, stamp 유지)
        self.result_msg.header = msg.header

        predictions = detect_result.pred[0]

        # ▼ 텐서/리스트 어떤 형식이든 안전하게 변환
        if hasattr(predictions, "cpu"):
            predictions = predictions.cpu().numpy()

        # ▼ [P0] 빈 결과면: 두 토픽 모두 "빈 메시지" 발행 후 종료
        if predictions is None or len(predictions) == 0:
            # 빈 Detection2DArray 발행
            self.yolo_result_pub.publish(self.result_msg)

            # 빈 ObjectsInfo 발행
            object_msg = ObjectsInfo()
            object_msg.objects = []
            self.object_pub.publish(object_msg)
            return

        boxes = predictions[:, :4]  # x1,y1,x2,y2 (이 좌표들은 640x640 기준)
        scores = predictions[:, 4]
        categories = predictions[:, 5]


        # 5) 원본 해상도로 복원(스케일백)
        sx = orig_w / 640.0
        sy = orig_h / 640.0

        objects_info = []  # 프레임당 한 번만 퍼블리시하려면 루프 바깥에서 모으기
        for i in range(len(categories)):
            x1, y1, x2, y2 = boxes[i]
            # 스케일백
            x1 = int(x1 * sx); x2 = int(x2 * sx)
            y1 = int(y1 * sy); y2 = int(y2 * sy)

            name = detect_result.names[int(categories[i])]

            detection2d = Detection2D()
            if (os.environ.get("ROS_DISTRO") or "").lower().startswith("galactic"):
                detection2d.bbox.center.x = (x1 + x2) / 2.0
                detection2d.bbox.center.y = (y1 + y2) / 2.0
            else:
                detection2d.bbox.center.position.x = (x1 + x2) / 2.0
                detection2d.bbox.center.position.y = (y1 + y2) / 2.0
            detection2d.bbox.size_x = float(x2 - x1)
            detection2d.bbox.size_y = float(y2 - y1)

            obj_pose = ObjectHypothesisWithPose()
            obj_pose.hypothesis.class_id = name
            obj_pose.hypothesis.score = float(scores[i])
            detection2d.results.append(obj_pose)
            self.result_msg.detections.append(detection2d)

            # (옵션) 드로잉/ObjectsInfo 구성
            if self.show_result or self.pub_result_img:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(image, f"{name}:{obj_pose.hypothesis.score:.2f}", (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            oi = ObjectInfo()
            oi.class_name = name
            oi.box = [x1, y1, x2, y2]
            oi.score = round(float(scores[i]), 2)
            oi.width = orig_w
            oi.height = orig_h
            objects_info.append(oi)

        # 커스텀 메시지 퍼블리시(프레임당 한 번)
        object_msg = ObjectsInfo()
        object_msg.objects = objects_info
        self.object_pub.publish(object_msg)

        # 결과 발행/표시
        if self.show_result:
            self.fps.update()
            image = self.fps.show_fps(image)
            cv2.imshow('result', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        if self.pub_result_img:
            result_img_msg = self.bridge.cv2_to_imgmsg(image, encoding="rgb8")
            result_img_msg.header = msg.header
            self.result_img_pub.publish(result_img_msg)

        self.yolo_result_pub.publish(self.result_msg)

  

def main():
    rclpy.init()
    rclpy.spin(YoloV5Ros2())
    rclpy.shutdown()

if __name__ == "__main__":
    main()

