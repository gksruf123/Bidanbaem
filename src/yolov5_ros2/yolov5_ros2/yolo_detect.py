import os
import cv2
import json
import numpy as np
import onnxruntime as ort
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import message_filters
from interfaces.msg import ObjectInfo, ObjectsInfo
from std_srvs.srv import Trigger
from yolov5_ros2 import fps as fps

from yolov5_ros2.cv_tool import px2xy

# Get ROS distribution and package path
ros_distribution = os.environ.get("ROS_DISTRO")
package_share_directory = os.path.join(os.path.dirname(__file__), "..", "config")


class YoloV5Ros2(Node):
    def __init__(self):
        super().__init__('yolov5_ros2')
        self.get_logger().info(f"Current ROS 2 distribution: {ros_distribution}")
        self.fps = fps.FPS()

        # Parameters
        self.declare_parameter("device", "cpu", ParameterDescriptor(
            name="device", description="Compute device selection, default: cpu, options: cpu/cuda:0"
        ))
        self.declare_parameter("model", "new_model", ParameterDescriptor(
            name="model", description="ONNX model name without extension"
        ))
        self.declare_parameter("image_topic", "/ascamera/camera_publisher/rgb0/image", ParameterDescriptor(
            name="image_topic", description="RGB image topic"
        ))
        self.declare_parameter("show_result", False, ParameterDescriptor(
            name="show_result", description="Display results"
        ))
        self.declare_parameter("pub_result_img", False, ParameterDescriptor(
            name="pub_result_img", description="Publish result image"
        ))

        # Services
        self.create_service(Trigger, '/yolov5/start', self.start_srv_callback)
        self.create_service(Trigger, '/yolov5/stop', self.stop_srv_callback)
        self.create_service(Trigger, '~/init_finish', self.get_node_state)

        # Load ONNX model
        model_name = self.get_parameter('model').value
        model_path = os.path.join(package_share_directory, f"{model_name}.onnx")
        label_path = os.path.join(package_share_directory, f"{model_name}.names.json")

        self.device = self.get_parameter('device').value
        providers = ['CUDAExecutionProvider'] if 'cuda' in self.device else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)

        with open(label_path, "r") as f:
            self.label_dict = json.load(f)

        # Publishers
        self.yolo_result_pub = self.create_publisher(Detection2DArray, "yolo_result", 10)
        self.result_msg = Detection2DArray()
        self.object_pub = self.create_publisher(ObjectsInfo, '~/object_detect', 1)
        self.result_img_pub = self.create_publisher(Image, "result_img", 10)

        # Subscribers (RGB + Depth)
        rgb_sub = message_filters.Subscriber(self, Image, '/ascamera/camera_publisher/rgb0/image')
        depth_sub = message_filters.Subscriber(self, Image, '/ascamera/camera_publisher/depth0/image_raw')
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.05)
        ts.registerCallback(self.image_callback)

        # Bridge
        self.bridge = CvBridge()

        self.show_result = self.get_parameter('show_result').value
        self.pub_result_img = self.get_parameter('pub_result_img').value

    def get_node_state(self, request, response):
        response.success = True
        return response

    def start_srv_callback(self, request, response):
        self.get_logger().info("Start YOLOv5 detection")
        self.start = True
        response.success = True
        response.message = "start"
        return response

    def stop_srv_callback(self, request, response):
        self.get_logger().info("Stop YOLOv5 detection")
        self.start = False
        response.success = True
        response.message = "stop"
        return response

    def preprocess(self, image):
        """Resize and normalize the image for ONNX input"""
        img_resized = cv2.resize(image, (640, 640))
        img_rgb = img_resized.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_rgb, (2, 0, 1))  # HWC -> CHW
        img_batch = np.expand_dims(img_transposed, axis=0)
        return img_batch

    def image_callback(self, rgb_msg, depth_msg):
        # 1. Convert ROS Image to OpenCV
        image = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')

        # 2. Depth inpainting (0 → 주변값)
        depth_uint16 = depth.astype(np.uint16)
        mask = (depth_uint16 == 0).astype('uint8')
        depth_inpaint = cv2.inpaint(depth_uint16, mask, 2, cv2.INPAINT_TELEA)
        depth = depth_inpaint.astype(np.float32)

        # 3. ONNX inference
        detect_result = self.yolov5.predict(image)  # detect_result: num_detections x 7 (x1, y1, x2, y2, conf, cls_conf, cls_idx)
        predictions = detect_result  # ONNX output 직접 사용

        self.result_msg.detections.clear()
        self.result_msg.header.frame_id = "camera"
        self.result_msg.header.stamp = self.get_clock().now().to_msg()

        objects_info = []
        for pred in predictions:
            # pred: [x1, y1, x2, y2, conf, cls_conf, cls_idx]
            if len(pred) == 7:
                x1, y1, x2, y2, conf, _, cls_idx = pred
            else:
                # 안전하게 길이가 달라져도 처리
                x1, y1, x2, y2, conf, cls_idx = pred[:6]

            name = detect_result.names[int(cls_idx)]

            # BBox 계산
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            # Depth
            box_distance = depth[int(center_y), int(center_x)]
            fence_distance = depth[10, 320]

            # Detection2D msg
            detection2d = Detection2D()
            if ros_distribution == 'galactic':
                detection2d.bbox.center.x = center_x
                detection2d.bbox.center.y = center_y
            else:
                detection2d.bbox.center.position.x = center_x
                detection2d.bbox.center.position.y = center_y

            detection2d.bbox.size_x = float(x2 - x1)
            detection2d.bbox.size_y = float(y2 - y1)

            obj_pose = ObjectHypothesisWithPose()
            obj_pose.hypothesis.class_id = name
            obj_pose.hypothesis.score = float(conf)
            detection2d.results.append(obj_pose)
            self.result_msg.detections.append(detection2d)

            # Draw results
            if self.show_result or self.pub_result_img:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{name}:{conf:.2f}", (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # ObjectInfo msg
            h, w = image.shape[:2]
            object_info = ObjectInfo()
            object_info.class_name = name
            object_info.box = [x1, y1, x2, y2]
            object_info.score = round(float(conf), 2)
            object_info.width = w
            object_info.height = h
            object_info.distance = int(box_distance)
            object_info.fence_distance = int(fence_distance)
            objects_info.append(object_info)

        object_msg = ObjectsInfo()
        object_msg.objects = objects_info
        self.object_pub.publish(object_msg)

        # Display & publish
        if self.show_result:
            self.fps.update()
            image = self.fps.show_fps(image)
            cv2.imshow('result', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        if self.pub_result_img:
            result_img_msg = self.bridge.cv2_to_imgmsg(image, encoding="rgb8")
            result_img_msg.header = rgb_msg.header
            self.result_img_pub.publish(result_img_msg)

        if len(predictions) > 0:
            self.yolo_result_pub.publish(self.result_msg)



def main():
    rclpy.init()
    rclpy.spin(YoloV5Ros2())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
