import os
import json
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose, Detection2D
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import message_filters
import onnxruntime

from interfaces.msg import ObjectInfo, ObjectsInfo
import yolov5_ros2.fps as fps

ros_distribution = os.environ.get("ROS_DISTRO")
package_share_directory = os.path.join(os.getcwd(), 'install/yolov5_ros2/share/yolov5_ros2')  # 필요시 경로 수정

# -----------------------------
# ONNX용 YOLOv5 래퍼
# -----------------------------
class YoloV5ONNX:
    def __init__(self, model_path, class_names_path, device='cpu'):
        self.device = device
        self.session = onnxruntime.InferenceSession(model_path)
        with open(class_names_path, 'r') as f:
            self.names = json.load(f)

    def predict(self, image):
        """
        image: np.array (H,W,C) RGB
        return: np.array [num_detections, 7] -> x1,y1,x2,y2,conf,cls_conf,cls_idx
        """
        img = cv2.resize(image, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))[None]  # (1,3,H,W)

        outputs = self.session.run(None, {self.session.get_inputs()[0].name: img})
        return outputs[0]  # shape: [num_detections, 7]

# -----------------------------
# ROS2 노드
# -----------------------------
class YoloV5Ros2(Node):
    def __init__(self):
        super().__init__('yolov5_ros2')
        self.get_logger().info(f"Current ROS 2 distribution: {ros_distribution}")
        self.fps = fps.FPS()

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter("device", "cpu", ParameterDescriptor(name="device", description="cpu or cuda"))
        self.declare_parameter("model", "new_model", ParameterDescriptor(name="model", description="ONNX model name"))
        self.declare_parameter("show_result", False, ParameterDescriptor(name="show_result", description="Display image"))
        self.declare_parameter("pub_result_img", False, ParameterDescriptor(name="pub_result_img", description="Publish image"))

        self.show_result = self.get_parameter('show_result').value
        self.pub_result_img = self.get_parameter('pub_result_img').value

        # -----------------------------
        # Load ONNX model
        # -----------------------------
        model_name = self.get_parameter('model').value
        model_path = os.path.join(package_share_directory, 'config', model_name + ".onnx")
        label_path = os.path.join(package_share_directory, 'config', model_name + ".names.json")
        device = self.get_parameter('device').value

        self.yolov5 = YoloV5ONNX(model_path, label_path, device)

        # -----------------------------
        # Publishers & CvBridge
        # -----------------------------
        self.bridge = CvBridge()
        self.yolo_result_pub = self.create_publisher(Detection2DArray, "yolo_result", 10)
        self.result_msg = Detection2DArray()
        self.object_pub = self.create_publisher(ObjectsInfo, '~/object_detect', 1)
        self.result_img_pub = self.create_publisher(Image, "result_img", 10)

        # -----------------------------
        # Image subscriber
        # -----------------------------
        rgb_sub = message_filters.Subscriber(self, Image, '/ascamera/camera_publisher/rgb0/image')
        depth_sub = message_filters.Subscriber(self, Image, '/ascamera/camera_publisher/depth0/image_raw')
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.05)
        ts.registerCallback(self.image_callback)

        # -----------------------------
        # Services
        # -----------------------------
        self.create_service(Trigger, '/yolov5/start', self.start_srv_callback)
        self.create_service(Trigger, '/yolov5/stop', self.stop_srv_callback) 
        self.create_service(Trigger, '~/init_finish', self.get_node_state)
        self.start = True

    # -----------------------------
    # Service callbacks
    # -----------------------------
    def get_node_state(self, request, response):
        response.success = True
        return response

    def start_srv_callback(self, request, response):
        self.get_logger().info("start yolov5 detect")
        self.start = True
        response.success = True
        response.message = "start"
        return response

    def stop_srv_callback(self, request, response):
        self.get_logger().info("stop yolov5 detect")
        self.start = False
        response.success = True
        response.message = "stop"
        return response

    # -----------------------------
    # Image callback
    # -----------------------------
    def image_callback(self, rgb_msg, depth_msg):
        if not self.start:
            return

        image = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')

        # 깊이 보정
        depth_uint16 = depth.astype(np.uint16)
        mask = (depth_uint16 == 0).astype('uint8')
        depth_inpaint = cv2.inpaint(depth_uint16, mask, 2, cv2.INPAINT_TELEA)
        depth = depth_inpaint.astype(np.float32)

        # -----------------------------
        # ONNX 추론
        # -----------------------------
        preds = self.yolov5.predict(image)  # shape: [num_detections,7]

        self.result_msg.detections.clear()
        self.result_msg.header.frame_id = "camera"
        self.result_msg.header.stamp = self.get_clock().now().to_msg()
        objects_info = []

        for pred in preds:
            # pred = [x1,y1,x2,y2,conf,cls_conf,cls_idx]
            x1, y1, x2, y2, conf, _, cls_idx = pred
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            name = self.yolov5.names[int(cls_idx)]
            score = float(conf)

            # Detection2D
            detection2d = Detection2D()
            if ros_distribution == 'galactic':
                detection2d.bbox.center.x = (x1+x2)/2.0
                detection2d.bbox.center.y = (y1+y2)/2.0
            else:
                detection2d.bbox.center.position.x = (x1+x2)/2.0
                detection2d.bbox.center.position.y = (y1+y2)/2.0
            detection2d.bbox.size_x = float(x2-x1)
            detection2d.bbox.size_y = float(y2-y1)
            obj_pose = ObjectHypothesisWithPose()
            obj_pose.hypothesis.class_id = name
            obj_pose.hypothesis.score = score
            detection2d.results.append(obj_pose)
            self.result_msg.detections.append(detection2d)

            # ObjectInfo
            h, w = image.shape[:2]
            box_distance = depth[int((y1+y2)/2), int((x1+x2)/2)]
            fence_distance = depth[10, w//2]
            obj_info = ObjectInfo()
            obj_info.class_name = name
            obj_info.box = [x1, y1, x2, y2]
            obj_info.score = round(score, 2)
            obj_info.width = w
            obj_info.height = h
            obj_info.distance = int(box_distance)
            obj_info.fence_distance = int(fence_distance)
            objects_info.append(obj_info)

            # Draw
            if self.show_result or self.pub_result_img:
                cv2.rectangle(image, (x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(image,f"{name}:{score:.2f}",(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

        # Publish
        object_msg = ObjectsInfo()
        object_msg.objects = objects_info
        self.object_pub.publish(object_msg)
        if len(preds)>0:
            self.yolo_result_pub.publish(self.result_msg)

        if self.show_result:
            self.fps.update()
            image = self.fps.show_fps(image)
            cv2.imshow('result', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        if self.pub_result_img:
            result_img_msg = self.bridge.cv2_to_imgmsg(image, encoding="rgb8")
            result_img_msg.header = rgb_msg.header
            self.result_img_pub.publish(result_img_msg)


# -----------------------------
# Main
#
def main():
    rclpy.init()
    rclpy.spin(YoloV5Ros2())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
