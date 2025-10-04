import os
import numpy as np
import cv2
import onnxruntime
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import ParameterDescriptor
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose, Detection2D
from sensor_msgs.msg import Image
from interfaces.msg import ObjectInfo, ObjectsInfo
import message_filters
from std_srvs.srv import Trigger

class YoloV5ONNXNode(Node):
    def __init__(self):
        super().__init__('yolov5_onnx_ros2')

        # ROS 파라미터
        self.declare_parameter("device", "cpu", ParameterDescriptor(
            name="device", description="Compute device selection, default: cpu"))
        self.declare_parameter("model", "new_model", ParameterDescriptor(
            name="model", description="ONNX model name without extension"))
        self.declare_parameter("show_result", True, ParameterDescriptor(
            name="show_result", description="Display detection results"))
        self.declare_parameter("pub_result_img", True, ParameterDescriptor(
            name="pub_result_img", description="Publish result image"))

        # 파라미터 가져오기
        device = self.get_parameter("device").value
        model_name = self.get_parameter("model").value
        self.show_result = self.get_parameter("show_result").value
        self.pub_result_img = self.get_parameter("pub_result_img").value

        # 모델 로드
        package_share_directory = get_package_share_directory('yolov5_ros2')
        model_path = os.path.join(package_share_directory, "config", f"{model_name}.onnx")
        names_path = os.path.join(package_share_directory, "config", f"{model_name}.names.json")
        import json
        with open(names_path, "r") as f:
            self.class_names = json.load(f)

        providers = ['CPUExecutionProvider'] if device == 'cpu' else ['CUDAExecutionProvider']
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        # Publisher
        self.result_pub = self.create_publisher(Detection2DArray, "yolo_result", 10)
        self.result_img_pub = self.create_publisher(Image, "result_img", 10)
        self.object_pub = self.create_publisher(ObjectsInfo, "object_detect", 10)

        # CvBridge
        self.bridge = CvBridge()

        # Subscriber with message_filters
        rgb_sub = message_filters.Subscriber(self, Image, '/ascamera/camera_publisher/rgb0/image')
        depth_sub = message_filters.Subscriber(self, Image, '/ascamera/camera_publisher/depth0/image_raw')
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.05)
        ts.registerCallback(self.image_callback)

    def preprocess(self, image):
        img_resized = cv2.resize(image, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        # [1,3,640,640]
        img_transposed = np.transpose(img_normalized, (2,0,1))[np.newaxis, :]
        return img_transposed

    def image_callback(self, rgb_msg, depth_msg):
        image = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")

        # Depth 0 보간
        mask = (depth == 0).astype('uint8')
        depth = cv2.inpaint(depth.astype(np.uint16), mask, 2, cv2.INPAINT_TELEA).astype(np.float32)

        input_tensor = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        preds = outputs[0][0]  # [num_detections, 85]

        result_msg = Detection2DArray()
        result_msg.header.frame_id = "camera"
        result_msg.header.stamp = self.get_clock().now().to_msg()
        objects_info = []

        orig_h, orig_w = image.shape[:2]
        pred_h, pred_w = 640, 640  # 모델 입력 크기

        for pred in preds:
            x1, y1, x2, y2 = map(int, pred[:4])

            # pred[:4] 는 x1, y1, x2, y2
            x1, y1, x2, y2 = pred[:4]
            x1 = int(x1 * orig_w / pred_w)
            x2 = int(x2 * orig_w / pred_w)
            y1 = int(y1 * orig_h / pred_h)
            y2 = int(y2 * orig_h / pred_h)
            object_conf = float(pred[4])
            class_probs = pred[5:]
            cls_idx = int(np.argmax(class_probs))
            score = object_conf * float(class_probs[cls_idx])
            name = self.class_names[str(cls_idx)]

            # ROS Detection2D
            det2d = Detection2D()
            det2d.id = name
            det2d.bbox.center.position.x = (x1 + x2) / 2
            det2d.bbox.center.position.y = (y1 + y2) / 2
            det2d.bbox.size_x = float(x2 - x1)
            det2d.bbox.size_y = float(y2 - y1)

            obj_pose = ObjectHypothesisWithPose()
            obj_pose.hypothesis.class_id = name
            obj_pose.hypothesis.score = score
            det2d.results.append(obj_pose)
            result_msg.detections.append(det2d)

            # ObjectsInfo
            obj_info = ObjectInfo()
            obj_info.class_name = name
            obj_info.box = [x1, y1, x2, y2]
            obj_info.score = round(score, 2)
            obj_info.width = image.shape[1]
            obj_info.height = image.shape[0]
            obj_info.distance = int(depth[int((y1+y2)/2), int((x1+x2)/2)])
            objects_info.append(obj_info)

            # Draw box
            if self.show_result or self.pub_result_img:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(image, f"{name}:{score:.2f}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # Publish
        self.result_pub.publish(result_msg)

        obj_msg = ObjectsInfo()
        obj_msg.objects = objects_info
        self.object_pub.publish(obj_msg)

        if self.pub_result_img:
            img_msg = self.bridge.cv2_to_imgmsg(image, encoding="rgb8")
            img_msg.header = rgb_msg.header
            self.result_img_pub.publish(img_msg)

        if self.show_result:
            cv2.imshow("result", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)


def main():
    rclpy.init()
    node = YoloV5ONNXNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
