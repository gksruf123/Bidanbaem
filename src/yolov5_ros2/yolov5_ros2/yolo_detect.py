#!/usr/bin/env python3
from math import frexp
from traceback import print_tb
# import torch
# from yolov5 import YOLOv5

import message_filters
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

# --- ONNX 전용 최소 추가 ---
import onnxruntime as ort
import numpy as np
import json

def _load_class_names(base_dir, model_basename):
    # config/<model>.{json,yaml,yml,names} 순으로 탐색
    candidates = [
        os.path.join(base_dir, "config", f"{model_basename}.names.json"),
        os.path.join(base_dir, "config", f"{model_basename}.json"),
        os.path.join(base_dir, "config", f"{model_basename}.yaml"),
        os.path.join(base_dir, "config", f"{model_basename}.yml"),
        os.path.join(base_dir, "config", f"{model_basename}.names"),
    ]
    for p in candidates:
        if os.path.exists(p):
            ext = os.path.splitext(p)[1].lower()
            try:
                if ext == ".json":
                    with open(p, "r") as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        return [data[str(i)] for i in range(len(data))]
                    return list(data)
                if ext in (".yaml", ".yml"):
                    with open(p, "r") as f:
                        data = yaml.safe_load(f)
                    if isinstance(data, dict) and "names" in data:
                        return list(data["names"])
                    if isinstance(data, list):
                        return data
                with open(p, "r") as f:
                    return [line.strip() for line in f if line.strip()]
            except Exception:
                pass
    # fallback
    return [f"cls_{i}" for i in range(80)]

def _nms_numpy(boxes_xyxy, scores, iou_thres=0.45):
    if len(boxes_xyxy) == 0: return []
    x1, y1, x2, y2 = [boxes_xyxy[:, i] for i in range(4)]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(int(i))
        if order.size == 1: break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep

class OrtYoloCompat:
    """
    .predict(img_rgb) -> obj with:
      - .pred[0] = ndarray (N,6): [x1,y1,x2,y2,conf,cls]
      - .names = class names
    """
    class _PredObj:
        def __init__(self, pred, names):
            self.pred = [pred]
            self.names = names

    def __init__(self, onnx_path, class_names, img_size=640, device="cpu", conf_thres=0.25, iou_thres=0.45):
        providers = ["CPUExecutionProvider"]
        if "cuda" in str(device).lower():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.inp_name = self.sess.get_inputs()[0].name
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.names = class_names
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def _pre(self, img_rgb):
        h, w = img_rgb.shape[:2]
        inp = cv2.resize(img_rgb, (self.img_size, self.img_size))
        inp = inp.transpose(2, 0, 1).astype(np.float32) / 255.0
        inp = np.expand_dims(inp, 0)
        return inp, (w / float(self.img_size)), (h / float(self.img_size))

    def _ensure_xyxy(self, boxes):
        if len(boxes) == 0: return boxes
        smp = boxes[: min(10, len(boxes))]
        looks_xyxy = np.mean((smp[:,2] > smp[:,0]) & (smp[:,3] > smp[:,1])) > 0.6
        if looks_xyxy: return boxes
        cx, cy, w, h = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        x1 = cx - w/2.0; y1 = cy - h/2.0; x2 = cx + w/2.0; y2 = cy + h/2.0
        return np.stack([x1,y1,x2,y2], axis=1)

    def predict(self, img_rgb):
        inp, sx, sy = self._pre(img_rgb)
        outs = self.sess.run(self.out_names, {self.inp_name: inp})
        pred = np.squeeze(outs[0], 0)  # (N, no)
        if pred.ndim != 2 or pred.shape[1] < 6:
            raise RuntimeError(f"Unexpected ONNX output shape: {pred.shape}")
        boxes = pred[:, :4]
        obj = pred[:, 4]
        cls_prob = pred[:, 5:]
        cls_ids = np.argmax(cls_prob, axis=1)
        cls_scores = cls_prob[np.arange(len(cls_ids)), cls_ids]
        scores = obj * cls_scores

        keep = scores >= self.conf_thres
        boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
        boxes = self._ensure_xyxy(boxes)
        boxes[:, [0,2]] *= sx; boxes[:, [1,3]] *= sy

        keep_idx = _nms_numpy(boxes, scores, self.iou_thres)
        boxes, scores, cls_ids = boxes[keep_idx], scores[keep_idx], cls_ids[keep_idx]
        if len(boxes) == 0:
            out = np.zeros((0,6), dtype=np.float32)
        else:
            out = np.concatenate([
                boxes.astype(np.float32),
                scores.reshape(-1,1).astype(np.float32),
                cls_ids.reshape(-1,1).astype(np.float32)
            ], axis=1)
        return self._PredObj(out, self.names)
# --- /ONNX 전용 최소 추가 ---

# Get the ROS distribution version and set the shared directory for YoloV5 configuration files.
ros_distribution = os.environ.get("ROS_DISTRO")
package_share_directory = get_package_share_directory('yolov5_ros2')

class YoloV5Ros2(Node):
    def __init__(self):
        super().__init__('yolov5_ros2')
        self.get_logger().info(f"Current ROS 2 distribution: {ros_distribution}")
        self.fps = fps.FPS()

        self.declare_parameter("device", "cuda", ParameterDescriptor(
            name="device", description="Compute device selection, default: cpu, options: cuda:0"))
        self.declare_parameter("model", "yolov5s", ParameterDescriptor(
            name="model", description="Default model selection: yolov5s"))
        self.declare_parameter("image_topic", "/ascamera/camera_publisher/rgb0/image", ParameterDescriptor(
            name="image_topic", description="Image topic, default: /ascamera/camera_publisher/rgb0/image"))
        self.declare_parameter("show_result", False, ParameterDescriptor(
            name="show_result", description="Whether to display detection results, default: False"))
        self.declare_parameter("pub_result_img", False, ParameterDescriptor(
            name="pub_result_img", description="Whether to publish detection result images, default: False"))

        # 추가: ONNX 추론 파라미터
        self.declare_parameter("conf_thres", 0.25, ParameterDescriptor(
            name="conf_thres", description="confidence threshold"))
        self.declare_parameter("iou_thres", 0.45, ParameterDescriptor(
            name="iou_thres", description="IoU threshold for NMS"))
        self.declare_parameter("img_size", 640, ParameterDescriptor(
            name="img_size", description="inference image size (square)"))

        self.create_service(Trigger, '/yolov5/start', self.start_srv_callback)
        self.create_service(Trigger, '/yolov5/stop', self.stop_srv_callback) 
        self.create_service(Trigger, '~/init_finish', self.get_node_state)

        # Load the model: .pt -> .onnx 로 변경
        model_base = self.get_parameter('model').value
        model_path = os.path.join(package_share_directory, "config", f"{model_base}.onnx")
        device = self.get_parameter('device').value

        conf_thres = float(self.get_parameter("conf_thres").value)
        iou_thres = float(self.get_parameter("iou_thres").value)
        img_size  = int(self.get_parameter("img_size").value)

        # 클래스 이름 자동 로드(있으면 사용, 없으면 cls_#)
        class_names = _load_class_names(package_share_directory, model_base)

        # 기존 self.yolov5 를 ONNX 호환 래퍼로 대체 (predict/ names 시그니처 동일)
        self.yolov5 = OrtYoloCompat(
            onnx_path=model_path,
            class_names=class_names,
            img_size=img_size,
            device=device,
            conf_thres=conf_thres,
            iou_thres=iou_thres
        )

        # Publishers
        self.yolo_result_pub = self.create_publisher(Detection2DArray, "yolo_result", 10)
        self.result_msg = Detection2DArray()
        self.object_pub = self.create_publisher(ObjectsInfo, '~/object_detect', 1)
        self.result_img_pub = self.create_publisher(Image, "result_img", 10)

        # Subscriber
        # image_topic = self.get_parameter('image_topic').value
        # self.image_sub = self.create_subscription(Image, image_topic, self.image_callback, 10)
        rgb_sub = message_filters.Subscriber(self, Image, '/ascamera/camera_publisher/rgb0/image')
        depth_sub = message_filters.Subscriber(self, Image, '/ascamera/camera_publisher/depth0/image_raw')
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.05)
        ts.registerCallback(self.image_callback)

        # Bridge & flags
        self.bridge = CvBridge()
        self.show_result = self.get_parameter('show_result').value
        self.pub_result_img = self.get_parameter('pub_result_img').value
        self.start = True

    def get_node_state(self, request, response):
        response.success = True
        return response

    def start_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "start yolov5 detect (onnx)")
        self.start = True
        response.success = True
        response.message = "start"
        return response

    def stop_srv_callback(self, request, response):
        self.get_logger().info('\033[1;32m%s\033[0m' % "stop yolov5 detect (onnx)")
        self.start = False
        response.success = True
        response.message = "stop"
        return response

    def image_callback(self, rgb_msg, depth_msg):
        if not self.start:
            return

        image = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')
        detect_result = self.yolov5.predict(image)  # 동일 시그니처 유지

        # 깊이 값 보정 (0 → 주변값으로 보간)
        depth_uint16 = depth.astype(np.uint16)  # inpaint는 8/16bit만 지원
        mask = (depth_uint16 == 0).astype('uint8')    # 0인 부분을 마스크로 지정
        depth_inpaint = cv2.inpaint(depth_uint16, mask, 2, cv2.INPAINT_TELEA)

        # 다시 float로 변환 (필요하다면)
        depth = depth_inpaint.astype(np.float32)

        self.result_msg.detections.clear()
        self.result_msg.header.frame_id = "camera"
        self.result_msg.header.stamp = self.get_clock().now().to_msg()

        predictions = detect_result.pred[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        categories = predictions[:, 5]

        objects_info = []
        h, w = image.shape[:2]

        for index in range(len(categories)):
            cid = int(categories[index])
            name = detect_result.names[cid] if cid < len(detect_result.names) else f"cls_{cid}"
            self.get_logger().info(f"\033[1;31m**{name}**\033[0m")

            det = Detection2D()
            det.id = name
            x1, y1, x2, y2 = boxes[index]
            x1 = int(x1); y1 = int(y1); x2 = int(x2); y2 = int(y2)
            cx = (x1 + x2) / 2.0; cy = (y1 + y2) / 2.0

            box_distance = depth[int(cy), int(cx)]
            fence_distance = depth[10, 320]

            if ros_distribution == 'galactic':
                det.bbox.center.x = cx; det.bbox.center.y = cy
            else:
                det.bbox.center.position.x = cx; det.bbox.center.position.y = cy

            det.bbox.size_x = float(x2 - x1)
            det.bbox.size_y = float(y2 - y1)

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = name
            hyp.hypothesis.score = float(scores[index])
            det.results.append(hyp)
            self.result_msg.detections.append(det)

            if self.show_result or self.pub_result_img:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{name}:{hyp.hypothesis.score:.2f}", (x1, y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.waitKey(1)

            oi = ObjectInfo()
            oi.class_name = name
            oi.box = [x1, y1, x2, y2]
            oi.score = round(float(scores[index]), 2)
            oi.width = w; oi.height = h
            oi.distance = int(box_distance)
            oi.fence_distance = int(fence_distance)
            objects_info.append(oi)

        object_msg = ObjectsInfo()
        object_msg.objects = objects_info
        self.object_pub.publish(object_msg)

        if self.show_result:
            self.fps.update()
            image = self.fps.show_fps(image)
            cv2.imshow('result', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        if self.pub_result_img:
            result_img_msg = self.bridge.cv2_to_imgmsg(image, encoding="rgb8")
            result_img_msg.header = rgb_msg.header
            self.result_img_pub.publish(result_img_msg)

        if len(categories) > 0:
            self.yolo_result_pub.publish(self.result_msg)

def main():
    rclpy.init()
    rclpy.spin(YoloV5Ros2())
    rclpy.shutdown()

if __name__ == "__main__":
    main()