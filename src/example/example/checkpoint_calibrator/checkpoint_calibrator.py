#!/usr/bin/env python3
import rclpy, math, time, numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from interfaces.msg import ObjectsInfo
from cv_bridge import CvBridge

class Calibrator(Node):
    def __init__(self):
        super().__init__("checkpoint_calibrator")
        self.declare_parameter("rgb_topic", "/ascamera/camera_publisher/rgb0/image")
        self.declare_parameter("depth_topic", "/ascamera/camera_publisher/depth0/image")
        self.declare_parameter("classes", ["crosswalk","go","right","green","red","park"])
        self.declare_parameter("hfov_deg", 69.0)
        self.declare_parameter("samples", 30)

        self.rgb_topic   = self.get_parameter("rgb_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.target_classes = list(self.get_parameter("classes").value)
        self.hfov = math.radians(float(self.get_parameter("hfov_deg").value))
        self.samples = int(self.get_parameter("samples").value)

        self.bridge = CvBridge()
        self.rgb = None; self.depth = None; self.objs = []
        self.create_subscription(Image, self.rgb_topic, self.rgb_cb, 10)
        self.create_subscription(Image, self.depth_topic, self.depth_cb, 10)
        self.create_subscription(ObjectsInfo, "/yolov5_ros2/object_detect", self.obj_cb, 10)

        self.get_logger().info("Calibrator ready. Hold still at checkpoint...")
        self.timer = self.create_timer(0.05, self.tick)

        self.bearings, self.dists = [], []
        self.done = False

    def rgb_cb(self, msg):  self.rgb = self.bridge.imgmsg_to_cv2(msg, "rgb8")
    def depth_cb(self, msg): self.depth = self.bridge.imgmsg_to_cv2(msg)
    def obj_cb(self, msg):   self.objs = msg.objects

    def _bbox_center_bearing_deg(self, bbox, w):
        x1,_,x2,_ = bbox
        cx = 0.5*(x1+x2)
        bearing = ((cx - w/2.0)/(w/2.0))*(self.hfov/2.0)
        return math.degrees(-bearing)  # 오른쪽 음수

    def _depth_at_bbox_center(self, bbox):
        if self.depth is None: return None
        x1,y1,x2,y2 = map(int, bbox)
        h,w = self.depth.shape[:2]
        x1=max(0,min(w-1,x1)); x2=max(0,min(w-1,x2))
        y1=max(0,min(h-1,y1)); y2=max(0,min(h-1,y2))
        cx=int(0.5*(x1+x2)); cy=int(0.5*(y1+y2))
        xs=slice(max(0,cx-2),min(w,cx+3)); ys=slice(max(0,cy-2),min(h,cy+3))
        patch=self.depth[ys,xs]
        if patch.size==0: return None
        d=float(np.median(patch))
        if d>5.0: d*=0.001  # mm→m 가정
        return d if 0.05<d<10.0 else None

    def tick(self):
        if self.done or self.rgb is None or not self.objs: return
        H,W = self.rgb.shape[:2]

        # 타겟 클래스 우선순위대로 검색
        for cls in self.target_classes:
            cand = [o for o in self.objs if o.class_name==cls]
            if not cand: continue
            # 점수 최고 하나 선택
            o = max(cand, key=lambda x: x.score)
            bearing = self._bbox_center_bearing_deg(o.box, W)
            dist = self._depth_at_bbox_center(o.box)
            if dist is None: return
            self.bearings.append(bearing); self.dists.append(dist)
            self.get_logger().info(f"[sample {len(self.bearings)}/{self.samples}] {cls}: "
                                   f"bearing={bearing:.2f}°, dist={dist:.3f} m")
            break

        if len(self.bearings) >= self.samples:
            b = np.array(self.bearings); d = np.array(self.dists)
            bearing_med = float(np.median(b)); dist_med = float(np.median(d))
            bearing_mad = float(np.median(np.abs(b-bearing_med)))
            dist_mad = float(np.median(np.abs(d-dist_med)))
            self.get_logger().info("==== RESULT (copy to CHECKPOINT_EXPECTED) ====")
            self.get_logger().info(f"bearing_deg: {bearing_med:.2f}   (robust± ~{1.5*bearing_mad:.2f})")
            self.get_logger().info(f"distance_m : {dist_med:.3f}   (robust± ~{1.5*dist_mad:.3f})")
            self.get_logger().info("=============================================")
            self.done = True

def main():
    rclpy.init()
    n = Calibrator()
    try:
        rclpy.spin(n)
    finally:
        n.destroy_node(); rclpy.shutdown()

if __name__ == "__main__":
    main()
