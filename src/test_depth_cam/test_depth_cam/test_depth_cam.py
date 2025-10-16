import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import message_filters  # 메시지 동기화용


class RGBDVisualizer(Node):
    def __init__(self):
        super().__init__('rgbd_visualizer')
        self.bridge = CvBridge()

        # RGB, Depth Subscriber (message_filters 사용)
        rgb_sub = message_filters.Subscriber(self, Image, '/ascamera/camera_publisher/rgb0/image')
        depth_sub = message_filters.Subscriber(self, Image, '/ascamera/camera_publisher/depth0/image_raw')

        # ApproximateTimeSynchronizer: 약간의 시간 오차 허용
        ts = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=10, slop=0.1)
        ts.registerCallback(self.sync_callback)

    def sync_callback(self, rgb_msg, depth_msg):
        # ROS Image → OpenCV
        rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

        # 깊이 값 보정 (0 → 주변값으로 보간)
        depth_uint16 = depth_image.astype(np.uint16)  # inpaint는 8/16bit만 지원
        mask = (depth_uint16 == 0).astype('uint8')    # 0인 부분을 마스크로 지정
        depth_inpaint = cv2.inpaint(depth_uint16, mask, 2, cv2.INPAINT_TELEA)

        # 다시 float로 변환 (필요하다면)
        depth_image = depth_inpaint.astype(np.float32)

        img = rgb_image.copy()
        h, w = img.shape[:2]
        cx = w // 2  # 중앙 x 좌표

        step = 30
        for y in range(0, h, step):
            depth_value = depth_image[y, cx]

            # 깊이 값이 유효할 때만 표시
            if np.isfinite(depth_value) and depth_value > 0:
                distance = depth_value / 1000.0  # mm → m (센서 단위에 따라 조정)
                cv2.circle(img, (cx, y), 5, (0, 0, 255), -1)
                cv2.putText(img, f"{distance:.2f}m", (cx + 10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow("RGB with Depth Points", img)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = RGBDVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
