from pathlib import Path
import cv2
import sys
import json
import numpy as np
from functools import partial

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


CONFIG_FILE = "LAB-cal.json"
WINDOW_NAME = "LAB Filter"
TB_L_MIN = "L Min"
TB_L_MAX = "L Max"
TB_A_MIN = "A Min"
TB_A_MAX = "A Max"
TB_B_MIN = "B Min"
TB_B_MAX = "B Max"

# Initial color range values
l_min = 0
a_min = 0
b_min = 0
l_max = 255
a_max = 255
b_max = 255


def update_color_value(x, color, is_min):
    global l_min, a_min, b_min, l_max, a_max, b_max
    match color:
        case "L":
            if is_min:
                l_min = x
            else:
                l_max = x
        case "A":
            if is_min:
                a_min = x
            else:
                a_max = x
        case "B":
            if is_min:
                b_min = x
            else:
                b_max = x


def load_config(config_path):
    global l_min, a_min, b_min, l_max, a_max, b_max
    if not Path(config_path).exists():
        return
    with open(config_path, "r") as f:
        d = json.load(f)
        l_min = d["l_min"]
        a_min = d["a_min"]
        b_min = d["b_min"]
        l_max = d["l_max"]
        a_max = d["a_max"]
        b_max = d["b_max"]


def save_config(config_path):
    with open(config_path, "w") as f:
        json.dump(
            {
                "l_min": l_min,
                "a_min": a_min,
                "b_min": b_min,
                "l_max": l_max,
                "a_max": a_max,
                "b_max": b_max,
            },
            f,
            indent=2,
        )


def update_trackbar_positions():
    cv2.setTrackbarPos(TB_L_MIN, WINDOW_NAME, l_min)
    cv2.setTrackbarPos(TB_L_MAX, WINDOW_NAME, l_max)
    cv2.setTrackbarPos(TB_A_MIN, WINDOW_NAME, a_min)
    cv2.setTrackbarPos(TB_A_MAX, WINDOW_NAME, a_max)
    cv2.setTrackbarPos(TB_B_MIN, WINDOW_NAME, b_min)
    cv2.setTrackbarPos(TB_B_MAX, WINDOW_NAME, b_max)


class Sub(Node):
    def __init__(self):
        super().__init__("cam_sub")
        self.bridge = CvBridge()
        self.latest_img = None

        self.rgb_sub = self.create_subscription(
            Image, "/ascamera/camera_publisher/rgb0/image", self.callback, 1
        )

    def callback(self, msg: Image):
        self.latest_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")


def ui_loop(node: Sub):
    global l_min, a_min, b_min, l_max, a_max, b_max

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.01)

        if node.latest_img is None:
            continue

        img = node.latest_img
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        lower = np.array([l_min, a_min, b_min])
        upper = np.array([l_max, a_max, b_max])
        mask = cv2.inRange(lab, lower, upper)
        cv2.imshow('mask', mask)

        h, w = mask.shape
        floodfilled = cv2.bitwise_not(mask)
        kernel = np.ones((3, 3), np.uint8)
        mask_ff = cv2.erode(floodfilled, kernel, iterations=1)
        mask_ff = np.zeros((h+2, w+2), np.uint8)  # OpenCV requires padding

        seed_point = (w//2, h//2)  # inside the white region
        cv2.floodFill(floodfilled, mask_ff, seed_point, 255)

        masked_img = cv2.bitwise_and(img, img, mask=mask)
        ff_region = mask_ff[1:-1, 1:-1]   # remove the +2 padding
        ff_img = cv2.bitwise_and(img, img, mask=ff_region)

        # Same for flood-filled region
        # ff_vis = cv2.applyColorMap(floodfilled, cv2.COLORMAP_JET)
        cv2.imshow("FloodFill Mask", floodfilled)

        cv2.rectangle(img, (w//2, h//2), (w//2 + 3, h//2 + 3), (255, 255, 255), 2)
        combined = np.hstack((img, masked_img, ff_img))
        cv2.imshow("LAB Filter Result", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 115:  # 's'
            save_config(CONFIG_FILE)
            print(f"File saved to {CONFIG_FILE}")

    cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = Sub()

    # Load config if exists
    load_config(CONFIG_FILE)

    cv2.namedWindow(WINDOW_NAME)
    cv2.resizeWindow(WINDOW_NAME, 800, 200)
    cv2.createTrackbar(TB_L_MIN, WINDOW_NAME, l_min, 255,
                       partial(update_color_value, color="L", is_min=True))
    cv2.createTrackbar(TB_L_MAX, WINDOW_NAME, l_max, 255,
                       partial(update_color_value, color="L", is_min=False))
    cv2.createTrackbar(TB_A_MIN, WINDOW_NAME, a_min, 255,
                       partial(update_color_value, color="A", is_min=True))
    cv2.createTrackbar(TB_A_MAX, WINDOW_NAME, a_max, 255,
                       partial(update_color_value, color="A", is_min=False))
    cv2.createTrackbar(TB_B_MIN, WINDOW_NAME, b_min, 255,
                       partial(update_color_value, color="B", is_min=True))
    cv2.createTrackbar(TB_B_MAX, WINDOW_NAME, b_max, 255,
                       partial(update_color_value, color="B", is_min=False))

    update_trackbar_positions()
    print(f"Loaded config: L[{l_min}-{l_max}], A[{a_min}-{a_max}], B[{b_min}-{b_max}]")

    ui_loop(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
