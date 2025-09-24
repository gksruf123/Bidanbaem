#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from interfaces.msg import ObjectInfo, ObjectsInfo


class Subscriber(Node):
    def __init__(self):
        # TODO: Implement node initialization
        super().__init__('subscribe')

        # subscribe
        # TODO: Register callback
        self.inference_sub = self.create_subscription(
            ObjectsInfo,
            '/yolov5_ros2/object_detect',
            self.listener_callback,   
            10               
        )


    def listener_callback(self, msg):
        # TODO: Implemnt a callback
        for obj in msg.objects:
            self.get_logger().info(f"class_name: {obj.class_name}, score: {obj.score}")


def main(args=None):
    rclpy.init(args=args)
    node = Subscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
