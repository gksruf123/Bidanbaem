import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from interfaces.msg import ObjectsInfo
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class MSSDNode(Node):
    def __init__(self, name='MS Self Drive'):
        rclpy.init()
        super().__init__(name, allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.name = name
        self.is_running = True
        self.image_queue
        self.cv_bridge = CvBridge()
        self.classes = ['crosswalk', 'right', 'park', 'red', 'green', 'go']
        # self.create_subscription(Image, '/ascamera/camera_publisher/rgb0/image', self.image_callback, 1)
        # self.create_subscription(ObjectsInfo, '/yolov5_ros2/object_detect', self.object_detect_callback, 1)

        self.pos = None
        self.ang = None
        # TODO: Try SLAM with depth camera or 2d lidar.
        # Utilize odometer to check movement or use lidar to check absolute position

        # TODO: Lane keeping on both left and right side. 

        # TODO: Vector based movement, angle the car to face vector dir. 

        self.wheel_pub = self.create_publisher(Twist, '/controller/cmd_vel', 1)
        self.timer = self.create_timer(0.0, self.init_process) # run init_process asap

    def init_process(self):
        self.timer.cancel() 

        self.stop_movement() # reset wheel

    def stop_movement(self):
        self.wheel_pub.publish(Twist())

    def image_callback(self):
        pass
    def object_detect_callback(self):
        pass

def main():
    node = MSSDNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()

if __name__ == '__main__':
    main()