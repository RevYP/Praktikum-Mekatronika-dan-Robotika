#!/usr/bin/env python3
import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraDisplay(Node):
    def __init__(self):
        super().__init__('camera_display')
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.declare_parameter('show_cv', True)
        self.declare_parameter('window_name', 'Camera Raw')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.show_cv = self.get_parameter('show_cv').get_parameter_value().bool_value
        self.window_name = self.get_parameter('window_name').get_parameter_value().string_value
        self.width = self.get_parameter('width').get_parameter_value().integer_value
        self.height = self.get_parameter('height').get_parameter_value().integer_value
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, input_topic, self.image_cb, 10)
        if self.show_cv:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.width, self.height)
        self.get_logger().info(f'Camera reader subscribed to {input_topic}, show_cv={self.show_cv}')

    def image_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            if self.show_cv:
                cv2.imshow(self.window_name, frame)
                cv2.waitKey(1)
        except Exception as exc:
            self.get_logger().error(f'Camera read error: {exc}')

    def destroy_node(self):
        if self.show_cv:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraDisplay()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
