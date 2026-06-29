#!/usr/bin/env python3
import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image


class RvizDisplay(Node):
    def __init__(self):
        super().__init__('rviz_display')
        self.declare_parameter('raw_topic', '/camera/image_raw')
        self.declare_parameter('yolo_topic', '/yolo/annotated')
        self.declare_parameter('show_raw', True)
        self.declare_parameter('show_yolo', True)
        raw_topic = self.get_parameter('raw_topic').get_parameter_value().string_value
        yolo_topic = self.get_parameter('yolo_topic').get_parameter_value().string_value
        self.show_raw = self.get_parameter('show_raw').get_parameter_value().bool_value
        self.show_yolo = self.get_parameter('show_yolo').get_parameter_value().bool_value
        self.bridge = CvBridge()
        self.raw_sub = self.create_subscription(Image, raw_topic, self.raw_cb, 10)
        self.yolo_sub = self.create_subscription(Image, yolo_topic, self.yolo_cb, 10)
        if self.show_raw:
            cv2.namedWindow('Raw Camera', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Raw Camera', 640, 480)
        if self.show_yolo:
            cv2.namedWindow('YOLO Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('YOLO Detection', 640, 480)
        self.get_logger().info(f'Display raw={raw_topic}, yolo={yolo_topic}')

    def raw_cb(self, msg):
        if not self.show_raw:
            return
        try:
            cv2.imshow('Raw Camera', self.bridge.imgmsg_to_cv2(msg, 'bgr8'))
            cv2.waitKey(1)
        except Exception as exc:
            self.get_logger().error(f'Raw display error: {exc}')

    def yolo_cb(self, msg):
        if not self.show_yolo:
            return
        try:
            cv2.imshow('YOLO Detection', self.bridge.imgmsg_to_cv2(msg, 'bgr8'))
            cv2.waitKey(1)
        except Exception as exc:
            self.get_logger().error(f'YOLO display error: {exc}')

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RvizDisplay()
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
