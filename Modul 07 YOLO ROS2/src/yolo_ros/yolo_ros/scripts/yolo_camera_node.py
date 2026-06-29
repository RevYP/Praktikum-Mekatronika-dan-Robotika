#!/usr/bin/env python3
import os

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO


class YoloCamera(Node):
    def __init__(self):
        super().__init__('yolo_camera')
        base_dir = '/home/otomasi/Praktikum-Mekatronika-dan-Robotika/Modul07 ROS YOLO/ROS_YOLO/src/yolo_ros'
        default_model = os.path.join(base_dir, 'models', 'yolov8n.pt')
        self.declare_parameter('model_path', default_model)
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.declare_parameter('show_cv', True)
        self.declare_parameter('window_name', 'YOLO Camera')
        self.declare_parameter('confidence', 0.25)
        model_path = self.get_parameter('model_path').get_parameter_value().string_value or default_model
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.show_cv = self.get_parameter('show_cv').get_parameter_value().bool_value
        self.window_name = self.get_parameter('window_name').get_parameter_value().string_value
        self.confidence = self.get_parameter('confidence').get_parameter_value().double_value
        self.bridge = CvBridge()
        self.model = YOLO(model_path)
        self.sub = self.create_subscription(Image, input_topic, self.image_cb, 10)
        if self.show_cv:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 640, 480)
        self.get_logger().info(f'YOLO camera subscribed to {input_topic}, model={model_path}, show_cv={self.show_cv}')

    def image_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            results = self.model(frame, conf=self.confidence, verbose=False)
            annotated = results[0].plot()
            if self.show_cv:
                cv2.imshow(self.window_name, annotated)
                cv2.waitKey(1)
        except Exception as exc:
            self.get_logger().error(f'YOLO camera error: {exc}')

    def destroy_node(self):
        if self.show_cv:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloCamera()
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
