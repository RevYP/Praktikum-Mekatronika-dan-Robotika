#!/usr/bin/env python3
import json
import os

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO


class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        base_dir = '/home/otomasi/Praktikum-Mekatronika-dan-Robotika/Modul07 ROS YOLO/ROS_YOLO/src/yolo_ros'
        default_model = os.path.join(base_dir, 'models', 'yolov8n.pt')
        self.declare_parameter('model_path', default_model)
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.declare_parameter('output_topic', '/yolo/annotated')
        self.declare_parameter('detection_topic', '/yolo/detections')
        self.declare_parameter('confidence', 0.25)
        self.declare_parameter('show_cv', False)
        self.declare_parameter('window_name', 'YOLO Detection')
        model_path = self.get_parameter('model_path').get_parameter_value().string_value or default_model
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        detection_topic = self.get_parameter('detection_topic').get_parameter_value().string_value
        self.confidence = self.get_parameter('confidence').get_parameter_value().double_value
        self.show_cv = self.get_parameter('show_cv').get_parameter_value().bool_value
        self.window_name = self.get_parameter('window_name').get_parameter_value().string_value
        self.bridge = CvBridge()
        self.model = YOLO(model_path)
        self.sub = self.create_subscription(Image, input_topic, self.image_cb, 10)
        self.img_pub = self.create_publisher(Image, output_topic, 10)
        self.det_pub = self.create_publisher(String, detection_topic, 10)
        if self.show_cv:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 640, 480)
        self.get_logger().info(f'YOLO model: {model_path}')
        self.get_logger().info(f'Input: {input_topic}, annotated: {output_topic}, detections: {detection_topic}')

    def image_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            results = self.model(frame, conf=self.confidence, verbose=False)
            detections = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                detections.append({
                    'class_id': cls_id,
                    'class': self.model.names[cls_id],
                    'confidence': round(float(box.conf[0].item()), 4),
                    'bbox_xyxy': [int(x1), int(y1), int(x2), int(y2)],
                    'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                    'area': int((x2 - x1) * (y2 - y1))
                })
            annotated = results[0].plot()
            self.img_pub.publish(self.bridge.cv2_to_imgmsg(annotated, 'bgr8'))
            det_msg = String()
            det_msg.data = json.dumps(detections)
            self.det_pub.publish(det_msg)
            if self.show_cv:
                cv2.imshow(self.window_name, annotated)
                cv2.waitKey(1)
        except Exception as exc:
            self.get_logger().error(f'YOLO detection error: {exc}')

    def destroy_node(self):
        if self.show_cv:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
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
