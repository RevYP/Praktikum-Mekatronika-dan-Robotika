#!/usr/bin/env python3
import json
import time

import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .config import load_config


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.config = load_config()['camera']
        self.publisher = self.create_publisher(String, 'camera/frame_info', 10)
        self.capture = cv2.VideoCapture(int(self.config['index']))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.config['width']))
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.config['height']))
        self.timer = self.create_timer(0.033, self.publish_frame_info)
        self.frame_count = 0
        if not self.capture.isOpened():
            self.get_logger().error('Kamera tidak ditemukan')

    def publish_frame_info(self):
        ok, frame = self.capture.read()
        if not ok:
            return
        if self.config.get('mirror', True):
            frame = cv2.flip(frame, 1)
        self.frame_count += 1
        height, width = frame.shape[:2]
        msg = String()
        msg.data = json.dumps({'frame': self.frame_count, 'width': width, 'height': height, 'time': time.time()})
        self.publisher.publish(msg)

    def destroy_node(self):
        if self.capture:
            self.capture.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
