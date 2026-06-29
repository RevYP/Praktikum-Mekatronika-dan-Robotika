#!/usr/bin/env python3
import json
import time

import cv2
import mediapipe as mp
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .config import FINGER_NAMES, load_config


class MediaPipeNode(Node):
    def __init__(self):
        super().__init__('mediapipe_node')
        self.config = load_config()
        cam = self.config['camera']
        mp_cfg = self.config['mediapipe']
        self.publisher = self.create_publisher(String, 'mediapipe/fingers', 10)
        self.capture = cv2.VideoCapture(int(cam['index']))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam['width']))
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam['height']))
        self.mirror = bool(cam.get('mirror', True))
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=float(mp_cfg['min_detection_confidence']),
            min_tracking_confidence=float(mp_cfg['min_tracking_confidence']),
        )
        self.timer = self.create_timer(0.05, self.process_frame)
        self.last_states = None

    def process_frame(self):
        ok, frame = self.capture.read()
        if not ok:
            return
        if self.mirror:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        states = [False] * 5
        detected = bool(result.multi_hand_landmarks)
        if detected:
            lm = result.multi_hand_landmarks[0].landmark
            states[0] = lm[4].x < lm[3].x
            states[1] = lm[8].y < lm[6].y
            states[2] = lm[12].y < lm[10].y
            states[3] = lm[16].y < lm[14].y
            states[4] = lm[20].y < lm[18].y
        if states == self.last_states and detected:
            return
        self.last_states = list(states)
        msg = String()
        msg.data = json.dumps({'detected': detected, 'fingers': dict(zip(FINGER_NAMES, states)), 'states': states, 'time': time.time()})
        self.publisher.publish(msg)

    def destroy_node(self):
        if self.capture:
            self.capture.release()
        self.hands.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MediaPipeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
