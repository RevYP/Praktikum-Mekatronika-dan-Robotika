#!/usr/bin/env python3
import os
import shutil

import rclpy
import yaml
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO


class TrainingNode(Node):
    def __init__(self):
        super().__init__('training_node')
        base_dir = '/home/otomasi/Praktikum-Mekatronika-dan-Robotika/Modul07 ROS YOLO/ROS_YOLO/src/yolo_ros'
        self.default_data_dir = os.path.join(base_dir, 'training_data')
        self.default_output_dir = os.path.join(base_dir, 'models')
        self.declare_parameter('data_dir', self.default_data_dir)
        self.declare_parameter('base_model', os.path.join(base_dir, 'models', 'yolov8n.pt'))
        self.declare_parameter('model_output', os.path.join(self.default_output_dir, 'custom_yolov8.pt'))
        self.declare_parameter('epochs', 50)
        self.declare_parameter('batch_size', 16)
        self.declare_parameter('img_size', 640)
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('auto_start', False)
        self.data_dir = self.get_parameter('data_dir').get_parameter_value().string_value
        self.base_model = self.get_parameter('base_model').get_parameter_value().string_value
        self.model_output = self.get_parameter('model_output').get_parameter_value().string_value
        self.epochs = self.get_parameter('epochs').get_parameter_value().integer_value
        self.batch_size = self.get_parameter('batch_size').get_parameter_value().integer_value
        self.img_size = self.get_parameter('img_size').get_parameter_value().integer_value
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.status_pub = self.create_publisher(String, '/yolo/training_status', 10)
        self.get_logger().info(f'Dataset: {self.data_dir}')
        self.get_logger().info(f'Output model: {self.model_output}')
        if self.get_parameter('auto_start').get_parameter_value().bool_value:
            self.create_timer(1.0, self.train_once)

    def train_once(self):
        if hasattr(self, 'started'):
            return
        self.started = True
        self.train_model()

    def ensure_dataset_structure(self):
        for split in ('train', 'valid', 'test'):
            os.makedirs(os.path.join(self.data_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.data_dir, split, 'labels'), exist_ok=True)
        os.makedirs(self.default_output_dir, exist_ok=True)

    def dataset_yaml(self):
        self.ensure_dataset_structure()
        yaml_path = os.path.join(self.data_dir, 'data.yaml')
        if os.path.exists(yaml_path):
            return yaml_path
        data = {
            'path': self.data_dir,
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'names': {0: 'object'}
        }
        with open(yaml_path, 'w', encoding='utf-8') as file:
            yaml.safe_dump(data, file, sort_keys=False)
        return yaml_path

    def has_training_images(self):
        train_images = os.path.join(self.data_dir, 'train', 'images')
        return any(name.lower().endswith(('.jpg', '.jpeg', '.png')) for name in os.listdir(train_images))

    def publish_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(text)

    def train_model(self):
        yaml_path = self.dataset_yaml()
        if not self.has_training_images():
            self.publish_status(f'Dataset kosong. Paste hasil export YOLOv8 ke {self.data_dir}')
            return False
        try:
            self.publish_status('Training YOLOv8 dimulai')
            model = YOLO(self.base_model)
            results = model.train(
                data=yaml_path,
                epochs=self.epochs,
                batch=self.batch_size,
                imgsz=self.img_size,
                device=self.device,
                name='ros_yolo_custom'
            )
            best_pt = os.path.join(str(results.save_dir), 'weights', 'best.pt')
            shutil.copy2(best_pt, self.model_output)
            self.publish_status(f'Training selesai. Model disimpan ke {self.model_output}')
            return True
        except Exception as exc:
            self.publish_status(f'Training gagal: {exc}')
            return False


def main(args=None):
    rclpy.init(args=args)
    node = TrainingNode()
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
