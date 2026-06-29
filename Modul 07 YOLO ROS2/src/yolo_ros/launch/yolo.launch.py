#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model_path = LaunchConfiguration('model_path')
    camera_topic = LaunchConfiguration('camera_topic')
    show_cv = LaunchConfiguration('show_cv')
    confidence = LaunchConfiguration('confidence')
    return LaunchDescription([
        DeclareLaunchArgument('model_path', default_value='', description='Path model YOLO .pt'),
        DeclareLaunchArgument('camera_topic', default_value='/camera/image_raw', description='Topic kamera raw'),
        DeclareLaunchArgument('show_cv', default_value='true', description='Tampilkan window OpenCV raw dan YOLO'),
        DeclareLaunchArgument('confidence', default_value='0.25', description='Confidence threshold YOLO'),
        Node(
            package='yolo_ros',
            executable='yolo_node',
            name='yolo_detector',
            parameters=[{
                'model_path': model_path,
                'input_topic': camera_topic,
                'confidence': confidence,
                'show_cv': False,
            }],
            output='screen'
        ),
        Node(
            package='yolo_ros',
            executable='rviz_display',
            name='rviz_display',
            parameters=[{
                'raw_topic': camera_topic,
                'yolo_topic': '/yolo/annotated',
                'show_raw': show_cv,
                'show_yolo': show_cv,
            }],
            output='screen'
        ),
    ])
