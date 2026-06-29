from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mediapipe_modbus',
            executable='mediapipe_node',
            name='mediapipe_node',
            output='screen',
        ),
        Node(
            package='mediapipe_modbus',
            executable='gui_node',
            name='gui_node',
            output='screen',
        ),
    ])
