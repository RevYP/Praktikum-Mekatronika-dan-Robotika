import os

from ament_index_python.packages import get_package_share_path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():
    dobot_description_path = get_package_share_path('dobot_description')
    default_model_path = dobot_description_path / 'model/magician_standalone.urdf.xacro'
    default_rviz_config_path = dobot_description_path / 'rviz/urdf_full.rviz'

    tool_arg = DeclareLaunchArgument(
        name='tool',
        default_value='gripper',
        choices=['suction_cup', 'gripper', 'extended_gripper', 'pen', 'none'],
        description='Type of tool attached to Dobot',
    )

    dof_arg = DeclareLaunchArgument(
        name='DOF',
        default_value='4',
        choices=['3', '4'],
        description='Number of DOF',
    )

    camera_arg = DeclareLaunchArgument(
        name='use_camera',
        default_value='false',
        choices=['true', 'false'],
        description='Add camera to model',
    )

    rviz_arg = DeclareLaunchArgument(
        name='rvizconfig',
        default_value=str(default_rviz_config_path),
        description='Path to rviz config',
    )

    dummy_node = Node(
        package='dobot_dummy',
        executable='dummy_node',
        name='dobot_dummy_node',
        output='screen',
    )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': Command([
                'xacro "',
                str(default_model_path),
                '" DOF:=', LaunchConfiguration('DOF'),
                ' use_camera:=', LaunchConfiguration('use_camera'),
                ' tool:=', LaunchConfiguration('tool'),
            ])
        }],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rvizconfig')],
    )

    return LaunchDescription([
        tool_arg,
        dof_arg,
        camera_arg,
        rviz_arg,
        dummy_node,
        robot_state_publisher_node,
        rviz_node,
    ])
