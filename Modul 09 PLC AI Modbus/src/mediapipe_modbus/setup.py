from setuptools import setup
import os
from glob import glob

package_name = 'mediapipe_modbus'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='otomasi',
    maintainer_email='otomasi@todo.todo',
    description='ROS2 MediaPipe Hand Gesture + Modbus TCP Controller with PyQt5 GUI',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = mediapipe_modbus.camera_node:main',
            'mediapipe_node = mediapipe_modbus.mediapipe_node:main',
            'gui_node = mediapipe_modbus.gui_node:main',
            'modbus_node = mediapipe_modbus.modbus_node:main',
        ],
    },
)
