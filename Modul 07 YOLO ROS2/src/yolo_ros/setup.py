from glob import glob
from setuptools import setup

package_name = 'yolo_ros'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name, f'{package_name}.scripts'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/yolo.launch.py']),
        ('share/' + package_name + '/config', ['config/yolo_params.yaml']),
        ('share/' + package_name + '/models', glob('models/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Student',
    maintainer_email='student@otomasi.id',
    description='YOLOv8 ROS 2 Integration',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_node = yolo_ros.scripts.yolo_node:main',
            'camera_node = yolo_ros.scripts.camera_node:main',
            'yolo_camera_node = yolo_ros.scripts.yolo_camera_node:main',
            'training_node = yolo_ros.scripts.training_node:main',
            'rviz_display = yolo_ros.scripts.rviz_display:main',
        ],
    },
)
