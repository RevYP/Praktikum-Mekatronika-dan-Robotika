import os
from glob import glob
from setuptools import setup

package_name = 'dobot_dummy'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='otomasi',
    maintainer_email='otomasi@local',
    description='Dummy simulator for Dobot Magician (no hardware required)',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dummy_node = dobot_dummy.dummy_node:main',
        ],
    },
)
