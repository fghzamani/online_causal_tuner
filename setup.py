from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'online_causal_tuner'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name] if os.path.exists('resource/' + package_name) else []),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='forough',
    maintainer_email='forough@todo.todo',
    description='Offline causal model training and online dynamic parameter tuner for Nav2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train_causal_models = online_causal_tuner.train_causal_models:main',
            'online_tuner_node = online_causal_tuner.online_tuner_node:main',
            'dynamic_obstacle_controller = online_causal_tuner.dynamic_obstacle_controller:main',
        ],
    },
)
