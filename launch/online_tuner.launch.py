import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('online_causal_tuner')

    default_config = os.path.join(pkg_share, 'config', 'default_tuner_config.yaml')
    default_model = os.path.join(os.getcwd(), 'models', 'causal_tuner_models.pkl')

    config_arg = DeclareLaunchArgument(
        'tuner_params_file',
        default_value=default_config,
        description='Full path to tuner YAML parameters file'
    )
    model_arg = DeclareLaunchArgument(
        'model_path',
        default_value=default_model,
        description='Full path to trained causal models pickle artifact'
    )

    tuner_node = Node(
        package='online_causal_tuner',
        executable='online_tuner_node',
        name='online_causal_tuner',
        output='screen',
        parameters=[
            LaunchConfiguration('tuner_params_file'),
            {'model_path': LaunchConfiguration('model_path')},
        ]
    )

    return LaunchDescription([
        config_arg,
        model_arg,
        tuner_node,
    ])
