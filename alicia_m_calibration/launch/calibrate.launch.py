"""Run eye-in-hand or eye-to-hand calibration from its YAML config."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def start(context):
    mode = LaunchConfiguration('mode').perform(context)
    if mode not in ('eye_in_hand', 'eye_to_hand'):
        raise ValueError('mode must be eye_in_hand or eye_to_hand')
    config = Path(get_package_share_directory('alicia_m_calibration'))
    config = config / 'config' / f'{mode}.yaml'
    return [Node(
        package='alicia_m_calibration', executable='calibrate.py',
        output='screen', parameters=[str(config)])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='eye_in_hand'),
        OpaqueFunction(function=start),
    ])
