"""Verify an eye-in-hand or eye-to-hand result."""

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
        package='alicia_m_calibration', executable='verify.py',
        output='screen', parameters=[str(config), {
            'result': LaunchConfiguration('result')}])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='eye_in_hand'),
        DeclareLaunchArgument('result', default_value=''),
        OpaqueFunction(function=start),
    ])
