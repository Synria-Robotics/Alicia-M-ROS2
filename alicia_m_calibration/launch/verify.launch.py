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
    overrides = {'result': LaunchConfiguration('result')}
    marker_size = LaunchConfiguration('marker_size').perform(context)
    marker_id = LaunchConfiguration('marker_id').perform(context)
    if marker_size:
        overrides['marker_size'] = float(marker_size)
    if marker_id:
        overrides['marker_id'] = int(marker_id)
    return [Node(
        package='alicia_m_calibration', executable='verify.py',
        output='screen', parameters=[str(config), overrides])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='eye_in_hand'),
        DeclareLaunchArgument('result', default_value=''),
        DeclareLaunchArgument(
            'marker_size', default_value='',
            description='Override result marker side length in metres'),
        DeclareLaunchArgument(
            'marker_id', default_value='',
            description='Override result marker ID'),
        OpaqueFunction(function=start),
    ])
