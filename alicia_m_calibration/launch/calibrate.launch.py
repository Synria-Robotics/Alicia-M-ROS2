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
    overrides = {}
    values = {
        'marker_size': ('marker_size', float),
        'marker_id': ('marker_id', int),
        'min_samples': ('samples', int),
        'calibration_method': ('method', str),
    }
    for argument, (parameter, convert) in values.items():
        value = LaunchConfiguration(argument).perform(context)
        if value:
            overrides[parameter] = convert(value)
    return [Node(
        package='alicia_m_calibration', executable='calibrate.py',
        output='screen', parameters=[str(config), overrides])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='eye_in_hand'),
        DeclareLaunchArgument(
            'marker_size', default_value='',
            description='ArUco marker side length in metres; defaults to mode YAML'),
        DeclareLaunchArgument(
            'marker_id', default_value='',
            description='ArUco marker ID; defaults to mode YAML'),
        DeclareLaunchArgument(
            'min_samples', default_value='',
            description='Minimum valid samples; defaults to mode YAML'),
        DeclareLaunchArgument(
            'calibration_method', default_value='',
            choices=['', 'auto', 'tsai', 'park', 'horaud', 'daniilidis', 'andreff'],
            description='OpenCV hand-eye method; defaults to mode YAML'),
        OpaqueFunction(function=start),
    ])
