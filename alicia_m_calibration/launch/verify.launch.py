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
    camera_name = LaunchConfiguration('camera_name').perform(context).strip('/')
    camera_namespace = LaunchConfiguration(
        'camera_namespace').perform(context).rstrip('/')
    if camera_name:
        prefix = f'{camera_namespace}/{camera_name}/color'
        overrides['camera_topic'] = f'{prefix}/image_raw'
        overrides['camera_info_topic'] = f'{prefix}/camera_info'
    for argument in ('camera_topic', 'camera_info_topic'):
        value = LaunchConfiguration(argument).perform(context)
        if value:
            overrides[argument] = value
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
        DeclareLaunchArgument(
            'camera_name', default_value='',
            description=(
                'Optional RealSense camera_name; omit to use configured topics '
                'or auto-discover the only publishing color camera')),
        DeclareLaunchArgument(
            'camera_namespace', default_value='/camera',
            description='Namespace containing the named RealSense camera'),
        DeclareLaunchArgument(
            'camera_topic', default_value='',
            description='Explicit image topic; overrides camera_name'),
        DeclareLaunchArgument(
            'camera_info_topic', default_value='',
            description='Explicit CameraInfo topic; overrides camera_name'),
        OpaqueFunction(function=start),
    ])
