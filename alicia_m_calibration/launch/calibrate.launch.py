"""Run eye-in-hand or eye-to-hand calibration from its YAML config."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml


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
    reference_pose = LaunchConfiguration('reference_pose').perform(context)
    if reference_pose:
        parsed = yaml.safe_load(reference_pose)
        if not isinstance(parsed, list) or len(parsed) != 6:
            raise ValueError('reference_pose must be a list of six numbers')
        overrides['reference_pose'] = [float(item) for item in parsed]
    for argument in ('pose_amplitudes',):
        value = LaunchConfiguration(argument).perform(context)
        if value:
            parsed = yaml.safe_load(value)
            if not isinstance(parsed, list) or len(parsed) != 6:
                raise ValueError(f'{argument} must be a list of six numbers')
            overrides[argument] = [float(item) for item in parsed]
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
            'calibration_method',
            choices=['tsai', 'park', 'horaud', 'daniilidis', 'andreff'],
            description='Required OpenCV hand-eye calibration method'),
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
        DeclareLaunchArgument(
            'reference_pose', default_value='',
            description='Six-joint reference pose; defaults to mode YAML'),
        DeclareLaunchArgument(
            'pose_amplitudes', default_value='',
            description='Maximum six-joint offsets used by pose generation'),
        OpaqueFunction(function=start),
    ])
