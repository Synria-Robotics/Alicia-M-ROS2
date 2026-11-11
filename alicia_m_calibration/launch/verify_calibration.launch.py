"""One-command hardware, camera, MoveIt, and calibration verification."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def include(package, launch_file, arguments, condition=None):
    source = Path(get_package_share_directory(package)) / 'launch' / launch_file
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(source)),
        launch_arguments=arguments.items(), condition=condition)


def start(context):
    mode = LaunchConfiguration('mode').perform(context)
    if mode not in ('eye_in_hand', 'eye_to_hand'):
        raise ValueError('mode must be eye_in_hand or eye_to_hand')

    camera_name = LaunchConfiguration('camera_name').perform(context)
    if not camera_name:
        camera_name = 'cam1' if mode == 'eye_to_hand' else 'camera'

    robot = include(
        'alicia_m_bringup', 'moveit_hardware.launch.py', {
            'serial_port': LaunchConfiguration('serial_port'),
            'control_mode': LaunchConfiguration('control_mode'),
            'use_rviz': LaunchConfiguration('use_rviz'),
        }, IfCondition(LaunchConfiguration('start_robot')))
    camera = include(
        'realsense2_camera', 'rs_launch.py', {
            'camera_name': camera_name,
            'enable_depth': 'false',
            'enable_infra1': 'false',
            'enable_infra2': 'false',
        }, IfCondition(LaunchConfiguration('start_camera')))
    verifier = include(
        'alicia_m_calibration', 'verify.launch.py', {
            'mode': mode,
            'result': LaunchConfiguration('result'),
            'marker_id': LaunchConfiguration('marker_id'),
            'marker_size': LaunchConfiguration('marker_size'),
            'camera_name': camera_name,
        })
    return [robot, camera, verifier]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'mode', default_value='eye_to_hand',
            choices=['eye_in_hand', 'eye_to_hand']),
        DeclareLaunchArgument('result', default_value=''),
        DeclareLaunchArgument('marker_id', default_value='0'),
        DeclareLaunchArgument('marker_size', default_value='0.0765'),
        DeclareLaunchArgument('serial_port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('control_mode', default_value='pv'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument(
            'start_robot', default_value='true',
            description='Start hardware, MoveIt, and RViz'),
        DeclareLaunchArgument(
            'start_camera', default_value='true',
            description='Start the RealSense camera'),
        DeclareLaunchArgument(
            'camera_name', default_value='',
            description='Defaults to cam1 for eye-to-hand and camera otherwise'),
        OpaqueFunction(function=start),
    ])
