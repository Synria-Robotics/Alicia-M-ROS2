"""
hardware.launch.py - 仅启动硬件（ros2_control + 控制器）

启动内容:
  - robot_state_publisher (加载带真实硬件插件的 URDF)
  - ros2_control controller_manager
  - joint_state_broadcaster
  - arm_controller (JointTrajectoryController)
  - gripper_controller (JointTrajectoryController)

参数:
  - serial_port: 串口设备路径 (默认 /dev/ttyACM0, 可选 /dev/ttyUSB0)
  - control_mode: 控制模式 pv 或 mit (默认 pv)
  - baudrate: 波特率 (默认 1000000)

用法:
  ros2 launch alicia_m_bringup hardware.launch.py
  ros2 launch alicia_m_bringup hardware.launch.py serial_port:=/dev/ttyUSB0
  ros2 launch alicia_m_bringup hardware.launch.py control_mode:=mit
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessStart
from launch.substitutions import (
    Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # ======================== 启动参数 ========================
    declared_args = [
        DeclareLaunchArgument(
            'serial_port', default_value='/dev/ttyACM0',
            description='串口设备路径 (如 /dev/ttyACM0 或 /dev/ttyUSB0)'),
        DeclareLaunchArgument(
            'control_mode', default_value='pv',
            description='控制模式: pv (位置+速度) 或 mit'),
        DeclareLaunchArgument(
            'baudrate', default_value='1000000',
            description='串口波特率'),
        DeclareLaunchArgument(
            'default_speed', default_value='1.0',
            description='默认关节速度 (rad/s)'),
    ]

    serial_port = LaunchConfiguration('serial_port')
    control_mode = LaunchConfiguration('control_mode')
    baudrate = LaunchConfiguration('baudrate')
    default_speed = LaunchConfiguration('default_speed')

    # ======================== URDF 生成 ========================
    moveit_config_pkg = FindPackageShare('alicia_m_moveit_config')

    robot_description_content = Command([
        FindExecutable(name='xacro'), ' ',
        PathJoinSubstitution([
            moveit_config_pkg, 'config',
            'alicia_m_v1_1_follower.urdf.xacro'
        ]),
        ' ros2_control_plugin:=alicia_m_driver/AliciaHardwareInterface',
        ' serial_port:=', serial_port,
        ' baudrate:=', baudrate,
        ' control_mode:=', control_mode,
        ' default_speed:=', default_speed,
    ])

    robot_description = {
        'robot_description': ParameterValue(robot_description_content, value_type=str)
    }

    # ======================== 控制器配置 ========================
    bringup_pkg = FindPackageShare('alicia_m_bringup')
    controllers_yaml = PathJoinSubstitution([
        bringup_pkg, 'config', 'ros2_controllers.yaml'
    ])

    # ======================== 节点 ========================
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[robot_description],
    )

    control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[robot_description, controllers_yaml],
        output='both',
    )

    # 控制器在 controller_manager 启动后延迟 spawn
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['arm_controller', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    gripper_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gripper_controller', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    # controller_manager 启动后延时 spawn 控制器，避免竞争
    delayed_controllers = TimerAction(
        period=3.0,
        actions=[
            joint_state_broadcaster_spawner,
            arm_controller_spawner,
            gripper_controller_spawner,
        ],
    )

    return LaunchDescription(
        declared_args + [
            robot_state_publisher_node,
            control_node,
            delayed_controllers,
        ]
    )
