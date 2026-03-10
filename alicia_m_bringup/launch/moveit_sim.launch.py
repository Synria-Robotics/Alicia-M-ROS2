"""
moveit_sim.launch.py - 仅启动 MoveIt2 RViz 仿真（无真实硬件）

使用 mock_components/GenericSystem 作为虚拟硬件，
可在无机械臂的情况下进行运动规划测试和可视化。

启动内容:
  - robot_state_publisher (加载带 mock 硬件插件的 URDF)
  - ros2_control controller_manager (mock 硬件)
  - static_transform_publisher (virtual_joint: world -> base_link)
  - move_group (MoveIt2 运动规划)
  - rviz2 (MoveIt2 可视化)

用法:
  ros2 launch alicia_m_bringup moveit_sim.launch.py
"""

import os
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import (
    Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def load_yaml(package_name, file_path):
    """从包中加载 YAML 文件"""
    full_path = os.path.join(
        get_package_share_directory(package_name), file_path)
    with open(full_path, 'r') as f:
        return yaml.safe_load(f)


def generate_launch_description():
    # ======================== 启动参数 ========================
    declared_args = [
        DeclareLaunchArgument(
            'use_rviz', default_value='true',
            description='是否启动 RViz'),
    ]

    use_rviz = LaunchConfiguration('use_rviz')

    # ======================== 包路径 ========================
    moveit_config_pkg = 'alicia_m_moveit_config'
    moveit_config_share = FindPackageShare(moveit_config_pkg)
    bringup_share = FindPackageShare('alicia_m_bringup')

    # ======================== URDF (Mock 硬件) ========================
    robot_description_content = Command([
        FindExecutable(name='xacro'), ' ',
        PathJoinSubstitution([
            moveit_config_share, 'config',
            'alicia_m_v1_1_follower.urdf.xacro'
        ]),
        ' ros2_control_plugin:=mock_components/GenericSystem',
    ])
    robot_description = {
        'robot_description': ParameterValue(robot_description_content, value_type=str)
    }

    # ======================== MoveIt2 配置 ========================
    robot_description_semantic_content = Command([
        FindExecutable(name='xacro'), ' ',
        PathJoinSubstitution([
            moveit_config_share, 'config',
            'alicia_m_v1_1_follower.srdf'
        ]),
    ])
    robot_description_semantic = {
        'robot_description_semantic': ParameterValue(
            robot_description_semantic_content, value_type=str)
    }

    kinematics_yaml = load_yaml(moveit_config_pkg, 'config/kinematics.yaml')
    joint_limits_yaml = load_yaml(moveit_config_pkg, 'config/joint_limits.yaml')
    pilz_cartesian_limits_yaml = load_yaml(
        moveit_config_pkg, 'config/pilz_cartesian_limits.yaml')

    planning_pipeline_config = {
        'default_planning_pipeline': 'ompl',
        'planning_pipelines': ['ompl', 'pilz_industrial_motion_planner'],
        'ompl': {
            'planning_plugin': 'ompl_interface/OMPLPlanner',
            'request_adapters':
                'default_planner_request_adapters/AddTimeOptimalParameterization '
                'default_planner_request_adapters/ResolveConstraintFrames '
                'default_planner_request_adapters/FixWorkspaceBounds '
                'default_planner_request_adapters/FixStartStateBounds '
                'default_planner_request_adapters/FixStartStateCollision '
                'default_planner_request_adapters/FixStartStatePathConstraints',
            'start_state_max_bounds_error': 0.1,
        },
        'pilz_industrial_motion_planner': {
            'planning_plugin': 'pilz_industrial_motion_planner/CommandPlanner',
            'request_adapters': '',
        },
    }

    moveit_controllers_yaml = load_yaml(
        moveit_config_pkg, 'config/moveit_controllers.yaml')

    trajectory_execution = {
        'moveit_manage_controllers': True,
        'trajectory_execution.allowed_execution_duration_scaling': 1.2,
        'trajectory_execution.allowed_goal_duration_margin': 0.5,
        'trajectory_execution.allowed_start_tolerance': 0.01,
    }

    planning_scene_monitor_parameters = {
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
    }

    # ======================== 控制器配置 ========================
    controllers_yaml = PathJoinSubstitution([
        bringup_share, 'config', 'ros2_controllers.yaml'
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

    # 虚拟关节 TF (world -> base_link)
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'world', 'base_link'],
        output='screen',
    )

    # MoveIt2 move_group
    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml,
            planning_pipeline_config,
            trajectory_execution,
            moveit_controllers_yaml,
            planning_scene_monitor_parameters,
            joint_limits_yaml,
            pilz_cartesian_limits_yaml,
        ],
    )

    # RViz
    rviz_config_file = PathJoinSubstitution([
        moveit_config_share, 'config', 'moveit.rviz'
    ])
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml,
            planning_pipeline_config,
            joint_limits_yaml,
        ],
        output='screen',
        condition=IfCondition(use_rviz),
    )

    # 延迟启动控制器
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
            static_tf_node,
            move_group_node,
            rviz_node,
            delayed_controllers,
        ]
    )
