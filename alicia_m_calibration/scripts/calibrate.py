#!/usr/bin/env python3
"""Unified Alicia-M eye-in-hand and eye-to-hand calibration."""

import time

from calibration_common import (
    ArucoTracker, save_result, solve, transform, transform_from_msg)
from control_msgs.action import FollowJointTrajectory, GripperCommand
from cv_bridge import CvBridge
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import Constraints, JointConstraint
from moveit_msgs.srv import GetMotionPlan
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

JOINTS = [f'joint{index}' for index in range(1, 7)]
EYE_IN_HAND_POSES = [
    [0.00, -1.09, -0.87, 0.00, -0.77, 0.00],
    [-0.12, -1.09, -0.87, 0.00, -0.77, 0.00],
    [0.12, -1.09, -0.87, 0.00, -0.77, 0.00],
    [-0.20, -1.09, -0.87, 0.00, -0.77, 0.00],
    [0.00, -1.09, -0.87, 0.00, -0.77, 0.40],
    [0.00, -1.09, -0.87, 0.00, -0.77, -0.40],
    [0.00, -1.09, -0.87, 0.00, -0.77, 0.80],
    [0.00, -1.09, -0.87, 0.00, -0.65, 0.00],
    [0.00, -1.09, -0.87, 0.00, -0.90, 0.00],
    [0.00, -1.09, -0.87, 0.00, -0.55, 0.00],
    [0.00, -1.09, -0.87, 0.25, -0.77, 0.00],
    [0.00, -1.09, -0.87, -0.25, -0.77, 0.00],
    [0.00, -1.09, -0.87, 0.40, -0.77, 0.00],
    [0.15, -1.09, -0.87, 0.00, -0.77, 0.50],
    [-0.15, -1.09, -0.87, 0.00, -0.77, -0.50],
    [0.10, -1.09, -0.87, 0.00, -0.77, -0.60],
    [0.00, -1.09, -0.87, 0.20, -0.70, 0.30],
    [0.00, -1.09, -0.87, -0.20, -0.85, -0.30],
    [0.00, -1.09, -0.87, 0.15, -0.60, 0.50],
    [-0.10, -1.09, -0.87, -0.15, -0.70, -0.40],
]


def eye_to_hand_poses(arm_id):
    reference = (
        [-0.38, -1.34, -0.52, -0.24, 1.05, 0.02]
        if arm_id == 'slave'
        else [0.37, -1.18, -0.34, 0.15, 0.99, -0.11]
    )

    def pose(*offsets):
        offsets += (0.0,) * (6 - len(offsets))
        return [value + offset for value, offset in zip(reference, offsets)]

    return [
        pose(), pose(.08), pose(-.08), pose(.15), pose(-.15),
        pose(0, .06), pose(0, -.06), pose(0, .12),
        pose(0, 0, .06), pose(0, 0, -.06),
        pose(0, 0, 0, .10), pose(0, 0, 0, -.10),
        pose(0, 0, 0, 0, -.10), pose(0, 0, 0, 0, .10),
        pose(0, 0, 0, 0, 0, .20), pose(0, 0, 0, 0, 0, -.20),
        pose(0, 0, 0, 0, 0, .35), pose(.08, .06), pose(-.08, -.06),
        pose(.08, 0, 0, 0, -.08, .20),
        pose(-.08, 0, 0, 0, .08, -.20),
    ]


class Calibration(Node):
    def __init__(self):
        super().__init__('alicia_m_calibration')
        self.declare_parameter('mode', 'eye_in_hand')
        self.mode = self.get_parameter('mode').value
        if self.mode not in ('eye_in_hand', 'eye_to_hand'):
            raise ValueError('mode must be eye_in_hand or eye_to_hand')
        eye_in_hand = self.mode == 'eye_in_hand'
        defaults = {
            'aruco_dict': 'DICT_4X4_50',
            'marker_size': .0765 if eye_in_hand else .05,
            'marker_id': 0,
            'camera_topic': ('/camera/camera/color/image_raw' if eye_in_hand
                             else '/cam1/cam1/color/image_raw'),
            'camera_info_topic': ('/camera/camera/color/camera_info'
                                  if eye_in_hand
                                  else '/cam1/cam1/color/camera_info'),
            'base_link': 'base_link',
            'tool_link': 'tool0',
            'robot_model': 'alicia_m_v1_2_follower',
            'samples': 15 if eye_in_hand else 18,
            'method': 'horaud' if eye_in_hand else 'park',
            'output': ('hand_eye_calibration_result.yaml' if eye_in_hand
                       else 'eye_to_hand_calibration_result.yaml'),
            'arm_id': 'master',
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
            setattr(self, name, self.get_parameter(name).value)

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.tracker = ArucoTracker(
            self.aruco_dict, self.marker_id, self.marker_size,
            max_translation_std=.005 if eye_in_hand else .01)
        self.data = []
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, self.camera_topic, self.on_image, qos)
        self.create_subscription(
            CameraInfo, self.camera_info_topic, self.on_camera_info, 10)

        self.arm = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory')
        self.execute = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self.plan = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        self.gripper = ActionClient(
            self, GripperCommand, '/gripper_controller/gripper_cmd')
        self.poses = (EYE_IN_HAND_POSES if eye_in_hand
                      else eye_to_hand_poses(self.arm_id))
        self.get_logger().info(
            f'{self.mode}, marker={self.marker_id}, size={self.marker_size} m')

    def on_camera_info(self, message):
        if self.camera_matrix is None:
            self.camera_matrix = np.asarray(message.k).reshape(3, 3)
            self.dist_coeffs = np.asarray(message.d)

    def on_image(self, message):
        if self.camera_matrix is None:
            return
        image = self.bridge.imgmsg_to_cv2(message, 'bgr8')
        self.tracker.update(image, self.camera_matrix, self.dist_coeffs)

    def wait(self, future, timeout):
        start = time.monotonic()
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=.05)
            if time.monotonic() - start > timeout:
                return None
        return future.result() if future.done() else None

    def services_ready(self):
        if not (self.arm.wait_for_server(timeout_sec=10)
                and self.execute.wait_for_server(timeout_sec=10)
                and self.plan.wait_for_service(timeout_sec=10)):
            return False
        start = time.monotonic()
        while self.camera_matrix is None and time.monotonic() - start < 30:
            rclpy.spin_once(self, timeout_sec=.1)
        return self.camera_matrix is not None

    def plan_to(self, positions):
        request = GetMotionPlan.Request()
        motion = request.motion_plan_request
        motion.group_name = 'arm'
        motion.num_planning_attempts = 5
        motion.allowed_planning_time = 5.0
        motion.max_velocity_scaling_factor = .3
        motion.max_acceleration_scaling_factor = .3
        motion.start_state.is_diff = True
        constraints = Constraints(name='joint_goal')
        constraints.joint_constraints = [
            JointConstraint(joint_name=name, position=float(value),
                            tolerance_above=.01, tolerance_below=.01, weight=1.)
            for name, value in zip(JOINTS, positions)
        ]
        motion.goal_constraints = [constraints]
        response = self.wait(self.plan.call_async(request), 10)
        if response and response.motion_plan_response.error_code.val == 1:
            return response.motion_plan_response.trajectory
        return None

    def execute_trajectory(self, trajectory):
        handle = self.wait(self.execute.send_goal_async(
            ExecuteTrajectory.Goal(trajectory=trajectory)), 10)
        if not handle or not handle.accepted:
            return False
        point = trajectory.joint_trajectory.points[-1]
        timeout = point.time_from_start.sec + point.time_from_start.nanosec / 1e9 + 15
        result = self.wait(handle.get_result_async(), timeout)
        return bool(result and result.result.error_code.val == 1)

    def move(self, positions, duration=4.):
        trajectory = self.plan_to(positions)
        if trajectory is not None and self.execute_trajectory(trajectory):
            time.sleep(.5)
            return True
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory(
            joint_names=JOINTS,
            points=[JointTrajectoryPoint(
                positions=list(map(float, positions)),
                time_from_start=Duration(seconds=duration).to_msg())])
        handle = self.wait(self.arm.send_goal_async(goal), 10)
        if not handle or not handle.accepted:
            return False
        result = self.wait(handle.get_result_async(), duration + 10)
        return bool(result and result.result.error_code == 0)

    def open_gripper(self):
        if self.mode != 'eye_in_hand' or not self.gripper.wait_for_server(5):
            return
        goal = GripperCommand.Goal()
        goal.command.position = 0.
        goal.command.max_effort = 10.
        handle = self.wait(self.gripper.send_goal_async(goal), 5)
        if handle and handle.accepted:
            self.wait(handle.get_result_async(), 5)

    def sample(self):
        try:
            message = self.tf_buffer.lookup_transform(
                self.base_link, self.tool_link, rclpy.time.Time(),
                timeout=Duration(seconds=2))
        except Exception as error:
            self.get_logger().error(f'TF lookup failed: {error}')
            return False
        marker = self.tracker.stable_pose
        if marker is None:
            return False
        camera_target = transform(*marker)
        self.data.append((transform_from_msg(message), camera_target))
        self.get_logger().info(
            f'Sample {len(self.data)}/{self.samples}, '
            f'distance={np.linalg.norm(camera_target[:3, 3]):.3f} m')
        return True

    def collect(self):
        if not self.services_ready():
            return False
        self.open_gripper()
        if not self.move(self.poses[0]):
            return False
        for index, pose in enumerate(self.poses, 1):
            if len(self.data) >= self.samples:
                break
            self.get_logger().info(f'Pose {index}/{len(self.poses)}')
            if not self.move(pose):
                continue
            end = time.monotonic() + 2
            while time.monotonic() < end:
                rclpy.spin_once(self, timeout_sec=.05)
            self.tracker.reset()
            end = time.monotonic() + 5
            while time.monotonic() < end and self.tracker.stable_pose is None:
                rclpy.spin_once(self, timeout_sec=.05)
            self.sample()
        self.move(self.poses[0], 3.)
        return len(self.data) >= 3

    def run(self):
        if not self.collect():
            return False
        method, result, all_results = solve(self.data, self.mode, self.method)
        for name, item in all_results.items():
            self.get_logger().info(
                f'{name}: t={np.round(item["pose"][:3, 3], 4)}')
        path = save_result(
            self.output, mode=self.mode, pose=result['pose'],
            samples=len(self.data), method=method,
            dictionary=self.tracker.dictionary_name,
            marker_id=self.marker_id, marker_size=self.marker_size,
            robot_model=self.robot_model,
            quality=result.get('quality'),
            arm_id=self.arm_id if self.mode == 'eye_to_hand' else '')
        self.get_logger().info(f'Saved {path}')
        return True


def main():
    rclpy.init()
    node = Calibration()
    success = False
    try:
        success = node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
