#!/usr/bin/python3
"""Unified Alicia-M eye-in-hand and eye-to-hand calibration."""

import time

from calibration_common import (
    METHODS, ArucoTracker, save_result, solve, transform, transform_from_msg)
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
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

JOINTS = [f'joint{index}' for index in range(1, 7)]
DEFAULT_REFERENCE_POSES = {
    'eye_in_hand': [0.00, -1.09, -0.87, 0.00, -0.77, 0.00],
    'eye_to_hand': [0.70, -1.40, -0.34, 0.15, 0.99, -0.11],
}

# Maximum safe offset from the start pose for each joint.  Zero keeps a joint
# fixed.  These are motion envelopes, not a fixed list of calibration poses.
POSE_AMPLITUDES = {
    'eye_in_hand': [0.16, 0.00, 0.00, 0.24, 0.13, 0.50],
    'eye_to_hand': [0.15, 0.12, 0.06, 0.10, 0.10, 0.35],
}


def generate_calibration_poses(start, amplitudes, samples):
    """Generate diverse joint targets around one user-provided start pose."""
    start = np.asarray(start, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)
    if start.shape != (6,) or amplitudes.shape != (6,):
        raise ValueError('start pose and amplitudes must contain six joints')
    active = np.flatnonzero(amplitudes).tolist()
    if len(active) < 2:
        raise ValueError('at least two joints need non-zero pose amplitudes')

    # Generate extra candidates so missed ArUco detections do not force a
    # complete restart. Collection still stops at the requested sample count.
    candidate_count = max(int(samples) + 8, (int(samples) * 3 + 1) // 2)
    poses = [start.tolist()]

    # Single-axis motions establish strong, independently observable rotation.
    for scale in (0.5, -0.5, 1.0, -1.0):
        for axis in active:
            pose = start.copy()
            pose[axis] += amplitudes[axis] * scale
            poses.append(pose.tolist())
            if len(poses) >= candidate_count:
                return poses

    # Add automatically generated coupled motions only when more candidates
    # are needed. Half-range offsets keep the target near the start view.
    pairs = [(a, b) for index, a in enumerate(active)
             for b in active[index + 1:]]
    patterns = ((0.5, 0.5), (0.5, -0.5),
                (-0.5, 0.5), (-0.5, -0.5))
    for first_scale, second_scale in patterns:
        for first, second in pairs:
            pose = start.copy()
            pose[first] += amplitudes[first] * first_scale
            pose[second] += amplitudes[second] * second_scale
            poses.append(pose.tolist())
            if len(poses) >= candidate_count:
                return poses
    return poses


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
            'marker_size': .0765,
            'marker_id': 0,
            'camera_topic': ('/camera/camera/color/image_raw' if eye_in_hand
                             else '/camera/cam1/color/image_raw'),
            'camera_info_topic': ('/camera/camera/color/camera_info'
                                  if eye_in_hand
                                  else '/camera/cam1/color/camera_info'),
            'base_link': 'base_link',
            'tool_link': 'tool0',
            'robot_model': 'alicia_m_v1_2_follower',
            'reference_pose': DEFAULT_REFERENCE_POSES[self.mode],
            'pose_amplitudes': POSE_AMPLITUDES[self.mode],
            'samples': 15 if eye_in_hand else 18,
            'method': '',
            'output': ('hand_eye_calibration_result.yaml' if eye_in_hand
                       else 'eye_to_hand_calibration_result.yaml'),
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
            setattr(self, name, self.get_parameter(name).value)
        if self.method not in METHODS:
            choices = ', '.join(METHODS)
            raise ValueError(
                f'method must be explicitly set to one of: {choices}')

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.camera_frame = None
        self.dist_coeffs = None
        self.joint_positions = None
        self.tracker = ArucoTracker(
            self.aruco_dict, self.marker_id, self.marker_size,
            max_translation_std=.005 if eye_in_hand else .01)
        self.data = []
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._resolve_camera_topics()
        image_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                               history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(
            Image, self.camera_topic, self.on_image,
            self._publisher_qos(self.camera_topic, image_qos))
        self.create_subscription(
            CameraInfo, self.camera_info_topic, self.on_camera_info,
            self._publisher_qos(self.camera_info_topic, 10))
        self.create_subscription(JointState, '/joint_states',
                                 self.on_joint_state, 10)

        self.arm = ActionClient(
            self, FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory')
        self.execute = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self.plan = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        self.gripper = ActionClient(
            self, GripperCommand, '/gripper_controller/gripper_cmd')
        self.poses = generate_calibration_poses(
            self.reference_pose, self.pose_amplitudes, self.samples)
        self.get_logger().info(
            'Calibration parameters: '
            f'mode={self.mode}, aruco_dict={self.aruco_dict}, '
            f'marker_id={self.marker_id}, marker_size={self.marker_size} m, '
            f'camera_topic={self.camera_topic}, '
            f'camera_info_topic={self.camera_info_topic}, '
            f'base_link={self.base_link}, tool_link={self.tool_link}, '
            f'robot_model={self.robot_model}, samples={self.samples}, '
            f'method={self.method}, '
            f'reference_pose={np.round(self.reference_pose, 3)}, '
            f'candidate_poses={len(self.poses)}, output={self.output}')

    def _topic_publishers(self, topic):
        return self.get_publishers_info_by_topic(topic)

    def _publisher_qos(self, topic, default):
        publishers = self._topic_publishers(topic)
        if not publishers:
            return default
        profile = publishers[0].qos_profile
        if profile.history == HistoryPolicy.UNKNOWN:
            profile.history = HistoryPolicy.KEEP_LAST
        if profile.depth in (0, None):
            profile.depth = 10
        return profile

    def _resolve_camera_topics(self):
        """Use the configured camera, or the one color camera that is publishing."""
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if (self._topic_publishers(self.camera_info_topic)
                    and self._topic_publishers(self.camera_topic)):
                return
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._topic_publishers(self.camera_info_topic):
            return
        candidates = []
        for name, types in self.get_topic_names_and_types():
            if (name.endswith('/color/camera_info')
                    and 'sensor_msgs/msg/CameraInfo' in types
                    and self._topic_publishers(name)):
                candidates.append(name)
        if len(candidates) != 1:
            return
        info = candidates[0]
        image = info[:-len('camera_info')] + 'image_raw'
        if not self._topic_publishers(image):
            return
        self.get_logger().warning(
            f'No publisher on {self.camera_info_topic}; using {info}')
        self.camera_info_topic = info
        self.camera_topic = image

    def on_camera_info(self, message):
        if self.camera_matrix is None:
            self.camera_matrix = np.asarray(message.k).reshape(3, 3)
            self.dist_coeffs = np.asarray(message.d)
            self.camera_frame = message.header.frame_id

    def on_image(self, message):
        if self.camera_matrix is None:
            return
        image = self.bridge.imgmsg_to_cv2(message, 'bgr8')
        self.tracker.update(image, self.camera_matrix, self.dist_coeffs)

    def on_joint_state(self, message):
        positions = dict(zip(message.name, message.position))
        if all(name in positions for name in JOINTS):
            self.joint_positions = np.asarray(
                [positions[name] for name in JOINTS])

    def wait_until_reached(self, target, timeout=3., tolerance=.03):
        target = np.asarray(target)
        end = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=.05)
            if (self.joint_positions is not None and
                    np.max(np.abs(self.joint_positions - target)) <= tolerance):
                return True
        if self.joint_positions is None:
            self.get_logger().error('No /joint_states feedback received')
        else:
            error = np.abs(self.joint_positions - target)
            self.get_logger().error(
                f'Robot did not reach target; max joint error='
                f'{np.max(error):.3f} rad, errors={np.round(error, 3)}')
        return False

    def wait(self, future, timeout):
        start = time.monotonic()
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=.05)
            if time.monotonic() - start > timeout:
                return None
        return future.result() if future.done() else None

    def services_ready(self):
        missing = []
        if not self.arm.wait_for_server(timeout_sec=10):
            missing.append('/arm_controller/follow_joint_trajectory')
        if not self.execute.wait_for_server(timeout_sec=10):
            missing.append('/execute_trajectory')
        if not self.plan.wait_for_service(timeout_sec=10):
            missing.append('/plan_kinematic_path')
        if missing:
            self.get_logger().error(
                'MoveIt interfaces are not available: ' + ', '.join(missing))
            return False
        start = time.monotonic()
        while self.camera_matrix is None and time.monotonic() - start < 30:
            rclpy.spin_once(self, timeout_sec=.1)
        if self.camera_matrix is None:
            self.get_logger().error(
                f'No CameraInfo on {self.camera_info_topic}. '
                f'Expected images on {self.camera_topic}.')
            return False
        return True

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
        return bool(result and result.result.error_code.val == 1 and
                    self.wait_until_reached(point.positions))

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
        return bool(result and result.result.error_code == 0 and
                    self.wait_until_reached(positions))

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
            self.get_logger().error('Failed to reach the first calibration pose')
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
            if not self.sample():
                self.get_logger().warning(
                    f'Pose {index}: no stable ArUco detection; sample skipped')
        self.move(self.poses[0], 3.)
        if len(self.data) < self.samples:
            self.get_logger().error(
                f'Collected only {len(self.data)}/{self.samples} valid samples; '
                'calibration result was not saved')
            return False
        return True

    def run(self):
        if not self.collect():
            return False
        method, result, all_results = solve(self.data, self.mode, self.method)
        calibration_label = ('tool_camera_xyz_m' if self.mode == 'eye_in_hand'
                             else 'base_camera_xyz_m')
        invariant_label = ('base_marker_xyz_m' if self.mode == 'eye_in_hand'
                           else 'tool_marker_xyz_m')
        for name, item in all_results.items():
            quality = item.get('quality', {})
            self.get_logger().info(
                f'{name}: {calibration_label}='
                f'{np.round(item["pose"][:3, 3], 4)}, '
                f'{invariant_label}='
                f'{np.round(item["invariant_pose"][:3, 3], 4)}, '
                f't_rmse={quality.get("translation_rmse_m", float("nan")):.6f} m, '
                f'r_rmse={quality.get("rotation_rmse_deg", float("nan")):.3f} deg')
        self.get_logger().info(f'Selected calibration method: {method}')
        path = save_result(
            self.output, mode=self.mode, pose=result['pose'],
            samples=len(self.data), method=method,
            dictionary=self.tracker.dictionary_name,
            marker_id=self.marker_id, marker_size=self.marker_size,
            robot_model=self.robot_model, quality=result.get('quality'),
            camera_frame=self.camera_frame or 'camera_color_optical_frame')
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
