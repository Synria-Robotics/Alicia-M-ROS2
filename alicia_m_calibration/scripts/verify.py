#!/usr/bin/python3
"""Publish a calibration result and fresh ArUco observations to MoveIt."""

import time

from calibration_common import (
    ArucoTracker, load_result, transform_from_msg)
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, TransformStamped
from moveit_msgs.msg import CollisionObject
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from shape_msgs.msg import SolidPrimitive
from tf2_ros import (
    Buffer, StaticTransformBroadcaster, TransformBroadcaster,
    TransformListener)


class Verification(Node):
    def __init__(self):
        super().__init__('calibration_verification')
        self.declare_parameter('result', '')
        self.declare_parameter('output', 'hand_eye_calibration_result.yaml')
        self.declare_parameter('marker_size', 0.0)
        self.declare_parameter('marker_id', -1)
        self.declare_parameter('camera_topic', '/camera/camera/color/image_raw')
        self.declare_parameter(
            'camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('base_link', 'base_link')
        self.declare_parameter('tool_link', 'tool0')
        self.declare_parameter('camera_link', '')
        self.declare_parameter('marker_frame', 'aruco_marker_frame')
        self.declare_parameter('collision_topic', '/collision_object')
        self.declare_parameter('collision_object_id', 'detected_aruco_marker')
        self.declare_parameter('marker_thickness', 0.002)
        self.declare_parameter('publish_collision_object', True)
        self.declare_parameter('collision_publish_period', 0.2)
        filename = (self.get_parameter('result').value
                    or self.get_parameter('output').value)
        self.path, self.record, self.parent_optical = load_result(filename)
        self.camera_matrix = None
        self.camera_frame = None
        self.dist_coeffs = None
        self.bridge = CvBridge()
        marker_size = self.get_parameter('marker_size').value
        marker_id = self.get_parameter('marker_id').value
        marker_size = (marker_size if marker_size > 0 else
                       self.record.get('aruco_marker_size', .0765))
        marker_id = (marker_id if marker_id >= 0 else
                     self.record.get('aruco_marker_id', 0))
        self.tracker = ArucoTracker(
            self.record.get('aruco_dictionary', 'DICT_4X4_50'),
            marker_id, marker_size, min_frames=1)
        self.get_logger().info(
            f'Verification marker={marker_id}, size={marker_size} m')
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.marker_broadcaster = TransformBroadcaster(self)
        self.marker_frame = self.get_parameter('marker_frame').value
        self.base_link = self.get_parameter('base_link').value
        self.tool_link = self.get_parameter('tool_link').value
        self.marker_size = float(marker_size)
        self.last_collision_time = None
        self.collision_publisher = self.create_publisher(
            CollisionObject,
            self.get_parameter('collision_topic').value, 10)
        self.timer = self.create_timer(.2, self.publish_calibration)
        self.camera_topic = self.get_parameter('camera_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self._resolve_camera_topics()
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(
            Image, self.camera_topic,
            self.on_image, qos)
        self.create_subscription(
            CameraInfo, self.camera_info_topic,
            self.on_camera_info, 10)

    def _topic_publishers(self, topic):
        return self.get_publishers_info_by_topic(topic)

    def _resolve_camera_topics(self):
        """Use the configured camera, or the one color camera that is publishing."""
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if (self._topic_publishers(self.camera_info_topic)
                    and self._topic_publishers(self.camera_topic)):
                return
            rclpy.spin_once(self, timeout_sec=0.1)
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

    @staticmethod
    def message(parent, child, pose, stamp):
        quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()
        result = TransformStamped()
        result.header.stamp = stamp
        result.header.frame_id = parent
        result.child_frame_id = child
        result.transform.translation.x = float(pose[0, 3])
        result.transform.translation.y = float(pose[1, 3])
        result.transform.translation.z = float(pose[2, 3])
        result.transform.rotation.x = float(quaternion[0])
        result.transform.rotation.y = float(quaternion[1])
        result.transform.rotation.z = float(quaternion[2])
        result.transform.rotation.w = float(quaternion[3])
        return result

    def publish_calibration(self):
        optical_frame = self.record['child_frame_id']
        camera_link = self.get_parameter('camera_link').value
        if not camera_link:
            suffix = '_color_optical_frame'
            camera_link = (optical_frame[:-len(suffix)] + '_link'
                           if optical_frame.endswith(suffix)
                           else optical_frame + '_link')
        try:
            internal = self.buffer.lookup_transform(
                camera_link, optical_frame, rclpy.time.Time())
        except Exception:
            return
        parent_link = self.parent_optical @ np.linalg.inv(
            transform_from_msg(internal))
        message = self.message(
            self.record['frame_id'], camera_link, parent_link,
            self.get_clock().now().to_msg())
        self.static_broadcaster.sendTransform(message)
        self.timer.cancel()
        self.get_logger().info(
            f'Published {message.header.frame_id} -> {camera_link} from '
            f'{self.path}')

    def calibrated_camera_pose(self, stamp):
        """Return base->current camera optical frame for this observation."""
        calibrated_frame = self.record['child_frame_id']
        current_frame = self.camera_frame or calibrated_frame
        observation_time = rclpy.time.Time.from_msg(stamp)
        calibrated_to_current = np.eye(4)
        if current_frame != calibrated_frame:
            relative = self.buffer.lookup_transform(
                calibrated_frame, current_frame, observation_time,
                timeout=Duration(seconds=.1))
            calibrated_to_current = transform_from_msg(relative)
        if self.record['type'] == 'eye_to_hand':
            return self.parent_optical @ calibrated_to_current
        try:
            base_to_tool = self.buffer.lookup_transform(
                self.base_link, self.tool_link, observation_time,
                timeout=Duration(seconds=.1))
        except Exception:
            # Some hardware publishes joint TF slightly behind the camera
            # timestamp.  Verification should still work with the freshest
            # robot pose instead of dropping every image.
            base_to_tool = self.buffer.lookup_transform(
                self.base_link, self.tool_link, rclpy.time.Time(),
                timeout=Duration(seconds=.1))
        return (transform_from_msg(base_to_tool) @ self.parent_optical @
                calibrated_to_current)

    @staticmethod
    def pose_message(pose):
        quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()
        result = Pose()
        result.position.x = float(pose[0, 3])
        result.position.y = float(pose[1, 3])
        result.position.z = float(pose[2, 3])
        result.orientation.x = float(quaternion[0])
        result.orientation.y = float(quaternion[1])
        result.orientation.z = float(quaternion[2])
        result.orientation.w = float(quaternion[3])
        return result

    def publish_to_moveit(self, marker_pose, stamp):
        if not self.get_parameter('publish_collision_object').value:
            return
        now = self.get_clock().now()
        period = self.get_parameter('collision_publish_period').value
        if (self.last_collision_time is not None and
                (now - self.last_collision_time).nanoseconds < period * 1e9):
            return
        self.last_collision_time = now
        collision = CollisionObject()
        collision.header.frame_id = self.base_link
        collision.header.stamp = stamp
        collision.id = self.get_parameter('collision_object_id').value
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [
            self.marker_size, self.marker_size,
            float(self.get_parameter('marker_thickness').value)]
        collision.primitives = [primitive]
        collision.primitive_poses = [self.pose_message(marker_pose)]
        collision.operation = CollisionObject.ADD
        self.collision_publisher.publish(collision)

    def on_image(self, message):
        if self.camera_matrix is None:
            return
        image = self.bridge.imgmsg_to_cv2(message, 'bgr8')
        detection = self.tracker.update(
            image, self.camera_matrix, self.dist_coeffs)
        if not detection.found:
            return
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec(detection.rvec).as_matrix()
        pose[:3, 3] = detection.tvec
        try:
            base_camera = self.calibrated_camera_pose(message.header.stamp)
            base_marker = base_camera @ pose
        except Exception as error:
            self.get_logger().warning(
                f'Cannot transform ArUco observation to {self.base_link}: '
                f'{error}', throttle_duration_sec=2.)
            return
        self.marker_broadcaster.sendTransform(self.message(
            self.base_link, self.marker_frame, base_marker,
            message.header.stamp))
        self.publish_to_moveit(base_marker, message.header.stamp)


def main():
    rclpy.init()
    node = Verification()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
