#!/usr/bin/env python3
"""Publish a calibration result and the observed ArUco marker TF."""

from calibration_common import (
    ArucoTracker, load_result, transform_from_msg)
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import (
    Buffer, StaticTransformBroadcaster, TransformBroadcaster,
    TransformListener)


class Verification(Node):
    def __init__(self):
        super().__init__('calibration_verification')
        self.declare_parameter('result', '')
        self.declare_parameter('output', 'hand_eye_calibration_result.yaml')
        self.declare_parameter('camera_topic', '/camera/camera/color/image_raw')
        self.declare_parameter(
            'camera_info_topic', '/camera/camera/color/camera_info')
        filename = (self.get_parameter('result').value
                    or self.get_parameter('output').value)
        self.path, self.record, self.parent_optical = load_result(filename)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.bridge = CvBridge()
        self.tracker = ArucoTracker(
            self.record.get('aruco_dictionary', 'DICT_4X4_50'),
            self.record.get('aruco_marker_id', 0),
            self.record.get('aruco_marker_size', .0765), min_frames=1)
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.marker_broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(.2, self.publish_calibration)
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(
            Image, self.get_parameter('camera_topic').value,
            self.on_image, qos)
        self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_topic').value,
            self.on_camera_info, 10)

    def on_camera_info(self, message):
        if self.camera_matrix is None:
            self.camera_matrix = np.asarray(message.k).reshape(3, 3)
            self.dist_coeffs = np.asarray(message.d)

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
        try:
            internal = self.buffer.lookup_transform(
                'camera_link', 'camera_color_optical_frame', rclpy.time.Time())
        except Exception:
            return
        parent_link = self.parent_optical @ np.linalg.inv(
            transform_from_msg(internal))
        message = self.message(
            self.record['frame_id'], 'camera_link', parent_link,
            self.get_clock().now().to_msg())
        self.static_broadcaster.sendTransform(message)
        self.timer.cancel()
        self.get_logger().info(
            f'Published {message.header.frame_id} -> camera_link from {self.path}')

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
        self.marker_broadcaster.sendTransform(self.message(
            'camera_color_optical_frame', 'aruco_marker_frame',
            pose, message.header.stamp))


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
