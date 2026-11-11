"""Shared ArUco, transform, solver, and result-file utilities."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import yaml


DICTIONARIES = {
    name: getattr(cv2.aruco, name)
    for name in (
        'DICT_4X4_50', 'DICT_4X4_100', 'DICT_4X4_250', 'DICT_4X4_1000',
        'DICT_5X5_50', 'DICT_5X5_100', 'DICT_5X5_250', 'DICT_5X5_1000',
        'DICT_6X6_50', 'DICT_6X6_100', 'DICT_6X6_250', 'DICT_6X6_1000',
        'DICT_7X7_50', 'DICT_7X7_100', 'DICT_7X7_250', 'DICT_7X7_1000',
        'DICT_ARUCO_ORIGINAL',
    )
}
AUTO_DICTIONARIES = (
    'DICT_4X4_50', 'DICT_5X5_50', 'DICT_6X6_50', 'DICT_7X7_50',
    'DICT_ARUCO_ORIGINAL',
)
METHODS = {
    'tsai': cv2.CALIB_HAND_EYE_TSAI,
    'park': cv2.CALIB_HAND_EYE_PARK,
    'horaud': cv2.CALIB_HAND_EYE_HORAUD,
    'daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS,
    'andreff': cv2.CALIB_HAND_EYE_ANDREFF,
}


@dataclass
class Detection:
    corners: list
    ids: np.ndarray | None
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None

    @property
    def found(self):
        return self.rvec is not None


class ArucoTracker:
    """Detect one marker and fuse a short history into a stable pose."""

    def __init__(self, dictionary, marker_id, marker_size, *, min_frames=5,
                 max_translation_std=0.005, history_size=10):
        self.dictionary_name = dictionary
        self.marker_id = int(marker_id)
        self.marker_size = float(marker_size)
        self.min_frames = min_frames
        self.max_translation_std = max_translation_std
        self.history_size = history_size
        self.history = []
        self.last = Detection([], None)
        self.params = (
            cv2.aruco.DetectorParameters_create()
            if hasattr(cv2.aruco, 'DetectorParameters_create')
            else cv2.aruco.DetectorParameters()
        )
        self.params.minMarkerDistanceRate = 0.03
        # Hand-eye translation is very sensitive to marker corner noise.
        # Refine detected corners before estimating the planar square pose.
        self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.params.cornerRefinementWinSize = 5
        self.params.cornerRefinementMaxIterations = 50
        self.params.cornerRefinementMinAccuracy = 0.01
        self.new_api = (
            hasattr(cv2.aruco, 'ArucoDetector')
            and tuple(map(int, cv2.__version__.split('.')[:2])) >= (4, 7)
        )
        self.auto = dictionary.lower() == 'auto'
        if not self.auto:
            self._select(dictionary if dictionary in DICTIONARIES
                         else 'DICT_4X4_50')

    def _select(self, name):
        self.dictionary_name = name
        self.dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARIES[name])
        self.detector = (
            cv2.aruco.ArucoDetector(self.dictionary, self.params)
            if self.new_api else None
        )

    def _detect(self, gray):
        if self.new_api:
            return self.detector.detectMarkers(gray)
        return cv2.aruco.detectMarkers(
            gray, self.dictionary, parameters=self.params)

    def _find(self, gray):
        if not self.auto:
            return self._detect(gray)
        for name in AUTO_DICTIONARIES:
            self._select(name)
            result = self._detect(gray)
            ids = result[1]
            if ids is not None and self.marker_id in ids.flatten():
                self.auto = False
                return result
        self.dictionary_name = 'auto'
        return [], None, []

    def update(self, image, camera_matrix, dist_coeffs):
        corners, ids, _ = self._find(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        result = Detection(corners, ids)
        if ids is not None:
            matches = np.flatnonzero(ids.flatten() == self.marker_id)
            if matches.size:
                half = self.marker_size / 2
                object_points = np.array([
                    [-half, half, 0], [half, half, 0],
                    [half, -half, 0], [-half, -half, 0],
                ], dtype=np.float64)
                image_points = np.asarray(
                    corners[int(matches[0])][0], dtype=np.float64)
                ok, rvec, tvec = cv2.solvePnP(
                    object_points, image_points,
                    camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ok:
                    rvec, tvec = cv2.solvePnPRefineLM(
                        object_points, image_points, camera_matrix,
                        dist_coeffs, rvec, tvec)
                    result.rvec = rvec.reshape(3)
                    result.tvec = tvec.reshape(3)
        self.last = result
        self.history.append((result.rvec, result.tvec))
        self.history = self.history[-self.history_size:]
        return result

    def reset(self):
        self.history.clear()

    @property
    def stable_pose(self):
        valid = [(r, t) for r, t in self.history if r is not None]
        if len(valid) < self.min_frames:
            return None
        rvecs, tvecs = map(np.asarray, zip(*valid))
        if np.max(np.std(tvecs, axis=0)) > self.max_translation_std:
            return None
        return (
            Rotation.from_rotvec(rvecs).mean().as_matrix(),
            np.median(tvecs, axis=0),
        )


def transform(rotation=np.eye(3), translation=np.zeros(3)):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(rotation).reshape(3, 3)
    result[:3, 3] = np.asarray(translation).reshape(3)
    return result


def transform_from_msg(message):
    t = message.transform.translation
    q = message.transform.rotation
    return transform(
        Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(),
        [t.x, t.y, t.z])


def solve(samples, mode, requested_method):
    """Run OpenCV methods and select the requested valid solution."""
    if len(samples) < 3:
        raise ValueError('at least three samples are required')
    if requested_method not in METHODS:
        raise ValueError(
            f'calibration method must be one of: {", ".join(METHODS)}')
    mounts = [sample[0] for sample in samples]
    targets = [sample[1] for sample in samples]
    grippers = ([np.linalg.inv(pose) for pose in mounts]
                if mode == 'eye_to_hand' else mounts)
    results = {}
    for name, method in METHODS.items():
        try:
            rotation, translation = cv2.calibrateHandEye(
                [pose[:3, :3] for pose in grippers],
                [pose[:3, 3].reshape(3, 1) for pose in grippers],
                [pose[:3, :3] for pose in targets],
                [pose[:3, 3].reshape(3, 1) for pose in targets],
                method=method)
            pose = transform(rotation, translation)
            item = {'pose': pose}
            # The invariant pose is base->target for eye-in-hand and
            # tool->target for eye-to-hand.  Measuring its dispersion gives
            # both modes the same meaningful residual quality metric.
            invariants = (
                [mount @ pose @ target for mount, target in samples]
                if mode == 'eye_in_hand' else
                [np.linalg.inv(mount) @ pose @ target
                 for mount, target in samples]
            )
            xyz = np.array([value[:3, 3] for value in invariants])
            t_rmse = np.sqrt(np.mean(np.sum(
                (xyz - xyz.mean(axis=0)) ** 2, axis=1)))
            rotations = Rotation.from_matrix(
                [value[:3, :3] for value in invariants])
            mean_rotation = rotations.mean()
            r_error = (mean_rotation.inv() * rotations).magnitude()
            r_rmse = np.sqrt(np.mean(r_error ** 2))
            item['quality'] = {
                'translation_rmse_m': float(t_rmse),
                'rotation_rmse_deg': float(np.degrees(r_rmse)),
            }
            item['invariant_pose'] = transform(
                mean_rotation.as_matrix(), xyz.mean(axis=0))
            results[name] = item
        except cv2.error:
            continue
    if requested_method not in results:
        raise RuntimeError(
            f'requested calibration method {requested_method!r} failed')
    return requested_method, results[requested_method], results


def package_config(filename):
    path = Path(filename).expanduser()
    if path.is_absolute():
        return path
    share = Path(get_package_share_directory('alicia_m_calibration'))
    source = share.parents[3] / 'alicia_m_calibration' / 'config' / path
    return source if source.parent.exists() else share / 'config' / path


def save_result(filename, *, mode, pose, samples, method, dictionary,
                marker_id, marker_size, robot_model, quality=None, arm_id='',
                camera_frame='camera_color_optical_frame'):
    rotation = Rotation.from_matrix(pose[:3, :3])
    quaternion = rotation.as_quat()
    euler = rotation.as_euler('xyz', degrees=True)
    if mode == 'eye_in_hand':
        key, parent, child = (
            'hand_eye_calibration', 'tool0', camera_frame)
    else:
        key, parent, child = (
            'eye_to_hand_calibration', 'base_link', camera_frame)
    record = {
        'type': mode,
        'frame_id': parent,
        'child_frame_id': child,
        'robot_model': robot_model,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'samples_count': samples,
        'calibration_method': method,
        'aruco_dictionary': dictionary,
        'aruco_marker_id': marker_id,
        'aruco_marker_size': marker_size,
        'transform': {
            'translation': dict(zip('xyz', map(float, pose[:3, 3]))),
            'rotation': {
                'quaternion': dict(zip('xyzw', map(float, quaternion))),
                'euler_xyz_deg': dict(zip(
                    ('roll', 'pitch', 'yaw'), map(float, euler))),
            },
            'matrix_4x4': pose.tolist(),
        },
    }
    if quality:
        record['quality'] = quality
    if arm_id:
        record['arm_id'] = arm_id
    path = package_config(filename)
    path.write_text(yaml.safe_dump({key: record}, sort_keys=False),
                    encoding='utf-8')
    return path


def load_result(filename):
    path = package_config(filename)
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    key = next(key for key in (
        'hand_eye_calibration', 'eye_to_hand_calibration') if key in data)
    record = data[key]
    value = record['transform']
    xyz = value['translation']
    quat = value['rotation']['quaternion']
    pose = transform(
        Rotation.from_quat([quat[key] for key in 'xyzw']).as_matrix(),
        [xyz[key] for key in 'xyz'])
    return path, record, pose
