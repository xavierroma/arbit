"""
Type definitions for Arbit Python bindings
"""

from ctypes import Structure, c_double, c_float, c_uint32, c_uint64, c_ulong, c_bool, c_uint8, POINTER
from enum import IntEnum
from typing import List, Optional
import numpy as np


class PixelFormat(IntEnum):
    """Pixel format enumeration"""
    BGRA8 = 0
    RGBA8 = 1
    NV12 = 2
    YV12 = 3
    DEPTH16 = 4


class MotionState(IntEnum):
    """Motion state enumeration"""
    STATIONARY = 0
    SLOW = 1
    FAST = 2


class TrackStatus(IntEnum):
    """Track status enumeration"""
    CONVERGED = 0
    DIVERGED = 1
    OUT_OF_BOUNDS = 2


class ArbitCameraIntrinsics(Structure):
    """Camera intrinsics structure"""
    _fields_ = [
        ("fx", c_double),
        ("fy", c_double),
        ("cx", c_double),
        ("cy", c_double),
        ("skew", c_double),
        ("width", c_uint32),
        ("height", c_uint32),
        ("distortion_len", c_ulong),
        ("distortion", POINTER(c_double)),
    ]


class ArbitFrameTimestamps(Structure):
    """Frame timestamp structure"""
    _fields_ = [
        ("capture_seconds", c_double),
        ("pipeline_seconds", c_double),
        ("latency_seconds", c_double),
    ]


class ArbitCameraFrame(Structure):
    """Camera frame structure"""
    _fields_ = [
        ("timestamp_seconds", c_double),
        ("intrinsics", ArbitCameraIntrinsics),
        ("pixel_format", c_uint32),
        ("bytes_per_row", c_ulong),
        ("data", POINTER(c_uint8)),
        ("data_len", c_ulong),
    ]


class ArbitCameraSample(Structure):
    """Camera sample output structure"""
    _fields_ = [
        ("timestamps", ArbitFrameTimestamps),
        ("intrinsics", ArbitCameraIntrinsics),
        ("pixel_format", c_uint32),
        ("bytes_per_row", c_ulong),
    ]


class ArbitImuSample(Structure):
    """IMU sample structure"""
    _fields_ = [
        ("timestamp_seconds", c_double),
        ("accel_x", c_double),
        ("accel_y", c_double),
        ("accel_z", c_double),
        ("gyro_x", c_double),
        ("gyro_y", c_double),
        ("gyro_z", c_double),
    ]


class ArbitTransform(Structure):
    """4x4 transformation matrix (column-major)"""
    _fields_ = [
        ("elements", c_double * 16),
    ]
    
    def to_matrix(self) -> np.ndarray:
        """Convert to numpy 4x4 matrix"""
        return np.array(self.elements).reshape((4, 4))
    
    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "ArbitTransform":
        """Create from numpy 4x4 matrix"""
        if matrix.shape != (4, 4):
            raise ValueError("Matrix must be 4x4")
        transform = cls()
        for i, val in enumerate(matrix.flatten()):
            transform.elements[i] = float(val)
        return transform
    
    @classmethod
    def identity(cls) -> "ArbitTransform":
        """Create identity transform"""
        transform = cls()
        transform.elements[0] = 1.0
        transform.elements[5] = 1.0
        transform.elements[10] = 1.0
        transform.elements[15] = 1.0
        return transform


class ArbitImuState(Structure):
    """IMU state structure"""
    _fields_ = [
        ("has_rotation_prior", c_bool),
        ("rotation_prior_radians", c_double),
        ("has_motion_state", c_bool),
        ("motion_state", c_uint32),  # See MotionState enum
        ("has_gravity", c_bool),
        ("gravity_down", c_double * 3),
        ("gravity_samples", c_uint32),
        ("preintegration_count", c_uint32),
    ]


class ArbitTwoViewSummary(Structure):
    """Two-view initialization summary"""
    _fields_ = [
        ("inliers", c_uint32),
        ("average_error", c_double),
        ("rotation", c_double * 9),
        ("translation", c_double * 3),
    ]


class ArbitRelocalizationSummary(Structure):
    """Relocalization summary"""
    _fields_ = [
        ("pose", ArbitTransform),
        ("inliers", c_uint32),
        ("average_error", c_double),
    ]


class ArbitFrameState(Structure):
    """Comprehensive frame state"""
    _fields_ = [
        ("track_count", c_uint32),
        ("has_two_view", c_bool),
        ("two_view", ArbitTwoViewSummary),
        ("has_relocalization", c_bool),
        ("relocalization", ArbitRelocalizationSummary),
        ("keyframe_count", c_uint64),
        ("landmark_count", c_uint64),
        ("anchor_count", c_uint64),
        ("imu", ArbitImuState),
    ]


class ArbitTrackedPoint(Structure):
    """Tracked feature point"""
    _fields_ = [
        ("initial_x", c_float),
        ("initial_y", c_float),
        ("refined_x", c_float),
        ("refined_y", c_float),
        ("residual", c_float),
        ("iterations", c_uint32),
        ("status", c_uint32),  # See TrackStatus enum
        ("track_id", c_uint64),
    ]

class ArbitFeatDescriptor(Structure):
    """Raw feature descriptor from the FFI layer"""
    _fields_ = [
        ("level", c_uint32),
        ("seed_x", c_float),
        ("seed_y", c_float),
        ("score", c_float),
        ("angle", c_float),
        ("data_len", c_ulong),
        ("data", c_uint8 * 32),
    ]


class ArbitMatch(Structure):
    """Raw feature match from the FFI layer"""
    _fields_ = [
        ("query_idx", c_uint32),
        ("train_idx", c_uint32),
        ("distance", c_uint32),
        ("query_x", c_float),
        ("query_y", c_float),
        ("train_x", c_float),
        ("train_y", c_float),
    ]


class FeatDescriptor:
    """Python-friendly feature descriptor wrapper"""

    def __init__(self, native: ArbitFeatDescriptor):
        self._native = native

    @property
    def level(self) -> int:
        """Image pyramid level where the feature was detected"""
        return int(self._native.level)

    @property
    def position(self) -> np.ndarray:
        """Feature position `(x, y)` in pixel coordinates"""
        return np.array([self._native.seed_x, self._native.seed_y], dtype=np.float32)

    @property
    def score(self) -> float:
        """Detector response score"""
        return float(self._native.score)

    @property
    def angle(self) -> float:
        """Descriptor orientation in radians"""
        return float(self._native.angle)

    @property
    def angle_degrees(self) -> float:
        """Descriptor orientation in degrees"""
        return float(np.degrees(self._native.angle))

    @property
    def data_len(self) -> int:
        """Number of valid bytes in the descriptor"""
        return min(int(self._native.data_len), len(self._native.data))

    @property
    def data(self) -> np.ndarray:
        """Descriptor bytes as a NumPy array"""
        length = self.data_len
        return np.array(self._native.data[:length], dtype=np.uint8)

    def bytes(self) -> bytes:
        """Descriptor bytes as a Python bytes object"""
        length = self.data_len
        return bytes(self._native.data[:length])

class ArbitPoseSample(Structure):
    """Trajectory pose sample"""
    _fields_ = [
        ("x", c_double),
        ("y", c_double),
        ("z", c_double),
    ]


class ArbitProjectedAnchor(Structure):
    """Anchor projected into camera frame"""
    _fields_ = [
        ("anchor_id", c_uint64),
        ("pose", ArbitTransform),
        ("created_from_keyframe", c_uint64),
        ("has_keyframe", c_bool),
        ("normalized_u", c_double),
        ("normalized_v", c_double),
        ("pixel_x", c_float),
        ("pixel_y", c_float),
        ("depth", c_double),
    ]


class ArbitProjectedLandmark(Structure):
    """Landmark projected into camera frame"""
    _fields_ = [
        ("landmark_id", c_uint64),
        ("world_x", c_double),
        ("world_y", c_double),
        ("world_z", c_double),
        ("normalized_u", c_double),
        ("normalized_v", c_double),
        ("pixel_x", c_float),
        ("pixel_y", c_float),
        ("depth", c_double),
    ]


# Python-friendly wrappers
class Transform:
    """Python-friendly transform wrapper"""
    def __init__(self, matrix: Optional[np.ndarray] = None):
        if matrix is None:
            self._native = ArbitTransform.identity()
        else:
            self._native = ArbitTransform.from_matrix(matrix)
    
    @property
    def matrix(self) -> np.ndarray:
        """Get 4x4 transformation matrix"""
        return self._native.to_matrix()
    
    @property
    def translation(self) -> np.ndarray:
        """Get translation vector [x, y, z]"""
        m = self.matrix
        return m[:3, 3]
    
    @property
    def rotation(self) -> np.ndarray:
        """Get 3x3 rotation matrix"""
        m = self.matrix
        return m[:3, :3]


class FrameState:
    """Python-friendly frame state wrapper"""
    def __init__(self, native: ArbitFrameState):
        self._native = native
    
    @property
    def track_count(self) -> int:
        return self._native.track_count
    
    @property
    def keyframe_count(self) -> int:
        return self._native.keyframe_count
    
    @property
    def landmark_count(self) -> int:
        return self._native.landmark_count
    
    @property
    def anchor_count(self) -> int:
        return self._native.anchor_count
    
    @property
    def has_two_view(self) -> bool:
        return self._native.has_two_view
    
    @property
    def imu_state(self) -> "ImuState":
        return ImuState(self._native.imu)


class ImuState:
    """Python-friendly IMU state wrapper"""
    def __init__(self, native: ArbitImuState):
        self._native = native
    
    @property
    def has_gravity(self) -> bool:
        return self._native.has_gravity
    
    @property
    def gravity_down(self) -> Optional[np.ndarray]:
        if not self.has_gravity:
            return None
        return np.array([
            self._native.gravity_down[0],
            self._native.gravity_down[1],
            self._native.gravity_down[2],
        ])
    
    @property
    def motion_state(self) -> Optional[MotionState]:
        if not self._native.has_motion_state:
            return None
        return MotionState(self._native.motion_state)
    
    @property
    def rotation_prior_degrees(self) -> Optional[float]:
        if not self._native.has_rotation_prior:
            return None
        return np.degrees(self._native.rotation_prior_radians)


class Match:
    """Python-friendly feature match wrapper"""
    def __init__(self, native: ArbitMatch):
        self._native = native
    
    @property
    def query_idx(self) -> int:
        """Index of the query descriptor"""
        return int(self._native.query_idx)
    
    @property
    def train_idx(self) -> int:
        """Index of the train descriptor"""
        return int(self._native.train_idx)
    
    @property
    def distance(self) -> int:
        """Hamming distance between descriptors"""
        return int(self._native.distance)
    
    @property
    def query_position(self) -> np.ndarray:
        """Query descriptor position (x, y)"""
        return np.array([self._native.query_x, self._native.query_y], dtype=np.float32)
    
    @property
    def train_position(self) -> np.ndarray:
        """Train descriptor position (x, y)"""
        return np.array([self._native.train_x, self._native.train_y], dtype=np.float32)
    
    def __repr__(self) -> str:
        return f"Match(query_idx={self.query_idx}, train_idx={self.train_idx}, distance={self.distance})"
