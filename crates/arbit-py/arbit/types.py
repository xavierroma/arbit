"""
Type definitions for Arbit Python bindings (v2 API).
"""

from __future__ import annotations

from ctypes import (
    Structure,
    POINTER,
    c_bool,
    c_double,
    c_float,
    c_uint8,
    c_uint32,
    c_uint64,
    c_ulong,
)
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np


class PixelFormat(IntEnum):
    BGRA8 = 0
    RGBA8 = 1
    NV12 = 2
    YV12 = 3
    DEPTH16 = 4


class TrackingState(IntEnum):
    INITIALIZING = 0
    TRACKING = 1
    RELOCALIZING = 2
    LOST = 3


class ArbitV2CameraIntrinsics(Structure):
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


class ArbitV2CameraFrame(Structure):
    _fields_ = [
        ("timestamp_seconds", c_double),
        ("intrinsics", ArbitV2CameraIntrinsics),
        ("pixel_format", c_uint32),
        ("bytes_per_row", c_ulong),
        ("data", POINTER(c_uint8)),
        ("data_len", c_ulong),
    ]


class ArbitV2ImuSample(Structure):
    _fields_ = [
        ("timestamp_seconds", c_double),
        ("accel_x", c_double),
        ("accel_y", c_double),
        ("accel_z", c_double),
        ("gyro_x", c_double),
        ("gyro_y", c_double),
        ("gyro_z", c_double),
    ]


class ArbitV2TrackingSnapshot(Structure):
    _fields_ = [
        ("state", c_uint32),
        ("frame_id", c_uint64),
        ("track_count", c_uint32),
        ("inlier_count", c_uint32),
        ("pose_wc", c_double * 16),
    ]


class ArbitV2BackendSnapshot(Structure):
    _fields_ = [
        ("keyframe_count", c_uint64),
        ("loop_closure_events", c_uint64),
        ("relocalization_ready", c_bool),
    ]


class ArbitV2MapSnapshot(Structure):
    _fields_ = [
        ("landmark_count", c_uint64),
        ("anchor_count", c_uint64),
    ]


class ArbitV2RuntimeMetricsSnapshot(Structure):
    _fields_ = [
        ("frame_queue_depth", c_ulong),
        ("imu_queue_depth", c_ulong),
        ("keyframe_queue_depth", c_ulong),
        ("backend_queue_depth", c_ulong),
        ("dropped_frames", c_uint64),
        ("frontend_ms_median", c_double),
        ("frontend_ms_p95", c_double),
        ("end_to_end_ms_p95", c_double),
    ]


class ArbitV2Snapshot(Structure):
    _fields_ = [
        ("timestamp_seconds", c_double),
        ("tracking", ArbitV2TrackingSnapshot),
        ("backend", ArbitV2BackendSnapshot),
        ("map", ArbitV2MapSnapshot),
        ("metrics", ArbitV2RuntimeMetricsSnapshot),
    ]


class ArbitV2Anchor(Structure):
    _fields_ = [
        ("anchor_id", c_uint64),
        ("pose_wc", c_double * 16),
        ("created_from_keyframe", c_uint64),
        ("has_keyframe", c_bool),
        ("last_observed_frame", c_uint64),
    ]


@dataclass
class Transform:
    matrix: np.ndarray

    @classmethod
    def identity(cls) -> "Transform":
        return cls(matrix=np.eye(4, dtype=np.float64))

    def as_flat_row_major(self) -> np.ndarray:
        arr = np.asarray(self.matrix, dtype=np.float64)
        if arr.shape != (4, 4):
            raise ValueError("Transform matrix must be 4x4")
        return arr.reshape(16)


@dataclass
class TrackingSnapshot:
    state: TrackingState
    frame_id: int
    track_count: int
    inlier_count: int
    pose_wc: np.ndarray


@dataclass
class BackendSnapshot:
    keyframe_count: int
    loop_closure_events: int
    relocalization_ready: bool


@dataclass
class MapSnapshot:
    landmark_count: int
    anchor_count: int


@dataclass
class RuntimeMetricsSnapshot:
    frame_queue_depth: int
    imu_queue_depth: int
    keyframe_queue_depth: int
    backend_queue_depth: int
    dropped_frames: int
    frontend_ms_median: float
    frontend_ms_p95: float
    end_to_end_ms_p95: float


@dataclass
class Snapshot:
    timestamp_seconds: float
    tracking: TrackingSnapshot
    backend: BackendSnapshot
    map: MapSnapshot
    metrics: RuntimeMetricsSnapshot

    @classmethod
    def from_native(cls, native: ArbitV2Snapshot) -> "Snapshot":
        tracking_state = TrackingState(native.tracking.state)
        pose = np.array(native.tracking.pose_wc, dtype=np.float64).reshape((4, 4))
        return cls(
            timestamp_seconds=float(native.timestamp_seconds),
            tracking=TrackingSnapshot(
                state=tracking_state,
                frame_id=int(native.tracking.frame_id),
                track_count=int(native.tracking.track_count),
                inlier_count=int(native.tracking.inlier_count),
                pose_wc=pose,
            ),
            backend=BackendSnapshot(
                keyframe_count=int(native.backend.keyframe_count),
                loop_closure_events=int(native.backend.loop_closure_events),
                relocalization_ready=bool(native.backend.relocalization_ready),
            ),
            map=MapSnapshot(
                landmark_count=int(native.map.landmark_count),
                anchor_count=int(native.map.anchor_count),
            ),
            metrics=RuntimeMetricsSnapshot(
                frame_queue_depth=int(native.metrics.frame_queue_depth),
                imu_queue_depth=int(native.metrics.imu_queue_depth),
                keyframe_queue_depth=int(native.metrics.keyframe_queue_depth),
                backend_queue_depth=int(native.metrics.backend_queue_depth),
                dropped_frames=int(native.metrics.dropped_frames),
                frontend_ms_median=float(native.metrics.frontend_ms_median),
                frontend_ms_p95=float(native.metrics.frontend_ms_p95),
                end_to_end_ms_p95=float(native.metrics.end_to_end_ms_p95),
            ),
        )


@dataclass
class Anchor:
    anchor_id: int
    pose_wc: np.ndarray
    created_from_keyframe: Optional[int]
    last_observed_frame: int

    @classmethod
    def from_native(cls, native: ArbitV2Anchor) -> "Anchor":
        pose = np.array(native.pose_wc, dtype=np.float64).reshape((4, 4))
        keyframe = int(native.created_from_keyframe) if native.has_keyframe else None
        return cls(
            anchor_id=int(native.anchor_id),
            pose_wc=pose,
            created_from_keyframe=keyframe,
            last_observed_frame=int(native.last_observed_frame),
        )
