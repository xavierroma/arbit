"""
Arbit v2 Processing Engine - Python wrapper for Rust FFI.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import POINTER, byref, c_bool, c_double, c_uint64, c_void_p
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .types import (
    Anchor,
    ArbitV2Anchor,
    ArbitV2CameraFrame,
    ArbitV2CameraIntrinsics,
    ArbitV2ImuSample,
    ArbitV2Snapshot,
    PixelFormat,
    Snapshot,
    Transform,
)


def find_library() -> Path:
    env_override = os.environ.get("ARBIT_FFI_PATH")
    if env_override:
        candidate = Path(env_override).expanduser().resolve()
        if candidate.exists():
            return candidate
        raise RuntimeError(f"ARBIT_FFI_PATH does not exist: {candidate}")

    import sys

    if sys.platform == "darwin":
        lib_name = "libarbit_ffi.dylib"
    elif sys.platform.startswith("linux"):
        lib_name = "libarbit_ffi.so"
    elif sys.platform.startswith("win"):
        lib_name = "arbit_ffi.dll"
    else:
        lib_name = "libarbit_ffi.so"

    package_dir = Path(__file__).parent
    bundled_lib = package_dir / "native" / lib_name
    if bundled_lib.exists():
        return bundled_lib

    workspace_root = package_dir.parent.parent.parent
    for config in ["release", "debug"]:
        candidate = workspace_root / "target" / config / lib_name
        if candidate.exists():
            return candidate

    raise RuntimeError(
        f"Could not find {lib_name}. Build with cargo build -p arbit-ffi or set ARBIT_FFI_PATH"
    )


class ArbitEngine:
    def __init__(self, library_path: Optional[Path] = None):
        if library_path is None:
            library_path = find_library()

        self._lib = ctypes.CDLL(str(library_path))
        self._setup_functions()

        self._handle = self._lib.arbit_v2_context_create()
        if not self._handle:
            raise RuntimeError("Failed to create Arbit v2 context")

    def _setup_functions(self) -> None:
        lib = self._lib

        lib.arbit_init_logging.argtypes = []
        lib.arbit_init_logging.restype = None

        lib.arbit_v2_context_create.argtypes = []
        lib.arbit_v2_context_create.restype = c_void_p

        lib.arbit_v2_context_destroy.argtypes = [c_void_p]
        lib.arbit_v2_context_destroy.restype = None

        lib.arbit_v2_ingest_frame.argtypes = [c_void_p, POINTER(ArbitV2CameraFrame)]
        lib.arbit_v2_ingest_frame.restype = c_bool

        lib.arbit_v2_ingest_imu.argtypes = [c_void_p, POINTER(ArbitV2ImuSample)]
        lib.arbit_v2_ingest_imu.restype = c_bool

        lib.arbit_v2_get_snapshot.argtypes = [c_void_p, POINTER(ArbitV2Snapshot)]
        lib.arbit_v2_get_snapshot.restype = c_bool

        lib.arbit_v2_create_anchor.argtypes = [
            c_void_p,
            POINTER(c_double),
            c_bool,
            c_uint64,
            POINTER(c_uint64),
        ]
        lib.arbit_v2_create_anchor.restype = c_bool

        lib.arbit_v2_query_anchor.argtypes = [c_void_p, c_uint64, POINTER(ArbitV2Anchor)]
        lib.arbit_v2_query_anchor.restype = c_bool

        lib.arbit_v2_reset_session.argtypes = [c_void_p]
        lib.arbit_v2_reset_session.restype = c_bool

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            self._lib.arbit_v2_context_destroy(self._handle)

    def ingest_frame(self, frame: "CameraFrame") -> bool:
        native_frame = frame.to_native()
        return bool(self._lib.arbit_v2_ingest_frame(self._handle, byref(native_frame)))

    def ingest_imu(self, sample: "ImuSample") -> bool:
        native = sample.to_native()
        return bool(self._lib.arbit_v2_ingest_imu(self._handle, byref(native)))

    def get_snapshot(self) -> Snapshot:
        native = ArbitV2Snapshot()
        ok = self._lib.arbit_v2_get_snapshot(self._handle, byref(native))
        if not ok:
            raise RuntimeError("Failed to retrieve v2 snapshot")
        return Snapshot.from_native(native)

    def create_anchor(self, transform: Optional[Transform] = None, keyframe_hint: Optional[int] = None) -> int:
        if transform is None:
            transform = Transform.identity()

        flat = transform.as_flat_row_major()
        c_pose = (c_double * 16)(*flat)
        out_anchor = c_uint64()
        has_hint = keyframe_hint is not None
        hint_value = c_uint64(0 if keyframe_hint is None else int(keyframe_hint))

        ok = self._lib.arbit_v2_create_anchor(
            self._handle,
            c_pose,
            c_bool(has_hint),
            hint_value,
            byref(out_anchor),
        )
        if not ok:
            raise RuntimeError("Failed to create anchor")
        return int(out_anchor.value)

    def query_anchor(self, anchor_id: int) -> Optional[Anchor]:
        native = ArbitV2Anchor()
        ok = self._lib.arbit_v2_query_anchor(self._handle, c_uint64(anchor_id), byref(native))
        if not ok:
            return None
        return Anchor.from_native(native)

    def reset_session(self) -> None:
        ok = self._lib.arbit_v2_reset_session(self._handle)
        if not ok:
            raise RuntimeError("Failed to reset session")

    # Compatibility alias for existing downstream callers.
    def get_frame_state(self) -> Snapshot:
        return self.get_snapshot()

    @staticmethod
    def init_logging(library_path: Optional[Path] = None) -> None:
        if library_path is None:
            library_path = find_library()
        lib = ctypes.CDLL(str(library_path))
        lib.arbit_init_logging.argtypes = []
        lib.arbit_init_logging.restype = None
        lib.arbit_init_logging()


class CameraFrame:
    def __init__(
        self,
        timestamp: float,
        image: np.ndarray,
        intrinsics: Tuple[float, float, float, float],
        pixel_format: PixelFormat = PixelFormat.BGRA8,
    ):
        self.timestamp = float(timestamp)
        self.image = np.ascontiguousarray(image)
        self.intrinsics = intrinsics
        self.pixel_format = pixel_format

    def to_native(self) -> ArbitV2CameraFrame:
        fx, fy, cx, cy = self.intrinsics
        height, width = self.image.shape[:2]

        intr = ArbitV2CameraIntrinsics(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            skew=0.0,
            width=width,
            height=height,
            distortion_len=0,
            distortion=None,
        )

        return ArbitV2CameraFrame(
            timestamp_seconds=self.timestamp,
            intrinsics=intr,
            pixel_format=int(self.pixel_format),
            bytes_per_row=self.image.strides[0],
            data=self.image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            data_len=self.image.nbytes,
        )


class ImuSample:
    def __init__(
        self,
        timestamp: float,
        accel: Tuple[float, float, float],
        gyro: Tuple[float, float, float],
    ):
        self.timestamp = float(timestamp)
        self.accel = accel
        self.gyro = gyro

    def to_native(self) -> ArbitV2ImuSample:
        ax, ay, az = self.accel
        gx, gy, gz = self.gyro
        return ArbitV2ImuSample(
            timestamp_seconds=self.timestamp,
            accel_x=ax,
            accel_y=ay,
            accel_z=az,
            gyro_x=gx,
            gyro_y=gy,
            gyro_z=gz,
        )
