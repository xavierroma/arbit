"""
Arbit Processing Engine - Python wrapper for Rust FFI
"""

import ctypes
from ctypes import c_void_p, c_bool, c_ulong, c_uint64, c_double, POINTER, byref
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np

from .types import (
    ArbitCameraFrame,
    ArbitCameraIntrinsics,
    ArbitCameraSample,
    ArbitFeatDescriptor,
    ArbitImuSample,
    ArbitFrameState,
    ArbitImuState,
    ArbitTrackedPoint,
    ArbitPoseSample,
    ArbitTransform,
    ArbitProjectedAnchor,
    ArbitProjectedLandmark,
    ArbitMatch,
    PixelFormat,
    Transform,
    FrameState,
    ImuState,
    FeatDescriptor,
    Match,
)


def find_library() -> Path:
    """Find the arbit_ffi library"""
    import sys
    
    # Determine library name based on platform
    if sys.platform == "darwin":
        lib_name = "libarbit_ffi.dylib"
    elif sys.platform.startswith("linux"):
        lib_name = "libarbit_ffi.so"
    elif sys.platform.startswith("win"):
        lib_name = "arbit_ffi.dll"
    else:
        lib_name = "libarbit_ffi.so"  # fallback
    
    # __file__ is at: arbit/engine.py
    package_dir = Path(__file__).parent
    
    # First, check for bundled native library (installed package)
    bundled_lib = package_dir / "native" / lib_name
    if bundled_lib.exists():
        print(f"Found library at: {bundled_lib} (bundled)")
        return bundled_lib
    
    # For development, look for workspace target directory
    # From arbit/engine.py -> arbit-py/ -> crates/ -> project_root/
    workspace_root = package_dir.parent.parent.parent
    
    # Look in common build locations
    possible_paths = [
        workspace_root / "target" / "debug" / lib_name,
        workspace_root / "target" / "release" / lib_name,
    ]
    
    # If we're in a development tree, check if Cargo.toml exists
    if (workspace_root / "Cargo.toml").exists():
        for path in possible_paths:
            if path.exists():
                print(f"Found library at: {path} (workspace)")
                return path
    
    # Last resort: search from current directory upward for workspace root
    current = Path.cwd()
    for _ in range(10):  # Search up to 10 levels
        if (current / "Cargo.toml").exists():
            for config in ["debug", "release"]:
                path = current / "target" / config / lib_name
                if path.exists():
                    print(f"Found library at: {path} (discovered)")
                    return path
        if current.parent == current:  # Reached filesystem root
            break
        current = current.parent
    
    # Provide helpful error message
    attempted = [str(bundled_lib)] + [str(p) for p in possible_paths]
    attempted_str = "\n  ".join(attempted)
    raise RuntimeError(
        f"Could not find {lib_name}.\n"
        f"Options:\n"
        f"  1. Build it: cargo build -p arbit-ffi\n"
        f"  2. Run build script: crates/arbit-py/scripts/build-native.sh\n"
        f"  3. Set ARBIT_FFI_PATH environment variable\n"
        f"Searched in:\n  {attempted_str}"
    )


class ArbitEngine:
    """
    Python wrapper for Arbit Processing Engine
    
    This provides a high-level interface to the visual-inertial SLAM engine.
    
    Example:
        >>> engine = ArbitEngine()
        >>> 
        >>> # Ingest camera frames
        >>> frame = CameraFrame(
        ...     timestamp=0.0,
        ...     image=np.zeros((480, 640, 4), dtype=np.uint8),
        ...     intrinsics=(800.0, 800.0, 320.0, 240.0),
        ...     pixel_format=PixelFormat.BGRA8
        ... )
        >>> engine.ingest_frame(frame)
        >>> 
        >>> # Ingest IMU samples
        >>> imu = ImuSample(
        ...     timestamp=0.0,
        ...     accel=(0.0, 0.0, -9.81),
        ...     gyro=(0.0, 0.0, 0.0)
        ... )
        >>> engine.ingest_imu(imu)
        >>> 
        >>> # Query state
        >>> state = engine.get_frame_state()
        >>> print(f"Landmarks: {state.landmark_count}")
    """
    
    def __init__(self, library_path: Optional[Path] = None):
        """
        Initialize the Arbit engine
        
        Args:
            library_path: Path to libarbit_ffi library (auto-detected if None)
        """
        if library_path is None:
            library_path = find_library()
        
        self._lib = ctypes.CDLL(str(library_path))
        self._setup_functions()
        
        # Create context
        self._handle = self._lib.arbit_context_create()
        if not self._handle:
            raise RuntimeError("Failed to create Arbit context")
    
    def _setup_functions(self):
        """Setup function signatures"""
        lib = self._lib
        
        # Logging
        lib.arbit_init_logging.argtypes = []
        lib.arbit_init_logging.restype = None
        
        # Context management
        lib.arbit_context_create.argtypes = []
        lib.arbit_context_create.restype = c_void_p
        
        lib.arbit_context_destroy.argtypes = [c_void_p]
        lib.arbit_context_destroy.restype = None
        
        # Frame ingestion
        lib.arbit_ingest_frame.argtypes = [c_void_p, POINTER(ArbitCameraFrame), c_void_p]
        lib.arbit_ingest_frame.restype = c_bool
        
        # IMU ingestion
        lib.arbit_ingest_imu.argtypes = [c_void_p, ArbitImuSample]
        lib.arbit_ingest_imu.restype = c_bool
        
        # State queries
        lib.arbit_get_frame_state.argtypes = [c_void_p, POINTER(ArbitFrameState)]
        lib.arbit_get_frame_state.restype = c_bool
        
        lib.arbit_get_imu_state.argtypes = [c_void_p, POINTER(ArbitImuState)]
        lib.arbit_get_imu_state.restype = c_bool
        
        # Tracking queries
        lib.arbit_get_tracked_points.argtypes = [c_void_p, POINTER(ArbitTrackedPoint), c_ulong]
        lib.arbit_get_tracked_points.restype = c_ulong

        lib.arbit_get_descriptors.argtypes = [c_void_p, POINTER(ArbitFeatDescriptor), c_ulong]
        lib.arbit_get_descriptors.restype = c_ulong
        
        lib.arbit_match_descriptors.argtypes = [
            POINTER(ArbitFeatDescriptor), c_ulong,  # query descriptors
            POINTER(ArbitFeatDescriptor), c_ulong,  # train descriptors
            POINTER(ArbitMatch), c_ulong,            # output matches
            c_bool,                                   # cross_check
            ctypes.c_uint32,                         # max_distance
        ]
        lib.arbit_match_descriptors.restype = c_ulong

        lib.arbit_get_trajectory.argtypes = [c_void_p, POINTER(ArbitPoseSample), c_ulong]
        lib.arbit_get_trajectory.restype = c_ulong

        
        # # Anchor management
        # lib.arbit_create_anchor.argtypes = [c_void_p, POINTER(ArbitTransform), POINTER(c_uint64)]
        # lib.arbit_create_anchor.restype = c_bool
        
        # lib.arbit_get_anchor.argtypes = [c_void_p, c_uint64, POINTER(ArbitTransform)]
        # lib.arbit_get_anchor.restype = c_bool
        
        # lib.arbit_place_anchor_at_screen_point.argtypes = [
        #     c_void_p, c_double, c_double, c_double, POINTER(c_uint64)
        # ]
        # lib.arbit_place_anchor_at_screen_point.restype = c_bool
        
        # lib.arbit_get_visible_anchors.argtypes = [c_void_p, POINTER(ArbitProjectedAnchor), c_ulong]
        # lib.arbit_get_visible_anchors.restype = c_ulong
        
        # lib.arbit_get_visible_landmarks.argtypes = [
        #     c_void_p, POINTER(ArbitProjectedLandmark), c_ulong
        # ]
        # lib.arbit_get_visible_landmarks.restype = c_ulong
        
        # # Map I/O
        # lib.arbit_save_map.argtypes = [c_void_p, c_void_p, c_ulong, POINTER(c_ulong)]
        # lib.arbit_save_map.restype = c_bool
        
        # lib.arbit_load_map.argtypes = [c_void_p, c_void_p, c_ulong]
        # lib.arbit_load_map.restype = c_bool
    
    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, '_handle') and self._handle:
            self._lib.arbit_context_destroy(self._handle)
    
    def ingest_frame(self, frame: "CameraFrame") -> bool:
        """
        Ingest a camera frame
        
        Args:
            frame: Camera frame with image data and intrinsics
            
        Returns:
            True if frame was successfully processed
        """
        native_frame = frame.to_native()
        # Create output sample to receive processed frame info
        out_sample = ArbitCameraSample()
        result = self._lib.arbit_ingest_frame(self._handle, byref(native_frame), byref(out_sample))
        return bool(result)
    
    def ingest_imu(self, sample: "ImuSample") -> bool:
        """
        Ingest an IMU sample (accelerometer + gyroscope)
        
        Args:
            sample: IMU sample with accelerometer and gyroscope readings
            
        Returns:
            True if sample was successfully processed
        """
        native_sample = sample.to_native()
        result = self._lib.arbit_ingest_imu(self._handle, native_sample)
        return bool(result)
    
    def get_frame_state(self) -> FrameState:
        """
        Get comprehensive frame state
        
        Returns:
            FrameState with tracking, map, and IMU information
        """
        state = ArbitFrameState()
        if not self._lib.arbit_get_frame_state(self._handle, byref(state)):
            raise RuntimeError("Failed to get frame state")
        return FrameState(state)
    
    def get_imu_state(self) -> ImuState:
        """
        Get IMU state
        
        Returns:
            ImuState with gravity, motion, and preintegration info
        """
        state = ArbitImuState()
        if not self._lib.arbit_get_imu_state(self._handle, byref(state)):
            raise RuntimeError("Failed to get IMU state")
        return ImuState(state)
    
    def get_trajectory(self, max_points: int = 2048) -> np.ndarray:
        """
        Get camera trajectory

        Args:
            max_points: Maximum number of trajectory points to return
            
        Returns:
            Nx3 numpy array of trajectory positions
        """
        points = (ArbitPoseSample * max_points)()
        count = self._lib.arbit_get_trajectory(self._handle, points, max_points)

        trajectory = np.zeros((count, 3))
        for i in range(count):
            trajectory[i] = [points[i].x, points[i].y, points[i].z]

        return trajectory

    def get_descriptors(self, max_descriptors: int = 1024) -> List[FeatDescriptor]:
        """
        Get feature descriptors from the most recent keyframe

        Args:
            max_descriptors: Maximum number of descriptors to return

        Returns:
            List of `FeatDescriptor` wrappers containing 32-byte descriptor data
        """
        buffer = (ArbitFeatDescriptor * max_descriptors)()
        count = self._lib.arbit_get_descriptors(self._handle, buffer, max_descriptors)
        return [FeatDescriptor(buffer[i]) for i in range(count)]
    
    @staticmethod
    def match_descriptors(
        query_descriptors: List[FeatDescriptor],
        train_descriptors: List[FeatDescriptor],
        cross_check: bool = True,
        max_distance: int = 80,
        max_matches: int = 2048,
    ) -> List[Match]:
        """
        Match two sets of feature descriptors using Hamming distance
        
        Args:
            query_descriptors: Query descriptor list (e.g., from current keyframe)
            train_descriptors: Train descriptor list (e.g., from previous keyframe)
            cross_check: Enable cross-check matching (mutual nearest neighbors)
            max_distance: Maximum Hamming distance threshold (0 = no filtering)
            max_matches: Maximum number of matches to return
            
        Returns:
            List of Match objects containing matched descriptor indices and positions
            
        Example:
            >>> descriptors_kf1 = engine.get_descriptors()
            >>> # ... process more frames ...
            >>> descriptors_kf2 = engine.get_descriptors()
            >>> matches = ArbitEngine.match_descriptors(descriptors_kf1, descriptors_kf2)
        """
        # Convert Python FeatDescriptor wrappers back to ctypes structures
        query_buffer = (ArbitFeatDescriptor * len(query_descriptors))()
        for i, desc in enumerate(query_descriptors):
            query_buffer[i] = desc._native
            
        train_buffer = (ArbitFeatDescriptor * len(train_descriptors))()
        for i, desc in enumerate(train_descriptors):
            train_buffer[i] = desc._native
        
        # Allocate output buffer
        matches_buffer = (ArbitMatch * max_matches)()
        
        # Load library (static method, so we need to get it)
        lib = ctypes.CDLL(str(find_library()))
        
        # Call FFI function
        count = lib.arbit_match_descriptors(
            query_buffer, len(query_descriptors),
            train_buffer, len(train_descriptors),
            matches_buffer, max_matches,
            cross_check,
            max_distance,
        )
        
        # Convert to Python-friendly Match objects
        return [Match(matches_buffer[i]) for i in range(count)]
    
    def create_anchor(self, transform: Optional[Transform] = None) -> int:
        """
        Create a new anchor
        
        Args:
            transform: Anchor pose (uses current camera pose if None)
            
        Returns:
            Anchor ID
        """
        if transform is None:
            transform = Transform()
        
        anchor_id = c_uint64()
        if not self._lib.arbit_create_anchor(
            self._handle, byref(transform._native), byref(anchor_id)
        ):
            raise RuntimeError("Failed to create anchor")
        
        return anchor_id.value
    
    def place_anchor_at_point(
        self, normalized_u: float, normalized_v: float, depth: float = 1.0
    ) -> int:
        """
        Place an anchor by raycasting from a screen point
        
        Args:
            normalized_u: Normalized U coordinate [0, 1]
            normalized_v: Normalized V coordinate [0, 1]
            depth: Distance along ray in meters
            
        Returns:
            Anchor ID
        """
        anchor_id = c_uint64()
        if not self._lib.arbit_place_anchor_at_screen_point(
            self._handle, normalized_u, normalized_v, depth, byref(anchor_id)
        ):
            raise RuntimeError("Failed to place anchor")
        
        return anchor_id.value
    
    def get_visible_landmarks(self, max_count: int = 1000) -> List[dict]:
        """
        Get all visible landmarks in the current camera frame
        
        Args:
            max_count: Maximum number of landmarks to return
            
        Returns:
            List of landmark dictionaries with id, position, pixel coords
        """
        landmarks = (ArbitProjectedLandmark * max_count)()
        count = self._lib.arbit_get_visible_landmarks(self._handle, landmarks, max_count)
        
        result = []
        for i in range(count):
            lm = landmarks[i]
            result.append({
                'id': lm.landmark_id,
                'world_position': np.array([lm.world_x, lm.world_y, lm.world_z]),
                'pixel': np.array([lm.pixel_x, lm.pixel_y]),
                'normalized': np.array([lm.normalized_u, lm.normalized_v]),
                'depth': lm.depth,
            })
        
        return result
    
    def save_map(self) -> bytes:
        """
        Save map to binary data
        
        Returns:
            Binary map data
        """
        # Get required size
        required = c_ulong()
        self._lib.arbit_save_map(self._handle, None, 0, byref(required))
        
        # Allocate and save
        buffer = (ctypes.c_uint8 * required.value)()
        written = c_ulong()
        
        if not self._lib.arbit_save_map(
            self._handle, buffer, required.value, byref(written)
        ):
            raise RuntimeError("Failed to save map")
        
        return bytes(buffer[:written.value])
    
    def load_map(self, data: bytes) -> None:
        """
        Load map from binary data
        
        Args:
            data: Binary map data
        """
        buffer = (ctypes.c_uint8 * len(data))(*data)
        if not self._lib.arbit_load_map(self._handle, buffer, len(data)):
            raise RuntimeError("Failed to load map")

    @staticmethod
    def init_logging(library_path: Optional[Path] = None, verbose: bool = True):
        """
        Initialize Arbit logging (call before creating engine)
        
        Args:
            library_path: Path to libarbit_ffi library (auto-detected if None)
            verbose: Whether to print library location
        """
        if library_path is None:
            # Temporarily suppress find_library output if verbose=False
            import sys
            import io
            old_stdout = sys.stdout
            try:
                if not verbose:
                    sys.stdout = io.StringIO()
                library_path = find_library()
            finally:
                sys.stdout = old_stdout
        
        lib = ctypes.CDLL(str(library_path))
        lib.arbit_init_logging.argtypes = []
        lib.arbit_init_logging.restype = None
        lib.arbit_init_logging()


class CameraFrame:
    """Camera frame with image data and intrinsics"""
    
    def __init__(
        self,
        timestamp: float,
        image: np.ndarray,
        intrinsics: Tuple[float, float, float, float],  # (fx, fy, cx, cy)
        pixel_format: PixelFormat = PixelFormat.BGRA8,
    ):
        """
        Create a camera frame
        
        Args:
            timestamp: Frame timestamp in seconds
            image: Image data as numpy array (H x W x C)
            intrinsics: Camera intrinsics (fx, fy, cx, cy)
            pixel_format: Pixel format of image data
        """
        self.timestamp = timestamp
        self.image = image
        self.intrinsics = intrinsics
        self.pixel_format = pixel_format
    
    def to_native(self) -> ArbitCameraFrame:
        """Convert to native FFI structure"""
        fx, fy, cx, cy = self.intrinsics
        height, width = self.image.shape[:2]
        
        # Ensure contiguous memory
        image_data = np.ascontiguousarray(self.image)
        
        intrinsics = ArbitCameraIntrinsics()
        intrinsics.fx = fx
        intrinsics.fy = fy
        intrinsics.cx = cx
        intrinsics.cy = cy
        intrinsics.skew = 0.0
        intrinsics.width = width
        intrinsics.height = height
        intrinsics.distortion_len = 0
        intrinsics.distortion = None
        
        frame = ArbitCameraFrame()
        frame.timestamp_seconds = self.timestamp
        frame.intrinsics = intrinsics
        frame.pixel_format = self.pixel_format.value
        frame.bytes_per_row = image_data.strides[0]
        frame.data = image_data.ctypes.data_as(POINTER(ctypes.c_uint8))
        frame.data_len = image_data.nbytes
        
        # Keep reference to prevent garbage collection
        self._image_data = image_data
        
        return frame


class ImuSample:
    """IMU sample with accelerometer and gyroscope"""
    
    def __init__(
        self,
        timestamp: float,
        accel: Tuple[float, float, float],  # m/s²
        gyro: Tuple[float, float, float],   # rad/s
    ):
        """
        Create an IMU sample
        
        Args:
            timestamp: Sample timestamp in seconds
            accel: Accelerometer reading (x, y, z) in m/s²
            gyro: Gyroscope reading (x, y, z) in rad/s
        """
        self.timestamp = timestamp
        self.accel = accel
        self.gyro = gyro
    
    def to_native(self) -> ArbitImuSample:
        """Convert to native FFI structure"""
        sample = ArbitImuSample()
        sample.timestamp_seconds = self.timestamp
        sample.accel_x, sample.accel_y, sample.accel_z = self.accel
        sample.gyro_x, sample.gyro_y, sample.gyro_z = self.gyro
        return sample
