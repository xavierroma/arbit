# Arbit Python Bindings

Python bindings for the Arbit visual-inertial SLAM engine.

## Features

- **Visual-Inertial SLAM**: Real-time tracking and mapping
- **Native Performance**: Rust-powered processing engine
- **Self-Contained**: Bundled native libraries, no external dependencies
- **Easy to Use**: Pythonic API with NumPy integration

## Installation

### From Source (Development)

```bash
# Build the native library and install in editable mode
cd crates/arbit-py
./scripts/build-native.sh
pip install -e .
```

### Building a Wheel

```bash
cd crates/arbit-py
python -m build
pip install dist/arbit-*.whl
```

The build process automatically compiles the Rust FFI library and bundles it into the package.

## Quick Start

```python
import arbit
import numpy as np

# Optional: Enable debug logging
arbit.init_logging()

# Create engine instance
engine = arbit.ArbitEngine()

# Create a camera frame
frame_data = np.zeros((480, 640, 4), dtype=np.uint8)  # BGRA image
camera_frame = arbit.CameraFrame(
    timestamp=0.0,
    image=frame_data,
    intrinsics=(800.0, 800.0, 320.0, 240.0),  # fx, fy, cx, cy
    pixel_format=arbit.PixelFormat.BGRA8
)

# Ingest the frame
success = engine.ingest_frame(camera_frame)

# Ingest IMU data
imu = arbit.ImuSample(
    timestamp=0.0,
    accel=(0.0, 0.0, -9.81),  # m/s²
    gyro=(0.0, 0.0, 0.0)       # rad/s
)
engine.ingest_imu(imu)

# Query state
state = engine.get_frame_state()
print(f"Landmarks: {state.landmark_count}")
print(f"Keyframes: {state.keyframe_count}")

# Get trajectory
trajectory = engine.get_trajectory()
print(f"Trajectory: {trajectory.shape}")
```

## API Reference

### Core Classes

#### `ArbitEngine`

Main SLAM engine interface.

**Methods:**
- `__init__()` - Create a new engine instance
- `ingest_frame(frame: CameraFrame) -> bool` - Process a camera frame
- `ingest_imu(sample: ImuSample) -> bool` - Process an IMU sample
- `get_frame_state() -> FrameState` - Get current tracking state
- `get_imu_state() -> ImuState` - Get IMU state
- `get_trajectory() -> np.ndarray` - Get camera trajectory (Nx3)
- `create_anchor(transform: Optional[Transform]) -> int` - Create an anchor
- `get_visible_landmarks() -> List[dict]` - Get visible landmarks
- `save_map() -> bytes` - Serialize map to bytes
- `load_map(data: bytes)` - Load map from bytes

**Static Methods:**
- `init_logging(verbose: bool = True)` - Initialize debug logging

#### `CameraFrame`

Camera frame with image data and calibration.

```python
frame = arbit.CameraFrame(
    timestamp=0.0,          # seconds
    image=numpy_array,      # HxWxC uint8
    intrinsics=(fx, fy, cx, cy),
    pixel_format=arbit.PixelFormat.BGRA8
)
```

#### `ImuSample`

IMU measurement with accelerometer and gyroscope.

```python
sample = arbit.ImuSample(
    timestamp=0.0,
    accel=(ax, ay, az),  # m/s²
    gyro=(wx, wy, wz)     # rad/s
)
```

### Enums

#### `PixelFormat`
- `BGRA8` - 8-bit BGRA (4 channels)
- `RGBA8` - 8-bit RGBA (4 channels)
- `NV12` - YUV 420 semi-planar
- `YV12` - YUV 420 planar

### State Types

#### `FrameState`
- `track_count` - Number of tracked points
- `keyframe_count` - Number of keyframes in map
- `landmark_count` - Number of 3D landmarks
- `anchor_count` - Number of user anchors

#### `ImuState`
- `has_gravity` - Whether gravity is estimated
- `gravity_down` - Gravity direction vector
- `has_motion_state` - Whether motion is classified
- `motion_state` - Motion classification (0=Stationary, 1=Slow, 2=Fast)

## Development

### Building the Native Library

The native Rust library is automatically built when you install the package, but you can also build it manually:

```bash
cd crates/arbit-py
./scripts/build-native.sh
```

This compiles `arbit-ffi` in release mode and copies the shared library to `arbit/native/`.

### Environment Variables

- `RUST_LOG=debug` - Enable debug logging from Rust engine
- `ARBIT_FFI_PATH=/path/to/lib` - Override library location
- `ARBIT_BUILD_CONFIG=debug` - Build in debug mode (default: release)
- `ARBIT_SKIP_BUILD=1` - Skip automatic build during pip install

### Running Tests

```bash
cd crates/arbit-py
pytest
```

### Examples

See the `examples/arbit-py/` directory for complete examples:
- `process_video.py` - Process video file with SLAM

## Architecture

The Python bindings use `ctypes` to interface with the Rust FFI layer (`arbit-ffi`). The native library is bundled during the build process, making the package self-contained.

```
Python Code
    ↓ (ctypes)
arbit-ffi (C API)
    ↓
arbit-engine (Rust)
    ↓
arbit-core (SLAM algorithms)
```

## License

MIT

## Contributing

See [QUICKSTART.md](QUICKSTART.md) for development setup instructions.
