# ARBIT CLI - Video Processing Tool

A command-line tool to run the ARBIT engine on recorded video sessions with optional IMU data.

## System Dependencies

### FFmpeg

The CLI requires FFmpeg libraries to be installed on your system.

**macOS:**
```bash
brew install ffmpeg
```

## Building

Once FFmpeg is installed:

```bash
cargo build -p arbit-cli
```

## Usage

### Process a video-only session:
```bash
cargo run -p arbit-cli -- process-session video.mp4
```

### Process a video with IMU data:
```bash
cargo run -p arbit-cli -- process-session video.mp4 imu.csv --output ./results
```

### Specify camera intrinsics via FOV:
```bash
cargo run -p arbit-cli -- process-session video.mp4 --fov 70.0
```

### Process with custom settings:
```bash
cargo run -p arbit-cli -- process-session video.mp4 imu.csv \
    --output ./output \
    --skip-frames 1 \
    --max-frames 100 \
    --verbose
```

### Save and load maps:
```bash
# Save map after processing
cargo run -p arbit-cli -- process-session video.mp4 --save-map scene.map

# Load existing map for relocalization
cargo run -p arbit-cli -- process-session video2.mp4 --map-file scene.map
```

## IMU Data Format

The IMU file should be a CSV with the following columns:

```csv
timestamp_seconds,accel_x,accel_y,accel_z
0.000000,-0.123,9.810,0.045
0.010000,-0.125,9.808,0.047
```

- **timestamp_seconds**: Seconds since session start (float64)
- **accel_x, accel_y, accel_z**: Accelerometer data in m/s²

## Output

The tool generates a `results.json` file containing:
- Camera trajectory
- Per-frame statistics (tracks, inliers, processing time)
- IMU statistics (if IMU data provided)
- Processing summary

Example output structure:
```json
{
  "metadata": {
    "session_name": "video",
    "frame_count": 150,
    "duration_seconds": 5.0
  },
  "trajectory": [...],
  "frame_stats": [...],
  "imu_stats": {...},
  "summary": {
    "total_frames": 150,
    "keyframes_created": 12,
    "landmarks_created": 1450
  }
}
```

## Camera Intrinsics

The tool supports multiple ways to specify camera intrinsics:

### 1. Estimate from FOV (default):
```bash
--fov 60.0  # Horizontal FOV in degrees
```

### 2. Load from YAML file:
```bash
--intrinsics intrinsics.yaml
```

Example `intrinsics.yaml`:
```yaml
width: 1920
height: 1080
fx: 1200.0
fy: 1200.0
cx: 960.0
cy: 540.0
skew: 0.0
distortion:  # Optional
  - -0.15
  - 0.08
```

### 3. Use iPhone model defaults:
```bash
--iphone-model 14Pro  # Supported: 14Pro, 13Pro, 12Pro
```

## Architecture

The CLI follows a clean provider pattern:

```
VideoFrame (CLI-specific) → VideoCameraProvider → CameraSample (generic) → ProcessingEngine
```

This allows the engine to remain completely provider-agnostic and work with any video source.

## Development

Run tests:
```bash
cargo test -p arbit-cli
```

Run with debug logging:
```bash
cargo run -p arbit-cli -- process-session video.mp4 --verbose
```

Format code:
```bash
cargo fmt --all
```

## Troubleshooting

### FFmpeg not found
If you see errors about FFmpeg during build, ensure it's installed:
```bash
brew install ffmpeg  # macOS
```

### Video decoding errors
Ensure your video file is in a supported format (MP4, MOV, AVI, etc.)

### IMU parsing errors
Check that your CSV file follows the expected format with proper headers

