use arbit_core::math::CameraIntrinsics;
use arbit_core::time::{SystemClock, TimestampPolicy};
use arbit_providers::{CameraSample, PixelFormat};

use crate::types::VideoFrame;

/// Video camera provider that converts VideoFrame to CameraSample
///
/// This follows the same pattern as IosCameraProvider, converting source-specific
/// types (VideoFrame) to generic engine types (CameraSample) with proper timestamp
/// management.
pub struct VideoCameraProvider {
    timestamp_policy: TimestampPolicy<SystemClock>,
}

impl VideoCameraProvider {
    pub fn new() -> Self {
        Self {
            timestamp_policy: TimestampPolicy::new(),
        }
    }

    /// Convert VideoFrame (CLI-specific) to CameraSample (generic)
    pub fn ingest_frame(
        &mut self,
        frame: VideoFrame,
        intrinsics: CameraIntrinsics,
    ) -> CameraSample {
        let timestamps = self.timestamp_policy.ingest_capture(frame.timestamp);

        CameraSample {
            timestamps,
            intrinsics,
            pixel_format: PixelFormat::Bgra8,
            bytes_per_row: frame.bytes_per_row,
            data: frame.data,
        }
    }
}

impl Default for VideoCameraProvider {
    fn default() -> Self {
        Self::new()
    }
}
