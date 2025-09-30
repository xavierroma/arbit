use std::sync::Arc;
use std::time::Duration;

use arbit_core::time::Clock;
use arbit_core::{
    math::{CameraIntrinsics, DistortionModel},
    time::{FrameTimestamps, SystemClock, TimestampPolicy},
};
use nalgebra::Matrix3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Bgra8,
    Rgba8,
    Nv12,
    Yv12,
    Depth16,
}

#[derive(Debug, Clone)]
pub struct CameraSample {
    pub timestamps: FrameTimestamps,
    pub intrinsics: CameraIntrinsics,
    pub pixel_format: PixelFormat,
    pub bytes_per_row: usize,
    pub data: Arc<[u8]>,
}

impl CameraSample {
    pub fn intrinsic_matrix(&self) -> Matrix3<f64> {
        self.intrinsics.matrix()
    }

    pub fn resolution(&self) -> (u32, u32) {
        (self.intrinsics.width, self.intrinsics.height)
    }

    pub fn has_distortion(&self) -> bool {
        self.intrinsics.has_distortion()
    }
}

#[derive(Debug, Clone)]
pub struct ArKitIntrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub skew: f64,
    pub width: u32,
    pub height: u32,
    pub distortion: Option<Vec<f64>>,
}

impl From<ArKitIntrinsics> for CameraIntrinsics {
    fn from(value: ArKitIntrinsics) -> Self {
        let distortion = value
            .distortion
            .map(DistortionModel::Custom)
            .unwrap_or(DistortionModel::None);
        CameraIntrinsics::new(
            value.fx,
            value.fy,
            value.cx,
            value.cy,
            value.skew,
            value.width,
            value.height,
            distortion,
        )
    }
}

#[derive(Debug, Clone)]
pub struct ArKitFrame {
    pub timestamp: Duration,
    pub intrinsics: ArKitIntrinsics,
    pub pixel_format: PixelFormat,
    pub bytes_per_row: usize,
    pub data: Arc<[u8]>,
}

#[derive(Debug, Clone)]
pub struct IosCameraProvider<C: Clock = SystemClock> {
    timestamp_policy: TimestampPolicy<C>,
}

impl IosCameraProvider<SystemClock> {
    pub fn new() -> Self {
        Self {
            timestamp_policy: TimestampPolicy::new(),
        }
    }
}

impl<C: Clock> IosCameraProvider<C> {
    pub fn with_policy(policy: TimestampPolicy<C>) -> Self {
        Self {
            timestamp_policy: policy,
        }
    }

    pub fn ingest_frame(&mut self, frame: ArKitFrame) -> CameraSample {
        let timestamps = self.timestamp_policy.ingest_capture(frame.timestamp);
        let intrinsics = CameraIntrinsics::from(frame.intrinsics);

        CameraSample {
            timestamps,
            intrinsics,
            pixel_format: frame.pixel_format,
            bytes_per_row: frame.bytes_per_row,
            data: frame.data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[derive(Clone)]
    struct MockClock {
        times: RefCell<Vec<Duration>>,
    }

    impl MockClock {
        fn new(times: Vec<Duration>) -> Self {
            Self {
                times: RefCell::new(times),
            }
        }
    }

    impl Clock for MockClock {
        fn now(&mut self) -> Duration {
            let mut times = self.times.borrow_mut();
            if times.len() == 1 {
                times[0]
            } else {
                times.remove(0)
            }
        }
    }

    #[test]
    fn ingest_frame_wraps_intrinsics() {
        let policy = TimestampPolicy::with_clock(MockClock::new(vec![
            Duration::from_millis(5),
            Duration::from_millis(6),
        ]));
        let mut provider = IosCameraProvider::with_policy(policy);

        let frame = ArKitFrame {
            timestamp: Duration::from_millis(2),
            intrinsics: ArKitIntrinsics {
                fx: 800.0,
                fy: 820.0,
                cx: 400.0,
                cy: 300.0,
                skew: 0.0,
                width: 800,
                height: 600,
                distortion: Some(vec![0.1, -0.01, 0.0, 0.0]),
            },
            pixel_format: PixelFormat::Bgra8,
            bytes_per_row: 800 * 4,
            data: Arc::from(vec![0u8; 800 * 600 * 4]),
        };

        let sample = provider.ingest_frame(frame);

        assert_eq!(sample.intrinsics.fx, 800.0);
        assert_eq!(sample.intrinsics.fy, 820.0);
        assert_eq!(sample.intrinsics.width, 800);
        assert_eq!(sample.intrinsics.height, 600);
        assert_eq!(sample.pixel_format, PixelFormat::Bgra8);
        assert!(matches!(
            sample.intrinsics.distortion,
            DistortionModel::Custom(_)
        ));
        assert_eq!(sample.timestamps.latency, Duration::from_millis(3));

        let (width, height) = sample.resolution();
        assert_eq!(width, 800);
        assert_eq!(height, 600);
        assert!(sample.has_distortion());

        let matrix = sample.intrinsic_matrix();
        assert_eq!(matrix[(0, 0)], 800.0);
        assert_eq!(matrix[(0, 1)], 0.0);
        assert_eq!(matrix[(0, 2)], 400.0);
        assert_eq!(matrix[(1, 1)], 820.0);
        assert_eq!(matrix[(1, 2)], 300.0);
    }
}
