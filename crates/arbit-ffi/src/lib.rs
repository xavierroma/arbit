use std::ptr;
use std::time::Duration;

use arbit_core::math::{CameraIntrinsics, DistortionModel};
use arbit_core::time::{FrameTimestamps, SystemClock, TimestampPolicy};

struct CaptureContext {
    timestamp_policy: TimestampPolicy<SystemClock>,
}

impl Default for CaptureContext {
    fn default() -> Self {
        Self {
            timestamp_policy: TimestampPolicy::new(),
        }
    }
}

#[repr(C)]
pub struct ArbitCaptureContextHandle {
    inner: CaptureContext,
}

impl Default for ArbitCaptureContextHandle {
    fn default() -> Self {
        Self {
            inner: CaptureContext::default(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ArbitPixelFormat {
    Bgra8 = 0,
    Rgba8 = 1,
    Nv12 = 2,
    Yv12 = 3,
    Depth16 = 4,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ArbitCameraIntrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub skew: f64,
    pub width: u32,
    pub height: u32,
    pub distortion_len: usize,
    pub distortion: *const f64,
}

impl Default for ArbitCameraIntrinsics {
    fn default() -> Self {
        Self {
            fx: 0.0,
            fy: 0.0,
            cx: 0.0,
            cy: 0.0,
            skew: 0.0,
            width: 0,
            height: 0,
            distortion_len: 0,
            distortion: ptr::null(),
        }
    }
}

impl ArbitCameraIntrinsics {
    fn distortion_coeffs(&self) -> Option<Vec<f64>> {
        if self.distortion_len == 0 || self.distortion.is_null() {
            return None;
        }

        let slice = unsafe { std::slice::from_raw_parts(self.distortion, self.distortion_len) };
        Some(slice.to_vec())
    }

    fn to_internal(&self) -> CameraIntrinsics {
        let distortion = self
            .distortion_coeffs()
            .map(DistortionModel::Custom)
            .unwrap_or(DistortionModel::None);

        CameraIntrinsics::new(
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.skew,
            self.width,
            self.height,
            distortion,
        )
    }
}

impl From<&CameraIntrinsics> for ArbitCameraIntrinsics {
    fn from(value: &CameraIntrinsics) -> Self {
        Self {
            fx: value.fx,
            fy: value.fy,
            cx: value.cx,
            cy: value.cy,
            skew: value.skew,
            width: value.width,
            height: value.height,
            distortion_len: 0,
            distortion: ptr::null(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ArbitCameraFrame {
    pub timestamp_seconds: f64,
    pub intrinsics: ArbitCameraIntrinsics,
    pub pixel_format: ArbitPixelFormat,
    pub bytes_per_row: usize,
    pub data: *const u8,
    pub data_len: usize,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ArbitFrameTimestamps {
    pub capture_seconds: f64,
    pub pipeline_seconds: f64,
    pub latency_seconds: f64,
}

impl From<FrameTimestamps> for ArbitFrameTimestamps {
    fn from(value: FrameTimestamps) -> Self {
        Self {
            capture_seconds: value.capture.as_duration().as_secs_f64(),
            pipeline_seconds: value.pipeline.as_duration().as_secs_f64(),
            latency_seconds: value.latency.as_secs_f64(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ArbitCameraSample {
    pub timestamps: ArbitFrameTimestamps,
    pub intrinsics: ArbitCameraIntrinsics,
    pub pixel_format: ArbitPixelFormat,
    pub bytes_per_row: usize,
}

impl ArbitCameraSample {
    fn from_internal(
        timestamps: FrameTimestamps,
        intrinsics: &CameraIntrinsics,
        pixel_format: ArbitPixelFormat,
        bytes_per_row: usize,
    ) -> Self {
        Self {
            timestamps: timestamps.into(),
            intrinsics: intrinsics.into(),
            pixel_format,
            bytes_per_row,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ArbitTransform {
    pub elements: [f64; 16],
}

impl Default for ArbitTransform {
    fn default() -> Self {
        Self {
            elements: [0.0; 16],
        }
    }
}

fn duration_from_seconds(seconds: f64) -> Duration {
    if seconds <= 0.0 {
        Duration::from_secs(0)
    } else {
        Duration::from_secs_f64(seconds)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn arbit_capture_context_new() -> *mut ArbitCaptureContextHandle {
    Box::into_raw(Box::new(ArbitCaptureContextHandle::default()))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_capture_context_free(handle: *mut ArbitCaptureContextHandle) {
    if handle.is_null() {
        return;
    }

    unsafe {
        drop(Box::from_raw(handle));
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_ingest_camera_frame(
    handle: *mut ArbitCaptureContextHandle,
    frame: *const ArbitCameraFrame,
    out_sample: *mut ArbitCameraSample,
) -> bool {
    if handle.is_null() || frame.is_null() || out_sample.is_null() {
        return false;
    }

    let context = unsafe { &mut (*handle).inner };
    let frame = unsafe { &*frame };

    let capture = duration_from_seconds(frame.timestamp_seconds);
    let timestamps = context.timestamp_policy.ingest_capture(capture);
    let intrinsics = frame.intrinsics.to_internal();

    unsafe {
        (*out_sample) = ArbitCameraSample::from_internal(
            timestamps,
            &intrinsics,
            frame.pixel_format,
            frame.bytes_per_row,
        );
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_zero_for_negative_input() {
        assert_eq!(duration_from_seconds(-1.0), Duration::from_secs(0));
    }

    #[test]
    fn ingest_path_returns_monotonic_timestamps() {
        let handle = arbit_capture_context_new();
        assert!(!handle.is_null());

        let frame = ArbitCameraFrame {
            timestamp_seconds: 0.5,
            intrinsics: ArbitCameraIntrinsics {
                fx: 500.0,
                fy: 500.0,
                cx: 320.0,
                cy: 240.0,
                skew: 0.0,
                width: 640,
                height: 480,
                distortion_len: 0,
                distortion: ptr::null(),
            },
            pixel_format: ArbitPixelFormat::Bgra8,
            bytes_per_row: 640 * 4,
            data: ptr::null(),
            data_len: 0,
        };

        let mut sample = ArbitCameraSample {
            timestamps: ArbitFrameTimestamps {
                capture_seconds: 0.0,
                pipeline_seconds: 0.0,
                latency_seconds: 0.0,
            },
            intrinsics: ArbitCameraIntrinsics::default(),
            pixel_format: ArbitPixelFormat::Bgra8,
            bytes_per_row: 0,
        };

        unsafe {
            assert!(arbit_ingest_camera_frame(
                handle,
                &frame,
                &mut sample as *mut _
            ));
            assert!(sample.timestamps.capture_seconds >= 0.5);
            assert!(sample.timestamps.pipeline_seconds >= 0.0);
            assert!(sample.timestamps.latency_seconds >= 0.0);
        }

        unsafe {
            arbit_capture_context_free(handle);
        }
    }

    #[test]
    fn successive_ingest_calls_are_monotonic() {
        let handle = arbit_capture_context_new();
        assert!(!handle.is_null());

        let mut sample = ArbitCameraSample {
            timestamps: ArbitFrameTimestamps {
                capture_seconds: 0.0,
                pipeline_seconds: 0.0,
                latency_seconds: 0.0,
            },
            intrinsics: ArbitCameraIntrinsics::default(),
            pixel_format: ArbitPixelFormat::Bgra8,
            bytes_per_row: 0,
        };

        unsafe {
            let mut frame = ArbitCameraFrame {
                timestamp_seconds: 1.0,
                intrinsics: ArbitCameraIntrinsics::default(),
                pixel_format: ArbitPixelFormat::Bgra8,
                bytes_per_row: 0,
                data: ptr::null(),
                data_len: 0,
            };
            assert!(arbit_ingest_camera_frame(handle, &frame, &mut sample));
            let capture_a = sample.timestamps.capture_seconds;

            frame.timestamp_seconds = 0.5;
            assert!(arbit_ingest_camera_frame(handle, &frame, &mut sample));
            let capture_b = sample.timestamps.capture_seconds;

            assert!(capture_b > capture_a);

            arbit_capture_context_free(handle);
        }
    }
}
