use std::ptr;
use std::slice;
use std::time::Duration;

use arbit_core::img::{ImageBuffer, Pyramid, PyramidLevel, build_pyramid};
use arbit_core::imu::{GravityEstimate, GravityEstimator};
use arbit_core::init::two_view::{
    FeatureMatch, RansacParams, TwoViewInitialization, TwoViewInitializer,
};
use arbit_core::math::se3::TransformSE3;
use arbit_core::math::{CameraIntrinsics, DistortionModel};
use arbit_core::time::{FrameTimestamps, SystemClock, TimestampPolicy};
use arbit_core::track::{
    FeatureGridConfig, FeatureSeeder, LucasKanadeConfig, TrackObservation, TrackOutcome, Tracker,
};
use log::{debug, info, warn};
use nalgebra::{Translation3, UnitQuaternion, Vector2, Vector3};

struct PyramidLevelCache {
    octave: u32,
    scale: f32,
    width: u32,
    height: u32,
    bytes_per_row: usize,
    pixels: Vec<u8>,
}

struct CaptureContext {
    timestamp_policy: TimestampPolicy<SystemClock>,
    last_intrinsics: Option<CameraIntrinsics>,
    prev_pyramid: Option<Pyramid>,
    pyramid_cache: Vec<PyramidLevelCache>,
    seeder: FeatureSeeder,
    tracker: Tracker,
    last_tracks: Vec<TrackObservation>,
    last_two_view: Option<TwoViewInitialization>,
    two_view_initializer: TwoViewInitializer,
    trajectory: Vec<Vector3<f64>>,
    current_pose: TransformSE3,
    last_gravity: Option<GravityEstimate>,
    gravity_estimator: GravityEstimator,
}

impl Default for CaptureContext {
    fn default() -> Self {
        Self {
            timestamp_policy: TimestampPolicy::new(),
            last_intrinsics: None,
            prev_pyramid: None,
            pyramid_cache: Vec::new(),
            seeder: FeatureSeeder::new(FeatureGridConfig::default()),
            tracker: Tracker::new(LucasKanadeConfig::default()),
            last_tracks: Vec::new(),
            last_two_view: None,
            two_view_initializer: TwoViewInitializer::new(RansacParams::default()),
            trajectory: vec![Vector3::new(0.0, 0.0, 0.0)],
            current_pose: TransformSE3::identity(),
            last_gravity: None,
            gravity_estimator: GravityEstimator::new(0.75),
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
pub struct ArbitPyramidLevelView {
    pub octave: u32,
    pub scale: f32,
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: usize,
    pub pixels: *const u8,
    pub pixels_len: usize,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ArbitTrackStatus {
    Converged = 0,
    Diverged = 1,
    OutOfBounds = 2,
}

impl From<TrackOutcome> for ArbitTrackStatus {
    fn from(value: TrackOutcome) -> Self {
        match value {
            TrackOutcome::Converged => ArbitTrackStatus::Converged,
            TrackOutcome::Diverged => ArbitTrackStatus::Diverged,
            TrackOutcome::OutOfBounds => ArbitTrackStatus::OutOfBounds,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ArbitTrackedPoint {
    pub initial_x: f32,
    pub initial_y: f32,
    pub refined_x: f32,
    pub refined_y: f32,
    pub residual: f32,
    pub iterations: u32,
    pub status: ArbitTrackStatus,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ArbitTwoViewSummary {
    pub inliers: u32,
    pub average_error: f64,
    pub rotation: [f64; 9],
    pub translation: [f64; 3],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ArbitPoseSample {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ArbitAccelerometerSample {
    pub ax: f64,
    pub ay: f64,
    pub az: f64,
    pub dt_seconds: f64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ArbitGravityEstimate {
    pub down: [f64; 3],
    pub samples: u32,
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

fn extract_image_buffer(
    frame: &ArbitCameraFrame,
    intrinsics: &CameraIntrinsics,
) -> Option<ImageBuffer> {
    if frame.data.is_null() || frame.data_len == 0 {
        return None;
    }

    let width = intrinsics.width as usize;
    let height = intrinsics.height as usize;
    if width == 0 || height == 0 {
        return None;
    }

    // More lenient validation - use minimum required bytes
    let min_required = width.saturating_mul(height).saturating_mul(4); // BGRA8 = 4 bytes per pixel
    if frame.data_len < min_required {
        return None;
    }

    // Use the smaller of provided length or calculated requirement
    let actual_len = frame
        .data_len
        .min(frame.bytes_per_row.saturating_mul(height));

    let bytes = unsafe { slice::from_raw_parts(frame.data, actual_len) };
    Some(ImageBuffer::from_bgra8(
        bytes,
        width,
        height,
        frame.bytes_per_row,
    ))
}

fn encode_luma(level: &PyramidLevel) -> Vec<u8> {
    level
        .image
        .data()
        .iter()
        .map(|value| value.clamp(0.0, 255.0) as u8)
        .collect()
}

fn normalize_pixel(pixel: Vector2<f32>, intrinsics: &CameraIntrinsics) -> Vector2<f64> {
    Vector2::new(
        (pixel.x as f64 - intrinsics.cx) / intrinsics.fx,
        (pixel.y as f64 - intrinsics.cy) / intrinsics.fy,
    )
}

fn build_feature_matches(
    observations: &[TrackObservation],
    prev_intrinsics: &CameraIntrinsics,
    curr_intrinsics: &CameraIntrinsics,
) -> Vec<FeatureMatch> {
    observations
        .iter()
        .filter(|obs| matches!(obs.outcome, TrackOutcome::Converged))
        .map(|obs| FeatureMatch {
            normalized_a: normalize_pixel(obs.initial, prev_intrinsics),
            normalized_b: normalize_pixel(obs.refined, curr_intrinsics),
        })
        .collect()
}

fn update_pose(current: &TransformSE3, two_view: &TwoViewInitialization) -> TransformSE3 {
    let rotation = UnitQuaternion::from_rotation_matrix(&two_view.rotation);
    let translation = Translation3::from(two_view.translation.normalize() * 0.05);
    let delta = TransformSE3::from_parts(translation, rotation);
    current * delta
}

#[unsafe(no_mangle)]
pub extern "C" fn arbit_capture_context_new() -> *mut ArbitCaptureContextHandle {
    Box::into_raw(Box::new(ArbitCaptureContextHandle::default()))
}

#[unsafe(no_mangle)]
pub extern "C" fn arbit_init_logging() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
        .filter_level(log::LevelFilter::Debug)
        .init();
    info!("ARBIT logging initialized");
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
    let prev_intrinsics = context.last_intrinsics.clone();

    debug!(
        "Ingesting frame at timestamp: {:.3}s",
        frame.timestamp_seconds
    );
    debug!(
        "Previous pyramid exists: {}",
        context.prev_pyramid.is_some()
    );
    debug!(
        "Previous intrinsics exists: {}",
        context.last_intrinsics.is_some()
    );

    if let Some(image) = extract_image_buffer(frame, &intrinsics) {
        debug!(
            "Successfully extracted image buffer: {}x{}",
            image.width(),
            image.height()
        );
        let pyramid = build_pyramid(&image, 3);
        let levels = pyramid.levels().len();
        debug!("Built pyramid with {} levels", levels);
        context.pyramid_cache = pyramid
            .levels()
            .iter()
            .map(|level| PyramidLevelCache {
                octave: level.octave as u32,
                scale: level.scale,
                width: level.image.width() as u32,
                height: level.image.height() as u32,
                bytes_per_row: level.image.width(),
                pixels: encode_luma(level),
            })
            .collect();

        if let Some(prev) = context.prev_pyramid.as_ref() {
            let seeds = context.seeder.seed(&prev.levels()[0]);
            debug!("Seeded {} features for tracking", seeds.len());
            let mut tracks = Vec::with_capacity(seeds.len());
            for seed in seeds.iter().take(256) {
                let observation = context.tracker.track(prev, &pyramid, seed.position);
                tracks.push(observation);
            }
            let converged_tracks = tracks
                .iter()
                .filter(|t| matches!(t.outcome, TrackOutcome::Converged))
                .count();
            debug!(
                "Computed {} tracks, {} converged",
                tracks.len(),
                converged_tracks
            );
            context.last_tracks = tracks;
            debug!("Stored {} tracks for next frame", context.last_tracks.len());

            if let Some(prev_intr) = prev_intrinsics {
                let matches = build_feature_matches(&context.last_tracks, &prev_intr, &intrinsics);
                debug!(
                    "Found {} feature matches for two-view initialization",
                    matches.len()
                );
                if matches.len() >= 8 {
                    debug!(
                        "Attempting two-view initialization with {} matches",
                        matches.len()
                    );
                    if let Some(two_view) = context.two_view_initializer.estimate(&matches) {
                        context.current_pose = update_pose(&context.current_pose, &two_view);
                        context
                            .trajectory
                            .push(context.current_pose.translation.vector);
                        if context.trajectory.len() > 2048 {
                            let trim = context.trajectory.len() - 2048;
                            context.trajectory.drain(0..trim);
                        }
                        context.last_two_view = Some(two_view);
                        debug!(
                            "Two-view initialization successful, trajectory length: {}",
                            context.trajectory.len()
                        );
                    } else {
                        context.last_two_view = None;
                        debug!("Two-view initialization failed");
                    }
                } else {
                    context.last_two_view = None;
                    debug!(
                        "Insufficient matches ({}) for two-view initialization",
                        matches.len()
                    );
                }
            } else {
                debug!("No previous intrinsics available for two-view initialization");
            }
        }

        context.prev_pyramid = Some(pyramid);
    } else {
        warn!(
            "Failed to extract image buffer for frame at {:.3}s",
            frame.timestamp_seconds
        );
        context.pyramid_cache.clear();
        context.last_tracks.clear();
    }

    context.last_intrinsics = Some(intrinsics.clone());

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

#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_capture_context_pyramid_levels(
    handle: *mut ArbitCaptureContextHandle,
    out_levels: *mut ArbitPyramidLevelView,
    max_levels: usize,
) -> usize {
    if handle.is_null() || out_levels.is_null() || max_levels == 0 {
        return 0;
    }

    let context = unsafe { &mut (*handle).inner };
    let count = context.pyramid_cache.len().min(max_levels);
    let dest = unsafe { slice::from_raw_parts_mut(out_levels, count) };
    for (dst, cache) in dest.iter_mut().zip(context.pyramid_cache.iter()) {
        *dst = ArbitPyramidLevelView {
            octave: cache.octave,
            scale: cache.scale,
            width: cache.width,
            height: cache.height,
            bytes_per_row: cache.bytes_per_row,
            pixels: cache.pixels.as_ptr(),
            pixels_len: cache.pixels.len(),
        };
    }
    count
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_capture_context_tracked_points(
    handle: *mut ArbitCaptureContextHandle,
    out_points: *mut ArbitTrackedPoint,
    max_points: usize,
) -> usize {
    if handle.is_null() || out_points.is_null() || max_points == 0 {
        return 0;
    }

    let context = unsafe { &mut (*handle).inner };
    let count = context.last_tracks.len().min(max_points);
    let dest = unsafe { slice::from_raw_parts_mut(out_points, count) };
    for (dst, track) in dest.iter_mut().zip(context.last_tracks.iter()) {
        *dst = ArbitTrackedPoint {
            initial_x: track.initial.x,
            initial_y: track.initial.y,
            refined_x: track.refined.x,
            refined_y: track.refined.y,
            residual: track.residual,
            iterations: track.iterations,
            status: track.outcome.into(),
        };
    }
    count
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_capture_context_two_view(
    handle: *mut ArbitCaptureContextHandle,
    out_summary: *mut ArbitTwoViewSummary,
) -> bool {
    if handle.is_null() || out_summary.is_null() {
        return false;
    }

    let context = unsafe { &mut (*handle).inner };
    let Some(result) = context.last_two_view.as_ref() else {
        return false;
    };

    let matrix = result.rotation.matrix();
    let mut rotation = [0.0f64; 9];
    for row in 0..3 {
        for col in 0..3 {
            rotation[row * 3 + col] = matrix[(row, col)];
        }
    }

    let translation = [
        result.translation.x,
        result.translation.y,
        result.translation.z,
    ];

    unsafe {
        *out_summary = ArbitTwoViewSummary {
            inliers: result.inliers.len() as u32,
            average_error: result.average_sampson_error,
            rotation,
            translation,
        };
    }
    true
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_capture_context_trajectory(
    handle: *mut ArbitCaptureContextHandle,
    out_points: *mut ArbitPoseSample,
    max_points: usize,
) -> usize {
    if handle.is_null() || out_points.is_null() || max_points == 0 {
        return 0;
    }

    let context = unsafe { &mut (*handle).inner };
    let count = context.trajectory.len().min(max_points);
    let dest = unsafe { slice::from_raw_parts_mut(out_points, count) };
    for (dst, pos) in dest.iter_mut().zip(context.trajectory.iter()) {
        *dst = ArbitPoseSample {
            x: pos.x,
            y: pos.y,
            z: pos.z,
        };
    }
    count
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_ingest_accelerometer_sample(
    handle: *mut ArbitCaptureContextHandle,
    sample: ArbitAccelerometerSample,
) -> bool {
    if handle.is_null() {
        return false;
    }

    let context = unsafe { &mut (*handle).inner };
    let accel = Vector3::new(sample.ax, sample.ay, sample.az);
    let dt = if sample.dt_seconds.is_finite() && sample.dt_seconds > 0.0 {
        sample.dt_seconds
    } else {
        1.0 / 120.0
    };

    context.last_gravity = context.gravity_estimator.update(accel, dt);
    true
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_capture_context_gravity(
    handle: *mut ArbitCaptureContextHandle,
    out_estimate: *mut ArbitGravityEstimate,
) -> bool {
    if handle.is_null() || out_estimate.is_null() {
        return false;
    }

    let context = unsafe { &mut (*handle).inner };
    let Some(gravity) = context.last_gravity.as_ref() else {
        return false;
    };

    let down = gravity.down().into_inner();
    unsafe {
        *out_estimate = ArbitGravityEstimate {
            down: [down.x, down.y, down.z],
            samples: context.gravity_estimator.sample_count() as u32,
        };
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
