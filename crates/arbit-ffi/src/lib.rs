use std::ptr;
use std::slice;
use std::sync::Arc;
use std::time::Duration;

use arbit_core::math::CameraIntrinsics;
use arbit_core::math::se3::TransformSE3;
use arbit_core::time::FrameTimestamps;
use arbit_core::track::TrackOutcome;
use arbit_engine::ProcessingEngine;
use arbit_providers::{ArKitFrame, ArKitIntrinsics, CameraSample, IosCameraProvider, PixelFormat};
use log::{info, warn};
use nalgebra::{Matrix4, Translation3, UnitQuaternion, Vector3};

struct CaptureContext {
    engine: ProcessingEngine,
    provider: IosCameraProvider,
}

impl Default for CaptureContext {
    fn default() -> Self {
        Self {
            engine: ProcessingEngine::new(),
            provider: IosCameraProvider::new(),
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

        let slice = unsafe { slice::from_raw_parts(self.distortion, self.distortion_len) };
        Some(slice.to_vec())
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
    fn from_sample(sample: &CameraSample) -> Self {
        Self {
            timestamps: sample.timestamps.into(),
            intrinsics: (&sample.intrinsics).into(),
            pixel_format: pixel_format_to_ffi(sample.pixel_format),
            bytes_per_row: sample.bytes_per_row,
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
pub struct ArbitImuSample {
    pub timestamp_seconds: f64,
    pub accel_x: f64,
    pub accel_y: f64,
    pub accel_z: f64,
    pub gyro_x: f64,
    pub gyro_y: f64,
    pub gyro_z: f64,
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

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ArbitRelocalizationSummary {
    pub pose: ArbitTransform,
    pub inliers: u32,
    pub average_error: f64,
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

    let Some(arkit_frame) = build_arkit_frame(frame) else {
        warn!("Failed to convert camera frame for processing");
        return false;
    };

    // Use IosCameraProvider to convert ArKitFrame → CameraSample
    let sample = context.provider.ingest_frame(arkit_frame);

    // Pass the generic CameraSample to the engine
    context.engine.ingest_camera_sample(&sample);

    unsafe {
        *out_sample = ArbitCameraSample::from_sample(&sample);
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
    let levels = context.engine.pyramid_levels();
    let count = levels.len().min(max_levels);
    let dest = unsafe { slice::from_raw_parts_mut(out_levels, count) };
    for (dst, cache) in dest.iter_mut().zip(levels.iter()) {
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
    let tracks = context.engine.tracked_points();
    let count = tracks.len().min(max_points);
    let dest = unsafe { slice::from_raw_parts_mut(out_points, count) };
    for (dst, track) in dest.iter_mut().zip(tracks.iter()) {
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
    let Some(result) = context.engine.latest_two_view() else {
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
    let trajectory = context.engine.trajectory();
    let count = trajectory.len().min(max_points);
    let dest = unsafe { slice::from_raw_parts_mut(out_points, count) };
    for (dst, pos) in dest.iter_mut().zip(trajectory.iter()) {
        *dst = ArbitPoseSample {
            x: pos.x,
            y: pos.y,
            z: pos.z,
        };
    }
    count
}

/// Ingest a full 6DOF IMU sample (accelerometer + gyroscope).
/// This is the preferred method for feeding IMU data as it enables IMU preintegration.
///
/// * `timestamp_seconds` - Timestamp in seconds
/// * `accel_*` - Accelerometer reading in m/s² (x, y, z)
/// * `gyro_*` - Gyroscope reading in rad/s (x, y, z)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_ingest_imu_sample(
    handle: *mut ArbitCaptureContextHandle,
    sample: ArbitImuSample,
) -> bool {
    if handle.is_null() {
        return false;
    }

    let context = unsafe { &mut (*handle).inner };
    context.engine.ingest_imu_sample(
        sample.timestamp_seconds,
        (sample.gyro_x, sample.gyro_y, sample.gyro_z),
        (sample.accel_x, sample.accel_y, sample.accel_z),
    );
    true
}

/// Returns the last IMU rotation prior (in radians) if available.
///
/// * `out_rotation_radians` - Output pointer for rotation magnitude
///
/// Returns true if a rotation prior is available.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_last_imu_rotation_prior(
    handle: *mut ArbitCaptureContextHandle,
    out_rotation_radians: *mut f64,
) -> bool {
    if handle.is_null() || out_rotation_radians.is_null() {
        return false;
    }

    let context = unsafe { &mut (*handle).inner };
    if let Some(rotation) = context.engine.last_imu_rotation_prior() {
        unsafe {
            *out_rotation_radians = rotation;
        }
        true
    } else {
        false
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ArbitMotionState {
    pub state: u32, // 0 = Stationary, 1 = SlowMotion, 2 = FastMotion
}

/// Returns the last motion state if available.
///
/// * `out_state` - Output pointer for motion state (0=Stationary, 1=Slow, 2=Fast)
///
/// Returns true if a motion state is available.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_last_motion_state(
    handle: *mut ArbitCaptureContextHandle,
    out_state: *mut ArbitMotionState,
) -> bool {
    if handle.is_null() || out_state.is_null() {
        return false;
    }

    let context = unsafe { &mut (*handle).inner };
    if let Some(state_str) = context.engine.last_motion_state() {
        let state_code = match state_str.as_str() {
            "Stationary" => 0,
            "SlowMotion" => 1,
            "FastMotion" => 2,
            _ => 0,
        };
        unsafe {
            *out_state = ArbitMotionState { state: state_code };
        }
        true
    } else {
        false
    }
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
    let Some(gravity) = context.engine.gravity_estimate() else {
        return false;
    };

    let down = gravity.down().into_inner();
    unsafe {
        *out_estimate = ArbitGravityEstimate {
            down: [down.x, down.y, down.z],
            samples: context.engine.gravity_sample_count(),
        };
    }
    true
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_capture_context_map_stats(
    handle: *mut ArbitCaptureContextHandle,
    out_keyframes: *mut u64,
    out_landmarks: *mut u64,
    out_anchors: *mut u64,
) -> bool {
    if handle.is_null() {
        return false;
    }

    let context = unsafe { &mut (*handle).inner };
    let (keyframes, landmarks, anchors) = context.engine.map_stats();

    if !out_keyframes.is_null() {
        unsafe {
            *out_keyframes = keyframes;
        }
    }
    if !out_landmarks.is_null() {
        unsafe {
            *out_landmarks = landmarks;
        }
    }
    if !out_anchors.is_null() {
        unsafe {
            *out_anchors = anchors;
        }
    }
    true
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_capture_context_list_anchors(
    handle: *mut ArbitCaptureContextHandle,
    out_ids: *mut u64,
    max_ids: usize,
) -> usize {
    if handle.is_null() || out_ids.is_null() || max_ids == 0 {
        return 0;
    }

    let context = unsafe { &mut (*handle).inner };
    let mut ids = context.engine.anchor_ids();
    ids.sort_unstable();
    let count = ids.len().min(max_ids);
    let dest = unsafe { slice::from_raw_parts_mut(out_ids, count) };
    for (dst, id) in dest.iter_mut().zip(ids.iter()) {
        *dst = *id;
    }
    count
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_capture_context_create_anchor(
    handle: *mut ArbitCaptureContextHandle,
    pose: *const ArbitTransform,
    out_id: *mut u64,
) -> bool {
    if handle.is_null() || pose.is_null() {
        return false;
    }

    let context = unsafe { &mut (*handle).inner };
    let pose = unsafe { &*pose };
    let Some(transform) = transform_from_ffi(pose) else {
        return false;
    };

    let id = context.engine.create_anchor(transform);
    if !out_id.is_null() {
        unsafe {
            *out_id = id;
        }
    }
    true
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_capture_context_resolve_anchor(
    handle: *mut ArbitCaptureContextHandle,
    anchor_id: u64,
    out_pose: *mut ArbitTransform,
) -> bool {
    if handle.is_null() || out_pose.is_null() {
        return false;
    }

    let context = unsafe { &mut (*handle).inner };
    let Some(anchor) = context.engine.resolve_anchor(anchor_id) else {
        return false;
    };

    let pose = transform_to_ffi(&anchor.pose);
    unsafe {
        *out_pose = pose;
    }
    true
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_capture_context_update_anchor(
    handle: *mut ArbitCaptureContextHandle,
    anchor_id: u64,
    pose: *const ArbitTransform,
) -> bool {
    if handle.is_null() || pose.is_null() {
        return false;
    }

    let context = unsafe { &mut (*handle).inner };
    let pose = unsafe { &*pose };
    let Some(transform) = transform_from_ffi(pose) else {
        return false;
    };

    context.engine.update_anchor(anchor_id, transform)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_capture_context_last_relocalization(
    handle: *mut ArbitCaptureContextHandle,
    out_summary: *mut ArbitRelocalizationSummary,
) -> bool {
    if handle.is_null() || out_summary.is_null() {
        return false;
    }

    let context = unsafe { &mut (*handle).inner };
    let Some(result) = context.engine.last_relocalization() else {
        return false;
    };

    let pose = transform_to_ffi(&result.pose);
    unsafe {
        *out_summary = ArbitRelocalizationSummary {
            pose,
            inliers: result.inliers.len() as u32,
            average_error: result.average_reprojection_error,
        };
    }
    true
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_capture_context_save_map(
    handle: *mut ArbitCaptureContextHandle,
    out_buffer: *mut u8,
    buffer_len: usize,
    out_written: *mut usize,
) -> bool {
    if handle.is_null() {
        return false;
    }

    let context = unsafe { &mut (*handle).inner };
    let bytes = match context.engine.save_map() {
        Ok(bytes) => bytes,
        Err(err) => {
            warn!("Failed to serialize map: {err}");
            return false;
        }
    };

    if !out_written.is_null() {
        unsafe {
            *out_written = bytes.len();
        }
    }

    if out_buffer.is_null() || buffer_len < bytes.len() {
        return false;
    }

    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), out_buffer, bytes.len());
    }
    true
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_capture_context_load_map(
    handle: *mut ArbitCaptureContextHandle,
    data: *const u8,
    data_len: usize,
) -> bool {
    if handle.is_null() || data.is_null() || data_len == 0 {
        return false;
    }

    let context = unsafe { &mut (*handle).inner };
    let bytes = unsafe { slice::from_raw_parts(data, data_len) };
    match context.engine.load_map(bytes) {
        Ok(()) => true,
        Err(err) => {
            warn!("Failed to load map: {err}");
            false
        }
    }
}

fn build_arkit_frame(frame: &ArbitCameraFrame) -> Option<ArKitFrame> {
    let pixel_format = pixel_format_from_ffi(frame.pixel_format)?;
    let intrinsics = arkit_intrinsics_from_ffi(&frame.intrinsics);

    let width = intrinsics.width as usize;
    let height = intrinsics.height as usize;
    if width == 0 || height == 0 {
        return None;
    }

    if frame.data.is_null() || frame.data_len == 0 {
        return None;
    }

    let expected = frame.bytes_per_row.saturating_mul(height);
    let actual_len = frame.data_len.min(expected);
    let bytes = unsafe { slice::from_raw_parts(frame.data, actual_len) };
    let data = Arc::from(bytes.to_vec());

    let timestamp = if frame.timestamp_seconds <= 0.0 {
        Duration::from_secs(0)
    } else {
        Duration::from_secs_f64(frame.timestamp_seconds)
    };

    Some(ArKitFrame {
        timestamp,
        intrinsics,
        pixel_format,
        bytes_per_row: frame.bytes_per_row,
        data,
    })
}

fn pixel_format_from_ffi(format: ArbitPixelFormat) -> Option<PixelFormat> {
    match format {
        ArbitPixelFormat::Bgra8 => Some(PixelFormat::Bgra8),
        ArbitPixelFormat::Rgba8 => Some(PixelFormat::Rgba8),
        ArbitPixelFormat::Nv12 => Some(PixelFormat::Nv12),
        ArbitPixelFormat::Yv12 => Some(PixelFormat::Yv12),
        ArbitPixelFormat::Depth16 => Some(PixelFormat::Depth16),
    }
}

fn pixel_format_to_ffi(format: PixelFormat) -> ArbitPixelFormat {
    match format {
        PixelFormat::Bgra8 => ArbitPixelFormat::Bgra8,
        PixelFormat::Rgba8 => ArbitPixelFormat::Rgba8,
        PixelFormat::Nv12 => ArbitPixelFormat::Nv12,
        PixelFormat::Yv12 => ArbitPixelFormat::Yv12,
        PixelFormat::Depth16 => ArbitPixelFormat::Depth16,
    }
}

fn arkit_intrinsics_from_ffi(intrinsics: &ArbitCameraIntrinsics) -> ArKitIntrinsics {
    ArKitIntrinsics {
        fx: intrinsics.fx,
        fy: intrinsics.fy,
        cx: intrinsics.cx,
        cy: intrinsics.cy,
        skew: intrinsics.skew,
        width: intrinsics.width,
        height: intrinsics.height,
        distortion: intrinsics.distortion_coeffs(),
    }
}

fn transform_to_ffi(transform: &TransformSE3) -> ArbitTransform {
    let mut output = ArbitTransform::default();
    let matrix = transform.to_homogeneous();
    for (idx, value) in matrix.iter().enumerate() {
        output.elements[idx] = *value;
    }
    output
}

fn transform_from_ffi(raw: &ArbitTransform) -> Option<TransformSE3> {
    let matrix = Matrix4::from_row_slice(&raw.elements);
    let last_row = matrix.row(3);
    if last_row[3].abs() < f64::EPSILON {
        return None;
    }
    if last_row[0].abs() > 1e-6 || last_row[1].abs() > 1e-6 || last_row[2].abs() > 1e-6 {
        return None;
    }
    if (last_row[3] - 1.0).abs() > 1e-6 {
        return None;
    }
    let rotation_matrix = matrix.fixed_view::<3, 3>(0, 0).into_owned();
    let translation = Vector3::new(matrix[(0, 3)], matrix[(1, 3)], matrix[(2, 3)]);
    let rotation = UnitQuaternion::from_matrix(&rotation_matrix);
    Some(TransformSE3::from_parts(
        Translation3::from(translation),
        rotation,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

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
            assert!(!arbit_ingest_camera_frame(
                handle,
                &frame,
                &mut sample as *mut _
            ));
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
            assert!(!arbit_ingest_camera_frame(handle, &frame, &mut sample));

            frame.timestamp_seconds = 0.5;
            assert!(!arbit_ingest_camera_frame(handle, &frame, &mut sample));

            arbit_capture_context_free(handle);
        }
    }

    #[test]
    fn anchors_round_trip_via_ffi() {
        let handle = arbit_capture_context_new();
        assert!(!handle.is_null());

        let mut keyframes = 0u64;
        let mut landmarks = 0u64;
        let mut anchors = 0u64;
        unsafe {
            assert!(arbit_capture_context_map_stats(
                handle,
                &mut keyframes,
                &mut landmarks,
                &mut anchors
            ));
        }
        assert_eq!(anchors, 0);

        let identity = ArbitTransform {
            elements: [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        };
        let mut anchor_id = 0u64;
        unsafe {
            assert!(arbit_capture_context_create_anchor(
                handle,
                &identity as *const _,
                &mut anchor_id as *mut _
            ));
        }

        unsafe {
            assert!(arbit_capture_context_map_stats(
                handle,
                &mut keyframes,
                &mut landmarks,
                &mut anchors
            ));
        }
        assert_eq!(anchors, 1);

        let mut resolved = ArbitTransform::default();
        unsafe {
            assert!(arbit_capture_context_resolve_anchor(
                handle,
                anchor_id,
                &mut resolved as *mut _
            ));
        }
        assert_eq!(resolved.elements, identity.elements);

        unsafe {
            arbit_capture_context_free(handle);
        }
    }

    #[test]
    fn map_save_and_load_round_trip() {
        let handle = arbit_capture_context_new();
        assert!(!handle.is_null());

        let identity = ArbitTransform {
            elements: [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        };
        let mut anchor_id = 0u64;
        unsafe {
            assert!(arbit_capture_context_create_anchor(
                handle,
                &identity as *const _,
                &mut anchor_id as *mut _
            ));
        }

        let mut required: usize = 0;
        unsafe {
            assert!(!arbit_capture_context_save_map(
                handle,
                ptr::null_mut(),
                0,
                &mut required
            ));
        }
        assert!(required > 0);

        let mut buffer = vec![0u8; required];
        let mut written = 0usize;
        let saved = unsafe {
            arbit_capture_context_save_map(handle, buffer.as_mut_ptr(), buffer.len(), &mut written)
        };
        assert!(saved);
        assert_eq!(written, buffer.len());

        let new_handle = arbit_capture_context_new();
        assert!(!new_handle.is_null());
        let loaded =
            unsafe { arbit_capture_context_load_map(new_handle, buffer.as_ptr(), buffer.len()) };
        assert!(loaded);

        unsafe {
            arbit_capture_context_free(handle);
            arbit_capture_context_free(new_handle);
        }
    }
}
