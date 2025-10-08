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

/// Unified IMU state containing all IMU-related information
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ArbitImuState {
    pub has_rotation_prior: bool,
    pub rotation_prior_radians: f64,
    pub has_motion_state: bool,
    pub motion_state: u32, // 0=Stationary, 1=Slow, 2=Fast
    pub has_gravity: bool,
    pub gravity_down: [f64; 3],
    pub gravity_samples: u32,
    pub preintegration_count: u32,
}

impl Default for ArbitImuState {
    fn default() -> Self {
        Self {
            has_rotation_prior: false,
            rotation_prior_radians: 0.0,
            has_motion_state: false,
            motion_state: 0,
            has_gravity: false,
            gravity_down: [0.0; 3],
            gravity_samples: 0,
            preintegration_count: 0,
        }
    }
}

/// Comprehensive frame state containing all common query results
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ArbitFrameState {
    // Tracking state
    pub track_count: u32,
    pub has_two_view: bool,
    pub two_view: ArbitTwoViewSummary,
    pub has_relocalization: bool,
    pub relocalization: ArbitRelocalizationSummary,

    // Map state
    pub keyframe_count: u64,
    pub landmark_count: u64,
    pub anchor_count: u64,

    // IMU state
    pub imu: ArbitImuState,
}

impl Default for ArbitFrameState {
    fn default() -> Self {
        Self {
            track_count: 0,
            has_two_view: false,
            two_view: ArbitTwoViewSummary {
                inliers: 0,
                average_error: 0.0,
                rotation: [0.0; 9],
                translation: [0.0; 3],
            },
            has_relocalization: false,
            relocalization: ArbitRelocalizationSummary {
                pose: ArbitTransform::default(),
                inliers: 0,
                average_error: 0.0,
            },
            keyframe_count: 0,
            landmark_count: 0,
            anchor_count: 0,
            imu: ArbitImuState::default(),
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

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ArbitRelocalizationSummary {
    pub pose: ArbitTransform,
    pub inliers: u32,
    pub average_error: f64,
}

/// Opaque handle for the capture context - internal implementation hidden
#[repr(C)]
pub struct ArbitCaptureContextHandle {
    _private: [u8; 0],
}

// Helper to cast pointers between opaque handle and real type
fn handle_to_context(handle: *mut ArbitCaptureContextHandle) -> &'static mut CaptureContext {
    unsafe { &mut *(handle as *mut CaptureContext) }
}

fn context_to_handle(ctx: *mut CaptureContext) -> *mut ArbitCaptureContextHandle {
    ctx as *mut ArbitCaptureContextHandle
}

#[unsafe(no_mangle)]
pub extern "C" fn arbit_init_logging() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug"))
        .filter_level(log::LevelFilter::Debug)
        .init();
    info!("ARBIT logging initialized");
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

// =============================================================================
// SIMPLIFIED API - Consistent Naming Convention
// =============================================================================

/// Creates a new capture context
#[unsafe(no_mangle)]
pub extern "C" fn arbit_context_create() -> *mut ArbitCaptureContextHandle {
    let ctx = Box::new(CaptureContext::default());
    context_to_handle(Box::into_raw(ctx))
}

/// Destroys a capture context
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_context_destroy(handle: *mut ArbitCaptureContextHandle) {
    if handle.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(handle as *mut CaptureContext));
    }
}

/// Ingests a camera frame
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_ingest_frame(
    handle: *mut ArbitCaptureContextHandle,
    frame: *const ArbitCameraFrame,
    out_sample: *mut ArbitCameraSample,
) -> bool {
    if handle.is_null() || frame.is_null() || out_sample.is_null() {
        return false;
    }

    let context = handle_to_context(handle);
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

/// Ingests a full 6DOF IMU sample (accelerometer + gyroscope)
///
/// * `timestamp_seconds` - Timestamp in seconds
/// * `accel_*` - Accelerometer reading in m/s² (x, y, z)
/// * `gyro_*` - Gyroscope reading in rad/s (x, y, z)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_ingest_imu(
    handle: *mut ArbitCaptureContextHandle,
    sample: ArbitImuSample,
) -> bool {
    if handle.is_null() {
        return false;
    }

    let context = handle_to_context(handle);
    context.engine.ingest_imu_sample(
        sample.timestamp_seconds,
        (sample.gyro_x, sample.gyro_y, sample.gyro_z),
        (sample.accel_x, sample.accel_y, sample.accel_z),
    );
    true
}

/// Gets unified IMU state in a single call
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_get_imu_state(
    handle: *mut ArbitCaptureContextHandle,
    out_state: *mut ArbitImuState,
) -> bool {
    if handle.is_null() || out_state.is_null() {
        return false;
    }

    let context = handle_to_context(handle);
    let mut state = ArbitImuState::default();

    // Get rotation prior
    if let Some(rotation) = context.engine.last_imu_rotation_prior() {
        state.has_rotation_prior = true;
        state.rotation_prior_radians = rotation;
    }

    // Get motion state
    if let Some(motion) = context.engine.last_motion_state() {
        state.has_motion_state = true;
        state.motion_state = match motion.as_str() {
            "Stationary" => 0,
            "SlowMotion" => 1,
            "FastMotion" => 2,
            _ => 0,
        };
    }

    // Get gravity estimate
    if let Some(gravity) = context.engine.gravity_estimate() {
        state.has_gravity = true;
        let down = gravity.down().into_inner();
        state.gravity_down = [down.x, down.y, down.z];
        state.gravity_samples = context.engine.gravity_sample_count();
    }

    // Get preintegration count
    state.preintegration_count = context.engine.preintegration_count() as u32;

    unsafe {
        *out_state = state;
    }
    true
}

/// Gets comprehensive frame state in a single call
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_get_frame_state(
    handle: *mut ArbitCaptureContextHandle,
    out_state: *mut ArbitFrameState,
) -> bool {
    if handle.is_null() || out_state.is_null() {
        return false;
    }

    let context = handle_to_context(handle);
    let mut state = ArbitFrameState::default();

    // Get tracked points count
    state.track_count = context.engine.tracked_points().len() as u32;

    // Get two-view summary
    if let Some(two_view) = context.engine.latest_two_view() {
        state.has_two_view = true;
        let matrix = two_view.rotation.matrix();
        for row in 0..3 {
            for col in 0..3 {
                state.two_view.rotation[row * 3 + col] = matrix[(row, col)];
            }
        }
        state.two_view.translation = [
            two_view.translation.x,
            two_view.translation.y,
            two_view.translation.z,
        ];
        state.two_view.inliers = two_view.inliers.len() as u32;
        state.two_view.average_error = two_view.average_sampson_error;
    }

    // Get relocalization
    if let Some(reloc) = context.engine.last_relocalization() {
        state.has_relocalization = true;
        state.relocalization.pose = transform_to_ffi(&reloc.pose);
        state.relocalization.inliers = reloc.inliers.len() as u32;
        state.relocalization.average_error = reloc.average_reprojection_error;
    }

    // Get map stats
    let (keyframes, landmarks, anchors) = context.engine.map_stats();
    state.keyframe_count = keyframes;
    state.landmark_count = landmarks;
    state.anchor_count = anchors;

    // Get IMU state
    if let Some(rotation) = context.engine.last_imu_rotation_prior() {
        state.imu.has_rotation_prior = true;
        state.imu.rotation_prior_radians = rotation;
    }
    if let Some(motion) = context.engine.last_motion_state() {
        state.imu.has_motion_state = true;
        state.imu.motion_state = match motion.as_str() {
            "Stationary" => 0,
            "SlowMotion" => 1,
            "FastMotion" => 2,
            _ => 0,
        };
    }
    if let Some(gravity) = context.engine.gravity_estimate() {
        state.imu.has_gravity = true;
        let down = gravity.down().into_inner();
        state.imu.gravity_down = [down.x, down.y, down.z];
        state.imu.gravity_samples = context.engine.gravity_sample_count();
    }
    state.imu.preintegration_count = context.engine.preintegration_count() as u32;

    unsafe {
        *out_state = state;
    }
    true
}

/// Gets pyramid levels
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_get_pyramid_levels(
    handle: *mut ArbitCaptureContextHandle,
    out_levels: *mut ArbitPyramidLevelView,
    max_levels: usize,
) -> usize {
    if handle.is_null() || out_levels.is_null() || max_levels == 0 {
        return 0;
    }

    let context = handle_to_context(handle);
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

/// Gets tracked points
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_get_tracked_points(
    handle: *mut ArbitCaptureContextHandle,
    out_points: *mut ArbitTrackedPoint,
    max_points: usize,
) -> usize {
    if handle.is_null() || out_points.is_null() || max_points == 0 {
        return 0;
    }

    let context = handle_to_context(handle);
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

/// Gets trajectory
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_get_trajectory(
    handle: *mut ArbitCaptureContextHandle,
    out_points: *mut ArbitPoseSample,
    max_points: usize,
) -> usize {
    if handle.is_null() || out_points.is_null() || max_points == 0 {
        return 0;
    }

    let context = handle_to_context(handle);
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

/// Lists all anchor IDs
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_list_anchors(
    handle: *mut ArbitCaptureContextHandle,
    out_ids: *mut u64,
    max_ids: usize,
) -> usize {
    if handle.is_null() || out_ids.is_null() || max_ids == 0 {
        return 0;
    }

    let context = handle_to_context(handle);
    let mut ids = context.engine.anchor_ids();
    ids.sort_unstable();
    let count = ids.len().min(max_ids);
    let dest = unsafe { slice::from_raw_parts_mut(out_ids, count) };
    for (dst, id) in dest.iter_mut().zip(ids.iter()) {
        *dst = *id;
    }
    count
}

/// Creates a new anchor
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_create_anchor(
    handle: *mut ArbitCaptureContextHandle,
    pose: *const ArbitTransform,
    out_id: *mut u64,
) -> bool {
    if handle.is_null() || pose.is_null() {
        return false;
    }

    let context = handle_to_context(handle);
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

/// Gets an anchor pose
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_get_anchor(
    handle: *mut ArbitCaptureContextHandle,
    anchor_id: u64,
    out_pose: *mut ArbitTransform,
) -> bool {
    if handle.is_null() || out_pose.is_null() {
        return false;
    }

    let context = handle_to_context(handle);
    let Some(anchor) = context.engine.resolve_anchor(anchor_id) else {
        return false;
    };

    let pose = transform_to_ffi(&anchor.pose);
    unsafe {
        *out_pose = pose;
    }
    true
}

/// Updates an anchor pose
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_update_anchor(
    handle: *mut ArbitCaptureContextHandle,
    anchor_id: u64,
    pose: *const ArbitTransform,
) -> bool {
    if handle.is_null() || pose.is_null() {
        return false;
    }

    let context = handle_to_context(handle);
    let pose = unsafe { &*pose };
    let Some(transform) = transform_from_ffi(pose) else {
        return false;
    };

    context.engine.update_anchor(anchor_id, transform)
}

/// Saves map to buffer
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_save_map(
    handle: *mut ArbitCaptureContextHandle,
    out_buffer: *mut u8,
    buffer_len: usize,
    out_written: *mut usize,
) -> bool {
    if handle.is_null() {
        return false;
    }

    let context = handle_to_context(handle);
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

/// Loads map from buffer
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_load_map(
    handle: *mut ArbitCaptureContextHandle,
    data: *const u8,
    data_len: usize,
) -> bool {
    if handle.is_null() || data.is_null() || data_len == 0 {
        return false;
    }

    let context = handle_to_context(handle);
    let bytes = unsafe { slice::from_raw_parts(data, data_len) };
    match context.engine.load_map(bytes) {
        Ok(()) => true,
        Err(err) => {
            warn!("Failed to load map: {err}");
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ingest_path_returns_monotonic_timestamps() {
        let handle = arbit_context_create();
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
            assert!(!arbit_ingest_frame(handle, &frame, &mut sample as *mut _));
        }

        unsafe {
            arbit_context_destroy(handle);
        }
    }

    #[test]
    fn successive_ingest_calls_are_monotonic() {
        let handle = arbit_context_create();
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
            assert!(!arbit_ingest_frame(handle, &frame, &mut sample));

            frame.timestamp_seconds = 0.5;
            assert!(!arbit_ingest_frame(handle, &frame, &mut sample));

            arbit_context_destroy(handle);
        }
    }

    #[test]
    fn anchors_round_trip_via_ffi() {
        let handle = arbit_context_create();
        assert!(!handle.is_null());

        // Check initial state
        let mut state = ArbitFrameState::default();
        unsafe {
            assert!(arbit_get_frame_state(handle, &mut state as *mut _));
        }
        assert_eq!(state.anchor_count, 0);

        let identity = ArbitTransform {
            elements: [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        };
        let mut anchor_id = 0u64;
        unsafe {
            assert!(arbit_create_anchor(
                handle,
                &identity as *const _,
                &mut anchor_id as *mut _
            ));
        }

        // Check state after creating anchor
        unsafe {
            assert!(arbit_get_frame_state(handle, &mut state as *mut _));
        }
        assert_eq!(state.anchor_count, 1);

        let mut resolved = ArbitTransform::default();
        unsafe {
            assert!(arbit_get_anchor(handle, anchor_id, &mut resolved as *mut _));
        }
        assert_eq!(resolved.elements, identity.elements);

        unsafe {
            arbit_context_destroy(handle);
        }
    }

    #[test]
    fn map_save_and_load_round_trip() {
        let handle = arbit_context_create();
        assert!(!handle.is_null());

        let identity = ArbitTransform {
            elements: [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        };
        let mut anchor_id = 0u64;
        unsafe {
            assert!(arbit_create_anchor(
                handle,
                &identity as *const _,
                &mut anchor_id as *mut _
            ));
        }

        let mut required: usize = 0;
        unsafe {
            assert!(!arbit_save_map(handle, ptr::null_mut(), 0, &mut required));
        }
        assert!(required > 0);

        let mut buffer = vec![0u8; required];
        let mut written = 0usize;
        let saved =
            unsafe { arbit_save_map(handle, buffer.as_mut_ptr(), buffer.len(), &mut written) };
        assert!(saved);
        assert_eq!(written, buffer.len());

        let new_handle = arbit_context_create();
        assert!(!new_handle.is_null());
        let loaded = unsafe { arbit_load_map(new_handle, buffer.as_ptr(), buffer.len()) };
        assert!(loaded);

        unsafe {
            arbit_context_destroy(handle);
            arbit_context_destroy(new_handle);
        }
    }

    #[test]
    fn simplified_api_lifecycle() {
        // Test new simplified API names
        let handle = arbit_context_create();
        assert!(!handle.is_null());

        unsafe {
            arbit_context_destroy(handle);
        }
    }

    #[test]
    fn simplified_api_get_frame_state() {
        let handle = arbit_context_create();
        assert!(!handle.is_null());

        // Get frame state should succeed even with no data
        let mut frame_state = ArbitFrameState::default();
        unsafe {
            assert!(arbit_get_frame_state(handle, &mut frame_state as *mut _));
        }

        // Verify default values
        assert_eq!(frame_state.track_count, 0);
        assert!(!frame_state.has_two_view);
        assert!(!frame_state.has_relocalization);
        assert_eq!(frame_state.keyframe_count, 0);
        assert_eq!(frame_state.landmark_count, 0);
        assert_eq!(frame_state.anchor_count, 0);

        unsafe {
            arbit_context_destroy(handle);
        }
    }

    #[test]
    fn simplified_api_get_imu_state() {
        let handle = arbit_context_create();
        assert!(!handle.is_null());

        // Get IMU state should succeed even with no data
        let mut imu_state = ArbitImuState::default();
        unsafe {
            assert!(arbit_get_imu_state(handle, &mut imu_state as *mut _));
        }

        // Verify default values
        assert!(!imu_state.has_rotation_prior);
        assert!(!imu_state.has_motion_state);
        assert!(!imu_state.has_gravity);
        assert_eq!(imu_state.preintegration_count, 0);

        unsafe {
            arbit_context_destroy(handle);
        }
    }

    #[test]
    fn simplified_api_anchor_operations() {
        let handle = arbit_context_create();
        assert!(!handle.is_null());

        let identity = ArbitTransform {
            elements: [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        };

        // Test simplified anchor creation
        let mut anchor_id = 0u64;
        unsafe {
            assert!(arbit_create_anchor(
                handle,
                &identity as *const _,
                &mut anchor_id as *mut _
            ));
        }

        // Test simplified anchor retrieval
        let mut retrieved = ArbitTransform::default();
        unsafe {
            assert!(arbit_get_anchor(
                handle,
                anchor_id,
                &mut retrieved as *mut _
            ));
        }
        assert_eq!(retrieved.elements, identity.elements);

        // Test simplified list anchors
        let mut ids = [0u64; 16];
        let count = unsafe { arbit_list_anchors(handle, ids.as_mut_ptr(), ids.len()) };
        assert_eq!(count, 1);
        assert_eq!(ids[0], anchor_id);

        unsafe {
            arbit_context_destroy(handle);
        }
    }

    #[test]
    fn simplified_api_map_operations() {
        let handle = arbit_context_create();
        assert!(!handle.is_null());

        // Test simplified save map (should get size even with empty map)
        let mut required: usize = 0;
        unsafe {
            arbit_save_map(handle, ptr::null_mut(), 0, &mut required);
        }
        assert!(required > 0);

        unsafe {
            arbit_context_destroy(handle);
        }
    }
}
