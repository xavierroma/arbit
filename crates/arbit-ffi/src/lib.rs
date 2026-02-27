use std::ptr;
use std::slice;
use std::sync::Arc;
use std::time::Duration;

use arbit_core::contracts::{
    AnchorSnapshot, EngineSnapshot, MapRepository, TrackingState, identity_pose,
};
use arbit_engine::SlamEngine;
use arbit_providers::{ArKitFrame, ArKitIntrinsics, CameraSample, IosCameraProvider, PixelFormat};
use log::{info, warn};
use tracing_subscriber::{EnvFilter, fmt};

struct CaptureContext {
    engine: SlamEngine,
    provider: IosCameraProvider,
}

impl Default for CaptureContext {
    fn default() -> Self {
        Self {
            engine: SlamEngine::new(),
            provider: IosCameraProvider::new(),
        }
    }
}

#[repr(C)]
pub struct ArbitCaptureContextHandle {
    _private: [u8; 0],
}

fn handle_to_context(handle: *mut ArbitCaptureContextHandle) -> &'static mut CaptureContext {
    unsafe { &mut *(handle as *mut CaptureContext) }
}

fn context_to_handle(ctx: *mut CaptureContext) -> *mut ArbitCaptureContextHandle {
    ctx as *mut ArbitCaptureContextHandle
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ArbitV2PixelFormat {
    Bgra8 = 0,
    Rgba8 = 1,
    Nv12 = 2,
    Yv12 = 3,
    Depth16 = 4,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ArbitV2CameraIntrinsics {
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

impl Default for ArbitV2CameraIntrinsics {
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

impl ArbitV2CameraIntrinsics {
    fn distortion_coeffs(&self) -> Option<Vec<f64>> {
        if self.distortion_len == 0 || self.distortion.is_null() {
            return None;
        }
        let coeffs = unsafe { slice::from_raw_parts(self.distortion, self.distortion_len) };
        Some(coeffs.to_vec())
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ArbitV2CameraFrame {
    pub timestamp_seconds: f64,
    pub intrinsics: ArbitV2CameraIntrinsics,
    pub pixel_format: ArbitV2PixelFormat,
    pub bytes_per_row: usize,
    pub data: *const u8,
    pub data_len: usize,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ArbitV2ImuSample {
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
pub struct ArbitV2TrackingSnapshot {
    pub state: u32,
    pub frame_id: u64,
    pub track_count: u32,
    pub inlier_count: u32,
    pub pose_wc: [f64; 16],
}

impl Default for ArbitV2TrackingSnapshot {
    fn default() -> Self {
        Self {
            state: 0,
            frame_id: 0,
            track_count: 0,
            inlier_count: 0,
            pose_wc: identity_pose(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct ArbitV2BackendSnapshot {
    pub keyframe_count: u64,
    pub loop_closure_events: u64,
    pub relocalization_ready: bool,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct ArbitV2MapSnapshot {
    pub landmark_count: u64,
    pub anchor_count: u64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct ArbitV2RuntimeMetricsSnapshot {
    pub frame_queue_depth: usize,
    pub imu_queue_depth: usize,
    pub keyframe_queue_depth: usize,
    pub backend_queue_depth: usize,
    pub dropped_frames: u64,
    pub frontend_ms_median: f64,
    pub frontend_ms_p95: f64,
    pub end_to_end_ms_p95: f64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ArbitV2Snapshot {
    pub timestamp_seconds: f64,
    pub tracking: ArbitV2TrackingSnapshot,
    pub backend: ArbitV2BackendSnapshot,
    pub map: ArbitV2MapSnapshot,
    pub metrics: ArbitV2RuntimeMetricsSnapshot,
}

impl Default for ArbitV2Snapshot {
    fn default() -> Self {
        Self {
            timestamp_seconds: 0.0,
            tracking: ArbitV2TrackingSnapshot::default(),
            backend: ArbitV2BackendSnapshot::default(),
            map: ArbitV2MapSnapshot::default(),
            metrics: ArbitV2RuntimeMetricsSnapshot::default(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ArbitV2Anchor {
    pub anchor_id: u64,
    pub pose_wc: [f64; 16],
    pub created_from_keyframe: u64,
    pub has_keyframe: bool,
    pub last_observed_frame: u64,
}

impl Default for ArbitV2Anchor {
    fn default() -> Self {
        Self {
            anchor_id: 0,
            pose_wc: identity_pose(),
            created_from_keyframe: 0,
            has_keyframe: false,
            last_observed_frame: 0,
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn arbit_init_logging() {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_timer(fmt::time::uptime())
        .with_level(true)
        .with_ansi(false)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .try_init()
        .ok();
    info!("ARBIT logging initialized");
}

#[unsafe(no_mangle)]
pub extern "C" fn arbit_v2_context_create() -> *mut ArbitCaptureContextHandle {
    let context = Box::new(CaptureContext::default());
    context_to_handle(Box::into_raw(context))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_v2_context_destroy(handle: *mut ArbitCaptureContextHandle) {
    if handle.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(handle as *mut CaptureContext));
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_v2_ingest_frame(
    handle: *mut ArbitCaptureContextHandle,
    frame: *const ArbitV2CameraFrame,
) -> bool {
    if handle.is_null() || frame.is_null() {
        return false;
    }

    let context = handle_to_context(handle);
    let frame = unsafe { &*frame };

    let Some(arkit_frame) = build_v2_arkit_frame(frame) else {
        warn!("v2 ingest_frame rejected invalid frame");
        return false;
    };

    let sample = context.provider.ingest_frame(arkit_frame);
    context.engine.ingest_frame(&sample)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_v2_ingest_imu(
    handle: *mut ArbitCaptureContextHandle,
    sample: *const ArbitV2ImuSample,
) -> bool {
    if handle.is_null() || sample.is_null() {
        return false;
    }

    let context = handle_to_context(handle);
    let sample = unsafe { &*sample };

    context.engine.ingest_imu(
        sample.timestamp_seconds,
        [sample.accel_x, sample.accel_y, sample.accel_z],
        [sample.gyro_x, sample.gyro_y, sample.gyro_z],
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_v2_get_snapshot(
    handle: *mut ArbitCaptureContextHandle,
    out_snapshot: *mut ArbitV2Snapshot,
) -> bool {
    if handle.is_null() || out_snapshot.is_null() {
        return false;
    }

    let context = handle_to_context(handle);
    let snapshot = context.engine.snapshot();
    unsafe {
        *out_snapshot = snapshot_to_v2(&snapshot);
    }
    true
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_v2_create_anchor(
    handle: *mut ArbitCaptureContextHandle,
    pose_wc: *const f64,
    has_keyframe_hint: bool,
    keyframe_hint: u64,
    out_anchor_id: *mut u64,
) -> bool {
    if handle.is_null() || out_anchor_id.is_null() {
        return false;
    }

    let context = handle_to_context(handle);
    let pose = if pose_wc.is_null() {
        identity_pose()
    } else {
        let values = unsafe { slice::from_raw_parts(pose_wc, 16) };
        let mut pose = [0.0; 16];
        pose.copy_from_slice(values);
        pose
    };

    let keyframe = has_keyframe_hint.then_some(keyframe_hint);
    let anchor_id = context.engine.create_anchor(pose, keyframe);
    unsafe {
        *out_anchor_id = anchor_id;
    }
    true
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_v2_query_anchor(
    handle: *mut ArbitCaptureContextHandle,
    anchor_id: u64,
    out_anchor: *mut ArbitV2Anchor,
) -> bool {
    if handle.is_null() || out_anchor.is_null() {
        return false;
    }

    let context = handle_to_context(handle);
    let Some(anchor) = context.engine.query_anchor(anchor_id) else {
        return false;
    };

    unsafe {
        *out_anchor = anchor_to_v2(&anchor);
    }
    true
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_v2_reset_session(handle: *mut ArbitCaptureContextHandle) -> bool {
    if handle.is_null() {
        return false;
    }

    let context = handle_to_context(handle);
    context.engine.reset_session();
    true
}

fn snapshot_to_v2(snapshot: &EngineSnapshot) -> ArbitV2Snapshot {
    ArbitV2Snapshot {
        timestamp_seconds: snapshot.timestamp_seconds,
        tracking: ArbitV2TrackingSnapshot {
            state: tracking_state_to_u32(snapshot.tracking.state),
            frame_id: snapshot.tracking.frame_id,
            track_count: snapshot.tracking.track_count,
            inlier_count: snapshot.tracking.inlier_count,
            pose_wc: snapshot.tracking.pose_wc,
        },
        backend: ArbitV2BackendSnapshot {
            keyframe_count: snapshot.backend.keyframe_count,
            loop_closure_events: snapshot.backend.loop_closure_events,
            relocalization_ready: snapshot.backend.relocalization_ready,
        },
        map: ArbitV2MapSnapshot {
            landmark_count: snapshot.map.landmark_count,
            anchor_count: snapshot.map.anchor_count,
        },
        metrics: ArbitV2RuntimeMetricsSnapshot {
            frame_queue_depth: snapshot.metrics.frame_queue_depth,
            imu_queue_depth: snapshot.metrics.imu_queue_depth,
            keyframe_queue_depth: snapshot.metrics.keyframe_queue_depth,
            backend_queue_depth: snapshot.metrics.backend_queue_depth,
            dropped_frames: snapshot.metrics.dropped_frames,
            frontend_ms_median: snapshot.metrics.frontend_ms_median,
            frontend_ms_p95: snapshot.metrics.frontend_ms_p95,
            end_to_end_ms_p95: snapshot.metrics.end_to_end_ms_p95,
        },
    }
}

fn anchor_to_v2(anchor: &AnchorSnapshot) -> ArbitV2Anchor {
    ArbitV2Anchor {
        anchor_id: anchor.anchor_id,
        pose_wc: anchor.pose_wc,
        created_from_keyframe: anchor.created_from_keyframe.unwrap_or(0),
        has_keyframe: anchor.created_from_keyframe.is_some(),
        last_observed_frame: anchor.last_observed_frame,
    }
}

fn tracking_state_to_u32(state: TrackingState) -> u32 {
    match state {
        TrackingState::Initializing => 0,
        TrackingState::Tracking => 1,
        TrackingState::Relocalizing => 2,
        TrackingState::Lost => 3,
    }
}

fn build_v2_arkit_frame(frame: &ArbitV2CameraFrame) -> Option<ArKitFrame> {
    let pixel_format = pixel_format_from_v2(frame.pixel_format)?;
    let intrinsics = arkit_intrinsics_from_v2(&frame.intrinsics);

    if frame.data.is_null() || frame.data_len == 0 {
        return None;
    }

    let height = intrinsics.height as usize;
    if height == 0 || frame.bytes_per_row == 0 {
        return None;
    }

    let expected = frame.bytes_per_row.saturating_mul(height);
    let copied = frame.data_len.min(expected);
    let bytes = unsafe { slice::from_raw_parts(frame.data, copied) };

    Some(ArKitFrame {
        timestamp: duration_from_seconds(frame.timestamp_seconds),
        intrinsics,
        pixel_format,
        bytes_per_row: frame.bytes_per_row,
        data: Arc::from(bytes.to_vec()),
    })
}

fn arkit_intrinsics_from_v2(intrinsics: &ArbitV2CameraIntrinsics) -> ArKitIntrinsics {
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

fn pixel_format_from_v2(format: ArbitV2PixelFormat) -> Option<PixelFormat> {
    match format {
        ArbitV2PixelFormat::Bgra8 => Some(PixelFormat::Bgra8),
        ArbitV2PixelFormat::Rgba8 => Some(PixelFormat::Rgba8),
        ArbitV2PixelFormat::Nv12 => Some(PixelFormat::Nv12),
        ArbitV2PixelFormat::Yv12 => Some(PixelFormat::Yv12),
        ArbitV2PixelFormat::Depth16 => Some(PixelFormat::Depth16),
    }
}

fn duration_from_seconds(seconds: f64) -> Duration {
    if seconds.is_finite() && seconds >= 0.0 {
        Duration::from_secs_f64(seconds)
    } else {
        Duration::from_secs(0)
    }
}

fn v2_from_v1_pixel_format(format: ArbitPixelFormat) -> ArbitV2PixelFormat {
    match format {
        ArbitPixelFormat::Bgra8 => ArbitV2PixelFormat::Bgra8,
        ArbitPixelFormat::Rgba8 => ArbitV2PixelFormat::Rgba8,
        ArbitPixelFormat::Nv12 => ArbitV2PixelFormat::Nv12,
        ArbitPixelFormat::Yv12 => ArbitV2PixelFormat::Yv12,
        ArbitPixelFormat::Depth16 => ArbitV2PixelFormat::Depth16,
    }
}

fn v1_intrinsics_to_v2(intrinsics: &ArbitCameraIntrinsics) -> ArbitV2CameraIntrinsics {
    ArbitV2CameraIntrinsics {
        fx: intrinsics.fx,
        fy: intrinsics.fy,
        cx: intrinsics.cx,
        cy: intrinsics.cy,
        skew: intrinsics.skew,
        width: intrinsics.width,
        height: intrinsics.height,
        distortion_len: intrinsics.distortion_len,
        distortion: intrinsics.distortion,
    }
}

fn sample_to_v1(sample: &CameraSample) -> ArbitCameraSample {
    ArbitCameraSample {
        timestamps: ArbitFrameTimestamps {
            capture_seconds: sample.timestamps.capture.as_duration().as_secs_f64(),
            pipeline_seconds: sample.timestamps.pipeline.as_duration().as_secs_f64(),
            latency_seconds: sample.timestamps.latency.as_secs_f64(),
        },
        intrinsics: ArbitCameraIntrinsics {
            fx: sample.intrinsics.fx,
            fy: sample.intrinsics.fy,
            cx: sample.intrinsics.cx,
            cy: sample.intrinsics.cy,
            skew: sample.intrinsics.skew,
            width: sample.intrinsics.width,
            height: sample.intrinsics.height,
            distortion_len: 0,
            distortion: ptr::null(),
        },
        pixel_format: match sample.pixel_format {
            PixelFormat::Bgra8 => ArbitPixelFormat::Bgra8,
            PixelFormat::Rgba8 => ArbitPixelFormat::Rgba8,
            PixelFormat::Nv12 => ArbitPixelFormat::Nv12,
            PixelFormat::Yv12 => ArbitPixelFormat::Yv12,
            PixelFormat::Depth16 => ArbitPixelFormat::Depth16,
        },
        bytes_per_row: sample.bytes_per_row,
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

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ArbitCameraSample {
    pub timestamps: ArbitFrameTimestamps,
    pub intrinsics: ArbitCameraIntrinsics,
    pub pixel_format: ArbitPixelFormat,
    pub bytes_per_row: usize,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ArbitImuState {
    pub has_rotation_prior: bool,
    pub rotation_prior_radians: f64,
    pub has_motion_state: bool,
    pub motion_state: u32,
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

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct ArbitTwoViewSummary {
    pub inliers: u32,
    pub average_error: f64,
    pub rotation: [f64; 9],
    pub translation: [f64; 3],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ArbitTransform {
    pub elements: [f64; 16],
}

impl Default for ArbitTransform {
    fn default() -> Self {
        Self {
            elements: identity_pose(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct ArbitRelocalizationSummary {
    pub pose: ArbitTransform,
    pub inliers: u32,
    pub average_error: f64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ArbitFrameState {
    pub track_count: u32,
    pub has_two_view: bool,
    pub two_view: ArbitTwoViewSummary,
    pub has_relocalization: bool,
    pub relocalization: ArbitRelocalizationSummary,
    pub keyframe_count: u64,
    pub landmark_count: u64,
    pub anchor_count: u64,
    pub imu: ArbitImuState,
}

impl Default for ArbitFrameState {
    fn default() -> Self {
        Self {
            track_count: 0,
            has_two_view: false,
            two_view: ArbitTwoViewSummary::default(),
            has_relocalization: false,
            relocalization: ArbitRelocalizationSummary::default(),
            keyframe_count: 0,
            landmark_count: 0,
            anchor_count: 0,
            imu: ArbitImuState::default(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ArbitTrackStatus {
    Converged = 0,
    Diverged = 1,
    OutOfBounds = 2,
}

impl Default for ArbitTrackStatus {
    fn default() -> Self {
        Self::Converged
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
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
#[derive(Debug, Copy, Clone, Default)]
pub struct ArbitTrackedPoint {
    pub initial_x: f32,
    pub initial_y: f32,
    pub refined_x: f32,
    pub refined_y: f32,
    pub residual: f32,
    pub iterations: u32,
    pub status: ArbitTrackStatus,
    pub track_id: u64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct ArbitPoseSample {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct ArbitFeatDescriptor {
    pub level: u32,
    pub seed_x: f32,
    pub seed_y: f32,
    pub score: f32,
    pub angle: f32,
    pub data_len: usize,
    pub data: [u8; 32],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct ArbitMatch {
    pub query_idx: u32,
    pub train_idx: u32,
    pub distance: u32,
    pub query_x: f32,
    pub query_y: f32,
    pub train_x: f32,
    pub train_y: f32,
}

#[unsafe(no_mangle)]
pub extern "C" fn arbit_context_create() -> *mut ArbitCaptureContextHandle {
    arbit_v2_context_create()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_context_destroy(handle: *mut ArbitCaptureContextHandle) {
    unsafe { arbit_v2_context_destroy(handle) }
}

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

    let v2_frame = ArbitV2CameraFrame {
        timestamp_seconds: frame.timestamp_seconds,
        intrinsics: v1_intrinsics_to_v2(&frame.intrinsics),
        pixel_format: v2_from_v1_pixel_format(frame.pixel_format),
        bytes_per_row: frame.bytes_per_row,
        data: frame.data,
        data_len: frame.data_len,
    };

    let Some(arkit_frame) = build_v2_arkit_frame(&v2_frame) else {
        return false;
    };

    let sample = context.provider.ingest_frame(arkit_frame);
    let ok = context.engine.ingest_frame(&sample);
    if !ok {
        return false;
    }

    unsafe {
        *out_sample = sample_to_v1(&sample);
    }
    true
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_get_frame_state(
    handle: *mut ArbitCaptureContextHandle,
    out_state: *mut ArbitFrameState,
) -> bool {
    if handle.is_null() || out_state.is_null() {
        return false;
    }

    let context = handle_to_context(handle);
    let snapshot = context.engine.snapshot();

    let mut state = ArbitFrameState::default();
    state.track_count = snapshot.tracking.track_count;
    state.has_relocalization = snapshot.backend.relocalization_ready;
    state.keyframe_count = snapshot.backend.keyframe_count;
    state.landmark_count = snapshot.map.landmark_count;
    state.anchor_count = snapshot.map.anchor_count;

    unsafe {
        *out_state = state;
    }

    true
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_get_pyramid_levels(
    _handle: *mut ArbitCaptureContextHandle,
    _out_levels: *mut ArbitPyramidLevelView,
    _max_levels: usize,
) -> usize {
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_get_descriptors(
    _handle: *mut ArbitCaptureContextHandle,
    _out_descriptors: *mut ArbitFeatDescriptor,
    _max_descriptors: usize,
) -> usize {
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_match_descriptors(
    _query_descriptors: *const ArbitFeatDescriptor,
    _query_count: usize,
    _train_descriptors: *const ArbitFeatDescriptor,
    _train_count: usize,
    _out_matches: *mut ArbitMatch,
    _max_matches: usize,
    _cross_check: bool,
    _max_distance: u32,
) -> usize {
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_get_tracked_points(
    _handle: *mut ArbitCaptureContextHandle,
    _out_points: *mut ArbitTrackedPoint,
    _max_points: usize,
) -> usize {
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_get_trajectory(
    _handle: *mut ArbitCaptureContextHandle,
    _out_points: *mut ArbitPoseSample,
    _max_points: usize,
) -> usize {
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v2_lifecycle_and_snapshot_work() {
        let handle = arbit_v2_context_create();
        assert!(!handle.is_null());

        let mut snapshot = ArbitV2Snapshot::default();
        unsafe {
            assert!(arbit_v2_get_snapshot(handle, &mut snapshot as *mut _));
        }

        assert_eq!(snapshot.tracking.frame_id, 0);
        assert_eq!(snapshot.map.anchor_count, 0);

        unsafe {
            arbit_v2_context_destroy(handle);
        }
    }

    #[test]
    fn v1_lifecycle_and_frame_state_work() {
        let handle = arbit_context_create();
        assert!(!handle.is_null());

        let mut frame_state = ArbitFrameState::default();
        unsafe {
            assert!(arbit_get_frame_state(handle, &mut frame_state as *mut _));
        }
        assert_eq!(frame_state.track_count, 0);

        unsafe {
            arbit_context_destroy(handle);
        }
    }
}
