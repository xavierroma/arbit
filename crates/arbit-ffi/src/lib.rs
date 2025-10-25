use std::ptr;
use std::slice;
use std::sync::Arc;
use std::time::Duration;

use arbit_core::math::CameraIntrinsics;
use arbit_core::math::se3::TransformSE3;
use arbit_core::time::FrameTimestamps;
use arbit_core::track::{
    FastSeeder, FeatDescriptorExtractor, FeatureSeed, FeatureSeederTrait, OrbDescriptor,
    TrackOutcome,
};
use arbit_engine::ProcessingEngine;
use arbit_providers::{ArKitFrame, ArKitIntrinsics, CameraSample, IosCameraProvider, PixelFormat};
use log::{info, warn};
use tracing_subscriber::{EnvFilter, fmt};

struct CaptureContext<
    S: FeatureSeederTrait = FastSeeder,
    D: FeatDescriptorExtractor = OrbDescriptor,
> {
    engine: ProcessingEngine<S, D>,
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

const ORB_DESCRIPTOR_LEN: usize = OrbDescriptor::LEN;

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
    pub track_id: u64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ArbitFeatDescriptor {
    pub level: u32,
    pub seed_x: f32,
    pub seed_y: f32,
    pub score: f32,
    pub angle: f32,
    pub data_len: usize,
    pub data: [u8; ORB_DESCRIPTOR_LEN],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ArbitMatch {
    pub query_idx: u32,
    pub train_idx: u32,
    pub distance: u32,
    pub query_x: f32,
    pub query_y: f32,
    pub train_x: f32,
    pub train_y: f32,
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
pub struct ArbitProjectedAnchor {
    pub anchor_id: u64,
    pub pose: ArbitTransform,
    pub created_from_keyframe: u64,
    pub has_keyframe: bool,
    pub normalized_u: f64,
    pub normalized_v: f64,
    pub pixel_x: f32,
    pub pixel_y: f32,
    pub depth: f64,
}

impl Default for ArbitProjectedAnchor {
    fn default() -> Self {
        Self {
            anchor_id: 0,
            pose: ArbitTransform::default(),
            created_from_keyframe: 0,
            has_keyframe: false,
            normalized_u: 0.0,
            normalized_v: 0.0,
            pixel_x: 0.0,
            pixel_y: 0.0,
            depth: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ArbitProjectedLandmark {
    pub landmark_id: u64,
    pub world_x: f64,
    pub world_y: f64,
    pub world_z: f64,
    pub normalized_u: f64,
    pub normalized_v: f64,
    pub pixel_x: f32,
    pub pixel_y: f32,
    pub depth: f64,
}

impl Default for ArbitProjectedLandmark {
    fn default() -> Self {
        Self {
            landmark_id: 0,
            world_x: 0.0,
            world_y: 0.0,
            world_z: 0.0,
            normalized_u: 0.0,
            normalized_v: 0.0,
            pixel_x: 0.0,
            pixel_y: 0.0,
            depth: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ArbitMapDebugSnapshot {
    pub camera_x: f64,
    pub camera_y: f64,
    pub camera_z: f64,
    pub camera_rotation: [f64; 9],
    pub landmark_count: u64,
    pub keyframe_count: u64,
    pub anchor_count: u64,
}

impl Default for ArbitMapDebugSnapshot {
    fn default() -> Self {
        Self {
            camera_x: 0.0,
            camera_y: 0.0,
            camera_z: 0.0,
            camera_rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            landmark_count: 0,
            keyframe_count: 0,
            anchor_count: 0,
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
    // Initialize tracing subscriber with span timing enabled
    tracing_subscriber::fmt()
        .with_target(false)
        .with_timer(fmt::time::uptime())
        .with_level(true)
        .with_ansi(false) // Disable ANSI colors for clean output
        .with_span_events(fmt::format::FmtSpan::CLOSE) // Show span timing!
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug")),
        )
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

    if intrinsics.fx.abs() < 1e-6 || intrinsics.fy.abs() < 1e-6 {
        warn!(
            "Rejecting frame with invalid focal lengths: fx={}, fy={}",
            intrinsics.fx, intrinsics.fy
        );
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
        let matrix = two_view.rotation_c2c1.matrix();
        for row in 0..3 {
            for col in 0..3 {
                state.two_view.rotation[row * 3 + col] = matrix[(row, col)];
            }
        }
        state.two_view.translation = [
            two_view.translation_c2c1.x,
            two_view.translation_c2c1.y,
            two_view.translation_c2c1.z,
        ];
        state.two_view.inliers = two_view.inliers.len() as u32;
        state.two_view.average_error = two_view.average_sampson_error;
    }

    // Get map stats
    let (keyframes, landmarks, anchors) = context.engine.map_stats();
    state.keyframe_count = keyframes;
    state.landmark_count = landmarks;
    state.anchor_count = anchors;

    // // Get IMU state
    // if let Some(rotation) = context.engine.last_imu_rotation_prior() {
    //     state.imu.has_rotation_prior = true;
    //     state.imu.rotation_prior_radians = rotation;
    // }
    // if let Some(motion) = context.engine.last_motion_state() {
    //     state.imu.has_motion_state = true;
    //     state.imu.motion_state = match motion.as_str() {
    //         "Stationary" => 0,
    //         "SlowMotion" => 1,
    //         "FastMotion" => 2,
    //         _ => 0,
    //     };
    // }
    if let Some(gravity) = context.engine.gravity_estimate() {
        state.imu.has_gravity = true;
        let down = gravity.down().into_inner();
        state.imu.gravity_down = [down.x, down.y, down.z];
        state.imu.gravity_samples = context.engine.gravity_sample_count();
    }
    // state.imu.preintegration_count = context.engine.preintegration_count() as u32;

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

/// Gets feature descriptors from the most recent keyframe
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_get_descriptors(
    handle: *mut ArbitCaptureContextHandle,
    out_descriptors: *mut ArbitFeatDescriptor,
    max_descriptors: usize,
) -> usize {
    if handle.is_null() || out_descriptors.is_null() || max_descriptors == 0 {
        return 0;
    }

    let context = handle_to_context(handle);
    let descriptors = context.engine.descriptors();
    let count = descriptors.len().min(max_descriptors);
    let dest = unsafe { slice::from_raw_parts_mut(out_descriptors, count) };

    for (dst, descriptor) in dest.iter_mut().zip(descriptors.iter()) {
        let seed = &descriptor.seed;
        let bytes = descriptor.bytes();
        debug_assert!(bytes.len() <= ORB_DESCRIPTOR_LEN);

        dst.level = seed.level as u32;
        dst.seed_x = seed.px_uv.x;
        dst.seed_y = seed.px_uv.y;
        dst.score = seed.score;
        dst.angle = descriptor.angle;
        dst.data_len = bytes.len();
        dst.data = [0u8; ORB_DESCRIPTOR_LEN];
        dst.data[..bytes.len()].copy_from_slice(bytes);
    }

    count
}

/// Matches two sets of feature descriptors using Hamming distance
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_match_descriptors(
    query_descriptors: *const ArbitFeatDescriptor,
    query_count: usize,
    train_descriptors: *const ArbitFeatDescriptor,
    train_count: usize,
    out_matches: *mut ArbitMatch,
    max_matches: usize,
    cross_check: bool,
    max_distance: u32,
) -> usize {
    if query_descriptors.is_null()
        || train_descriptors.is_null()
        || out_matches.is_null()
        || query_count == 0
        || train_count == 0
        || max_matches == 0
    {
        return 0;
    }

    let query = unsafe { slice::from_raw_parts(query_descriptors, query_count) };
    let train = unsafe { slice::from_raw_parts(train_descriptors, train_count) };
    let dest = unsafe { slice::from_raw_parts_mut(out_matches, max_matches) };

    use arbit_core::track::feat_descriptor::FeatDescriptor;
    use arbit_core::track::feat_matcher::HammingFeatMatcher;
    use nalgebra::Vector2;

    // Convert FFI descriptors to internal format
    let query_descriptors: Vec<FeatDescriptor<[u8; ORB_DESCRIPTOR_LEN]>> = query
        .iter()
        .map(|d| {
            let mut data = [0u8; ORB_DESCRIPTOR_LEN];
            let len = d.data_len.min(ORB_DESCRIPTOR_LEN);
            data[..len].copy_from_slice(&d.data[..len]);
            FeatDescriptor {
                seed: FeatureSeed {
                    level: d.level as usize,
                    level_scale: 1.0,
                    px_uv: Vector2::new(d.seed_x, d.seed_y),
                    score: d.score,
                },
                angle: d.angle,
                data,
            }
        })
        .collect();

    let train_descriptors: Vec<FeatDescriptor<[u8; ORB_DESCRIPTOR_LEN]>> = train
        .iter()
        .map(|d| {
            let mut data = [0u8; ORB_DESCRIPTOR_LEN];
            let len = d.data_len.min(ORB_DESCRIPTOR_LEN);
            data[..len].copy_from_slice(&d.data[..len]);
            FeatDescriptor {
                seed: FeatureSeed {
                    level: d.level as usize,
                    level_scale: 1.0,
                    px_uv: Vector2::new(d.seed_x, d.seed_y),
                    score: d.score,
                },
                angle: d.angle,
                data,
            }
        })
        .collect();

    // Create matcher with configuration
    let matcher = HammingFeatMatcher {
        cross_check,
        max_distance: if max_distance > 0 {
            Some(max_distance)
        } else {
            None
        },
        ratio_threshold: None, // Could expose this as parameter if needed
    };

    // Perform matching
    let matches = matcher.match_feats(&query_descriptors, &train_descriptors);

    // Convert matches to FFI format
    let count = matches.len().min(max_matches);
    for (i, m) in matches.iter().take(count).enumerate() {
        dest[i] = ArbitMatch {
            query_idx: m.query_idx as u32,
            train_idx: m.train_idx as u32,
            distance: m.distance,
            query_x: query[m.query_idx].seed_x,
            query_y: query[m.query_idx].seed_y,
            train_x: train[m.train_idx].seed_x,
            train_y: train[m.train_idx].seed_y,
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
            initial_x: track.initial_px_uv.x,
            initial_y: track.initial_px_uv.y,
            refined_x: track.refined_px_uv.x,
            refined_y: track.refined_px_uv.y,
            residual: track.residual,
            iterations: track.iterations,
            status: track.outcome.into(),
            track_id: track.id.unwrap_or(0),
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
}
