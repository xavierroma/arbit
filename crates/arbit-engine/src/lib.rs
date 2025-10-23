pub mod types;

#[cfg(feature = "debug-server")]
pub mod debug_server;

mod track_manager;

use std::collections::{HashMap, HashSet, VecDeque};
use std::error::Error;
use std::sync::{Arc, Mutex};

use arbit_core::img::{build_pyramid, ImageBuffer, Pyramid, PyramidLevel};
use arbit_core::imu::{
    AccelBiasEstimator, GravityEstimate, GravityEstimator, GyroBiasEstimator, ImuPreintegrator,
    MotionDetector, MotionState, MotionStats, PreintegratedImu, PreintegrationConfig,
};
use arbit_core::init::two_view::{
    FeatureMatch, TwoViewInitialization, TwoViewInitializationParams, TwoViewInitializer,
};
use arbit_core::map::{self, Anchor, KeyframeData, MapIoError, WorldMap};
use arbit_core::math::se3::TransformSE3;
use arbit_core::math::CameraIntrinsics;
use arbit_core::relocalize::{PnPObservation, PnPRansac, PnPRansacParams, PnPResult};
use arbit_core::time::FrameTimestamps;
use arbit_core::track::{
    DescriptorBuffer, FastSeeder, FastSeederConfig, FeatDescriptor, FeatDescriptorExtractor,
    FeatureSeederTrait, HammingFeatMatcher, LucasKanadeConfig, OrbDescriptor, TrackObservation,
    TrackOutcome, Tracker,
};
use arbit_core::vo::{FrameObservation, VoLoop, VoLoopConfig, VoStatus};
use arbit_providers::CameraSample;
use log::{debug, info, warn};
use nalgebra::{Matrix3x4, Point3, Translation3, UnitQuaternion, Vector2, Vector3};
use tracing::debug_span;

use crate::track_manager::{drop_near_live, TrackConfig, TrackManager};

/// Processing constants grouped into a configuration structure.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub pyramid_octaves: usize,
    pub baseline_scale: f64,
    pub max_trajectory_points: usize,
    pub min_keyframe_landmarks: usize,
    pub relocalization_cell_threshold: f32,
    pub relocalization_candidates: usize,
    pub relocalization_min_inliers: usize,
    pub pnp_iterations: usize,
    pub pnp_threshold: f64,

    pub frame_window_size: usize,
    pub two_view_max_lookback: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            pyramid_octaves: 4,
            baseline_scale: 0.05,
            max_trajectory_points: 2048,
            min_keyframe_landmarks: 12,
            relocalization_cell_threshold: 0.12,
            relocalization_candidates: 3,
            relocalization_min_inliers: 12,
            pnp_iterations: 512,
            pnp_threshold: 1e-2,
            frame_window_size: 10,
            two_view_max_lookback: 10,
        }
    }
}

/// Cached luma pyramid level exposed to higher layers (FFI/UI).
#[derive(Clone)]
pub struct PyramidLevelCache {
    pub octave: u32,
    pub scale: f32,
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: usize,
    pub pixels: Vec<u8>,
}

/// Tracking state distinguishing between initialization and normal tracking phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrackingState {
    /// Waiting for first frame and feature initialization
    Uninitialized,
    /// First frame received, tracking features but no map yet
    TrackingPreInit,
    /// Map initialized, using PnP for pose estimation
    Tracking,
}

/// An anchor projected into the current camera frame with screen coordinates.
#[derive(Debug, Clone)]
pub struct ProjectedAnchor {
    /// The underlying anchor data
    pub anchor: Anchor,
    /// Normalized image coordinates (u, v) in range [0, 1]
    pub normalized_u: f64,
    pub normalized_v: f64,
    /// Pixel coordinates in the current frame
    pub pixel_x: f32,
    pub pixel_y: f32,
    /// Depth from camera (distance along optical axis)
    pub depth: f64,
}

/// A landmark projected into the current camera frame for debugging visualization.
#[derive(Debug, Clone)]
pub struct ProjectedLandmark {
    pub landmark_id: u64,
    pub world_position: Point3<f64>,
    pub normalized_u: f64,
    pub normalized_v: f64,
    pub pixel_x: f32,
    pub pixel_y: f32,
    pub depth: f64,
}

/// Debug snapshot of the map state for visualization.
#[derive(Debug, Clone)]
pub struct MapDebugSnapshot {
    pub camera_position: Vector3<f64>,
    pub camera_rotation: [f64; 9], // 3x3 rotation matrix (row-major)
    pub landmark_count: usize,
    pub keyframe_count: usize,
    pub anchor_count: usize,
}

#[derive(Debug, Clone)]
pub struct FrameData {
    // Identification
    pub frame_index: u64,
    pub timestamp_seconds: f64,

    // Visual data
    pub image: ImageBuffer,
    pub pyramid: Pyramid,
    pub intrinsics: CameraIntrinsics,

    // Tracking results
    pub tracks: Vec<TrackObservation>,

    // Pose estimation (if available)
    pub pose: Option<TransformSE3>,
    pub gravity: Option<GravityEstimate>,

    // Optional: IMU preintegration from previous frame
    pub preintegrated_imu: Option<PreintegratedImu>,
}

pub struct KeyFrameData<D: DescriptorBuffer> {
    pub frame_data: FrameData,
    pub descriptors: Vec<FeatDescriptor<D>>,
}
/// Processing engine that encapsulates the full camera/IMU pipeline.
pub struct ProcessingEngine<S: FeatureSeederTrait, D: FeatDescriptorExtractor> {
    config: EngineConfig,
    tracking_state: TrackingState,

    frame_window: VecDeque<FrameData>,
    keyframe_window: VecDeque<KeyFrameData<D::Storage>>,
    // last_intrinsics: Option<CameraIntrinsics>,
    // prev_pyramid: Option<Pyramid>,
    // pyramid_cache: Vec<PyramidLevelCache>,
    // last_tracks: Vec<TrackObservation>,
    last_two_view: Option<TwoViewInitialization>,
    trajectory: Vec<Vector3<f64>>,
    current_pose: TransformSE3,
    last_gravity: Option<GravityEstimate>,
    map: WorldMap,
    pnp: PnPRansac,
    frame_index: u64,
    last_relocalization: Option<PnPResult>,
    last_keyframe_id: Option<u64>,
    // Track indexed landmarks for PnP
    tracked_landmark_ids: HashMap<u64, u64>,
    // IMU preintegration components
    last_motion_stats: Option<MotionStats>,
    last_preintegrated: Option<PreintegratedImu>,
    preintegration_count: usize,

    feat_detector: S,
    feat_descriptor: D,
    feat_matcher: HammingFeatMatcher,
    tracker: Arc<Tracker>,
    track_manager: Mutex<TrackManager<Tracker>>,
    two_view_initializer: TwoViewInitializer,
    gravity_estimator: GravityEstimator,
    vo_loop: VoLoop,
    preintegrator: Option<ImuPreintegrator>,
    motion_detector: Option<MotionDetector>,
    gyro_bias_estimator: Option<GyroBiasEstimator>,
    accel_bias_estimator: Option<AccelBiasEstimator>,
}

impl ProcessingEngine<FastSeeder, OrbDescriptor> {
    // ========================================================================
    // CONSTRUCTION & CONFIGURATION
    // ========================================================================

    /// Construct a new engine using default configuration values.
    pub fn new() -> Self {
        Self::with_config(EngineConfig::default())
    }

    /// Construct a new engine with a custom configuration.
    pub fn with_config(config: EngineConfig) -> Self {
        let pnp = PnPRansac::new(PnPRansacParams {
            iterations: config.pnp_iterations,
            threshold: config.pnp_threshold,
            min_inliers: config.relocalization_min_inliers,
        });

        let tracker = Arc::new(Tracker::new(LucasKanadeConfig::default()));
        let track_manager = TrackManager::new(tracker.clone(), TrackConfig::default());

        let engine = Self {
            config,
            tracking_state: TrackingState::Uninitialized,
            frame_window: VecDeque::new(),
            keyframe_window: VecDeque::new(),
            feat_detector: FastSeeder::new(FastSeederConfig::default()),
            feat_descriptor: OrbDescriptor::new(),
            feat_matcher: HammingFeatMatcher::default(),
            tracker,
            track_manager: Mutex::new(track_manager),
            last_two_view: None,
            two_view_initializer: TwoViewInitializer::new(TwoViewInitializationParams::default()),
            trajectory: vec![Vector3::new(0.0, 0.0, 0.0)],
            current_pose: TransformSE3::identity(),
            last_gravity: None,
            gravity_estimator: GravityEstimator::new(0.75),
            map: WorldMap::new(),
            pnp,
            vo_loop: VoLoop::new(VoLoopConfig::default()),
            frame_index: 0,
            last_relocalization: None,
            last_keyframe_id: None,
            tracked_landmark_ids: HashMap::new(),
            preintegrator: None,
            motion_detector: None,
            gyro_bias_estimator: None,
            accel_bias_estimator: None,
            last_motion_stats: None,
            last_preintegrated: None,
            preintegration_count: 0,
        };

        engine
    }
}

impl<S: FeatureSeederTrait, D: FeatDescriptorExtractor> ProcessingEngine<S, D> {
    /// Returns a reference to the active engine configuration.
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Returns the current frame index.
    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    // ========================================================================
    // MAIN CAMERA PROCESSING PIPELINE
    // ========================================================================

    /// Create a snapshot of the current engine state.
    pub fn snapshot(&self, timestamp: f64) -> types::EngineSnapshot {
        types::EngineSnapshot::from_engine(self, timestamp, self.frame_index)
    }

    /// Ingest a camera sample, updating the internal state.
    ///
    /// Main processing pipeline:
    /// 1. IMU preintegration
    /// 2. Image extraction
    /// 3. Pyramid construction
    /// 4. Feature seeding
    /// 5. Feature tracking
    /// 6. Pose estimation (initialization or PnP)
    pub fn ingest_camera_sample(&mut self, sample: &CameraSample) {
        // Step 2: Extract image buffer
        let image = match self.extract_image_buffer(sample) {
            Some(image) => image,
            None => {
                warn!(
                    "Failed to extract image buffer for frame at {:.3}s",
                    sample.timestamps.capture.as_duration().as_secs_f64()
                );
                return;
            }
        };
        let intrinsics = sample.intrinsics.clone();
        // Step 3: Build pyramid
        let pyramid = self.step_build_pyramid(&image);

        let mut frame_data = FrameData {
            frame_index: self.frame_index,
            timestamp_seconds: sample.timestamps.capture.as_duration().as_secs_f64(),
            image,
            pyramid,
            intrinsics,
            tracks: Vec::new(),
            pose: None,
            gravity: None,
            preintegrated_imu: None,
        };

        let Some(prev_frame) = self.current_frame() else {
            warn!("No previous frame found");
            self.frame_window.push_front(frame_data);
            self.frame_index += 1;
            return;
        };

        // Step 4 & 5: Seed and track features
        let tracks = self.step_seed_and_track_features(
            &prev_frame.pyramid,
            &frame_data.pyramid,
            &frame_data.intrinsics,
            prev_frame.frame_index,
            frame_data.frame_index,
        );
        frame_data.tracks = tracks;

        let is_key_frame = self.frame_index % 60 == 0;
        if is_key_frame {
            let _span = debug_span!("feat_descriptor").entered();
            let curr_frame = frame_data.clone();
            let seeds = self.feat_detector.seed(&curr_frame.pyramid);
            let descriptors = self.feat_descriptor.describe(&frame_data.pyramid, &seeds);
            let keyframe_data = KeyFrameData {
                frame_data: curr_frame,
                descriptors,
            };
            self.keyframe_window.push_front(keyframe_data);
        }

        self.frame_window.push_front(frame_data);
        if self.frame_window.len() > self.config.frame_window_size {
            self.frame_window.pop_back();
        }
        self.frame_index += 1;

        // Step 6: Estimate pose (initialization or tracking)
        match self.tracking_state {
            TrackingState::Uninitialized => {
                self.tracking_state = TrackingState::TrackingPreInit;
            }
            TrackingState::TrackingPreInit => {
                if !is_key_frame {
                    return;
                }
                if let Some(two_view) = self.run_two_view_initialization() {
                    self.last_two_view = Some(two_view);
                    self.tracking_state = TrackingState::Tracking;
                    //TODO: Insert keyframe into map
                }
            }
            TrackingState::Tracking => {
                // TODO: Implement PnP tracking
            }
        }
    }

    // ========================================================================
    // PUBLIC ACCESSORS: State Inspection
    // ========================================================================

    fn current_frame(&self) -> Option<&FrameData> {
        self.frame_window.front()
    }

    /// Cached pyramid levels for inspection.
    pub fn pyramid_levels(&self) -> Vec<PyramidLevelCache> {
        self.current_frame()
            .map(|frame| {
                frame
                    .pyramid
                    .levels()
                    .iter()
                    .map(|level| PyramidLevelCache {
                        octave: level.octave as u32,
                        scale: level.scale,
                        width: level.image.width() as u32,
                        height: level.image.height() as u32,
                        bytes_per_row: level.image.width(),
                        pixels: self.encode_luma(level),
                    })
                    .collect::<Vec<PyramidLevelCache>>()
            })
            .unwrap_or_default()
    }

    /// Last tracked feature observations.
    pub fn tracked_points(&self) -> &[TrackObservation] {
        self.current_frame()
            .map(|f| f.tracks.as_slice())
            .unwrap_or(&[])
    }

    pub fn descriptors(&self) -> &[FeatDescriptor<D::Storage>] {
        self.keyframe_window
            .front()
            .map(|kf| kf.descriptors.as_slice())
            .unwrap_or(&[])
    }

    /// Latest two-view initialization summary, if available.
    pub fn latest_two_view(&self) -> Option<&TwoViewInitialization> {
        self.last_two_view.as_ref()
    }

    /// Current estimated trajectory.
    pub fn trajectory(&self) -> &[Vector3<f64>] {
        &self.trajectory
    }

    /// Latest gravity estimate from IMU fusion.
    pub fn gravity_estimate(&self) -> Option<&GravityEstimate> {
        self.last_gravity.as_ref()
    }

    /// Number of accelerometer samples ingested by the gravity estimator.
    pub fn gravity_sample_count(&self) -> u32 {
        self.gravity_estimator.sample_count() as u32
    }

    /// Latest relocalization result, if any.
    pub fn last_relocalization(&self) -> Option<&PnPResult> {
        self.last_relocalization.as_ref()
    }

    /// Aggregate map statistics (keyframes, landmarks, anchors).
    pub fn map_stats(&self) -> (u64, u64, u64) {
        (
            self.map.keyframe_count() as u64,
            self.map.landmark_count() as u64,
            self.map.anchor_count() as u64,
        )
    }

    /// Current estimated camera pose.
    pub fn current_pose(&self) -> &TransformSE3 {
        &self.current_pose
    }

    // ========================================================================
    // INTERNAL HELPERS: Image Processing
    // ========================================================================

    fn extract_image_buffer(&self, sample: &CameraSample) -> Option<ImageBuffer> {
        let width = sample.intrinsics.width as usize;
        let height = sample.intrinsics.height as usize;
        if width == 0 || height == 0 {
            return None;
        }

        let min_required = width.saturating_mul(height).saturating_mul(4);
        if sample.data.len() < min_required {
            return None;
        }

        Some(ImageBuffer::from_bgra8(
            sample.data.as_ref(),
            width,
            height,
            sample.bytes_per_row,
        ))
    }

    /// Step 3: Build image pyramid for multi-scale feature tracking.
    fn step_build_pyramid(&self, image: &ImageBuffer) -> Pyramid {
        let _span = debug_span!("step_build_pyramid").entered();
        let pyramid = build_pyramid(&image, self.config.pyramid_octaves);
        let levels = pyramid.levels().len();

        debug!("      [3] Pyramid: ({} levels)", levels);

        pyramid
    }

    /// Step 4 & 5: Advance persistent tracks and promote new seeds.
    fn step_seed_and_track_features(
        &self,
        prev_pyramid: &Pyramid,
        curr_pyramid: &Pyramid,
        intrinsics: &CameraIntrinsics,
        frame_prev: u64,
        frame_curr: u64,
    ) -> Vec<TrackObservation> {
        let mut manager = self.track_manager.lock().expect("track manager poisoned");

        // Step 4a: advance existing live tracks
        let (stats, mut advanced) = manager.advance_alive(
            prev_pyramid,
            curr_pyramid,
            intrinsics,
            frame_prev,
            frame_curr,
        );
        debug!(
            "      [4] Tracking: advanced={}, killed={} (no_converge={}, fb_fail={}, res_fail={})",
            stats.advanced, stats.killed, stats.no_converge, stats.fb_fail, stats.res_fail
        );

        if !manager.need_more_features() {
            debug!("      [4] No more features needed");
            return advanced;
        }

        debug!("      [4] Detecting new candidates");

        // Step 4b: detect new candidates on the previous frame
        let mut seeds = {
            let _span = debug_span!("seed_features").entered();
            let raw = self.feat_detector.seed(prev_pyramid);
            debug!("      [4] Seeded ({} seeds)", raw.len());
            raw
        };

        let live_prev = manager.live_points_at(frame_prev);
        seeds = drop_near_live(seeds, live_prev, manager.config.r_detect);
        debug!("      [4] Dropped near live: remaining {}", seeds.len());

        // Step 5: track candidates forward and evaluate quality
        let mut candidates: Vec<TrackObservation> = Vec::with_capacity(seeds.len());
        for seed in seeds.iter() {
            let mut observation = self.tracker.track_with_prior(
                prev_pyramid,
                curr_pyramid,
                seed.position,
                None,
                Some(intrinsics),
            );

            if matches!(observation.outcome, TrackOutcome::Converged) {
                let backward = self.tracker.track_with_prior(
                    curr_pyramid,
                    prev_pyramid,
                    observation.refined,
                    None,
                    Some(intrinsics),
                );
                observation.fb_err = (backward.refined - seed.position).norm();
            } else {
                observation.fb_err = f32::MAX;
            }

            observation.score = seed.score;
            observation.id = None;
            candidates.push(observation);
        }

        manager.promote(&mut candidates, frame_prev, frame_curr);

        advanced.extend(candidates.into_iter().filter(|c| c.id.is_some()));
        advanced
    }

    // ========================================================================
    // POSE ESTIMATION: Two-View Initialization & PnP Tracking
    // ========================================================================

    /// Run two-view initialization to bootstrap the map.
    /// Creates the initial map with gravity-aligned coordinate frame.
    fn run_two_view_initialization(&mut self) -> Option<TwoViewInitialization> {
        let _span = debug_span!("two_view_initialization").entered();

        // Extract and clone data from frame window to avoid borrowing issues
        let Some(curr_frame) = self.keyframe_window.front() else {
            debug!("No current frame available for two-view initialization");
            return None;
        };

        // Try to find a good frame pair for initialization
        let mut two_view_result: Option<(TwoViewInitialization, CameraIntrinsics)> = None;

        for frame_i in 1..self.keyframe_window.len() {
            let Some(older_keyframe) = self.keyframe_window.get(frame_i) else {
                debug!("No older frame available for two-view initialization");
                return None;
            };
            let prev_intrinsics = &older_keyframe.frame_data.intrinsics;

            // Track features from older frame to current frame
            let matches_res = self
                .feat_matcher
                .match_feats(&older_keyframe.descriptors, &curr_frame.descriptors);

            let matches: Vec<FeatureMatch> = matches_res
                .iter()
                .map(|m| FeatureMatch {
                    normalized_a: older_keyframe.descriptors[m.query_idx]
                        .seed
                        .position
                        .cast::<f64>(),
                    normalized_b: curr_frame.descriptors[m.train_idx]
                        .seed
                        .position
                        .cast::<f64>(),
                })
                .collect();

            if matches.len() < 8 {
                debug!(
                    "Insufficient matches ({}) for two-view initialization",
                    matches.len()
                );
                continue;
            }

            // Estimate essential matrix and decompose into R, t
            let two_view = {
                let _two_view_span = debug_span!("two_view_estimation").entered();
                let result = self.two_view_initializer.estimate(&matches);
                if let Some(ref tv) = result {
                    info!(target: "arbit_engine", "✅ Two-view SUCCESS: {} inliers, {} landmarks", 
                          tv.inliers.len(), tv.landmarks.len());
                }
                result
            };

            if let Some(two_view) = two_view {
                // Success! Store the results and break from the loop
                two_view_result = Some((two_view.clone(), prev_intrinsics.clone()));
                break;
            } else {
                warn!(target: "arbit_engine", "⚠️  Two-view FAILED - trying next frame pair");
                continue;
            }
        }

        // Check if we found a valid initialization
        let Some((two_view, prev_intrinsics)) = two_view_result else {
            debug!("No valid frame pair found for two-view initialization");
            return None;
        };

        // Apply gravity alignment if available
        let world_alignment = if let Some(gravity) = &self.last_gravity {
            align_world_to_gravity(gravity)
        } else {
            warn!(target: "arbit_engine", "No gravity estimate available for initialization, using identity alignment");
            TransformSE3::identity()
        };

        // Scale and apply world alignment
        let scaled_two_view = two_view.scaled(self.config.baseline_scale);
        let landmark_count = scaled_two_view.landmarks.len();

        let initial_pose = world_alignment.inverse();
        self.current_pose = initial_pose.clone();

        info!(target: "arbit_engine", "   Initial pose set to T_cam_world (inverted gravity alignment)");

        // Update pose for second frame
        self.current_pose = update_pose(&initial_pose, &scaled_two_view);
        self.trajectory.push(self.current_pose.translation.vector);

        info!(
            target: "arbit_engine",
            "✅ Map initialized with {} landmarks, gravity-aligned world frame. STATE: TrackingPreInit → Tracking",
            landmark_count
        );

        Some(two_view)
    }

    fn encode_luma(&self, level: &PyramidLevel) -> Vec<u8> {
        level
            .image
            .data()
            .iter()
            .map(|value| value.clamp(0.0, 255.0) as u8)
            .collect()
    }
}

fn update_pose(current: &TransformSE3, two_view: &TwoViewInitialization) -> TransformSE3 {
    let rotation = UnitQuaternion::from_rotation_matrix(&two_view.rotation);
    let translation = Translation3::from(two_view.translation);
    let delta = TransformSE3::from_parts(translation, rotation);
    current * delta
}

/// Aligns the world coordinate frame so that the Y-axis points up (opposite to gravity).
/// Returns a transform that rotates the camera frame to align with gravity.
fn align_world_to_gravity(gravity: &GravityEstimate) -> TransformSE3 {
    use nalgebra::Rotation3;

    // Gravity direction in device frame (down)
    let g_device = gravity.down().into_inner();

    // We want world Y-axis to point up (opposite to gravity)
    let world_up = Vector3::new(0.0, 1.0, 0.0);

    // Compute rotation axis: cross product of gravity (down) and world up
    let axis = g_device.cross(&-world_up);
    let axis_norm = axis.norm();

    // If gravity is already aligned (or anti-aligned) with Y, return identity or 180° flip
    if axis_norm < 1e-6 {
        let dot = g_device.dot(&-world_up);
        if dot > 0.0 {
            // Gravity points down, world Y points up: already aligned
            return TransformSE3::identity();
        } else {
            // Gravity points up: rotate 180° around X
            let rotation = Rotation3::from_axis_angle(&Vector3::x_axis(), std::f64::consts::PI);
            return TransformSE3::from_parts(Translation3::identity(), rotation.into());
        }
    }

    // Rotation angle: angle between gravity and -world_up
    let angle = g_device.angle(&-world_up);
    let axis_normalized = axis / axis_norm;

    // Build rotation
    let rotation =
        Rotation3::from_axis_angle(&nalgebra::Unit::new_unchecked(axis_normalized), angle);

    TransformSE3::from_parts(Translation3::identity(), rotation.into())
}

impl Default for ProcessingEngine<FastSeeder, OrbDescriptor> {
    fn default() -> Self {
        Self::new()
    }
}
