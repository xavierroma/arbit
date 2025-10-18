pub mod types;

#[cfg(feature = "debug-server")]
pub mod debug_server;

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use arbit_core::img::{build_pyramid, ImageBuffer, Pyramid, PyramidLevel};
use arbit_core::imu::{
    AccelBiasEstimator, GravityEstimate, GravityEstimator, GyroBiasEstimator, ImuPreintegrator,
    MotionDetector, MotionState, MotionStats, PreintegratedImu, PreintegrationConfig,
};
use arbit_core::init::two_view::{
    FeatureMatch, RansacParams, TwoViewInitialization, TwoViewInitializer,
};
use arbit_core::map::{
    self, build_descriptor, Anchor, DescriptorInfo, DescriptorSample, KeyframeData, MapIoError,
    WorldMap,
};
use arbit_core::math::se3::TransformSE3;
use arbit_core::math::CameraIntrinsics;
use arbit_core::relocalize::{PnPObservation, PnPRansac, PnPRansacParams, PnPResult};
use arbit_core::time::FrameTimestamps;
use arbit_core::track::{
    FeatureGridConfig, FeatureSeeder, LucasKanadeConfig, TrackObservation, TrackOutcome, Tracker,
};
use arbit_core::vo::{FrameObservation, VoLoop, VoLoopConfig, VoStatus};
use arbit_providers::CameraSample;
use log::{debug, info, warn};
use nalgebra::Point3;
use nalgebra::{Translation3, UnitQuaternion, Vector2, Vector3};

/// Processing constants grouped into a configuration structure.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub baseline_scale: f64,
    pub max_trajectory_points: usize,
    pub min_keyframe_landmarks: usize,
    pub relocalization_cell_threshold: f32,
    pub relocalization_candidates: usize,
    pub relocalization_min_inliers: usize,
    pub pnp_iterations: usize,
    pub pnp_threshold: f64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            baseline_scale: 0.05,
            max_trajectory_points: 2048,
            min_keyframe_landmarks: 12,
            relocalization_cell_threshold: 0.12,
            relocalization_candidates: 3,
            relocalization_min_inliers: 12,
            pnp_iterations: 512,
            pnp_threshold: 1e-2,
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

/// Processing engine that encapsulates the full camera/IMU pipeline.
pub struct ProcessingEngine {
    config: EngineConfig,
    tracking_state: TrackingState,
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
    map: WorldMap,
    pnp: PnPRansac,
    vo_loop: VoLoop,
    frame_index: u64,
    last_relocalization: Option<PnPResult>,
    last_keyframe_id: Option<u64>,
    // Track indexed landmarks for PnP
    tracked_landmark_ids: HashMap<usize, u64>,
    // IMU preintegration components
    preintegrator: Option<ImuPreintegrator>,
    motion_detector: Option<MotionDetector>,
    gyro_bias_estimator: Option<GyroBiasEstimator>,
    accel_bias_estimator: Option<AccelBiasEstimator>,
    last_motion_stats: Option<MotionStats>,
    last_preintegrated: Option<PreintegratedImu>,
    preintegration_count: usize,
}

impl ProcessingEngine {
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

        let mut engine = Self {
            config,
            tracking_state: TrackingState::Uninitialized,
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

        // Enable IMU preintegration by default
        engine.enable_imu_preintegration(PreintegrationConfig::default());

        engine
    }

    /// Enables IMU preintegration for improved 6DOF estimation.
    /// Call this to activate gyroscope + accelerometer fusion.
    pub fn enable_imu_preintegration(&mut self, config: PreintegrationConfig) {
        info!("Enabling IMU preintegration with config: {:?}", config);
        self.preintegrator = Some(ImuPreintegrator::new(
            config,
            Vector3::zeros(), // Initial gyro bias
            Vector3::zeros(), // Initial accel bias
        ));
        self.motion_detector = Some(MotionDetector::new(200)); // 1 second window at 200 Hz
        self.gyro_bias_estimator = Some(GyroBiasEstimator::new(2000)); // 10 seconds at 200 Hz
        self.accel_bias_estimator = Some(AccelBiasEstimator::new(2000, 9.80665));
    }

    /// Disables IMU preintegration (e.g., for visual-only mode).
    pub fn disable_imu_preintegration(&mut self) {
        info!("Disabling IMU preintegration");
        self.preintegrator = None;
        self.motion_detector = None;
        self.gyro_bias_estimator = None;
        self.accel_bias_estimator = None;
        self.last_motion_stats = None;
        self.last_preintegrated = None;
    }

    /// Returns whether IMU preintegration is currently enabled.
    pub fn has_preintegration(&self) -> bool {
        self.preintegrator.is_some()
    }

    /// Returns a reference to the active engine configuration.
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Ingests an IMU sample (gyroscope + accelerometer) for preintegration.
    /// Call this for each IMU sample between camera frames.
    ///
    /// * `timestamp` - Timestamp in seconds
    /// * `gyro` - Gyroscope reading in rad/s (x, y, z)
    /// * `accel` - Accelerometer reading in m/s² (x, y, z)
    pub fn ingest_imu_sample(
        &mut self,
        timestamp: f64,
        gyro: (f64, f64, f64),
        accel: (f64, f64, f64),
    ) {
        let gyro_vec = Vector3::new(gyro.0, gyro.1, gyro.2);
        let accel_vec = Vector3::new(accel.0, accel.1, accel.2);

        // Update gravity estimator (uses accelerometer only)
        let dt = 0.005; // Assume 200 Hz IMU rate
        if let Some(gravity_est) = self.gravity_estimator.update(accel_vec, dt) {
            self.last_gravity = Some(gravity_est.clone());

            // Update motion detector if enabled
            if let Some(ref mut detector) = self.motion_detector {
                if let Some(ref gravity) = self.last_gravity {
                    detector.set_gravity(gravity.as_vector(9.80665));
                }
                let motion_stats = detector.update(gyro_vec, accel_vec);
                self.last_motion_stats = Some(motion_stats);

                // Update bias estimators during stationary periods
                if motion_stats.state == MotionState::Stationary {
                    if let Some(ref mut gyro_bias_est) = self.gyro_bias_estimator {
                        gyro_bias_est.update(gyro_vec);
                    }
                    if let Some(ref mut accel_bias_est) = self.accel_bias_estimator {
                        accel_bias_est.update(accel_vec);
                    }
                }
            }

            // Integrate IMU if preintegrator is enabled
            if let Some(ref mut preint) = self.preintegrator {
                let timestamp_duration = std::time::Duration::from_secs_f64(timestamp);
                let gravity_vec = gravity_est.as_vector(9.80665);
                preint.integrate(timestamp_duration, gyro_vec, accel_vec, gravity_vec);
            }
        }
    }

    /// Finishes the current IMU preintegration interval and returns the result.
    /// Returns the preintegrated IMU measurements if preintegration is enabled.
    pub fn finish_imu_preintegration(&mut self) -> Option<PreintegratedImu> {
        if let Some(ref mut preint) = self.preintegrator {
            // Update bias estimates if available
            if let Some(ref gyro_bias_est) = self.gyro_bias_estimator {
                if gyro_bias_est.is_valid() {
                    let gyro_bias = gyro_bias_est.bias();
                    let accel_bias = self
                        .accel_bias_estimator
                        .as_ref()
                        .and_then(|est| {
                            if est.is_valid() {
                                Some(est.bias())
                            } else {
                                None
                            }
                        })
                        .unwrap_or_else(Vector3::zeros);
                    preint.set_biases(gyro_bias, accel_bias);
                }
            }

            let result = preint.finish();
            self.last_preintegrated = Some(result.clone());
            self.preintegration_count += 1;

            debug!(
                "IMU preintegration #{}: {} samples over {:.3}s, rotation={:.3}°",
                self.preintegration_count,
                result.sample_count,
                result.delta_time.as_secs_f64(),
                result.delta_rotation.log().norm().to_degrees()
            );

            Some(result)
        } else {
            None
        }
    }

    /// Returns the last IMU rotation prior (in radians) if available.
    pub fn last_imu_rotation_prior(&self) -> Option<f64> {
        self.last_preintegrated
            .as_ref()
            .map(|p| p.delta_rotation.log().norm())
    }

    /// Returns the last motion state as a string if available.
    pub fn last_motion_state(&self) -> Option<String> {
        self.last_motion_stats
            .as_ref()
            .map(|s| format!("{:?}", s.state))
    }

    /// Returns the preintegration count (number of intervals completed).
    pub fn preintegration_count(&self) -> usize {
        self.preintegration_count
    }

    /// Returns the current frame index.
    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    /// Create a snapshot of the current engine state.
    pub fn snapshot(&self, timestamp: f64) -> types::EngineSnapshot {
        types::EngineSnapshot::from_engine(self, timestamp, self.frame_index)
    }

    /// Ingest a camera sample, updating the internal state.
    pub fn ingest_camera_sample(&mut self, sample: &CameraSample) {
        use std::time::Instant;

        // Finish IMU preintegration for the interval leading up to this frame.
        // This provides rotation/velocity/position priors for feature tracking.
        let _preintegrated = self.finish_imu_preintegration();

        let intrinsics = sample.intrinsics.clone();
        let prev_intrinsics = self.last_intrinsics.clone();
        let timestamp_seconds = sample.timestamps.capture.as_duration().as_secs_f64();

        let extract_start = Instant::now();
        if let Some(image) = self.extract_image_buffer(sample, &intrinsics) {
            let extract_elapsed = extract_start.elapsed().as_secs_f64() * 1000.0;
            debug!("    Image extraction: {:.2}ms", extract_elapsed);

            let process_start = Instant::now();
            self.process_image_buffer(
                image,
                &intrinsics,
                prev_intrinsics.as_ref(),
                timestamp_seconds,
            );
            let process_elapsed = process_start.elapsed().as_secs_f64() * 1000.0;
            debug!("    Image processing: {:.2}ms", process_elapsed);
        } else {
            warn!(
                "Failed to extract image buffer for frame at {:.3}s",
                timestamp_seconds
            );
            self.pyramid_cache.clear();
            self.last_tracks.clear();
        }

        let descriptor_start = Instant::now();
        let descriptor_current = build_descriptor(
            &self.last_tracks,
            intrinsics.width,
            intrinsics.height,
            DescriptorSample::Refined,
        );
        let descriptor_elapsed = descriptor_start.elapsed().as_secs_f64() * 1000.0;
        debug!("    Descriptor build: {:.2}ms", descriptor_elapsed);

        let vo_start = Instant::now();
        self.handle_vo_update(timestamp_seconds, &descriptor_current);
        let vo_elapsed = vo_start.elapsed().as_secs_f64() * 1000.0;
        debug!("    VO update: {:.2}ms", vo_elapsed);

        self.last_intrinsics = Some(intrinsics);
    }

    /// Ingest an accelerometer sample.
    pub fn ingest_accelerometer(&mut self, ax: f64, ay: f64, az: f64, dt: f64) {
        let dt = if dt.is_finite() && dt > 0.0 {
            dt
        } else {
            1.0 / 120.0
        };
        let accel = Vector3::new(ax, ay, az);
        self.last_gravity = self.gravity_estimator.update(accel, dt);
    }

    /// Cached pyramid levels for inspection.
    pub fn pyramid_levels(&self) -> &[PyramidLevelCache] {
        &self.pyramid_cache
    }

    /// Last tracked feature observations.
    pub fn tracked_points(&self) -> &[TrackObservation] {
        &self.last_tracks
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

    /// Snapshot of anchor identifiers currently tracked.
    pub fn anchor_ids(&self) -> Vec<u64> {
        self.map.anchors().map(|anchor| anchor.id).collect()
    }

    /// Resolve an anchor by identifier.
    pub fn resolve_anchor(&self, anchor_id: u64) -> Option<&Anchor> {
        self.map.resolve_anchor(anchor_id)
    }

    /// Update an anchor pose.
    pub fn update_anchor(&mut self, anchor_id: u64, pose: TransformSE3) -> bool {
        self.map.update_anchor_pose(anchor_id, pose)
    }

    /// Create a new anchor and return its identifier.
    pub fn create_anchor(&mut self, pose: TransformSE3) -> u64 {
        let hint = self.last_keyframe_id;
        self.map.create_anchor(pose, hint)
    }

    /// Remove an anchor by identifier. Returns true if the anchor existed and was removed.
    pub fn remove_anchor(&mut self, anchor_id: u64) -> bool {
        self.map.remove_anchor(anchor_id)
    }

    /// Place an anchor at a screen position by raycasting into the scene.
    /// Returns the anchor ID if successful.
    ///
    /// - `normalized_u`, `normalized_v`: Normalized image coordinates in range [0, 1]
    /// - `depth`: Distance along the ray in meters (default: 1.0)
    pub fn place_anchor_at_screen_point(
        &mut self,
        normalized_u: f64,
        normalized_v: f64,
        depth: f64,
    ) -> Option<u64> {
        let intrinsics = self.last_intrinsics.as_ref()?;

        // Convert normalized [0,1] to pixel coordinates
        let pixel_x = normalized_u * intrinsics.width as f64;
        let pixel_y = normalized_v * intrinsics.height as f64;

        // Back-project to normalized camera coordinates (ray direction)
        let ndc_x = (pixel_x - intrinsics.cx) / intrinsics.fx;
        let ndc_y = (pixel_y - intrinsics.cy) / intrinsics.fy;

        // Ray direction in camera space (normalized)
        let ray_camera = Vector3::new(ndc_x, ndc_y, 1.0).normalize();

        // Transform ray from camera space to world space using current pose
        let rotation = self.current_pose.rotation.to_rotation_matrix();
        let ray_world = rotation * ray_camera;

        // Calculate anchor position: camera position + ray * depth
        let camera_position = self.current_pose.translation.vector;
        let anchor_position = camera_position + ray_world * depth;

        // Create pose at anchor position (identity rotation)
        let anchor_pose = TransformSE3::from_parts(
            Translation3::from(anchor_position),
            UnitQuaternion::identity(),
        );

        let anchor_id = self.create_anchor(anchor_pose);
        info!(
            target: "arbit_engine",
            "Placed anchor #{} at world position [{:.2}, {:.2}, {:.2}] (depth: {:.2}m)",
            anchor_id,
            anchor_position.x,
            anchor_position.y,
            anchor_position.z,
            depth
        );

        Some(anchor_id)
    }

    /// Get all anchors that are visible in the current camera frame with their projected coordinates.
    /// Returns anchors that are:
    /// - In front of the camera (positive depth)
    /// - Within the frame bounds
    pub fn get_visible_anchors(&self) -> Vec<ProjectedAnchor> {
        let Some(intrinsics) = &self.last_intrinsics else {
            return Vec::new();
        };

        self.map
            .anchors()
            .filter_map(|anchor| self.project_anchor_internal(anchor, intrinsics))
            .collect()
    }

    /// Helper to project a single anchor to screen space
    fn project_anchor_internal(
        &self,
        anchor: &Anchor,
        intrinsics: &CameraIntrinsics,
    ) -> Option<ProjectedAnchor> {
        // Extract world position from anchor pose (translation component)
        let world_position = Point3::from(anchor.pose.translation.vector);

        // Transform to camera frame
        let camera_point = self.current_pose.transform_point(&world_position);

        // Check if in front of camera
        if camera_point.z <= 0.0 {
            return None;
        }

        // Project to normalized image coordinates
        let normalized_u = camera_point.x / camera_point.z;
        let normalized_v = camera_point.y / camera_point.z;

        // Convert to pixel coordinates
        let pixel_x = (normalized_u * intrinsics.fx + intrinsics.cx) as f32;
        let pixel_y = (normalized_v * intrinsics.fy + intrinsics.cy) as f32;

        // Check if within frame bounds
        if pixel_x < 0.0
            || pixel_x >= intrinsics.width as f32
            || pixel_y < 0.0
            || pixel_y >= intrinsics.height as f32
        {
            return None;
        }

        Some(ProjectedAnchor {
            anchor: anchor.clone(),
            normalized_u,
            normalized_v,
            pixel_x,
            pixel_y,
            depth: camera_point.z,
        })
    }

    /// Serialize the current map to a binary payload.
    pub fn save_map(&self) -> Result<Vec<u8>, MapIoError> {
        self.map.to_bytes()
    }

    /// Load a map from the provided binary payload.
    pub fn load_map(&mut self, bytes: &[u8]) -> Result<(), MapIoError> {
        self.map.load_from_bytes(bytes)?;
        self.last_keyframe_id = self.map.max_keyframe_id();
        self.last_relocalization = None;
        self.vo_loop.reset();
        Ok(())
    }

    fn extract_image_buffer(
        &self,
        sample: &CameraSample,
        intrinsics: &CameraIntrinsics,
    ) -> Option<ImageBuffer> {
        let width = intrinsics.width as usize;
        let height = intrinsics.height as usize;
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

    fn process_image_buffer(
        &mut self,
        image: ImageBuffer,
        intrinsics: &CameraIntrinsics,
        prev_intrinsics: Option<&CameraIntrinsics>,
        timestamp_seconds: f64,
    ) {
        use std::time::Instant;

        debug!(
            "Successfully extracted image buffer: {}x{}",
            image.width(),
            image.height()
        );

        let pyramid_start = Instant::now();
        let pyramid = build_pyramid(&image, 3);
        let levels = pyramid.levels().len();
        let pyramid_elapsed = pyramid_start.elapsed().as_secs_f64() * 1000.0;
        debug!(
            "      Pyramid build: {:.2}ms ({} levels)",
            pyramid_elapsed, levels
        );

        let cache_start = Instant::now();
        self.pyramid_cache = pyramid
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
            .collect();
        let cache_elapsed = cache_start.elapsed().as_secs_f64() * 1000.0;
        debug!("      Pyramid cache: {:.2}ms", cache_elapsed);

        if let Some(prev_pyramid) = &self.prev_pyramid {
            let seed_start = Instant::now();
            let seeds = self.seeder.seed(&prev_pyramid.levels()[0]);
            let seed_elapsed = seed_start.elapsed().as_secs_f64() * 1000.0;
            debug!(
                "      Feature seeding: {:.2}ms ({} seeds)",
                seed_elapsed,
                seeds.len()
            );

            let track_start = Instant::now();
            let mut tracks = Vec::with_capacity(seeds.len());
            for seed in seeds.iter().take(256) {
                // Use IMU rotation prior if available to improve tracking
                // let rotation_prior = self.last_preintegrated.as_ref().map(|p| &p.delta_rotation);
                let observation = self.tracker.track_with_prior(
                    prev_pyramid,
                    &pyramid,
                    seed.position,
                    None,
                    Some(intrinsics),
                );
                tracks.push(observation);
            }
            let track_elapsed = track_start.elapsed().as_secs_f64() * 1000.0;

            let converged_tracks = tracks
                .iter()
                .filter(|t| matches!(t.outcome, TrackOutcome::Converged))
                .count();
            debug!(
                "      Feature tracking: {:.2}ms ({} tracks, {} converged)",
                track_elapsed,
                tracks.len(),
                converged_tracks
            );

            self.last_tracks = tracks.clone();

            // State machine: handle initialization vs tracking
            match self.tracking_state {
                TrackingState::Uninitialized => {
                    // First frame: just start tracking
                    self.tracking_state = TrackingState::TrackingPreInit;
                    debug!("First frame received, state -> TrackingPreInit");
                }
                TrackingState::TrackingPreInit => {
                    // Second frame onwards: try two-view initialization
                    if let Some(prev_intr) = prev_intrinsics {
                        self.handle_initialization(
                            &tracks,
                            prev_intr,
                            intrinsics,
                            timestamp_seconds,
                        );
                    }
                }
                TrackingState::Tracking => {
                    // Map initialized: use PnP for pose estimation
                    self.handle_tracking_with_pnp(&tracks, intrinsics);
                }
            }
        }

        self.prev_pyramid = Some(pyramid);

        let descriptor_current = build_descriptor(
            &self.last_tracks,
            intrinsics.width,
            intrinsics.height,
            DescriptorSample::Refined,
        );
        self.handle_vo_update(timestamp_seconds, &descriptor_current);
    }

    /// Handles two-view initialization and gravity-aligned map creation
    fn handle_initialization(
        &mut self,
        tracks: &[TrackObservation],
        prev_intrinsics: &CameraIntrinsics,
        curr_intrinsics: &CameraIntrinsics,
        _timestamp_seconds: f64,
    ) {
        let descriptor_start = Instant::now();
        let descriptor_prev = build_descriptor(
            tracks,
            prev_intrinsics.width,
            prev_intrinsics.height,
            DescriptorSample::Initial,
        );
        let descriptor_elapsed = descriptor_start.elapsed().as_secs_f64() * 1000.0;
        debug!("      Descriptor build (prev): {:.2}ms", descriptor_elapsed);

        let match_start = Instant::now();
        let matches_indexed = build_feature_matches(tracks, prev_intrinsics, curr_intrinsics);
        let matches: Vec<FeatureMatch> = matches_indexed
            .iter()
            .map(|(_, feature)| *feature)
            .collect();
        let match_elapsed = match_start.elapsed().as_secs_f64() * 1000.0;

        debug!(
            "      Feature matching: {:.2}ms ({} matches)",
            match_elapsed,
            matches.len()
        );

        if matches.len() < 8 {
            debug!(
                "Insufficient matches ({}) for two-view initialization",
                matches.len()
            );
            return;
        }

        let two_view_start = Instant::now();
        let Some(two_view) = self.two_view_initializer.estimate(&matches) else {
            let two_view_elapsed = two_view_start.elapsed().as_secs_f64() * 1000.0;
            debug!("      Two-view init: {:.2}ms (FAILED)", two_view_elapsed);
            return;
        };

        let two_view_elapsed = two_view_start.elapsed().as_secs_f64() * 1000.0;
        debug!("      Two-view init: {:.2}ms (SUCCESS)", two_view_elapsed);

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

        // Set initial pose: first camera at aligned origin
        let initial_pose = world_alignment.clone();
        self.current_pose = initial_pose.clone();

        // Insert first keyframe at aligned origin
        let keyframe_start = Instant::now();
        self.try_insert_keyframe_with_landmarks(
            &initial_pose,
            descriptor_prev,
            &scaled_two_view,
            &matches_indexed,
        );
        let keyframe_elapsed = keyframe_start.elapsed().as_secs_f64() * 1000.0;
        debug!("      Keyframe insertion: {:.2}ms", keyframe_elapsed);

        // Update pose for second frame
        self.current_pose = update_pose(&initial_pose, &scaled_two_view);
        self.trajectory.push(self.current_pose.translation.vector);

        // Transition to tracking mode
        self.tracking_state = TrackingState::Tracking;
        self.last_two_view = Some(scaled_two_view);

        info!(
            "Map initialized with {} landmarks, gravity-aligned world frame",
            landmark_count
        );
    }

    /// Handles pose estimation using PnP with tracked features
    fn handle_tracking_with_pnp(
        &mut self,
        tracks: &[TrackObservation],
        intrinsics: &CameraIntrinsics,
    ) {
        // Build observations from tracked features that have known landmark IDs
        let mut observations = Vec::new();

        for (track_idx, track) in tracks.iter().enumerate() {
            if !matches!(track.outcome, TrackOutcome::Converged) {
                continue;
            }

            // Check if we have a landmark ID for this track
            if let Some(&landmark_id) = self.tracked_landmark_ids.get(&track_idx) {
                // Get the 3D position from the map
                if let Some(landmark) = self.map.landmark(landmark_id) {
                    let normalized = normalize_pixel(track.refined, intrinsics);
                    observations.push(PnPObservation {
                        world_point: landmark.position,
                        normalized_image: normalized,
                    });
                }
            }
        }

        debug!(
            "      PnP tracking: {} observations from {} tracks",
            observations.len(),
            tracks.len()
        );

        if observations.len() < self.config.relocalization_min_inliers {
            warn!(
                "Insufficient PnP observations ({} < {}), attempting relocalization",
                observations.len(),
                self.config.relocalization_min_inliers
            );
            // Fall back to relocalization
            self.tracking_state = TrackingState::TrackingPreInit;
            self.tracked_landmark_ids.clear();
            return;
        }

        let pnp_start = Instant::now();
        let Some(result) = self.pnp.estimate(&observations) else {
            let pnp_elapsed = pnp_start.elapsed().as_secs_f64() * 1000.0;
            warn!(
                "      PnP estimation: {:.2}ms (FAILED), falling back",
                pnp_elapsed
            );
            self.tracking_state = TrackingState::TrackingPreInit;
            self.tracked_landmark_ids.clear();
            return;
        };

        let pnp_elapsed = pnp_start.elapsed().as_secs_f64() * 1000.0;
        debug!(
            "      PnP estimation: {:.2}ms ({} inliers, error: {:.4})",
            pnp_elapsed,
            result.inliers.len(),
            result.average_reprojection_error
        );

        // Update current pose
        self.current_pose = result.pose.clone();
        self.trajectory.push(self.current_pose.translation.vector);
        if self.trajectory.len() > self.config.max_trajectory_points {
            let trim = self.trajectory.len() - self.config.max_trajectory_points;
            self.trajectory.drain(0..trim);
        }
    }

    /// Inserts keyframe and maintains landmark-to-track mapping for PnP tracking
    fn try_insert_keyframe_with_landmarks(
        &mut self,
        prev_pose: &TransformSE3,
        descriptor_info: DescriptorInfo,
        two_view: &TwoViewInitialization,
        matches_indexed: &[(usize, FeatureMatch)],
    ) {
        if !self.map.should_insert_keyframe(prev_pose) {
            debug!("Keyframe gating rejected pose insertion");
            return;
        }

        let DescriptorInfo {
            descriptor,
            normalized_points,
            converged_tracks,
            ..
        } = descriptor_info;

        if converged_tracks < self.config.min_keyframe_landmarks {
            debug!(
                "Insufficient converged tracks for keyframe ({} < {})",
                converged_tracks, self.config.min_keyframe_landmarks
            );
            return;
        }

        if two_view.landmarks.len() < self.config.min_keyframe_landmarks {
            debug!(
                "Two-view result provided {} landmarks (need {})",
                two_view.landmarks.len(),
                self.config.min_keyframe_landmarks
            );
            return;
        }

        let normalized_lookup: HashMap<usize, Vector2<f32>> =
            normalized_points.into_iter().collect();
        let track_lookup: Vec<usize> = matches_indexed.iter().map(|(idx, _)| *idx).collect();

        let mut features = Vec::new();
        let mut track_to_match: HashMap<usize, usize> = HashMap::new();

        for (match_idx, landmark) in &two_view.landmarks {
            if let Some(track_idx) = track_lookup.get(*match_idx) {
                if let Some(normalized) = normalized_lookup.get(track_idx) {
                    let world_point = prev_pose.transform_point(landmark);
                    features.push((*normalized, world_point));
                    track_to_match.insert(*track_idx, *match_idx);
                }
            }
        }

        if features.len() < self.config.min_keyframe_landmarks {
            debug!(
                "Matched landmarks below threshold ({} < {})",
                features.len(),
                self.config.min_keyframe_landmarks
            );
            return;
        }

        // Get the starting landmark ID before insertion
        let starting_landmark_id = self.map.landmark_count() as u64;

        if let Some(_keyframe_id) =
            self.map
                .insert_keyframe(prev_pose.clone(), descriptor, features.clone())
        {
            info!(
                "Committed keyframe {} with {} landmarks",
                _keyframe_id,
                features.len()
            );
            self.last_keyframe_id = Some(_keyframe_id);

            // Update landmark tracking: map track indices to landmark IDs
            self.tracked_landmark_ids.clear();
            for (feature_idx, (track_idx, _match_idx)) in track_to_match.iter().enumerate() {
                let landmark_id = starting_landmark_id + feature_idx as u64;
                self.tracked_landmark_ids.insert(*track_idx, landmark_id);
            }

            debug!(
                "Initialized {} tracked landmarks for PnP",
                self.tracked_landmark_ids.len()
            );
        }
    }

    fn handle_vo_update(&mut self, timestamp_seconds: f64, descriptor_info: &DescriptorInfo) {
        if descriptor_info.total_tracks == 0 {
            self.frame_index = self.frame_index.saturating_add(1);
            return;
        }

        let observation = FrameObservation {
            frame_index: self.frame_index,
            timestamp_seconds,
            pose: self.current_pose.clone(),
            track_count: descriptor_info.total_tracks,
            inlier_ratio: descriptor_info.inlier_ratio(),
            forward_backward_error: descriptor_info.average_residual,
        };

        let status = self.vo_loop.process(observation);
        self.frame_index = self.frame_index.saturating_add(1);

        if matches!(status, VoStatus::Lost) {
            debug!("VO reported loss; attempting relocalization");
            if let Some(result) = self.attempt_relocalization(descriptor_info) {
                self.current_pose = result.pose.clone();
                self.last_two_view = None;
                self.last_relocalization = Some(result.clone());
                self.trajectory.push(self.current_pose.translation.vector);
                if self.trajectory.len() > self.config.max_trajectory_points {
                    let trim = self.trajectory.len() - self.config.max_trajectory_points;
                    self.trajectory.drain(0..trim);
                }
                self.vo_loop.reset();
                info!(
                    "Relocalized with {} inliers (avg reprojection {:.4})",
                    result.inliers.len(),
                    result.average_reprojection_error
                );
            }
        }
    }

    fn attempt_relocalization(&self, descriptor_info: &DescriptorInfo) -> Option<PnPResult> {
        if descriptor_info.normalized_points.is_empty() || self.map.is_empty() {
            return None;
        }

        let descriptor = descriptor_info.descriptor.clone();
        let candidates = self
            .map
            .query(&descriptor, self.config.relocalization_candidates);
        if candidates.is_empty() {
            return None;
        }

        for candidate in candidates {
            let observations =
                self.build_pnp_observations(candidate, &descriptor_info.normalized_points);
            debug!(
                "Keyframe {} yielded {} candidate observations",
                candidate.id,
                observations.len()
            );
            if observations.len() < self.config.relocalization_min_inliers {
                continue;
            }
            if let Some(result) = self.pnp.estimate(&observations) {
                return Some(result);
            }
        }

        None
    }

    fn build_pnp_observations(
        &self,
        keyframe: &KeyframeData,
        points: &[(usize, Vector2<f32>)],
    ) -> Vec<PnPObservation> {
        let mut used = HashSet::new();
        let mut observations = Vec::new();

        for (_, normalized) in points {
            let cell = map::cell_for_normalized(normalized);
            let mut best: Option<(f32, &map::KeyframeFeature)> = None;
            for feature in keyframe.features_in_cell(cell) {
                if used.contains(&feature.landmark_id) {
                    continue;
                }
                let dist = (feature.normalized - *normalized).norm();
                if dist > self.config.relocalization_cell_threshold {
                    continue;
                }
                match best {
                    Some((current, _)) if current <= dist => {}
                    _ => best = Some((dist, feature)),
                }
            }

            if let Some((_, feature)) = best {
                used.insert(feature.landmark_id);
                observations.push(PnPObservation {
                    world_point: feature.world,
                    normalized_image: Vector2::new(normalized.x as f64, normalized.y as f64),
                });
            }
        }

        observations
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

fn build_feature_matches(
    observations: &[TrackObservation],
    prev_intrinsics: &CameraIntrinsics,
    curr_intrinsics: &CameraIntrinsics,
) -> Vec<(usize, FeatureMatch)> {
    observations
        .iter()
        .enumerate()
        .filter_map(|(index, obs)| {
            if !matches!(obs.outcome, TrackOutcome::Converged) {
                return None;
            }
            Some((
                index,
                FeatureMatch {
                    normalized_a: normalize_pixel(obs.initial, prev_intrinsics),
                    normalized_b: normalize_pixel(obs.refined, curr_intrinsics),
                },
            ))
        })
        .collect()
}

fn normalize_pixel(pixel: Vector2<f32>, intrinsics: &CameraIntrinsics) -> Vector2<f64> {
    Vector2::new(
        (pixel.x as f64 - intrinsics.cx) / intrinsics.fx,
        (pixel.y as f64 - intrinsics.cy) / intrinsics.fy,
    )
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

impl Default for ProcessingEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience accessor for trajectory timestamps.
pub fn timestamps_to_seconds(timestamps: &FrameTimestamps) -> (f64, f64, f64) {
    (
        timestamps.capture.as_duration().as_secs_f64(),
        timestamps.pipeline.as_duration().as_secs_f64(),
        timestamps.latency.as_secs_f64(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use arbit_providers::{ArKitFrame, ArKitIntrinsics, IosCameraProvider, PixelFormat};
    use std::sync::Arc;
    use std::time::Duration;

    fn sample_camera_sample() -> CameraSample {
        // Use IosCameraProvider to properly create a CameraSample with valid timestamps
        let mut provider = IosCameraProvider::new();
        let intrinsics = ArKitIntrinsics {
            fx: 800.0,
            fy: 800.0,
            cx: 320.0,
            cy: 240.0,
            skew: 0.0,
            width: 640,
            height: 480,
            distortion: None,
        };
        let bytes_per_row = 640 * 4;
        let buffer = vec![0u8; 480 * bytes_per_row];

        let frame = ArKitFrame {
            timestamp: Duration::from_millis(0),
            intrinsics,
            pixel_format: PixelFormat::Bgra8,
            bytes_per_row,
            data: Arc::from(buffer),
        };

        provider.ingest_frame(frame)
    }

    #[test]
    fn engine_ingests_sample() {
        let mut engine = ProcessingEngine::new();
        let sample = sample_camera_sample();
        engine.ingest_camera_sample(&sample);
        assert_eq!(sample.intrinsics.width, 640);
        assert!(!engine.pyramid_levels().is_empty());
        assert!(engine.trajectory().len() >= 1);
    }

    #[test]
    fn save_and_load_map_round_trip() {
        let mut engine = ProcessingEngine::new();
        engine.create_anchor(TransformSE3::identity());
        let data = engine.save_map().expect("save");
        assert!(engine.load_map(&data).is_ok());
    }
}
