use std::collections::{HashMap, HashSet};

use arbit_core::img::{build_pyramid, ImageBuffer, Pyramid, PyramidLevel};
use arbit_core::imu::{GravityEstimate, GravityEstimator};
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
use arbit_core::time::{FrameTimestamps, SystemClock, TimestampPolicy};
use arbit_core::track::{
    FeatureGridConfig, FeatureSeeder, LucasKanadeConfig, TrackObservation, TrackOutcome, Tracker,
};
use arbit_core::vo::{FrameObservation, VoLoop, VoLoopConfig, VoStatus};
use arbit_providers::{ArKitFrame, CameraSample};
use log::{debug, info, warn};
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

/// Processing engine that encapsulates the full camera/IMU pipeline.
pub struct ProcessingEngine {
    config: EngineConfig,
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
    map: WorldMap,
    pnp: PnPRansac,
    vo_loop: VoLoop,
    frame_index: u64,
    last_relocalization: Option<PnPResult>,
    last_keyframe_id: Option<u64>,
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

        Self {
            config,
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
            map: WorldMap::new(),
            pnp,
            vo_loop: VoLoop::new(VoLoopConfig::default()),
            frame_index: 0,
            last_relocalization: None,
            last_keyframe_id: None,
        }
    }

    /// Ingest a camera frame, updating the internal state. Returns a camera sample with
    /// monotonic timestamps when successful; returns `None` if the frame could not be processed.
    pub fn ingest_camera_frame(&mut self, frame: &ArKitFrame) -> Option<CameraSample> {
        let timestamps = self.timestamp_policy.ingest_capture(frame.timestamp);
        let intrinsics = CameraIntrinsics::from(frame.intrinsics.clone());
        let prev_intrinsics = self.last_intrinsics.clone();

        let sample = CameraSample {
            timestamps,
            intrinsics: intrinsics.clone(),
            pixel_format: frame.pixel_format,
            bytes_per_row: frame.bytes_per_row,
            data: frame.data.clone(),
        };

        if let Some(image) = self.extract_image_buffer(frame, &intrinsics) {
            self.process_image_buffer(
                image,
                &intrinsics,
                prev_intrinsics.as_ref(),
                frame.timestamp.as_secs_f64(),
            );
        } else {
            warn!(
                "Failed to extract image buffer for frame at {:.3}s",
                frame.timestamp.as_secs_f64()
            );
            self.pyramid_cache.clear();
            self.last_tracks.clear();
        }

        let descriptor_current = build_descriptor(
            &self.last_tracks,
            intrinsics.width,
            intrinsics.height,
            DescriptorSample::Refined,
        );
        self.handle_vo_update(frame.timestamp.as_secs_f64(), &descriptor_current);

        self.last_intrinsics = Some(intrinsics);

        Some(sample)
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
        frame: &ArKitFrame,
        intrinsics: &CameraIntrinsics,
    ) -> Option<ImageBuffer> {
        let width = intrinsics.width as usize;
        let height = intrinsics.height as usize;
        if width == 0 || height == 0 {
            return None;
        }

        let min_required = width.saturating_mul(height).saturating_mul(4);
        if frame.data.len() < min_required {
            return None;
        }

        Some(ImageBuffer::from_bgra8(
            frame.data.as_ref(),
            width,
            height,
            frame.bytes_per_row,
        ))
    }

    fn process_image_buffer(
        &mut self,
        image: ImageBuffer,
        intrinsics: &CameraIntrinsics,
        prev_intrinsics: Option<&CameraIntrinsics>,
        timestamp_seconds: f64,
    ) {
        debug!(
            "Successfully extracted image buffer: {}x{}",
            image.width(),
            image.height()
        );

        let pyramid = build_pyramid(&image, 3);
        let levels = pyramid.levels().len();
        debug!("Built pyramid with {} levels", levels);

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

        if let Some(prev_pyramid) = &self.prev_pyramid {
            let seeds = self.seeder.seed(&prev_pyramid.levels()[0]);
            debug!("Seeded {} features for tracking", seeds.len());

            let mut tracks = Vec::with_capacity(seeds.len());
            for seed in seeds.iter().take(256) {
                let observation = self.tracker.track(prev_pyramid, &pyramid, seed.position);
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

            self.last_tracks = tracks;

            if let Some(prev_intr) = prev_intrinsics {
                let descriptor_prev = build_descriptor(
                    &self.last_tracks,
                    prev_intr.width,
                    prev_intr.height,
                    DescriptorSample::Initial,
                );
                let matches_indexed =
                    build_feature_matches(&self.last_tracks, prev_intr, intrinsics);
                let matches: Vec<FeatureMatch> = matches_indexed
                    .iter()
                    .map(|(_, feature)| *feature)
                    .collect();

                debug!(
                    "Found {} feature matches for two-view initialization",
                    matches.len()
                );

                if matches.len() >= 8 {
                    debug!(
                        "Attempting two-view initialization with {} matches",
                        matches.len()
                    );

                    if let Some(two_view) = self.two_view_initializer.estimate(&matches) {
                        let prev_pose = self.current_pose.clone();
                        let scaled_two_view = two_view.scaled(self.config.baseline_scale);
                        self.try_insert_keyframe(
                            &prev_pose,
                            descriptor_prev,
                            &scaled_two_view,
                            &matches_indexed,
                        );

                        self.current_pose = update_pose(&self.current_pose, &scaled_two_view);
                        self.trajectory.push(self.current_pose.translation.vector);
                        if self.trajectory.len() > self.config.max_trajectory_points {
                            let trim = self.trajectory.len() - self.config.max_trajectory_points;
                            self.trajectory.drain(0..trim);
                        }
                        self.last_two_view = Some(scaled_two_view);
                        debug!(
                            "Two-view initialization successful, trajectory length: {}",
                            self.trajectory.len()
                        );
                    } else {
                        self.last_two_view = None;
                        debug!("Two-view initialization failed");
                    }
                } else {
                    self.last_two_view = None;
                    debug!(
                        "Insufficient matches ({}) for two-view initialization",
                        matches.len()
                    );
                }
            } else {
                debug!("No previous intrinsics available for two-view initialization");
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

    fn try_insert_keyframe(
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
        for (match_idx, landmark) in &two_view.landmarks {
            if let Some(track_idx) = track_lookup.get(*match_idx) {
                if let Some(normalized) = normalized_lookup.get(track_idx) {
                    let world_point = prev_pose.transform_point(landmark);
                    features.push((*normalized, world_point));
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

        if let Some(id) = self
            .map
            .insert_keyframe(prev_pose.clone(), descriptor, features)
        {
            info!("Committed keyframe {}", id);
            self.last_keyframe_id = Some(id);
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
    use arbit_providers::{ArKitIntrinsics, PixelFormat};
    use std::sync::Arc;
    use std::time::Duration;

    fn frame() -> ArKitFrame {
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
        let bytes_per_row = (intrinsics.width as usize) * 4;
        let buffer = vec![0u8; (intrinsics.height as usize) * bytes_per_row];
        ArKitFrame {
            timestamp: Duration::from_millis(0),
            intrinsics,
            pixel_format: PixelFormat::Bgra8,
            bytes_per_row,
            data: Arc::from(buffer),
        }
    }

    #[test]
    fn engine_ingests_frame() {
        let mut engine = ProcessingEngine::new();
        let sample = engine.ingest_camera_frame(&frame()).expect("sample");
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
