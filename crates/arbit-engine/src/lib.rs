pub mod types;

#[cfg(feature = "debug-server")]
pub mod debug_server;

mod track_manager;

use std::collections::VecDeque;

use std::sync::{Arc, Mutex};

use arbit_core::db::KeyframeDescriptor;
use arbit_core::img::{build_pyramid, GrayImage, Pyramid, RgbaImage};
use arbit_core::imu::{GravityEstimate, GravityEstimator, PreintegratedImu};
use arbit_core::init::two_view::{
    FeatureMatch, TwoViewInitialization, TwoViewInitializationParams, TwoViewInitializer,
};
use arbit_core::map::{Anchor, WorldMap};
use arbit_core::math::se3::TransformSE3;
use arbit_core::math::CameraIntrinsics;
use arbit_core::relocalize::{PnPRansac, PnPRansacParams};
use arbit_core::track::{
    DescriptorBuffer, FastSeeder, FastSeederConfig, FeatDescriptor, FeatDescriptorExtractor,
    FeatureSeederTrait, HammingFeatMatcher, LKTracker, LucasKanadeConfig, OrbDescriptor,
    TrackObservation, TrackOutcome,
};
use arbit_providers::CameraSample;
use log::{debug, info, warn};
use nalgebra::{Point3, Rotation3, Translation3, UnitQuaternion, Vector2, Vector3};
use tracing::debug_span;

use crate::track_manager::{TrackConfig, TrackManager};

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
    pub bytes_per_row: u32,
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
    pub norm_xy: Vector2<f64>,
    /// Pixel coordinates in the current frame
    pub px_uv: Vector2<f32>,
    /// Depth from camera (distance along optical axis)
    pub depth: f64,
}

/// A landmark projected into the current camera frame for debugging visualization.
#[derive(Debug, Clone)]
pub struct ProjectedLandmark {
    pub landmark_id: u64,
    pub world_xyz: Point3<f64>,
    pub norm_xy: Vector2<f64>,
    pub px_uv: Vector2<f32>,
    pub depth: f64,
}

/// Debug snapshot of the map state for visualization.
#[derive(Debug, Clone)]
pub struct MapDebugSnapshot {
    pub cam_xyz: Vector3<f64>,
    pub cam_rotation: [f64; 9], // 3x3 rotation matrix (row-major)
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
    pub image: RgbaImage,
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
    last_two_view: Option<TwoViewInitialization>,
    trajectory: Vec<Vector3<f64>>,
    /// pose_wc: camera→world. PnP returns pose_cw (world→camera); we store its inverse here.
    pose_wc: TransformSE3,
    last_gravity: Option<GravityEstimate>,
    map: WorldMap,
    frame_index: u64,
    feat_detector: S,
    feat_descriptor: D,
    feat_matcher: HammingFeatMatcher,
    tracker: Arc<LKTracker>,
    track_manager: Mutex<TrackManager<LKTracker>>,
    two_view_initializer: TwoViewInitializer,
    gravity_estimator: GravityEstimator,
    _pnp: PnPRansac,
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

        let tracker = Arc::new(LKTracker::new(LucasKanadeConfig::default()));
        let track_manager = TrackManager::new(
            LKTracker::new(LucasKanadeConfig::default()),
            TrackConfig::default(),
        );

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
            pose_wc: TransformSE3::identity(),
            last_gravity: None,
            gravity_estimator: GravityEstimator::new(0.75),
            map: WorldMap::new(),
            frame_index: 0,
            _pnp: pnp,
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
        let image = RgbaImage::from_raw(
            sample.intrinsics.width,
            sample.intrinsics.height,
            sample.data.to_vec(),
        )
        .unwrap();
        let gray_image = GrayImage::from_raw(
            image.width() as u32,
            image.height() as u32,
            image.clone().into_vec(),
        )
        .unwrap();
        let intrinsics = sample.intrinsics.clone();
        // Step 3: Build pyramid
        let pyramid = self.step_build_pyramid(&gray_image);

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

        let prev_frame_opt = self.current_frame().cloned();
        self.frame_window.push_front(frame_data.clone());
        if self.frame_window.len() > self.config.frame_window_size {
            self.frame_window.pop_back();
        }
        self.frame_index += 1;
        let Some(prev_frame) = prev_frame_opt else {
            warn!("No previous frame found");
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

        // Step 6: Estimate pose (initialization or tracking)
        match self.tracking_state {
            TrackingState::Uninitialized => {
                self.tracking_state = TrackingState::TrackingPreInit;
            }
            TrackingState::TrackingPreInit => {
                let is_key_frame = self.frame_index % 10 == 0;
                if !is_key_frame {
                    return;
                }
                let _span = debug_span!("feat_descriptor").entered();
                let curr_frame = frame_data.clone();
                let seeds = self.feat_detector.seed(&curr_frame.pyramid);
                let descriptors = self.feat_descriptor.describe(&curr_frame.pyramid, &seeds);
                let keyframe_data = KeyFrameData {
                    frame_data: curr_frame,
                    descriptors,
                };
                if let Some(two_view) = self.run_two_view_initialization() {
                    self.last_two_view = Some(two_view.clone());
                    self.tracking_state = TrackingState::Tracking;
                    //Dummy descriptor for now
                    let frame_descriptor = KeyframeDescriptor::from_slice(&[0.0; 5]);
                    // self.map
                    //     .insert_keyframe(self.pose_wc, frame_descriptor, two_view.landmarks_c1);
                } else {
                    self.keyframe_window.push_front(keyframe_data);
                }
            }
            TrackingState::Tracking => {
                // let pnp_observations = frame_data
                //     .tracks
                //     .iter()
                //     .map(|t| PnPObservation {
                //         world_point: project_normalized_to_world(
                //             pos_px_to_normalized(t.initial, t.level_scale, &frame_data.intrinsics),
                //             &frame_data.pose,
                //         ),
                //         normalized_image: pos_px_to_normalized(
                //             t.refined,
                //             t.level_scale,
                //             &frame_data.intrinsics,
                //         ),
                //     })
                //     .collect::<Vec<_>>();
                // let pnp_result = self.pnp.estimate(&pnp_observations);
                // if let Some(pnp_result) = pnp_result {
                //     // pose_cw from PnP
                //     let pose_cw = pnp_result.pose_cw;
                //     let pose_wc = pose_cw.inverse();
                //     self.pose_wc = pose_wc;
                //     self.trajectory.push(self.pose_wc.translation.vector);
                // }
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
                        width: level.image.width(),
                        height: level.image.height(),
                        bytes_per_row: level.image.width(),
                        pixels: level.image.clone().into_vec(),
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
        &self.pose_wc
    }

    /// Step 3: Build image pyramid for multi-scale feature tracking.
    fn step_build_pyramid(&self, image: &GrayImage) -> Pyramid {
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

        // Step 4b: detect new candidates on the previous frame
        let seeds = {
            let _span = debug_span!("seed_features").entered();
            let raw = self.feat_detector.seed(prev_pyramid);
            debug!("      [4] Seeded ({} seeds)", raw.len());
            raw
        };
        let new_tracks = manager.seed_tracks(
            &seeds,
            frame_prev,
            frame_curr,
            prev_pyramid,
            curr_pyramid,
            intrinsics,
        );
        advanced.extend(new_tracks);
        advanced
    }

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
            let curr_intrinsics = &curr_frame.frame_data.intrinsics;

            let matches: Vec<FeatureMatch> = matches_res
                .iter()
                .map(|m| {
                    let qa = &older_keyframe.descriptors[m.query_idx].seed;
                    let tb = &curr_frame.descriptors[m.train_idx].seed;

                    let norm_xy_a = px_uv_to_norm_xy(qa.px_uv, qa.level_scale, prev_intrinsics);
                    let norm_xy_b = px_uv_to_norm_xy(tb.px_uv, tb.level_scale, curr_intrinsics);

                    FeatureMatch {
                        norm_xy_a,
                        norm_xy_b,
                    }
                })
                .collect();

            // Estimate essential matrix and decompose into R, t
            let two_view = {
                let _two_view_span = debug_span!("two_view_estimation").entered();
                let result = self.two_view_initializer.estimate(&matches);
                if let Some(ref tv) = result {
                    info!(target: "arbit_engine", "✅ Two-view SUCCESS: {} inliers, {} landmarks", 
                          tv.inliers.len(), tv.landmarks_c1.len());
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
        let Some((two_view, _prev_intrinsics)) = two_view_result else {
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
        let rotated_two_view =
            scaled_two_view.rotate_world_orientation(&Rotation3::from(world_alignment.rotation));

        // Camera 2 → Camera 1 (where World = Camera 1)
        let rotation_c2c1 = rotated_two_view.rotation_c2c1.matrix();
        let translation_c2c1 = rotated_two_view.translation_c2c1;
        let rotation_c1c2 = rotation_c2c1.transpose(); // Camera 2 → Camera 1 = Camera 2 → World
        let translation_c1c2 = -(rotation_c1c2.transpose() * translation_c2c1); // Camera 2 → World translation

        let pose_c1c2 = TransformSE3::from_parts(
            Translation3::from(translation_c1c2),
            UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix(&rotation_c1c2)),
        );

        self.pose_wc = pose_c1c2.clone();
        self.trajectory.push(self.pose_wc.translation.vector);

        let frame_descriptor_a = KeyframeDescriptor::from_slice(&[0.0; 5]); // TODO: real descriptor
        let landmakrs_len = rotated_two_view.landmarks_c1.len();
        // let features_with_colors: Vec<(Point2<f64>, Point3<f64>, [u8; 3])> = rotated_two_view
        //     .landmarks_c1
        //     .iter()
        //     .zip(&matches)
        //     .map(|(world_xyz, m)| {
        //         // Get the pixel coordinates for the feature in frame A
        //         let px_x = (m.norm_xy_a.x * K.fx + K.cx) as u32;
        //         let px_y = (m.norm_xy_a.y * K.fy + K.cy) as u32;

        //         // Sample the pixel color from the frame
        //         let pixel = frame.img.get_pixel(
        //             px_x.min(frame.img.width() - 1),
        //             px_y.min(frame.img.height() - 1),
        //         );
        //         let color_rgb = [pixel[0], pixel[1], pixel[2]];

        //         // Return (normalized_xy, world_xyz, color)
        //         (
        //             Point2::new(m.norm_xy_a.x, m.norm_xy_a.y),
        //             *world_xyz,
        //             color_rgb,
        //         )
        //     })
        //     .collect();
        // self.map.insert_keyframe(
        //     TransformSE3::identity(), // Camera 1 -> World (Identity since 1 = World)
        //     frame_descriptor_a,
        //     features_with_colors, // Landmarks in Camera 1 frame = World frame
        // );

        info!(
            target: "arbit_engine",
            "✅ Map initialized: Camera 1 (world origin) with {} landmarks, Camera 2 at [{:.3}, {:.3}, {:.3}]",
            landmakrs_len,
            self.pose_wc.translation.x,
            self.pose_wc.translation.y,
            self.pose_wc.translation.z
        );

        Some(rotated_two_view)
    }
}

fn _update_pose(current: &TransformSE3, two_view: &TwoViewInitialization) -> TransformSE3 {
    let rotation = UnitQuaternion::from_rotation_matrix(&two_view.rotation_c2c1);
    let translation = Translation3::from(two_view.translation_c2c1);
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

fn px_uv_to_norm_xy(
    px_uv: Vector2<f32>,
    level_scale: f32,
    intrinsics: &CameraIntrinsics,
) -> nalgebra::Vector2<f64> {
    let u = px_uv.x as f64 / level_scale as f64;
    let v = px_uv.y as f64 / level_scale as f64;

    let x = (u - intrinsics.cx) / (intrinsics.fx);
    let y = (v - intrinsics.cy) / (intrinsics.fy);
    nalgebra::Vector2::new(x, y)
}
