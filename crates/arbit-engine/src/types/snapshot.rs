use crate::ProcessingEngine;
use arbit_core::track::{DescriptorBuffer, FeatDescriptorExtractor, FeatureSeederTrait};
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

#[cfg(feature = "debug-server")]
use utoipa::ToSchema;

/// Snapshot of engine state at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "debug-server", derive(ToSchema))]
pub struct EngineSnapshot {
    pub timestamp: f64,
    pub frame_index: u64,
    pub tracking: TrackingMetrics,
    pub imu: ImuMetrics,
    pub map: MapMetrics,
    pub pose: PoseSnapshot,
}

/// Tracking metrics at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "debug-server", derive(ToSchema))]
pub struct TrackingMetrics {
    pub total_tracks: usize,
    pub converged_tracks: usize,
    pub lost_tracks: usize,
    pub average_residual: Option<f64>,
}

/// IMU-related metrics at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "debug-server", derive(ToSchema))]
pub struct ImuMetrics {
    pub has_preintegration: bool,
    pub gravity_estimate: Option<[f64; 3]>,
    pub gravity_sample_count: u32,
    pub motion_state: Option<String>,
    pub rotation_prior_rad: Option<f64>,
    pub preintegration_count: usize,
}

/// Map statistics at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "debug-server", derive(ToSchema))]
pub struct MapMetrics {
    pub keyframe_count: u64,
    pub landmark_count: u64,
    pub anchor_count: u64,
}

/// Camera pose snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "debug-server", derive(ToSchema))]
pub struct PoseSnapshot {
    pub translation: [f64; 3],
    pub rotation_quaternion: [f64; 4], // [x, y, z, w]
}

/// Trajectory point (pose over time)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "debug-server", derive(ToSchema))]
pub struct TrajectoryPoint {
    pub timestamp: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub qw: f64,
    pub qx: f64,
    pub qy: f64,
    pub qz: f64,
}

impl EngineSnapshot {
    /// Create a snapshot from the current engine state
    pub fn from_engine<S: FeatureSeederTrait, D: FeatDescriptorExtractor>(
        engine: &ProcessingEngine<S, D>,
        timestamp: f64,
        frame_index: u64,
    ) -> Self {
        Self {
            timestamp,
            frame_index,
            tracking: TrackingMetrics::from_engine(engine),
            imu: ImuMetrics::from_engine(engine),
            map: MapMetrics::from_engine(engine),
            pose: PoseSnapshot::from_engine(engine),
        }
    }
}

impl TrackingMetrics {
    pub fn from_engine<S: FeatureSeederTrait, D: FeatDescriptorExtractor>(
        engine: &ProcessingEngine<S, D>,
    ) -> Self {
        let tracks = engine.tracked_points();
        let total_tracks = tracks.len();

        let converged_tracks = tracks
            .iter()
            .filter(|t| matches!(t.outcome, arbit_core::track::TrackOutcome::Converged))
            .count();

        let lost_tracks = total_tracks - converged_tracks;

        // Calculate average residual for converged tracks
        let residuals: Vec<f64> = tracks
            .iter()
            .filter(|t| matches!(t.outcome, arbit_core::track::TrackOutcome::Converged))
            .map(|t| t.residual as f64)
            .collect();

        let average_residual = if residuals.is_empty() {
            None
        } else {
            Some(residuals.iter().sum::<f64>() / residuals.len() as f64)
        };

        Self {
            total_tracks,
            converged_tracks,
            lost_tracks,
            average_residual,
        }
    }
}

impl ImuMetrics {
    pub fn from_engine<S: FeatureSeederTrait, D: FeatDescriptorExtractor>(
        engine: &ProcessingEngine<S, D>,
    ) -> Self {
        let gravity_estimate = engine.gravity_estimate().map(|g| {
            let down = g.down().into_inner();
            [down.x, down.y, down.z]
        });

        let gravity_sample_count = engine.gravity_sample_count();
        let motion_state = engine.last_motion_state();
        let rotation_prior_rad = engine.last_imu_rotation_prior();
        let preintegration_count = engine.preintegration_count();

        Self {
            has_preintegration: engine.has_preintegration(),
            gravity_estimate,
            gravity_sample_count,
            motion_state,
            rotation_prior_rad,
            preintegration_count,
        }
    }
}

impl MapMetrics {
    pub fn from_engine<S: FeatureSeederTrait, D: FeatDescriptorExtractor>(
        engine: &ProcessingEngine<S, D>,
    ) -> Self {
        let (keyframe_count, landmark_count, anchor_count) = engine.map_stats();

        Self {
            keyframe_count,
            landmark_count,
            anchor_count,
        }
    }
}

impl PoseSnapshot {
    pub fn from_engine<S: FeatureSeederTrait, D: FeatDescriptorExtractor>(
        engine: &ProcessingEngine<S, D>,
    ) -> Self {
        let pose = engine.current_pose();
        let quat = pose.rotation.quaternion();

        Self {
            translation: [pose.translation.x, pose.translation.y, pose.translation.z],
            rotation_quaternion: [quat.i, quat.j, quat.k, quat.w],
        }
    }
}

impl TrajectoryPoint {
    /// Create a trajectory point from a pose and timestamp
    pub fn from_pose(pose: &arbit_core::math::se3::TransformSE3, timestamp: f64) -> Self {
        let quat = pose.rotation.quaternion();

        Self {
            timestamp,
            x: pose.translation.x,
            y: pose.translation.y,
            z: pose.translation.z,
            qw: quat.w,
            qx: quat.i,
            qy: quat.j,
            qz: quat.k,
        }
    }

    /// Create a trajectory point from a Vector3 position and timestamp
    pub fn from_position(position: &Vector3<f64>, timestamp: f64) -> Self {
        Self {
            timestamp,
            x: position.x,
            y: position.y,
            z: position.z,
            qw: 1.0,
            qx: 0.0,
            qy: 0.0,
            qz: 0.0,
        }
    }
}
