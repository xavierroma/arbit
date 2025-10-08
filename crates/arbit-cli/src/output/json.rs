use arbit_engine::types::{EngineSnapshot, TrajectoryPoint};
use serde::{Deserialize, Serialize};

/// Complete processing output in JSON format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingOutput {
    pub metadata: Metadata,
    pub trajectory: Vec<TrajectoryPoint>,
    pub snapshots: Vec<EngineSnapshot>,
    pub frame_stats: Vec<FrameStat>,
    pub imu_stats: Option<ImuStats>,
    pub summary: Summary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    pub session_name: String,
    pub video_file: String,
    pub imu_file: Option<String>,
    pub processing_timestamp: String,
    pub frame_count: usize,
    pub imu_sample_count: usize,
    pub duration_seconds: f64,
}

/// Per-frame statistics (CLI-specific, includes processing time)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameStat {
    pub frame: usize,
    pub timestamp: f64,
    pub tracks: usize,
    pub inliers: usize,
    pub processing_ms: f64,
    pub gravity_x: Option<f64>,
    pub gravity_y: Option<f64>,
    pub gravity_z: Option<f64>,
    pub imu_rotation_prior: Option<f64>,
    pub motion_state: Option<String>,
}

/// IMU statistics summary (CLI-specific, aggregated over session)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImuStats {
    pub total_samples: usize,
    pub has_gyroscope_data: bool,
    pub gravity_estimate_count: usize,
    pub average_gravity_magnitude: f64,
    pub gyro_bias_x: f64,
    pub gyro_bias_y: f64,
    pub gyro_bias_z: f64,
    pub accel_bias_x: f64,
    pub accel_bias_y: f64,
    pub accel_bias_z: f64,
    pub preintegration_intervals: usize,
    pub average_motion_state: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Summary {
    pub total_frames: usize,
    pub keyframes_created: u64,
    pub landmarks_created: u64,
    pub average_processing_ms: f64,
    pub imu_fusion_enabled: bool,
}
