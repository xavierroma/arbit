use arbit_core::math::{CameraIntrinsics, DistortionModel};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::errors::{Result, VideoProcessingError};

/// Configuration for video processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    /// Target frame rate for processing
    #[serde(default = "default_frame_rate")]
    pub frame_rate: f64,
    /// Skip N frames between processing
    #[serde(default)]
    pub skip_frames: usize,
    /// Maximum number of frames to process
    pub max_frames: Option<usize>,
    /// Intrinsics source configuration
    pub intrinsics: IntrinsicsConfig,
    /// IMU configuration
    #[serde(default)]
    pub imu: ImuConfig,
    /// Output directory
    #[serde(default = "default_output_dir")]
    pub output_dir: PathBuf,
    /// Verbose logging
    #[serde(default)]
    pub verbose: bool,
}

fn default_frame_rate() -> f64 {
    30.0
}

fn default_output_dir() -> PathBuf {
    PathBuf::from("./arbit-output")
}

/// Configuration for IMU processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImuConfig {
    /// Enable IMU preintegration for improved 6DOF estimates
    #[serde(default = "default_true")]
    pub enable_preintegration: bool,
    /// Gyroscope measurement noise density (rad/s/√Hz)
    #[serde(default = "default_gyro_noise")]
    pub gyro_noise_density: f64,
    /// Accelerometer measurement noise density (m/s²/√Hz)
    #[serde(default = "default_accel_noise")]
    pub accel_noise_density: f64,
    /// Gyroscope bias random walk (rad/s²/√Hz)
    #[serde(default = "default_gyro_bias_walk")]
    pub gyro_bias_random_walk: f64,
    /// Accelerometer bias random walk (m/s³/√Hz)
    #[serde(default = "default_accel_bias_walk")]
    pub accel_bias_random_walk: f64,
    /// Gravity time constant for estimation (seconds)
    #[serde(default = "default_gravity_time_const")]
    pub gravity_time_constant: f64,
    /// Enable motion-aware scale monitoring
    #[serde(default = "default_true")]
    pub motion_aware_scale: bool,
}

impl Default for ImuConfig {
    fn default() -> Self {
        Self {
            enable_preintegration: true,
            gyro_noise_density: default_gyro_noise(),
            accel_noise_density: default_accel_noise(),
            gyro_bias_random_walk: default_gyro_bias_walk(),
            accel_bias_random_walk: default_accel_bias_walk(),
            gravity_time_constant: default_gravity_time_const(),
            motion_aware_scale: true,
        }
    }
}

fn default_true() -> bool {
    true
}

fn default_gyro_noise() -> f64 {
    1.7e-4 // rad/s/√Hz - typical for consumer IMUs
}

fn default_accel_noise() -> f64 {
    2.0e-3 // m/s²/√Hz
}

fn default_gyro_bias_walk() -> f64 {
    1.9e-5 // rad/s²/√Hz
}

fn default_accel_bias_walk() -> f64 {
    3.0e-3 // m/s³/√Hz
}

fn default_gravity_time_const() -> f64 {
    1.0 // seconds - balances responsiveness and smoothness
}

/// Configuration for camera intrinsics
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "source")]
pub enum IntrinsicsConfig {
    /// Load from YAML calibration file
    #[serde(rename = "file")]
    FromFile { path: PathBuf },

    /// Estimate from horizontal FOV and resolution
    #[serde(rename = "fov")]
    EstimateFromFov { horizontal_fov_deg: f64 },

    /// Use known iPhone model defaults
    #[serde(rename = "iphone")]
    DefaultIPhoneModel { model: String },

    /// User-provided explicit parameters
    #[serde(rename = "explicit")]
    Explicit {
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        width: u32,
        height: u32,
        #[serde(default)]
        skew: f64,
        distortion: Option<Vec<f64>>,
    },
}

/// Intrinsics loaded from YAML file
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IntrinsicsFile {
    width: u32,
    height: u32,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    #[serde(default)]
    skew: f64,
    distortion: Option<Vec<f64>>,
}

impl IntrinsicsConfig {
    /// Load camera intrinsics based on the configuration
    pub fn load(&self, video_width: u32, video_height: u32) -> Result<CameraIntrinsics> {
        match self {
            IntrinsicsConfig::FromFile { path } => {
                let contents = std::fs::read_to_string(path).map_err(|e| {
                    VideoProcessingError::IntrinsicsLoad(format!(
                        "Failed to read file {}: {}",
                        path.display(),
                        e
                    ))
                })?;

                let file: IntrinsicsFile = serde_yaml::from_str(&contents)?;

                if file.width != video_width || file.height != video_height {
                    log::warn!(
                        "Intrinsics resolution ({}x{}) doesn't match video resolution ({}x{})",
                        file.width,
                        file.height,
                        video_width,
                        video_height
                    );
                }

                let distortion = file
                    .distortion
                    .map(DistortionModel::Custom)
                    .unwrap_or(DistortionModel::None);

                Ok(CameraIntrinsics::new(
                    file.fx,
                    file.fy,
                    file.cx,
                    file.cy,
                    file.skew,
                    file.width,
                    file.height,
                    distortion,
                ))
            }

            IntrinsicsConfig::EstimateFromFov { horizontal_fov_deg } => {
                let fov_rad = horizontal_fov_deg.to_radians();
                let fx = (video_width as f64 / 2.0) / (fov_rad / 2.0).tan();
                let fy = fx; // Assume square pixels

                Ok(CameraIntrinsics::new(
                    fx,
                    fy,
                    video_width as f64 / 2.0,
                    video_height as f64 / 2.0,
                    0.0,
                    video_width,
                    video_height,
                    DistortionModel::None,
                ))
            }

            IntrinsicsConfig::DefaultIPhoneModel { model } => {
                // Default intrinsics for common iPhone models
                let (fx, fy) = match model.as_str() {
                    "14Pro" => (1200.0, 1200.0),
                    "13Pro" => (1150.0, 1150.0),
                    "12Pro" => (1100.0, 1100.0),
                    _ => {
                        return Err(VideoProcessingError::IntrinsicsInvalid(format!(
                            "Unknown iPhone model: {}. Supported models: 14Pro, 13Pro, 12Pro",
                            model
                        )));
                    }
                };

                Ok(CameraIntrinsics::new(
                    fx,
                    fy,
                    video_width as f64 / 2.0,
                    video_height as f64 / 2.0,
                    0.0,
                    video_width,
                    video_height,
                    DistortionModel::None,
                ))
            }

            IntrinsicsConfig::Explicit {
                fx,
                fy,
                cx,
                cy,
                width,
                height,
                skew,
                distortion,
            } => {
                if *width != video_width || *height != video_height {
                    log::warn!(
                        "Explicit intrinsics resolution ({}x{}) doesn't match video resolution ({}x{})",
                        width,
                        height,
                        video_width,
                        video_height
                    );
                }

                let dist = distortion
                    .as_ref()
                    .map(|d| DistortionModel::Custom(d.clone()))
                    .unwrap_or(DistortionModel::None);

                Ok(CameraIntrinsics::new(
                    *fx, *fy, *cx, *cy, *skew, *width, *height, dist,
                ))
            }
        }
    }
}
