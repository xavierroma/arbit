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
