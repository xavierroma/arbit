use arbit_core::map::MapIoError;
use thiserror::Error;

/// Errors that can occur during video processing
#[derive(Debug, Error)]
pub enum VideoProcessingError {
    #[error("Failed to decode video: {0}")]
    VideoDecoding(String),

    #[error("IMU file format error at line {line}: {message}")]
    ImuFormat { line: usize, message: String },

    #[error("Timestamp synchronization failed: video and IMU timestamps don't overlap")]
    TimestampSync,

    #[error("Failed to load intrinsics: {0}")]
    IntrinsicsLoad(String),

    #[error("Invalid intrinsics: {0}")]
    IntrinsicsInvalid(String),

    #[error("Engine processing error: {0}")]
    EngineError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Map serialization error: {0}")]
    MapIo(#[from] MapIoError),

    #[error("CSV parsing error: {0}")]
    CsvError(#[from] csv::Error),

    #[error("JSON serialization error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("YAML parsing error: {0}")]
    YamlError(#[from] serde_yaml::Error),

    #[error("Video file not found: {0}")]
    VideoFileNotFound(String),

    #[error("IMU file not found: {0}")]
    ImuFileNotFound(String),

    #[error("No frames found in video")]
    NoFrames,

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

pub type Result<T> = std::result::Result<T, VideoProcessingError>;
