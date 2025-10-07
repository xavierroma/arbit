use std::path::PathBuf;

/// Complete video session data (video + optional IMU)
#[derive(Debug, Clone)]
pub struct SessionData {
    /// Path to the video file
    pub video_file: PathBuf,
    /// Optional path to IMU data file
    pub imu_file: Option<PathBuf>,
    /// Session name (derived from video filename)
    pub name: String,
}

impl SessionData {
    pub fn new(video_file: PathBuf, imu_file: Option<PathBuf>) -> Self {
        let name = video_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unnamed_session")
            .to_string();

        Self {
            video_file,
            imu_file,
            name,
        }
    }

    pub fn has_imu(&self) -> bool {
        self.imu_file.is_some()
    }
}
