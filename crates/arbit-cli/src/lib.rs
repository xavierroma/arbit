pub mod errors;
pub mod imu;
pub mod output;
pub mod providers;
pub mod types;
pub mod video;

use std::path::Path;
use std::time::Instant;

use arbit_engine::ProcessingEngine;
use log::{debug, info};

use crate::errors::Result;
use crate::imu::ImuParser;
use crate::output::{AnalysisCollector, FrameStat, ProcessingOutput};
use crate::providers::VideoCameraProvider;
use crate::types::{ProcessingConfig, SessionData};
use crate::video::VideoDecoder;

/// Video processor that runs the ARBIT engine on recorded sessions
pub struct VideoProcessor {
    engine: ProcessingEngine,
    provider: VideoCameraProvider,
    config: ProcessingConfig,
}

impl VideoProcessor {
    pub fn new(config: ProcessingConfig) -> Self {
        Self {
            engine: ProcessingEngine::new(),
            provider: VideoCameraProvider::new(),
            config,
        }
    }

    /// Process a video session (with optional IMU data)
    pub fn process_session<P: AsRef<Path>>(
        &mut self,
        video_path: P,
        imu_path: Option<P>,
    ) -> Result<ProcessingOutput> {
        // Initialize FFmpeg
        VideoDecoder::init()?;

        // Create session data
        let session = SessionData::new(
            video_path.as_ref().to_path_buf(),
            imu_path.as_ref().map(|p| p.as_ref().to_path_buf()),
        );

        info!("Processing session: {}", session.name);

        // Decode video
        let decoder = VideoDecoder::open(&session.video_file)?;
        let video_frames = decoder.decode_frames(&session.video_file)?;
        info!("Decoded {} frames", video_frames.len());

        // Load intrinsics
        let intrinsics = self
            .config
            .intrinsics
            .load(decoder.width(), decoder.height())?;
        info!(
            "Loaded intrinsics: {}x{}, fx={:.2}, fy={:.2}",
            intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy
        );

        // Load IMU data if available
        let mut analysis = AnalysisCollector::new(session.clone());
        let imu_samples = if let Some(ref imu_path) = imu_path {
            let samples = ImuParser::parse_file(imu_path)?;
            analysis.set_imu_samples(samples.clone());
            Some(samples)
        } else {
            None
        };

        // Process frames
        let mut imu_index = 0;
        let duration = decoder.duration().as_secs_f64();

        for (frame_idx, video_frame) in video_frames.iter().enumerate() {
            // Skip frames if configured
            if frame_idx % (self.config.skip_frames + 1) != 0 {
                continue;
            }

            // Check max frames limit
            if let Some(max) = self.config.max_frames {
                if frame_idx >= max {
                    break;
                }
            }

            let frame_start = Instant::now();
            let timestamp = video_frame.timestamp.as_secs_f64();

            info!(
                ">>> Frame {} starting (timestamp: {:.3}s)",
                frame_idx, timestamp
            );

            // Feed IMU samples up to this frame's timestamp
            let imu_start = Instant::now();
            let mut imu_count = 0;
            if let Some(ref samples) = imu_samples {
                while imu_index < samples.len() {
                    let imu_sample = &samples[imu_index];
                    if imu_sample.timestamp_secs() > timestamp {
                        break;
                    }

                    let (ax, ay, az) = imu_sample.accel();
                    let dt = if imu_index > 0 {
                        (imu_sample.timestamp_secs() - samples[imu_index - 1].timestamp_secs())
                            .max(0.001)
                    } else {
                        1.0 / 100.0 // Assume 100Hz
                    };

                    self.engine.ingest_accelerometer(ax, ay, az, dt);
                    analysis.increment_gravity_count();
                    imu_index += 1;
                    imu_count += 1;
                }
            }
            let imu_elapsed = imu_start.elapsed().as_secs_f64() * 1000.0;
            if imu_count > 0 {
                debug!(
                    "  IMU processing: {:.2}ms ({} samples)",
                    imu_elapsed, imu_count
                );
            }

            // Convert video frame to camera sample
            let conversion_start = Instant::now();
            let camera_sample = self
                .provider
                .ingest_frame(video_frame.clone(), intrinsics.clone());
            let conversion_elapsed = conversion_start.elapsed().as_secs_f64() * 1000.0;
            debug!("  Frame conversion: {:.2}ms", conversion_elapsed);

            // Process frame through engine
            let engine_start = Instant::now();
            self.engine.ingest_camera_sample(&camera_sample);
            let engine_elapsed = engine_start.elapsed().as_secs_f64() * 1000.0;
            info!("  Engine processing: {:.2}ms", engine_elapsed);

            let processing_ms = frame_start.elapsed().as_secs_f64() * 1000.0;

            // Collect statistics
            let stats_start = Instant::now();
            let tracks = self.engine.tracked_points().len();
            let inliers = self
                .engine
                .tracked_points()
                .iter()
                .filter(|t| matches!(t.outcome, arbit_core::track::TrackOutcome::Converged))
                .count();

            let (gravity_x, gravity_y, gravity_z) =
                if let Some(gravity) = self.engine.gravity_estimate() {
                    let down = gravity.down().into_inner();
                    (Some(down.x), Some(down.y), Some(down.z))
                } else {
                    (None, None, None)
                };

            analysis.add_pose(self.engine.current_pose().clone());

            analysis.add_frame_stat(FrameStat {
                frame: frame_idx,
                timestamp,
                tracks,
                inliers,
                processing_ms,
                gravity_x,
                gravity_y,
                gravity_z,
            });
            let stats_elapsed = stats_start.elapsed().as_secs_f64() * 1000.0;
            debug!("  Statistics collection: {:.2}ms", stats_elapsed);

            info!(
                "<<< Frame {} complete: {:.2}ms total (tracks={}, inliers={})",
                frame_idx, processing_ms, tracks, inliers
            );

            if self.config.verbose && frame_idx % 10 == 0 {
                debug!(
                    "Frame {}: tracks={}, inliers={}, {:.2}ms",
                    frame_idx, tracks, inliers, processing_ms
                );
            }
        }

        // Get final statistics from engine
        let trajectory = self.engine.trajectory().to_vec();
        let (keyframes, landmarks, _anchors) = self.engine.map_stats();

        info!(
            "Processing complete: {} frames, {} keyframes, {} landmarks",
            video_frames.len(),
            keyframes,
            landmarks
        );

        Ok(analysis.finalize(trajectory, keyframes, landmarks, duration))
    }

    /// Save the current map to a file
    pub fn save_map<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let bytes = self.engine.save_map()?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Load a map from a file
    pub fn load_map<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let bytes = std::fs::read(path)?;
        self.engine.load_map(&bytes)?;
        Ok(())
    }
}
