use std::sync::Arc;

use crate::math::CameraIntrinsics;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PixelFormat {
    Bgra8,
    Rgba8,
    Nv12,
    Yv12,
    Depth16,
}

#[derive(Debug, Clone)]
pub struct FramePacket {
    pub frame_id: u64,
    pub timestamp_seconds: f64,
    pub intrinsics: CameraIntrinsics,
    pub pixel_format: PixelFormat,
    pub bytes_per_row: usize,
    pub data: Arc<[u8]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ImuPacket {
    pub timestamp_seconds: f64,
    pub accel_mps2: [f64; 3],
    pub gyro_rps: [f64; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackingState {
    Initializing,
    Tracking,
    Relocalizing,
    Lost,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackingSnapshot {
    pub state: TrackingState,
    pub frame_id: u64,
    pub track_count: u32,
    pub inlier_count: u32,
    pub pose_wc: [f64; 16],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendSnapshot {
    pub keyframe_count: u64,
    pub loop_closure_events: u64,
    pub relocalization_ready: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MapSnapshot {
    pub landmark_count: u64,
    pub anchor_count: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnchorSnapshot {
    pub anchor_id: u64,
    pub pose_wc: [f64; 16],
    pub created_from_keyframe: Option<u64>,
    pub last_observed_frame: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuntimeMetricsSnapshot {
    pub frame_queue_depth: usize,
    pub imu_queue_depth: usize,
    pub keyframe_queue_depth: usize,
    pub backend_queue_depth: usize,
    pub dropped_frames: u64,
    pub frontend_ms_median: f64,
    pub frontend_ms_p95: f64,
    pub end_to_end_ms_p95: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineSnapshot {
    pub timestamp_seconds: f64,
    pub tracking: TrackingSnapshot,
    pub backend: BackendSnapshot,
    pub map: MapSnapshot,
    pub metrics: RuntimeMetricsSnapshot,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrontendOutput {
    pub frame_id: u64,
    pub timestamp_seconds: f64,
    pub pose_wc: [f64; 16],
    pub track_count: u32,
    pub inlier_count: u32,
    pub tracking_state: TrackingState,
    pub keyframe_candidate: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KeyframeCandidate {
    pub frame_id: u64,
    pub timestamp_seconds: f64,
    pub pose_wc: [f64; 16],
    pub track_count: u32,
    pub inlier_count: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendUpdate {
    pub keyframe_count: u64,
    pub landmark_count: u64,
    pub loop_closure_events: u64,
    pub relocalization_ready: bool,
    pub correction_pose_wc: [f64; 16],
}

pub trait FrontendProcessor: Send + Sync + 'static {
    fn process_frame(&mut self, frame: &FramePacket, imu_window: &[ImuPacket]) -> FrontendOutput;
    fn reset(&mut self);
}

pub trait BackendOptimizer: Send + Sync + 'static {
    fn ingest_candidate(&mut self, candidate: &KeyframeCandidate) -> BackendUpdate;
    fn loop_closure_tick(&mut self) -> Option<BackendUpdate>;
    fn reset(&mut self);
}

pub trait PlaceRecognizer: Send + Sync + 'static {
    fn query(&self, candidate: &KeyframeCandidate, max_results: usize) -> Vec<u64>;
}

pub trait MapRepository: Send + Sync + 'static {
    fn create_anchor(&mut self, pose_wc: [f64; 16], keyframe_hint: Option<u64>) -> u64;
    fn query_anchor(&self, anchor_id: u64) -> Option<AnchorSnapshot>;
    fn anchor_count(&self) -> u64;
    fn reset(&mut self);
}

pub const fn identity_pose() -> [f64; 16] {
    [
        1.0, 0.0, 0.0, 0.0, // row 0
        0.0, 1.0, 0.0, 0.0, // row 1
        0.0, 0.0, 1.0, 0.0, // row 2
        0.0, 0.0, 0.0, 1.0, // row 3
    ]
}

impl Default for EngineSnapshot {
    fn default() -> Self {
        Self {
            timestamp_seconds: 0.0,
            tracking: TrackingSnapshot {
                state: TrackingState::Initializing,
                frame_id: 0,
                track_count: 0,
                inlier_count: 0,
                pose_wc: identity_pose(),
            },
            backend: BackendSnapshot {
                keyframe_count: 0,
                loop_closure_events: 0,
                relocalization_ready: false,
            },
            map: MapSnapshot {
                landmark_count: 0,
                anchor_count: 0,
            },
            metrics: RuntimeMetricsSnapshot {
                frame_queue_depth: 0,
                imu_queue_depth: 0,
                keyframe_queue_depth: 0,
                backend_queue_depth: 0,
                dropped_frames: 0,
                frontend_ms_median: 0.0,
                frontend_ms_p95: 0.0,
                end_to_end_ms_p95: 0.0,
            },
        }
    }
}
