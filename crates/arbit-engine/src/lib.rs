#[cfg(feature = "debug-server")]
pub mod debug_server;

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use arbit_backend::{GraphBackend, GraphBackendConfig};
use arbit_core::contracts::{
    identity_pose, AnchorSnapshot, BackendOptimizer, EngineSnapshot, FramePacket, FrontendProcessor,
    ImuPacket, KeyframeCandidate, MapRepository, PixelFormat,
};
use arbit_frontend::{CpuFrontend, CpuFrontendConfig};
use arbit_providers::{CameraSample, PixelFormat as ProviderPixelFormat};
use crossbeam::queue::ArrayQueue;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub frame_queue_size: usize,
    pub imu_queue_size: usize,
    pub keyframe_queue_size: usize,
    pub backend_queue_size: usize,
    pub keyframe_interval: u64,
    pub min_tracks_for_keyframe: u32,
    pub run_workers: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            frame_queue_size: 3,
            imu_queue_size: 4_096,
            keyframe_queue_size: 32,
            backend_queue_size: 8,
            keyframe_interval: 10,
            min_tracks_for_keyframe: 120,
            run_workers: true,
        }
    }
}

#[derive(Debug)]
struct RuntimeState {
    frame_queue: ArrayQueue<FramePacket>,
    imu_queue: ArrayQueue<ImuPacket>,
    keyframe_queue: ArrayQueue<KeyframeCandidate>,
    backend_update_queue: ArrayQueue<arbit_core::contracts::BackendUpdate>,
    snapshot: ArcSwap<EngineSnapshot>,
    anchors: Mutex<HashMap<u64, AnchorSnapshot>>,
    next_frame_id: AtomicU64,
    next_anchor_id: AtomicU64,
    dropped_frames: AtomicU64,
    running: AtomicBool,
}

impl RuntimeState {
    fn anchor_count(&self) -> u64 {
        self.anchors
            .lock()
            .expect("anchors mutex poisoned")
            .len() as u64
    }
}

pub struct SlamEngine {
    config: EngineConfig,
    state: Arc<RuntimeState>,
    workers: Vec<JoinHandle<()>>,
}

impl SlamEngine {
    pub fn new() -> Self {
        Self::with_config(EngineConfig::default())
    }

    pub fn with_config(config: EngineConfig) -> Self {
        let state = Arc::new(RuntimeState {
            frame_queue: ArrayQueue::new(config.frame_queue_size),
            imu_queue: ArrayQueue::new(config.imu_queue_size),
            keyframe_queue: ArrayQueue::new(config.keyframe_queue_size),
            backend_update_queue: ArrayQueue::new(config.backend_queue_size),
            snapshot: ArcSwap::new(Arc::new(EngineSnapshot::default())),
            anchors: Mutex::new(HashMap::new()),
            next_frame_id: AtomicU64::new(0),
            next_anchor_id: AtomicU64::new(1),
            dropped_frames: AtomicU64::new(0),
            running: AtomicBool::new(true),
        });

        let mut workers = Vec::new();
        if config.run_workers {
            workers.push(spawn_frontend_worker(state.clone(), config.clone()));
            workers.push(spawn_backend_worker(state.clone()));
            workers.push(spawn_loop_worker(state.clone()));
        }

        Self {
            config,
            state,
            workers,
        }
    }

    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    pub fn ingest_frame(&self, sample: &CameraSample) -> bool {
        let height = sample.intrinsics.height as usize;
        if height == 0 || sample.bytes_per_row == 0 || sample.data.is_empty() {
            return false;
        }

        let frame_id = self.state.next_frame_id.fetch_add(1, Ordering::SeqCst) + 1;
        let packet = FramePacket {
            frame_id,
            timestamp_seconds: sample.timestamps.capture.as_duration().as_secs_f64(),
            intrinsics: sample.intrinsics.clone(),
            pixel_format: pixel_format_from_provider(sample.pixel_format),
            bytes_per_row: sample.bytes_per_row,
            data: sample.data.clone(),
        };

        bounded_latest_push(
            &self.state.frame_queue,
            packet,
            Some(&self.state.dropped_frames),
        );
        true
    }

    pub fn ingest_imu(
        &self,
        timestamp_seconds: f64,
        accel_mps2: [f64; 3],
        gyro_rps: [f64; 3],
    ) -> bool {
        let imu = ImuPacket {
            timestamp_seconds,
            accel_mps2,
            gyro_rps,
        };

        bounded_latest_push(&self.state.imu_queue, imu, None);
        true
    }

    pub fn snapshot(&self) -> EngineSnapshot {
        self.state.snapshot.load().as_ref().clone()
    }

    pub fn reset_session(&self) {
        clear_queue(&self.state.frame_queue);
        clear_queue(&self.state.imu_queue);
        clear_queue(&self.state.keyframe_queue);
        clear_queue(&self.state.backend_update_queue);

        self.state
            .anchors
            .lock()
            .expect("anchors mutex poisoned")
            .clear();

        self.state.next_frame_id.store(0, Ordering::SeqCst);
        self.state.next_anchor_id.store(1, Ordering::SeqCst);
        self.state.dropped_frames.store(0, Ordering::SeqCst);
        self.state.snapshot.store(Arc::new(EngineSnapshot::default()));
    }
}

impl MapRepository for SlamEngine {
    fn create_anchor(&mut self, pose_wc: [f64; 16], keyframe_hint: Option<u64>) -> u64 {
        let anchor_id = self.state.next_anchor_id.fetch_add(1, Ordering::SeqCst);
        let frame_id = self.state.next_frame_id.load(Ordering::Relaxed);
        let anchor = AnchorSnapshot {
            anchor_id,
            pose_wc,
            created_from_keyframe: keyframe_hint,
            last_observed_frame: frame_id,
        };

        self.state
            .anchors
            .lock()
            .expect("anchors mutex poisoned")
            .insert(anchor_id, anchor);

        let mut snapshot = self.snapshot();
        snapshot.map.anchor_count = self.state.anchor_count();
        self.state.snapshot.store(Arc::new(snapshot));
        anchor_id
    }

    fn query_anchor(&self, anchor_id: u64) -> Option<AnchorSnapshot> {
        self.state
            .anchors
            .lock()
            .expect("anchors mutex poisoned")
            .get(&anchor_id)
            .cloned()
    }

    fn anchor_count(&self) -> u64 {
        self.state.anchor_count()
    }

    fn reset(&mut self) {
        self.reset_session();
    }
}

impl Drop for SlamEngine {
    fn drop(&mut self) {
        self.state.running.store(false, Ordering::SeqCst);
        while let Some(worker) = self.workers.pop() {
            let _ = worker.join();
        }
    }
}

impl Default for SlamEngine {
    fn default() -> Self {
        Self::new()
    }
}

fn spawn_frontend_worker(state: Arc<RuntimeState>, config: EngineConfig) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut frontend = CpuFrontend::new(CpuFrontendConfig {
            keyframe_interval: config.keyframe_interval,
            min_tracks_for_keyframe: config.min_tracks_for_keyframe,
        });
        let mut latencies_ms: VecDeque<f64> = VecDeque::with_capacity(120);

        while state.running.load(Ordering::Acquire) {
            let Some(frame) = state.frame_queue.pop() else {
                thread::sleep(Duration::from_millis(1));
                continue;
            };

            let started = Instant::now();
            let mut imu_window = Vec::new();
            while let Some(imu) = state.imu_queue.pop() {
                imu_window.push(imu);
            }

            let output = frontend.process_frame(&frame, &imu_window);
            if output.keyframe_candidate {
                let candidate = KeyframeCandidate {
                    frame_id: output.frame_id,
                    timestamp_seconds: output.timestamp_seconds,
                    pose_wc: output.pose_wc,
                    track_count: output.track_count,
                    inlier_count: output.inlier_count,
                };
                bounded_latest_push(&state.keyframe_queue, candidate, None);
            }

            let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
            if latencies_ms.len() == 120 {
                latencies_ms.pop_front();
            }
            latencies_ms.push_back(elapsed_ms);

            let mut snapshot = state.snapshot.load().as_ref().clone();
            snapshot.timestamp_seconds = frame.timestamp_seconds;
            snapshot.tracking.frame_id = output.frame_id;
            snapshot.tracking.track_count = output.track_count;
            snapshot.tracking.inlier_count = output.inlier_count;
            snapshot.tracking.pose_wc = output.pose_wc;
            snapshot.tracking.state = output.tracking_state;
            snapshot.map.anchor_count = state.anchor_count();
            snapshot.metrics.frame_queue_depth = state.frame_queue.len();
            snapshot.metrics.imu_queue_depth = state.imu_queue.len();
            snapshot.metrics.keyframe_queue_depth = state.keyframe_queue.len();
            snapshot.metrics.backend_queue_depth = state.backend_update_queue.len();
            snapshot.metrics.dropped_frames = state.dropped_frames.load(Ordering::Relaxed);
            snapshot.metrics.frontend_ms_median = percentile(&latencies_ms, 0.5);
            snapshot.metrics.frontend_ms_p95 = percentile(&latencies_ms, 0.95);
            snapshot.metrics.end_to_end_ms_p95 = snapshot.metrics.frontend_ms_p95;
            state.snapshot.store(Arc::new(snapshot));
        }
    })
}

fn spawn_backend_worker(state: Arc<RuntimeState>) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut backend = GraphBackend::new(GraphBackendConfig::default());

        while state.running.load(Ordering::Acquire) {
            let Some(candidate) = state.keyframe_queue.pop() else {
                thread::sleep(Duration::from_millis(1));
                continue;
            };

            let update = backend.ingest_candidate(&candidate);
            bounded_latest_push(&state.backend_update_queue, update.clone(), None);
            apply_backend_update(&state, &update);

            if let Some(loop_update) = backend.loop_closure_tick() {
                bounded_latest_push(&state.backend_update_queue, loop_update.clone(), None);
                apply_backend_update(&state, &loop_update);
            }
        }
    })
}

fn spawn_loop_worker(state: Arc<RuntimeState>) -> JoinHandle<()> {
    thread::spawn(move || {
        while state.running.load(Ordering::Acquire) {
            thread::sleep(Duration::from_secs(2));

            let mut snapshot = state.snapshot.load().as_ref().clone();
            if snapshot.backend.keyframe_count == 0 {
                continue;
            }

            snapshot.backend.loop_closure_events =
                snapshot.backend.loop_closure_events.saturating_add(1);
            snapshot.backend.relocalization_ready = true;
            state.snapshot.store(Arc::new(snapshot));
        }
    })
}

fn apply_backend_update(
    state: &RuntimeState,
    update: &arbit_core::contracts::BackendUpdate,
) {
    let mut snapshot = state.snapshot.load().as_ref().clone();
    snapshot.backend.keyframe_count = update.keyframe_count;
    snapshot.backend.loop_closure_events = update.loop_closure_events;
    snapshot.backend.relocalization_ready = update.relocalization_ready;
    snapshot.map.landmark_count = update.landmark_count;
    snapshot.metrics.backend_queue_depth = state.backend_update_queue.len();
    state.snapshot.store(Arc::new(snapshot));
}

fn bounded_latest_push<T>(
    queue: &ArrayQueue<T>,
    mut value: T,
    drop_counter: Option<&AtomicU64>,
) {
    loop {
        match queue.push(value) {
            Ok(()) => return,
            Err(returned) => {
                value = returned;
                let _ = queue.pop();
                if let Some(counter) = drop_counter {
                    counter.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }
}

fn clear_queue<T>(queue: &ArrayQueue<T>) {
    while queue.pop().is_some() {}
}

fn percentile(samples: &VecDeque<f64>, q: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted = samples.iter().copied().collect::<Vec<_>>();
    sorted.sort_by(f64::total_cmp);
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn pixel_format_from_provider(format: ProviderPixelFormat) -> PixelFormat {
    match format {
        ProviderPixelFormat::Bgra8 => PixelFormat::Bgra8,
        ProviderPixelFormat::Rgba8 => PixelFormat::Rgba8,
        ProviderPixelFormat::Nv12 => PixelFormat::Nv12,
        ProviderPixelFormat::Yv12 => PixelFormat::Yv12,
        ProviderPixelFormat::Depth16 => PixelFormat::Depth16,
    }
}

pub fn default_anchor_pose() -> [f64; 16] {
    identity_pose()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arbit_core::math::{CameraIntrinsics, DistortionModel};
    use arbit_core::time::{FrameTimestamps, MonotonicTime};
    use arbit_providers::{CameraSample, PixelFormat};

    use super::*;

    fn sample_camera(timestamp: f64) -> CameraSample {
        CameraSample {
            timestamps: FrameTimestamps {
                capture: MonotonicTime::from_duration(Duration::from_secs_f64(timestamp)),
                pipeline: MonotonicTime::from_duration(Duration::from_secs_f64(timestamp)),
                latency: Duration::from_millis(0),
            },
            intrinsics: CameraIntrinsics::new(
                600.0,
                600.0,
                320.0,
                240.0,
                0.0,
                640,
                480,
                DistortionModel::None,
            ),
            pixel_format: PixelFormat::Bgra8,
            bytes_per_row: 640 * 4,
            data: Arc::from(vec![0_u8; 640 * 480 * 4]),
        }
    }

    #[test]
    fn reset_session_clears_anchors() {
        let mut engine = SlamEngine::with_config(EngineConfig {
            run_workers: false,
            ..EngineConfig::default()
        });

        let anchor_id = engine.create_anchor(default_anchor_pose(), None);
        assert!(engine.query_anchor(anchor_id).is_some());

        engine.reset_session();
        assert_eq!(engine.anchor_count(), 0);
        assert!(engine.query_anchor(anchor_id).is_none());
    }

    #[test]
    fn frame_ingest_is_non_blocking_when_queue_is_full() {
        let engine = SlamEngine::with_config(EngineConfig {
            frame_queue_size: 1,
            run_workers: false,
            ..EngineConfig::default()
        });

        assert!(engine.ingest_frame(&sample_camera(0.0)));
        assert!(engine.ingest_frame(&sample_camera(0.033)));

        let snapshot = engine.snapshot();
        assert_eq!(snapshot.metrics.dropped_frames, 0);
    }

    #[test]
    fn frontend_worker_updates_tracking_snapshot() {
        let engine = SlamEngine::new();

        assert!(engine.ingest_frame(&sample_camera(0.0)));
        thread::sleep(Duration::from_millis(30));

        let snapshot = engine.snapshot();
        assert!(snapshot.tracking.frame_id >= 1);
        assert!(matches!(
            snapshot.tracking.state,
            arbit_core::contracts::TrackingState::Initializing
                | arbit_core::contracts::TrackingState::Tracking
        ));
    }
}
