use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use arbit_core::contracts::EngineSnapshot;
use arbit_core::time::{Clock, TimestampPolicy};
use arbit_engine::{EngineConfig, SlamEngine};
use arbit_providers::{ArKitFrame, ArKitIntrinsics, IosCameraProvider, PixelFormat};

const WIDTH: u32 = 1_280;
const HEIGHT: u32 = 720;

#[derive(Clone)]
struct DeterministicClock {
    now: Duration,
    step: Duration,
}

impl Default for DeterministicClock {
    fn default() -> Self {
        Self {
            now: Duration::from_millis(0),
            step: Duration::from_millis(1),
        }
    }
}

impl Clock for DeterministicClock {
    fn now(&mut self) -> Duration {
        self.now = self.now.saturating_add(self.step);
        self.now
    }
}

type PerfProvider = IosCameraProvider<DeterministicClock>;

fn build_sample(
    provider: &mut PerfProvider,
    frame_data: &Arc<[u8]>,
    timestamp_seconds: f64,
) -> arbit_providers::CameraSample {
    provider.ingest_frame(ArKitFrame {
        timestamp: Duration::from_secs_f64(timestamp_seconds),
        intrinsics: ArKitIntrinsics {
            fx: 900.0,
            fy: 900.0,
            cx: (WIDTH as f64) * 0.5,
            cy: (HEIGHT as f64) * 0.5,
            skew: 0.0,
            width: WIDTH,
            height: HEIGHT,
            distortion: None,
        },
        pixel_format: PixelFormat::Bgra8,
        bytes_per_row: (WIDTH as usize) * 4,
        data: frame_data.clone(),
    })
}

fn ingest_sequence(
    engine: &SlamEngine,
    provider: &mut PerfProvider,
    frame_data: &Arc<[u8]>,
    frames: usize,
    nominal_fps: f64,
) {
    for idx in 0..frames {
        let ts = (idx as f64) / nominal_fps;
        let sample = build_sample(provider, frame_data, ts);
        assert!(engine.ingest_frame(&sample));

        if idx % 2 == 0 {
            assert!(engine.ingest_imu(ts, [0.0, 0.0, -9.81], [0.0, 0.0, 0.0]));
        }
    }
}

fn wait_for_target(engine: &SlamEngine, frame_id: u64, timeout: Duration) -> EngineSnapshot {
    let start = Instant::now();
    loop {
        let snapshot = engine.snapshot();
        if snapshot.tracking.frame_id >= frame_id && snapshot.metrics.frame_queue_depth == 0 {
            return snapshot;
        }

        if start.elapsed() >= timeout {
            return snapshot;
        }

        thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn realtime_gate_cpu_only_720p() {
    let engine = SlamEngine::with_config(EngineConfig {
        frame_queue_size: 1_024,
        imu_queue_size: 8_192,
        keyframe_queue_size: 256,
        backend_queue_size: 64,
        ..EngineConfig::default()
    });
    let mut provider =
        IosCameraProvider::with_policy(TimestampPolicy::with_clock(DeterministicClock::default()));
    let frame_data: Arc<[u8]> = vec![127_u8; (WIDTH as usize) * (HEIGHT as usize) * 4].into();

    let total_frames = 900_usize;
    let start = Instant::now();
    ingest_sequence(&engine, &mut provider, &frame_data, total_frames, 30.0);
    let snapshot = wait_for_target(&engine, total_frames as u64, Duration::from_secs(6));

    let elapsed_seconds = start.elapsed().as_secs_f64();
    let throughput_fps = (snapshot.tracking.frame_id as f64) / elapsed_seconds.max(1e-6);

    assert!(
        throughput_fps >= 30.0,
        "throughput gate failed: {throughput_fps:.2} FPS"
    );
    assert!(
        snapshot.metrics.frontend_ms_median < 18.0,
        "frontend median gate failed: {:.3} ms",
        snapshot.metrics.frontend_ms_median
    );
    assert!(
        snapshot.metrics.frontend_ms_p95 < 25.0,
        "frontend p95 gate failed: {:.3} ms",
        snapshot.metrics.frontend_ms_p95
    );
    assert!(
        snapshot.metrics.end_to_end_ms_p95 < 50.0,
        "end-to-end p95 gate failed: {:.3} ms",
        snapshot.metrics.end_to_end_ms_p95
    );
}

#[test]
#[ignore = "Nightly soak gate"]
fn soak_gate_queue_stability() {
    let engine = SlamEngine::with_config(EngineConfig {
        frame_queue_size: 2_048,
        imu_queue_size: 16_384,
        keyframe_queue_size: 512,
        backend_queue_size: 128,
        ..EngineConfig::default()
    });
    let mut provider =
        IosCameraProvider::with_policy(TimestampPolicy::with_clock(DeterministicClock::default()));
    let frame_data: Arc<[u8]> = vec![63_u8; (WIDTH as usize) * (HEIGHT as usize) * 4].into();

    let total_frames = 6_000_usize;
    let pacing = Duration::from_millis(1);
    let mut next_tick = Instant::now();
    for idx in 0..total_frames {
        let ts = (idx as f64) / 30.0;
        let sample = build_sample(&mut provider, &frame_data, ts);
        assert!(engine.ingest_frame(&sample));
        if idx % 2 == 0 {
            assert!(engine.ingest_imu(ts, [0.0, 0.0, -9.81], [0.0, 0.0, 0.0]));
        }
        next_tick += pacing;
        let now = Instant::now();
        if next_tick > now {
            thread::sleep(next_tick - now);
        }
    }

    let snapshot = wait_for_target(&engine, total_frames as u64, Duration::from_secs(20));
    let processed = snapshot.tracking.frame_id as f64;
    let drop_rate = (snapshot.metrics.dropped_frames as f64) / (total_frames as f64);

    assert!(
        processed >= (total_frames as f64) * 0.95,
        "soak progress gate failed: processed {processed} of {total_frames}"
    );
    assert!(
        drop_rate < 0.05,
        "soak drop-rate gate failed: {drop_rate:.4}"
    );
}
