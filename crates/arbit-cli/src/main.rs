use std::sync::Arc;
use std::time::Duration;

use arbit_providers::{ArKitFrame, ArKitIntrinsics, IosCameraProvider, PixelFormat};

fn main() {
    let mut provider = IosCameraProvider::new();

    let simulated_frames = build_sample_frames();

    println!(
        "Booting ARBIT frame ingestion demo ({} samples)...",
        simulated_frames.len()
    );

    for (index, frame) in simulated_frames.into_iter().enumerate() {
        let sample = provider.ingest_frame(frame);
        let capture_ms = sample.timestamps.capture.as_duration().as_secs_f64() * 1_000.0;
        let pipeline_ms = sample.timestamps.pipeline.as_duration().as_secs_f64() * 1_000.0;
        let latency_ms = sample.timestamps.latency.as_secs_f64() * 1_000.0;

        println!(
            "Frame {index}: capture={capture_ms:.3} ms pipeline={pipeline_ms:.3} ms latency={latency_ms:.3} ms format={:?} resolution={}x{}",
            sample.pixel_format, sample.intrinsics.width, sample.intrinsics.height
        );
    }

    println!("Demo complete — timestamps remain monotonic and latency is measured per frame.");
}

fn build_sample_frames() -> Vec<ArKitFrame> {
    let intrinsics = ArKitIntrinsics {
        fx: 1_200.0,
        fy: 1_200.0,
        cx: 960.0,
        cy: 720.0,
        skew: 0.0,
        width: 1_920,
        height: 1_440,
        distortion: None,
    };

    let bytes_per_pixel: usize = 4;
    let buffer_len = (intrinsics.width as usize) * (intrinsics.height as usize) * bytes_per_pixel;
    let bytes_per_row = (intrinsics.width as usize) * bytes_per_pixel;
    let pixel_data: Arc<[u8]> = Arc::from(vec![0u8; buffer_len]);

    // Simulate three frames arriving from the device with increasing capture timestamps.
    vec![
        ArKitFrame {
            timestamp: Duration::from_millis(0),
            intrinsics: intrinsics.clone(),
            pixel_format: PixelFormat::Bgra8,
            bytes_per_row: bytes_per_row,
            data: pixel_data.clone(),
        },
        ArKitFrame {
            timestamp: Duration::from_millis(16),
            intrinsics: intrinsics.clone(),
            pixel_format: PixelFormat::Bgra8,
            bytes_per_row: bytes_per_row,
            data: pixel_data.clone(),
        },
        ArKitFrame {
            timestamp: Duration::from_millis(32),
            intrinsics,
            pixel_format: PixelFormat::Bgra8,
            bytes_per_row: bytes_per_row,
            data: pixel_data,
        },
    ]
}
