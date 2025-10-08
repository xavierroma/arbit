use log::debug;
use nalgebra::Vector3;
use std::collections::VecDeque;

/// Size of sliding window for motion detection (1 second at 200 Hz)
const MOTION_WINDOW_SIZE: usize = 200;

/// Motion state classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionState {
    /// Device is stationary (suitable for bias estimation)
    Stationary,
    /// Device is moving slowly
    SlowMotion,
    /// Device is moving rapidly
    FastMotion,
}

/// Thresholds for motion classification (empirically tuned for handheld devices)
const GYRO_STATIONARY_THRESHOLD: f64 = 0.05; // rad/s (~3 deg/s)
const GYRO_FAST_THRESHOLD: f64 = 0.5; // rad/s (~30 deg/s)
const ACCEL_STATIONARY_THRESHOLD: f64 = 0.3; // m/s²
const ACCEL_FAST_THRESHOLD: f64 = 2.0; // m/s²

/// Motion statistics computed over a sliding window
#[derive(Debug, Clone, Copy)]
pub struct MotionStats {
    /// RMS of gyroscope magnitude over window
    pub gyro_rms: f64,
    /// RMS of acceleration variance (deviation from gravity)
    pub accel_rms: f64,
    /// Current motion state classification
    pub state: MotionState,
}

/// Detects device motion state using a sliding window of IMU samples.
/// Used to trigger bias re-estimation and adjust tracking parameters.
#[derive(Debug, Clone)]
pub struct MotionDetector {
    gyro_window: VecDeque<Vector3<f64>>,
    accel_window: VecDeque<Vector3<f64>>,
    capacity: usize,
    gravity_estimate: Option<Vector3<f64>>,
}

impl MotionDetector {
    /// Creates a new motion detector with given window capacity
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self {
            gyro_window: VecDeque::with_capacity(capacity),
            accel_window: VecDeque::with_capacity(capacity),
            capacity,
            gravity_estimate: None,
        }
    }

    /// Creates a detector with default settings
    pub fn default() -> Self {
        Self::new(MOTION_WINDOW_SIZE)
    }

    /// Sets the gravity estimate for better acceleration variance computation
    pub fn set_gravity(&mut self, gravity: Vector3<f64>) {
        self.gravity_estimate = Some(gravity);
    }

    /// Ingests new IMU samples and returns updated motion statistics
    pub fn update(&mut self, gyro: Vector3<f64>, accel: Vector3<f64>) -> MotionStats {
        // Maintain sliding windows
        if self.gyro_window.len() == self.capacity {
            self.gyro_window.pop_front();
            self.accel_window.pop_front();
        }
        self.gyro_window.push_back(gyro);
        self.accel_window.push_back(accel);

        // Compute RMS of gyroscope magnitude
        let gyro_rms = if self.gyro_window.is_empty() {
            0.0
        } else {
            let sum_sq: f64 = self.gyro_window.iter().map(|g| g.norm_squared()).sum();
            (sum_sq / self.gyro_window.len() as f64).sqrt()
        };

        // Compute RMS of acceleration variance from gravity
        let accel_rms = if self.accel_window.is_empty() {
            0.0
        } else {
            let gravity = self
                .gravity_estimate
                .unwrap_or_else(|| Vector3::new(0.0, 9.80665, 0.0));
            let sum_sq: f64 = self
                .accel_window
                .iter()
                .map(|a| (a - gravity).norm_squared())
                .sum();
            (sum_sq / self.accel_window.len() as f64).sqrt()
        };

        // Classify motion state
        let state =
            if gyro_rms < GYRO_STATIONARY_THRESHOLD && accel_rms < ACCEL_STATIONARY_THRESHOLD {
                MotionState::Stationary
            } else if gyro_rms > GYRO_FAST_THRESHOLD || accel_rms > ACCEL_FAST_THRESHOLD {
                MotionState::FastMotion
            } else {
                MotionState::SlowMotion
            };

        debug!(target: "arbit_core::imu",
            "Motion state: {:?}, gyro_rms={:.4} rad/s, accel_rms={:.4} m/s²",
            state, gyro_rms, accel_rms
        );

        MotionStats {
            gyro_rms,
            accel_rms,
            state,
        }
    }

    /// Returns the number of samples currently in the window
    pub fn window_size(&self) -> usize {
        self.gyro_window.len()
    }

    /// Clears the motion history
    pub fn reset(&mut self) {
        self.gyro_window.clear();
        self.accel_window.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn detects_stationary_state() {
        let mut detector = MotionDetector::new(50);
        detector.set_gravity(Vector3::new(0.0, 9.8, 0.0));

        // Feed stationary samples
        for _ in 0..60 {
            let gyro = Vector3::new(0.001, 0.002, -0.001);
            let accel = Vector3::new(0.05, 9.82, 0.03);
            let stats = detector.update(gyro, accel);

            if detector.window_size() >= 50 {
                assert_eq!(stats.state, MotionState::Stationary);
                assert!(stats.gyro_rms < GYRO_STATIONARY_THRESHOLD);
                assert!(stats.accel_rms < ACCEL_STATIONARY_THRESHOLD);
            }
        }
    }

    #[test]
    fn detects_fast_motion() {
        let mut detector = MotionDetector::new(50);
        detector.set_gravity(Vector3::new(0.0, 9.8, 0.0));

        // Feed fast motion samples
        for i in 0..60 {
            let phase = (i as f64) * 0.1;
            let gyro = Vector3::new(
                0.6 * phase.sin(),
                0.7 * phase.cos(),
                0.5 * (phase * 1.5).sin(),
            );
            let accel = Vector3::new(
                3.0 * phase.cos(),
                9.8 + 2.5 * phase.sin(),
                2.0 * phase.sin(),
            );
            let stats = detector.update(gyro, accel);

            if detector.window_size() >= 50 {
                assert_eq!(stats.state, MotionState::FastMotion);
            }
        }
    }

    #[test]
    fn detects_slow_motion() {
        let mut detector = MotionDetector::new(50);
        detector.set_gravity(Vector3::new(0.0, 9.8, 0.0));

        // Feed slow motion samples
        for i in 0..60 {
            let phase = (i as f64) * 0.05;
            let gyro = Vector3::new(0.1 * phase.sin(), 0.08 * phase.cos(), 0.05);
            let accel = Vector3::new(
                0.5 * phase.cos(),
                9.8 + 0.6 * phase.sin(),
                0.4 * phase.sin(),
            );
            let stats = detector.update(gyro, accel);

            if detector.window_size() >= 50 {
                assert_eq!(stats.state, MotionState::SlowMotion);
            }
        }
    }

    #[test]
    fn rms_computation_is_accurate() {
        let mut detector = MotionDetector::new(100);

        // Known gyro samples
        let gyro_samples = vec![
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];

        for gyro in &gyro_samples {
            detector.update(*gyro, Vector3::new(0.0, 9.8, 0.0));
        }

        let stats = detector.update(Vector3::zeros(), Vector3::new(0.0, 9.8, 0.0));

        // Expected RMS: sqrt((1 + 1 + 1 + 0) / 4) = sqrt(0.75) ≈ 0.866
        assert_relative_eq!(stats.gyro_rms, 0.866, epsilon = 0.01);
    }
}
