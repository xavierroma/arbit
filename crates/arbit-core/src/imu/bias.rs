use log::debug;
use nalgebra::Vector3;
use std::collections::VecDeque;

/// Default window size for bias estimation (10 seconds at 200 Hz)
const DEFAULT_BIAS_WINDOW: usize = 2000;

/// Minimum samples required before bias estimate is considered valid
const MIN_SAMPLES_FOR_VALID_ESTIMATE: usize = 100;

/// Threshold for detecting stationary state (rad/s for gyro, m/s² for accel)
const GYRO_STATIONARY_THRESHOLD: f64 = 0.05; // ~3 deg/s
const ACCEL_STATIONARY_VARIANCE_THRESHOLD: f64 = 0.5; // m/s²

/// Estimates gyroscope bias by averaging samples during stationary periods.
/// Gyroscope bias drifts over time and must be continuously re-estimated.
#[derive(Debug, Clone)]
pub struct GyroBiasEstimator {
    window: VecDeque<Vector3<f64>>,
    capacity: usize,
    sum: Vector3<f64>,
    estimate: Vector3<f64>,
}

impl GyroBiasEstimator {
    /// Creates a new gyroscope bias estimator with given window capacity
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self {
            window: VecDeque::with_capacity(capacity),
            capacity,
            sum: Vector3::zeros(),
            estimate: Vector3::zeros(),
        }
    }

    /// Creates an estimator with default settings
    pub fn default() -> Self {
        Self::new(DEFAULT_BIAS_WINDOW)
    }

    /// Ingests a new gyroscope sample. Should only be called during stationary periods
    /// for accurate bias estimation.
    pub fn update(&mut self, gyro: Vector3<f64>) {
        // Remove oldest if at capacity
        if self.window.len() == self.capacity {
            if let Some(oldest) = self.window.pop_front() {
                self.sum -= oldest;
            }
        }

        self.window.push_back(gyro);
        self.sum += gyro;

        // Update estimate
        if !self.window.is_empty() {
            self.estimate = self.sum / (self.window.len() as f64);
        }

        debug!(target: "arbit_core::imu",
            "Gyro bias updated: bias=[{:.6}, {:.6}, {:.6}] rad/s, window_size={}",
            self.estimate.x, self.estimate.y, self.estimate.z, self.window.len()
        );
    }

    /// Returns the current bias estimate. Returns zero vector if insufficient samples.
    pub fn bias(&self) -> Vector3<f64> {
        if self.window.len() < MIN_SAMPLES_FOR_VALID_ESTIMATE {
            Vector3::zeros()
        } else {
            self.estimate
        }
    }

    /// Returns whether the estimator has enough samples for a valid estimate
    pub fn is_valid(&self) -> bool {
        self.window.len() >= MIN_SAMPLES_FOR_VALID_ESTIMATE
    }

    /// Returns the number of samples in the window
    pub fn sample_count(&self) -> usize {
        self.window.len()
    }

    /// Clears all samples (useful when motion state changes)
    pub fn reset(&mut self) {
        self.window.clear();
        self.sum = Vector3::zeros();
        debug!(target: "arbit_core::imu", "Gyro bias estimator reset");
    }
}

/// Estimates accelerometer bias using the same approach as gyroscope.
/// Less critical than gyro bias but still useful for accuracy.
#[derive(Debug, Clone)]
pub struct AccelBiasEstimator {
    window: VecDeque<Vector3<f64>>,
    capacity: usize,
    sum: Vector3<f64>,
    estimate: Vector3<f64>,
    gravity_magnitude: f64,
}

impl AccelBiasEstimator {
    /// Creates a new accelerometer bias estimator
    pub fn new(capacity: usize, expected_gravity: f64) -> Self {
        assert!(capacity > 0);
        assert!(expected_gravity > 0.0);
        Self {
            window: VecDeque::with_capacity(capacity),
            capacity,
            sum: Vector3::zeros(),
            estimate: Vector3::zeros(),
            gravity_magnitude: expected_gravity,
        }
    }

    /// Creates an estimator with default settings (1g gravity)
    pub fn default() -> Self {
        Self::new(DEFAULT_BIAS_WINDOW, 9.80665)
    }

    /// Ingests a new accelerometer sample during stationary periods.
    /// The bias is computed relative to expected gravity magnitude.
    pub fn update(&mut self, accel: Vector3<f64>) {
        if self.window.len() == self.capacity {
            if let Some(oldest) = self.window.pop_front() {
                self.sum -= oldest;
            }
        }

        self.window.push_back(accel);
        self.sum += accel;

        if !self.window.is_empty() {
            let mean_accel = self.sum / (self.window.len() as f64);
            let measured_magnitude = mean_accel.norm();

            // Bias is the deviation from expected 1g
            let magnitude_error = measured_magnitude - self.gravity_magnitude;
            self.estimate = if measured_magnitude > f64::EPSILON {
                mean_accel.normalize() * magnitude_error
            } else {
                Vector3::zeros()
            };
        }

        debug!(target: "arbit_core::imu",
            "Accel bias updated: bias=[{:.6}, {:.6}, {:.6}] m/s², window_size={}",
            self.estimate.x, self.estimate.y, self.estimate.z, self.window.len()
        );
    }

    /// Returns the current bias estimate
    pub fn bias(&self) -> Vector3<f64> {
        if self.window.len() < MIN_SAMPLES_FOR_VALID_ESTIMATE {
            Vector3::zeros()
        } else {
            self.estimate
        }
    }

    /// Returns whether the estimator has enough samples
    pub fn is_valid(&self) -> bool {
        self.window.len() >= MIN_SAMPLES_FOR_VALID_ESTIMATE
    }

    /// Returns the number of samples in the window
    pub fn sample_count(&self) -> usize {
        self.window.len()
    }

    /// Clears all samples
    pub fn reset(&mut self) {
        self.window.clear();
        self.sum = Vector3::zeros();
        self.estimate = Vector3::zeros();
        debug!(target: "arbit_core::imu", "Accel bias estimator reset");
    }
}

/// Detects whether the device is stationary based on gyro and accel variance
pub fn is_stationary(gyro_samples: &[Vector3<f64>], accel_samples: &[Vector3<f64>]) -> bool {
    if gyro_samples.is_empty() || accel_samples.is_empty() {
        return false;
    }

    // Check gyro variance
    let gyro_mean =
        gyro_samples.iter().fold(Vector3::zeros(), |acc, g| acc + g) / (gyro_samples.len() as f64);
    let gyro_variance = gyro_samples
        .iter()
        .map(|g| (g - gyro_mean).norm_squared())
        .sum::<f64>()
        / (gyro_samples.len() as f64);

    if gyro_variance.sqrt() > GYRO_STATIONARY_THRESHOLD {
        return false;
    }

    // Check accel variance
    let accel_mean = accel_samples
        .iter()
        .fold(Vector3::zeros(), |acc, a| acc + a)
        / (accel_samples.len() as f64);
    let accel_variance = accel_samples
        .iter()
        .map(|a| (a - accel_mean).norm_squared())
        .sum::<f64>()
        / (accel_samples.len() as f64);

    accel_variance.sqrt() < ACCEL_STATIONARY_VARIANCE_THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn gyro_bias_converges_to_mean() {
        let mut estimator = GyroBiasEstimator::new(500);
        let true_bias = Vector3::new(0.01, -0.02, 0.015); // rad/s

        for _ in 0..600 {
            let noise = Vector3::new(
                (rand::random::<f64>() - 0.5) * 0.001,
                (rand::random::<f64>() - 0.5) * 0.001,
                (rand::random::<f64>() - 0.5) * 0.001,
            );
            estimator.update(true_bias + noise);
        }

        assert!(estimator.is_valid());
        let estimated = estimator.bias();
        assert_relative_eq!(estimated.x, true_bias.x, epsilon = 0.002);
        assert_relative_eq!(estimated.y, true_bias.y, epsilon = 0.002);
        assert_relative_eq!(estimated.z, true_bias.z, epsilon = 0.002);
    }

    #[test]
    fn gyro_bias_invalid_until_sufficient_samples() {
        let mut estimator = GyroBiasEstimator::new(500);
        assert!(!estimator.is_valid());

        for _ in 0..50 {
            estimator.update(Vector3::new(0.01, 0.0, 0.0));
        }
        assert!(!estimator.is_valid());

        for _ in 0..100 {
            estimator.update(Vector3::new(0.01, 0.0, 0.0));
        }
        assert!(estimator.is_valid());
    }

    #[test]
    fn stationary_detection_works() {
        // Stationary case
        let gyro_still = vec![
            Vector3::new(0.001, 0.002, -0.001),
            Vector3::new(0.002, 0.001, -0.002),
            Vector3::new(0.001, 0.002, -0.001),
        ];
        let accel_still = vec![
            Vector3::new(0.1, 9.8, 0.0),
            Vector3::new(0.1, 9.81, 0.0),
            Vector3::new(0.1, 9.80, 0.0),
        ];
        assert!(is_stationary(&gyro_still, &accel_still));

        // Moving case
        let gyro_moving = vec![
            Vector3::new(0.1, 0.2, -0.1),
            Vector3::new(0.3, 0.15, -0.2),
            Vector3::new(0.2, 0.25, -0.15),
        ];
        let accel_moving = vec![
            Vector3::new(1.0, 8.0, 2.0),
            Vector3::new(2.0, 7.0, 3.0),
            Vector3::new(1.5, 8.5, 2.5),
        ];
        assert!(!is_stationary(&gyro_moving, &accel_moving));
    }

    #[test]
    fn reset_clears_estimator() {
        let mut estimator = GyroBiasEstimator::new(100);
        for _ in 0..200 {
            estimator.update(Vector3::new(0.01, 0.0, 0.0));
        }
        assert!(estimator.is_valid());

        estimator.reset();
        assert!(!estimator.is_valid());
        assert_eq!(estimator.sample_count(), 0);
    }
}
