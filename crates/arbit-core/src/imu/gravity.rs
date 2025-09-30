use log::{debug, warn};
use nalgebra::{UnitVector3, Vector3};

/// Holds the running gravity estimate along with helper accessors to derive
/// world-aligned bases.
#[derive(Debug, Clone)]
pub struct GravityEstimate {
    gravity_dir: UnitVector3<f64>,
}

impl GravityEstimate {
    /// Returns the unit vector pointing towards gravity (down direction) in the device frame.
    pub fn down(&self) -> UnitVector3<f64> {
        self.gravity_dir
    }

    /// Returns the unit vector opposite to gravity (up direction) in the device frame.
    pub fn up(&self) -> UnitVector3<f64> {
        UnitVector3::new_unchecked(-self.gravity_dir.into_inner())
    }
}

/// Exponential moving-average gravity estimator that assumes IMU samples are provided
/// in m/s². The estimator is tuned via a time constant controlling responsiveness to
/// new measurements. A longer time constant yields a smoother (but slower) estimate.
#[derive(Debug, Clone)]
pub struct GravityEstimator {
    time_constant: f64,
    estimate: Option<Vector3<f64>>,
    samples: usize,
}

impl GravityEstimator {
    /// Creates a new estimator. `time_constant_secs` controls how quickly the estimate reacts
    /// to new data; values in the 0.5–2.0 s range work well for handheld devices.
    pub fn new(time_constant_secs: f64) -> Self {
        assert!(time_constant_secs.is_finite() && time_constant_secs > 0.0);
        Self {
            time_constant: time_constant_secs,
            estimate: None,
            samples: 0,
        }
    }

    /// Ingests an accelerometer sample and updates the gravity estimate.
    /// * `accel` — accelerometer reading in m/s² expressed in device coordinates.
    /// * `dt` — time delta since the previous sample in seconds.
    pub fn update(&mut self, accel: Vector3<f64>, dt: f64) -> Option<GravityEstimate> {
        if !dt.is_finite() || dt <= 0.0 {
            warn!(target: "arbit_core::imu", "Invalid time delta: {:.6}s (must be finite and positive)", dt);
            return self.estimate.as_ref().map(|vec| GravityEstimate {
                gravity_dir: UnitVector3::new_normalize(*vec),
            });
        }

        let alpha = dt / (self.time_constant + dt);
        self.samples += 1;

        let current = match self.estimate {
            Some(prev) => prev * (1.0 - alpha) + accel * alpha,
            None => accel,
        };

        if current.magnitude() < f64::EPSILON {
            warn!(target: "arbit_core::imu", "Accelerometer reading magnitude too small: {:.6}", current.magnitude());
            return self.estimate.as_ref().map(|vec| GravityEstimate {
                gravity_dir: UnitVector3::new_normalize(*vec),
            });
        }

        self.estimate = Some(current);

        if let Some(ref estimate) = self.estimate {
            debug!(target: "arbit_core::imu", "Gravity estimate updated (sample {}): magnitude={:.4}, alpha={:.4}",
                   self.samples, estimate.magnitude(), alpha);
        }

        self.estimate.as_ref().map(|vec| GravityEstimate {
            gravity_dir: UnitVector3::new_normalize(*vec),
        })
    }

    /// Returns how many samples have been ingested.
    pub fn sample_count(&self) -> usize {
        self.samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn gravity_converges_with_noise() {
        let mut estimator = GravityEstimator::new(0.75);
        let true_gravity = Vector3::new(0.1, -9.8, 0.2);
        let dt = 1.0 / 200.0;

        let mut estimate = estimator.update(true_gravity, dt).unwrap();
        for k in 1..6_000 {
            let phase = (k as f64) * 0.01;
            let noise = Vector3::new(0.02 * phase.sin(), -0.03 * phase.cos(), 0.015 * phase.sin());
            estimate = estimator.update(true_gravity + noise, dt).unwrap();
        }

        let down = estimate.down();
        let expected = UnitVector3::new_normalize(true_gravity);
        let angle = down.angle(&expected);
        assert!(angle <= std::f64::consts::PI / 90.0); // <=2°
        assert_relative_eq!(down.norm(), 1.0, epsilon = 1e-6);
        assert_relative_eq!(expected.norm(), 1.0, epsilon = 1e-6);
    }
}
