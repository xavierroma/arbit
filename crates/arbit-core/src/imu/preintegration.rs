use log::{debug, warn};
use nalgebra::Vector3;
use std::time::Duration;

use crate::math::{SE3, SO3};

/// Configuration parameters for IMU preintegration.
/// These should be tuned based on the specific IMU sensor characteristics.
#[derive(Debug, Clone)]
pub struct PreintegrationConfig {
    /// Gyroscope measurement noise density (rad/s/√Hz)
    pub gyro_noise_density: f64,
    /// Accelerometer measurement noise density (m/s²/√Hz)
    pub accel_noise_density: f64,
    /// Gyroscope bias random walk (rad/s²/√Hz)
    pub gyro_bias_random_walk: f64,
    /// Accelerometer bias random walk (m/s³/√Hz)
    pub accel_bias_random_walk: f64,
}

impl Default for PreintegrationConfig {
    fn default() -> Self {
        // Conservative defaults suitable for consumer-grade IMUs (iPhone, etc.)
        Self {
            gyro_noise_density: 1.7e-4,     // rad/s/√Hz
            accel_noise_density: 2.0e-3,    // m/s²/√Hz
            gyro_bias_random_walk: 1.9e-5,  // rad/s²/√Hz
            accel_bias_random_walk: 3.0e-3, // m/s³/√Hz
        }
    }
}

/// Preintegrated IMU measurements between two camera frames.
/// Represents the relative motion (rotation, velocity, position) accumulated
/// from integrating gyroscope and accelerometer readings.
/// The rotation delta is used as a prior to improve Lucas-Kanade tracking convergence.
#[derive(Debug, Clone)]
pub struct PreintegratedImu {
    /// Integrated rotation delta (body frame i to body frame j)
    pub delta_rotation: SO3,
    /// Integrated velocity change
    pub delta_velocity: Vector3<f64>,
    /// Integrated position change
    pub delta_position: Vector3<f64>,
    /// Time span covered by this preintegration
    pub delta_time: Duration,
    /// Number of IMU samples integrated
    pub sample_count: usize,
}

impl PreintegratedImu {
    /// Creates a zero preintegration (identity transformation)
    pub fn zero() -> Self {
        Self {
            delta_rotation: SO3::identity(),
            delta_velocity: Vector3::zeros(),
            delta_position: Vector3::zeros(),
            delta_time: Duration::ZERO,
            sample_count: 0,
        }
    }

    /// Converts the preintegrated measurements to an SE(3) transformation.
    /// Note: This only includes rotation and position, not velocity.
    pub fn to_se3(&self) -> SE3 {
        SE3::from_parts(self.delta_rotation.clone(), self.delta_position)
    }
}

/// IMU preintegrator that accumulates measurements between camera frames.
///
/// The `ProcessingEngine` automatically calls `finish()` when each new camera frame
/// is ingested, completing the IMU integration interval and providing rotation priors
/// for Lucas-Kanade tracking. Implements the preintegration theory from "On-Manifold
/// Preintegration for Real-Time Visual-Inertial Odometry" (Forster et al., 2017).
#[derive(Debug, Clone)]
pub struct ImuPreintegrator {
    #[allow(dead_code)] // Stored for future use (e.g., covariance propagation)
    config: PreintegrationConfig,
    delta_rotation: SO3,
    delta_velocity: Vector3<f64>,
    delta_position: Vector3<f64>,
    accumulated_time: Duration,
    sample_count: usize,
    gyro_bias: Vector3<f64>,
    accel_bias: Vector3<f64>,
    last_timestamp: Option<Duration>,
}

impl ImuPreintegrator {
    /// Creates a new preintegrator with given configuration and initial biases
    pub fn new(
        config: PreintegrationConfig,
        gyro_bias: Vector3<f64>,
        accel_bias: Vector3<f64>,
    ) -> Self {
        Self {
            config,
            delta_rotation: SO3::identity(),
            delta_velocity: Vector3::zeros(),
            delta_position: Vector3::zeros(),
            accumulated_time: Duration::ZERO,
            sample_count: 0,
            gyro_bias,
            accel_bias,
            last_timestamp: None,
        }
    }

    /// Creates a preintegrator with default configuration and zero biases
    pub fn default() -> Self {
        Self::new(
            PreintegrationConfig::default(),
            Vector3::zeros(),
            Vector3::zeros(),
        )
    }

    /// Updates the bias estimates. This should be called when new bias estimates
    /// become available, typically after stationary periods.
    pub fn set_biases(&mut self, gyro_bias: Vector3<f64>, accel_bias: Vector3<f64>) {
        self.gyro_bias = gyro_bias;
        self.accel_bias = accel_bias;
        debug!(target: "arbit_core::imu",
            "Updated IMU biases: gyro=[{:.6}, {:.6}, {:.6}], accel=[{:.6}, {:.6}, {:.6}]",
            gyro_bias.x, gyro_bias.y, gyro_bias.z,
            accel_bias.x, accel_bias.y, accel_bias.z
        );
    }

    /// Integrates a single IMU measurement.
    /// * `timestamp` - timestamp of this measurement
    /// * `gyro` - gyroscope reading in rad/s (body frame)
    /// * `accel` - accelerometer reading in m/s² (body frame, includes gravity reaction force)
    /// * `gravity` - gravity vector in world frame, same direction as accel when stationary (e.g., (0, 9.8, 0) if accel points up)
    pub fn integrate(
        &mut self,
        timestamp: Duration,
        gyro: Vector3<f64>,
        accel: Vector3<f64>,
        gravity: Vector3<f64>,
    ) {
        // Compute time delta
        let dt = if let Some(last) = self.last_timestamp {
            let dt_secs = (timestamp.as_secs_f64() - last.as_secs_f64()).abs();
            if dt_secs < 1e-6 || dt_secs > 1.0 {
                warn!(target: "arbit_core::imu",
                    "Suspicious IMU time delta: {:.6}s, skipping integration", dt_secs
                );
                return;
            }
            dt_secs
        } else {
            // First sample, no integration yet
            self.last_timestamp = Some(timestamp);
            return;
        };

        self.last_timestamp = Some(timestamp);

        // Remove biases
        let gyro_corrected = gyro - self.gyro_bias;
        let accel_corrected = accel - self.accel_bias;

        // Integrate rotation using midpoint method
        // ΔR_{ij} = ΔR_{ij} * Exp(ω̂ * dt)
        let rotation_increment = SO3::exp(&(gyro_corrected * dt));
        let new_quat = self.delta_rotation.unit_quaternion() * rotation_increment.unit_quaternion();
        self.delta_rotation = SO3::from_unit_quaternion(new_quat);

        // Rotate acceleration to world frame and remove gravity
        // a_world = R_{ij} * a_body - g
        let accel_world = self.delta_rotation.unit_quaternion() * accel_corrected - gravity;

        // Integrate velocity and position using Euler method
        // Δv_{ij} = Δv_{ij} + a_world * dt
        // Δp_{ij} = Δp_{ij} + Δv_{ij} * dt + 0.5 * a_world * dt²
        self.delta_position += self.delta_velocity * dt + accel_world * (0.5 * dt * dt);
        self.delta_velocity += accel_world * dt;

        self.accumulated_time += Duration::from_secs_f64(dt);
        self.sample_count += 1;

        if self.sample_count % 50 == 0 {
            debug!(target: "arbit_core::imu",
                "Preintegrated {} samples over {:.3}s: rot_angle={:.3}°, vel_norm={:.3} m/s, pos_norm={:.3} m",
                self.sample_count,
                self.accumulated_time.as_secs_f64(),
                self.delta_rotation.log().norm().to_degrees(),
                self.delta_velocity.norm(),
                self.delta_position.norm()
            );
        }
    }

    /// Returns the current preintegrated measurements and resets the integrator
    /// for the next interval.
    ///
    /// **This is called automatically by `ProcessingEngine::ingest_camera_sample()`**
    /// to complete the IMU integration interval and obtain rotation/velocity/position
    /// priors for tracking. You typically don't need to call this manually.
    pub fn finish(&mut self) -> PreintegratedImu {
        let result = PreintegratedImu {
            delta_rotation: self.delta_rotation.clone(),
            delta_velocity: self.delta_velocity,
            delta_position: self.delta_position,
            delta_time: self.accumulated_time,
            sample_count: self.sample_count,
        };

        debug!(target: "arbit_core::imu",
            "Finished preintegration: {} samples, {:.3}s, rotation={:.2}°, position={:.3}m",
            result.sample_count,
            result.delta_time.as_secs_f64(),
            result.delta_rotation.log().norm().to_degrees(),
            result.delta_position.norm()
        );

        // Reset for next interval
        self.reset();

        result
    }

    /// Resets the preintegrator to start a new interval
    pub fn reset(&mut self) {
        self.delta_rotation = SO3::identity();
        self.delta_velocity = Vector3::zeros();
        self.delta_position = Vector3::zeros();
        self.accumulated_time = Duration::ZERO;
        self.sample_count = 0;
        self.last_timestamp = None;
    }

    /// Returns the current number of integrated samples
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Returns the current accumulated time
    pub fn accumulated_time(&self) -> Duration {
        self.accumulated_time
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn preintegration_pure_rotation() {
        let mut preint = ImuPreintegrator::default();
        // Gravity vector should match accelerometer reading when stationary
        let gravity = Vector3::new(0.0, 9.80665, 0.0);

        // Simulate pure rotation around z-axis at 10 deg/s for 1 second
        let angular_velocity = 10_f64.to_radians(); // rad/s
        let dt = 0.005; // 200 Hz
        let num_samples = 200;

        for i in 0..num_samples {
            let timestamp = Duration::from_secs_f64(i as f64 * dt);
            let gyro = Vector3::new(0.0, 0.0, angular_velocity);
            let accel = Vector3::new(0.0, 9.80665, 0.0); // Stationary, just gravity reaction
            preint.integrate(timestamp, gyro, accel, gravity);
        }

        let result = preint.finish();

        // Should have rotated ~10 degrees
        let rotation_angle = result.delta_rotation.log().norm();
        assert_relative_eq!(rotation_angle, 10_f64.to_radians(), epsilon = 0.05);

        // Should have minimal translation
        // Note: Some drift expected due to Euler integration and frame transformation during rotation
        assert!(
            result.delta_position.norm() < 0.5,
            "Position drift: {:.6}",
            result.delta_position.norm()
        );
    }

    #[test]
    fn preintegration_pure_translation() {
        let mut preint = ImuPreintegrator::default();
        let gravity = Vector3::new(0.0, 9.80665, 0.0);

        // Simulate constant acceleration in x direction for 1 second
        let acceleration = 1.0; // m/s²
        let dt = 0.005; // 200 Hz
        let num_samples = 200;

        for i in 0..num_samples {
            let timestamp = Duration::from_secs_f64(i as f64 * dt);
            let gyro = Vector3::zeros(); // No rotation
            let accel = Vector3::new(acceleration, 9.80665, 0.0); // Gravity + linear accel
            preint.integrate(timestamp, gyro, accel, gravity);
        }

        let result = preint.finish();

        // Should have minimal rotation
        let rotation_angle = result.delta_rotation.log().norm();
        assert!(
            rotation_angle < 0.01,
            "Unexpected rotation: {:.6} rad",
            rotation_angle
        );

        // Kinematic equation: s = 0.5 * a * t²
        // Expected: 0.5 * 1.0 * 1.0² = 0.5 m
        assert_relative_eq!(result.delta_position.x, 0.5, epsilon = 0.1);
        assert!(
            result.delta_position.y.abs() < 0.01,
            "Y position drift: {:.6}",
            result.delta_position.y
        );
        assert!(
            result.delta_position.z.abs() < 0.01,
            "Z position drift: {:.6}",
            result.delta_position.z
        );
    }

    #[test]
    fn bias_correction_works() {
        let gyro_bias = Vector3::new(0.1, 0.0, 0.0);
        let accel_bias = Vector3::zeros();
        let mut preint =
            ImuPreintegrator::new(PreintegrationConfig::default(), gyro_bias, accel_bias);

        let gravity = Vector3::new(0.0, 9.80665, 0.0);
        let dt = 0.005;
        let num_samples = 200;

        // Feed measurements that exactly equal the bias
        for i in 0..num_samples {
            let timestamp = Duration::from_secs_f64(i as f64 * dt);
            let gyro = gyro_bias; // Measurement equals bias
            let accel = Vector3::new(0.0, 9.80665, 0.0); // Stationary
            preint.integrate(timestamp, gyro, accel, gravity);
        }

        let result = preint.finish();

        // After bias correction, should have minimal rotation
        let rotation_angle = result.delta_rotation.log().norm();
        assert!(rotation_angle < 0.01, "Rotation angle: {}", rotation_angle);
    }

    #[test]
    fn reset_clears_state() {
        let mut preint = ImuPreintegrator::default();
        let gravity = Vector3::new(0.0, -9.80665, 0.0);

        // Integrate some samples
        for i in 0..50 {
            let timestamp = Duration::from_secs_f64(i as f64 * 0.005);
            preint.integrate(
                timestamp,
                Vector3::new(0.1, 0.0, 0.0),
                Vector3::new(0.0, 9.8, 0.0),
                gravity,
            );
        }

        assert!(preint.sample_count() > 0);

        preint.reset();

        assert_eq!(preint.sample_count(), 0);
        assert_eq!(preint.accumulated_time(), Duration::ZERO);
        assert!(preint.delta_rotation.log().norm() < 1e-9);
    }
}
