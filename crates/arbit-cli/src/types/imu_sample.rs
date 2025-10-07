use std::time::Duration;

/// IMU data point from accelerometer
#[derive(Debug, Clone)]
pub struct ImuSample {
    /// Timestamp of the sample
    pub timestamp: Duration,
    /// Accelerometer data in m/s² (x, y, z)
    pub accelerometer: (f64, f64, f64),
}

impl ImuSample {
    pub fn new(timestamp: Duration, accel_x: f64, accel_y: f64, accel_z: f64) -> Self {
        Self {
            timestamp,
            accelerometer: (accel_x, accel_y, accel_z),
        }
    }

    /// Get the accelerometer components
    pub fn accel(&self) -> (f64, f64, f64) {
        self.accelerometer
    }

    /// Get the timestamp in seconds
    pub fn timestamp_secs(&self) -> f64 {
        self.timestamp.as_secs_f64()
    }
}
