use std::time::Duration;

/// IMU data point from accelerometer and gyroscope
#[derive(Debug, Clone)]
pub struct ImuSample {
    /// Timestamp of the sample
    pub timestamp: Duration,
    /// Accelerometer data in m/s² (x, y, z)
    pub accelerometer: (f64, f64, f64),
    /// Gyroscope data in rad/s (x, y, z)
    pub gyroscope: (f64, f64, f64),
}

impl ImuSample {
    /// Creates a new IMU sample with full 6DOF sensor data
    pub fn new(
        timestamp: Duration,
        accel_x: f64,
        accel_y: f64,
        accel_z: f64,
        gyro_x: f64,
        gyro_y: f64,
        gyro_z: f64,
    ) -> Self {
        Self {
            timestamp,
            accelerometer: (accel_x, accel_y, accel_z),
            gyroscope: (gyro_x, gyro_y, gyro_z),
        }
    }

    /// Creates an IMU sample with only accelerometer data (gyro zeroed)
    /// For backward compatibility with data that lacks gyroscope measurements
    pub fn from_accel_only(timestamp: Duration, accel_x: f64, accel_y: f64, accel_z: f64) -> Self {
        Self {
            timestamp,
            accelerometer: (accel_x, accel_y, accel_z),
            gyroscope: (0.0, 0.0, 0.0),
        }
    }

    /// Get the accelerometer components in m/s²
    pub fn accel(&self) -> (f64, f64, f64) {
        self.accelerometer
    }

    /// Get the gyroscope components in rad/s
    pub fn gyro(&self) -> (f64, f64, f64) {
        self.gyroscope
    }

    /// Get the timestamp in seconds
    pub fn timestamp_secs(&self) -> f64 {
        self.timestamp.as_secs_f64()
    }
}
