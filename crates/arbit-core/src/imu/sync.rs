use std::collections::VecDeque;
use std::time::Duration;

const EPSILON: Duration = Duration::from_nanos(1);

/// Maintains a running estimate of the constant time offset between IMU and camera timestamps.
#[derive(Debug, Clone)]
pub struct TimeOffsetEstimator {
    window: VecDeque<f64>,
    capacity: usize,
}

impl TimeOffsetEstimator {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self {
            window: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Ingests a timestamp pair and returns the current offset estimate in seconds.
    /// The offset is defined as `imu_time - reference_time`.
    pub fn ingest(&mut self, imu_time: Duration, reference_time: Duration) -> f64 {
        let diff_nanos = (imu_time.as_nanos() as i128) - (reference_time.as_nanos() as i128);
        let diff_seconds = diff_nanos as f64 / 1e9;

        if self.window.len() == self.capacity {
            self.window.pop_front();
        }
        self.window.push_back(diff_seconds);

        self.window.iter().sum::<f64>() / (self.window.len() as f64)
    }

    pub fn len(&self) -> usize {
        self.window.len()
    }

    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }

    pub fn clear(&mut self) {
        self.window.clear();
    }
}

/// Enforces monotonic IMU timestamps to guard against sensor glitches while keeping
/// the data usable by downstream consumers.
#[derive(Debug, Clone, Default)]
pub struct ImuTimestampTracker {
    last: Option<Duration>,
}

impl ImuTimestampTracker {
    pub fn new() -> Self {
        Self { last: None }
    }

    pub fn ingest(&mut self, raw: Duration) -> Duration {
        let next = match self.last {
            Some(prev) if raw <= prev => prev + EPSILON,
            _ => raw,
        };
        self.last = Some(next);
        next
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn estimator_converges_to_positive_offset() {
        let mut estimator = TimeOffsetEstimator::new(32);
        let true_offset = Duration::from_millis(8);

        for k in 0..64 {
            let camera_time = Duration::from_millis(k * 5);
            let imu_time = camera_time + true_offset;
            let estimate = estimator.ingest(imu_time, camera_time);
            assert!(estimator.len() <= 32);
            assert!(estimate > 0.0);
        }

        let estimate = estimator.ingest(Duration::from_millis(321), Duration::from_millis(313));
        assert_relative_eq!(estimate, true_offset.as_secs_f64(), epsilon = 1e-3);
    }

    #[test]
    fn estimator_handles_negative_offset() {
        let mut estimator = TimeOffsetEstimator::new(64);
        let true_offset = Duration::from_millis(10);

        for k in 0..40 {
            let camera_time = Duration::from_millis((k + 3) * 4);
            let imu_time = camera_time.checked_sub(true_offset).unwrap();
            let estimate = estimator.ingest(imu_time, camera_time);
            assert!(estimate < 0.0);
        }

        let estimate = estimator.ingest(Duration::from_millis(500), Duration::from_millis(510));
        assert_relative_eq!(estimate, -true_offset.as_secs_f64(), epsilon = 1e-3);
    }

    #[test]
    fn tracker_enforces_monotonicity() {
        let mut tracker = ImuTimestampTracker::new();
        let mut last = tracker.ingest(Duration::from_millis(0));
        for &candidate in &[
            Duration::from_millis(5),
            Duration::from_millis(4),
            Duration::from_millis(6),
        ] {
            let corrected = tracker.ingest(candidate);
            assert!(corrected > last);
            last = corrected;
        }
    }
}
