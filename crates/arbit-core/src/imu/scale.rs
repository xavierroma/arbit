use nalgebra::Vector3;
use std::collections::VecDeque;

const GRAVITY_MS2: f64 = 9.80665;
const DEFAULT_WINDOW: usize = 240; // four seconds at 60 Hz

/// Rolling estimate of accelerometer norm to monitor metric scale drift.
#[derive(Debug, Clone)]
pub struct ScaleDriftMonitor {
    window: VecDeque<f64>,
    capacity: usize,
    sum: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaleEstimate {
    /// Ratio of recent |a| average versus 1 g.
    pub norm_ratio: f64,
    /// Percent drift relative to 1 g.
    pub drift_percent: f64,
}

impl ScaleDriftMonitor {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self {
            window: VecDeque::with_capacity(capacity),
            capacity,
            sum: 0.0,
        }
    }

    pub fn default() -> Self {
        Self::new(DEFAULT_WINDOW)
    }

    pub fn update(&mut self, accel: Vector3<f64>) -> ScaleEstimate {
        let norm = accel.norm();
        if self.window.len() == self.capacity {
            if let Some(oldest) = self.window.pop_front() {
                self.sum -= oldest;
            }
        }
        self.window.push_back(norm);
        self.sum += norm;

        let avg = if self.window.is_empty() {
            norm
        } else {
            self.sum / (self.window.len() as f64)
        };

        let ratio = avg / GRAVITY_MS2;
        ScaleEstimate {
            norm_ratio: ratio,
            drift_percent: (ratio - 1.0).abs() * 100.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn scale_monitor_stays_within_two_percent() {
        let mut monitor = ScaleDriftMonitor::new(600); // 10 seconds at 60 Hz
        let dt = 1.0 / 60.0;
        let mut last = Vector3::new(0.3, GRAVITY_MS2 - 0.2, -0.1);
        let mut estimate = monitor.update(last);
        for k in 1..600 {
            let wobble = (k as f64) * dt * std::f64::consts::TAU;
            let accel = Vector3::new(
                0.3 + 0.05 * wobble.sin(),
                (GRAVITY_MS2 - 0.15) + 0.07 * wobble.cos(),
                -0.1 + 0.03 * wobble.sin(),
            );
            last = accel;
            estimate = monitor.update(accel);
        }

        assert!(estimate.drift_percent < 2.0);
        assert_relative_eq!(estimate.norm_ratio, 1.0, epsilon = 0.02);
        assert!(monitor.window.len() <= 600);
        assert!(last.norm() > 0.0);
    }
}
