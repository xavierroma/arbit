use log::{debug, warn};
use nalgebra::Vector3;
use std::collections::VecDeque;

use super::motion_detector::MotionState;

const GRAVITY_MS2: f64 = 9.80665;
const DEFAULT_WINDOW: usize = 240; // four seconds at 60 Hz

/// Rolling estimate of accelerometer norm to monitor metric scale drift.
/// Optionally filters updates based on motion state for more accurate estimates.
#[derive(Debug, Clone)]
pub struct ScaleDriftMonitor {
    window: VecDeque<f64>,
    capacity: usize,
    sum: f64,
    stationary_only: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaleEstimate {
    /// Ratio of recent |a| average versus 1 g.
    pub norm_ratio: f64,
    /// Percent drift relative to 1 g.
    pub drift_percent: f64,
    /// Whether this estimate was computed during stationary period
    pub is_stationary: bool,
}

impl ScaleDriftMonitor {
    /// Creates a new scale drift monitor
    /// * `capacity` - size of rolling window
    /// * `stationary_only` - if true, only updates during stationary periods
    pub fn new(capacity: usize, stationary_only: bool) -> Self {
        assert!(capacity > 0);
        Self {
            window: VecDeque::with_capacity(capacity),
            capacity,
            sum: 0.0,
            stationary_only,
        }
    }

    /// Creates a monitor with default settings (updates during any motion state)
    pub fn default() -> Self {
        Self::new(DEFAULT_WINDOW, false)
    }

    /// Creates a monitor that only updates during stationary periods
    pub fn stationary_only() -> Self {
        Self::new(DEFAULT_WINDOW, true)
    }

    /// Updates the scale estimate, optionally filtering based on motion state.
    ///
    /// * `accel` - accelerometer reading in m/s²
    /// * `motion_state` - optional motion classification; if `None`, always updates
    ///
    /// If `stationary_only` mode is enabled and device is moving, returns last estimate
    /// without updating the window.
    pub fn update(
        &mut self,
        accel: Vector3<f64>,
        motion_state: Option<MotionState>,
    ) -> ScaleEstimate {
        let is_stationary = motion_state
            .map(|s| s == MotionState::Stationary)
            .unwrap_or(false);

        // Skip update if we require stationary but device is moving
        if self.stationary_only && motion_state.is_some() && !is_stationary {
            debug!(target: "arbit_core::imu", 
                "Scale monitor: skipping update during {:?} (stationary_only mode)", motion_state.unwrap());
            return self.compute_estimate(is_stationary);
        }

        let norm = accel.norm();
        if let Some(state) = motion_state {
            debug!(target: "arbit_core::imu", 
                "Scale monitor update: accel norm={:.4}, motion={:?}, window size={}", 
                norm, state, self.window.len());
        } else {
            debug!(target: "arbit_core::imu", 
                "Scale monitor update: accel norm={:.4}, window size={}", 
                norm, self.window.len());
        }

        if self.window.len() == self.capacity {
            if let Some(oldest) = self.window.pop_front() {
                self.sum -= oldest;
            }
        }
        self.window.push_back(norm);
        self.sum += norm;

        self.compute_estimate(is_stationary)
    }

    fn compute_estimate(&self, is_stationary: bool) -> ScaleEstimate {
        let avg = if self.window.is_empty() {
            GRAVITY_MS2
        } else {
            self.sum / (self.window.len() as f64)
        };

        let ratio = avg / GRAVITY_MS2;
        let drift_percent = (ratio - 1.0).abs() * 100.0;

        debug!(target: "arbit_core::imu", 
            "Scale estimate: avg={:.4}, ratio={:.4}, drift={:.2}%, stationary={}",
            avg, ratio, drift_percent, is_stationary);

        if drift_percent > 5.0 {
            warn!(target: "arbit_core::imu", 
                "Large scale drift detected: {:.2}% (ratio: {:.4})", drift_percent, ratio);
        }

        ScaleEstimate {
            norm_ratio: ratio,
            drift_percent,
            is_stationary,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn scale_monitor_stays_within_two_percent() {
        let mut monitor = ScaleDriftMonitor::new(600, false); // 10 seconds at 60 Hz
        let dt = 1.0 / 60.0;
        let mut last = Vector3::new(0.3, GRAVITY_MS2 - 0.2, -0.1);
        let mut estimate = monitor.update(last, None);
        for k in 1..600 {
            let wobble = (k as f64) * dt * std::f64::consts::TAU;
            let accel = Vector3::new(
                0.3 + 0.05 * wobble.sin(),
                (GRAVITY_MS2 - 0.15) + 0.07 * wobble.cos(),
                -0.1 + 0.03 * wobble.sin(),
            );
            last = accel;
            estimate = monitor.update(accel, None);
        }

        assert!(estimate.drift_percent < 2.0);
        assert_relative_eq!(estimate.norm_ratio, 1.0, epsilon = 0.02);
        assert!(monitor.window.len() <= 600);
        assert!(last.norm() > 0.0);
    }

    #[test]
    fn stationary_only_mode_filters_motion() {
        let mut monitor = ScaleDriftMonitor::stationary_only();

        // Feed stationary samples
        for _ in 0..50 {
            let accel = Vector3::new(0.1, GRAVITY_MS2, 0.0);
            monitor.update(accel, Some(MotionState::Stationary));
        }
        assert!(monitor.window.len() == 50);

        // Feed moving samples - should be rejected
        for _ in 0..50 {
            let accel = Vector3::new(2.0, GRAVITY_MS2 + 1.0, 1.0);
            monitor.update(accel, Some(MotionState::FastMotion));
        }
        assert!(monitor.window.len() == 50); // Window didn't grow

        // Feed stationary again
        for _ in 0..10 {
            let accel = Vector3::new(0.1, GRAVITY_MS2, 0.0);
            monitor.update(accel, Some(MotionState::Stationary));
        }
        assert!(monitor.window.len() == 60); // Window grew
    }
}
