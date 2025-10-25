pub mod db;
pub mod img;
pub mod imu;
pub mod init;
pub mod logs;
pub mod map;
pub mod math;
pub mod relocalize;
pub mod time;
pub mod track;

#[cfg(test)]
mod tests {
    use crate::math::se3::SE3;
    use crate::math::so3::SO3;
    use crate::time::{Clock, TimestampPolicy};
    use std::cell::RefCell;
    use std::time::Duration;

    #[test]
    fn se3_round_trip_log_exp() {
        let rotation = SO3::from_axis_angle(
            &nalgebra::Vector3::new(0.0, 1.0, 0.0),
            std::f64::consts::FRAC_PI_4,
        );
        let translation = nalgebra::Vector3::new(1.0, 2.0, 3.0);
        let pose = SE3::from_parts(rotation, translation);
        let xi = pose.log();
        let reconstructed = SE3::exp(&xi);

        let delta_translation = (reconstructed.translation() - pose.translation()).norm();
        let delta_rotation = (reconstructed.rotation().log() - pose.rotation().log()).norm();

        assert!(delta_translation < 1e-9);
        assert!(delta_rotation < 1e-9);
    }

    #[test]
    fn monotonic_bridge_enforces_order() {
        struct MockClock {
            times: RefCell<Vec<Duration>>,
        }

        impl MockClock {
            fn new(times: Vec<Duration>) -> Self {
                Self {
                    times: RefCell::new(times),
                }
            }
        }

        impl Clock for MockClock {
            fn now(&mut self) -> Duration {
                let mut times = self.times.borrow_mut();
                if times.len() == 1 {
                    times[0]
                } else {
                    times.remove(0)
                }
            }
        }

        let clock = MockClock::new(vec![Duration::from_millis(12), Duration::from_millis(11)]);
        let mut policy = TimestampPolicy::with_clock(clock);

        let sample_a = policy.ingest_capture(Duration::from_millis(10));
        assert_eq!(sample_a.latency, Duration::from_millis(2));

        // Provide out-of-order capture/pipeline times; bridge should clamp to monotonic progression.
        let sample_b = policy.ingest_capture(Duration::from_millis(9));

        assert!(sample_b.capture.as_duration() >= sample_a.capture.as_duration());
        assert!(sample_b.pipeline.as_duration() >= sample_a.pipeline.as_duration());
        assert!(sample_b.latency >= Duration::from_nanos(0));
    }
}
