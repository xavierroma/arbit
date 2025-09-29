use std::time::{Duration, Instant};

const EPSILON: Duration = Duration::from_nanos(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct MonotonicTimestamp(Duration);

impl MonotonicTimestamp {
    pub fn as_duration(&self) -> Duration {
        self.0
    }

    fn from_duration(duration: Duration) -> Self {
        Self(duration)
    }
}

pub trait Clock {
    fn now(&mut self) -> Duration;
}

#[derive(Debug, Clone)]
pub struct SystemClock {
    origin: Instant,
}

impl Default for SystemClock {
    fn default() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl Clock for SystemClock {
    fn now(&mut self) -> Duration {
        self.origin.elapsed()
    }
}

#[derive(Debug, Clone)]
pub struct TimestampPolicy<C: Clock> {
    clock: C,
    last_capture: Option<MonotonicTimestamp>,
    last_pipeline: Option<MonotonicTimestamp>,
}

impl TimestampPolicy<SystemClock> {
    pub fn new() -> Self {
        Self::with_clock(SystemClock::default())
    }
}

impl<C: Clock> TimestampPolicy<C> {
    pub fn with_clock(clock: C) -> Self {
        Self {
            clock,
            last_capture: None,
            last_pipeline: None,
        }
    }

    pub fn ingest_capture(&mut self, capture: Duration) -> FrameTimestamps {
        let capture_ts = self.enforce_capture_monotonic(capture);
        let pipeline_now = self.clock.now();
        let latency = pipeline_now.checked_sub(capture).unwrap_or_default();
        let pipeline_ts = self.enforce_pipeline_monotonic(pipeline_now);
        FrameTimestamps::new(capture_ts, pipeline_ts, latency)
    }

    fn enforce_capture_monotonic(&mut self, candidate: Duration) -> MonotonicTimestamp {
        let next = self.next_monotonic(candidate, &self.last_capture);
        self.last_capture = Some(next);
        next
    }

    fn enforce_pipeline_monotonic(&mut self, candidate: Duration) -> MonotonicTimestamp {
        let next = self.next_monotonic(candidate, &self.last_pipeline);
        self.last_pipeline = Some(next);
        next
    }

    fn next_monotonic(
        &self,
        candidate: Duration,
        last: &Option<MonotonicTimestamp>,
    ) -> MonotonicTimestamp {
        match last {
            Some(prev) if candidate <= prev.0 => {
                MonotonicTimestamp::from_duration(prev.0.checked_add(EPSILON).unwrap_or(prev.0))
            }
            _ => MonotonicTimestamp::from_duration(candidate),
        }
    }
}

pub type MonotonicClock = TimestampPolicy<SystemClock>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameTimestamps {
    pub capture: MonotonicTimestamp,
    pub pipeline: MonotonicTimestamp,
    pub latency: Duration,
}

impl FrameTimestamps {
    fn new(capture: MonotonicTimestamp, pipeline: MonotonicTimestamp, latency: Duration) -> Self {
        Self {
            capture,
            pipeline,
            latency,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

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

    #[test]
    fn capture_monotonicity_enforced() {
        let clock = MockClock::new(vec![Duration::from_millis(1); 3]);
        let mut policy = TimestampPolicy::with_clock(clock);

        let a = policy.ingest_capture(Duration::from_millis(0)).capture;
        let b = policy.ingest_capture(Duration::from_millis(5)).capture;
        let c = policy.ingest_capture(Duration::from_millis(4)).capture;

        assert!(a.as_duration() < b.as_duration());
        assert!(b.as_duration() < c.as_duration());
    }

    #[test]
    fn pipeline_monotonicity_with_duplicate_samples() {
        let clock = MockClock::new(vec![
            Duration::from_millis(1),
            Duration::from_millis(1),
            Duration::from_millis(1),
        ]);
        let mut policy = TimestampPolicy::with_clock(clock);

        let first = policy.ingest_capture(Duration::from_millis(0)).pipeline;
        let second = policy.ingest_capture(Duration::from_millis(1)).pipeline;
        let third = policy.ingest_capture(Duration::from_millis(2)).pipeline;

        assert!(first.as_duration() < second.as_duration());
        assert!(second.as_duration() < third.as_duration());
    }

    #[test]
    fn latency_is_saturated_at_zero() {
        let clock = MockClock::new(vec![Duration::from_millis(1), Duration::from_millis(2)]);
        let mut policy = TimestampPolicy::with_clock(clock);

        let first = policy.ingest_capture(Duration::from_millis(5));
        assert_eq!(first.latency, Duration::from_millis(0));

        let second = policy.ingest_capture(Duration::from_millis(1));
        assert_eq!(second.latency, Duration::from_millis(1));
    }
}
