use crate::math::se3::TransformSE3;

#[derive(Debug, Clone, PartialEq)]
pub struct FrameLogEntry {
    pub frame_index: u64,
    pub timestamp_seconds: f64,
    pub pose: TransformSE3,
    pub track_count: usize,
    pub inlier_ratio: f32,
    pub forward_backward_error: f32,
}

#[derive(Debug, Default, Clone)]
pub struct ReplayLog {
    entries: Vec<FrameLogEntry>,
}

impl ReplayLog {
    pub fn push(&mut self, entry: FrameLogEntry) {
        if let Some(last) = self.entries.last() {
            assert!(
                entry.timestamp_seconds >= last.timestamp_seconds,
                "Log entries must be time-ordered"
            );
        }
        self.entries.push(entry);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &FrameLogEntry> {
        self.entries.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::se3::TransformSE3;
    use nalgebra::{Translation3, UnitQuaternion};

    #[test]
    fn log_enforces_monotonic_timestamps() {
        let mut log = ReplayLog::default();
        let pose =
            TransformSE3::from_parts(Translation3::new(0.0, 0.0, 0.0), UnitQuaternion::identity());
        log.push(FrameLogEntry {
            frame_index: 0,
            timestamp_seconds: 0.1,
            pose: pose.clone(),
            track_count: 100,
            inlier_ratio: 0.9,
            forward_backward_error: 0.3,
        });
        log.push(FrameLogEntry {
            frame_index: 1,
            timestamp_seconds: 0.2,
            pose: pose.clone(),
            track_count: 95,
            inlier_ratio: 0.88,
            forward_backward_error: 0.35,
        });
        assert_eq!(log.len(), 2);
    }

    #[test]
    #[should_panic]
    fn log_panics_on_time_regression() {
        let mut log = ReplayLog::default();
        let pose = TransformSE3::identity();
        log.push(FrameLogEntry {
            frame_index: 0,
            timestamp_seconds: 0.2,
            pose: pose.clone(),
            track_count: 10,
            inlier_ratio: 0.5,
            forward_backward_error: 1.0,
        });
        log.push(FrameLogEntry {
            frame_index: 1,
            timestamp_seconds: 0.19,
            pose,
            track_count: 8,
            inlier_ratio: 0.4,
            forward_backward_error: 1.1,
        });
    }
}
