use arbit_core::contracts::{
    BackendOptimizer, BackendUpdate, KeyframeCandidate, PlaceRecognizer, identity_pose,
};

#[derive(Debug, Clone)]
pub struct GraphBackendConfig {
    pub min_inliers_for_stability: u32,
}

impl Default for GraphBackendConfig {
    fn default() -> Self {
        Self {
            min_inliers_for_stability: 48,
        }
    }
}

#[derive(Debug)]
pub struct GraphBackend {
    config: GraphBackendConfig,
    keyframe_count: u64,
    landmark_count: u64,
    loop_closure_events: u64,
    relocalization_ready: bool,
}

impl GraphBackend {
    pub fn new(config: GraphBackendConfig) -> Self {
        Self {
            config,
            keyframe_count: 0,
            landmark_count: 0,
            loop_closure_events: 0,
            relocalization_ready: false,
        }
    }
}

impl Default for GraphBackend {
    fn default() -> Self {
        Self::new(GraphBackendConfig::default())
    }
}

impl BackendOptimizer for GraphBackend {
    fn ingest_candidate(&mut self, candidate: &KeyframeCandidate) -> BackendUpdate {
        self.keyframe_count = self.keyframe_count.saturating_add(1);
        self.landmark_count = self
            .landmark_count
            .saturating_add((candidate.inlier_count as u64).max(24));

        if candidate.inlier_count >= self.config.min_inliers_for_stability {
            self.relocalization_ready = true;
        }

        BackendUpdate {
            keyframe_count: self.keyframe_count,
            landmark_count: self.landmark_count,
            loop_closure_events: self.loop_closure_events,
            relocalization_ready: self.relocalization_ready,
            correction_pose_wc: identity_pose(),
        }
    }

    fn loop_closure_tick(&mut self) -> Option<BackendUpdate> {
        if self.keyframe_count == 0 || self.keyframe_count % 25 != 0 {
            return None;
        }

        self.loop_closure_events = self.loop_closure_events.saturating_add(1);
        Some(BackendUpdate {
            keyframe_count: self.keyframe_count,
            landmark_count: self.landmark_count,
            loop_closure_events: self.loop_closure_events,
            relocalization_ready: self.relocalization_ready,
            correction_pose_wc: identity_pose(),
        })
    }

    fn reset(&mut self) {
        self.keyframe_count = 0;
        self.landmark_count = 0;
        self.loop_closure_events = 0;
        self.relocalization_ready = false;
    }
}

#[derive(Debug, Default)]
pub struct NullPlaceRecognizer;

impl PlaceRecognizer for NullPlaceRecognizer {
    fn query(&self, _candidate: &KeyframeCandidate, _max_results: usize) -> Vec<u64> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_promotes_relocalization_when_inliers_are_good() {
        let mut backend = GraphBackend::default();
        let candidate = KeyframeCandidate {
            frame_id: 12,
            timestamp_seconds: 0.4,
            pose_wc: identity_pose(),
            track_count: 140,
            inlier_count: 80,
        };

        let update = backend.ingest_candidate(&candidate);
        assert_eq!(update.keyframe_count, 1);
        assert!(update.relocalization_ready);
        assert!(update.landmark_count >= 24);
    }
}
