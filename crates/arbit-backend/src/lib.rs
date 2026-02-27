use arbit_core::contracts::{
    BackendOptimizer, BackendUpdate, KeyframeCandidate, PlaceRecognizer, identity_pose,
};
use arbit_native::{
    BowHistogram, BowKernelAdapter, GtsamKernelAdapter, PoseGraphEdge, PoseGraphSolveResult,
};

#[derive(Debug, Clone)]
pub struct GraphBackendConfig {
    pub min_inliers_for_stability: u32,
    pub optimization_window: usize,
    pub loop_closure_similarity_threshold: f32,
}

impl Default for GraphBackendConfig {
    fn default() -> Self {
        Self {
            min_inliers_for_stability: 48,
            optimization_window: 24,
            loop_closure_similarity_threshold: 0.82,
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
    keyframe_history: Vec<KeyframeCandidate>,
    bow_database: Vec<(u64, BowHistogram)>,
    optimizer: GtsamKernelAdapter,
    place_bow: BowKernelAdapter,
}

impl GraphBackend {
    pub fn new(config: GraphBackendConfig) -> Self {
        Self {
            config,
            keyframe_count: 0,
            landmark_count: 0,
            loop_closure_events: 0,
            relocalization_ready: false,
            keyframe_history: Vec::new(),
            bow_database: Vec::new(),
            optimizer: GtsamKernelAdapter,
            place_bow: BowKernelAdapter::default(),
        }
    }

    fn optimize_recent_window(&self) -> Option<PoseGraphSolveResult> {
        let history_len = self.keyframe_history.len();
        if history_len < 2 {
            return None;
        }

        let start = history_len.saturating_sub(self.config.optimization_window);
        let window = &self.keyframe_history[start..history_len];

        let mut edges = Vec::with_capacity(window.len().saturating_sub(1));
        for idx in 1..window.len() {
            let prev = &window[idx - 1];
            let curr = &window[idx];
            edges.push(PoseGraphEdge {
                from: idx - 1,
                to: idx,
                delta_xyz: [
                    curr.pose_wc[3] - prev.pose_wc[3],
                    curr.pose_wc[7] - prev.pose_wc[7],
                    curr.pose_wc[11] - prev.pose_wc[11],
                ],
                sigma: 1.0,
            });
        }

        self.optimizer
            .optimize_translation_graph(window.len(), &edges)
            .ok()
    }

    fn compose_translation_pose(translation: [f64; 3]) -> [f64; 16] {
        let mut pose = identity_pose();
        pose[3] = translation[0];
        pose[7] = translation[1];
        pose[11] = translation[2];
        pose
    }

    fn descriptor_from_candidate(candidate: &KeyframeCandidate) -> [u8; 32] {
        let mut seed = candidate.frame_id
            ^ ((candidate.track_count as u64) << 17)
            ^ ((candidate.inlier_count as u64) << 29)
            ^ ((candidate.timestamp_seconds.to_bits()) << 7);
        let mut out = [0_u8; 32];
        for byte in &mut out {
            seed ^= seed << 7;
            seed ^= seed >> 9;
            seed ^= seed << 8;
            *byte = (seed & 0xff) as u8;
        }
        out
    }

    fn update_loop_closure_state(&mut self, candidate: &KeyframeCandidate) {
        let descriptor = Self::descriptor_from_candidate(candidate);
        let Ok(query_hist) = self.place_bow.encode_descriptors(&[descriptor]) else {
            return;
        };

        let query_db = self
            .bow_database
            .iter()
            .filter(|(id, _)| candidate.frame_id.saturating_sub(*id) > 12)
            .cloned()
            .collect::<Vec<_>>();

        if !query_db.is_empty() {
            if let Ok(top) = self.place_bow.query_top_k(&query_hist, &query_db, 1)
                && let Some((_, score)) = top.first()
                && *score >= self.config.loop_closure_similarity_threshold
            {
                self.loop_closure_events = self.loop_closure_events.saturating_add(1);
                self.relocalization_ready = true;
            }
        }

        self.bow_database.push((candidate.frame_id, query_hist));
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

        self.keyframe_history.push(candidate.clone());
        self.update_loop_closure_state(candidate);

        let correction_pose_wc = self
            .optimize_recent_window()
            .and_then(|solved| solved.translations.last().copied())
            .map(Self::compose_translation_pose)
            .unwrap_or_else(identity_pose);

        BackendUpdate {
            keyframe_count: self.keyframe_count,
            landmark_count: self.landmark_count,
            loop_closure_events: self.loop_closure_events,
            relocalization_ready: self.relocalization_ready,
            correction_pose_wc,
        }
    }

    fn loop_closure_tick(&mut self) -> Option<BackendUpdate> {
        if self.keyframe_count == 0 || self.keyframe_count % 25 != 0 {
            return None;
        }

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
        self.keyframe_history.clear();
        self.bow_database.clear();
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

    fn candidate(frame_id: u64, z: f64, inliers: u32) -> KeyframeCandidate {
        let mut pose = identity_pose();
        pose[11] = z;
        KeyframeCandidate {
            frame_id,
            timestamp_seconds: frame_id as f64 / 30.0,
            pose_wc: pose,
            track_count: inliers.max(32),
            inlier_count: inliers,
        }
    }

    #[test]
    fn backend_promotes_relocalization_when_inliers_are_good() {
        let mut backend = GraphBackend::default();
        let update = backend.ingest_candidate(&candidate(12, -0.02, 80));

        assert_eq!(update.keyframe_count, 1);
        assert!(update.relocalization_ready);
        assert!(update.landmark_count >= 24);
    }

    #[test]
    fn backend_returns_non_identity_correction_after_multiple_keyframes() {
        let mut backend = GraphBackend::default();
        for idx in 0..6 {
            let z = -0.03 * idx as f64;
            let _ = backend.ingest_candidate(&candidate(idx as u64 + 1, z, 72));
        }

        let update = backend.ingest_candidate(&candidate(7, -0.18, 72));
        assert!(update.correction_pose_wc[11] < -0.05);
    }
}
