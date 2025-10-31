use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use arbit_core::img::{Pyramid, PyramidLevel};
use arbit_core::math::CameraIntrinsics;
use arbit_core::track::{FeatureSeed, LKTracker, TrackObservation, TrackOutcome};
use nalgebra::Vector2;
use tracing::info;

#[derive(Debug, Clone, Copy)]
pub struct Observation {
    pub frame_id: u64,
    pub px_uv: Vector2<f32>,
    pub level_scale: f32,
}

impl Observation {
    pub fn new(frame_id: u64, px_uv: Vector2<f32>, level_scale: f32) -> Self {
        Self {
            frame_id,
            px_uv,
            level_scale,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Track {
    pub id: u64,
    pub obs: Vec<Observation>,
    pub alive: bool,
    pub last_seen: u64,
    pub age: u32,
    pub score: f32,
}

#[derive(Debug, Clone)]
pub struct TrackConfig {
    pub r_promote: f32,
    pub fb_th: f32,
    pub res_th: f32,
    pub target_tracks: usize,
}

impl Default for TrackConfig {
    fn default() -> Self {
        Self {
            r_promote: 4.0,
            fb_th: 5.0,
            res_th: 10.0,
            target_tracks: 800,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct AdvanceStats {
    pub advanced: usize,
    pub killed: usize,
    pub no_converge: usize,
    pub fb_fail: usize,
    pub res_fail: usize,
}

pub trait FlowTracker: Send + Sync + 'static {
    fn track(
        &self,
        prev: &PyramidLevel,
        curr: &PyramidLevel,
        seed: Vector2<f32>,
        prior: Option<Vector2<f32>>,
        intrinsics: &CameraIntrinsics,
    ) -> TrackObservation;
}

pub struct TrackManager<T: FlowTracker> {
    pub tracks: HashMap<u64, Track>,
    pub next_id: AtomicU64,
    pub tracker: T,
    pub config: TrackConfig,
}

impl<T: FlowTracker> TrackManager<T> {
    pub fn new(tracker: T, config: TrackConfig) -> Self {
        Self {
            tracks: HashMap::new(),
            next_id: AtomicU64::new(1),
            tracker,
            config,
        }
    }

    pub fn advance_alive(
        &mut self,
        prev_pyr: &Pyramid,
        curr_pyr: &Pyramid,
        intr: &CameraIntrinsics,
        frame_prev: u64,
        frame_curr: u64,
    ) -> (AdvanceStats, Vec<TrackObservation>) {
        let mut stats = AdvanceStats::default();
        let mut advanced = Vec::new();

        for track in self
            .tracks
            .values_mut()
            .filter(|t| t.alive && t.last_seen == frame_prev)
        {
            let Some(p_prev) = track.obs.last().cloned() else {
                continue;
            };
            let (prev_pyr_level, curr_pyr_level) = (
                &prev_pyr
                    .levels()
                    .iter()
                    .find(|l| l.scale == p_prev.level_scale)
                    .unwrap(),
                &curr_pyr
                    .levels()
                    .iter()
                    .find(|l| l.scale == p_prev.level_scale)
                    .unwrap(),
            );
            let forward =
                self.tracker
                    .track(prev_pyr_level, curr_pyr_level, p_prev.px_uv, None, intr);
            if !matches!(forward.outcome, TrackOutcome::Converged) {
                track.alive = false;
                stats.killed += 1;
                stats.no_converge += 1;
                info!(
                    "Track {} killed: forward tracking did not converge (outcome: {:?})",
                    track.id, forward.outcome
                );
                continue;
            }

            let back = self.tracker.track(
                curr_pyr_level,
                prev_pyr_level,
                forward.refined_px_uv,
                None,
                intr,
            );
            if !matches!(back.outcome, TrackOutcome::Converged) {
                track.alive = false;
                stats.killed += 1;
                stats.fb_fail += 1;
                info!(
                    "Track {} killed: backward tracking did not converge (outcome: {:?})",
                    track.id, back.outcome
                );
                continue;
            }

            let fb_err = (back.refined_px_uv - p_prev.px_uv).norm();
            if fb_err > self.config.fb_th {
                track.alive = false;
                stats.killed += 1;
                stats.fb_fail += 1;
                info!(
                    "Track {} killed: FB error {:.2}px > threshold {:.2}px",
                    track.id, fb_err, self.config.fb_th
                );
                continue;
            }

            if forward.residual > self.config.res_th {
                track.alive = false;
                stats.killed += 1;
                stats.res_fail += 1;
                info!(
                    "Track {} killed: residual {:.3} > threshold {:.3}",
                    track.id, forward.residual, self.config.res_th
                );
                continue;
            }

            track.obs.push(Observation::new(
                frame_curr,
                forward.refined_px_uv,
                p_prev.level_scale,
            ));
            track.last_seen = frame_curr;
            track.age = track.age.saturating_add(1);

            let mut obs = forward;
            obs.fb_err = fb_err;
            obs.id = Some(track.id);
            obs.score = track.score;
            advanced.push(obs);

            stats.advanced += 1;
        }

        (stats, advanced)
    }

    pub fn need_more_features(&self) -> bool {
        self.alive_count() < self.config.target_tracks
    }

    pub fn seed_tracks(
        &mut self,
        prev_frame_seeds: &Vec<FeatureSeed>,
        prev_frame_id: u64,
        curr_frame_id: u64,
        prev_pyr: &Pyramid,
        curr_pyr: &Pyramid,
        intr: &CameraIntrinsics,
    ) -> Vec<TrackObservation> {
        let mut new_tracks = Vec::new();
        for seed in prev_frame_seeds.iter() {
            let mut obs = self.tracker.track(
                &prev_pyr
                    .levels()
                    .iter()
                    .find(|l| l.scale == seed.level_scale)
                    .unwrap(),
                &curr_pyr
                    .levels()
                    .iter()
                    .find(|l| l.scale == seed.level_scale)
                    .unwrap(),
                seed.px_uv,
                None,
                intr,
            );
            if matches!(obs.outcome, TrackOutcome::Converged) {
                let backward = self.tracker.track(
                    &curr_pyr
                        .levels()
                        .iter()
                        .find(|l| l.scale == seed.level_scale)
                        .unwrap(),
                    &prev_pyr
                        .levels()
                        .iter()
                        .find(|l| l.scale == seed.level_scale)
                        .unwrap(),
                    obs.refined_px_uv,
                    None,
                    intr,
                );
                obs.fb_err = (backward.refined_px_uv - seed.px_uv).norm();
                info!("Tracking for seed: {:?}; Forward obs: {:?}; Backward obs: {:?}; FB error: {:.2}px", seed, obs, backward, obs.fb_err);
            } else {
                info!("Forward tracking did not converge for seed: {:?}", seed);
                obs.fb_err = f32::MAX;
            }
            obs.score = seed.score;
            obs.id = None;
            new_tracks.push(obs);
        }
        self.promote(&new_tracks, prev_frame_id, curr_frame_id)
    }

    pub fn promote(
        &mut self,
        observations: &Vec<TrackObservation>,
        frame_prev: u64,
        frame_curr: u64,
    ) -> Vec<TrackObservation> {
        let mut accepted: Vec<(usize, f32)> = Vec::new();
        for (idx, obs) in observations.iter().enumerate() {
            if !matches!(obs.outcome, TrackOutcome::Converged) {
                info!(
                    "Track {} not promoted: tracking did not converge",
                    obs.id.unwrap_or(0)
                );
                continue;
            }
            if obs.fb_err > self.config.fb_th || obs.residual > self.config.res_th {
                info!("Track {} not promoted: FB error {:.2}px > threshold {:.2}px or residual {:.3} > threshold {:.3}",
                    obs.id.unwrap_or(0), obs.fb_err, self.config.fb_th, obs.residual, self.config.res_th
                );
                continue;
            }
            if self.near_existing(&obs.refined_px_uv, frame_curr, self.config.r_promote) {
                info!(
                    "Track {} not promoted: near existing track",
                    obs.id.unwrap_or(0)
                );
                continue;
            }
            accepted.push((idx, obs.score));
        }

        accepted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut promoted = Vec::new();
        for (idx, score) in accepted {
            let Some(obs) = observations.get(idx).cloned() else {
                continue;
            };
            let id = self.next_id.fetch_add(1, Ordering::SeqCst);
            let track = Track {
                id,
                obs: vec![
                    Observation::new(frame_prev, obs.initial_px_uv, obs.level_scale),
                    Observation::new(frame_curr, obs.refined_px_uv, obs.level_scale),
                ],
                alive: true,
                last_seen: frame_curr,
                age: 1,
                score,
            };
            info!("Promoting track {} with score {:.3}", id, score);
            self.tracks.insert(id, track);
            let mut obs = obs.clone();
            obs.id = Some(id);
            promoted.push(obs);
        }
        promoted
    }

    pub fn alive_count(&self) -> usize {
        self.tracks.values().filter(|t| t.alive).count()
    }

    fn near_existing(&self, px_uv: &Vector2<f32>, frame_id: u64, radius: f32) -> bool {
        if radius <= 0.0 {
            return false;
        }
        let radius_sq = radius * radius;
        self.tracks.values().any(|track| {
            if !track.alive {
                return false;
            }
            track
                .obs
                .iter()
                .rev()
                .find(|obs| obs.frame_id == frame_id)
                .map(|obs| (obs.px_uv - *px_uv).norm_squared() <= radius_sq)
                .unwrap_or(false)
        })
    }
}

impl FlowTracker for LKTracker {
    fn track(
        &self,
        prev: &PyramidLevel,
        curr: &PyramidLevel,
        seed: Vector2<f32>,
        _prior: Option<Vector2<f32>>,
        intrinsics: &CameraIntrinsics,
    ) -> TrackObservation {
        LKTracker::track(self, prev, curr, seed, None, Some(intrinsics))
    }
}
