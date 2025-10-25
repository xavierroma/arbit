use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use arbit_core::img::{Pyramid, PyramidLevel};
use arbit_core::math::CameraIntrinsics;
use arbit_core::track::{FeatureSeed, TrackObservation, TrackOutcome, Tracker};
use log::trace;
use nalgebra::Vector2;

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
    pub nms_radius: f32,
    pub grid_cell: f32,
    pub per_cell_cap: usize,
    pub r_detect: f32,
    pub r_promote: f32,
    pub fb_th: f32,
    pub res_th: f32,
    pub target_tracks: usize,
    pub score_th: f32,
}

impl Default for TrackConfig {
    fn default() -> Self {
        Self {
            nms_radius: 6.0,
            grid_cell: 32.0,
            per_cell_cap: 3,
            r_detect: 10.0,
            r_promote: 6.0,
            fb_th: 1.2,
            res_th: 1.0,
            target_tracks: 800,
            score_th: 0.8,
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
    pub tracker: Arc<T>,
    pub config: TrackConfig,
}

impl<T: FlowTracker> TrackManager<T> {
    pub fn new(tracker: Arc<T>, config: TrackConfig) -> Self {
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
                trace!(
                    "Track {} killed: forward tracking did not converge (outcome: {:?})",
                    track.id,
                    forward.outcome
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
                trace!(
                    "Track {} killed: backward tracking did not converge (outcome: {:?})",
                    track.id,
                    back.outcome
                );
                continue;
            }

            let fb_err = (back.refined_px_uv - p_prev.px_uv).norm();
            if fb_err > self.config.fb_th {
                track.alive = false;
                stats.killed += 1;
                stats.fb_fail += 1;
                trace!(
                    "Track {} killed: FB error {:.2}px > threshold {:.2}px",
                    track.id,
                    fb_err,
                    self.config.fb_th
                );
                continue;
            }

            if forward.residual > self.config.res_th {
                track.alive = false;
                stats.killed += 1;
                stats.res_fail += 1;
                trace!(
                    "Track {} killed: residual {:.3} > threshold {:.3}",
                    track.id,
                    forward.residual,
                    self.config.res_th
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

    pub fn promote(
        &mut self,
        observations: &mut Vec<TrackObservation>,
        frame_prev: u64,
        frame_curr: u64,
    ) {
        let mut accepted: Vec<(usize, f32)> = Vec::new();
        for (idx, obs) in observations.iter().enumerate() {
            if !matches!(obs.outcome, TrackOutcome::Converged) {
                continue;
            }
            if obs.fb_err > self.config.fb_th || obs.residual > self.config.res_th {
                continue;
            }
            if self.near_existing(&obs.refined_px_uv, frame_curr, self.config.r_promote) {
                continue;
            }
            accepted.push((idx, obs.score));
        }

        accepted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

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
            trace!("Promoting track {} with score {:.3}", id, score);
            self.tracks.insert(id, track);
            if let Some(entry) = observations.get_mut(idx) {
                entry.id = Some(id);
            }
        }
    }

    pub fn alive_count(&self) -> usize {
        self.tracks.values().filter(|t| t.alive).count()
    }

    pub fn live_points_at(&self, frame_id: u64) -> Vec<Vector2<f32>> {
        self.tracks
            .values()
            .filter(|t| t.alive)
            .filter_map(|t| {
                t.obs
                    .iter()
                    .rev()
                    .find(|obs| obs.frame_id == frame_id)
                    .map(|obs| obs.px_uv)
            })
            .collect()
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

pub fn non_max_suppression(mut points: Vec<FeatureSeed>, radius: f32) -> Vec<FeatureSeed> {
    if points.is_empty() {
        return points;
    }
    let mut result: Vec<FeatureSeed> = Vec::with_capacity(points.len());
    points.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let radius_sq = radius * radius;
    'outer: for candidate in points.into_iter() {
        for existing in result.iter() {
            if (existing.px_uv - candidate.px_uv).norm_squared() <= radius_sq {
                continue 'outer;
            }
        }
        result.push(candidate);
    }
    result
}

pub fn grid_cap(points: Vec<FeatureSeed>, cell: f32, cap: usize) -> Vec<FeatureSeed> {
    if cap == 0 {
        return Vec::new();
    }
    let mut grid: HashMap<(i32, i32), usize> = HashMap::new();
    let mut result = Vec::with_capacity(points.len());
    for seed in points.into_iter() {
        let key = (
            (seed.px_uv.x / cell).floor() as i32,
            (seed.px_uv.y / cell).floor() as i32,
        );
        let count = grid.entry(key).or_insert(0);
        if *count < cap {
            result.push(seed);
            *count += 1;
        }
    }
    result
}

pub fn drop_near_live(
    points: Vec<FeatureSeed>,
    live: Vec<Vector2<f32>>,
    radius: f32,
) -> Vec<FeatureSeed> {
    if live.is_empty() {
        return points;
    }
    let radius_sq = radius * radius;
    points
        .into_iter()
        .filter(|seed| {
            live.iter()
                .all(|p| (p - seed.px_uv).norm_squared() > radius_sq)
        })
        .collect()
}

pub fn apply_score_threshold(mut points: Vec<FeatureSeed>, percentile: f32) -> Vec<FeatureSeed> {
    if points.is_empty() {
        return points;
    }
    let percentile = percentile.clamp(0.0, 1.0);
    if percentile <= 0.0 {
        return points;
    }
    points.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let keep_fraction = (1.0 - percentile).max(0.0);
    let keep_count = ((points.len() as f32) * keep_fraction).ceil() as usize;
    let keep_count = keep_count.max(1).min(points.len());
    points.truncate(keep_count);
    points
}

impl FlowTracker for Tracker {
    fn track(
        &self,
        prev: &PyramidLevel,
        curr: &PyramidLevel,
        seed: Vector2<f32>,
        _prior: Option<Vector2<f32>>,
        intrinsics: &CameraIntrinsics,
    ) -> TrackObservation {
        Tracker::track(self, prev, curr, seed, None, Some(intrinsics))
    }
}
