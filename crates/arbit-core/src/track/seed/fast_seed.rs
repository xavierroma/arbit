use super::{FeatureGridConfig, FeatureSeed, FeatureSeederTrait};
use crate::{
    img::{Pyramid, pyramid::PyramidLevel},
    track::seed::utils::radius_nms,
};
use log::{debug, trace};
use nalgebra::Vector2;

/// Standard FAST detector samples a circle of 16 points at radius 3 pixels
const FAST_CIRCLE_RADIUS: usize = 3;

const FAST_CIRCLE_OFFSETS: [(isize, isize); 16] = [
    (0, -3),
    (1, -3),
    (2, -2),
    (3, -1),
    (3, 0),
    (3, 1),
    (2, 2),
    (1, 3),
    (0, 3),
    (-1, 3),
    (-2, 2),
    (-3, 1),
    (-3, 0),
    (-3, -1),
    (-2, -2),
    (-1, -3),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastDetectorType {
    Type5_8,
    Type7_12,
    Type9_16,
}

impl FastDetectorType {
    #[inline]
    fn contiguous_arc_length(self) -> usize {
        match self {
            Self::Type5_8 => 5,
            Self::Type7_12 => 7,
            Self::Type9_16 => 9,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FastDetectorConfig {
    pub intensity_threshold: f32,
    pub nonmax_suppression: bool,
    pub detector_type: FastDetectorType,
}

impl Default for FastDetectorConfig {
    fn default() -> Self {
        Self {
            intensity_threshold: 1.0,
            nonmax_suppression: true,
            detector_type: FastDetectorType::Type9_16,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FastSeederConfig {
    pub grid: FeatureGridConfig,
    pub detector: FastDetectorConfig,
}

impl Default for FastSeederConfig {
    fn default() -> Self {
        Self {
            grid: FeatureGridConfig::default(),
            detector: FastDetectorConfig::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FastSeeder {
    config: FastSeederConfig,
    arc_length: usize,
}

impl FastSeeder {
    pub fn new(config: FastSeederConfig) -> Self {
        let mut detector = config.detector;
        detector.intensity_threshold = detector.intensity_threshold.max(0.0);

        let arc_length = detector.detector_type.contiguous_arc_length().min(16);

        trace!(
            "FAST seeder config: threshold {:.1}, nonmax {}, arc length {}",
            detector.intensity_threshold, detector.nonmax_suppression, arc_length
        );

        Self {
            config: FastSeederConfig {
                grid: config.grid,
                detector,
            },
            arc_length,
        }
    }
}

impl FeatureSeederTrait for FastSeeder {
    fn seed(&self, pyramid: &Pyramid) -> Vec<FeatureSeed> {
        let mut seeds = Vec::new();
        for level in pyramid.levels() {
            let width = level.image.width() as usize;
            let height = level.image.height() as usize;
            let grid_cfg = self.config.grid;
            let detector_cfg = self.config.detector;

            let cell = grid_cfg.cell_size.max(4);
            let per_cell_cap = grid_cfg.per_cell_cap.max(1);
            let max_features = grid_cfg.max_features.max(per_cell_cap);
            let response_threshold = grid_cfg.response_threshold.max(0.0);
            let nms_radius = grid_cfg.nms_radius_px.max(0.0);
            let lk_radius = grid_cfg.window_radius.max(1);
            let border_margin = FAST_CIRCLE_RADIUS.max(lk_radius);

            if width <= border_margin * 2 || height <= border_margin * 2 {
                return Vec::new();
            }

            let x_lo = border_margin;
            let x_hi = width - border_margin;
            let y_lo = border_margin;
            let y_hi = height - border_margin;

            let cells_x = ((x_hi - x_lo) + cell - 1) / cell;
            let cells_y = ((y_hi - y_lo) + cell - 1) / cell;

            let mut score_map = vec![0.0f32; width * height];
            let mut candidates: Vec<Candidate> = Vec::new();

            for y in y_lo..y_hi {
                for x in x_lo..x_hi {
                    if let Some(score) = fast_corner_score(
                        level,
                        x,
                        y,
                        &FAST_CIRCLE_OFFSETS,
                        detector_cfg.intensity_threshold,
                        self.arc_length,
                    ) {
                        let idx = y * width + x;
                        score_map[idx] = score;
                        candidates.push(Candidate { x, y, score });
                    }
                }
            }

            trace!(
                "FAST detected {} raw corners above threshold {:.1}",
                candidates.len(),
                detector_cfg.intensity_threshold
            );

            if candidates.is_empty() {
                debug!(
                    "FAST seeding in {}x{} (oct {}) → 0 (raw)",
                    width, height, level.octave
                );
                continue;
            }

            let candidates = if detector_cfg.nonmax_suppression {
                apply_nonmax_suppression(&candidates, &score_map, width, height)
            } else {
                candidates
            };

            let mut buckets: Vec<Vec<FeatureSeed>> =
                vec![Vec::with_capacity(per_cell_cap); cells_x * cells_y];

            for candidate in candidates.into_iter() {
                if candidate.score < response_threshold {
                    continue;
                }

                if candidate.x < x_lo
                    || candidate.x >= x_hi
                    || candidate.y < y_lo
                    || candidate.y >= y_hi
                {
                    continue;
                }

                let cx = (candidate.x - x_lo) / cell;
                let cy = (candidate.y - y_lo) / cell;
                if cx >= cells_x || cy >= cells_y {
                    continue;
                }

                let bucket_idx = cy * cells_x + cx;
                let bucket = &mut buckets[bucket_idx];
                insert_sorted(
                    bucket,
                    FeatureSeed {
                        level: level.octave,
                        level_scale: level.scale,
                        px_uv: Vector2::new(candidate.x as f32 + 0.5, candidate.y as f32 + 0.5),
                        score: candidate.score,
                    },
                );

                if bucket.len() > per_cell_cap {
                    bucket.pop();
                }
            }

            let mut seeds_level: Vec<FeatureSeed> = buckets.into_iter().flatten().collect();
            if seeds_level.is_empty() {
                debug!(
                    "FAST seeding in {}x{} (oct {}) → 0 (grid filtered)",
                    width, height, level.octave
                );
                continue;
            }

            seeds_level.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut seeds_level = if nms_radius > 0.0 {
                radius_nms(seeds_level, nms_radius, max_features)
            } else {
                seeds_level
            };

            if seeds_level.len() > max_features {
                seeds_level.truncate(max_features);
            }

            debug!(
                "FAST seeding in {}x{} (oct {}, cell {}, K {}, NMS {:.1}px) → {} (max {})",
                width,
                height,
                level.octave,
                cell,
                per_cell_cap,
                nms_radius,
                seeds_level.len(),
                max_features
            );

            seeds.extend(seeds_level);
        }

        seeds
    }
}

#[derive(Debug, Clone)]
struct Candidate {
    x: usize,
    y: usize,
    score: f32,
}

fn fast_corner_score(
    level: &PyramidLevel,
    x: usize,
    y: usize,
    offsets: &[(isize, isize); 16],
    threshold: f32,
    arc_length: usize,
) -> Option<f32> {
    let image = &level.image;
    let center = image.get_pixel(x as u32, y as u32).0[0] as f32;
    let high = center + threshold;
    let low = center - threshold;

    let mut brighter = 0;
    let mut darker = 0;
    for &idx in &[0usize, 4, 8, 12] {
        let (dx, dy) = offsets[idx];
        let sample = image
            .get_pixel((x as isize + dx) as u32, (y as isize + dy) as u32)
            .0[0] as f32;
        if sample >= high {
            brighter += 1;
        } else if sample <= low {
            darker += 1;
        }
    }

    if brighter < 3 && darker < 3 {
        return None;
    }

    let mut circle_vals = [0.0f32; 16];
    for (i, &(dx, dy)) in offsets.iter().enumerate() {
        circle_vals[i] = image
            .get_pixel((x as isize + dx) as u32, (y as isize + dy) as u32)
            .0[0] as f32;
    }

    let mut classifications = [0i8; 32];
    for i in 0..16 {
        let val = circle_vals[i];
        classifications[i] = if val > high {
            1
        } else if val < low {
            -1
        } else {
            0
        };
        classifications[i + 16] = classifications[i];
    }

    let mut idx = 0usize;
    let mut is_corner = false;
    let mut best_bright = 0.0f32;
    let mut best_dark = 0.0f32;

    while idx < classifications.len() {
        match classifications[idx] {
            1 => {
                let mut len = 0usize;
                let mut sum = 0.0f32;
                while idx + len < classifications.len() && classifications[idx + len] == 1 {
                    let circle_idx = (idx + len) % 16;
                    sum += circle_vals[circle_idx] - center;
                    len += 1;
                }
                if len >= arc_length {
                    is_corner = true;
                    best_bright = best_bright.max(sum);
                }
                idx += len;
            }
            -1 => {
                let mut len = 0usize;
                let mut sum = 0.0f32;
                while idx + len < classifications.len() && classifications[idx + len] == -1 {
                    let circle_idx = (idx + len) % 16;
                    sum += center - circle_vals[circle_idx];
                    len += 1;
                }
                if len >= arc_length {
                    is_corner = true;
                    best_dark = best_dark.max(sum);
                }
                idx += len;
            }
            _ => {
                idx += 1;
            }
        }
    }

    if is_corner {
        Some(best_bright.max(best_dark))
    } else {
        None
    }
}

fn apply_nonmax_suppression(
    candidates: &[Candidate],
    score_map: &[f32],
    width: usize,
    height: usize,
) -> Vec<Candidate> {
    let mut filtered = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let x = candidate.x;
        let y = candidate.y;
        let score = candidate.score;
        let x_min = x.saturating_sub(1);
        let x_max = (x + 1).min(width - 1);
        let y_min = y.saturating_sub(1);
        let y_max = (y + 1).min(height - 1);

        let mut is_max = true;
        for ny in y_min..=y_max {
            for nx in x_min..=x_max {
                if nx == x && ny == y {
                    continue;
                }
                let neighbor_score = score_map[ny * width + nx];
                if neighbor_score > score {
                    is_max = false;
                    break;
                }
            }
            if !is_max {
                break;
            }
        }

        if is_max {
            filtered.push(candidate.clone());
        }
    }
    filtered
}

fn insert_sorted(bucket: &mut Vec<FeatureSeed>, seed: FeatureSeed) {
    let mut insert_pos = 0;
    while insert_pos < bucket.len() && bucket[insert_pos].score > seed.score {
        insert_pos += 1;
    }
    bucket.insert(insert_pos, seed);
}
