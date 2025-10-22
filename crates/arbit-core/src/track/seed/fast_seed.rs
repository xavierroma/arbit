use super::{FeatureGridConfig, FeatureSeed, FeatureSeederTrait};
use crate::{img::pyramid::PyramidLevel, track::seed::utils::radius_nms};
use log::{debug, trace};
use nalgebra::Vector2;

const BASE_CIRCLE_OFFSETS: [(isize, isize); 16] = [
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
    pub circle_radius: usize,
}

impl Default for FastDetectorConfig {
    fn default() -> Self {
        Self {
            intensity_threshold: 20.0,
            nonmax_suppression: true,
            detector_type: FastDetectorType::Type9_16,
            circle_radius: 3,
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
    circle_offsets: [(isize, isize); 16],
    arc_length: usize,
}

impl FastSeeder {
    pub fn new(config: FastSeederConfig) -> Self {
        let mut detector = config.detector;
        detector.intensity_threshold = detector.intensity_threshold.max(0.0);
        detector.circle_radius = detector.circle_radius.max(2);

        let arc_length = detector.detector_type.contiguous_arc_length().min(16);
        let circle_offsets = compute_circle_offsets(detector.circle_radius);

        trace!(
            "FAST seeder config: threshold {:.1}, nonmax {}, radius {}, arc length {}",
            detector.intensity_threshold,
            detector.nonmax_suppression,
            detector.circle_radius,
            arc_length
        );

        Self {
            config: FastSeederConfig {
                grid: config.grid,
                detector,
            },
            circle_offsets,
            arc_length,
        }
    }
}

impl FeatureSeederTrait for FastSeeder {
    fn seed(&self, level: &PyramidLevel) -> Vec<FeatureSeed> {
        let width = level.image.width();
        let height = level.image.height();
        let grid_cfg = self.config.grid;
        let detector_cfg = self.config.detector;

        let cell = grid_cfg.cell_size.max(4);
        let per_cell_cap = grid_cfg.per_cell_cap.max(1);
        let max_features = grid_cfg.max_features.max(per_cell_cap);
        let response_threshold = grid_cfg.response_threshold.max(0.0);
        let nms_radius = grid_cfg.nms_radius_px.max(0.0);
        let lk_radius = grid_cfg.window_radius.max(1);
        let circle_radius = detector_cfg.circle_radius.max(lk_radius);

        if width <= circle_radius * 2 || height <= circle_radius * 2 {
            return Vec::new();
        }

        let x_lo = circle_radius;
        let x_hi = width - circle_radius;
        let y_lo = circle_radius;
        let y_hi = height - circle_radius;

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
                    &self.circle_offsets,
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
            return Vec::new();
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
                    position: Vector2::new(candidate.x as f32 + 0.5, candidate.y as f32 + 0.5),
                    score: candidate.score,
                },
            );

            if bucket.len() > per_cell_cap {
                bucket.pop();
            }
        }

        let mut seeds: Vec<FeatureSeed> = buckets.into_iter().flatten().collect();
        if seeds.is_empty() {
            debug!(
                "FAST seeding in {}x{} (oct {}) → 0 (grid filtered)",
                width, height, level.octave
            );
            return seeds;
        }

        seeds.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut seeds = if nms_radius > 0.0 {
            radius_nms(seeds, nms_radius, max_features)
        } else {
            seeds
        };

        if seeds.len() > max_features {
            seeds.truncate(max_features);
        }

        debug!(
            "FAST seeding in {}x{} (oct {}, cell {}, K {}, NMS {:.1}px) → {} (max {})",
            width,
            height,
            level.octave,
            cell,
            per_cell_cap,
            nms_radius,
            seeds.len(),
            max_features
        );

        seeds
    }
}

#[derive(Debug, Clone)]
struct Candidate {
    x: usize,
    y: usize,
    score: f32,
}

fn compute_circle_offsets(radius: usize) -> [(isize, isize); 16] {
    if radius == 3 {
        return BASE_CIRCLE_OFFSETS;
    }

    let scale = radius as f32 / 3.0;
    let mut offsets = [(0isize, 0isize); 16];
    for (i, &(x, y)) in BASE_CIRCLE_OFFSETS.iter().enumerate() {
        offsets[i] = (
            (x as f32 * scale).round() as isize,
            (y as f32 * scale).round() as isize,
        );
    }
    offsets
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
    let center = image.get(x, y);
    let high = center + threshold;
    let low = center - threshold;

    let mut brighter = 0;
    let mut darker = 0;
    for &idx in &[0usize, 4, 8, 12] {
        let (dx, dy) = offsets[idx];
        let sample = image.get((x as isize + dx) as usize, (y as isize + dy) as usize);
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
        circle_vals[i] = image.get((x as isize + dx) as usize, (y as isize + dy) as usize);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::img::pyramid::{ImageBuffer, build_pyramid};

    #[test]
    fn fast_seeder_detects_checker_corners() {
        let width = 32;
        let height = 32;
        let mut bytes = vec![0u8; width * height * 4];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 4;
                let bright = (x >= 16) ^ (y >= 16);
                let value = if bright { 255 } else { 32 };
                bytes[idx] = value;
                bytes[idx + 1] = value;
                bytes[idx + 2] = value;
                bytes[idx + 3] = 255;
            }
        }
        let gray = ImageBuffer::from_bgra8(&bytes, width, height, width * 4);
        let pyramid = build_pyramid(&gray, 1);
        let level = &pyramid.levels()[0];

        let seeder = FastSeeder::new(FastSeederConfig {
            grid: FeatureGridConfig::default(),
            detector: FastDetectorConfig::default(),
        });

        let seeds = seeder.seed(level);
        assert!(!seeds.is_empty(), "expected FAST seeder to find corners");
        assert!(seeds.len() <= 32);
        assert!(seeds.iter().all(|s| s.score >= 5.0));
        assert!(
            seeds.iter().any(|s| {
                (s.position.x - 16.5).abs() < 1.5 && (s.position.y - 16.5).abs() < 1.5
            })
        );
    }
}
