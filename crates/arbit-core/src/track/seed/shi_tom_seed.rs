use crate::img::pyramid::PyramidLevel;
use crate::track::seed::utils::radius_nms;
use crate::track::seed::{FeatureSeed, FeatureSeederTrait};
use log::debug;
use nalgebra::Vector2;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy)]
pub struct ShiTomasiGridConfig {
    pub cell_size: usize,
    pub max_features: usize,
    pub response_threshold: f32,
    pub per_cell_cap: usize,
    pub nms_radius_px: f32,
    pub window_radius: usize,
}

impl Default for ShiTomasiGridConfig {
    fn default() -> Self {
        Self {
            cell_size: 24,
            max_features: 200,
            response_threshold: 10.0,
            per_cell_cap: 5,
            nms_radius_px: 10.0,
            window_radius: 10,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ShiTomasiSeeder {
    config: ShiTomasiGridConfig,
}

impl ShiTomasiSeeder {
    pub fn new(config: ShiTomasiGridConfig) -> Self {
        Self { config }
    }

    /// Computes Shi-Tomasi corner response (minimum eigenvalue of structure tensor)
    ///
    /// The structure tensor captures the distribution of gradient directions in a local window:
    ///
    /// ```text
    /// M = [ixx  ixy]   where  ixx = ∑Iₓ²,  iyy = ∑Iᵧ²,  ixy = ∑IₓIᵧ
    ///     [ixy  iyy]
    /// ```
    ///
    /// The minimum eigenvalue λ_min = (trace - √(trace² - 4·det)) / 2 indicates:
    /// - **High λ_min**: Strong gradients in all directions → **Corner** 🎯
    /// - **Low λ_min**: Weak gradient in at least one direction → Edge or flat region
    ///
    /// This makes λ_min an excellent corner detector: corners have strong responses
    /// in multiple directions, while edges are strong in only one direction.
    #[inline]
    fn shi_tomasi_response(
        level: &PyramidLevel,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> f32 {
        // Early bounds check - if we can't fit a 3x3 window, return 0
        if x < 1 || y < 1 || x >= width - 1 || y >= height - 1 {
            return 0.0;
        }

        let gx_values: [f32; 9] = [
            level.grad_x.get(x - 1, y - 1),
            level.grad_x.get(x, y - 1),
            level.grad_x.get(x + 1, y - 1),
            level.grad_x.get(x - 1, y),
            level.grad_x.get(x, y),
            level.grad_x.get(x + 1, y),
            level.grad_x.get(x - 1, y + 1),
            level.grad_x.get(x, y + 1),
            level.grad_x.get(x + 1, y + 1),
        ];

        let gy_values: [f32; 9] = [
            level.grad_y.get(x - 1, y - 1),
            level.grad_y.get(x, y - 1),
            level.grad_y.get(x + 1, y - 1),
            level.grad_y.get(x - 1, y),
            level.grad_y.get(x, y),
            level.grad_y.get(x + 1, y),
            level.grad_y.get(x - 1, y + 1),
            level.grad_y.get(x, y + 1),
            level.grad_y.get(x + 1, y + 1),
        ];

        // Iterator pattern that LLVM auto-unrolls and vectorizes
        let (ixx, iyy, ixy) = gx_values
            .iter()
            .zip(gy_values.iter())
            .fold((0.0f32, 0.0f32, 0.0f32), |(ixx, iyy, ixy), (&gx, &gy)| {
                (ixx + gx * gx, iyy + gy * gy, ixy + gx * gy)
            });

        let trace = ixx + iyy;
        let det = ixx * iyy - ixy * ixy;
        let discriminant = trace * trace - 4.0 * det;

        if discriminant < 0.0 {
            return 0.0;
        }

        0.5 * (trace - discriminant.sqrt())
    }
}

impl FeatureSeederTrait for ShiTomasiSeeder {
    fn seed(&self, level: &PyramidLevel) -> Vec<FeatureSeed> {
        let width = level.image.width();
        let height = level.image.height();

        // --- Parameters & guards ---
        let cell = self.config.cell_size.max(4);
        let per_cell_cap = self.config.per_cell_cap.max(1);
        let max_features = self.config.max_features.max(per_cell_cap);
        let thr = self.config.response_threshold.max(0.0);
        let nms_radius = self.config.nms_radius_px.max(0.0);
        let r = self.config.window_radius.max(1); // LK patch margin (in L0 pixels)

        // Define a safe ROI so LK windows won’t go OOB later
        if width <= 2 * r || height <= 2 * r {
            return Vec::new();
        }
        let x_lo = r;
        let x_hi = width - r;
        let y_lo = r;
        let y_hi = height - r;

        // integer ceil-div without nightly/unstable methods
        let cells_x = ((x_hi - x_lo) + cell - 1) / cell;
        let cells_y = ((y_hi - y_lo) + cell - 1) / cell;

        // --- 1) Per-cell top-K (parallel over rows of cells) ---
        // We avoid "nested parallel iterators" by parallelizing only outer y-rows,
        // and doing the inner x-loop sequentially. Then we reduce Vecs.
        let seeds: Vec<FeatureSeed> = (0..cells_y)
            .into_par_iter()
            .map(|cy| {
                let mut row_seeds: Vec<FeatureSeed> = Vec::with_capacity(cells_x * per_cell_cap);

                let y_min = y_lo + cy * cell;
                let y_max = (y_min + cell).min(y_hi);

                for cx in 0..cells_x {
                    let x_min = x_lo + cx * cell;
                    let x_max = (x_min + cell).min(x_hi);

                    // Tiny sorted list of best corners in this cell (desc by score)
                    let mut best: Vec<(f32, Vector2<f32>)> = Vec::with_capacity(per_cell_cap);

                    for y in y_min..y_max {
                        for x in x_min..x_max {
                            let score = Self::shi_tomasi_response(level, x, y, width, height);
                            if score < thr {
                                continue;
                            }

                            let pos = Vector2::new(x as f32 + 0.5, y as f32 + 0.5);

                            // Insert by score (desc), keep size <= per_cell_cap
                            let mut i = 0;
                            while i < best.len() && best[i].0 > score {
                                i += 1;
                            }
                            best.insert(i, (score, pos));
                            if best.len() > per_cell_cap {
                                best.pop();
                            }
                        }
                    }

                    for (score, position) in best.into_iter() {
                        row_seeds.push(FeatureSeed {
                            level: level.octave,
                            position,
                            score,
                        });
                    }
                }

                row_seeds
            })
            .reduce(
                || Vec::new(),
                |mut a, mut b| {
                    a.append(&mut b);
                    a
                },
            );

        // Early out if nothing scored above threshold
        if seeds.is_empty() {
            debug!(
                "Seeding features in {}x{} (oct {}, cell {}, per_cell {}) → 0",
                width, height, level.octave, cell, per_cell_cap
            );
            return seeds;
        }

        // --- 2) Global sort (desc) + radius-NMS across cell boundaries ---
        let mut seeds = seeds;
        seeds.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        if nms_radius > 0.0 {
            seeds = radius_nms(seeds, nms_radius, max_features);
        }

        // --- 3) Final cap ---
        if seeds.len() > max_features {
            seeds.truncate(max_features);
        }

        debug!(
            "Seeding features in {}x{} (oct {}, cell {}, K {}, NMS {:.1}px) → {} (max {})",
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::img::pyramid::{ImageBuffer, build_pyramid};

    #[test]
    fn seeder_picks_strong_gradients() {
        let width = 32;
        let height = 32;
        let mut bytes = vec![0u8; width * height * 4];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 4;
                let value = ((x + y) % 255) as u8;
                bytes[idx] = value;
                bytes[idx + 1] = value;
                bytes[idx + 2] = value;
                bytes[idx + 3] = 255;
            }
        }
        let gray = ImageBuffer::from_bgra8(&bytes, width, height, width * 4);
        let pyramid = build_pyramid(&gray, 1);
        let level = &pyramid.levels()[0];

        let seeder = ShiTomasiSeeder::new(ShiTomasiGridConfig {
            cell_size: 8,
            max_features: 16,
            response_threshold: 1.0,
            per_cell_cap: 10,
            nms_radius_px: 10.0,
            window_radius: 1,
        });

        let seeds = seeder.seed(level);
        assert!(!seeds.is_empty());
        assert!(seeds.len() <= 16);
        for seed in seeds {
            assert!(seed.score >= 1.0);
        }
    }
}
