use crate::img::pyramid::PyramidLevel;
use log::{debug, trace};
use nalgebra::Vector2;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy)]
pub struct FeatureSeed {
    pub level: usize,
    pub position: Vector2<f32>,
    pub score: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct FeatureGridConfig {
    pub cell_size: usize,
    pub max_features: usize,
    pub response_threshold: f32,
}

impl Default for FeatureGridConfig {
    fn default() -> Self {
        Self {
            cell_size: 24,
            max_features: 200,
            response_threshold: 10.0, // Adjusted for Shi-Tomasi scale
        }
    }
}

pub struct FeatureSeeder {
    config: FeatureGridConfig,
}

impl FeatureSeeder {
    pub fn new(config: FeatureGridConfig) -> Self {
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

    pub fn seed(&self, level: &PyramidLevel) -> Vec<FeatureSeed> {
        let cell = self.config.cell_size.max(4);
        let width = level.image.width();
        let height = level.image.height();

        debug!(
            "Seeding features in {}x{} image (octave {}, cell size {})",
            width, height, level.octave, cell
        );

        let cells_x = width.div_ceil(cell);
        let cells_y = height.div_ceil(cell);
        trace!("Processing {cells_x}x{cells_y} cells");

        let mut seeds: Vec<FeatureSeed> = (0..cells_y)
            .into_par_iter()
            .flat_map(|cy| {
                let y_min = cy * cell;
                let y_max = ((cy + 1) * cell).min(height);
                (0..cells_x).into_par_iter().filter_map(move |cx| {
                    let x_min = cx * cell;
                    let x_max = ((cx + 1) * cell).min(width);
                    let mut best_score = 0.0f32;
                    let mut best_position = None;

                    for y in y_min..y_max {
                        for x in x_min..x_max {
                            let score = Self::shi_tomasi_response(level, x, y, width, height);
                            if score > best_score {
                                best_score = score;
                                best_position = Some(Vector2::new(x as f32 + 0.5, y as f32 + 0.5));
                            }
                        }
                    }

                    if let Some(position) = best_position
                        && best_score >= self.config.response_threshold
                    {
                        Some(FeatureSeed {
                            level: level.octave,
                            position,
                            score: best_score,
                        })
                    } else {
                        None
                    }
                })
            })
            .collect();

        seeds.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        seeds.truncate(self.config.max_features);

        debug!(
            "Found {} feature seeds (max: {})",
            seeds.len(),
            self.config.max_features
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

        let seeder = FeatureSeeder::new(FeatureGridConfig {
            cell_size: 8,
            max_features: 16,
            response_threshold: 1.0,
        });

        let seeds = seeder.seed(level);
        assert!(!seeds.is_empty());
        assert!(seeds.len() <= 16);
        for seed in seeds {
            assert!(seed.score >= 1.0);
        }
    }
}
