use crate::img::pyramid::PyramidLevel;
use log::{debug, trace};
use nalgebra::Vector2;

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
    /// For structure tensor M = [∑Iₓ²    ∑IₓIᵧ]
    ///                           [∑IₓIᵧ   ∑Iᵧ² ]
    /// where the sum is over a window around (x, y)
    ///
    /// λ_min = (trace - √(trace² - 4·det)) / 2
    #[inline]
    fn shi_tomasi_response(
        level: &PyramidLevel,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> f32 {
        const WINDOW_SIZE: usize = 3; // 3x3 window
        const HALF_WINDOW: i32 = (WINDOW_SIZE / 2) as i32;

        let mut ixx = 0.0f32;
        let mut iyy = 0.0f32;
        let mut ixy = 0.0f32;

        // Sum structure tensor elements over window
        for dy in -HALF_WINDOW..=HALF_WINDOW {
            for dx in -HALF_WINDOW..=HALF_WINDOW {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;

                // Bounds check
                if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                    continue;
                }

                let gx = level.grad_x.get(nx as usize, ny as usize);
                let gy = level.grad_y.get(nx as usize, ny as usize);

                ixx += gx * gx;
                iyy += gy * gy;
                ixy += gx * gy;
            }
        }

        let trace = ixx + iyy;
        let det = ixx * iyy - ixy * ixy;

        // Discriminant check to avoid numerical issues
        let discriminant = trace * trace - 4.0 * det;
        if discriminant < 0.0 {
            return 0.0;
        }

        // Minimum eigenvalue
        0.5 * (trace - discriminant.sqrt())
    }

    pub fn seed(&self, level: &PyramidLevel) -> Vec<FeatureSeed> {
        let cell = self.config.cell_size.max(4);
        let mut seeds = Vec::new();
        let width = level.image.width();
        let height = level.image.height();

        debug!(
            "Seeding features in {}x{} image (octave {}, cell size {})",
            width, height, level.octave, cell
        );

        let cells_x = width.div_ceil(cell);
        let cells_y = height.div_ceil(cell);
        trace!("Processing {cells_x}x{cells_y} cells");

        for cy in 0..cells_y {
            let y_min = cy * cell;
            let y_max = ((cy + 1) * cell).min(height);
            for cx in 0..cells_x {
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
                    seeds.push(FeatureSeed {
                        level: level.octave,
                        position,
                        score: best_score,
                    });
                }
            }
        }

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
