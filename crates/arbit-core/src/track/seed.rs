use crate::img::pyramid::PyramidLevel;
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
            response_threshold: 25.0,
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

    pub fn seed(&self, level: &PyramidLevel) -> Vec<FeatureSeed> {
        let cell = self.config.cell_size.max(4);
        let mut seeds = Vec::new();
        let width = level.image.width();
        let height = level.image.height();

        let cells_x = (width + cell - 1) / cell;
        let cells_y = (height + cell - 1) / cell;

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
                        let gx = level.grad_x.get(x, y);
                        let gy = level.grad_y.get(x, y);
                        let score = gx * gx + gy * gy;
                        if score > best_score {
                            best_score = score;
                            best_position = Some(Vector2::new(x as f32 + 0.5, y as f32 + 0.5));
                        }
                    }
                }

                if let Some(position) = best_position {
                    if best_score >= self.config.response_threshold {
                        seeds.push(FeatureSeed {
                            level: level.octave,
                            position,
                            score: best_score,
                        });
                    }
                }
            }
        }

        seeds.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        seeds.truncate(self.config.max_features);
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
