use crate::img::Pyramid;
use nalgebra::Vector2;

pub mod fast_seed;
pub use fast_seed::{FastDetectorConfig, FastDetectorType, FastSeeder, FastSeederConfig};

pub mod shi_tom_seed;
pub use shi_tom_seed::{ShiTomasiGridConfig, ShiTomasiSeeder};

pub mod utils;

#[derive(Debug, Clone, Copy)]
pub struct FeatureGridConfig {
    pub cell_size: usize,
    pub max_features: usize,
    pub response_threshold: f32,
    pub per_cell_cap: usize,
    pub nms_radius_px: f32,
    pub window_radius: usize,
}

impl Default for FeatureGridConfig {
    fn default() -> Self {
        Self {
            cell_size: 24,
            max_features: 1000,
            response_threshold: 10.0,
            per_cell_cap: 100,
            nms_radius_px: 16.0,
            window_radius: 10,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FeatureSeed {
    pub level: usize,
    pub level_scale: f32,
    pub px_uv: Vector2<f32>,
    pub score: f32,
}

pub trait FeatureSeederTrait {
    fn seed(&self, pyramid: &Pyramid) -> Vec<FeatureSeed>;
}
