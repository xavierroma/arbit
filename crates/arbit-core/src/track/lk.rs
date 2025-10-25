use crate::img::pyramid::PyramidLevel;
use crate::math::{CameraIntrinsics, SO3};
use nalgebra::Vector2;
use tracing::{debug_span, error};

#[derive(Debug, Clone, Copy)]
pub struct LucasKanadeConfig {
    pub window_radius: usize,
    pub max_iterations: usize,
    pub epsilon: f32,
}

impl Default for LucasKanadeConfig {
    fn default() -> Self {
        Self {
            window_radius: 10,
            max_iterations: 30,
            epsilon: 0.03162, // ε² ≈ 1e-3
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrackOutcome {
    Converged,
    Diverged,
    OutOfBounds,
}

#[derive(Debug, Clone, Copy)]
pub struct TrackObservation {
    /// The initial position of the feature in the previous frame.
    pub initial_px_uv: Vector2<f32>,
    /// The refined position of the feature in the current frame.
    pub refined_px_uv: Vector2<f32>,
    /// The scale of the pyramid level at which the feature was tracked.
    pub level_scale: f32,
    /// The number of iterations used to refine the position.
    pub iterations: u32,
    /// The residual error of the refined position. Computed as the square root of the mean squared error.
    /// The smaller the residual, the better the tracking.
    pub residual: f32,
    /// The outcome of the tracking: converged, diverged, or out of bounds (not in the image).
    pub outcome: TrackOutcome,
    /// Forward-backward tracking error for health checks (pixels).
    pub fb_err: f32,
    /// Optional persistent track identifier. `None` for unpromoted seeds.
    pub id: Option<u64>,
    /// Normalized corner strength used when seeding.
    pub score: f32,
}

impl TrackObservation {
    /// Returns a stable identifier for this observation, falling back to an index-derived
    /// value when a persistent track ID is not yet available.
    pub fn identifier(&self, fallback_index: usize) -> u64 {
        self.id.unwrap_or((fallback_index as u64) | (1u64 << 63))
    }
}

pub struct Tracker {
    config: LucasKanadeConfig,
}

impl Tracker {
    pub fn new(config: LucasKanadeConfig) -> Self {
        Self { config }
    }

    /// Track with an optional rotation prior from IMU.
    pub fn track(
        &self,
        prev_level: &PyramidLevel,
        curr_level: &PyramidLevel,
        initial_px_uv: Vector2<f32>,
        rotation_prior: Option<&SO3>,
        intrinsics: Option<&CameraIntrinsics>,
    ) -> TrackObservation {
        let _span = debug_span!("track").entered();

        if prev_level.scale != curr_level.scale {
            error!("Previous and current pyramid levels must have the same scale");
            return TrackObservation {
                initial_px_uv,
                refined_px_uv: initial_px_uv,
                iterations: 0,
                residual: 0.0,
                outcome: TrackOutcome::Diverged,
                fb_err: 0.0,
                id: None,
                score: 0.0,
                level_scale: prev_level.scale,
            };
        }

        // Apply rotation prior to improve initial guess if available
        let mut current_base_px_uv = initial_px_uv;

        let mut total_iterations = 0u32;
        let mut residual = 0.0f32;

        let template_px_uv = scale_to_level(initial_px_uv, prev_level);
        let mut guess_px_uv = scale_to_level(current_base_px_uv, curr_level);

        match self.track_level(prev_level, curr_level, template_px_uv, &mut guess_px_uv) {
            Some((iters, level_residual)) => {
                total_iterations += iters;
                residual = level_residual;
                current_base_px_uv = guess_px_uv / curr_level.scale;
            }
            None => {
                return TrackObservation {
                    initial_px_uv,
                    refined_px_uv: current_base_px_uv,
                    iterations: total_iterations,
                    residual,
                    outcome: TrackOutcome::Diverged,
                    fb_err: 0.0,
                    id: None,
                    score: 0.0,
                    level_scale: prev_level.scale,
                };
            }
        }

        let result = TrackObservation {
            initial_px_uv,
            refined_px_uv: current_base_px_uv,
            iterations: total_iterations,
            residual,
            outcome: TrackOutcome::Converged,
            fb_err: 0.0,
            id: None,
            score: 0.0,
            level_scale: curr_level.scale,
        };

        result
    }

    fn track_level(
        &self,
        prev_level: &PyramidLevel,
        curr_level: &PyramidLevel,
        template_px_uv: Vector2<f32>,
        guess: &mut Vector2<f32>,
    ) -> Option<(u32, f32)> {
        let radius = self.config.window_radius as isize;
        let prev_width = prev_level.image.width() as isize;
        let prev_height = prev_level.image.height() as isize;
        let curr_width = curr_level.image.width() as isize;
        let curr_height = curr_level.image.height() as isize;

        if !window_within_bounds(template_px_uv, radius, prev_width, prev_height) {
            return None;
        }
        if !window_within_bounds(*guess, radius, curr_width, curr_height) {
            return None;
        }

        let mut iterations = 0u32;
        let mut residual = 0.0f32;
        let eps2 = self.config.epsilon * self.config.epsilon;

        while (iterations as usize) < self.config.max_iterations {
            iterations += 1;
            let mut gxx = 0.0f32;
            let mut gxy = 0.0f32;
            let mut gyy = 0.0f32;
            let mut bx = 0.0f32;
            let mut by = 0.0f32;
            let mut error_accum = 0.0f32;
            let mut sample_count = 0.0f32;

            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let offset = Vector2::new(dx as f32, dy as f32);
                    let template = template_px_uv + offset;
                    let target = *guess + offset;

                    if !point_within_bounds(template, prev_width, prev_height)
                        || !point_within_bounds(target, curr_width, curr_height)
                    {
                        continue;
                    }

                    let i0 = prev_level.image.sample(template.x, template.y);
                    let i1 = curr_level.image.sample(target.x, target.y);
                    let gx = prev_level.grad_x.sample(template.x, template.y);
                    let gy = prev_level.grad_y.sample(template.x, template.y);
                    let error = i1 - i0;

                    gxx += gx * gx;
                    gxy += gx * gy;
                    gyy += gy * gy;
                    bx += gx * error;
                    by += gy * error;
                    error_accum += error * error;
                    sample_count += 1.0;
                }
            }

            let lambda = 1e-4f32;
            let gxx = gxx + lambda;
            let gyy = gyy + lambda;
            let det = gxx * gyy - gxy * gxy;
            if det.abs() < 1e-6 {
                return None;
            }

            let inv00 = gyy / det;
            let inv01 = -gxy / det;
            let inv11 = gxx / det;

            let delta_x = -(inv00 * bx + inv01 * by);
            let delta_y = -(inv01 * bx + inv11 * by);
            let delta = Vector2::new(delta_x, delta_y);
            *guess += delta;

            if delta.dot(&delta) <= eps2 {
                residual = if sample_count > 0.0 {
                    (error_accum / sample_count).sqrt()
                } else {
                    0.0
                };
                return Some((iterations, residual));
            }
        }

        Some((iterations, residual))
    }
}

fn scale_to_level(position: Vector2<f32>, level: &PyramidLevel) -> Vector2<f32> {
    position * level.scale
}

fn window_within_bounds(
    position: Vector2<f32>,
    radius: isize,
    width: isize,
    height: isize,
) -> bool {
    let x = position.x as isize;
    let y = position.y as isize;
    x - radius >= 1 && x + radius < width - 1 && y - radius >= 1 && y + radius < height - 1
}

fn point_within_bounds(position: Vector2<f32>, width: isize, height: isize) -> bool {
    position.x >= 0.0
        && position.y >= 0.0
        && position.x < (width - 1) as f32
        && position.y < (height - 1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::img::pyramid::{ImageBuffer, build_pyramid};

    fn synthetic_translation(
        width: usize,
        height: usize,
        shift: (f32, f32),
    ) -> (ImageBuffer, ImageBuffer) {
        fn pattern(x: f32, y: f32) -> f32 {
            let a = (x * 0.15).sin();
            let b = (y * 0.12).cos();
            let c = (x * 0.05 + y * 0.09).sin();
            0.5 + 0.25 * a + 0.25 * b + 0.2 * c
        }

        let mut base1 = vec![0u8; width * height * 4];
        let mut base2 = vec![0u8; width * height * 4];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 4;
                let value_a = (pattern(x as f32, y as f32) * 255.0).clamp(0.0, 255.0) as u8;
                base1[idx] = value_a;
                base1[idx + 1] = value_a;
                base1[idx + 2] = value_a;
                base1[idx + 3] = 255;

                let shifted_x = x as f32 - shift.0;
                let shifted_y = y as f32 - shift.1;
                let value_b = (pattern(shifted_x, shifted_y) * 255.0).clamp(0.0, 255.0) as u8;
                base2[idx] = value_b;
                base2[idx + 1] = value_b;
                base2[idx + 2] = value_b;
                base2[idx + 3] = 255;
            }
        }

        (
            ImageBuffer::from_bgra8(&base1, width, height, width * 4),
            ImageBuffer::from_bgra8(&base2, width, height, width * 4),
        )
    }
}
