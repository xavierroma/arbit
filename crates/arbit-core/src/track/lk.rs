use crate::img::pyramid::{Pyramid, PyramidLevel};
use log::{debug, trace};
use nalgebra::Vector2;

#[derive(Debug, Clone, Copy)]
pub struct LucasKanadeConfig {
    pub window_radius: usize,
    pub max_iterations: usize,
    pub epsilon: f32,
}

impl Default for LucasKanadeConfig {
    fn default() -> Self {
        Self {
            window_radius: 4,
            max_iterations: 20,
            epsilon: 0.01,
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
    pub initial: Vector2<f32>,
    pub refined: Vector2<f32>,
    pub iterations: u32,
    pub residual: f32,
    pub outcome: TrackOutcome,
}

pub struct Tracker {
    config: LucasKanadeConfig,
}

impl Tracker {
    pub fn new(config: LucasKanadeConfig) -> Self {
        Self { config }
    }

    pub fn track(
        &self,
        prev: &Pyramid,
        curr: &Pyramid,
        initial_position: Vector2<f32>,
    ) -> TrackObservation {
        trace!(
            "Tracking feature from {:?} across {} pyramid levels",
            initial_position,
            prev.levels().len()
        );

        let mut current_base = initial_position;
        let mut total_iterations = 0u32;
        let mut residual = 0.0f32;

        let levels = prev.levels();
        if levels.is_empty() {
            debug!("No pyramid levels available for tracking");
            return TrackObservation {
                initial: initial_position,
                refined: initial_position,
                iterations: 0,
                residual: 0.0,
                outcome: TrackOutcome::Diverged,
            };
        }

        for level_idx in (0..levels.len()).rev() {
            let prev_level = &prev.levels()[level_idx];
            let curr_level = &curr.levels()[level_idx];
            let template_pos = scale_to_level(initial_position, prev_level);
            let mut guess = scale_to_level(current_base, curr_level);

            match self.track_level(prev_level, curr_level, template_pos, &mut guess) {
                Some((iters, level_residual)) => {
                    total_iterations += iters;
                    residual = level_residual;
                    current_base = guess / curr_level.scale;
                }
                None => {
                    return TrackObservation {
                        initial: initial_position,
                        refined: current_base,
                        iterations: total_iterations,
                        residual,
                        outcome: TrackOutcome::Diverged,
                    };
                }
            }
        }

        let result = TrackObservation {
            initial: initial_position,
            refined: current_base,
            iterations: total_iterations,
            residual,
            outcome: TrackOutcome::Converged,
        };

        debug!(
            "Tracking completed: {:?} -> {:?} ({} iterations, residual: {:.4})",
            initial_position, result.refined, total_iterations, residual
        );

        result
    }

    fn track_level(
        &self,
        prev_level: &PyramidLevel,
        curr_level: &PyramidLevel,
        template_pos: Vector2<f32>,
        guess: &mut Vector2<f32>,
    ) -> Option<(u32, f32)> {
        let radius = self.config.window_radius as isize;
        let prev_width = prev_level.image.width() as isize;
        let prev_height = prev_level.image.height() as isize;
        let curr_width = curr_level.image.width() as isize;
        let curr_height = curr_level.image.height() as isize;

        if !window_within_bounds(template_pos, radius, prev_width, prev_height) {
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
                    let template = template_pos + offset;
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

    #[test]
    fn tracker_recovers_translation() {
        let shift = (2.2f32, -3.4f32);
        let (frame_a, frame_b) = synthetic_translation(64, 64, shift);
        let pyr_a = build_pyramid(&frame_a, 3);
        let pyr_b = build_pyramid(&frame_b, 3);
        let tracker = Tracker::new(LucasKanadeConfig {
            window_radius: 3,
            max_iterations: 30,
            epsilon: 0.001,
        });
        let initial = Vector2::new(30.5, 30.5);
        let observation = tracker.track(&pyr_a, &pyr_b, initial);
        assert_eq!(observation.outcome, TrackOutcome::Converged);
        let expected = initial + Vector2::new(shift.0, shift.1);
        let error = (observation.refined - expected).norm();
        assert!(error < 0.5);
    }
}
