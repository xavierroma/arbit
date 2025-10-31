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

pub struct LKTracker {
    config: LucasKanadeConfig,
}

impl LKTracker {
    pub fn new(config: LucasKanadeConfig) -> Self {
        Self { config }
    }

    /// Track with an optional rotation prior from IMU.
    pub fn track(
        &self,
        prev_level: &PyramidLevel,
        curr_level: &PyramidLevel,
        initial_px_uv: Vector2<f32>,
        _rotation_prior: Option<&SO3>,
        _intrinsics: Option<&CameraIntrinsics>,
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

    /// Refines the position of a feature from prev_level to curr_level using Lucas-Kanade.
    ///
    /// This implements the inverse compositional Lucas-Kanade algorithm:
    /// 1. Extract template from prev_level centered at template_px_uv
    /// 2. Iteratively refine guess to minimize photometric error between template and curr_level
    /// 3. Solve a 2×2 linear system (Hessian) at each iteration to find the optical flow
    ///
    /// Returns (iterations, residual) on success, None if tracking fails.
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

        // Early exit if tracking window doesn't fit in images
        if !window_within_bounds(template_px_uv, radius, prev_width, prev_height) {
            return None;
        }
        if !window_within_bounds(*guess, radius, curr_width, curr_height) {
            return None;
        }

        let mut iterations = 0u32;
        let mut residual = 0.0f32;
        let convergence_threshold_sq = self.config.epsilon * self.config.epsilon;

        // Gauss-Newton iterations to refine the guess
        while (iterations as usize) < self.config.max_iterations {
            iterations += 1;

            // Accumulate Hessian (H) and steepest descent image (b)
            // We're solving: H * delta = -b
            let mut hessian_xx = 0.0f32;
            let mut hessian_xy = 0.0f32;
            let mut hessian_yy = 0.0f32;
            let mut steepest_x = 0.0f32;
            let mut steepest_y = 0.0f32;
            let mut error_sum_sq = 0.0f32;
            let mut sample_count = 0.0f32;

            // Loop over window around template position
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let offset = Vector2::new(dx as f32, dy as f32);
                    let template_pos = template_px_uv + offset;
                    let target_pos = *guess + offset;

                    // Skip pixels outside image bounds
                    if !point_within_bounds(template_pos, prev_width, prev_height)
                        || !point_within_bounds(target_pos, curr_width, curr_height)
                    {
                        continue;
                    }

                    // Sample intensities and gradients
                    let intensity_prev = prev_level
                        .image
                        .get_pixel(template_pos.x as u32, template_pos.y as u32)
                        .0[0];
                    let intensity_curr = curr_level
                        .image
                        .get_pixel(target_pos.x as u32, target_pos.y as u32)
                        .0[0];
                    let grad_x = prev_level
                        .grad_x
                        .get_pixel(template_pos.x as u32, template_pos.y as u32)
                        .0[0];
                    let grad_y = prev_level
                        .grad_y
                        .get_pixel(template_pos.x as u32, template_pos.y as u32)
                        .0[0];

                    // Photometric error: difference in intensities
                    let photometric_error = intensity_curr - intensity_prev;

                    // Accumulate Hessian: H = Σ(∇I * ∇I^T)
                    let gx = grad_x as f32;
                    let gy = grad_y as f32;
                    hessian_xx += gx * gx;
                    hessian_xy += gx * gy;
                    hessian_yy += gy * gy;

                    // Accumulate steepest descent: b = Σ(∇I * error)
                    steepest_x += gx * photometric_error as f32;
                    steepest_y += gy * photometric_error as f32;

                    // Track error for residual calculation
                    error_sum_sq += photometric_error as f32 * photometric_error as f32;
                    sample_count += 1.0;
                }
            }

            // Regularize Hessian for numerical stability (Levenberg-Marquardt damping)
            let damping = 1e-4f32;
            hessian_xx += damping;
            hessian_yy += damping;

            // Invert 2×2 Hessian matrix
            let determinant = hessian_xx * hessian_yy - hessian_xy * hessian_xy;
            if determinant.abs() < 1e-6 {
                // Matrix is singular, tracking fails
                return None;
            }

            let inv_det = 1.0 / determinant;
            let h_inv_00 = hessian_yy * inv_det;
            let h_inv_01 = -hessian_xy * inv_det;
            let h_inv_11 = hessian_xx * inv_det;

            // Solve for delta: delta = -H^(-1) * b
            let delta_x = -(h_inv_00 * steepest_x + h_inv_01 * steepest_y);
            let delta_y = -(h_inv_01 * steepest_x + h_inv_11 * steepest_y);
            let delta = Vector2::new(delta_x, delta_y);

            // Update guess position
            *guess += delta;

            if !window_within_bounds(*guess, radius, curr_width, curr_height) {
                return None;
            }

            // Check for convergence: if delta is small enough, we're done
            if delta.norm_squared() <= convergence_threshold_sq {
                residual = if sample_count > 0.0 {
                    (error_sum_sq / sample_count).sqrt()
                } else {
                    0.0
                };
                return Some((iterations, residual));
            }
        }

        // Max iterations reached without convergence
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
