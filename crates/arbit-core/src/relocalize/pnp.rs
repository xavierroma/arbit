use crate::math::se3::{SE3, TransformSE3, Twist};
use log::{debug, warn};
use nalgebra::{
    Matrix2x3, Matrix2x6, Matrix3, Matrix3x4, Point3, Rotation3, SMatrix, SVector, Translation3,
    Vector2, Vector3,
};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::index::sample;

#[derive(Debug, Clone, Copy)]
pub struct PnPObservation {
    pub world_xyz: Point3<f64>,
    pub norm_xy: Vector2<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct PnPRansacParams {
    pub iterations: usize,
    pub threshold: f64,
    pub min_inliers: usize,
}

impl Default for PnPRansacParams {
    fn default() -> Self {
        Self {
            iterations: 256,
            threshold: 1e-2,
            min_inliers: 12,
        }
    }
}

/// Result of PnP estimation.
///
/// pose_cw: world→camera. This transform maps world coordinates into the camera frame,
/// consistent with projection model x = K [R|t] X_world.
#[derive(Debug, Clone)]
pub struct PnPResult {
    /// pose_cw: world→camera. Maps world coordinates into camera coordinates.
    pub pose_cw: TransformSE3,
    pub inliers: Vec<usize>,
    pub average_reprojection_error: f64,
}

pub struct PnPRansac {
    params: PnPRansacParams,
}

impl PnPRansac {
    pub fn new(params: PnPRansacParams) -> Self {
        Self { params }
    }

    /// Estimate pose_cw (world→camera) from 2D-3D correspondences.
    ///
    /// Returns pose_cw: world→camera (maps world coords into camera coords).
    pub fn estimate(&self, observations: &[PnPObservation]) -> Option<PnPResult> {
        debug!(target: "arbit_core::pnp", "Starting PnP estimation with {} observations", observations.len());

        if observations.len() < 6 {
            warn!(target: "arbit_core::pnp", "Insufficient observations for PnP: {} < 6", observations.len());
            return None;
        }

        let mut rng = SmallRng::from_entropy();
        let mut best_inliers = Vec::new();
        let mut best_pose = None;

        let sample_size = observations.len().min(12).max(6);
        debug!(target: "arbit_core::pnp", "Using sample size {} for RANSAC", sample_size);

        for _ in 0..self.params.iterations {
            let sample_indices = sample(&mut rng, observations.len(), sample_size);
            let sample: Vec<_> = sample_indices
                .into_iter()
                .map(|i| observations[i])
                .collect();
            let Some(candidate) = solve_pnp_linear(&sample) else {
                continue;
            };

            let inliers = collect_inliers(
                observations,
                &candidate,
                self.params.threshold,
                self.params.min_inliers,
            );
            if inliers.len() > best_inliers.len() {
                best_inliers = inliers;
                best_pose = Some(candidate);
            }
        }

        if best_inliers.len() < self.params.min_inliers {
            return None;
        }

        let mut pose_cw = best_pose?;
        let inlier_observations: Vec<_> = best_inliers.iter().map(|&i| observations[i]).collect();

        // Try linear refinement with a subset (avoid large matrices)
        let linear_subset_size = 12.min(inlier_observations.len());
        let linear_subset: Vec<_> = inlier_observations
            .iter()
            .take(linear_subset_size)
            .cloned()
            .collect();
        if let Some(refined) = solve_pnp_linear(&linear_subset) {
            pose_cw = refined;
        }

        // Use nonlinear refinement with a subset of inliers for stability
        let subset_size = 20.min(inlier_observations.len());
        let subset: Vec<_> = inlier_observations
            .iter()
            .take(subset_size)
            .cloned()
            .collect();
        if let Some(refined) = refine_pose(pose_cw, &subset, 8) {
            pose_cw = refined;
        }

        let (avg_error, inliers) = evaluate_pose(
            observations,
            &pose_cw,
            self.params.threshold,
            self.params.min_inliers,
        );

        let inlier_count = inliers.len();
        let inlier_ratio = inlier_count as f64 / observations.len() as f64;

        debug!(target: "arbit_core::pnp", "PnP estimation completed: {} inliers ({:.1}%), avg error: {:.4}",
               inlier_count, inlier_ratio * 100.0, avg_error);

        if inlier_ratio < 0.5 {
            warn!(target: "arbit_core::pnp", "Low inlier ratio: {:.1}% may indicate poor pose estimate", inlier_ratio * 100.0);
        }

        Some(PnPResult {
            pose_cw,
            inliers,
            average_reprojection_error: avg_error,
        })
    }
}

fn collect_inliers(
    observations: &[PnPObservation],
    pose_cw: &TransformSE3,
    threshold: f64,
    min_required: usize,
) -> Vec<usize> {
    let mut errors: Vec<(f64, usize)> = observations
        .iter()
        .enumerate()
        .map(|(idx, obs)| (reprojection_error(pose_cw, obs), idx))
        .collect();

    errors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut inliers: Vec<usize> = errors
        .iter()
        .filter(|(err, _)| *err < threshold)
        .map(|(_, idx)| *idx)
        .collect();

    if inliers.len() < min_required {
        inliers = errors
            .into_iter()
            .take(min_required.min(observations.len()))
            .map(|(_, idx)| idx)
            .collect();
    }

    inliers
}

fn evaluate_pose(
    observations: &[PnPObservation],
    pose_cw: &TransformSE3,
    threshold: f64,
    min_required: usize,
) -> (f64, Vec<usize>) {
    let mut scored: Vec<(f64, usize)> = observations
        .iter()
        .enumerate()
        .map(|(idx, obs)| (reprojection_error(pose_cw, obs), idx))
        .collect();

    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut inliers: Vec<usize> = scored
        .iter()
        .filter(|(err, _)| *err < threshold)
        .map(|(_, idx)| *idx)
        .collect();

    if inliers.len() < min_required {
        inliers = scored
            .iter()
            .take(min_required.min(scored.len()))
            .map(|(_, idx)| *idx)
            .collect();
    }

    use std::collections::HashSet;
    let inlier_set: HashSet<usize> = inliers.iter().copied().collect();

    let average = if inlier_set.is_empty() {
        f64::MAX
    } else {
        let mut sum = 0.0;
        let mut count = 0.0;
        for (err, idx) in &scored {
            if inlier_set.contains(idx) {
                sum += *err;
                count += 1.0;
            }
        }
        if count > 0.0 { sum / count } else { f64::MAX }
    };

    (average, inliers)
}

/// Compute reprojection error for a given pose_cw mapping world→camera.
fn reprojection_error(pose_cw: &TransformSE3, obs: &PnPObservation) -> f64 {
    let cam_xyz = pose_cw.transform_point(&obs.world_xyz);
    if cam_xyz.z <= 0.0 {
        return f64::MAX;
    }
    let projected = Vector2::new(cam_xyz.x / cam_xyz.z, cam_xyz.y / cam_xyz.z);
    (projected - obs.norm_xy).norm()
}

fn solve_pnp_linear(observations: &[PnPObservation]) -> Option<TransformSE3> {
    if observations.len() < 6 {
        return None;
    }

    let mut a = nalgebra::DMatrix::<f64>::zeros(observations.len() * 2, 12);
    for (i, obs) in observations.iter().enumerate() {
        let p = obs.world_xyz.coords;
        let x = obs.norm_xy.x;
        let y = obs.norm_xy.y;
        let row = i * 2;

        a[(row, 0)] = p.x;
        a[(row, 1)] = p.y;
        a[(row, 2)] = p.z;
        a[(row, 3)] = 1.0;
        a[(row, 8)] = -x * p.x;
        a[(row, 9)] = -x * p.y;
        a[(row, 10)] = -x * p.z;
        a[(row, 11)] = -x;

        a[(row + 1, 4)] = p.x;
        a[(row + 1, 5)] = p.y;
        a[(row + 1, 6)] = p.z;
        a[(row + 1, 7)] = 1.0;
        a[(row + 1, 8)] = -y * p.x;
        a[(row + 1, 9)] = -y * p.y;
        a[(row + 1, 10)] = -y * p.z;
        a[(row + 1, 11)] = -y;
    }

    let svd = nalgebra::SVD::new(a, true, true);
    let v_t = svd.v_t?;
    let mut p_vec = v_t.row(v_t.nrows() - 1).transpose();

    // Normalize the solution vector
    if p_vec[11].abs() > 1e-10 {
        p_vec /= p_vec[11];
    }

    let mut p = Matrix3x4::zeros();
    for row in 0..3 {
        for col in 0..4 {
            p[(row, col)] = p_vec[row * 4 + col];
        }
    }

    let r_temp = p.fixed_view::<3, 3>(0, 0).into_owned();
    let svd_r = nalgebra::SVD::new(r_temp.clone(), true, true);
    let mut u = svd_r.u?;
    let vt = svd_r.v_t?;
    let det = u.determinant() * vt.determinant();
    if det < 0.0 {
        let mut row = u.column_mut(2);
        row *= -1.0;
    }
    let mut r = u * vt;
    let scale = (r_temp.column(0).norm() + r_temp.column(1).norm() + r_temp.column(2).norm()) / 3.0;
    if scale.abs() < 1e-9 {
        return None;
    }
    let mut t = p.column(3) / scale;

    if r.determinant() < 0.0 {
        r = -r;
        t = -t;
    }

    let rotation = Rotation3::from_matrix_unchecked(r);
    let translation = Vector3::new(t[0], t[1], t[2]);

    Some(TransformSE3::from_parts(
        Translation3::from(translation),
        rotation.into(),
    ))
}

/// Nonlinear refinement of pose_cw (world→camera) using Gauss-Newton.
fn refine_pose(
    mut pose_cw: TransformSE3,
    observations: &[PnPObservation],
    iterations: usize,
) -> Option<TransformSE3> {
    if observations.is_empty() {
        return Some(pose_cw);
    }

    for _ in 0..iterations {
        let mut jtj = SMatrix::<f64, 6, 6>::zeros();
        let mut jtr = SVector::<f64, 6>::zeros();
        let mut valid_samples = 0usize;

        for obs in observations {
            let cam_xyz = pose_cw.transform_point(&obs.world_xyz);
            if cam_xyz.z <= 0.0 {
                continue;
            }
            valid_samples += 1;

            let proj = Vector2::new(cam_xyz.x / cam_xyz.z, cam_xyz.y / cam_xyz.z);
            let residual = proj - obs.norm_xy;

            let jac_proj = Matrix2x3::new(
                1.0 / cam_xyz.z,
                0.0,
                -cam_xyz.x / (cam_xyz.z * cam_xyz.z),
                0.0,
                1.0 / cam_xyz.z,
                -cam_xyz.y / (cam_xyz.z * cam_xyz.z),
            );

            let skew = Matrix3::new(
                0.0, -cam_xyz.z, cam_xyz.y, cam_xyz.z, 0.0, -cam_xyz.x, -cam_xyz.y, cam_xyz.x, 0.0,
            );

            let jac_trans = jac_proj;
            let jac_rot = jac_proj * (-skew);

            let mut jac = Matrix2x6::zeros();
            jac.fixed_view_mut::<2, 3>(0, 0).copy_from(&jac_trans);
            jac.fixed_view_mut::<2, 3>(0, 3).copy_from(&jac_rot);

            let jac_t = jac.transpose();
            jtj += jac_t * jac;
            jtr += jac_t * residual;
        }

        if valid_samples < 6 {
            break;
        }

        let rhs = -jtr;
        let Some(delta) = jtj.full_piv_lu().solve(&rhs) else {
            break;
        };

        if delta.norm() < 1e-6 {
            break;
        }

        let twist = Twist::new(
            Vector3::new(delta[3], delta[4], delta[5]),
            Vector3::new(delta[0], delta[1], delta[2]),
        );
        let delta_pose = SE3::exp(&twist).to_isometry();
        pose_cw = delta_pose * pose_cw;
    }

    Some(pose_cw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn pnp_recovers_pose_with_outliers() {
        let true_rotation = Rotation3::from_axis_angle(&Vector3::z_axis(), 0.2);
        let true_translation = Vector3::new(0.1, -0.2, 0.5);

        let mut rng = SmallRng::seed_from_u64(42);
        let mut observations = Vec::new();
        for _ in 0..80 {
            let world = Point3::new(
                rng.gen_range(-0.6..0.6),
                rng.gen_range(-0.6..0.6),
                rng.gen_range(1.5..3.0),
            );
            let cam = true_rotation * world.coords + true_translation;
            let measurement = Vector2::new(cam.x / cam.z, cam.y / cam.z);

            observations.push(PnPObservation {
                world_xyz: world,
                norm_xy: measurement,
            });
        }

        for _ in 0..10 {
            observations.push(PnPObservation {
                world_xyz: Point3::new(
                    rng.gen_range(-1.0..1.0),
                    rng.gen_range(-1.0..1.0),
                    rng.gen_range(0.5..1.0),
                ),
                norm_xy: Vector2::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)),
            });
        }

        let estimator = PnPRansac::new(PnPRansacParams {
            iterations: 400,
            threshold: 0.5,
            min_inliers: 40,
        });

        let result = estimator.estimate(&observations).expect("pnp");
        assert!(!result.inliers.is_empty());
        assert!(result.average_reprojection_error < 0.05);
    }
}
