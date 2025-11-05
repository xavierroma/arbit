use log::trace;
use nalgebra::{Matrix3, Matrix3x4, Point2, Point3, Rotation3, SVector, Vector2, Vector3};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::index::sample;
use tracing::info;

#[derive(Debug, Clone, Copy)]
pub struct FeatureMatch {
    //Normalized image coordinates in keyframe A
    pub norm_xy_a: Vector2<f64>,
    // Normalized image coordinates in keyframe B
    pub norm_xy_b: Vector2<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct TwoViewInitializationParams {
    pub ransac_iterations: usize,
    pub ransac_threshold: f64,
    pub ransac_sample_size: usize,
    pub min_matches: usize,
    pub min_parallax: f64,
    /// Enable homography degeneracy test
    pub check_homography: bool,
    /// Minimum ratio of essential inliers to homography inliers (e.g., 0.45)
    pub homography_ratio_threshold: f64,
    /// Enable Levenberg-Marquardt refinement after RANSAC
    pub enable_lm_refinement: bool,
    /// Maximum LM iterations
    pub lm_max_iterations: usize,
}

impl Default for TwoViewInitializationParams {
    fn default() -> Self {
        Self {
            ransac_iterations: 1000,
            ransac_threshold: 2e-4,
            ransac_sample_size: 8,
            min_matches: 80,
            min_parallax: 1.0,
            check_homography: true,
            homography_ratio_threshold: 0.45,
            enable_lm_refinement: true,
            lm_max_iterations: 10,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TwoViewInitialization {
    pub essential: Matrix3<f64>,
    // Rotation from cam1 -> cam2
    pub rotation_c2c1: Rotation3<f64>,
    // Translation from cam1 -> cam2 (in cam1 frame)
    pub translation_c2c1: Vector3<f64>,
    pub inliers: Vec<usize>,
    pub average_sampson_error: f64,
    // 3D points in camera 1 frame (norm_xy, cam_xyz)
    pub landmarks_c1: Vec<(Point2<f64>, Point3<f64>)>,
}

#[derive(Debug, Clone, Copy)]
pub struct DecomposedEssential {
    pub rotation_c2c1: Rotation3<f64>,
    pub translation_c2c1: Vector3<f64>,
}

pub struct TwoViewInitializer {
    params: TwoViewInitializationParams,
}

impl TwoViewInitializer {
    pub fn new(params: TwoViewInitializationParams) -> Self {
        Self { params }
    }

    pub fn estimate(&self, matches: &[FeatureMatch]) -> Option<TwoViewInitialization> {
        info!(
            "Starting two-view initialization with {} feature matches",
            matches.len()
        );

        if matches.len() < self.params.min_matches {
            info!(
                "Insufficient matches for two-view initialization: {} < {}",
                matches.len(),
                self.params.min_matches,
            );
            return None;
        }
        let mut parallax_sum = 0.0;
        let mut parallax_samples = 0usize;

        // Validate input data is not degenerate while accumulating parallax
        for (i, m) in matches.iter().enumerate() {
            if !m.norm_xy_a.x.is_finite()
                || !m.norm_xy_a.y.is_finite()
                || !m.norm_xy_b.x.is_finite()
                || !m.norm_xy_b.y.is_finite()
            {
                info!("Match {} has non-finite coordinates, rejecting", i);
                return None;
            }

            let dir_a = Vector3::new(m.norm_xy_a.x, m.norm_xy_a.y, 1.0);
            let dir_b = Vector3::new(m.norm_xy_b.x, m.norm_xy_b.y, 1.0);
            let norm_a = dir_a.norm();
            let norm_b = dir_b.norm();

            if norm_a <= f64::EPSILON || norm_b <= f64::EPSILON {
                info!("Match {} has near-zero direction norm, rejecting", i);
                return None;
            }

            let cos_theta = (dir_a.dot(&dir_b) / (norm_a * norm_b)).clamp(-1.0, 1.0);
            parallax_sum += cos_theta.acos();
            parallax_samples += 1;
        }

        if parallax_samples == 0 {
            info!("No valid matches available to evaluate parallax, rejecting");
            return None;
        }

        let average_parallax_deg = (parallax_sum / parallax_samples as f64).to_degrees();
        if average_parallax_deg < self.params.min_parallax {
            info!(
                "Tracking not strong enough: average parallax {:.2}° < {:.2}°",
                average_parallax_deg, self.params.min_parallax,
            );
            return None;
        }

        // Check for sufficient spread (not all points at origin)
        let spread_a =
            matches.iter().map(|m| m.norm_xy_a.norm()).sum::<f64>() / matches.len() as f64;
        let spread_b =
            matches.iter().map(|m| m.norm_xy_b.norm()).sum::<f64>() / matches.len() as f64;

        if spread_a < 1e-6 || spread_b < 1e-6 {
            info!(
                "Matches have insufficient spread (a:{:.6}, b:{:.6}), rejecting",
                spread_a, spread_b
            );
            return None;
        }

        info!("Input validation passed, proceeding with RANSAC");

        let mut rng = SmallRng::from_entropy();
        let mut best_inliers = Vec::new();
        info!(
            "Running {} RANSAC iterations",
            self.params.ransac_iterations
        );

        for iter_count in 0..self.params.ransac_iterations {
            let sample_indices = sample(&mut rng, matches.len(), self.params.ransac_sample_size);
            let subset = sample_indices
                .iter()
                .map(|idx| matches[idx])
                .collect::<Vec<FeatureMatch>>();
            let e_candidate = estimate_essential(&subset);
            if !e_candidate.iter().all(|v| v.is_finite()) {
                trace!("RANSAC iteration {iter_count}: non-finite essential matrix, skipping");
                continue;
            }

            let inliers = score_inliers(matches, &e_candidate, self.params.ransac_threshold);
            if inliers.len() > best_inliers.len() {
                best_inliers = inliers;
                info!(
                    "RANSAC iteration {}: new best with {} inliers",
                    iter_count,
                    best_inliers.len()
                );
            }
        }

        if best_inliers.len() < 8 {
            info!(
                "No sufficient inliers found after RANSAC (best: {})",
                best_inliers.len()
            );
            return None;
        }

        // Check for degeneracy (planar scenes, pure rotation, etc.)
        if check_degeneracy(matches, &best_inliers, &self.params) {
            info!("Scene geometry is degenerate, rejecting initialization");
            return None;
        }

        let inlier_matches: Vec<_> = best_inliers.iter().map(|&i| matches[i]).collect();
        info!(
            "Refining essential matrix with {} inliers",
            inlier_matches.len()
        );
        let initial_e = estimate_essential(&inlier_matches);

        // Apply Levenberg-Marquardt refinement
        let refined_e = refine_essential_lm(&initial_e, matches, &best_inliers, &self.params);

        let decomposition = self.choose_pose(&refined_e, &inlier_matches)?;
        let avg_error = inlier_matches
            .iter()
            .map(|m| sampson_error(&refined_e, m))
            .sum::<f64>()
            / (inlier_matches.len() as f64);

        let p1_cam1 = Matrix3x4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        let p2_cam2 = compose_projection(
            &decomposition.rotation_c2c1,
            &decomposition.translation_c2c1,
        );

        let mut landmarks = Vec::with_capacity(best_inliers.len());

        for m in matches {
            if let Some(point_in_cam1) = triangulate(&p1_cam1, &p2_cam2, m) {
                let depth_in_cam1 = point_in_cam1.z;
                let point_in_cam2 = decomposition.rotation_c2c1 * point_in_cam1.coords
                    + decomposition.translation_c2c1;
                let depth_in_cam2 = point_in_cam2.z;
                if depth_in_cam1 > 0.0 && depth_in_cam2 > 0.0 {
                    landmarks.push((Point2::new(m.norm_xy_a.x, m.norm_xy_a.y), point_in_cam1));
                }
            }
        }
        if landmarks.len() < 10 {
            info!(
                "Two-view initialization failed: insufficient triangulations ({}/{})",
                landmarks.len(),
                best_inliers.len()
            );
            return None;
        }
        info!(
            "Two-view initialization successful: {} inliers, avg error: {:.6}",
            landmarks.len(),
            avg_error
        );

        Some(TwoViewInitialization {
            essential: refined_e,
            rotation_c2c1: decomposition.rotation_c2c1,
            translation_c2c1: decomposition.translation_c2c1,
            inliers: best_inliers,
            average_sampson_error: avg_error,
            landmarks_c1: landmarks,
        })
    }

    /// Like the ORB_SLAM method
    /// 1. ecompose the essential matrix into four possible poses
    /// 2. For each pose, triangulate all features
    /// 3. Score the pose based on the number of positive points, the average parallax, and the reprojection error
    /// 4. Sort the poses by score
    /// 5. Check if the best pose is significantly better than the second best, if not return None
    /// 6. Return the best pose (R_c2c1, t_c2c1)
    pub fn choose_pose(
        &self,
        essential: &Matrix3<f64>,
        matches: &[FeatureMatch],
    ) -> Option<DecomposedEssential> {
        let svd = nalgebra::SVD::new(*essential, true, true);
        let mut u = svd.u?;
        let mut vt = svd.v_t?;

        if u.determinant() < 0.0 {
            let mut col2 = u.column_mut(2);
            col2 *= -1.0;
        }
        if vt.determinant() < 0.0 {
            let mut row2 = vt.row_mut(2);
            row2 *= -1.0;
        }

        let w = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let r_c2c1_candidate1 = Rotation3::from_matrix_unchecked(u * w * vt);
        let r_c2c1_candidate2 = Rotation3::from_matrix_unchecked(u * w.transpose() * vt);
        let t_c2c1_direction = u.column(2).into_owned();

        let candidates = [
            (r_c2c1_candidate1, t_c2c1_direction),
            (r_c2c1_candidate1, -t_c2c1_direction),
            (r_c2c1_candidate2, t_c2c1_direction),
            (r_c2c1_candidate2, -t_c2c1_direction),
        ];
        let p_cam1 = Matrix3x4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

        let mut evaluations = Vec::new();

        // Evaluate each candidate thoroughly
        for (r_c2c1_candidate, t_c2c1_candidate) in candidates.iter() {
            let p_cam2 = compose_projection(r_c2c1_candidate, t_c2c1_candidate);

            let mut positive = 0usize;
            let mut parallax_sum = 0.0;
            let mut reproj_error_sum = 0.0;
            let mut valid_points = 0usize;

            for m in matches {
                if let Some(point_in_cam1) = triangulate(&p_cam1, &p_cam2, m) {
                    let depth_in_cam1 = point_in_cam1.z;
                    let point_in_cam2 = r_c2c1_candidate * point_in_cam1.coords + t_c2c1_candidate;
                    let depth_in_cam2 = point_in_cam2.z;

                    if depth_in_cam1 > 0.0 && depth_in_cam2 > 0.0 {
                        positive += 1;

                        // Compute parallax angle for this point
                        let ray1 = Vector3::new(m.norm_xy_a.x, m.norm_xy_a.y, 1.0).normalize();
                        let ray2 = Vector3::new(m.norm_xy_b.x, m.norm_xy_b.y, 1.0).normalize();
                        let cos_parallax = ray1.dot(&ray2).clamp(-1.0, 1.0);
                        parallax_sum += cos_parallax.acos();

                        // Compute reprojection error
                        let reproj1 = Vector2::new(
                            point_in_cam1.x / point_in_cam1.z,
                            point_in_cam1.y / point_in_cam1.z,
                        );
                        let reproj2 = Vector2::new(
                            point_in_cam2.x / point_in_cam2.z,
                            point_in_cam2.y / point_in_cam2.z,
                        );
                        let error1 = (m.norm_xy_a - reproj1).norm();
                        let error2 = (m.norm_xy_b - reproj2).norm();
                        reproj_error_sum += error1 + error2;

                        valid_points += 1;
                    }
                }
            }

            if valid_points > 0 {
                evaluations.push(CandidateEvaluation {
                    rotation_c2c1: *r_c2c1_candidate,
                    translation_c2c1: *t_c2c1_candidate,
                    positive_count: positive,
                    total_parallax: parallax_sum / valid_points as f64,
                    reprojection_error: reproj_error_sum / valid_points as f64,
                });
            }
        }

        if evaluations.is_empty() {
            info!("No valid candidate poses found");
            return None;
        }

        // Sort by positive count (descending), then by reprojection error (ascending)
        evaluations.sort_by(|a, b| {
            b.positive_count.cmp(&a.positive_count).then_with(|| {
                a.reprojection_error
                    .partial_cmp(&b.reprojection_error)
                    .unwrap()
            })
        });

        let best = &evaluations[0];

        // Check if the winner clearly dominates
        let min_positive_count = (matches.len() as f64 * 0.5) as usize; // At least 50% of matches
        let max_reproj_error = 0.05; // Normalized coordinates, so ~5% of image
        let min_parallax_deg = self.params.min_parallax; // Minimum average parallax in degrees

        if best.positive_count < min_positive_count {
            info!(
                "Best candidate has insufficient positive points: {}/{}",
                best.positive_count,
                matches.len()
            );
            return None;
        }

        if best.reprojection_error > max_reproj_error {
            info!(
                "Best candidate has excessive reprojection error: {:.6}",
                best.reprojection_error
            );
            return None;
        }

        if best.total_parallax.to_degrees() < min_parallax_deg {
            info!(
                "Best candidate has insufficient parallax: {:.2}°",
                best.total_parallax.to_degrees()
            );
            return None;
        }

        // Check for ambiguity: second-best should be clearly worse
        if evaluations.len() > 1 {
            let second_best = &evaluations[1];
            let ratio = best.positive_count as f64 / second_best.positive_count.max(1) as f64;

            if ratio < 1.5 {
                info!(
                    "Ambiguous reconstruction: best={} vs second={} (ratio={:.2})",
                    best.positive_count, second_best.positive_count, ratio
                );
                return None;
            }
        }

        info!(
            "Selected pose with {} positive points, {:.2}° parallax, {:.6} reproj error",
            best.positive_count,
            best.total_parallax.to_degrees(),
            best.reprojection_error
        );

        Some(DecomposedEssential {
            rotation_c2c1: best.rotation_c2c1,
            translation_c2c1: best.translation_c2c1.normalize(),
        })
    }
}

impl TwoViewInitialization {
    pub fn scaled(&self, scale: f64) -> Self {
        let mut scaled = self.clone();
        scaled.translation_c2c1 *= scale;
        scaled.landmarks_c1 = self
            .landmarks_c1
            .iter()
            .map(|(index, point)| {
                (
                    *index,
                    Point3::new(point.x * scale, point.y * scale, point.z * scale),
                )
            })
            .collect();
        scaled
    }
    pub fn rotate_world_orientation(&self, rotation: &Rotation3<f64>) -> Self {
        let mut rotated = self.clone();
        rotated.rotation_c2c1 = rotation * self.rotation_c2c1;
        rotated.translation_c2c1 = rotation * self.translation_c2c1;
        rotated.landmarks_c1 = self
            .landmarks_c1
            .iter()
            .map(|(index, point)| (*index, rotation * point))
            .collect();
        rotated
    }
}

/// Estimate the essential matrix from feature matches.
/// Returns R_c2c1 and t_c2c1.
fn estimate_essential(matches: &[FeatureMatch]) -> Matrix3<f64> {
    let (points_a, t_a) = normalize_points(matches.iter().map(|m| m.norm_xy_a).collect());
    let (points_b, t_b) = normalize_points(matches.iter().map(|m| m.norm_xy_b).collect());

    // Build constraint matrix A where each row encodes the epipolar constraint: x2^T * E * x1 = 0
    // Expanding: [x2, y2, 1]^T * E * [x1, y1, 1] = 0
    // Results in: [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1] * vec(E) = 0
    let mut a = nalgebra::DMatrix::<f64>::zeros(matches.len(), 9);
    for (row, (pa, pb)) in points_a.iter().zip(points_b.iter()).enumerate() {
        let x1 = pa.x;
        let y1 = pa.y;
        let x2 = pb.x;
        let y2 = pb.y;
        a[(row, 0)] = x2 * x1;
        a[(row, 1)] = x2 * y1;
        a[(row, 2)] = x2;
        a[(row, 3)] = y2 * x1;
        a[(row, 4)] = y2 * y1;
        a[(row, 5)] = y2;
        a[(row, 6)] = x1;
        a[(row, 7)] = y1;
        a[(row, 8)] = 1.0;
    }

    // Solve for E using SVD: the essential matrix is the right singular vector corresponding to the smallest singular value
    let svd = nalgebra::SVD::new(a, true, true);
    let v_t = svd.v_t.expect("SVD should provide V^T");
    let e_vec = v_t.row(v_t.nrows() - 1);
    let mut e = Matrix3::zeros();
    e[(0, 0)] = e_vec[0];
    e[(0, 1)] = e_vec[1];
    e[(0, 2)] = e_vec[2];
    e[(1, 0)] = e_vec[3];
    e[(1, 1)] = e_vec[4];
    e[(1, 2)] = e_vec[5];
    e[(2, 0)] = e_vec[6];
    e[(2, 1)] = e_vec[7];
    e[(2, 2)] = e_vec[8];

    // Denormalize: E' = T_b^T * E * T_a
    e = t_b.transpose() * e * t_a;
    enforce_rank2(&mut e);
    e
}

fn normalize_points(points: Vec<Vector2<f64>>) -> (Vec<Vector2<f64>>, Matrix3<f64>) {
    let centroid = points.iter().fold(Vector2::zeros(), |acc, p| acc + p) / (points.len() as f64);

    let mut mean_dist = 0.0;
    for p in &points {
        mean_dist += (p - centroid).norm();
    }
    mean_dist /= points.len() as f64;
    let scale = if mean_dist.abs() < f64::EPSILON {
        1.0
    } else {
        (2.0f64).sqrt() / mean_dist
    };

    let transform = Matrix3::new(
        scale,
        0.0,
        -scale * centroid.x,
        0.0,
        scale,
        -scale * centroid.y,
        0.0,
        0.0,
        1.0,
    );

    let transformed: Vec<_> = points
        .into_iter()
        .map(|p| {
            let v = Vector3::new(p.x, p.y, 1.0);
            let norm = transform * v;
            Vector2::new(norm.x / norm.z, norm.y / norm.z)
        })
        .collect();

    (transformed, transform)
}

fn enforce_rank2(e: &mut Matrix3<f64>) {
    let svd = nalgebra::SVD::new(*e, true, true);
    let mut singular = svd.singular_values;
    if singular.len() >= 3 {
        singular[2] = 0.0;
    }
    let u = svd.u.unwrap();
    let vt = svd.v_t.unwrap();
    *e = u * nalgebra::Matrix3::from_diagonal(&singular) * vt;
}

fn score_inliers(matches: &[FeatureMatch], essential: &Matrix3<f64>, threshold: f64) -> Vec<usize> {
    let mut inliers = Vec::new();
    for (idx, m) in matches.iter().enumerate() {
        let err = sampson_error(essential, m);
        if err < threshold {
            inliers.push(idx);
        }
    }
    inliers
}

/// Sampson error is the distance of the projected point from the epipolar line.
pub fn sampson_error(e: &Matrix3<f64>, match_pair: &FeatureMatch) -> f64 {
    let x1 = SVector::<f64, 3>::new(match_pair.norm_xy_a.x, match_pair.norm_xy_a.y, 1.0);
    let x2 = SVector::<f64, 3>::new(match_pair.norm_xy_b.x, match_pair.norm_xy_b.y, 1.0);
    let ex1 = e * x1;
    let etx2 = e.transpose() * x2;
    let denom = ex1.x.powi(2) + ex1.y.powi(2) + etx2.x.powi(2) + etx2.y.powi(2);
    if denom.abs() < 1e-12 {
        return f64::MAX;
    }
    let numerator = x2.transpose() * e * x1;
    let num_scalar = numerator[(0, 0)];
    (num_scalar * num_scalar / denom).abs()
}

#[derive(Debug, Clone, Copy)]
struct CandidateEvaluation {
    rotation_c2c1: Rotation3<f64>,
    translation_c2c1: Vector3<f64>,
    positive_count: usize,
    total_parallax: f64,
    reprojection_error: f64,
}

pub fn compose_projection(rotation: &Rotation3<f64>, translation: &Vector3<f64>) -> Matrix3x4<f64> {
    let r = rotation.matrix();
    Matrix3x4::new(
        r[(0, 0)],
        r[(0, 1)],
        r[(0, 2)],
        translation.x,
        r[(1, 0)],
        r[(1, 1)],
        r[(1, 2)],
        translation.y,
        r[(2, 0)],
        r[(2, 1)],
        r[(2, 2)],
        translation.z,
    )
}

/// Triangulate a 3D point from two camera projections and a feature match.
/// Uses Direct Linear Transformation (DLT) to solve for the 3D point.
pub fn triangulate(
    p1: &Matrix3x4<f64>,
    p2: &Matrix3x4<f64>,
    m: &FeatureMatch,
) -> Option<Point3<f64>> {
    let mut a = nalgebra::DMatrix::<f64>::zeros(4, 4);

    fill_triangulation_row(&mut a, 0, p1, m.norm_xy_a.x, 0);
    fill_triangulation_row(&mut a, 1, p1, m.norm_xy_a.y, 1);
    fill_triangulation_row(&mut a, 2, p2, m.norm_xy_b.x, 0);
    fill_triangulation_row(&mut a, 3, p2, m.norm_xy_b.y, 1);

    let svd = nalgebra::SVD::new(a, true, true);
    let v_t = svd.v_t?;
    let homog = v_t.row(v_t.nrows() - 1);
    if homog[3].abs() < 1e-12 {
        return None;
    }
    let point = Point3::new(
        homog[0] / homog[3],
        homog[1] / homog[3],
        homog[2] / homog[3],
    );
    Some(point)
}

fn fill_triangulation_row(
    a: &mut nalgebra::DMatrix<f64>,
    row: usize,
    projection: &Matrix3x4<f64>,
    value: f64,
    axis: usize,
) {
    let row_data = projection.row(axis);
    let third_row = projection.row(2);
    for col in 0..4 {
        a[(row, col)] = value * third_row[col] - row_data[col];
    }
}

/// Estimate a homography matrix from feature matches using DLT
fn estimate_homography(matches: &[FeatureMatch]) -> Matrix3<f64> {
    let (points_a, t_a) = normalize_points(matches.iter().map(|m| m.norm_xy_a).collect());
    let (points_b, t_b) = normalize_points(matches.iter().map(|m| m.norm_xy_b).collect());

    // Build constraint matrix A where each correspondence gives 2 rows
    // x' = Hx becomes: [-x, -y, -1, 0, 0, 0, x'*x, x'*y, x'] * h = 0
    //                  [0, 0, 0, -x, -y, -1, y'*x, y'*y, y'] * h = 0
    let mut a = nalgebra::DMatrix::<f64>::zeros(matches.len() * 2, 9);

    for (i, (pa, pb)) in points_a.iter().zip(points_b.iter()).enumerate() {
        let x = pa.x;
        let y = pa.y;
        let xp = pb.x;
        let yp = pb.y;

        // First row
        a[(i * 2, 0)] = -x;
        a[(i * 2, 1)] = -y;
        a[(i * 2, 2)] = -1.0;
        a[(i * 2, 6)] = xp * x;
        a[(i * 2, 7)] = xp * y;
        a[(i * 2, 8)] = xp;

        // Second row
        a[(i * 2 + 1, 3)] = -x;
        a[(i * 2 + 1, 4)] = -y;
        a[(i * 2 + 1, 5)] = -1.0;
        a[(i * 2 + 1, 6)] = yp * x;
        a[(i * 2 + 1, 7)] = yp * y;
        a[(i * 2 + 1, 8)] = yp;
    }

    let svd = nalgebra::SVD::new(a, true, true);
    let v_t = svd.v_t.expect("SVD should provide V^T");
    let h_vec = v_t.row(v_t.nrows() - 1);

    let mut h = Matrix3::new(
        h_vec[0], h_vec[1], h_vec[2], h_vec[3], h_vec[4], h_vec[5], h_vec[6], h_vec[7], h_vec[8],
    );

    // Denormalize: H' = T_b^(-1) * H * T_a
    let t_b_inv = Matrix3::new(
        1.0 / t_b[(0, 0)],
        0.0,
        -t_b[(0, 2)] / t_b[(0, 0)],
        0.0,
        1.0 / t_b[(1, 1)],
        -t_b[(1, 2)] / t_b[(1, 1)],
        0.0,
        0.0,
        1.0,
    );

    h = t_b_inv * h * t_a;
    h
}

/// Compute symmetric transfer error for a homography
fn homography_symmetric_error(h: &Matrix3<f64>, m: &FeatureMatch) -> f64 {
    let x1 = Vector3::new(m.norm_xy_a.x, m.norm_xy_a.y, 1.0);
    let x2 = Vector3::new(m.norm_xy_b.x, m.norm_xy_b.y, 1.0);

    // Forward: x2' = H * x1
    let x2_proj = h * x1;
    let x2_proj_norm = Vector2::new(x2_proj.x / x2_proj.z, x2_proj.y / x2_proj.z);
    let forward_error = (m.norm_xy_b - x2_proj_norm).norm_squared();

    // Backward: x1' = H^(-1) * x2
    if let Some(h_inv) = h.try_inverse() {
        let x1_proj = h_inv * x2;
        let x1_proj_norm = Vector2::new(x1_proj.x / x1_proj.z, x1_proj.y / x1_proj.z);
        let backward_error = (m.norm_xy_a - x1_proj_norm).norm_squared();
        forward_error + backward_error
    } else {
        f64::MAX
    }
}

/// Score homography inliers using symmetric transfer error
fn score_homography_inliers(
    matches: &[FeatureMatch],
    h: &Matrix3<f64>,
    threshold: f64,
) -> Vec<usize> {
    let mut inliers = Vec::new();
    for (idx, m) in matches.iter().enumerate() {
        let err = homography_symmetric_error(h, m);
        if err < threshold * threshold * 2.0 {
            // Account for symmetric error
            inliers.push(idx);
        }
    }
    inliers
}

/// Detect if the scene is degenerate (e.g., planar or pure rotation)
/// Returns true if the scene is degenerate (should reject essential matrix estimation)
fn check_degeneracy(
    matches: &[FeatureMatch],
    essential_inliers: &[usize],
    params: &TwoViewInitializationParams,
) -> bool {
    if !params.check_homography {
        return false;
    }

    info!("Running degeneracy check with homography test");

    // Run RANSAC for homography
    let mut rng = SmallRng::from_entropy();
    let mut best_h_inliers = Vec::new();

    let h_iterations = params.ransac_iterations / 2; // Use fewer iterations for homography
    for iter_count in 0..h_iterations {
        let sample_indices = sample(&mut rng, matches.len(), 4); // Homography needs 4 points
        let subset: Vec<FeatureMatch> = sample_indices.iter().map(|idx| matches[idx]).collect();

        let h_candidate = estimate_homography(&subset);
        if !h_candidate.iter().all(|v| v.is_finite()) {
            continue;
        }

        let inliers = score_homography_inliers(matches, &h_candidate, params.ransac_threshold);
        if inliers.len() > best_h_inliers.len() {
            best_h_inliers = inliers;
            trace!(
                "Homography RANSAC iteration {}: new best with {} inliers",
                iter_count,
                best_h_inliers.len()
            );
        }
    }

    info!(
        "Degeneracy check: E has {} inliers, H has {} inliers",
        essential_inliers.len(),
        best_h_inliers.len()
    );

    // If homography explains significantly more inliers, scene is likely planar
    let e_count = essential_inliers.len() as f64;
    let h_count = best_h_inliers.len() as f64;

    if h_count > e_count && e_count / h_count < params.homography_ratio_threshold {
        info!(
            "Scene appears degenerate: E/H ratio {:.3} < {:.3}",
            e_count / h_count,
            params.homography_ratio_threshold
        );
        return true;
    }

    false
}

/// Refine essential matrix using Levenberg-Marquardt optimization
/// Minimizes the sum of squared Sampson errors over all inliers
fn refine_essential_lm(
    initial_e: &Matrix3<f64>,
    matches: &[FeatureMatch],
    inliers: &[usize],
    params: &TwoViewInitializationParams,
) -> Matrix3<f64> {
    if !params.enable_lm_refinement {
        return *initial_e;
    }

    info!(
        "Refining essential matrix with LM over {} inliers",
        inliers.len()
    );

    // Parameterize E using 5 parameters (essential matrix has 5 DOF)
    // We'll use the representation: E = [t]_x * R
    // where R is a 3x3 rotation (3 params) and t is unit translation (2 params on sphere)

    let decomp = decompose_for_lm_init(initial_e);
    if decomp.is_none() {
        info!("Failed to decompose initial E for LM, returning original");
        return *initial_e;
    }

    let (mut rot_params, mut trans_params) = decomp.unwrap();

    let lambda_init = 0.01;
    let mut lambda = lambda_init;
    let lambda_up = 10.0;
    let lambda_down = 0.1;

    let inlier_matches: Vec<FeatureMatch> = inliers.iter().map(|&i| matches[i]).collect();
    let mut current_cost = compute_sampson_cost(
        &params_to_essential(&rot_params, &trans_params),
        &inlier_matches,
    );

    info!("LM initial cost: {:.6}", current_cost);

    for iteration in 0..params.lm_max_iterations {
        // Compute Jacobian and residuals
        let (jacobian, residuals) =
            compute_essential_jacobian(&rot_params, &trans_params, &inlier_matches);

        // Build normal equations: (J^T J + λI) δ = -J^T r
        let jtj = jacobian.transpose() * &jacobian;
        let jtr = jacobian.transpose() * &residuals;

        // Add damping
        let mut jtj_damped = jtj.clone();
        for i in 0..5 {
            jtj_damped[(i, i)] += lambda;
        }

        // Solve for parameter update
        if let Some(delta) = jtj_damped.try_inverse() {
            let delta = delta * (-jtr);

            // Apply update
            let new_rot_params = Vector3::new(
                rot_params.x + delta[0],
                rot_params.y + delta[1],
                rot_params.z + delta[2],
            );
            let new_trans_params =
                Vector2::new(trans_params.x + delta[3], trans_params.y + delta[4]);

            let new_e = params_to_essential(&new_rot_params, &new_trans_params);
            let new_cost = compute_sampson_cost(&new_e, &inlier_matches);

            if new_cost < current_cost {
                // Accept update
                let improvement = (current_cost - new_cost) / current_cost;
                trace!(
                    "LM iteration {}: cost {:.6} -> {:.6} (improvement: {:.2}%)",
                    iteration,
                    current_cost,
                    new_cost,
                    improvement * 100.0
                );

                rot_params = new_rot_params;
                trans_params = new_trans_params;
                current_cost = new_cost;
                lambda *= lambda_down;

                // Early termination if improvement is tiny
                if improvement < 1e-6 {
                    info!("LM converged after {} iterations", iteration + 1);
                    break;
                }
            } else {
                // Reject update, increase damping
                lambda *= lambda_up;
            }
        } else {
            lambda *= lambda_up;
        }
    }

    let refined_e = params_to_essential(&rot_params, &trans_params);
    info!("LM final cost: {:.6}", current_cost);

    refined_e
}

/// Decompose essential matrix into rotation and translation parameters for LM
fn decompose_for_lm_init(e: &Matrix3<f64>) -> Option<(Vector3<f64>, Vector2<f64>)> {
    let svd = nalgebra::SVD::new(*e, true, true);
    let mut u = svd.u?;
    let mut vt = svd.v_t?;

    if u.determinant() < 0.0 {
        u.column_mut(2).scale_mut(-1.0);
    }
    if vt.determinant() < 0.0 {
        vt.row_mut(2).scale_mut(-1.0);
    }

    let w = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let r = u * w * vt;
    let rotation = Rotation3::from_matrix_unchecked(r);

    // Convert rotation to axis-angle (3 params)
    let axis_angle = rotation.scaled_axis();

    // Get translation direction (unit vector) and parameterize on sphere
    let t = u.column(2).into_owned();
    let theta = t.y.atan2(t.x);
    let phi = t.z.asin();

    Some((axis_angle, Vector2::new(theta, phi)))
}

/// Convert LM parameters back to essential matrix
fn params_to_essential(rot_params: &Vector3<f64>, trans_params: &Vector2<f64>) -> Matrix3<f64> {
    // Rotation from axis-angle
    let rotation = Rotation3::new(*rot_params);

    // Translation from spherical coordinates
    let theta = trans_params.x;
    let phi = trans_params.y;
    let translation = Vector3::new(theta.cos() * phi.cos(), theta.sin() * phi.cos(), phi.sin());

    // E = [t]_x * R
    let t_skew = Matrix3::new(
        0.0,
        -translation.z,
        translation.y,
        translation.z,
        0.0,
        -translation.x,
        -translation.y,
        translation.x,
        0.0,
    );

    t_skew * rotation.matrix()
}

/// Compute total Sampson cost for a set of matches
fn compute_sampson_cost(e: &Matrix3<f64>, matches: &[FeatureMatch]) -> f64 {
    matches.iter().map(|m| sampson_error(e, m)).sum()
}

/// Compute Jacobian of Sampson errors with respect to essential matrix parameters
fn compute_essential_jacobian(
    rot_params: &Vector3<f64>,
    trans_params: &Vector2<f64>,
    matches: &[FeatureMatch],
) -> (nalgebra::DMatrix<f64>, nalgebra::DVector<f64>) {
    let n = matches.len();
    let mut jacobian = nalgebra::DMatrix::<f64>::zeros(n, 5);
    let mut residuals = nalgebra::DVector::<f64>::zeros(n);

    let e = params_to_essential(rot_params, trans_params);
    let epsilon = 1e-6;

    // Compute residuals
    for (i, m) in matches.iter().enumerate() {
        residuals[i] = sampson_error(&e, m);
    }

    // Numerical Jacobian (could be analytical but this is simpler and robust)
    for param_idx in 0..5 {
        let mut perturbed_rot = *rot_params;
        let mut perturbed_trans = *trans_params;

        if param_idx < 3 {
            perturbed_rot[param_idx] += epsilon;
        } else {
            perturbed_trans[param_idx - 3] += epsilon;
        }

        let e_perturbed = params_to_essential(&perturbed_rot, &perturbed_trans);

        for (i, m) in matches.iter().enumerate() {
            let error_perturbed = sampson_error(&e_perturbed, m);
            jacobian[(i, param_idx)] = (error_perturbed - residuals[i]) / epsilon;
        }
    }

    (jacobian, residuals)
}
