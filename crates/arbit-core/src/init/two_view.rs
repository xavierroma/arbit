use log::{debug, trace};
use nalgebra::{Matrix3, Matrix3x4, Point3, Rotation3, SVector, Vector2, Vector3};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::index::sample;

#[derive(Debug, Clone, Copy)]
pub struct FeatureMatch {
    pub normalized_a: Vector2<f64>,
    pub normalized_b: Vector2<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct TwoViewInitializationParams {
    pub ransac_iterations: usize,
    pub ransac_threshold: f64,
    pub ransac_sample_size: usize,
    pub min_matches: usize,
    pub min_parallax: f64,
}

impl Default for TwoViewInitializationParams {
    fn default() -> Self {
        Self {
            ransac_iterations: 200,
            ransac_threshold: 1e-3,
            ransac_sample_size: 8,
            min_matches: 100,
            min_parallax: 10.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TwoViewInitialization {
    pub essential: Matrix3<f64>,
    pub rotation: Rotation3<f64>,
    pub translation: Vector3<f64>,
    pub inliers: Vec<usize>,
    pub average_sampson_error: f64,
    pub landmarks: Vec<(usize, Point3<f64>)>,
}

#[derive(Debug, Clone, Copy)]
pub struct DecomposedEssential {
    pub rotation: Rotation3<f64>,
    pub translation: Vector3<f64>,
}

pub struct TwoViewInitializer {
    params: TwoViewInitializationParams,
}

impl TwoViewInitializer {
    pub fn new(params: TwoViewInitializationParams) -> Self {
        Self { params }
    }

    pub fn estimate(&self, matches: &[FeatureMatch]) -> Option<TwoViewInitialization> {
        debug!(
            "Starting two-view initialization with {} feature matches",
            matches.len()
        );

        if matches.len() < self.params.min_matches {
            debug!(
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
            if !m.normalized_a.x.is_finite()
                || !m.normalized_a.y.is_finite()
                || !m.normalized_b.x.is_finite()
                || !m.normalized_b.y.is_finite()
            {
                debug!("Match {} has non-finite coordinates, rejecting", i);
                return None;
            }

            let dir_a = Vector3::new(m.normalized_a.x, m.normalized_a.y, 1.0);
            let dir_b = Vector3::new(m.normalized_b.x, m.normalized_b.y, 1.0);
            let norm_a = dir_a.norm();
            let norm_b = dir_b.norm();

            if norm_a <= f64::EPSILON || norm_b <= f64::EPSILON {
                debug!("Match {} has near-zero direction norm, rejecting", i);
                return None;
            }

            let cos_theta = (dir_a.dot(&dir_b) / (norm_a * norm_b)).clamp(-1.0, 1.0);
            parallax_sum += cos_theta.acos();
            parallax_samples += 1;
        }

        if parallax_samples == 0 {
            debug!("No valid matches available to evaluate parallax, rejecting");
            return None;
        }

        let average_parallax_deg = (parallax_sum / parallax_samples as f64).to_degrees();
        if average_parallax_deg < self.params.min_parallax {
            debug!(
                "Tracking not strong enough: average parallax {:.2}° < {:.2}°",
                average_parallax_deg, self.params.min_parallax,
            );
            return None;
        }

        // Check for sufficient spread (not all points at origin)
        let spread_a =
            matches.iter().map(|m| m.normalized_a.norm()).sum::<f64>() / matches.len() as f64;
        let spread_b =
            matches.iter().map(|m| m.normalized_b.norm()).sum::<f64>() / matches.len() as f64;

        if spread_a < 1e-6 || spread_b < 1e-6 {
            debug!(
                "Matches have insufficient spread (a:{:.6}, b:{:.6}), rejecting",
                spread_a, spread_b
            );
            return None;
        }

        debug!("Input validation passed, proceeding with RANSAC");

        let mut rng = SmallRng::from_entropy();
        let mut best_inliers = Vec::new();
        debug!(
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
                trace!(
                    "RANSAC iteration {}: non-finite essential matrix, skipping",
                    iter_count
                );
                continue;
            }

            let inliers = score_inliers(matches, &e_candidate, self.params.ransac_threshold);
            if inliers.len() > best_inliers.len() {
                best_inliers = inliers;
                debug!(
                    "RANSAC iteration {}: new best with {} inliers",
                    iter_count,
                    best_inliers.len()
                );
            }
        }

        if best_inliers.len() < 8 {
            debug!(
                "No sufficient inliers found after RANSAC (best: {})",
                best_inliers.len()
            );
            return None;
        }

        let inlier_matches: Vec<_> = best_inliers.iter().map(|&i| matches[i]).collect();
        debug!(
            "Refining essential matrix with {} inliers",
            inlier_matches.len()
        );
        let refined_e = estimate_essential(&inlier_matches);

        let decomposition = choose_pose(&refined_e, &inlier_matches)?;
        let avg_error = inlier_matches
            .iter()
            .map(|m| sampson_error(&refined_e, m))
            .sum::<f64>()
            / (inlier_matches.len() as f64);

        let identity_projection =
            Matrix3x4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        let candidate_projection =
            compose_projection(&decomposition.rotation, &decomposition.translation);

        let mut landmarks = Vec::with_capacity(best_inliers.len());

        for &idx in &best_inliers {
            if let Some(point) =
                triangulate(&identity_projection, &candidate_projection, &matches[idx])
            {
                landmarks.push((idx, point));
            }
        }
        if landmarks.len() < 8 {
            debug!(
                "Two-view initialization failed: insufficient triangulations ({}/{})",
                landmarks.len(),
                best_inliers.len()
            );
            return None;
        }
        debug!(
            "Two-view initialization successful: {} inliers, avg error: {:.6}",
            inlier_matches.len(),
            avg_error
        );

        Some(TwoViewInitialization {
            essential: refined_e,
            rotation: decomposition.rotation,
            translation: decomposition.translation,
            inliers: best_inliers,
            average_sampson_error: avg_error,
            landmarks,
        })
    }
}

impl TwoViewInitialization {
    pub fn scaled(&self, scale: f64) -> Self {
        let mut scaled = self.clone();
        scaled.translation *= scale;
        scaled.landmarks = self
            .landmarks
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
}

// Estimate the essential matrix from a set of feature matches; this is in epipolar geometry, the R and t are the rotation and translation of the second camera relative to the first, which once known can be used to triangulate 3D points.
fn estimate_essential(matches: &[FeatureMatch]) -> Matrix3<f64> {
    let (points_a, t_a) = normalize_points(matches.iter().map(|m| m.normalized_a).collect());
    let (points_b, t_b) = normalize_points(matches.iter().map(|m| m.normalized_b).collect());

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
    let scale = if mean_dist.abs() < std::f64::EPSILON {
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

fn sampson_error(e: &Matrix3<f64>, match_pair: &FeatureMatch) -> f64 {
    let x1 = SVector::<f64, 3>::new(match_pair.normalized_a.x, match_pair.normalized_a.y, 1.0);
    let x2 = SVector::<f64, 3>::new(match_pair.normalized_b.x, match_pair.normalized_b.y, 1.0);
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

fn choose_pose(essential: &Matrix3<f64>, matches: &[FeatureMatch]) -> Option<DecomposedEssential> {
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
    let r1 = Rotation3::from_matrix_unchecked(u * w * vt);
    let r2 = Rotation3::from_matrix_unchecked(u * w.transpose() * vt);
    let t = u.column(2).into_owned();

    let candidates = [(r1, t), (r1, -t), (r2, t), (r2, -t)];

    let p1 = Matrix3x4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    let mut best = None;
    let mut best_positive = 0usize;

    for (rotation, t_vec) in candidates.iter() {
        let p2 = compose_projection(rotation, t_vec);

        let mut positive = 0usize;
        for m in matches {
            if let Some(point) = triangulate(&p1, &p2, m) {
                let depth1 = point.z;
                let cam2 = rotation * point.coords + t_vec;
                if depth1 > 0.0 && cam2.z > 0.0 {
                    positive += 1;
                }
            }
        }

        if positive > best_positive {
            best_positive = positive;
            best = Some(DecomposedEssential {
                rotation: *rotation,
                translation: t_vec.normalize(),
            });
        }
    }

    best
}

fn compose_projection(rotation: &Rotation3<f64>, translation: &Vector3<f64>) -> Matrix3x4<f64> {
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

fn triangulate(p1: &Matrix3x4<f64>, p2: &Matrix3x4<f64>, m: &FeatureMatch) -> Option<Point3<f64>> {
    let mut a = nalgebra::DMatrix::<f64>::zeros(4, 4);

    fill_triangulation_row(&mut a, 0, p1, m.normalized_a.x, 0);
    fill_triangulation_row(&mut a, 1, p1, m.normalized_a.y, 1);
    fill_triangulation_row(&mut a, 2, p2, m.normalized_b.x, 0);
    fill_triangulation_row(&mut a, 3, p2, m.normalized_b.y, 1);

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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::{Rng, SeedableRng};

    #[test]
    fn initializer_recovers_pose() {
        let rotation = Rotation3::from_axis_angle(&Vector3::z_axis(), 0.1);
        let translation = Vector3::new(0.3, 0.0, 0.1);

        let points = generate_scene(100, 1.5, 3.5);
        let matches = project_scene(&points, rotation, translation);

        let initializer = TwoViewInitializer::new(TwoViewInitializationParams {
            ransac_iterations: 300,
            ransac_threshold: 5e-4,
            ransac_sample_size: 16,
            min_matches: 100,
            min_parallax: 10.0,
        });

        let result = initializer
            .estimate(&matches)
            .expect("two view initialization");
        assert!(result.inliers.len() >= 80);
        assert!(result.average_sampson_error < 1e-3);
        assert!(!result.landmarks.is_empty());
        let first_landmark_index = result.landmarks[0].0;

        assert_relative_eq!(result.rotation.matrix(), rotation.matrix(), epsilon = 1e-2);
        assert_relative_eq!(
            result.translation.normalize(),
            translation.normalize(),
            epsilon = 1e-2
        );

        let scaled = result.scaled(0.5);
        assert_eq!(scaled.landmarks.len(), result.landmarks.len());
        assert!(scaled.translation.norm() < result.translation.norm());
        let scaled_first = scaled
            .landmarks
            .iter()
            .find(|(idx, _)| *idx == first_landmark_index)
            .expect("scaled landmark");
        let original_first = result
            .landmarks
            .iter()
            .find(|(idx, _)| *idx == first_landmark_index)
            .unwrap();
        assert!(scaled_first.1.coords.norm() < original_first.1.coords.norm());
    }

    #[test]
    fn initializer_rejects_small_sets() {
        let initializer = TwoViewInitializer::new(TwoViewInitializationParams::default());
        let matches = vec![
            FeatureMatch {
                normalized_a: Vector2::new(0.0, 0.0),
                normalized_b: Vector2::new(0.0, 0.0),
            };
            5
        ];
        assert!(initializer.estimate(&matches).is_none());
    }

    fn project_scene(
        points: &[Point3<f64>],
        rotation: Rotation3<f64>,
        translation: Vector3<f64>,
    ) -> Vec<FeatureMatch> {
        let mut matches = Vec::with_capacity(points.len());
        for p in points {
            let pa = project_point(p);
            let pb_cam = rotation * p.coords + translation;
            let pb = project_point(&Point3::from(pb_cam));
            matches.push(FeatureMatch {
                normalized_a: pa,
                normalized_b: pb,
            });
        }
        matches
    }

    fn project_point(point: &Point3<f64>) -> Vector2<f64> {
        Vector2::new(point.x / point.z, point.y / point.z)
    }

    fn generate_scene(count: usize, z_min: f64, z_max: f64) -> Vec<Point3<f64>> {
        let mut rng = SmallRng::seed_from_u64(42);
        let mut points = Vec::with_capacity(count);
        for _ in 0..count {
            let x = rng.gen_range(-0.6..0.6);
            let y = rng.gen_range(-0.6..0.6);
            let z = rng.gen_range(z_min..z_max);
            points.push(Point3::new(x, y, z));
        }
        points
    }
}
