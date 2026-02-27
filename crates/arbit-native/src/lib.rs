use std::cmp::Ordering;
use std::collections::HashMap;

use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum NativeKernelError {
    #[error("feature '{0}' is disabled")]
    FeatureDisabled(&'static str),
    #[error("invalid input: {0}")]
    InvalidInput(&'static str),
    #[error("optimization failed")]
    OptimizationFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelStatus {
    Ready,
    Disabled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeaturePoint {
    pub x: u32,
    pub y: u32,
    pub score: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryFeature {
    pub point: FeaturePoint,
    pub descriptor: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeatureMatch {
    pub query_idx: usize,
    pub train_idx: usize,
    pub distance: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoseGraphEdge {
    pub from: usize,
    pub to: usize,
    pub delta_xyz: [f64; 3],
    pub sigma: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PoseGraphSolveResult {
    pub translations: Vec<[f64; 3]>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BowHistogram {
    pub dimension: usize,
    pub indices: Vec<usize>,
    pub weights: Vec<f32>,
}

impl BowHistogram {
    fn new(dimension: usize, mut counts: HashMap<usize, f32>) -> Self {
        let mut items = counts.drain().collect::<Vec<_>>();
        items.sort_by_key(|(idx, _)| *idx);

        let mut indices = Vec::with_capacity(items.len());
        let mut weights = Vec::with_capacity(items.len());
        let mut norm_sq = 0.0_f32;

        for (idx, weight) in items {
            indices.push(idx);
            weights.push(weight);
            norm_sq += weight * weight;
        }

        let norm = norm_sq.sqrt();
        if norm > 0.0 {
            for w in &mut weights {
                *w /= norm;
            }
        }

        Self {
            dimension,
            indices,
            weights,
        }
    }

    fn cosine_similarity(&self, other: &Self) -> f32 {
        if self.dimension != other.dimension {
            return 0.0;
        }

        let mut lhs = 0usize;
        let mut rhs = 0usize;
        let mut dot = 0.0_f32;

        while lhs < self.indices.len() && rhs < other.indices.len() {
            match self.indices[lhs].cmp(&other.indices[rhs]) {
                Ordering::Less => lhs += 1,
                Ordering::Greater => rhs += 1,
                Ordering::Equal => {
                    dot += self.weights[lhs] * other.weights[rhs];
                    lhs += 1;
                    rhs += 1;
                }
            }
        }

        dot
    }
}

#[derive(Debug, Clone)]
pub struct OpenCvKernelAdapter {
    pub fast_threshold: u8,
}

impl Default for OpenCvKernelAdapter {
    fn default() -> Self {
        Self { fast_threshold: 16 }
    }
}

impl OpenCvKernelAdapter {
    pub fn status(&self) -> KernelStatus {
        if cfg!(feature = "native-opencv") {
            KernelStatus::Ready
        } else {
            KernelStatus::Disabled
        }
    }

    pub fn warmup(&self) -> Result<(), NativeKernelError> {
        #[cfg(feature = "native-opencv")]
        {
            let width = 64_u32;
            let height = 48_u32;
            let bytes_per_row = (width as usize) * 4;
            let mut frame = vec![0_u8; bytes_per_row * height as usize];
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let checker = ((x / 8) + (y / 8)) % 2;
                    let value = if checker == 0 { 16 } else { 240 };
                    let idx = y * bytes_per_row + x * 4;
                    frame[idx] = value;
                    frame[idx + 1] = value;
                    frame[idx + 2] = value;
                    frame[idx + 3] = 255;
                }
            }

            let feats = self.detect_and_describe_bgra(width, height, bytes_per_row, &frame, 64)?;
            if feats.is_empty() {
                return Err(NativeKernelError::OptimizationFailed);
            }
            Ok(())
        }

        #[cfg(not(feature = "native-opencv"))]
        {
            Err(NativeKernelError::FeatureDisabled("native-opencv"))
        }
    }

    pub fn detect_and_describe_bgra(
        &self,
        width: u32,
        height: u32,
        bytes_per_row: usize,
        data: &[u8],
        max_features: usize,
    ) -> Result<Vec<BinaryFeature>, NativeKernelError> {
        #[cfg(feature = "native-opencv")]
        {
            if is_low_texture_bgra(width, height, bytes_per_row, data)? {
                return Ok(Vec::new());
            }
            let gray = decode_bgra_to_gray(width, height, bytes_per_row, data)?;
            detect_and_describe_gray(&gray, self.fast_threshold, max_features)
        }

        #[cfg(not(feature = "native-opencv"))]
        {
            let _ = (width, height, bytes_per_row, data, max_features);
            Err(NativeKernelError::FeatureDisabled("native-opencv"))
        }
    }

    pub fn match_features(
        &self,
        query: &[BinaryFeature],
        train: &[BinaryFeature],
        max_distance: u32,
        cross_check: bool,
    ) -> Result<Vec<FeatureMatch>, NativeKernelError> {
        #[cfg(feature = "native-opencv")]
        {
            Ok(match_binary_features(
                query,
                train,
                max_distance,
                cross_check,
            ))
        }

        #[cfg(not(feature = "native-opencv"))]
        {
            let _ = (query, train, max_distance, cross_check);
            Err(NativeKernelError::FeatureDisabled("native-opencv"))
        }
    }
}

#[cfg(feature = "native-opencv")]
fn is_low_texture_bgra(
    width: u32,
    height: u32,
    bytes_per_row: usize,
    data: &[u8],
) -> Result<bool, NativeKernelError> {
    if width == 0 || height == 0 {
        return Err(NativeKernelError::InvalidInput(
            "image dimensions must be non-zero",
        ));
    }
    let min_stride = (width as usize) * 4;
    if bytes_per_row < min_stride {
        return Err(NativeKernelError::InvalidInput(
            "bytes_per_row smaller than packed BGRA row",
        ));
    }
    let required = bytes_per_row
        .checked_mul(height as usize)
        .ok_or(NativeKernelError::InvalidInput("frame buffer size overflow"))?;
    if data.len() < required {
        return Err(NativeKernelError::InvalidInput(
            "frame buffer shorter than expected",
        ));
    }

    let sample_step_x = (width.max(64) / 64) as usize;
    let sample_step_y = (height.max(48) / 48) as usize;
    let mut min_luma = u8::MAX;
    let mut max_luma = u8::MIN;

    for y in (0..height as usize).step_by(sample_step_y.max(1)) {
        let row = y * bytes_per_row;
        for x in (0..width as usize).step_by(sample_step_x.max(1)) {
            let idx = row + x * 4;
            let b = data[idx] as u16;
            let g = data[idx + 1] as u16;
            let r = data[idx + 2] as u16;
            let luma = ((77 * r + 150 * g + 29 * b) >> 8) as u8;
            min_luma = min_luma.min(luma);
            max_luma = max_luma.max(luma);
        }
    }

    Ok(max_luma.saturating_sub(min_luma) <= 6)
}

#[derive(Debug, Default, Clone)]
pub struct GtsamKernelAdapter;

impl GtsamKernelAdapter {
    pub fn status(&self) -> KernelStatus {
        if cfg!(feature = "native-gtsam") {
            KernelStatus::Ready
        } else {
            KernelStatus::Disabled
        }
    }

    pub fn warmup(&self) -> Result<(), NativeKernelError> {
        let edges = [
            PoseGraphEdge {
                from: 0,
                to: 1,
                delta_xyz: [0.0, 0.0, -0.05],
                sigma: 0.5,
            },
            PoseGraphEdge {
                from: 1,
                to: 2,
                delta_xyz: [0.0, 0.0, -0.05],
                sigma: 0.5,
            },
        ];
        let solved = self.optimize_translation_graph(3, &edges)?;
        if solved.translations.len() != 3 {
            return Err(NativeKernelError::OptimizationFailed);
        }
        Ok(())
    }

    pub fn optimize_translation_graph(
        &self,
        node_count: usize,
        edges: &[PoseGraphEdge],
    ) -> Result<PoseGraphSolveResult, NativeKernelError> {
        #[cfg(feature = "native-gtsam")]
        {
            solve_translation_graph(node_count, edges)
        }

        #[cfg(not(feature = "native-gtsam"))]
        {
            let _ = (node_count, edges);
            Err(NativeKernelError::FeatureDisabled("native-gtsam"))
        }
    }
}

#[derive(Debug, Clone)]
pub struct BowKernelAdapter {
    pub vocabulary_size: usize,
}

impl Default for BowKernelAdapter {
    fn default() -> Self {
        Self {
            vocabulary_size: 4_096,
        }
    }
}

impl BowKernelAdapter {
    pub fn status(&self) -> KernelStatus {
        if cfg!(feature = "native-bow") {
            KernelStatus::Ready
        } else {
            KernelStatus::Disabled
        }
    }

    pub fn warmup(&self) -> Result<(), NativeKernelError> {
        let descriptors = vec![[0_u8; 32], [255_u8; 32], [17_u8; 32]];
        let histogram = self.encode_descriptors(&descriptors)?;
        if histogram.indices.is_empty() {
            return Err(NativeKernelError::OptimizationFailed);
        }
        Ok(())
    }

    pub fn encode_descriptors(
        &self,
        descriptors: &[[u8; 32]],
    ) -> Result<BowHistogram, NativeKernelError> {
        #[cfg(feature = "native-bow")]
        {
            if self.vocabulary_size == 0 {
                return Err(NativeKernelError::InvalidInput(
                    "vocabulary size must be greater than zero",
                ));
            }

            let mut counts: HashMap<usize, f32> = HashMap::new();
            for descriptor in descriptors {
                let word_id = hash_descriptor(descriptor, self.vocabulary_size);
                *counts.entry(word_id).or_insert(0.0) += 1.0;
            }

            Ok(BowHistogram::new(self.vocabulary_size, counts))
        }

        #[cfg(not(feature = "native-bow"))]
        {
            let _ = descriptors;
            Err(NativeKernelError::FeatureDisabled("native-bow"))
        }
    }

    pub fn query_top_k(
        &self,
        query: &BowHistogram,
        database: &[(u64, BowHistogram)],
        max_results: usize,
    ) -> Result<Vec<(u64, f32)>, NativeKernelError> {
        #[cfg(feature = "native-bow")]
        {
            let mut scored = database
                .iter()
                .map(|(id, hist)| (*id, query.cosine_similarity(hist)))
                .collect::<Vec<_>>();
            scored.sort_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1));
            scored.truncate(max_results);
            Ok(scored)
        }

        #[cfg(not(feature = "native-bow"))]
        {
            let _ = (query, database, max_results);
            Err(NativeKernelError::FeatureDisabled("native-bow"))
        }
    }
}

fn hash_descriptor(descriptor: &[u8; 32], vocabulary_size: usize) -> usize {
    let mut hash = 0xcbf29ce484222325_u64;
    for b in descriptor {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    (hash as usize) % vocabulary_size
}

#[cfg(feature = "native-opencv")]
fn decode_bgra_to_gray(
    width: u32,
    height: u32,
    bytes_per_row: usize,
    data: &[u8],
) -> Result<image::GrayImage, NativeKernelError> {
    use image::{GrayImage, Luma};

    if width == 0 || height == 0 {
        return Err(NativeKernelError::InvalidInput(
            "image dimensions must be non-zero",
        ));
    }

    let min_stride = (width as usize) * 4;
    if bytes_per_row < min_stride {
        return Err(NativeKernelError::InvalidInput(
            "bytes_per_row smaller than packed BGRA row",
        ));
    }

    let required = bytes_per_row
        .checked_mul(height as usize)
        .ok_or(NativeKernelError::InvalidInput("frame buffer size overflow"))?;
    if data.len() < required {
        return Err(NativeKernelError::InvalidInput(
            "frame buffer shorter than expected",
        ));
    }

    let mut gray = GrayImage::new(width, height);
    for y in 0..height as usize {
        let row = y * bytes_per_row;
        for x in 0..width as usize {
            let idx = row + x * 4;
            let b = data[idx] as u16;
            let g = data[idx + 1] as u16;
            let r = data[idx + 2] as u16;
            let luma = ((77 * r + 150 * g + 29 * b) >> 8) as u8;
            gray.put_pixel(x as u32, y as u32, Luma([luma]));
        }
    }

    Ok(gray)
}

#[cfg(feature = "native-opencv")]
fn detect_and_describe_gray(
    gray: &image::GrayImage,
    fast_threshold: u8,
    max_features: usize,
) -> Result<Vec<BinaryFeature>, NativeKernelError> {
    use imageproc::corners::corners_fast9;

    if max_features == 0 {
        return Ok(Vec::new());
    }

    let mut candidates = corners_fast9(gray, fast_threshold)
        .into_iter()
        .map(|corner| (corner.x, corner.y, corner.score))
        .collect::<Vec<_>>();

    if candidates.is_empty() {
        candidates = fallback_gradient_candidates(gray);
    }
    candidates.sort_by(|lhs, rhs| rhs.2.total_cmp(&lhs.2));

    let mut features = Vec::with_capacity(max_features);
    for (x, y, score) in candidates {
        if features.len() >= max_features {
            break;
        }

        let Some(descriptor) = brief_descriptor(gray, x, y) else {
            continue;
        };

        features.push(BinaryFeature {
            point: FeaturePoint {
                x,
                y,
                score: score.clamp(0.0, u16::MAX as f32) as u16,
            },
            descriptor,
        });
    }

    Ok(features)
}

#[cfg(feature = "native-opencv")]
fn fallback_gradient_candidates(gray: &image::GrayImage) -> Vec<(u32, u32, f32)> {
    let width = gray.width();
    let height = gray.height();
    if width < 3 || height < 3 {
        return Vec::new();
    }

    let mut candidates = Vec::new();
    for y in (1..height - 1).step_by(2) {
        for x in (1..width - 1).step_by(2) {
            let center = gray.get_pixel(x, y).0[0] as i16;
            let dx = (gray.get_pixel(x + 1, y).0[0] as i16 - gray.get_pixel(x - 1, y).0[0] as i16)
                .unsigned_abs();
            let dy = (gray.get_pixel(x, y + 1).0[0] as i16 - gray.get_pixel(x, y - 1).0[0] as i16)
                .unsigned_abs();
            let lap = (gray.get_pixel(x + 1, y).0[0] as i16
                + gray.get_pixel(x - 1, y).0[0] as i16
                + gray.get_pixel(x, y + 1).0[0] as i16
                + gray.get_pixel(x, y - 1).0[0] as i16
                - 4 * center)
                .unsigned_abs();
            let score = (dx + dy + lap) as f32;
            if score >= 24.0 {
                candidates.push((x, y, score));
            }
        }
    }

    candidates
}

#[cfg(feature = "native-opencv")]
fn match_binary_features(
    query: &[BinaryFeature],
    train: &[BinaryFeature],
    max_distance: u32,
    cross_check: bool,
) -> Vec<FeatureMatch> {
    if query.is_empty() || train.is_empty() {
        return Vec::new();
    }

    let mut forward = Vec::new();
    for (query_idx, query_feature) in query.iter().enumerate() {
        let mut best = (usize::MAX, u32::MAX);
        let mut second_best = u32::MAX;
        for (train_idx, train_feature) in train.iter().enumerate() {
            let dist = hamming_distance(&query_feature.descriptor, &train_feature.descriptor);
            if dist < best.1 {
                second_best = best.1;
                best = (train_idx, dist);
            } else if dist < second_best {
                second_best = dist;
            }
        }

        if best.0 == usize::MAX || best.1 > max_distance {
            continue;
        }

        // Lowe-style ratio gate for descriptor ambiguity.
        if second_best < u32::MAX {
            let ratio = best.1 as f32 / (second_best as f32 + f32::EPSILON);
            if ratio > 0.92 {
                continue;
            }
        }

        forward.push(FeatureMatch {
            query_idx,
            train_idx: best.0,
            distance: best.1,
        });
    }

    if !cross_check {
        return forward;
    }

    let mut reverse_best = vec![usize::MAX; train.len()];
    for (train_idx, train_feature) in train.iter().enumerate() {
        let mut best_query = (usize::MAX, u32::MAX);
        for (query_idx, query_feature) in query.iter().enumerate() {
            let dist = hamming_distance(&train_feature.descriptor, &query_feature.descriptor);
            if dist < best_query.1 {
                best_query = (query_idx, dist);
            }
        }
        reverse_best[train_idx] = best_query.0;
    }

    forward
        .into_iter()
        .filter(|m| reverse_best[m.train_idx] == m.query_idx)
        .collect()
}

#[cfg(feature = "native-opencv")]
fn brief_descriptor(gray: &image::GrayImage, x: u32, y: u32) -> Option<[u8; 32]> {
    const HALF_PATCH: i32 = 15;

    let xi = x as i32;
    let yi = y as i32;
    if xi < HALF_PATCH
        || yi < HALF_PATCH
        || xi + HALF_PATCH >= gray.width() as i32
        || yi + HALF_PATCH >= gray.height() as i32
    {
        return None;
    }

    let mut desc = [0_u8; 32];
    for bit in 0..256_u32 {
        let (ax, ay, bx, by) = brief_pair(bit);
        let xa = (xi + ax) as u32;
        let ya = (yi + ay) as u32;
        let xb = (xi + bx) as u32;
        let yb = (yi + by) as u32;

        let pa = gray.get_pixel(xa, ya).0[0];
        let pb = gray.get_pixel(xb, yb).0[0];
        if pa < pb {
            desc[(bit / 8) as usize] |= 1_u8 << (bit % 8);
        }
    }

    Some(desc)
}

#[cfg(feature = "native-opencv")]
fn brief_pair(bit: u32) -> (i32, i32, i32, i32) {
    let a = lcg(bit.wrapping_mul(747_796_405).wrapping_add(2_891_336_453));
    let b = lcg(a.wrapping_mul(747_796_405).wrapping_add(2_891_336_453));

    let ax = ((a & 31) as i32) - 15;
    let ay = (((a >> 5) & 31) as i32) - 15;
    let bx = ((b & 31) as i32) - 15;
    let by = (((b >> 5) & 31) as i32) - 15;

    (ax, ay, bx, by)
}

#[cfg(feature = "native-opencv")]
fn lcg(seed: u32) -> u32 {
    seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
}

fn hamming_distance(lhs: &[u8; 32], rhs: &[u8; 32]) -> u32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(a, b)| (a ^ b).count_ones())
        .sum()
}

#[cfg(feature = "native-gtsam")]
fn solve_translation_graph(
    node_count: usize,
    edges: &[PoseGraphEdge],
) -> Result<PoseGraphSolveResult, NativeKernelError> {
    use tiny_solver::factors::{Factor, PriorFactor, na};
    use tiny_solver::{GaussNewtonOptimizer, Optimizer, OptimizerOptions, Problem};

    if node_count == 0 {
        return Err(NativeKernelError::InvalidInput(
            "pose graph must contain at least one node",
        ));
    }

    #[derive(Debug, Clone, Copy)]
    struct BetweenTranslationFactor {
        delta_xyz: [f64; 3],
    }

    impl<T: na::RealField> Factor<T> for BetweenTranslationFactor {
        fn residual_func(&self, params: &[na::DVector<T>]) -> na::DVector<T> {
            let from = &params[0];
            let to = &params[1];
            let dx = T::from_f64(self.delta_xyz[0]).unwrap();
            let dy = T::from_f64(self.delta_xyz[1]).unwrap();
            let dz = T::from_f64(self.delta_xyz[2]).unwrap();
            na::DVector::from_vec(vec![
                to[0].clone() - from[0].clone() - dx,
                to[1].clone() - from[1].clone() - dy,
                to[2].clone() - from[2].clone() - dz,
            ])
        }
    }

    let mut problem = Problem::new();

    // Fix gauge freedom by anchoring x0 at the origin.
    problem.add_residual_block(
        3,
        &["x0"],
        Box::new(PriorFactor {
            v: na::DVector::from_vec(vec![0.0, 0.0, 0.0]),
        }),
        None,
    );

    for edge in edges {
        if edge.from >= node_count || edge.to >= node_count {
            return Err(NativeKernelError::InvalidInput(
                "edge references node outside graph range",
            ));
        }

        let from_key = format!("x{}", edge.from);
        let to_key = format!("x{}", edge.to);
        problem.add_residual_block(
            3,
            &[from_key.as_str(), to_key.as_str()],
            Box::new(BetweenTranslationFactor {
                delta_xyz: edge.delta_xyz,
            }),
            None,
        );
    }

    let mut initial_values = HashMap::new();
    let mut current = [0.0_f64; 3];
    initial_values.insert("x0".to_string(), na::DVector::from_vec(vec![0.0, 0.0, 0.0]));
    for idx in 1..node_count {
        if let Some(edge) = edges.iter().find(|edge| edge.from + 1 == idx && edge.to == idx) {
            current[0] += edge.delta_xyz[0];
            current[1] += edge.delta_xyz[1];
            current[2] += edge.delta_xyz[2];
        }
        initial_values.insert(
            format!("x{idx}"),
            na::DVector::from_vec(vec![current[0], current[1], current[2]]),
        );
    }

    let optimizer = GaussNewtonOptimizer::new();
    let options = OptimizerOptions {
        max_iteration: 32,
        ..OptimizerOptions::default()
    };

    let solved = optimizer
        .optimize(&problem, &initial_values, Some(options))
        .ok_or(NativeKernelError::OptimizationFailed)?;

    let mut translations = vec![[0.0_f64; 3]; node_count];
    for (idx, slot) in translations.iter_mut().enumerate() {
        let key = format!("x{idx}");
        let value = solved.get(&key).ok_or(NativeKernelError::OptimizationFailed)?;
        if value.len() != 3 {
            return Err(NativeKernelError::OptimizationFailed);
        }
        *slot = [value[0], value[1], value[2]];
    }

    Ok(PoseGraphSolveResult { translations })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapters_report_expected_status_per_feature() {
        let opencv = OpenCvKernelAdapter::default();
        let gtsam = GtsamKernelAdapter;
        let bow = BowKernelAdapter::default();

        #[cfg(feature = "native-opencv")]
        assert_eq!(opencv.status(), KernelStatus::Ready);
        #[cfg(not(feature = "native-opencv"))]
        assert_eq!(opencv.status(), KernelStatus::Disabled);

        #[cfg(feature = "native-gtsam")]
        assert_eq!(gtsam.status(), KernelStatus::Ready);
        #[cfg(not(feature = "native-gtsam"))]
        assert_eq!(gtsam.status(), KernelStatus::Disabled);

        #[cfg(feature = "native-bow")]
        assert_eq!(bow.status(), KernelStatus::Ready);
        #[cfg(not(feature = "native-bow"))]
        assert_eq!(bow.status(), KernelStatus::Disabled);
    }

    #[test]
    #[cfg(feature = "native-opencv")]
    fn opencv_adapter_detects_and_matches_features() {
        let adapter = OpenCvKernelAdapter::default();

        let width = 96_u32;
        let height = 72_u32;
        let stride = (width as usize) * 4;
        let mut frame = vec![0_u8; stride * height as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let checker = ((x / 6) + (y / 6)) % 2;
                let value = if checker == 0 { 24 } else { 220 };
                let idx = y * stride + x * 4;
                frame[idx] = value;
                frame[idx + 1] = value;
                frame[idx + 2] = value;
                frame[idx + 3] = 255;
            }
        }

        let feats_a = adapter
            .detect_and_describe_bgra(width, height, stride, &frame, 120)
            .unwrap();
        assert!(!feats_a.is_empty());

        let matches = adapter.match_features(&feats_a, &feats_a, 0, true).unwrap();
        assert!(!matches.is_empty());
    }

    #[test]
    #[cfg(feature = "native-gtsam")]
    fn gtsam_adapter_solves_small_translation_graph() {
        let adapter = GtsamKernelAdapter;
        let edges = vec![
            PoseGraphEdge {
                from: 0,
                to: 1,
                delta_xyz: [0.0, 0.0, -0.1],
                sigma: 1.0,
            },
            PoseGraphEdge {
                from: 1,
                to: 2,
                delta_xyz: [0.0, 0.0, -0.1],
                sigma: 1.0,
            },
            PoseGraphEdge {
                from: 2,
                to: 3,
                delta_xyz: [0.0, 0.0, -0.1],
                sigma: 1.0,
            },
        ];

        let solved = adapter.optimize_translation_graph(4, &edges).unwrap();
        assert_eq!(solved.translations.len(), 4);
        assert!(solved.translations[3][2] < -0.25);
    }

    #[test]
    #[cfg(feature = "native-bow")]
    fn bow_adapter_ranks_similar_descriptors_first() {
        let adapter = BowKernelAdapter::default();

        let query_desc = vec![[17_u8; 32], [23_u8; 32], [89_u8; 32]];
        let near_desc = vec![[17_u8; 32], [23_u8; 32], [91_u8; 32]];
        let far_desc = vec![[240_u8; 32], [241_u8; 32], [242_u8; 32]];

        let query_hist = adapter.encode_descriptors(&query_desc).unwrap();
        let db = vec![
            (1_u64, adapter.encode_descriptors(&near_desc).unwrap()),
            (2_u64, adapter.encode_descriptors(&far_desc).unwrap()),
        ];

        let ranked = adapter.query_top_k(&query_hist, &db, 2).unwrap();
        assert_eq!(ranked[0].0, 1);
        assert!(ranked[0].1 >= ranked[1].1);
    }
}
