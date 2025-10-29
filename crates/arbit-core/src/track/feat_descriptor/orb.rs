use std::sync::OnceLock;

use super::{FeatDescriptor, FeatDescriptorExtractor};
use crate::img::Pyramid;
use crate::img::image_utils::bilinear_sample_luma;
use crate::img::pyramid::PyramidLevel;
use crate::track::FeatureSeed;

const ORB_DESCRIPTOR_BYTES: usize = 32;
const ORB_POINTS: usize = ORB_DESCRIPTOR_BYTES * 8;
const DEFAULT_PATCH_RADIUS: usize = 15; // ORB uses a 31x31 patch

/// Oriented FAST and Rotated BRIEF (ORB) descriptor extractor.
///
/// Each descriptor encodes 256 binary intensity tests drawn from a 31×31 patch
/// around the feature. Pairs of pixel offsets are rotated by the keypoint's
/// intensity-centroid angle and then compared; the results are packed into a
/// 32-byte bitstring that can be matched efficiently with Hamming distance.
#[derive(Debug, Clone)]
pub struct OrbDescriptor {
    patch_radius: usize,
}

impl OrbDescriptor {
    pub fn new() -> Self {
        Self {
            patch_radius: DEFAULT_PATCH_RADIUS,
        }
    }

    pub fn with_patch_radius(patch_radius: usize) -> Self {
        Self { patch_radius }
    }

    /// Returns the base BRIEF sampling pattern used by ORB.
    fn pattern(&self) -> &'static [PatternPair] {
        // Pattern only depends on the patch radius and descriptor length. We
        // keep a single shared instance derived from the default radius.
        // Custom radii reuse the same distribution scaled at runtime.
        pattern_cache()
    }

    fn describe_seed(
        &self,
        level: &PyramidLevel,
        seed: &FeatureSeed,
    ) -> FeatDescriptor<[u8; ORB_DESCRIPTOR_BYTES]> {
        let center_x = seed.px_uv.x;
        let center_y = seed.px_uv.y;
        let angle = compute_orientation(level, center_x, center_y, self.patch_radius as f32);
        let descriptor = build_descriptor(
            level,
            center_x,
            center_y,
            angle,
            self.patch_radius as f32,
            self.pattern(),
        );

        FeatDescriptor {
            seed: *seed,
            angle,
            data: descriptor,
        }
    }
}

impl Default for OrbDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatDescriptorExtractor for OrbDescriptor {
    type Storage = [u8; ORB_DESCRIPTOR_BYTES];

    const LEN: usize = ORB_DESCRIPTOR_BYTES;

    fn describe(
        &self,
        pyramid: &Pyramid,
        seeds: &[FeatureSeed],
    ) -> Vec<FeatDescriptor<Self::Storage>> {
        let levels = pyramid.levels();
        let mut output = Vec::with_capacity(seeds.len());
        for seed in seeds {
            if let Some(level) = levels.get(seed.level) {
                output.push(self.describe_seed(level, seed));
            }
        }
        output
    }
}

#[derive(Clone, Copy)]
struct PatternPair {
    p1: (f32, f32),
    p2: (f32, f32),
}

/// Lazily initialises the canonical ORB sampling pattern.
fn pattern_cache() -> &'static [PatternPair] {
    static CACHE: OnceLock<Vec<PatternPair>> = OnceLock::new();
    CACHE
        .get_or_init(|| generate_pattern(ORB_POINTS, DEFAULT_PATCH_RADIUS as f32))
        .as_slice()
}

fn generate_pattern(count: usize, radius: f32) -> Vec<PatternPair> {
    let mut rng = XorShift64::new(0xDEADBEEFCAFEBABEu64);
    let mut pairs = Vec::with_capacity(count);
    let radius_sq = radius * radius;
    while pairs.len() < count {
        let p1 = random_point(&mut rng, radius, radius_sq);
        let p2 = random_point(&mut rng, radius, radius_sq);
        pairs.push(PatternPair { p1, p2 });
    }
    pairs
}

fn random_point(rng: &mut XorShift64, radius: f32, radius_sq: f32) -> (f32, f32) {
    loop {
        let x = (rng.next_f32() * 2.0 - 1.0) * radius;
        let y = (rng.next_f32() * 2.0 - 1.0) * radius;
        if x * x + y * y <= radius_sq {
            return (x, y);
        }
    }
}

fn compute_orientation(level: &PyramidLevel, cx: f32, cy: f32, radius: f32) -> f32 {
    let radius = radius.round() as isize;
    let mut m01 = 0.0f32;
    let mut m10 = 0.0f32;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let intensity = bilinear_sample_luma(&level.image, cx + dx as f32, cy + dy as f32);
            m10 += dx as f32 * intensity;
            m01 += dy as f32 * intensity;
        }
    }

    if m10.abs() < f32::EPSILON && m01.abs() < f32::EPSILON {
        0.0
    } else {
        m01.atan2(m10)
    }
}

fn build_descriptor(
    level: &PyramidLevel,
    cx: f32,
    cy: f32,
    angle: f32,
    patch_radius: f32,
    pattern: &[PatternPair],
) -> [u8; ORB_DESCRIPTOR_BYTES] {
    let sin_theta = angle.sin();
    let cos_theta = angle.cos();
    let scale = if patch_radius == DEFAULT_PATCH_RADIUS as f32 {
        1.0
    } else {
        patch_radius / DEFAULT_PATCH_RADIUS as f32
    };

    let mut bytes = [0u8; ORB_DESCRIPTOR_BYTES];
    for (i, pair) in pattern.iter().enumerate() {
        let (x1, y1) = rotate_point(pair.p1, cos_theta, sin_theta, scale, cx, cy);
        let (x2, y2) = rotate_point(pair.p2, cos_theta, sin_theta, scale, cx, cy);

        let v1 = bilinear_sample_luma(&level.image, x1, y1);
        let v2 = bilinear_sample_luma(&level.image, x2, y2);

        if v1 < v2 {
            bytes[i / 8] |= 1 << (i & 7);
        }
    }
    bytes
}

fn rotate_point(
    point: (f32, f32),
    cos_theta: f32,
    sin_theta: f32,
    scale: f32,
    cx: f32,
    cy: f32,
) -> (f32, f32) {
    let (px, py) = point;
    let px = px * scale;
    let py = py * scale;
    let rx = cos_theta * px - sin_theta * py + cx;
    let ry = sin_theta * px + cos_theta * py + cy;
    (rx, ry)
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 7;
        x ^= x >> 9;
        x ^= x << 8;
        self.state = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        let bits = self.next_u64() >> 40; // Use upper 24 bits
        (bits as f32) / (1u64 << 24) as f32
    }
}
