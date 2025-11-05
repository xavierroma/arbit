use anyhow::{bail, Context, Result};
use imageproc::image::{self, imageops, ImageBuffer, Rgba};
use nalgebra::{Matrix3, Point2, Rotation3, UnitQuaternion, Vector2, Vector3};
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use arbit_core::{
    img::build_pyramid,
    init::two_view::{FeatureMatch, TwoViewInitializationParams, TwoViewInitializer},
    math::{projection::px_uv_to_norm_xy, CameraIntrinsics, DistortionModel},
    track::{
        FastSeeder, FastSeederConfig, FeatDescriptor, FeatDescriptorExtractor, FeatureSeederTrait,
        HammingFeatMatcher, Match, OrbDescriptor,
    },
};

use crate::OrbPipeline;

#[derive(Clone, Debug)]
struct ColmapPose {
    // world->camera: X_c = R * X_w + t  (COLMAP convention)
    r_cw: Rotation3<f64>,
    t_cw: Vector3<f64>,
}

pub fn run_benchmark(two_view_path: &Path) -> Result<()> {
    let cameras_txt = two_view_path.join("cameras.txt");
    let images_txt = two_view_path.join("images.txt");
    let img1_path = two_view_path.join("im0.png");
    let img2_path = two_view_path.join("im1.png");

    // 1) Load intrinsics (PINHOLE fx fy cx cy)
    let k = load_intrinsics(&cameras_txt).context("parse cameras.txt")?;
    println!("Intrinsics: {:?}", k);
    // 2) Load the two poses from images.txt (q_wxyz, t, world->camera)
    let (pose1, pose2, name1, name2) = load_two_poses(&images_txt).context("parse images.txt")?;
    println!("Pair: {}  vs  {}", name1, name2);
    println!("Pose1: {:?}", pose1);
    println!("Pose2: {:?}", pose2);

    // 3) Detect+match ORB features
    let (kps1, kps2, good_matches) = orb_match(&img1_path, &img2_path).context("ORB matching")?;

    // 4) Build normalized FeatureMatch array
    let feats = build_feature_matches(&kps1, &kps2, &good_matches, &k);

    // 5) Run your initializer
    let init = TwoViewInitializer::new(TwoViewInitializationParams {
        ransac_iterations: 500,
        ransac_threshold: 5e-4,
        ransac_sample_size: 16,
        min_matches: 80,
        min_parallax: 0.0,
    });
    let res = init.estimate(&feats).context("two-view init failed")?;
    println!(
        "Init: {} inliers, mean Sampson {:.3e}, landmarks {}",
        res.inliers.len(),
        res.average_sampson_error,
        res.landmarks_c1.len()
    );

    // 6) Compute GT relative pose (cam2 wrt cam1) from COLMAP world->camera
    let t21_gt_cam1 = gt_t_c2c1_in_cam1(&pose1, &pose2).normalize();
    println!("t_gt_cam1 (unnorm): {:?}", t21_gt_cam1);
    println!("t_est_cam1 (unnorm): {:?}", res.translation_c2c1);
    println!("t_gt_cam1 (norm):    {:?}", t21_gt_cam1.normalize());
    println!(
        "t_est_cam1 (norm):   {:?}",
        res.translation_c2c1.normalize()
    );
    // 7) Report pose errors (compare estimate against GT relative pose in the correct frame)
    // GT relative rotation: R_gt = R2_cw * R1_cw^T  (maps cam1 coords → cam2 coords)
    let r_gt = pose2.r_cw * pose1.r_cw.transpose();
    let r_err = rotation_angle_deg(&(r_gt.transpose() * res.rotation_c2c1));
    let t_err = angle_between_deg(&t21_gt_cam1, &res.translation_c2c1.normalize());
    println!(
        "Rotation error: {:>6.3} deg | Translation dir error: {:>6.3} deg",
        r_err, t_err
    );

    Ok(())
}

fn load_intrinsics(path: &Path) -> Result<CameraIntrinsics> {
    // COLMAP cameras.txt: "camera_id model width height params..."
    // For PINHOLE, params = fx fy cx cy
    let f = File::open(path)?;
    for line in BufReader::new(f).lines() {
        let line = line?;
        let s = line.trim();
        if s.is_empty() || s.starts_with('#') {
            continue;
        }
        let toks: Vec<&str> = s.split_whitespace().collect();
        if toks.len() >= 8 {
            // toks[1] = model
            let model = toks[1];
            if model != "PINHOLE" {
                bail!("Expected PINHOLE camera, got {model}");
            }
            let width = toks[2].parse::<u32>()?;
            let height = toks[3].parse::<u32>()?;
            let fx = toks[4].parse::<f64>()?;
            let fy = toks[5].parse::<f64>()?;
            let cx = toks[6].parse::<f64>()?;
            let cy = toks[7].parse::<f64>()?;
            return Ok(CameraIntrinsics {
                fx,
                fy,
                cx,
                cy,
                skew: 0.0,
                width,
                height,
                distortion: DistortionModel::None,
            });
        }
    }
    bail!("No valid PINHOLE camera line found in cameras.txt")
}

fn load_two_poses(path: &Path) -> Result<(ColmapPose, ColmapPose, String, String)> {
    // images.txt has blocks:
    // IMAGE_ID qw qx qy qz tx ty tz CAMERA_ID NAME
    // POINTS2D... (ignored)
    let mut poses: Vec<(ColmapPose, String)> = Vec::new();
    let f = File::open(path)?;
    let mut lines = BufReader::new(f).lines();
    while let Some(line) = lines.next() {
        let l = line?;
        let s = l.trim();
        if s.is_empty() || s.starts_with('#') {
            continue;
        }
        let toks: Vec<&str> = s.split_whitespace().collect();
        if toks.len() < 10 {
            continue;
        }

        // Parse quaternion (w, x, y, z) and translation t (world->camera)
        let qw = toks[1].parse::<f64>()?;
        let qx = toks[2].parse::<f64>()?;
        let qy = toks[3].parse::<f64>()?;
        let qz = toks[4].parse::<f64>()?;
        let tx = toks[5].parse::<f64>()?;
        let ty = toks[6].parse::<f64>()?;
        let tz = toks[7].parse::<f64>()?;
        let name = toks[9].to_string();

        let q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(qw, qx, qy, qz));
        let r_cw = Rotation3::from(q);
        let t_cw = Vector3::new(tx, ty, tz);
        poses.push((ColmapPose { r_cw, t_cw }, name));

        // Skip the corresponding POINTS2D line (if present)
        let _ = lines.next();
    }
    if poses.len() < 2 {
        bail!("Found fewer than 2 images in images.txt");
    }
    Ok((
        poses[0].0.clone(),
        poses[1].0.clone(),
        poses[0].1.clone(),
        poses[1].1.clone(),
    ))
}

fn orb_match(
    img1_path: &Path,
    img2_path: &Path,
) -> Result<(
    Vec<FeatDescriptor<[u8; 32]>>,
    Vec<FeatDescriptor<[u8; 32]>>,
    Vec<Match>,
)> {
    let img1 = image::open(img1_path)?.to_luma8();
    let img2 = image::open(img2_path)?.to_luma8();
    if img1.is_empty() || img2.is_empty() {
        bail!("failed to read images");
    }
    let fast_detector = FastSeeder::new(FastSeederConfig::default());
    let orb = OrbDescriptor::new();
    let matcher = HammingFeatMatcher::default();

    let pyramid1 = build_pyramid(&img1, 3);
    let pyramid2 = build_pyramid(&img2, 3);
    let feat1 = fast_detector.seed(&pyramid1);
    let feat2 = fast_detector.seed(&pyramid2);
    let desc1 = orb.describe(&pyramid1, &feat1);
    let desc2 = orb.describe(&pyramid2, &feat2);
    let matches = matcher.match_feats(&desc1, &desc2);
    Ok((desc1, desc2, matches))
}

fn build_feature_matches(
    desc1: &[FeatDescriptor<[u8; 32]>],
    desc2: &[FeatDescriptor<[u8; 32]>],
    matches: &[Match],
    k: &CameraIntrinsics,
) -> Vec<FeatureMatch> {
    let mut out = Vec::with_capacity(matches.len());
    for m in matches.iter() {
        let pt_a = &desc1[m.query_idx].seed.px_uv;
        let pt_b = &desc2[m.train_idx].seed.px_uv;

        out.push(FeatureMatch {
            norm_xy_a: px_uv_to_norm_xy(pt_a.x, pt_a.y, k),
            norm_xy_b: px_uv_to_norm_xy(pt_b.x, pt_b.y, k),
        });
    }
    out
}

fn gt_t_c2c1_in_cam1(p1: &ColmapPose, p2: &ColmapPose) -> Vector3<f64> {
    // Camera centers in world: C_w = -R_cw^T * t_cw
    let c1_w = -(p1.r_cw.transpose() * p1.t_cw);
    let c2_w = -(p2.r_cw.transpose() * p2.t_cw);

    // Baseline in world
    let b_w = c2_w - c1_w;

    // Express baseline in cam1 frame: v_c1 = R1 * v_w
    (p1.r_cw) * b_w
}

fn rotation_angle_deg(r: &Rotation3<f64>) -> f64 {
    let trace = r[(0, 0)] + r[(1, 1)] + r[(2, 2)];
    let cos = ((trace - 1.0) * 0.5).max(-1.0).min(1.0);
    cos.acos().to_degrees()
}

fn angle_between_deg(a: &Vector3<f64>, b: &Vector3<f64>) -> f64 {
    let da = a.normalize().dot(&b.normalize()).max(-1.0).min(1.0);
    da.acos().to_degrees()
}
