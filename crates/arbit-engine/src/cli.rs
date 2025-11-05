mod bench;
mod track_manager;
use crate::track_manager::{TrackConfig, TrackManager};
use arbit_core::db::KeyframeDescriptor;
use arbit_core::img::{build_pyramid, GrayImage, Pyramid, RgbaImage};
use arbit_core::init::{FeatureMatch, TwoViewInitializationParams, TwoViewInitializer};
use arbit_core::map::WorldMap;
use arbit_core::math::projection::{px_uv_to_norm_xy, CameraIntrinsics, DistortionModel};
use arbit_core::math::TransformSE3;
use arbit_core::relocalize::{PnPObservation, PnPRansac, PnPRansacParams};
use arbit_core::track::feat_descriptor::FeatDescriptorExtractor;
use arbit_core::track::{
    FastDetectorConfig, FastSeeder, FastSeederConfig, FeatDescriptor, FeatureGridConfig,
    FeatureSeed, FeatureSeederTrait, HammingFeatMatcher, LKTracker, LucasKanadeConfig,
    OrbDescriptor, TrackObservation, TrackOutcome,
};
use ffmpeg_next as ffmpeg;
use imageproc::image::{self, imageops, ImageBuffer, Rgba};
use nalgebra::{Matrix3, Point2, Point3, Rotation3, Translation3, UnitQuaternion, Vector2};
use rerun::external::re_log_encoding::external::lz4_flex::frame;
use rerun::Vec2D;
use std::collections::VecDeque;
use std::path::Path;
use tracing::{info, info_span};
use tracing_subscriber::{fmt, EnvFilter};
const K: CameraIntrinsics = CameraIntrinsics {
    fx: 840.164,
    fy: 840.164,
    cx: 640.0,
    cy: 360.0,
    skew: 0.0,
    width: 1280,
    height: 720,
    distortion: DistortionModel::None,
};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing subscriber to see logs
    tracing_subscriber::fmt()
        .with_target(false)
        .with_timer(fmt::time::uptime())
        .with_level(true)
        .with_ansi(false)
        .with_span_events(fmt::format::FmtSpan::CLOSE)
        .with_env_filter(EnvFilter::new("info"))
        .init();

    // Initialize FFmpeg
    ffmpeg::init()?;

    // Check command line arguments
    let args: Vec<String> = std::env::args().collect();
    let use_video = args.len() > 1 && args[1] == "--video";
    let use_benchmark = args.len() > 1 && args[1] == "--benchmark";

    if use_benchmark {
        let two_view_path = if args.len() > 2 {
            Path::new(&args[2])
        } else {
            Path::new("data/two_view")
        };

        bench::run_benchmark(&two_view_path)?;
        return Ok(());
    }

    let imgs = if use_video {
        let video_path = if args.len() > 2 {
            args[2].clone()
        } else {
            "src/data/bedroom.MOV".to_string()
        };

        println!("\n📹 Processing video: {}", video_path);

        let frame_skip = if args.len() > 4 {
            args[4].parse().unwrap_or(1)
        } else {
            1
        };

        println!(
            "   Frame skip: {} (processing every {}th frame)",
            frame_skip, frame_skip
        );

        extract_video_frames(&video_path, frame_skip)?
    } else {
        println!("\n📸 Processing image sequence");
        let paths = [
            "src/data/cafe1.png",
            "src/data/cafe2.png",
            "src/data/cafe3.png",
            "src/data/cafe4.png",
        ];

        paths
            .iter()
            .map(|path| {
                image::open(path)
                    .expect("No image found at provided path")
                    .to_rgba8()
            })
            .collect::<Vec<_>>()
    };

    println!("   Loaded {} frames", imgs.len());

    let mut tracker = Tracker::new();
    for img in imgs.iter() {
        let img_resized = imageops::resize(img, WIDTH, HEIGHT, imageops::FilterType::Lanczos3);
        let img_rotated = imageops::rotate90(&img_resized);
        info!("Tracking frame: {}", tracker.frame_window.len());
        let result = tracker.track(&img_rotated);
        match result {
            Ok(_) => (),
            Err(e) => tracing::error!("Error tracking frame: {}", e),
        };
    }

    Ok(())
}

/// Extract frames from a video file using FFmpeg
fn extract_video_frames(
    video_path: &str,
    frame_skip: usize,
) -> Result<Vec<RgbaImage>, Box<dyn std::error::Error>> {
    let mut input_ctx = ffmpeg::format::input(&video_path)?;

    // Find the video stream
    let video_stream_index = input_ctx
        .streams()
        .best(ffmpeg::media::Type::Video)
        .ok_or("No video stream found")?
        .index();

    // Get video stream info
    let stream = input_ctx.stream(video_stream_index).unwrap();
    let context_decoder = ffmpeg::codec::context::Context::from_parameters(stream.parameters())?;
    let mut decoder = context_decoder.decoder().video()?;

    println!(
        "   Video: {}x{} @ {:.2} fps",
        decoder.width(),
        decoder.height(),
        stream.avg_frame_rate().0 as f64 / stream.avg_frame_rate().1 as f64
    );

    // Create scaler to convert to RGBA
    let mut scaler = ffmpeg::software::scaling::context::Context::get(
        decoder.format(),
        decoder.width(),
        decoder.height(),
        ffmpeg::format::Pixel::RGBA,
        decoder.width(),
        decoder.height(),
        ffmpeg::software::scaling::Flags::BILINEAR,
    )?;

    let mut frames = Vec::new();
    let mut frame_count = 0;
    let mut decoded_count = 0;

    // Process packets
    for (stream, packet) in input_ctx.packets() {
        if stream.index() == video_stream_index {
            decoder.send_packet(&packet)?;

            let mut decoded_frame = ffmpeg::util::frame::video::Video::empty();
            while decoder.receive_frame(&mut decoded_frame).is_ok() {
                // Skip frames based on frame_skip parameter
                if decoded_count % frame_skip != 0 {
                    decoded_count += 1;
                    continue;
                }

                // Create a frame to hold the converted RGBA data
                let mut rgba_frame = ffmpeg::util::frame::video::Video::empty();
                scaler.run(&decoded_frame, &mut rgba_frame)?;

                // Convert FFmpeg frame to image::RgbaImage
                let width = rgba_frame.width();
                let height = rgba_frame.height();
                let data = rgba_frame.data(0);

                // Create RgbaImage from raw data
                if let Some(img) =
                    ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(width, height, data.to_vec())
                {
                    frames.push(img);
                    frame_count += 1;

                    if frame_count % 10 == 0 {
                        print!("\r   Extracted {} frames...", frame_count);
                        std::io::Write::flush(&mut std::io::stdout())?;
                    }
                }

                decoded_count += 1;
            }
        }
    }

    // Flush decoder
    decoder.send_eof()?;
    let mut decoded_frame = ffmpeg::util::frame::video::Video::empty();
    while decoder.receive_frame(&mut decoded_frame).is_ok() {
        if decoded_count % frame_skip != 0 {
            decoded_count += 1;
            continue;
        }

        let mut rgba_frame = ffmpeg::util::frame::video::Video::empty();
        scaler.run(&decoded_frame, &mut rgba_frame)?;

        let width = rgba_frame.width();
        let height = rgba_frame.height();
        let data = rgba_frame.data(0);

        if let Some(img) = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(width, height, data.to_vec())
        {
            frames.push(img);
            frame_count += 1;
        }

        decoded_count += 1;
    }

    println!("\r   Extracted {} frames total", frame_count);

    Ok(frames)
}

struct OrbPipeline {
    fast_detector: FastSeeder,
    orb_descriptor: OrbDescriptor,
    pub pyramid_levels: usize,
}

impl OrbPipeline {
    pub fn new(pyramid_levels: usize) -> Self {
        Self {
            pyramid_levels: pyramid_levels,
            fast_detector: FastSeeder::new(FastSeederConfig {
                grid: FeatureGridConfig::default(),
                detector: FastDetectorConfig::default(),
            }),
            orb_descriptor: OrbDescriptor::new(),
        }
    }
    fn process_image(&self, img: &RgbaImage) -> (Vec<FeatDescriptor<[u8; 32]>>, Pyramid) {
        let _span = info_span!("process_image");
        let gray_image = image::imageops::grayscale(&image::DynamicImage::ImageRgba8(img.clone()));
        let pyramid = build_pyramid(&gray_image, self.pyramid_levels);
        let features = self.compute_fast_features(&gray_image, &pyramid);
        let descriptors = self.orb_descriptor.describe(&pyramid, &features);
        (descriptors, pyramid)
    }
    fn compute_fast_features(
        &self,
        _image: &GrayImage,
        pyramid: &arbit_core::img::Pyramid,
    ) -> Vec<FeatureSeed> {
        let features = self.fast_detector.seed(pyramid);
        tracing::info!("Detected {} features", features.len());
        features
    }
}

#[derive(Clone)]
struct FrameData {
    pub frame_id: u64,
    pub img: RgbaImage,
    pub descriptors: Vec<FeatDescriptor<[u8; 32]>>,
    pub matches: Vec<FeatureMatch>,
    pub keyframe_id: Option<u64>,
    pub pyramid: Pyramid,
    pub tracks: Vec<TrackObservation>,
}

#[derive(Debug)]
enum TrackingState {
    Initial,
    Tracking,
    Lost,
}
struct Tracker {
    matcher: HammingFeatMatcher,
    two_view_initializer: TwoViewInitializer,
    pipeline: OrbPipeline,
    map: WorldMap,
    frame_window: VecDeque<FrameData>,
    tracking_state: TrackingState,
    rec: rerun::RecordingStream,
    pnp: PnPRansac,
    pose_wc: TransformSE3,
    pose_wc_traj: Vec<(TransformSE3, u64)>, //pose_wc, frame_id
    lk_tracker: LKTracker,
    flow_tracker: TrackManager<LKTracker>,
    tracked_descriptors: Vec<(u64, FeatDescriptor<[u8; 32]>, Point3<f64>)>,
}

impl Tracker {
    pub fn new() -> Self {
        Self {
            matcher: HammingFeatMatcher {
                cross_check: true,
                max_distance: Some(80),
                ratio_threshold: None,
            },
            two_view_initializer: TwoViewInitializer::new(TwoViewInitializationParams::default()),
            pipeline: OrbPipeline::new(1),
            map: WorldMap::new(),
            frame_window: VecDeque::new(),
            tracking_state: TrackingState::Initial,
            rec: rerun::RecordingStreamBuilder::new("arbit-orb-matcher")
                .spawn()
                .unwrap(),
            pnp: PnPRansac::new(PnPRansacParams::default()),
            pose_wc: TransformSE3::identity(),
            pose_wc_traj: Vec::new(),
            lk_tracker: LKTracker::new(LucasKanadeConfig::default()),
            flow_tracker: TrackManager::new(
                LKTracker::new(LucasKanadeConfig::default()),
                TrackConfig::default(),
            ),
            tracked_descriptors: Vec::new(),
        }
    }

    pub fn track(&mut self, img: &RgbaImage) -> Result<(), Box<dyn std::error::Error>> {
        let _span = info_span!("track");
        let (descriptors, pyramid) = self.pipeline.process_image(img);
        let mut frame_data = FrameData {
            frame_id: self.frame_window.len() as u64,
            img: img.clone(),
            descriptors,
            matches: Vec::new(),
            keyframe_id: None,
            pyramid,
            tracks: Vec::new(),
        };

        if self.frame_window.is_empty() {
            self.log_frame(&frame_data);
            self.frame_window.push_front(frame_data);
            return Ok(());
        }
        let prev_frame = self.frame_window.front().unwrap();
        let matches = self
            .matcher
            .match_feats(&prev_frame.descriptors, &frame_data.descriptors);
        let feat_matches = matches
            .iter()
            .map(|m| {
                let pt_a = &prev_frame.descriptors[m.query_idx].seed.px_uv;
                let pt_b = &frame_data.descriptors[m.train_idx].seed.px_uv;

                FeatureMatch {
                    norm_xy_a: px_uv_to_norm_xy(pt_a.x, pt_a.y, &K),
                    norm_xy_b: px_uv_to_norm_xy(pt_b.x, pt_b.y, &K),
                }
            })
            .collect::<Vec<_>>();
        self.log_matched_descriptors(&prev_frame, &frame_data, &matches);

        info!(
            "Tracking frameId: {}; state: {:?}",
            frame_data.frame_id, self.tracking_state
        );
        match self.tracking_state {
            TrackingState::Initial => {
                let _span = info_span!("TrackingState::Initial");
                let two_view_result = self.two_view_initializer.estimate(&feat_matches);
                if let Some(tv) = two_view_result {
                    let tv = tv.scaled(5.0);
                    // let rotated_two_view = scaled_two_view.rotate_world_orientation(
                    //     &Rotation3::from(TransformSE3::identity().rotation),
                    // );
                    let rotation_c2c1 = tv.rotation_c2c1.matrix();
                    let translation_c2c1 = tv.translation_c2c1;
                    let rotation_c1c2 = rotation_c2c1.transpose(); // Camera 2 → Camera 1 = Camera 2 → World
                    let translation_c1c2 = -(rotation_c1c2 * translation_c2c1); // Camera 2 → World translation

                    let pose_c1c2 = TransformSE3::from_parts(
                        Translation3::from(translation_c1c2),
                        UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix(
                            &rotation_c1c2,
                        )),
                    );
                    let frame_descriptor_a = KeyframeDescriptor::from_slice(&[0.0; 5]); // TODO: real descriptor

                    let features_with_colors: Vec<(
                        Point2<f64>,
                        Point3<f64>,
                        Option<Rgba<u8>>,
                        FeatDescriptor<[u8; 32]>,
                    )> = tv
                        .landmarks_c1
                        .iter()
                        .zip(feat_matches.iter().zip(matches.iter()))
                        .map(|(landmark, (feat_match, m))| {
                            let (norm_xy_a, cam_xyz) = landmark;
                            // Get the pixel coordinates for the feature in frame A
                            let px_x = (feat_match.norm_xy_a.x * K.fx + K.cx) as u32;
                            let px_y = (feat_match.norm_xy_a.y * K.fy + K.cy) as u32;

                            // Sample the pixel color from the frame
                            let pixel = prev_frame.img.get_pixel(
                                px_x.min(prev_frame.img.width() - 1),
                                px_y.min(prev_frame.img.height() - 1),
                            );
                            let descriptor = prev_frame.descriptors[m.query_idx].clone();

                            (*norm_xy_a, *cam_xyz, Some(*pixel), descriptor)
                        })
                        .collect::<Vec<_>>();

                    let cam_a_keyframe_id = self.map.insert_keyframe(
                        self.pose_wc,
                        frame_descriptor_a.clone(),
                        features_with_colors.clone(), // c1 is world so no transformation needed
                    );
                    self.rec
                        .log(
                            "two_view/cam1",
                            &rerun::Transform3D::from_mat3x3([
                                [1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0],
                            ]),
                        )
                        .ok();
                    self.rec
                        .log(
                            "two_view/cam1/image",
                            &rerun::Image::from_image(prev_frame.img.clone()).unwrap(),
                        )
                        .ok();

                    let pose_c2c1 = pose_c1c2.inverse();

                    self.rec
                        .log(
                            "two_view/cam2",
                            &rerun::Transform3D::from_translation_rotation_scale(
                                rerun::Vec3D::new(
                                    pose_c2c1.translation.vector.x as f32,
                                    pose_c2c1.translation.vector.y as f32,
                                    pose_c2c1.translation.vector.z as f32,
                                ),
                                rerun::Quaternion::from_xyzw([
                                    pose_c2c1.rotation.i as f32,
                                    pose_c2c1.rotation.j as f32,
                                    pose_c2c1.rotation.k as f32,
                                    pose_c2c1.rotation.w as f32,
                                ]),
                                1.0,
                            ),
                        )
                        .ok();
                    self.rec
                        .log(
                            "two_view/cam2/image",
                            &rerun::Image::from_image(frame_data.img.clone()).unwrap(),
                        )
                        .ok();

                    let points = features_with_colors
                        .iter()
                        .map(|(_, world_xyz, _, _)| {
                            [world_xyz.x as f32, world_xyz.y as f32, world_xyz.z as f32]
                        })
                        .collect::<Vec<_>>();
                    let colors = features_with_colors
                        .iter()
                        .map(|(_, _, color, _)| {
                            if let Some(color) = color {
                                rerun::Color::from_rgb(color[0], color[1], color[2])
                            } else {
                                rerun::Color::from_rgb(255, 0, 0)
                            }
                        })
                        .collect::<Vec<_>>();
                    self.rec
                        .log(
                            "map/landmarks",
                            &rerun::Points3D::new(points).with_colors(colors.clone()),
                        )
                        .ok();
                    self.rec
                        .log(
                            "map/landmarks/image",
                            &rerun::Image::from_image(prev_frame.img.clone()).unwrap(),
                        )
                        .ok();
                    let points_2d = features_with_colors
                        .iter()
                        .map(|(_, _, _, d)| [d.seed.px_uv.x as f32, d.seed.px_uv.y as f32])
                        .collect::<Vec<_>>();
                    self.rec
                        .log(
                            "map/landmarks/image/descriptors",
                            &rerun::Points2D::new(points_2d)
                                .with_radii([2.0])
                                .with_colors(colors),
                        )
                        .ok();

                    let keyframe_data = self.map.keyframe(cam_a_keyframe_id.unwrap()).unwrap();

                    self.tracked_descriptors.extend(
                        keyframe_data
                            .features()
                            .iter()
                            .map(|d| (d.landmark_id, d.descriptor.clone(), d.world_xyz)),
                    );
                    self.frame_window.front_mut().unwrap().keyframe_id = cam_a_keyframe_id;
                    frame_data.keyframe_id =
                        self.map
                            .insert_keyframe(pose_c1c2, frame_descriptor_a.clone(), vec![]);
                    self.pose_wc = pose_c1c2;
                    info!("Pose WC: {:?}", self.pose_wc);
                    self.pose_wc_traj.push((self.pose_wc, frame_data.frame_id));
                    self.tracking_state = TrackingState::Tracking;
                } else {
                    info!("Could not initiate two_view")
                }
            }
            TrackingState::Tracking => {
                let _span = info_span!("TrackingState::Tracking");
                // let prev_frame = self.frame_window.front().unwrap();
                // let (_stats, mut tracks) = self.flow_tracker.advance_alive(
                //     &prev_prev_frame.pyramid,
                //     &frame_data.pyramid,
                //     &K,
                //     prev_prev_frame.frame_id,
                //     frame_data.frame_id,
                // );
                // // if self.flow_tracker.need_more_features() {
                // let new_tracks = self.flow_tracker.seed_tracks(
                //     &prev_frame.features,
                //     prev_frame.frame_id,
                //     frame_data.frame_id,
                //     &prev_frame.pyramid,
                //     &frame_data.pyramid,
                //     &K,
                // );
                // tracks.extend(new_tracks);
                // }
                // frame_data.tracks = tracks;
                let known_descriptors = self
                    .tracked_descriptors
                    .iter()
                    .map(|(_, d, _)| d.clone())
                    .collect::<Vec<_>>();
                let feat_matches = self
                    .matcher
                    .match_feats(&known_descriptors, &frame_data.descriptors);

                let pnp_observations = feat_matches
                    .iter()
                    .map(|t| PnPObservation {
                        world_xyz: self.tracked_descriptors[t.query_idx].2,
                        norm_xy: px_uv_to_norm_xy(
                            frame_data.descriptors[t.train_idx].seed.px_uv.x,
                            frame_data.descriptors[t.train_idx].seed.px_uv.y,
                            &K,
                        ),
                    })
                    .collect::<Vec<_>>();
                let pnp_result = self.pnp.estimate(&pnp_observations);
                if let Some(pnp_result) = pnp_result {
                    self.pose_wc = pnp_result.pose_cw.inverse();
                    info!("Pose WC: {:?}", self.pose_wc);
                    self.pose_wc_traj.push((self.pose_wc, frame_data.frame_id));

                    self.rec
                        .log(
                            format!("pnp/cam{}", frame_data.frame_id),
                            &rerun::Transform3D::from_translation_rotation_scale(
                                rerun::Vec3D::new(
                                    pnp_result.pose_cw.translation.vector.x as f32,
                                    pnp_result.pose_cw.translation.vector.y as f32,
                                    pnp_result.pose_cw.translation.vector.z as f32,
                                ),
                                rerun::Quaternion::from_xyzw([
                                    pnp_result.pose_cw.rotation.i as f32,
                                    pnp_result.pose_cw.rotation.j as f32,
                                    pnp_result.pose_cw.rotation.k as f32,
                                    pnp_result.pose_cw.rotation.w as f32,
                                ]),
                                1.0,
                            ),
                        )
                        .ok();
                    self.rec
                        .log(
                            "two_view/cam2/image",
                            &rerun::Image::from_image(frame_data.img.clone()).unwrap(),
                        )
                        .ok();
                }
                // if self.map.should_insert_keyframe(&self.pose_wc) {}
            }
            TrackingState::Lost => {}
        }
        self.log_frame(&frame_data);
        self.frame_window.push_front(frame_data);
        Ok(())
    }

    fn log_frame(&self, frame_data: &FrameData) {
        // Set timeline to current frame
        let frame_count = self.frame_window.len() as i64;
        self.rec.set_time_sequence("frame", frame_count);

        // 1. Log the camera image
        let (width, height) = frame_data.img.dimensions();
        self.rec
            .log(
                "world/camera/image",
                &rerun::Image::from_image(frame_data.img.clone()).unwrap(),
            )
            .ok();

        // 2. Log camera intrinsics (pinhole projection)
        // This defines the projection from 3D camera space to 2D image
        self.rec
            .log(
                "world/camera/image",
                &rerun::Pinhole::from_focal_length_and_resolution(
                    [K.fx as f32, K.fy as f32],
                    [width as f32, height as f32],
                )
                .with_principal_point([K.cx as f32, K.cy as f32]),
            )
            .ok();

        // 3. Log camera pose (extrinsics)
        let pose_cw = self.pose_wc.inverse();
        let translation = pose_cw.translation.vector;
        let rotation = pose_cw.rotation;

        self.rec
            .log(
                "world/camera",
                &rerun::Transform3D::from_translation_rotation(
                    rerun::Vec3D::new(
                        translation.x as f32,
                        translation.y as f32,
                        translation.z as f32,
                    ),
                    rerun::Quaternion::from_xyzw([
                        rotation.i as f32,
                        rotation.j as f32,
                        rotation.k as f32,
                        rotation.w as f32,
                    ]),
                ),
            )
            .ok();

        // 4. Log 2D feature points detected in this frame
        if !frame_data.descriptors.is_empty() {
            let points_2d: Vec<[f32; 2]> = frame_data
                .descriptors
                .iter()
                .map(|d| [d.seed.px_uv.x, d.seed.px_uv.y])
                .collect();

            self.rec
                .log(
                    "world/camera/image/features",
                    &rerun::Points2D::new(points_2d)
                        .with_radii([2.0])
                        .with_colors([rerun::Color::from_rgb(34, 138, 167)]),
                )
                .ok();

            let points_2d: Vec<[f32; 2]> = frame_data
                .descriptors
                .iter()
                .map(|d| [d.seed.px_uv.x, d.seed.px_uv.y])
                .collect();
            let desc_colors: Vec<rerun::Color> = frame_data
                .descriptors
                .iter()
                .map(|d| rerun::Color::from_rgb(255, 0, 0))
                .collect();
            self.rec
                .log(
                    "world/camera/image/descriptors",
                    &rerun::Points2D::new(points_2d)
                        .with_radii([2.0])
                        .with_colors([rerun::Color::from_rgb(255, 0, 0)]),
                )
                .ok();
        }

        // 5. Log feature matches as lines between consecutive frames
        if !frame_data.matches.is_empty() && self.frame_window.len() >= 1 {
            let _prev_frame = self.frame_window.front().unwrap();

            // Create line segments connecting matched features
            let mut line_strips = Vec::new();
            for m in frame_data.matches.iter().take(100) {
                // Limit to avoid clutter
                // Convert normalized coordinates back to pixel coordinates
                let pt_a = [
                    (m.norm_xy_a.x as f32 * K.fx as f32 + K.cx as f32),
                    (m.norm_xy_a.y as f32 * K.fy as f32 + K.cy as f32),
                ];
                let pt_b = [
                    (m.norm_xy_b.x as f32 * K.fx as f32 + K.cx as f32),
                    (m.norm_xy_b.y as f32 * K.fy as f32 + K.cy as f32),
                ];
                line_strips.push(vec![pt_a, pt_b]);
            }

            if !line_strips.is_empty() {
                self.rec
                    .log(
                        "world/camera/image/matches",
                        &rerun::LineStrips2D::new(line_strips)
                            .with_colors([rerun::Color::from_rgb(0, 255, 0)]),
                    )
                    .ok();
            }
        }

        // 6. Log 3D map landmarks (reconstructed points)
        if self.map.landmark_count() > 0 {
            let landmarks: Vec<[f32; 3]> = self
                .map
                .landmarks_iter()
                .map(|lm| {
                    [
                        lm.world_xyz.x as f32,
                        lm.world_xyz.y as f32,
                        lm.world_xyz.z as f32,
                    ]
                })
                .collect();
            let colors: Vec<rerun::Color> = self
                .map
                .landmarks_iter()
                .map(|lm| {
                    if let Some(color) = lm.color {
                        rerun::Color::from_rgb(color[0], color[1], color[2])
                    } else {
                        rerun::Color::from_rgb(255, 255, 255)
                    }
                })
                .collect();
            self.rec
                .log(
                    "world/points",
                    &rerun::Points3D::new(landmarks)
                        .with_radii([1.0])
                        .with_colors(colors),
                )
                .ok();
        }
    }

    /// Log matched descriptors between two frames with visual lines connecting them
    fn log_matched_descriptors(
        &self,
        prev_frame: &FrameData,
        curr_frame: &FrameData,
        matches: &[arbit_core::track::feat_matcher::Match],
    ) {
        // Set timeline to current frame
        let frame_count = self.frame_window.len() as i64;
        self.rec.set_time_sequence("frame", frame_count);

        // Log previous frame image
        self.rec
            .log(
                "matching/prev_frame/image",
                &rerun::Image::from_image(prev_frame.img.clone()).unwrap(),
            )
            .ok();

        // Log current frame image
        self.rec
            .log(
                "matching/curr_frame/image",
                &rerun::Image::from_image(curr_frame.img.clone()).unwrap(),
            )
            .ok();

        // Log descriptors on previous frame
        if !prev_frame.descriptors.is_empty() {
            let points_2d: Vec<[f32; 2]> = prev_frame
                .descriptors
                .iter()
                .map(|d| [d.seed.px_uv.x, d.seed.px_uv.y])
                .collect();

            self.rec
                .log(
                    "matching/prev_frame/image/descriptors",
                    &rerun::Points2D::new(points_2d)
                        .with_radii([3.0])
                        .with_colors([rerun::Color::from_rgb(255, 100, 100)]),
                )
                .ok();
        }

        // Log descriptors on current frame
        if !curr_frame.descriptors.is_empty() {
            let points_2d: Vec<[f32; 2]> = curr_frame
                .descriptors
                .iter()
                .map(|d| [d.seed.px_uv.x, d.seed.px_uv.y])
                .collect();

            self.rec
                .log(
                    "matching/curr_frame/image/descriptors",
                    &rerun::Points2D::new(points_2d)
                        .with_radii([3.0])
                        .with_colors([rerun::Color::from_rgb(100, 100, 255)]),
                )
                .ok();
        }

        // Log match lines on previous frame
        if !matches.is_empty() {
            let mut line_strips_prev = Vec::new();
            for m in matches.iter().take(100) {
                let pt = [
                    prev_frame.descriptors[m.query_idx].seed.px_uv.x,
                    prev_frame.descriptors[m.query_idx].seed.px_uv.y,
                ];
                line_strips_prev.push(vec![pt, pt]); // Single point as line strip
            }

            if !line_strips_prev.is_empty() {
                self.rec
                    .log(
                        "matching/prev_frame/image/matched_points",
                        &rerun::LineStrips2D::new(line_strips_prev)
                            .with_colors([rerun::Color::from_rgb(0, 255, 0)]),
                    )
                    .ok();
            }
        }

        // Log match lines on current frame
        if !matches.is_empty() {
            let mut line_strips_curr = Vec::new();
            for m in matches.iter().take(100) {
                let pt = [
                    curr_frame.descriptors[m.train_idx].seed.px_uv.x,
                    curr_frame.descriptors[m.train_idx].seed.px_uv.y,
                ];
                line_strips_curr.push(vec![pt, pt]); // Single point as line strip
            }

            if !line_strips_curr.is_empty() {
                self.rec
                    .log(
                        "matching/curr_frame/image/matched_points",
                        &rerun::LineStrips2D::new(line_strips_curr)
                            .with_colors([rerun::Color::from_rgb(0, 255, 0)]),
                    )
                    .ok();
            }
        }

        // Create a combined view with both images side by side
        // and lines connecting matched descriptors
        if !matches.is_empty() {
            let (width_prev, height_prev) = prev_frame.img.dimensions();
            let (width_curr, height_curr) = curr_frame.img.dimensions();

            // Stack images horizontally
            let combined_width = width_prev + width_curr;
            let combined_height = height_prev.max(height_curr);

            // Create combined image
            let mut combined_img = RgbaImage::new(combined_width, combined_height);

            // Copy previous frame to left side
            for y in 0..height_prev {
                for x in 0..width_prev {
                    let pixel = prev_frame.img.get_pixel(x, y);
                    combined_img.put_pixel(x, y, *pixel);
                }
            }

            // Copy current frame to right side
            for y in 0..height_curr {
                for x in 0..width_curr {
                    let pixel = curr_frame.img.get_pixel(x, y);
                    combined_img.put_pixel(x + width_prev, y, *pixel);
                }
            }

            // Log combined image
            self.rec
                .log(
                    "matching/combined/image",
                    &rerun::Image::from_image(combined_img).unwrap(),
                )
                .ok();

            // Log match lines across the combined image
            let mut line_strips_combined = Vec::new();
            for m in matches.iter().take(200) {
                let pt_a = [
                    prev_frame.descriptors[m.query_idx].seed.px_uv.x,
                    prev_frame.descriptors[m.query_idx].seed.px_uv.y,
                ];
                let pt_b = [
                    curr_frame.descriptors[m.train_idx].seed.px_uv.x + width_prev as f32,
                    curr_frame.descriptors[m.train_idx].seed.px_uv.y,
                ];
                line_strips_combined.push(vec![pt_a, pt_b]);
            }

            if !line_strips_combined.is_empty() {
                self.rec
                    .log(
                        "matching/combined/image/match_lines",
                        &rerun::LineStrips2D::new(line_strips_combined)
                            .with_colors([rerun::Color::from_rgb(0, 255, 0)]),
                    )
                    .ok();
            }

            // Log matched descriptors on combined image
            let mut matched_points = Vec::new();
            for m in matches.iter().take(200) {
                matched_points.push([
                    prev_frame.descriptors[m.query_idx].seed.px_uv.x,
                    prev_frame.descriptors[m.query_idx].seed.px_uv.y,
                ]);
                matched_points.push([
                    curr_frame.descriptors[m.train_idx].seed.px_uv.x + width_prev as f32,
                    curr_frame.descriptors[m.train_idx].seed.px_uv.y,
                ]);
            }

            if !matched_points.is_empty() {
                self.rec
                    .log(
                        "matching/combined/image/matched_keypoints",
                        &rerun::Points2D::new(matched_points)
                            .with_radii([3.0])
                            .with_colors([rerun::Color::from_rgb(0, 255, 0)]),
                    )
                    .ok();
            }
        }
    }
}
