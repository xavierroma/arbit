mod track_manager;
use crate::track_manager::{TrackConfig, TrackManager};
use arbit_core::db::KeyframeDescriptor;
use arbit_core::img::{build_pyramid, GrayImage, Pyramid, RgbaImage};
use arbit_core::init::{FeatureMatch, TwoViewInitializationParams, TwoViewInitializer};
use arbit_core::map::WorldMap;
use arbit_core::math::projection::{CameraIntrinsics, DistortionModel};
use arbit_core::math::TransformSE3;
use arbit_core::relocalize::{PnPObservation, PnPRansac, PnPRansacParams};
use arbit_core::track::feat_descriptor::FeatDescriptorExtractor;
use arbit_core::track::{
    FastDetectorConfig, FastSeeder, FastSeederConfig, FeatDescriptor, FeatureGridConfig,
    FeatureSeed, FeatureSeederTrait, HammingFeatMatcher, LKTracker, LucasKanadeConfig,
    OrbDescriptor,
};
use ffmpeg_next as ffmpeg;
use imageproc::image::{self, imageops, ImageBuffer, Rgb, Rgba};
use nalgebra::{Point2, Point3, Rotation3, Translation3, UnitQuaternion, Vector2};
use std::collections::VecDeque;
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

/// Convert pixel coordinates to normalized camera coordinates
fn px_uv_to_norm_xy(px: f32, py: f32, intrinsics: &CameraIntrinsics) -> Vector2<f64> {
    Vector2::new(
        (px as f64 - intrinsics.cx) / intrinsics.fx,
        (py as f64 - intrinsics.cy) / intrinsics.fy,
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing subscriber to see logs
    tracing_subscriber::fmt()
        .with_target(false)
        .with_timer(fmt::time::uptime())
        .with_level(true)
        .with_ansi(false)
        .with_span_events(fmt::format::FmtSpan::CLOSE)
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug")),
        )
        .init();

    // Initialize FFmpeg
    ffmpeg::init()?;

    // Check command line arguments
    let args: Vec<String> = std::env::args().collect();
    let use_video = args.len() > 1 && args[1] == "--video";

    let imgs = if use_video {
        let video_path = if args.len() > 2 {
            args[2].clone()
        } else {
            "src/data/vid.MOV".to_string()
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
            "src/data/office1.png",
            "src/data/office2.png",
            // "src/data/img3.png",
            // "src/data/img4.png",
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
    fn process_image(
        &self,
        img: &RgbaImage,
    ) -> (Vec<FeatureSeed>, Vec<FeatDescriptor<[u8; 32]>>, Pyramid) {
        let gray_image = image::imageops::grayscale(&image::DynamicImage::ImageRgba8(img.clone()));
        let pyramid = build_pyramid(&gray_image, self.pyramid_levels);
        let features = self.compute_fast_features(&gray_image, &pyramid);
        let descriptors = self.orb_descriptor.describe(&pyramid, &features);
        (features, descriptors, pyramid)
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
    pub features: Vec<FeatureSeed>,
    pub descriptors: Vec<FeatDescriptor<[u8; 32]>>,
    pub matches: Vec<FeatureMatch>,
    pub keyframe_id: Option<u64>,
    pub pyramid: Pyramid,
}

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
    lk_tracker: LKTracker,
    flow_tracker: TrackManager<LKTracker>,
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
            lk_tracker: LKTracker::new(LucasKanadeConfig::default()),
            flow_tracker: TrackManager::new(
                LKTracker::new(LucasKanadeConfig::default()),
                TrackConfig::default(),
            ),
        }
    }

    pub fn track(&mut self, img: &RgbaImage) -> Result<(), Box<dyn std::error::Error>> {
        let (features, descriptors, pyramid) = self.pipeline.process_image(img);
        let mut frame_data = FrameData {
            frame_id: self.frame_window.len() as u64,
            img: img.clone(),
            features,
            descriptors,
            matches: Vec::new(),
            keyframe_id: None,
            pyramid,
        };

        if self.frame_window.len() < 2 {
            self.log_frame(&frame_data);
            self.frame_window.push_front(frame_data);
            return Ok(());
        }
        match self.tracking_state {
            TrackingState::Initial => {
                for frame in self.frame_window.iter_mut() {
                    let matches = self
                        .matcher
                        .match_feats(&frame.descriptors, &frame_data.descriptors)
                        .iter()
                        .map(|m| {
                            let pt_a = &frame.descriptors[m.query_idx].seed.px_uv;
                            let pt_b = &frame_data.descriptors[m.train_idx].seed.px_uv;

                            FeatureMatch {
                                norm_xy_a: px_uv_to_norm_xy(pt_a.x, pt_a.y, &K),
                                norm_xy_b: px_uv_to_norm_xy(pt_b.x, pt_b.y, &K),
                            }
                        })
                        .collect::<Vec<_>>();
                    let two_view_result = self.two_view_initializer.estimate(&matches);
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
                        )> = tv
                            .landmarks_c1
                            .iter()
                            .zip(&matches)
                            .map(|((norm_xy_a, cam_xyz), m)| {
                                // Get the pixel coordinates for the feature in frame A
                                let px_x = (m.norm_xy_a.x * K.fx + K.cx) as u32;
                                let px_y = (m.norm_xy_a.y * K.fy + K.cy) as u32;

                                // Sample the pixel color from the frame
                                let pixel = frame.img.get_pixel(
                                    px_x.min(frame.img.width() - 1),
                                    px_y.min(frame.img.height() - 1),
                                );

                                (*norm_xy_a, *cam_xyz, Some(*pixel))
                            })
                            .collect::<Vec<_>>();

                        let cam_a_keyframe_id = self.map.insert_keyframe(
                            self.pose_wc,
                            frame_descriptor_a.clone(),
                            features_with_colors, // c1 is world so no transformation needed
                        );
                        frame.keyframe_id = cam_a_keyframe_id;
                        frame_data.keyframe_id =
                            self.map
                                .insert_keyframe(pose_c1c2, frame_descriptor_a.clone(), vec![]);
                        self.pose_wc = pose_c1c2;
                        self.tracking_state = TrackingState::Tracking;
                    }
                }
            }
            TrackingState::Tracking => {
                let prev_frame = self.frame_window.front().unwrap();

                let (stats, mut tracks) = self.flow_tracker.advance_alive(
                    &prev_frame.pyramid,
                    &frame_data.pyramid,
                    &K,
                    prev_frame.frame_id,
                    frame_data.frame_id,
                );

                if self.flow_tracker.need_more_features() {
                    let new_tracks = self.flow_tracker.seed_tracks(
                        &prev_frame.features,
                        prev_frame.frame_id,
                        frame_data.frame_id,
                        &prev_frame.pyramid,
                        &frame_data.pyramid,
                        &K,
                    );
                    tracks.extend(new_tracks);
                }

                // let pnp_observations = frame_data
                //     .tracks
                //     .iter()
                //     .map(|t| PnPObservation {
                //         world_point: ,
                //         normalized_image: pos_px_to_normalized(
                //             t.refined,
                //             t.level_scale,
                //             &frame_data.intrinsics,
                //         ),
                //     })
                //     .collect::<Vec<_>>();
                // let pnp_result = self.pnp.estimate(&pnp_observations);
                // if let Some(pnp_result) = pnp_result {
                //     // pose_cw from PnP
                //     let pose_cw = pnp_result.pose_cw;
                //     let pose_wc = pose_cw.inverse();
                //     self.pose_wc = pose_wc;
                //     self.trajectory.push(self.pose_wc.translation.vector);
                // }
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
        let translation = self.pose_wc.translation.vector;
        let rotation = self.pose_wc.rotation;

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
        if !frame_data.features.is_empty() {
            let points_2d: Vec<[f32; 2]> = frame_data
                .features
                .iter()
                .map(|f| [f.px_uv.x, f.px_uv.y])
                .collect();

            self.rec
                .log(
                    "world/camera/image/features",
                    &rerun::Points2D::new(points_2d)
                        .with_radii([2.0])
                        .with_colors([rerun::Color::from_rgb(34, 138, 167)]),
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

        if self.map.landmark_count() > 0 && !frame_data.features.is_empty() {
            // Get camera center in world coordinates
            // pose_wc is camera-to-world, so translation gives camera position directly
            let camera_center_world = self.pose_wc.translation.vector;

            // For each keyframe feature, draw a line to its corresponding 3D landmark
            if let Some(keyframe_id) = frame_data.keyframe_id {
                if let Some(keyframe) = self.map.keyframe(keyframe_id) {
                    let observation_lines: Vec<Vec<[f32; 3]>> = keyframe
                        .features()
                        .iter()
                        .map(|feat| {
                            vec![
                                // Start: 3D landmark position
                                [
                                    feat.world_xyz.x as f32,
                                    feat.world_xyz.y as f32,
                                    feat.world_xyz.z as f32,
                                ],
                                // End: camera center
                                [
                                    camera_center_world.x as f32,
                                    camera_center_world.y as f32,
                                    camera_center_world.z as f32,
                                ],
                            ]
                        })
                        .collect();

                    if !observation_lines.is_empty() {
                        self.rec
                            .log(
                                "world/observations",
                                &rerun::LineStrips3D::new(observation_lines)
                                    .with_colors([rerun::Color::from_rgb(100, 200, 255)])
                                    .with_radii([1.0]),
                            )
                            .ok();
                    }
                }
            }
        }

        // 7. Log all keyframe poses in the map
        if self.map.keyframe_count() > 0 {
            for keyframe in self.map.keyframes() {
                // keyframe.pose is camera-to-world (pose of camera in world frame)
                let kf_trans = keyframe.pose.translation.vector;
                let kf_rot = keyframe.pose.rotation;

                self.rec
                    .log(
                        format!("world/keyframes/{}", keyframe.id),
                        &rerun::Transform3D::from_translation_rotation(
                            rerun::Vec3D::new(
                                kf_trans.x as f32,
                                kf_trans.y as f32,
                                kf_trans.z as f32,
                            ),
                            rerun::Quaternion::from_xyzw([
                                kf_rot.i as f32,
                                kf_rot.j as f32,
                                kf_rot.k as f32,
                                kf_rot.w as f32,
                            ]),
                        ),
                    )
                    .ok();

                // Optionally: log a small coordinate frame at each keyframe
                self.rec
                    .log(
                        format!("world/keyframes/{}", keyframe.id),
                        &rerun::Boxes3D::from_half_sizes([[0.05, 0.05, 0.05]])
                            .with_colors([rerun::Color::from_rgb(255, 0, 0)]),
                    )
                    .ok();
            }
        }

        // 8. Optional: Log tracking state
        let state_text = match self.tracking_state {
            TrackingState::Initial => "Initial",
            TrackingState::Tracking => "Tracking",
            TrackingState::Lost => "Lost",
        };

        self.rec
            .log(
                "tracking_state",
                &rerun::TextLog::new(state_text).with_level(match self.tracking_state {
                    TrackingState::Tracking => rerun::TextLogLevel::INFO,
                    TrackingState::Initial => rerun::TextLogLevel::WARN,
                    TrackingState::Lost => rerun::TextLogLevel::ERROR,
                }),
            )
            .ok();

        // self.rec
        //     .log(
        //         "stats/landmark_count",
        //         &rerun::Scalar::new(self.map.landmark_count() as f64),
        //     )
        //     .ok();

        // self.rec
        //     .log(
        //         "stats/keyframe_count",
        //         &rerun::Scalar::new(self.map.keyframe_count() as f64),
        //     )
        //     .ok();

        // self.rec
        //     .log(
        //         "stats/feature_count",
        //         &rerun::Scalar::new(frame_data.features.len() as f64),
        //     )
        //     .ok();
    }
}
