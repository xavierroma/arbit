use arbit_core::img::{build_pyramid, GrayImage, RgbaImage};
use arbit_core::track::feat_descriptor::FeatDescriptorExtractor;
use arbit_core::track::{
    FastSeeder, FastSeederConfig, FeatDescriptor, FeatureSeed, FeatureSeederTrait,
    HammingFeatMatcher, Match, OrbDescriptor,
};
use ffmpeg_next as ffmpeg;
use imageproc::image::{self, imageops, ImageBuffer, Rgba};
use tracing_subscriber::{fmt, EnvFilter};

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

    // Initialize Rerun
    let rec = rerun::RecordingStreamBuilder::new("arbit-orb-matcher").spawn()?;

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
            "src/data/img1.png",
            "src/data/img2.png",
            "src/data/img3.png",
            "src/data/img4.png",
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

    // Resize frames for consistent processing
    let target_width = 1080;
    let target_height = 1440;
    println!("   Resizing to {}x{}", target_width, target_height);

    let imgs = imgs
        .iter()
        .map(|img| {
            imageops::resize(
                img,
                target_width,
                target_height,
                imageops::FilterType::Lanczos3,
            )
        })
        .collect::<Vec<_>>();

    // Process images through ORB pipeline
    println!("\n🔍 Detecting features...");
    let pipeline = OrbPipeline::new();
    let descriptors = imgs
        .iter()
        .map(|img| pipeline.process_image(img).1)
        .collect::<Vec<_>>();

    // Match consecutive frames
    println!("\n🔗 Matching features...");
    let matcher = HammingFeatMatcher::default();
    let matches: Vec<Vec<Match>> = (0..imgs.len() - 1)
        .map(|i| matcher.match_feats(&descriptors[i], &descriptors[i + 1]))
        .collect();

    println!("\n=== Feature Detection Summary ===");
    for (i, desc) in descriptors.iter().enumerate() {
        println!("Frame {}: {} features detected", i, desc.len());
    }
    println!("\n=== Matching Summary ===");
    for (i, m) in matches.iter().enumerate() {
        println!("Frame {} → {}: {} matches", i, i + 1, m.len());
    }

    // Visualize as a temporal sequence
    println!("\n📊 Launching Rerun visualization...");
    visualize_video_sequence(&rec, &imgs, &descriptors, &matches)?;

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
}

impl OrbPipeline {
    pub fn new() -> Self {
        Self {
            fast_detector: FastSeeder::new(FastSeederConfig::default()),
            orb_descriptor: OrbDescriptor::new(),
        }
    }
    fn process_image(&self, img: &RgbaImage) -> (Vec<FeatureSeed>, Vec<FeatDescriptor<[u8; 32]>>) {
        let gray_image = image::imageops::grayscale(&image::DynamicImage::ImageRgba8(img.clone()));
        let pyramid = build_pyramid(&gray_image, 3);
        let features = self.compute_fast_features(&gray_image, &pyramid);
        let descriptors = self.orb_descriptor.describe(&pyramid, &features);
        (features, descriptors)
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

/// Visualize a video sequence with temporal navigation in Rerun
fn visualize_video_sequence(
    rec: &rerun::RecordingStream,
    imgs: &[RgbaImage],
    descriptors: &[Vec<FeatDescriptor<[u8; 32]>>],
    matches: &[Vec<Match>],
) -> Result<(), Box<dyn std::error::Error>> {
    use rerun::external::glam;

    // Log each frame with its features at specific time points
    for (frame_idx, (img, desc)) in imgs.iter().zip(descriptors.iter()).enumerate() {
        // Set the timeline for this frame
        rec.set_time_sequence("frame", frame_idx as i64);

        // Log the main camera image
        let img_data: Vec<u8> = img.as_raw().to_vec();
        let img_tensor = rerun::external::ndarray::Array3::from_shape_vec(
            (img.height() as usize, img.width() as usize, 4),
            img_data,
        )?;
        rec.log(
            "camera/image",
            &rerun::Image::from_color_model_and_tensor(rerun::ColorModel::RGBA, img_tensor)?,
        )?;

        // Log all detected features
        let all_points: Vec<glam::Vec2> = desc
            .iter()
            .map(|d| glam::vec2(d.seed.px_uv.x, d.seed.px_uv.y))
            .collect();

        rec.log(
            "camera/features/all",
            &rerun::Points2D::new(all_points)
                .with_radii([2.0])
                .with_colors([rerun::Color::from_rgb(255, 100, 100)]),
        )?;

        // Log matched features if we have matches for this frame
        if frame_idx < matches.len() {
            let frame_matches = &matches[frame_idx];
            let matched_points: Vec<glam::Vec2> = frame_matches
                .iter()
                .map(|m| {
                    let pt = &desc[m.query_idx].seed.px_uv;
                    glam::vec2(pt.x, pt.y)
                })
                .collect();

            rec.log(
                "camera/features/matched",
                &rerun::Points2D::new(matched_points)
                    .with_radii([4.0])
                    .with_colors([rerun::Color::from_rgb(100, 255, 100)]),
            )?;
        }

        // If there's a next frame, visualize side-by-side comparison
        if frame_idx < imgs.len() - 1 {
            visualize_frame_pair(
                rec,
                frame_idx,
                img,
                &imgs[frame_idx + 1],
                desc,
                &descriptors[frame_idx + 1],
                &matches[frame_idx],
            )?;
        }
    }

    println!("\n✨ Video sequence visualization ready!");
    println!("   Use the timeline at the bottom of Rerun to scrub through frames");
    println!("   Views available:");
    println!("   - camera/image: Main video feed with features");
    println!("   - comparison/: Side-by-side frame pairs with match lines");

    Ok(())
}

/// Visualize a pair of consecutive frames with matches
fn visualize_frame_pair(
    rec: &rerun::RecordingStream,
    _frame_idx: usize,
    img1: &RgbaImage,
    img2: &RgbaImage,
    descriptors1: &[FeatDescriptor<[u8; 32]>],
    descriptors2: &[FeatDescriptor<[u8; 32]>],
    matches: &[Match],
) -> Result<(), Box<dyn std::error::Error>> {
    use rerun::external::glam;

    let width1 = img1.width();
    let width2 = img2.width();
    let height = img1.height().max(img2.height());
    let total_width = width1 + width2;

    // Create side-by-side comparison image
    let mut combined = RgbaImage::new(total_width, height);
    imageops::overlay(&mut combined, img1, 0, 0);
    imageops::overlay(&mut combined, img2, width1 as i64, 0);

    let combined_data: Vec<u8> = combined.as_raw().to_vec();
    let combined_tensor = rerun::external::ndarray::Array3::from_shape_vec(
        (combined.height() as usize, combined.width() as usize, 4),
        combined_data,
    )?;
    rec.log(
        "comparison/side_by_side",
        &rerun::Image::from_color_model_and_tensor(rerun::ColorModel::RGBA, combined_tensor)?,
    )?;

    // Draw match lines
    let line_strips: Vec<Vec<glam::Vec2>> = matches
        .iter()
        .map(|m| {
            let pt1 = &descriptors1[m.query_idx].seed.px_uv;
            let pt2 = &descriptors2[m.train_idx].seed.px_uv;

            vec![
                glam::vec2(pt1.x, pt1.y),
                glam::vec2(pt2.x + width1 as f32, pt2.y),
            ]
        })
        .collect();

    rec.log(
        "comparison/match_lines",
        &rerun::LineStrips2D::new(line_strips).with_colors([rerun::Color::from_rgb(255, 255, 0)]),
    )?;

    // Draw match points
    let mut all_match_points = Vec::new();
    for m in matches {
        let pt1 = &descriptors1[m.query_idx].seed.px_uv;
        let pt2 = &descriptors2[m.train_idx].seed.px_uv;
        all_match_points.push(glam::vec2(pt1.x, pt1.y));
        all_match_points.push(glam::vec2(pt2.x + width1 as f32, pt2.y));
    }

    rec.log(
        "comparison/match_points",
        &rerun::Points2D::new(all_match_points)
            .with_radii([4.0])
            .with_colors([rerun::Color::from_rgb(100, 255, 100)]),
    )?;

    Ok(())
}
