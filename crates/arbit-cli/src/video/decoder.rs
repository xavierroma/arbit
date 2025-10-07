use ffmpeg_next as ffmpeg;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use crate::errors::{Result, VideoProcessingError};
use crate::types::VideoFrame;

/// Video decoder using FFmpeg
pub struct VideoDecoder {
    width: u32,
    height: u32,
    frame_rate: f64,
    duration: Duration,
}

impl VideoDecoder {
    /// Initialize FFmpeg (call once at program start)
    pub fn init() -> Result<()> {
        ffmpeg::init().map_err(|e| {
            VideoProcessingError::VideoDecoding(format!("Failed to initialize FFmpeg: {}", e))
        })
    }

    /// Open a video file and return decoder
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let input = ffmpeg::format::input(&path).map_err(|e| {
            VideoProcessingError::VideoFileNotFound(format!("{}: {}", path.as_ref().display(), e))
        })?;

        let video_stream = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or_else(|| {
                VideoProcessingError::VideoDecoding("No video stream found".to_string())
            })?;

        let video_codec_context = ffmpeg::codec::context::Context::from_parameters(
            video_stream.parameters(),
        )
        .map_err(|e| {
            VideoProcessingError::VideoDecoding(format!("Failed to create codec context: {}", e))
        })?;

        let decoder = video_codec_context.decoder().video().map_err(|e| {
            VideoProcessingError::VideoDecoding(format!("Failed to create decoder: {}", e))
        })?;

        let width = decoder.width();
        let height = decoder.height();
        let frame_rate = video_stream.avg_frame_rate();
        let frame_rate_f64 = frame_rate.numerator() as f64 / frame_rate.denominator() as f64;

        let duration_secs = video_stream.duration() as f64 * f64::from(video_stream.time_base());
        let duration = Duration::from_secs_f64(duration_secs);

        log::info!(
            "Opened video: {}x{} @ {:.2} fps, duration: {:.2}s",
            width,
            height,
            frame_rate_f64,
            duration_secs
        );

        Ok(Self {
            width,
            height,
            frame_rate: frame_rate_f64,
            duration,
        })
    }

    /// Decode all frames from the video file
    pub fn decode_frames<P: AsRef<Path>>(&self, path: P) -> Result<Vec<VideoFrame>> {
        let mut input = ffmpeg::format::input(&path).map_err(|e| {
            VideoProcessingError::VideoDecoding(format!("Failed to reopen video: {}", e))
        })?;

        let video_stream_index = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or_else(|| {
                VideoProcessingError::VideoDecoding("No video stream found".to_string())
            })?
            .index();

        let video_stream = input.stream(video_stream_index).ok_or_else(|| {
            VideoProcessingError::VideoDecoding("Video stream not found".to_string())
        })?;

        let time_base = video_stream.time_base();

        let video_codec_context = ffmpeg::codec::context::Context::from_parameters(
            video_stream.parameters(),
        )
        .map_err(|e| {
            VideoProcessingError::VideoDecoding(format!("Failed to create codec context: {}", e))
        })?;

        let mut decoder = video_codec_context.decoder().video().map_err(|e| {
            VideoProcessingError::VideoDecoding(format!("Failed to create decoder: {}", e))
        })?;

        let mut scaler = ffmpeg::software::scaling::Context::get(
            decoder.format(),
            decoder.width(),
            decoder.height(),
            ffmpeg::format::Pixel::BGRA,
            self.width,
            self.height,
            ffmpeg::software::scaling::Flags::BILINEAR,
        )
        .map_err(|e| {
            VideoProcessingError::VideoDecoding(format!("Failed to create scaler: {}", e))
        })?;

        let mut frames = Vec::new();
        let receive_frame = |decoder: &mut ffmpeg::decoder::Video,
                             scaler: &mut ffmpeg::software::scaling::Context|
         -> Result<Option<VideoFrame>> {
            let mut decoded = ffmpeg::frame::Video::empty();
            match decoder.receive_frame(&mut decoded) {
                Ok(()) => {
                    let timestamp = decoded.timestamp().unwrap_or(0);
                    let timestamp_secs = timestamp as f64 * f64::from(time_base);
                    let timestamp = Duration::from_secs_f64(timestamp_secs);

                    let mut bgra_frame = ffmpeg::frame::Video::empty();
                    scaler.run(&decoded, &mut bgra_frame).map_err(|e| {
                        VideoProcessingError::VideoDecoding(format!("Failed to scale frame: {}", e))
                    })?;

                    let bytes_per_row = bgra_frame.stride(0);
                    let data = bgra_frame.data(0);
                    let data_arc = Arc::from(data.to_vec());

                    Ok(Some(VideoFrame::new(
                        timestamp,
                        data_arc,
                        self.width,
                        self.height,
                        bytes_per_row,
                    )))
                }
                Err(ffmpeg::Error::Eof) => Ok(None),
                Err(ffmpeg::Error::Other {
                    errno: ffmpeg::error::EAGAIN,
                }) => Ok(None),
                Err(e) => Err(VideoProcessingError::VideoDecoding(format!(
                    "Failed to receive frame: {}",
                    e
                ))),
            }
        };

        for (stream, packet) in input.packets() {
            if stream.index() == video_stream_index {
                decoder.send_packet(&packet).map_err(|e| {
                    VideoProcessingError::VideoDecoding(format!("Failed to send packet: {}", e))
                })?;

                while let Some(frame) = receive_frame(&mut decoder, &mut scaler)? {
                    frames.push(frame);
                }
            }
        }

        // Flush decoder
        decoder.send_eof().map_err(|e| {
            VideoProcessingError::VideoDecoding(format!("Failed to send EOF: {}", e))
        })?;

        while let Some(frame) = receive_frame(&mut decoder, &mut scaler)? {
            frames.push(frame);
        }

        if frames.is_empty() {
            return Err(VideoProcessingError::NoFrames);
        }

        log::info!("Decoded {} frames", frames.len());
        Ok(frames)
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn frame_rate(&self) -> f64 {
        self.frame_rate
    }

    pub fn duration(&self) -> Duration {
        self.duration
    }
}
