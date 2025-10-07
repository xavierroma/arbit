use std::sync::Arc;
use std::time::Duration;

/// Represents a decoded video frame with metadata
#[derive(Clone)]
pub struct VideoFrame {
    /// Timestamp of the frame in the video
    pub timestamp: Duration,
    /// BGRA8 pixel data
    pub data: Arc<[u8]>,
    /// Frame width in pixels
    pub width: u32,
    /// Frame height in pixels
    pub height: u32,
    /// Bytes per row (stride)
    pub bytes_per_row: usize,
}

impl VideoFrame {
    pub fn new(
        timestamp: Duration,
        data: Arc<[u8]>,
        width: u32,
        height: u32,
        bytes_per_row: usize,
    ) -> Self {
        Self {
            timestamp,
            data,
            width,
            height,
            bytes_per_row,
        }
    }

    /// Get the expected data size for this frame
    pub fn expected_size(&self) -> usize {
        self.bytes_per_row * self.height as usize
    }

    /// Check if the frame data is valid
    pub fn is_valid(&self) -> bool {
        self.width > 0
            && self.height > 0
            && self.bytes_per_row >= (self.width as usize * 4)
            && self.data.len() >= self.expected_size()
    }
}
