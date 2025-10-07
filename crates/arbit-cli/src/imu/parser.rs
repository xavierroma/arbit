use csv::Reader;
use std::fs::File;
use std::path::Path;
use std::time::Duration;

use crate::errors::{Result, VideoProcessingError};
use crate::types::ImuSample;

/// Parser for IMU data from CSV files
pub struct ImuParser;

impl ImuParser {
    /// Parse IMU samples from a CSV file
    ///
    /// Expected format:
    /// timestamp_seconds,accel_x,accel_y,accel_z
    pub fn parse_file<P: AsRef<Path>>(path: P) -> Result<Vec<ImuSample>> {
        let file = File::open(path.as_ref()).map_err(|_| {
            VideoProcessingError::ImuFileNotFound(path.as_ref().display().to_string())
        })?;

        let mut reader = Reader::from_reader(file);
        let mut samples = Vec::new();

        for (line_number, result) in reader.records().enumerate() {
            let record = result.map_err(|e| VideoProcessingError::ImuFormat {
                line: line_number + 2, // +1 for header, +1 for 1-based indexing
                message: format!("CSV error: {}", e),
            })?;

            if record.len() < 4 {
                return Err(VideoProcessingError::ImuFormat {
                    line: line_number + 2,
                    message: format!("Expected at least 4 columns, found {}", record.len()),
                });
            }

            let timestamp_secs: f64 =
                record[0]
                    .parse()
                    .map_err(|e| VideoProcessingError::ImuFormat {
                        line: line_number + 2,
                        message: format!("Invalid timestamp: {}", e),
                    })?;

            let accel_x: f64 = record[1]
                .parse()
                .map_err(|e| VideoProcessingError::ImuFormat {
                    line: line_number + 2,
                    message: format!("Invalid accel_x: {}", e),
                })?;

            let accel_y: f64 = record[2]
                .parse()
                .map_err(|e| VideoProcessingError::ImuFormat {
                    line: line_number + 2,
                    message: format!("Invalid accel_y: {}", e),
                })?;

            let accel_z: f64 = record[3]
                .parse()
                .map_err(|e| VideoProcessingError::ImuFormat {
                    line: line_number + 2,
                    message: format!("Invalid accel_z: {}", e),
                })?;

            samples.push(ImuSample::new(
                Duration::from_secs_f64(timestamp_secs),
                accel_x,
                accel_y,
                accel_z,
            ));
        }

        if samples.is_empty() {
            return Err(VideoProcessingError::ImuFormat {
                line: 0,
                message: "No IMU samples found in file".to_string(),
            });
        }

        log::info!("Loaded {} IMU samples from file", samples.len());
        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn parse_valid_imu_file() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "timestamp_seconds,accel_x,accel_y,accel_z").unwrap();
        writeln!(file, "0.0,-0.123,9.810,0.045").unwrap();
        writeln!(file, "0.01,-0.125,9.808,0.047").unwrap();
        file.flush().unwrap();

        let samples = ImuParser::parse_file(file.path()).unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].accelerometer.1, 9.810);
    }
}
