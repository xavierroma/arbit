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
    /// Supported formats:
    /// - Full 6DOF: timestamp_seconds,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z
    /// - Accel-only (legacy): timestamp_seconds,accel_x,accel_y,accel_z
    pub fn parse_file<P: AsRef<Path>>(path: P) -> Result<Vec<ImuSample>> {
        let file = File::open(path.as_ref()).map_err(|_| {
            VideoProcessingError::ImuFileNotFound(path.as_ref().display().to_string())
        })?;

        let mut reader = Reader::from_reader(file);
        let mut samples = Vec::new();
        let mut has_gyro = None;

        for (line_number, result) in reader.records().enumerate() {
            let record = result.map_err(|e| VideoProcessingError::ImuFormat {
                line: line_number + 2, // +1 for header, +1 for 1-based indexing
                message: format!("CSV error: {}", e),
            })?;

            // Determine format from first data row
            if has_gyro.is_none() {
                if record.len() == 7 {
                    has_gyro = Some(true);
                    log::info!("Detected 6DOF IMU format (accel + gyro)");
                } else if record.len() == 4 {
                    has_gyro = Some(false);
                    log::warn!(
                        "Detected legacy 3DOF format (accel only). Gyroscope data will be zeroed."
                    );
                } else {
                    return Err(VideoProcessingError::ImuFormat {
                        line: line_number + 2,
                        message: format!(
                            "Expected 4 (accel-only) or 7 (accel+gyro) columns, found {}",
                            record.len()
                        ),
                    });
                }
            }

            if record.len() != if has_gyro.unwrap() { 7 } else { 4 } {
                return Err(VideoProcessingError::ImuFormat {
                    line: line_number + 2,
                    message: format!(
                        "Inconsistent column count: expected {}, found {}",
                        if has_gyro.unwrap() { 7 } else { 4 },
                        record.len()
                    ),
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

            let sample = if has_gyro.unwrap() {
                let gyro_x: f64 =
                    record[4]
                        .parse()
                        .map_err(|e| VideoProcessingError::ImuFormat {
                            line: line_number + 2,
                            message: format!("Invalid gyro_x: {}", e),
                        })?;

                let gyro_y: f64 =
                    record[5]
                        .parse()
                        .map_err(|e| VideoProcessingError::ImuFormat {
                            line: line_number + 2,
                            message: format!("Invalid gyro_y: {}", e),
                        })?;

                let gyro_z: f64 =
                    record[6]
                        .parse()
                        .map_err(|e| VideoProcessingError::ImuFormat {
                            line: line_number + 2,
                            message: format!("Invalid gyro_z: {}", e),
                        })?;

                ImuSample::new(
                    Duration::from_secs_f64(timestamp_secs),
                    accel_x,
                    accel_y,
                    accel_z,
                    gyro_x,
                    gyro_y,
                    gyro_z,
                )
            } else {
                ImuSample::from_accel_only(
                    Duration::from_secs_f64(timestamp_secs),
                    accel_x,
                    accel_y,
                    accel_z,
                )
            };

            samples.push(sample);
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
    fn parse_legacy_accel_only_file() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "timestamp_seconds,accel_x,accel_y,accel_z").unwrap();
        writeln!(file, "0.0,-0.123,9.810,0.045").unwrap();
        writeln!(file, "0.01,-0.125,9.808,0.047").unwrap();
        file.flush().unwrap();

        let samples = ImuParser::parse_file(file.path()).unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].accelerometer.1, 9.810);
        assert_eq!(samples[0].gyroscope, (0.0, 0.0, 0.0)); // Gyro zeroed
    }

    #[test]
    fn parse_full_6dof_file() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            "timestamp_seconds,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z"
        )
        .unwrap();
        writeln!(file, "0.0,-0.123,9.810,0.045,0.001,-0.002,0.003").unwrap();
        writeln!(file, "0.01,-0.125,9.808,0.047,0.002,-0.001,0.004").unwrap();
        file.flush().unwrap();

        let samples = ImuParser::parse_file(file.path()).unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].accelerometer.1, 9.810);
        assert_eq!(samples[0].gyroscope.0, 0.001);
        assert_eq!(samples[1].gyroscope.2, 0.004);
    }
}
