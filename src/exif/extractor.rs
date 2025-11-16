//! EXIF metadata extraction and parsing
//!
//! Extracts relevant metadata for color calibration including
//! white balance settings, color space, and camera information.

use crate::error::{AnalysisError, Result};
use std::path::Path;

/// EXIF metadata relevant to color analysis
#[derive(Debug, Clone)]
pub struct ColorMetadata {
    /// White balance mode (Auto, Daylight, etc.)
    pub white_balance_mode: Option<String>,
    /// White balance color temperature in Kelvin
    pub color_temperature: Option<f32>,
    /// Color space (sRGB, Adobe RGB, etc.)
    pub color_space: Option<String>,
    /// Flash usage
    pub flash_used: Option<bool>,
    /// Camera make and model
    pub camera_info: Option<CameraInfo>,
    /// ISO sensitivity
    pub iso: Option<u32>,
    /// Exposure settings
    pub exposure: Option<ExposureInfo>,
}

/// Camera identification information
#[derive(Debug, Clone)]
pub struct CameraInfo {
    pub make: String,
    pub model: String,
}

/// Exposure settings from EXIF
#[derive(Debug, Clone)]
pub struct ExposureInfo {
    /// Shutter speed in seconds
    pub shutter_speed: Option<f32>,
    /// Aperture f-number
    pub aperture: Option<f32>,
    /// Exposure compensation in EV
    pub exposure_compensation: Option<f32>,
}

/// EXIF metadata extractor
pub struct ExifExtractor;

impl ExifExtractor {
    /// Extract EXIF orientation value from image file
    ///
    /// Returns the EXIF orientation value (1-8) or 1 (normal) if not found.
    /// See: https://www.impulseadventure.com/photo/exif-orientation.html
    pub fn extract_orientation(image_path: &Path) -> u16 {
        use exif::{In, Reader, Tag};
        use std::fs::File;
        use std::io::BufReader;

        let file = match File::open(image_path) {
            Ok(f) => f,
            Err(_) => return 1, // Default: normal orientation
        };

        let exif = match Reader::new().read_from_container(&mut BufReader::new(&file)) {
            Ok(e) => e,
            Err(_) => return 1,
        };

        exif.get_field(Tag::Orientation, In::PRIMARY)
            .and_then(|f| f.value.get_uint(0))
            .map(|v| v as u16)
            .unwrap_or(1)
    }

    /// Extract color-relevant metadata from image file
    ///
    /// # Arguments
    ///
    /// * `image_path` - Path to image file
    ///
    /// # Returns
    ///
    /// Extracted metadata or error if file cannot be read
    pub fn extract_color_metadata(image_path: &Path) -> Result<ColorMetadata> {
        use exif::{In, Reader, Tag};
        use std::fs::File;
        use std::io::BufReader;

        // Open file
        let file = File::open(image_path)
            .map_err(|e| AnalysisError::image_load("Failed to open image file", e))?;

        // Read EXIF data with continue-on-error mode for robustness
        let exif = Reader::new()
            .continue_on_error(true)
            .read_from_container(&mut BufReader::new(&file))
            .or_else(|e| e.distill_partial_result(|_errors| {
                // Silently ignore errors, use partial data
            }))
            .map_err(|e| AnalysisError::exif("Failed to read EXIF data", e))?;

        // Extract white balance mode
        let white_balance_mode = exif.get_field(Tag::WhiteBalance, In::PRIMARY)
            .and_then(|f| f.value.get_uint(0))
            .and_then(|v| Self::parse_white_balance_mode(&v.to_string()));

        // Extract color temperature (if available)
        // Note: Using Tag::Temperature as ColorTemperature doesn't exist
        let color_temperature = exif.get_field(Tag::Temperature, In::PRIMARY)
            .and_then(|f| f.value.get_uint(0))
            .map(|v| v as f32);

        // Extract color space
        let color_space = exif.get_field(Tag::ColorSpace, In::PRIMARY)
            .and_then(|f| f.value.get_uint(0))
            .and_then(|v| Self::parse_color_space(v as u16));

        // Extract flash usage
        let flash_used = exif.get_field(Tag::Flash, In::PRIMARY)
            .and_then(|f| f.value.get_uint(0))
            .map(|v| v & 0x01 != 0); // Bit 0 indicates flash fired

        // Extract camera make and model
        let camera_info = Self::extract_camera_info_from_exif(&exif);

        // Extract ISO
        let iso = exif.get_field(Tag::ISOSpeed, In::PRIMARY)
            .or_else(|| exif.get_field(Tag::PhotographicSensitivity, In::PRIMARY))
            .and_then(|f| f.value.get_uint(0))
            .map(|v| v as u32);

        // Extract exposure info
        let exposure = Self::extract_exposure_info(&exif);

        Ok(ColorMetadata {
            white_balance_mode,
            color_temperature,
            color_space,
            flash_used,
            camera_info,
            iso,
            exposure,
        })
    }

    /// Parse white balance mode from EXIF string
    fn parse_white_balance_mode(wb_string: &str) -> Option<String> {
        // Map EXIF white balance values to standard names
        // EXIF 2.3 specification: 0 = Auto, 1 = Manual
        match wb_string {
            "0" => Some("Auto".to_string()),
            "1" => Some("Manual".to_string()),
            _ => Some(wb_string.to_string()),
        }
    }

    /// Parse color space from EXIF data
    fn parse_color_space(cs_value: u16) -> Option<String> {
        match cs_value {
            1 => Some("sRGB".to_string()),
            2 => Some("Adobe RGB".to_string()),
            65535 => Some("Uncalibrated".to_string()),
            _ => None,
        }
    }

    /// Extract camera make and model from EXIF data
    fn extract_camera_info_from_exif(exif: &exif::Exif) -> Option<CameraInfo> {
        use exif::{In, Tag, Value};

        let make = exif.get_field(Tag::Make, In::PRIMARY)
            .and_then(|f| match &f.value {
                Value::Ascii(v) => v.first().map(|bytes| {
                    String::from_utf8_lossy(bytes).trim().to_string()
                }),
                _ => None,
            });

        let model = exif.get_field(Tag::Model, In::PRIMARY)
            .and_then(|f| match &f.value {
                Value::Ascii(v) => v.first().map(|bytes| {
                    String::from_utf8_lossy(bytes).trim().to_string()
                }),
                _ => None,
            });

        match (make, model) {
            (Some(make), Some(model)) => Some(CameraInfo { make, model }),
            _ => None,
        }
    }

    /// Extract exposure information from EXIF data
    fn extract_exposure_info(exif: &exif::Exif) -> Option<ExposureInfo> {
        use exif::{In, Tag, Value};

        let shutter_speed = exif.get_field(Tag::ExposureTime, In::PRIMARY)
            .and_then(|f| match &f.value {
                Value::Rational(v) => v.first()
                    .and_then(|r| Self::rational_to_float(r.num, r.denom)),
                _ => None,
            });

        let aperture = exif.get_field(Tag::FNumber, In::PRIMARY)
            .and_then(|f| match &f.value {
                Value::Rational(v) => v.first()
                    .and_then(|r| Self::rational_to_float(r.num, r.denom)),
                _ => None,
            });

        let exposure_compensation = exif.get_field(Tag::ExposureBiasValue, In::PRIMARY)
            .and_then(|f| match &f.value {
                Value::SRational(v) => v.first()
                    .and_then(|r| {
                        if r.denom == 0 {
                            None
                        } else {
                            Some(r.num as f32 / r.denom as f32)
                        }
                    }),
                _ => None,
            });

        if shutter_speed.is_some() || aperture.is_some() || exposure_compensation.is_some() {
            Some(ExposureInfo {
                shutter_speed,
                aperture,
                exposure_compensation,
            })
        } else {
            None
        }
    }

    /// Convert EXIF rational to float
    fn rational_to_float(numerator: u32, denominator: u32) -> Option<f32> {
        if denominator == 0 {
            None
        } else {
            Some(numerator as f32 / denominator as f32)
        }
    }
}

impl Default for ColorMetadata {
    fn default() -> Self {
        Self {
            white_balance_mode: None,
            color_temperature: None,
            color_space: None,
            flash_used: None,
            camera_info: None,
            iso: None,
            exposure: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_space_parsing() {
        assert_eq!(ExifExtractor::parse_color_space(1), Some("sRGB".to_string()));
        assert_eq!(ExifExtractor::parse_color_space(2), Some("Adobe RGB".to_string()));
        assert_eq!(ExifExtractor::parse_color_space(65535), Some("Uncalibrated".to_string()));
        assert_eq!(ExifExtractor::parse_color_space(999), None);
    }

    #[test]
    fn test_rational_conversion() {
        assert_eq!(ExifExtractor::rational_to_float(1, 2), Some(0.5));
        assert_eq!(ExifExtractor::rational_to_float(100, 1), Some(100.0));
        assert_eq!(ExifExtractor::rational_to_float(1, 0), None);
    }

    #[test]
    fn test_color_metadata_default() {
        let metadata = ColorMetadata::default();
        assert!(metadata.white_balance_mode.is_none());
        assert!(metadata.color_temperature.is_none());
        assert!(metadata.camera_info.is_none());
    }

    #[test]
    fn test_white_balance_mode_parsing() {
        assert_eq!(
            ExifExtractor::parse_white_balance_mode("0"),
            Some("Auto".to_string())
        );
        assert_eq!(
            ExifExtractor::parse_white_balance_mode("1"),
            Some("Manual".to_string())
        );
        assert_eq!(
            ExifExtractor::parse_white_balance_mode("999"),
            Some("999".to_string())
        );
    }

    #[test]
    fn test_rational_edge_cases() {
        // Zero numerator
        assert_eq!(ExifExtractor::rational_to_float(0, 1), Some(0.0));

        // Large values
        assert_eq!(ExifExtractor::rational_to_float(1000000, 1), Some(1000000.0));

        // Fractional values
        assert_eq!(ExifExtractor::rational_to_float(1, 3), Some(1.0 / 3.0));
        assert_eq!(ExifExtractor::rational_to_float(2, 3), Some(2.0 / 3.0));
    }

    #[test]
    fn test_exposure_info_creation() {
        let exposure = ExposureInfo {
            shutter_speed: Some(0.01),
            aperture: Some(2.8),
            exposure_compensation: Some(-0.3),
        };

        assert_eq!(exposure.shutter_speed, Some(0.01));
        assert_eq!(exposure.aperture, Some(2.8));
        assert_eq!(exposure.exposure_compensation, Some(-0.3));
    }

    #[test]
    fn test_camera_info_creation() {
        let camera = CameraInfo {
            make: "Apple".to_string(),
            model: "iPhone 14 Pro".to_string(),
        };

        assert_eq!(camera.make, "Apple");
        assert_eq!(camera.model, "iPhone 14 Pro");
    }

    // Integration test with actual EXIF file would go here
    // This requires sample test images which we'll add later
    #[test]
    #[ignore] // Ignore until we have test images
    fn test_extract_color_metadata_from_file() {
        // TODO: Add test with sample JPEG/HEIC file containing EXIF data
        // let metadata = ExifExtractor::extract_color_metadata(Path::new("tests/sample.jpg")).unwrap();
        // assert!(metadata.camera_info.is_some());
    }
}