//! # Scan Colors
//!
//! A Rust crate for analyzing fountain pen ink colors from digital photographs.
//! 
//! This library provides calibrated color measurement by:
//! - Detecting and rectifying paper/card surfaces
//! - Estimating illuminant and applying white balance correction
//! - Isolating ink swatches from background
//! - Extracting representative colors with confidence metrics
//!
//! ## Example
//!
//! ```rust,no_run
//! use scan_colors::{analyze_swatch, ColorResult};
//! use std::path::Path;
//!
//! let result = analyze_swatch(Path::new("photo.jpg"))?;
//! println!("Lab: {:?}, Hex: {}", result.lab, result.hex);
//! # Ok::<(), scan_colors::AnalysisError>(())
//! ```

use std::path::Path;
use palette::{Lab, Lch, Srgb};
use serde::{Deserialize, Serialize};
use opencv::prelude::*;

pub mod error;
pub mod constants;
pub mod calibration;
pub mod detection;
pub mod color;
pub mod exif;

pub use error::{AnalysisError, Result};

/// Complete color analysis result with perceptual and display representations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColorResult {
    /// CIE Lab coordinates (perceptually uniform)
    pub lab: Lab,
    /// CIE LCh coordinates (cylindrical Lab representation)
    pub lch: Lch,
    /// sRGB color for display purposes
    pub srgb: Srgb,
    /// Hexadecimal color representation
    pub hex: String,
    /// Analysis confidence score (0.0 = low, 1.0 = high)
    pub confidence: f32,
}

/// Analyze a fountain pen ink swatch from an image file
///
/// This is the main entry point for color analysis. It processes an image
/// to extract a representative color from a fountain pen ink swatch.
///
/// # Arguments
///
/// * `image_path` - Path to the image file
///
/// # Returns
///
/// A `ColorResult` containing Lab, LCh, sRGB, hex representations and confidence score
///
/// # Errors
///
/// Returns `AnalysisError` if:
/// - Image cannot be loaded or is invalid format
/// - No swatch can be detected in the image
/// - Swatch area is too small for reliable analysis
/// - EXIF data cannot be processed
pub fn analyze_swatch(image_path: &Path) -> Result<ColorResult> {
    use crate::exif::extractor::ExifExtractor;
    use crate::detection::{PaperDetector, SwatchDetector};
    use crate::calibration::white_balance::WhiteBalanceEstimator;
    use crate::color::analysis::ColorAnalyzer;
    use crate::color::conversion::ColorConverter;

    // Step 1: Load image
    let mut image = opencv::imgcodecs::imread(
        image_path.to_str().ok_or_else(|| {
            AnalysisError::ProcessingError("Invalid image path encoding".into())
        })?,
        opencv::imgcodecs::IMREAD_COLOR,
    )
    .map_err(|e| AnalysisError::image_load("Failed to load image", e))?;

    if image.empty() {
        return Err(AnalysisError::ProcessingError("Image file is empty or corrupted".into()));
    }

    // Step 2: Extract EXIF metadata and apply orientation correction
    let _metadata = ExifExtractor::extract_color_metadata(image_path)
        .ok(); // Ignore errors, EXIF is optional (for future illuminant estimation)

    // Apply EXIF orientation correction if available
    image = apply_exif_orientation(image, image_path)?;

    // Step 3: Detect paper region and rectify perspective
    let paper_detector = PaperDetector::new();
    let paper_result = paper_detector.detect(&image)?;

    // Step 4: Estimate white balance from paper region
    // Note: foreign_object_mask has 0 = paper (not foreign), 255 = foreign objects
    // This matches what WhiteBalanceEstimator expects for paper_mask
    let wb_estimator = WhiteBalanceEstimator::new();
    let paper_color = wb_estimator.estimate_from_paper(
        &paper_result.rectified_image,
        &paper_result.foreign_object_mask,
    )?;

    // Step 5: Detect ink swatch region
    let swatch_detector = SwatchDetector::new();
    let swatch_result = swatch_detector.detect(
        &paper_result.rectified_image,
        &paper_result.foreign_object_mask,
        paper_color,
    )?;

    // Step 6: Extract representative color from swatch
    let color_analyzer = ColorAnalyzer::new();
    let color_analysis = color_analyzer.extract_color(
        &paper_result.rectified_image,
        &swatch_result.swatch_mask,
        paper_color,
    )?;

    // Step 7: Convert to multiple color spaces
    let converter = ColorConverter::new();
    let lch = converter.lab_to_lch(color_analysis.lab);
    let srgb = converter.lab_to_srgb(color_analysis.lab);
    let hex = converter.srgb_to_hex(srgb);

    // Step 8: Return complete result
    Ok(ColorResult {
        lab: color_analysis.lab,
        lch,
        srgb,
        hex,
        confidence: color_analysis.confidence,
    })
}

/// Apply EXIF orientation correction to image
///
/// Many cameras (especially smartphones) store images in one orientation but
/// record the intended orientation in EXIF metadata. OpenCV imread() doesn't
/// automatically apply this rotation, so we need to do it manually.
fn apply_exif_orientation(image: opencv::core::Mat, image_path: &Path) -> Result<opencv::core::Mat> {
    use crate::exif::extractor::ExifExtractor;

    // Get EXIF orientation (1-8, default 1 = normal)
    let orientation = ExifExtractor::extract_orientation(image_path);

    // Apply transformation based on orientation value
    // EXIF orientation values: https://www.impulseadventure.com/photo/exif-orientation.html
    let mut result = opencv::core::Mat::default();

    match orientation {
        1 => {
            // Normal - no rotation needed
            return Ok(image);
        }
        2 => {
            // Flip horizontal
            opencv::core::flip(&image, &mut result, 1)
                .map_err(|e| AnalysisError::ProcessingError(format!("Orientation flip failed: {}", e)))?;
        }
        3 => {
            // Rotate 180°
            opencv::core::rotate(&image, &mut result, opencv::core::ROTATE_180)
                .map_err(|e| AnalysisError::ProcessingError(format!("Orientation rotation failed: {}", e)))?;
        }
        4 => {
            // Flip vertical
            opencv::core::flip(&image, &mut result, 0)
                .map_err(|e| AnalysisError::ProcessingError(format!("Orientation flip failed: {}", e)))?;
        }
        5 => {
            // Rotate 90° CW + flip horizontal
            let mut temp = opencv::core::Mat::default();
            opencv::core::rotate(&image, &mut temp, opencv::core::ROTATE_90_CLOCKWISE)
                .map_err(|e| AnalysisError::ProcessingError(format!("Orientation rotation failed: {}", e)))?;
            opencv::core::flip(&temp, &mut result, 1)
                .map_err(|e| AnalysisError::ProcessingError(format!("Orientation flip failed: {}", e)))?;
        }
        6 => {
            // Rotate 90° CW
            opencv::core::rotate(&image, &mut result, opencv::core::ROTATE_90_CLOCKWISE)
                .map_err(|e| AnalysisError::ProcessingError(format!("Orientation rotation failed: {}", e)))?;
        }
        7 => {
            // Rotate 90° CCW + flip horizontal
            let mut temp = opencv::core::Mat::default();
            opencv::core::rotate(&image, &mut temp, opencv::core::ROTATE_90_COUNTERCLOCKWISE)
                .map_err(|e| AnalysisError::ProcessingError(format!("Orientation rotation failed: {}", e)))?;
            opencv::core::flip(&temp, &mut result, 1)
                .map_err(|e| AnalysisError::ProcessingError(format!("Orientation flip failed: {}", e)))?;
        }
        8 => {
            // Rotate 90° CCW
            opencv::core::rotate(&image, &mut result, opencv::core::ROTATE_90_COUNTERCLOCKWISE)
                .map_err(|e| AnalysisError::ProcessingError(format!("Orientation rotation failed: {}", e)))?;
        }
        _ => {
            // Unknown orientation, return as-is
            return Ok(image);
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_result_serialization() {
        let result = ColorResult {
            lab: Lab::new(50.0, 20.0, -30.0),
            lch: Lch::new(50.0, 36.06, 303.69),
            srgb: Srgb::new(0.2, 0.4, 0.8),
            hex: "#3366CC".to_string(),
            confidence: 0.85,
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: ColorResult = serde_json::from_str(&json).unwrap();
        
        assert_eq!(result, deserialized);
    }
}