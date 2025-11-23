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
    /// Munsell color notation (Hue Value/Chroma)
    pub munsell: String,
    /// ISCC-NBS color name (e.g., "dark purplish blue")
    pub color_name: String,
    /// ISCC-NBS tone modifier (e.g., "dark", "vivid")
    pub tone: String,
    /// Analysis confidence score (0.0 = low, 1.0 = high)
    pub confidence: f32,
}

/// Debug output containing intermediate processing images
#[derive(Debug, Clone)]
pub struct DebugOutput {
    /// Original image (before white balance correction, after orientation)
    pub original_image: opencv::core::Mat,
    /// White balance corrected rectified image (full card)
    pub corrected_image: opencv::core::Mat,
    /// Swatch fragment used for color extraction
    pub swatch_fragment: opencv::core::Mat,
    /// Binary mask showing swatch region
    pub swatch_mask: opencv::core::Mat,
}

/// Analyze a fountain pen ink swatch with a specified extraction method
///
/// This function is the same as `analyze_swatch` but allows specifying
/// the color extraction method for comparison and experimentation.
///
/// # Arguments
///
/// * `image_path` - Path to the image file
/// * `method` - Color extraction method to use
///
/// # Returns
///
/// `ColorResult` containing the extracted color in multiple representations
pub fn analyze_swatch_with_method(image_path: &Path, method: crate::color::ExtractionMethod) -> Result<ColorResult> {
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
    let metadata = ExifExtractor::extract_color_metadata(image_path)
        .ok(); // Ignore errors, EXIF is optional

    // Extract flash usage from EXIF (default to false if not available)
    let flash_used = metadata.as_ref()
        .and_then(|m| m.flash_used)
        .unwrap_or(false);

    // Apply EXIF orientation correction if available
    image = apply_exif_orientation(image, image_path)?;

    // Step 3: Detect paper region and rectify perspective
    let paper_detector = PaperDetector::new();
    let paper_result = paper_detector.detect(&image)?;

    // Step 4: Estimate white balance from paper region
    let wb_estimator = WhiteBalanceEstimator::new();
    let paper_color = wb_estimator.estimate_from_paper(
        &paper_result.rectified_image,
        &paper_result.foreign_object_mask,
    )?;

    // Step 4b: Apply white balance correction (adaptive based on flash)
    let corrected_image = wb_estimator.apply_correction(&paper_result.rectified_image, paper_color, flash_used)?;

    // After white balance correction, paper should be neutral white under D65
    // Target L*=92 (lowered from 95 to reduce washing out)
    let corrected_paper_color = Lab::new(92.0, 0.0, 0.0);

    // Step 5: Detect ink swatch region (using corrected image and corrected paper color)
    let swatch_detector = SwatchDetector::new();
    let swatch_result = swatch_detector.detect(
        &corrected_image,
        &paper_result.foreign_object_mask,
        corrected_paper_color,
    )?;

    // Step 6: Extract representative color from swatch using specified method (with flash awareness)
    let color_analyzer = ColorAnalyzer::new();
    let color_analysis = color_analyzer.extract_color(
        &corrected_image,
        &swatch_result.swatch_mask,
        corrected_paper_color,
        method,
        flash_used,
    )?;

    // Step 7: Convert to multiple color spaces
    let converter = ColorConverter::new();
    let lch = converter.lab_to_lch(color_analysis.lab);
    let srgb = converter.lab_to_srgb(color_analysis.lab);
    let hex = converter.srgb_to_hex(srgb);

    // Step 8: Convert to Munsell notation and ISCC-NBS color names
    let (munsell, color_name, tone) = srgb_to_munsell_and_names(srgb);

    // Step 9: Return complete result
    Ok(ColorResult {
        lab: color_analysis.lab,
        lch,
        srgb,
        hex,
        munsell,
        color_name,
        tone,
        confidence: color_analysis.confidence,
    })
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
    let metadata = ExifExtractor::extract_color_metadata(image_path)
        .ok(); // Ignore errors, EXIF is optional

    // Extract flash usage from EXIF (default to false if not available)
    let flash_used = metadata.as_ref()
        .and_then(|m| m.flash_used)
        .unwrap_or(false);

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

    // Step 4b: Apply white balance correction (adaptive based on flash)
    let corrected_image = wb_estimator.apply_correction(&paper_result.rectified_image, paper_color, flash_used)?;

    // After white balance correction, paper should be neutral white under D65
    // Target L*=92 (lowered from 95 to reduce washing out)
    let corrected_paper_color = Lab::new(92.0, 0.0, 0.0);

    // Step 5: Detect ink swatch region (using corrected image and corrected paper color)
    let swatch_detector = SwatchDetector::new();
    let swatch_result = swatch_detector.detect(
        &corrected_image,
        &paper_result.foreign_object_mask,
        corrected_paper_color,
    )?;

    // Step 6: Extract representative color from swatch (using corrected image and corrected paper color, with flash awareness)
    let color_analyzer = ColorAnalyzer::new();
    let color_analysis = color_analyzer.extract_color(
        &corrected_image,
        &swatch_result.swatch_mask,
        corrected_paper_color,
        crate::color::ExtractionMethod::MedianMean,
        flash_used,
    )?;

    // Step 7: Convert to multiple color spaces
    let converter = ColorConverter::new();
    let lch = converter.lab_to_lch(color_analysis.lab);
    let srgb = converter.lab_to_srgb(color_analysis.lab);
    let hex = converter.srgb_to_hex(srgb);

    // Step 8: Convert to Munsell notation and ISCC-NBS color names
    let (munsell, color_name, tone) = srgb_to_munsell_and_names(srgb);

    // Step 9: Return complete result
    Ok(ColorResult {
        lab: color_analysis.lab,
        lch,
        srgb,
        hex,
        munsell,
        color_name,
        tone,
        confidence: color_analysis.confidence,
    })
}

/// Analyze a fountain pen ink swatch with debug output
///
/// This function performs the same analysis as `analyze_swatch` but also returns
/// intermediate processing images for debugging and visualization.
///
/// # Arguments
///
/// * `image_path` - Path to the image file
///
/// # Returns
///
/// A tuple of (`ColorResult`, `DebugOutput`) containing color analysis and debug images
///
/// # Errors
///
/// Returns `AnalysisError` if analysis fails (same as `analyze_swatch`)
pub fn analyze_swatch_debug(image_path: &Path) -> Result<(ColorResult, DebugOutput)> {
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
    let metadata = ExifExtractor::extract_color_metadata(image_path)
        .ok(); // Ignore errors, EXIF is optional

    // Extract flash usage from EXIF (default to false if not available)
    let flash_used = metadata.as_ref()
        .and_then(|m| m.flash_used)
        .unwrap_or(false);

    // Apply EXIF orientation correction if available
    image = apply_exif_orientation(image, image_path)?;

    // Clone original image for debug output (after orientation, before processing)
    let original_image = image.clone();

    // Step 3: Detect paper region and rectify perspective
    let paper_detector = PaperDetector::new();
    let paper_result = paper_detector.detect(&image)?;

    // Step 4: Estimate white balance from paper region
    let wb_estimator = WhiteBalanceEstimator::new();
    let paper_color = wb_estimator.estimate_from_paper(
        &paper_result.rectified_image,
        &paper_result.foreign_object_mask,
    )?;

    // Step 4b: Apply white balance correction (adaptive based on flash)
    let corrected_image = wb_estimator.apply_correction(&paper_result.rectified_image, paper_color, flash_used)?;

    // After white balance correction, paper should be neutral white under D65
    // Target L*=92 (lowered from 95 to reduce washing out)
    let corrected_paper_color = Lab::new(92.0, 0.0, 0.0);

    // Step 5: Detect ink swatch region (using corrected image and corrected paper color)
    let swatch_detector = SwatchDetector::new();
    let swatch_result = swatch_detector.detect(
        &corrected_image,
        &paper_result.foreign_object_mask,
        corrected_paper_color,
    )?;

    // Step 6: Extract representative color from swatch (with flash awareness)
    let color_analyzer = ColorAnalyzer::new();
    let color_analysis = color_analyzer.extract_color(
        &corrected_image,
        &swatch_result.swatch_mask,
        corrected_paper_color,
        crate::color::ExtractionMethod::MedianMean,
        flash_used,
    )?;

    // Step 7: Convert to multiple color spaces
    let converter = ColorConverter::new();
    let lch = converter.lab_to_lch(color_analysis.lab);
    let srgb = converter.lab_to_srgb(color_analysis.lab);
    let hex = converter.srgb_to_hex(srgb);

    // Step 8: Convert to Munsell notation and ISCC-NBS color names
    let (munsell, color_name, tone) = srgb_to_munsell_and_names(srgb);

    // Step 9: Extract swatch fragment for debug output (using corrected image)
    let swatch_fragment = extract_swatch_fragment(&corrected_image, &swatch_result.swatch_mask)?;

    // Step 10: Return results and debug output
    let color_result = ColorResult {
        lab: color_analysis.lab,
        lch,
        srgb,
        hex,
        munsell,
        color_name,
        tone,
        confidence: color_analysis.confidence,
    };

    let debug_output = DebugOutput {
        original_image,
        corrected_image,
        swatch_fragment,
        swatch_mask: swatch_result.swatch_mask,
    };

    Ok((color_result, debug_output))
}

/// Extract the swatch fragment from the rectified image using the mask
///
/// Returns only the pixels that are actually used for color analysis (masked region).
/// Pixels outside the mask are set to black for visualization.
fn extract_swatch_fragment(image: &opencv::core::Mat, mask: &opencv::core::Mat) -> Result<opencv::core::Mat> {
    // Find bounding rectangle of the swatch region
    let mut contours = opencv::core::Vector::<opencv::core::Vector<opencv::core::Point>>::new();
    opencv::imgproc::find_contours(
        mask,
        &mut contours,
        opencv::imgproc::RETR_EXTERNAL,
        opencv::imgproc::CHAIN_APPROX_SIMPLE,
        opencv::core::Point::new(0, 0),
    )
    .map_err(|e| AnalysisError::ProcessingError(format!("Contour detection failed: {}", e)))?;

    if contours.is_empty() {
        return Err(AnalysisError::NoSwatchDetected("No swatch region in mask".into()));
    }

    // Find largest contour by area
    let mut best_contour_idx = 0;
    let mut best_area = 0.0;

    for i in 0..contours.len() {
        let contour = contours.get(i)
            .map_err(|e| AnalysisError::ProcessingError(format!("Contour access failed: {}", e)))?;
        let area = opencv::imgproc::contour_area(&contour, false)
            .map_err(|e| AnalysisError::ProcessingError(format!("Area calculation failed: {}", e)))?;

        if area > best_area {
            best_area = area;
            best_contour_idx = i;
        }
    }

    // Get bounding rectangle of largest contour
    let contour = contours.get(best_contour_idx)
        .map_err(|e| AnalysisError::ProcessingError(format!("Contour access failed: {}", e)))?;
    let rect = opencv::imgproc::bounding_rect(&contour)
        .map_err(|e| AnalysisError::ProcessingError(format!("Bounding rect failed: {}", e)))?;

    // Extract ROI from both image and mask
    let image_roi = opencv::core::Mat::roi(image, rect)
        .map_err(|e| AnalysisError::ProcessingError(format!("Image ROI extraction failed: {}", e)))?;
    let mask_roi = opencv::core::Mat::roi(mask, rect)
        .map_err(|e| AnalysisError::ProcessingError(format!("Mask ROI extraction failed: {}", e)))?;

    // Clone the image ROI
    let image_cropped = image_roi.try_clone()
        .map_err(|e| AnalysisError::ProcessingError(format!("Image clone failed: {}", e)))?;
    let mask_cropped = mask_roi.try_clone()
        .map_err(|e| AnalysisError::ProcessingError(format!("Mask clone failed: {}", e)))?;

    // Create output with masked pixels only (black background for non-mask pixels)
    let mut fragment = opencv::core::Mat::zeros(image_cropped.rows(), image_cropped.cols(), image_cropped.typ())
        .map_err(|e| AnalysisError::ProcessingError(format!("Output Mat creation failed: {}", e)))?
        .to_mat()
        .map_err(|e| AnalysisError::ProcessingError(format!("Mat conversion failed: {}", e)))?;

    // Copy only the masked pixels
    image_cropped.copy_to_masked(&mut fragment, &mask_cropped)
        .map_err(|e| AnalysisError::ProcessingError(format!("Masked copy failed: {}", e)))?;

    Ok(fragment)
}

/// Convert sRGB color to Munsell notation and ISCC-NBS color names
///
/// Returns (munsell_notation, color_name, tone)
fn srgb_to_munsell_and_names(srgb: Srgb) -> (String, String, String) {
    use munsellspace::{MunsellConverter, IsccNbsClassifier};

    // Convert sRGB [0.0-1.0] to [0-255]
    let r = (srgb.red * 255.0).round() as u8;
    let g = (srgb.green * 255.0).round() as u8;
    let b = (srgb.blue * 255.0).round() as u8;

    // Convert to Munsell using munsellspace
    let munsell_result = MunsellConverter::new()
        .and_then(|converter| converter.srgb_to_munsell([r, g, b]));

    match munsell_result {
        Ok(munsell_color) => {
            let munsell_str = munsell_color.to_string();

            // Try to get ISCC-NBS classification
            // munsell_color.hue is Option<String>, value is f64, chroma is Option<f64>
            let (color_name, tone) = match (&munsell_color.hue, munsell_color.chroma) {
                (Some(hue), Some(chroma)) => {
                    IsccNbsClassifier::new()
                        .ok()
                        .and_then(|classifier| {
                            classifier.classify_munsell(hue.as_str(), munsell_color.value, chroma).ok()
                        })
                        .flatten()
                        .map(|metadata| {
                            // ColorMetadata has iscc_nbs_descriptor() and alt_color_descriptor()
                            let primary_name = metadata.iscc_nbs_descriptor();
                            let alt_name = metadata.alt_color_descriptor();
                            (primary_name, alt_name)
                        })
                        .unwrap_or_else(|| ("N/A".to_string(), "N/A".to_string()))
                }
                _ => ("N/A".to_string(), "N/A".to_string()),
            };

            (munsell_str, color_name, tone)
        }
        Err(_) => ("N/A".to_string(), "N/A".to_string(), "N/A".to_string()),
    }
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
            munsell: "5PB 5/10".to_string(),
            color_name: "vivid blue".to_string(),
            tone: "vivid".to_string(),
            confidence: 0.85,
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: ColorResult = serde_json::from_str(&json).unwrap();

        assert_eq!(result, deserialized);
    }
}