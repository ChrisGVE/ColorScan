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
//! use inkswatch_colorscan::{analyze_swatch, ColorResult};
//! use std::path::Path;
//!
//! let result = analyze_swatch(Path::new("photo.jpg"))?;
//! println!("Lab: {:?}, Hex: {}", result.lab, result.hex);
//! # Ok::<(), inkswatch_colorscan::AnalysisError>(())
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
pub mod config;
pub mod image_loader;

pub use error::{AnalysisError, Result};
pub use config::PipelineConfig;

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
    /// ISCC-NBS alternate color name - more readable (e.g., "dark purplish blue")
    pub color_name: String,
    /// ISCC-NBS base color (e.g., "blue", "red", "gray")
    pub base_color: String,
    /// ISCC-NBS standard descriptor (e.g., "d. pB" for "dark purplish blue")
    pub tone: String,
    /// Analysis confidence score (0.0 = low, 1.0 = high)
    pub confidence: f32,
    /// Paper card color (hex) extracted from original image before WB correction
    /// Only available in swatch-first mode
    pub card_color_hex: Option<String>,
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
    use crate::image_loader::load_image;

    // Step 1: Load image using unified loader (supports JPEG, PNG, HEIC, etc.)
    let mut image = load_image(image_path)?;

    if image.empty() {
        return Err(AnalysisError::ProcessingError("Image file is empty or corrupted".into()));
    }

    // Step 2: Extract EXIF metadata and apply orientation correction
    let _metadata = ExifExtractor::extract_color_metadata(image_path)
        .ok(); // Ignore errors, EXIF is optional

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

    // Step 4b: Apply white balance correction
    let corrected_image = wb_estimator.apply_correction(&paper_result.rectified_image, paper_color)?;

    // After white balance correction, paper should be neutral white under D65
    let corrected_paper_color = Lab::new(95.0, 0.0, 0.0);

    // Step 5: Detect ink swatch region (using corrected image and corrected paper color)
    let swatch_detector = SwatchDetector::new();
    let swatch_result = swatch_detector.detect(
        &corrected_image,
        &paper_result.foreign_object_mask,
        corrected_paper_color,
    )?;

    // Step 6: Extract representative color from swatch using specified method
    let color_analyzer = ColorAnalyzer::new();
    let color_analysis = color_analyzer.extract_color(
        &corrected_image,
        &swatch_result.swatch_mask,
        corrected_paper_color,
        method,
    )?;

    // Step 7: Convert to multiple color spaces
    let converter = ColorConverter::new();
    let lch = converter.lab_to_lch(color_analysis.lab);
    let srgb = converter.lab_to_srgb(color_analysis.lab);
    let hex = converter.srgb_to_hex(srgb);

    // Step 8: Convert to Munsell notation and ISCC-NBS color names
    let (munsell, color_name, base_color, tone) = srgb_to_munsell_and_names(srgb);

    // Step 9: Return complete result
    Ok(ColorResult {
        lab: color_analysis.lab,
        lch,
        srgb,
        hex,
        munsell,
        color_name,
        base_color,
        tone,
        confidence: color_analysis.confidence,
        card_color_hex: None, // Not available in this pipeline
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
    use crate::image_loader::load_image;

    // Step 1: Load image using unified loader (supports JPEG, PNG, HEIC, etc.)
    let mut image = load_image(image_path)?;

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

    // Step 4b: Apply white balance correction
    let corrected_image = wb_estimator.apply_correction(&paper_result.rectified_image, paper_color)?;

    // After white balance correction, paper should be neutral white under D65
    let corrected_paper_color = Lab::new(95.0, 0.0, 0.0);

    // Step 5: Detect ink swatch region (using corrected image and corrected paper color)
    let swatch_detector = SwatchDetector::new();
    let swatch_result = swatch_detector.detect(
        &corrected_image,
        &paper_result.foreign_object_mask,
        corrected_paper_color,
    )?;

    // Step 6: Extract representative color from swatch (using corrected image and corrected paper color)
    let color_analyzer = ColorAnalyzer::new();
    let color_analysis = color_analyzer.extract_color(
        &corrected_image,
        &swatch_result.swatch_mask,
        corrected_paper_color,
        crate::color::ExtractionMethod::MedianMean,
    )?;

    // Step 7: Convert to multiple color spaces
    let converter = ColorConverter::new();
    let lch = converter.lab_to_lch(color_analysis.lab);
    let srgb = converter.lab_to_srgb(color_analysis.lab);
    let hex = converter.srgb_to_hex(srgb);

    // Step 8: Convert to Munsell notation and ISCC-NBS color names
    let (munsell, color_name, base_color, tone) = srgb_to_munsell_and_names(srgb);

    // Step 9: Return complete result
    Ok(ColorResult {
        lab: color_analysis.lab,
        lch,
        srgb,
        hex,
        munsell,
        color_name,
        base_color,
        tone,
        confidence: color_analysis.confidence,
        card_color_hex: None, // Not available in this pipeline
    })
}

/// Analyze a fountain pen ink swatch with configuration and debug output
///
/// This function performs the full analysis pipeline using parameters from the config
/// and returns intermediate processing images for debugging and validation.
///
/// # Arguments
///
/// * `image_path` - Path to the image file
/// * `config` - Pipeline configuration with all tunable parameters
///
/// # Returns
///
/// A tuple of (`ColorResult`, `DebugOutput`) containing color analysis and debug images
///
/// # Errors
///
/// Returns `AnalysisError` if analysis fails
pub fn analyze_swatch_debug_with_config(image_path: &Path, config: &PipelineConfig) -> Result<(ColorResult, DebugOutput)> {
    use crate::exif::extractor::ExifExtractor;
    use crate::detection::{PaperDetector, SwatchDetector};
    use crate::calibration::white_balance::WhiteBalanceEstimator;
    use crate::color::analysis::ColorAnalyzer;
    use crate::color::conversion::ColorConverter;
    use crate::image_loader::load_image;

    // Step 1: Load image using unified loader (supports JPEG, PNG, HEIC, etc.)
    let mut image = load_image(image_path)?;

    if image.empty() {
        return Err(AnalysisError::ProcessingError("Image file is empty or corrupted".into()));
    }

    // Step 2: Extract EXIF metadata and apply orientation correction
    let _metadata = ExifExtractor::extract_color_metadata(image_path)
        .ok(); // Ignore errors, EXIF is optional

    // Apply EXIF orientation correction if configured
    if config.preprocessing.exif_correction {
        image = apply_exif_orientation(image, image_path)?;
    }

    // Clone original image for debug output (after orientation, before processing)
    let original_image = image.clone();

    // Step 3: Detect paper region and rectify perspective (with config params)
    let paper_detector = PaperDetector::with_params(
        config.paper_detection.min_area_ratio,
        config.paper_detection.max_rectification_angle,
        config.paper_detection.poly_approx_epsilon,
    );
    let paper_result = paper_detector.detect(&image)?;

    // Step 4: Apply white balance correction if configured
    let corrected_image = if config.preprocessing.white_balance.enabled {
        let wb_estimator = WhiteBalanceEstimator::new();
        let paper_color = wb_estimator.estimate_from_paper(
            &paper_result.rectified_image,
            &paper_result.foreign_object_mask,
        )?;
        wb_estimator.apply_correction(&paper_result.rectified_image, paper_color)?
    } else {
        paper_result.rectified_image.clone()
    };

    // After white balance correction, paper should match target color
    let target_paper_lab: Lab = config.preprocessing.white_balance.target_paper.clone().into();

    // Step 5: Detect ink swatch region (with config params)
    let swatch_detector = SwatchDetector::with_params(
        config.swatch_detection.min_delta_e,
        config.swatch_detection.min_area_ratio,
        config.swatch_detection.max_area_ratio,
    );
    let swatch_result = swatch_detector.detect(
        &corrected_image,
        &paper_result.foreign_object_mask,
        target_paper_lab,
    )?;

    // Step 6: Extract representative color from swatch (with config params)
    let color_analyzer = ColorAnalyzer::with_params(
        config.color_extraction.min_ink_delta_e,
        config.color_extraction.outlier_percentile_low,
        config.color_extraction.outlier_percentile_high,
    );

    // Parse extraction method from config
    let method = match config.color_extraction.method.as_str() {
        "MedianMean" => crate::color::ExtractionMethod::MedianMean,
        "Darkest" => crate::color::ExtractionMethod::Darkest,
        "MostSaturated" => crate::color::ExtractionMethod::MostSaturated,
        "Mode" => crate::color::ExtractionMethod::Mode,
        _ => crate::color::ExtractionMethod::MedianMean, // Default fallback
    };

    let color_analysis = color_analyzer.extract_color(
        &corrected_image,
        &swatch_result.swatch_mask,
        target_paper_lab,
        method,
    )?;

    // Step 7: Convert to multiple color spaces
    let converter = ColorConverter::new();
    let lch = converter.lab_to_lch(color_analysis.lab);
    let srgb = converter.lab_to_srgb(color_analysis.lab);
    let hex = converter.srgb_to_hex(srgb);

    // Step 8: Convert to Munsell notation and ISCC-NBS color names
    let (munsell, color_name, base_color, tone) = srgb_to_munsell_and_names(srgb);

    // Step 9: Extract swatch fragment for debug output
    let swatch_fragment = extract_swatch_fragment(&corrected_image, &swatch_result.swatch_mask)?;

    // Step 10: Return results and debug output
    let color_result = ColorResult {
        lab: color_analysis.lab,
        lch,
        srgb,
        hex,
        munsell,
        color_name,
        base_color,
        tone,
        confidence: color_analysis.confidence,
        card_color_hex: None, // Not available in standard pipeline
    };

    let debug_output = DebugOutput {
        original_image,
        corrected_image,
        swatch_fragment,
        swatch_mask: swatch_result.swatch_mask,
    };

    Ok((color_result, debug_output))
}

/// Analyze a fountain pen ink swatch with debug output (using default config)
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
    let config = PipelineConfig::default_experiment_0();
    analyze_swatch_debug_with_config(image_path, &config)
}

/// Analyze using the swatch-first approach
///
/// This pipeline handles cases where rectangle detection finds the swatch
/// rather than the paper card. It estimates white balance from the paper band
/// OUTSIDE the detected rectangle, applies WB to the full image, then
/// finds the swatch within.
///
/// # Key differences from standard pipeline:
/// - WB estimated from band around detected rectangle (not from rectangle interior)
/// - WB applied to full original image (not cropped region)
/// - Works whether detected rectangle is paper or swatch
///
/// # Arguments
///
/// * `image_path` - Path to the image file
/// * `config` - Pipeline configuration with all tunable parameters
///
/// # Returns
///
/// A tuple of (`ColorResult`, `DebugOutput`) containing color analysis and debug images
pub fn analyze_swatch_first_with_config(image_path: &Path, config: &PipelineConfig) -> Result<(ColorResult, DebugOutput)> {
    use crate::exif::extractor::ExifExtractor;
    use crate::detection::{PaperDetector, SwatchDetector};
    use crate::calibration::white_balance::WhiteBalanceEstimator;
    use crate::color::analysis::ColorAnalyzer;
    use crate::color::conversion::ColorConverter;
    use crate::image_loader::load_image;

    // Step 1: Load image using unified loader (supports JPEG, PNG, HEIC, etc.)
    let mut image = load_image(image_path)?;

    if image.empty() {
        return Err(AnalysisError::ProcessingError("Image file is empty or corrupted".into()));
    }

    // Step 2: Extract EXIF metadata and apply orientation correction
    let _metadata = ExifExtractor::extract_color_metadata(image_path)
        .ok();

    if config.preprocessing.exif_correction {
        image = apply_exif_orientation(image, image_path)?;
    }

    // Clone original image for debug output
    let original_image = image.clone();

    // Step 3: Detect rectangle (could be paper or swatch - doesn't matter)
    // We only need the bounding box, not the rectified image
    let paper_detector = PaperDetector::with_params(
        config.paper_detection.min_area_ratio,
        config.paper_detection.max_rectification_angle,
        config.paper_detection.poly_approx_epsilon,
    );

    // Get edges and find contour to extract bounding box
    let rect_bounds = detect_rectangle_bounds(&image, &paper_detector)?;

    // Step 4: Estimate WB from paper band OUTSIDE the detected rectangle
    // This works whether the rectangle is paper or swatch - the band is paper in either case
    let wb_estimator = WhiteBalanceEstimator::new();
    let paper_band_result = wb_estimator.estimate_from_paper_band(&image, rect_bounds)?;

    // Step 5: Apply white balance correction to FULL original image
    let corrected_image = if config.preprocessing.white_balance.enabled {
        wb_estimator.apply_correction(&image, paper_band_result.paper_color)?
    } else {
        image.clone()
    };

    // After white balance correction, paper should match target color
    let target_paper_lab: palette::Lab = config.preprocessing.white_balance.target_paper.clone().into();

    // Step 6: Detect ink swatch in the WB-corrected full image
    // The swatch detector will find ink regions based on delta_E from paper
    let swatch_detector = SwatchDetector::with_params(
        config.swatch_detection.min_delta_e,
        config.swatch_detection.min_area_ratio,
        config.swatch_detection.max_area_ratio,
    );

    // Create empty foreign object mask for full image
    let foreign_mask = opencv::core::Mat::zeros(corrected_image.rows(), corrected_image.cols(), opencv::core::CV_8UC1)
        .map_err(|e| AnalysisError::ProcessingError(format!("Mask creation failed: {}", e)))?
        .to_mat()
        .map_err(|e| AnalysisError::ProcessingError(format!("Mask conversion failed: {}", e)))?;

    let swatch_result = swatch_detector.detect(
        &corrected_image,
        &foreign_mask,
        target_paper_lab,
    )?;

    // Step 7: Extract representative color from swatch
    let color_analyzer = ColorAnalyzer::with_params(
        config.color_extraction.min_ink_delta_e,
        config.color_extraction.outlier_percentile_low,
        config.color_extraction.outlier_percentile_high,
    );

    let method = match config.color_extraction.method.as_str() {
        "MedianMean" => crate::color::ExtractionMethod::MedianMean,
        "Darkest" => crate::color::ExtractionMethod::Darkest,
        "MostSaturated" => crate::color::ExtractionMethod::MostSaturated,
        "Mode" => crate::color::ExtractionMethod::Mode,
        _ => crate::color::ExtractionMethod::MedianMean,
    };

    let color_analysis = color_analyzer.extract_color(
        &corrected_image,
        &swatch_result.swatch_mask,
        target_paper_lab,
        method,
    )?;

    // Step 8: Convert to multiple color spaces
    let converter = ColorConverter::new();
    let lch = converter.lab_to_lch(color_analysis.lab);
    let srgb = converter.lab_to_srgb(color_analysis.lab);
    let hex = converter.srgb_to_hex(srgb);

    // Step 9: Convert to Munsell notation and ISCC-NBS color names
    let (munsell, color_name, base_color, tone) = srgb_to_munsell_and_names(srgb);

    // Step 10: Extract swatch fragment for debug output
    let swatch_fragment = extract_swatch_fragment(&corrected_image, &swatch_result.swatch_mask)?;

    // Step 11: Extract card color from original (pre-WB) image using inverse swatch mask
    let card_color_hex = extract_card_color(&original_image, &swatch_result.swatch_mask);

    // Step 12: Return results and debug output
    let color_result = ColorResult {
        lab: color_analysis.lab,
        lch,
        srgb,
        hex,
        munsell,
        color_name,
        base_color,
        tone,
        confidence: color_analysis.confidence,
        card_color_hex,
    };

    let debug_output = DebugOutput {
        original_image,
        corrected_image,
        swatch_fragment,
        swatch_mask: swatch_result.swatch_mask,
    };

    Ok((color_result, debug_output))
}

/// Detect rectangle bounds from image using edge detection
/// Returns bounding box of detected rectangle (swatch or paper)
fn detect_rectangle_bounds(image: &opencv::core::Mat, _paper_detector: &crate::detection::PaperDetector) -> Result<opencv::core::Rect> {
    use opencv::core::{Point, Rect, Vector};
    use opencv::imgproc::{approx_poly_dp, arc_length, find_contours, CHAIN_APPROX_SIMPLE, RETR_EXTERNAL};

    // Detect edges
    let mut gray = opencv::core::Mat::default();
    opencv::imgproc::cvt_color(image, &mut gray, opencv::imgproc::COLOR_BGR2GRAY, 0, opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT)
        .map_err(|e| AnalysisError::ProcessingError(format!("Grayscale conversion failed: {}", e)))?;

    let mut blurred = opencv::core::Mat::default();
    opencv::imgproc::gaussian_blur(
        &gray,
        &mut blurred,
        opencv::core::Size::new(5, 5),
        1.5,
        1.5,
        opencv::core::BORDER_CONSTANT,
        opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )
    .map_err(|e| AnalysisError::ProcessingError(format!("Gaussian blur failed: {}", e)))?;

    let mut edges = opencv::core::Mat::default();
    opencv::imgproc::canny(&blurred, &mut edges, 30.0, 90.0, 3, false)
        .map_err(|e| AnalysisError::ProcessingError(format!("Canny edge detection failed: {}", e)))?;

    let kernel = opencv::imgproc::get_structuring_element(
        opencv::imgproc::MORPH_RECT,
        opencv::core::Size::new(3, 3),
        Point::new(-1, -1),
    )
    .map_err(|e| AnalysisError::ProcessingError(format!("Kernel creation failed: {}", e)))?;

    let mut dilated = opencv::core::Mat::default();
    opencv::imgproc::dilate(
        &edges,
        &mut dilated,
        &kernel,
        Point::new(-1, -1),
        1,
        opencv::core::BORDER_CONSTANT,
        opencv::core::Scalar::all(0.0),
    )
    .map_err(|e| AnalysisError::ProcessingError(format!("Dilation failed: {}", e)))?;

    // Find contours
    let mut contours: Vector<Vector<Point>> = Vector::new();
    find_contours(&dilated, &mut contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point::new(0, 0))
        .map_err(|e| AnalysisError::ProcessingError(format!("Contour detection failed: {}", e)))?;

    let image_area = (image.rows() * image.cols()) as f64;
    let min_area = image_area * 0.05; // 5% minimum
    let max_area = image_area * 0.90; // 90% maximum - exclude image border contours

    let img_width = image.cols() as f64;
    let img_height = image.rows() as f64;
    let img_center_x = img_width / 2.0;
    let img_center_y = img_height / 2.0;

    let mut best_score = 0.0;
    let mut best_rect: Option<Rect> = None;

    for i in 0..contours.len() {
        let contour = contours.get(i)
            .map_err(|e| AnalysisError::ProcessingError(format!("Contour access failed: {}", e)))?;
        let area = opencv::imgproc::contour_area(&contour, false)
            .map_err(|e| AnalysisError::ProcessingError(format!("Area calculation failed: {}", e)))?;

        // Skip contours that are too small or too large (image border detection)
        if area < min_area || area > max_area {
            continue;
        }

        // Approximate to polygon
        let perimeter = arc_length(&contour, true)
            .map_err(|e| AnalysisError::ProcessingError(format!("Perimeter calculation failed: {}", e)))?;
        let epsilon = perimeter * 0.02;

        let mut approx: Vector<Point> = Vector::new();
        approx_poly_dp(&contour, &mut approx, epsilon, true)
            .map_err(|e| AnalysisError::ProcessingError(format!("Polygon approximation failed: {}", e)))?;

        // Only consider 4-sided polygons
        if approx.len() == 4 {
            let rect = opencv::imgproc::bounding_rect(&contour)
                .map_err(|e| AnalysisError::ProcessingError(format!("Bounding rect failed: {}", e)))?;

            let aspect = rect.width as f64 / rect.height as f64;
            if aspect >= 0.33 && aspect <= 3.0 {
                // Calculate centrality score
                let rect_center_x = rect.x as f64 + rect.width as f64 / 2.0;
                let rect_center_y = rect.y as f64 + rect.height as f64 / 2.0;
                let distance = ((rect_center_x - img_center_x).powi(2) + (rect_center_y - img_center_y).powi(2)).sqrt();
                let img_diagonal = (img_width.powi(2) + img_height.powi(2)).sqrt();
                let centrality = 1.0 - (distance / img_diagonal);

                let score = area * centrality;
                if score > best_score {
                    best_score = score;
                    best_rect = Some(rect);
                }
            }
        }
    }

    // Fallback to centered rectangle if no contour found
    best_rect.ok_or_else(|| {
        // Use center 50% of image as fallback
        let margin_x = (img_width * 0.25) as i32;
        let margin_y = (img_height * 0.25) as i32;
        let _fallback = Rect::new(margin_x, margin_y, image.cols() - 2 * margin_x, image.rows() - 2 * margin_y);
        AnalysisError::NoSwatchDetected("No rectangular region found - using center fallback".into())
    }).or_else(|_| {
        // Return center region as fallback
        let margin_x = (img_width * 0.25) as i32;
        let margin_y = (img_height * 0.25) as i32;
        Ok(Rect::new(margin_x, margin_y, image.cols() - 2 * margin_x, image.rows() - 2 * margin_y))
    })
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
/// Returns (munsell_notation, color_name, base_color, tone)
fn srgb_to_munsell_and_names(srgb: Srgb) -> (String, String, String, String) {
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
            let (color_name, base_color, tone) = match (&munsell_color.hue, munsell_color.chroma) {
                (Some(hue), Some(chroma)) => {
                    IsccNbsClassifier::new()
                        .ok()
                        .and_then(|classifier| {
                            classifier.classify_munsell(hue.as_str(), munsell_color.value, chroma).ok()
                        })
                        .flatten()
                        .map(|metadata| {
                            // ColorMetadata fields:
                            // - alt_color_name: base color for alternate descriptor
                            // - alt_color_descriptor(): full alternate name (e.g., "dark purplish blue")
                            // - iscc_nbs_descriptor(): full standard name
                            let color_name = metadata.alt_color_descriptor();
                            let base_color = metadata.alt_color_name.clone();
                            let standard_name = metadata.iscc_nbs_descriptor();
                            (color_name, base_color, standard_name)
                        })
                        .unwrap_or_else(|| ("N/A".to_string(), "N/A".to_string(), "N/A".to_string()))
                }
                _ => ("N/A".to_string(), "N/A".to_string(), "N/A".to_string()),
            };

            (munsell_str, color_name, base_color, tone)
        }
        Err(_) => ("N/A".to_string(), "N/A".to_string(), "N/A".to_string(), "N/A".to_string()),
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

/// Extract paper card color from original image using inverse of swatch mask
///
/// Samples paper pixels surrounding the ink swatch (within the detected rectangle)
/// from the original image before white balance correction.
///
/// # Arguments
/// * `original_image` - Original image before WB correction
/// * `swatch_mask` - Binary mask where ink pixels are white (255)
///
/// # Returns
/// Hex color string of the paper card, or None if extraction fails
fn extract_card_color(original_image: &opencv::core::Mat, swatch_mask: &opencv::core::Mat) -> Option<String> {
    use opencv::core::{Mat, Rect, Vector};
    use palette::{Lab, Srgb, IntoColor, white_point::D65};

    // Invert swatch mask to get paper pixels
    let mut paper_mask = Mat::default();
    let no_mask = Mat::default();
    if opencv::core::bitwise_not(swatch_mask, &mut paper_mask, &no_mask).is_err() {
        return None;
    }

    // Get bounding rect of swatch to focus on that region
    let mut points = Vector::<opencv::core::Point>::new();
    for y in 0..swatch_mask.rows() {
        for x in 0..swatch_mask.cols() {
            let pixel: u8 = *swatch_mask.at_2d(y, x).ok()?;
            if pixel > 0 {
                points.push(opencv::core::Point::new(x, y));
            }
        }
    }

    if points.len() < 10 {
        return None;
    }

    let bbox = opencv::imgproc::bounding_rect(&points).ok()?;

    // Expand bbox slightly to include surrounding paper (20% padding)
    let pad_x = (bbox.width as f64 * 0.2) as i32;
    let pad_y = (bbox.height as f64 * 0.2) as i32;
    let expanded = Rect::new(
        (bbox.x - pad_x).max(0),
        (bbox.y - pad_y).max(0),
        (bbox.width + 2 * pad_x).min(original_image.cols() - bbox.x + pad_x),
        (bbox.height + 2 * pad_y).min(original_image.rows() - bbox.y + pad_y),
    );

    // Extract paper pixels from expanded region using inverse mask
    let mut lab_values: Vec<[f32; 3]> = Vec::new();

    for y in expanded.y..(expanded.y + expanded.height).min(original_image.rows()) {
        for x in expanded.x..(expanded.x + expanded.width).min(original_image.cols()) {
            // Check if this is a paper pixel (not swatch)
            let mask_val: u8 = *paper_mask.at_2d(y, x).ok()?;
            if mask_val == 0 {
                continue; // This is a swatch pixel, skip
            }

            // Get BGR pixel from original image
            let pixel: &opencv::core::Vec3b = original_image.at_2d(y, x).ok()?;
            let b = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let r = pixel[2] as f32 / 255.0;

            // Convert to Lab
            let srgb: Srgb = Srgb::new(r, g, b);
            let lab: Lab<D65, f32> = srgb.into_color();

            // Filter out very dark or very bright pixels
            if lab.l > 30.0 && lab.l < 100.0 {
                lab_values.push([lab.l, lab.a, lab.b]);
            }
        }
    }

    if lab_values.len() < 100 {
        return None;
    }

    // Calculate median Lab values
    let mut l_vals: Vec<f32> = lab_values.iter().map(|v| v[0]).collect();
    let mut a_vals: Vec<f32> = lab_values.iter().map(|v| v[1]).collect();
    let mut b_vals: Vec<f32> = lab_values.iter().map(|v| v[2]).collect();

    l_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    a_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    b_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mid = lab_values.len() / 2;
    let median_lab: Lab<D65, f32> = Lab::new(l_vals[mid], a_vals[mid], b_vals[mid]);

    // Convert to sRGB and hex
    let card_srgb: Srgb = median_lab.into_color();
    let r = (card_srgb.red.clamp(0.0, 1.0) * 255.0).round() as u8;
    let g = (card_srgb.green.clamp(0.0, 1.0) * 255.0).round() as u8;
    let b = (card_srgb.blue.clamp(0.0, 1.0) * 255.0).round() as u8;

    Some(format!("#{:02X}{:02X}{:02X}", r, g, b))
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
            base_color: "blue".to_string(),
            tone: "vivid".to_string(),
            confidence: 0.85,
            card_color_hex: None,
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: ColorResult = serde_json::from_str(&json).unwrap();

        assert_eq!(result, deserialized);
    }
}