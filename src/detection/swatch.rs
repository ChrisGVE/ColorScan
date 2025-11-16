//! Ink swatch boundary detection with foreign object exclusion
//!
//! Implements swatch detection that:
//! - Segments ink regions from paper background using color difference
//! - Excludes foreign objects from detection
//! - Refines boundaries with morphological operations
//! - Returns binary mask suitable for color extraction
//!
//! Algorithm tag: `algo-swatch-boundary-detection`

use opencv::{
    core::{Mat, Point, Scalar, Size, Vector, BORDER_CONSTANT},
    imgproc::{
        contour_area, find_contours, morphology_ex, get_structuring_element,
        CHAIN_APPROX_SIMPLE, MORPH_CLOSE, MORPH_OPEN, MORPH_RECT, RETR_EXTERNAL,
        cvt_color, COLOR_BGR2Lab,
    },
    prelude::*,
    types::VectorOfPoint,
};
use palette::{Lab, Srgb};
use crate::{AnalysisError, Result};

/// Minimum color difference (ΔE) to separate ink from paper
const MIN_DELTA_E: f64 = 15.0;

/// Minimum swatch area as fraction of paper area (10%)
const MIN_SWATCH_AREA_RATIO: f64 = 0.10;

/// Maximum swatch area as fraction of paper area (80%)
const MAX_SWATCH_AREA_RATIO: f64 = 0.80;

/// Morphological kernel size
const MORPH_KERNEL_SIZE: i32 = 3;

/// Boundary erosion size (pixels)
const BOUNDARY_EROSION: i32 = 2;

/// Swatch detection result
#[derive(Debug, Clone)]
pub struct SwatchDetectionResult {
    /// Binary mask of ink swatch (true = ink)
    pub swatch_mask: Mat,
    /// Swatch boundary contour
    pub swatch_contour: VectorOfPoint,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
}

/// Swatch detector implementing color-based segmentation with foreign object exclusion
pub struct SwatchDetector {
    min_delta_e: f64,
    min_area_ratio: f64,
    max_area_ratio: f64,
}

impl Default for SwatchDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl SwatchDetector {
    /// Create a new swatch detector with default parameters
    pub fn new() -> Self {
        Self {
            min_delta_e: MIN_DELTA_E,
            min_area_ratio: MIN_SWATCH_AREA_RATIO,
            max_area_ratio: MAX_SWATCH_AREA_RATIO,
        }
    }

    /// Create a swatch detector with custom parameters
    pub fn with_params(min_delta_e: f64, min_area_ratio: f64, max_area_ratio: f64) -> Self {
        Self {
            min_delta_e,
            min_area_ratio,
            max_area_ratio,
        }
    }

    /// Detect ink swatch region in rectified paper image
    ///
    /// # Arguments
    ///
    /// * `image` - Rectified BGR image of paper
    /// * `foreign_mask` - Binary mask of foreign objects
    /// * `paper_color` - Estimated paper color in Lab space
    ///
    /// # Returns
    ///
    /// `SwatchDetectionResult` with swatch mask and metadata
    ///
    /// # Errors
    ///
    /// Returns `AnalysisError` if:
    /// - Image cannot be processed
    /// - No swatch region detected
    /// - Swatch area outside valid range
    pub fn detect(
        &self,
        image: &Mat,
        foreign_mask: &Mat,
        paper_color: Lab,
    ) -> Result<SwatchDetectionResult> {
        // Step 1: Color-based segmentation
        let binary_mask = self.segment_by_color(image, paper_color)?;

        // Apply foreign object mask
        let masked = self.apply_foreign_mask(&binary_mask, foreign_mask)?;

        // Step 2: Morphological refinement
        let refined = self.morphological_refinement(&masked)?;

        // Step 3: Contour analysis
        let swatch_contour = self.find_swatch_contour(&refined, image)?;

        // Step 4: Boundary refinement
        let final_mask = self.refine_boundary(&refined)?;

        // Step 5: Confidence scoring
        let confidence = self.compute_confidence(&swatch_contour, &final_mask, image)?;

        Ok(SwatchDetectionResult {
            swatch_mask: final_mask,
            swatch_contour,
            confidence,
        })
    }

    /// Segment image by color difference from paper
    fn segment_by_color(&self, image: &Mat, paper_color: Lab) -> Result<Mat> {
        // Convert image to Lab
        let mut lab_image = Mat::default();
        cvt_color(image, &mut lab_image, COLOR_BGR2Lab, 0)
            .map_err(|e| AnalysisError::ProcessingError(format!("Lab conversion failed: {}", e)))?;

        // Create binary mask based on color difference
        let mut binary = Mat::zeros(image.rows(), image.cols(), opencv::core::CV_8UC1)
            .map_err(|e| AnalysisError::ProcessingError(format!("Mask creation failed: {}", e)))?
            .to_mat()
            .map_err(|e| AnalysisError::ProcessingError(format!("Mask conversion failed: {}", e)))?;

        // Compute ΔE for each pixel
        for row in 0..image.rows() {
            for col in 0..image.cols() {
                let pixel = lab_image.at_2d::<opencv::core::Vec3b>(row, col)
                    .map_err(|e| AnalysisError::ProcessingError(format!("Pixel access failed: {}", e)))?;

                // OpenCV Lab: L*[0,255], a*[0,255], b*[0,255]
                // Convert to palette Lab: L[0,100], a[-128,127], b[-128,127]
                let l = (pixel[0] as f32) * 100.0 / 255.0;
                let a = (pixel[1] as f32) - 128.0;
                let b = (pixel[2] as f32) - 128.0;

                let pixel_lab = Lab::new(l, a, b);

                // Compute simple Euclidean distance (approximation of ΔE)
                let delta_e = self.compute_delta_e(pixel_lab, paper_color);

                if delta_e >= self.min_delta_e {
                    *binary.at_2d_mut::<u8>(row, col)
                        .map_err(|e| AnalysisError::ProcessingError(format!("Mask write failed: {}", e)))? = 255;
                }
            }
        }

        Ok(binary)
    }

    /// Compute simple Euclidean color difference (ΔE approximation)
    fn compute_delta_e(&self, color1: Lab, color2: Lab) -> f64 {
        let dl = color1.l - color2.l;
        let da = color1.a - color2.a;
        let db = color1.b - color2.b;
        ((dl * dl + da * da + db * db) as f64).sqrt()
    }

    /// Apply foreign object mask to exclude non-ink regions
    fn apply_foreign_mask(&self, binary: &Mat, foreign_mask: &Mat) -> Result<Mat> {
        // Invert foreign mask (we want to keep non-foreign regions)
        let mut inverted_mask = Mat::default();
        opencv::core::bitwise_not(foreign_mask, &mut inverted_mask, &Mat::default())
            .map_err(|e| AnalysisError::ProcessingError(format!("Mask inversion failed: {}", e)))?;

        // Apply mask
        let mut result = Mat::default();
        opencv::core::bitwise_and(binary, binary, &mut result, &inverted_mask)
            .map_err(|e| AnalysisError::ProcessingError(format!("Mask application failed: {}", e)))?;

        Ok(result)
    }

    /// Apply morphological operations to refine swatch mask
    fn morphological_refinement(&self, binary: &Mat) -> Result<Mat> {
        let kernel = get_structuring_element(
            MORPH_RECT,
            Size::new(MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
            Point::new(-1, -1),
        )
        .map_err(|e| AnalysisError::ProcessingError(format!("Kernel creation failed: {}", e)))?;

        // Opening: remove small noise
        let mut opened = Mat::default();
        morphology_ex(
            binary,
            &mut opened,
            MORPH_OPEN,
            &kernel,
            Point::new(-1, -1),
            1,
            BORDER_CONSTANT,
            Scalar::default(),
        )
        .map_err(|e| AnalysisError::ProcessingError(format!("Opening failed: {}", e)))?;

        // Closing: fill holes
        let mut closed = Mat::default();
        morphology_ex(
            &opened,
            &mut closed,
            MORPH_CLOSE,
            &kernel,
            Point::new(-1, -1),
            1,
            BORDER_CONSTANT,
            Scalar::default(),
        )
        .map_err(|e| AnalysisError::ProcessingError(format!("Closing failed: {}", e)))?;

        Ok(closed)
    }

    /// Find largest swatch contour within valid area range
    fn find_swatch_contour(&self, binary: &Mat, image: &Mat) -> Result<VectorOfPoint> {
        let mut contours = Vector::<VectorOfPoint>::new();
        find_contours(
            binary,
            &mut contours,
            RETR_EXTERNAL,
            CHAIN_APPROX_SIMPLE,
            Point::new(0, 0),
        )
        .map_err(|e| AnalysisError::ProcessingError(format!("Contour detection failed: {}", e)))?;

        if contours.is_empty() {
            return Err(AnalysisError::NoSwatchDetected("No ink regions found in image".into()));
        }

        let paper_area = (image.rows() * image.cols()) as f64;
        let min_area = paper_area * self.min_area_ratio;
        let max_area = paper_area * self.max_area_ratio;

        // Find largest contour within valid area range
        let mut best_contour: Option<VectorOfPoint> = None;
        let mut best_area = 0.0;

        for i in 0..contours.len() {
            let contour = contours.get(i)
                .map_err(|e| AnalysisError::ProcessingError(format!("Contour access failed: {}", e)))?;
            let area = contour_area(&contour, false)
                .map_err(|e| AnalysisError::ProcessingError(format!("Area calculation failed: {}", e)))?;

            if area >= min_area && area <= max_area && area > best_area {
                best_area = area;
                best_contour = Some(contour);
            }
        }

        best_contour.ok_or_else(|| {
            AnalysisError::SwatchTooSmall(
                format!(
                    "No swatch region found within valid area range ({:.0}%-{:.0}% of paper)",
                    self.min_area_ratio * 100.0,
                    self.max_area_ratio * 100.0
                )
            )
        })
    }

    /// Refine boundary by eroding slightly to avoid edge artifacts
    fn refine_boundary(&self, binary: &Mat) -> Result<Mat> {
        let kernel = get_structuring_element(
            MORPH_RECT,
            Size::new(BOUNDARY_EROSION, BOUNDARY_EROSION),
            Point::new(-1, -1),
        )
        .map_err(|e| AnalysisError::ProcessingError(format!("Kernel creation failed: {}", e)))?;

        let mut eroded = Mat::default();
        opencv::imgproc::erode(
            binary,
            &mut eroded,
            &kernel,
            Point::new(-1, -1),
            1,
            BORDER_CONSTANT,
            Scalar::default(),
        )
        .map_err(|e| AnalysisError::ProcessingError(format!("Erosion failed: {}", e)))?;

        Ok(eroded)
    }

    /// Compute confidence score for swatch detection
    fn compute_confidence(&self, contour: &VectorOfPoint, mask: &Mat, image: &Mat) -> Result<f32> {
        let area = contour_area(contour, false)
            .map_err(|e| AnalysisError::ProcessingError(format!("Area calculation failed: {}", e)))?;
        let perimeter = opencv::imgproc::arc_length(contour, true)
            .map_err(|e| AnalysisError::ProcessingError(format!("Perimeter calculation failed: {}", e)))?;

        // Compactness: how circular/regular the shape is
        let compactness = if perimeter > 0.0 {
            (4.0 * std::f64::consts::PI * area) / (perimeter * perimeter)
        } else {
            0.0
        };

        // Size score: preference for mid-range sizes
        let paper_area = (image.rows() * image.cols()) as f64;
        let size_ratio = area / paper_area;
        let size_score = if size_ratio >= 0.2 && size_ratio <= 0.6 {
            1.0
        } else {
            0.5
        };

        // Non-zero pixel ratio in mask
        let non_zero = opencv::core::count_non_zero(mask)
            .map_err(|e| AnalysisError::ProcessingError(format!("Count non-zero failed: {}", e)))? as f64;
        let mask_area = (mask.rows() * mask.cols()) as f64;
        let coverage = non_zero / mask_area;

        // Weighted combination
        let confidence = (0.3 * compactness + 0.4 * size_score + 0.3 * coverage) as f32;

        Ok(confidence.clamp(0.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swatch_detector_creation() {
        let detector = SwatchDetector::new();
        assert_eq!(detector.min_delta_e, MIN_DELTA_E);
        assert_eq!(detector.min_area_ratio, MIN_SWATCH_AREA_RATIO);
        assert_eq!(detector.max_area_ratio, MAX_SWATCH_AREA_RATIO);
    }

    #[test]
    fn test_swatch_detector_custom_params() {
        let detector = SwatchDetector::with_params(20.0, 0.15, 0.70);
        assert_eq!(detector.min_delta_e, 20.0);
        assert_eq!(detector.min_area_ratio, 0.15);
        assert_eq!(detector.max_area_ratio, 0.70);
    }

    #[test]
    fn test_delta_e_computation() {
        let detector = SwatchDetector::new();
        let color1 = Lab::new(50.0, 0.0, 0.0);
        let color2 = Lab::new(50.0, 0.0, 0.0);
        let delta_e = detector.compute_delta_e(color1, color2);
        assert!(delta_e < 0.001); // Should be essentially zero

        let color3 = Lab::new(60.0, 10.0, 10.0);
        let delta_e2 = detector.compute_delta_e(color1, color3);
        assert!(delta_e2 > 10.0); // Should be significant
    }

    // TODO: Add integration tests with sample images
    // - Test with clear ink swatch
    // - Test with gradient swatches
    // - Test with very light inks
    // - Test with foreign objects
    // - Test error cases (no swatch, too small, too large)
}
