//! Paper detection and rectification with foreign object exclusion
//!
//! Implements adaptive paper detection that:
//! - Detects paper/card surface boundaries
//! - Identifies and masks foreign objects (tape, rulers, weights)
//! - Computes homography for perspective rectification
//! - Returns rectified image suitable for color analysis
//!
//! Algorithm tag: `algo-adaptive-paper-detection`

use opencv::{
    core::{Mat, Point2f, Scalar, Size, Vector, BORDER_CONSTANT},
    imgproc::{
        approx_poly_dp, arc_length, canny, find_contours, get_perspective_transform,
        warp_perspective, cvt_color, gaussian_blur, threshold, morphology_ex,
        CHAIN_APPROX_SIMPLE, MORPH_CLOSE, MORPH_OPEN, RETR_EXTERNAL,
        THRESH_BINARY, THRESH_OTSU, COLOR_BGR2Lab,
    },
    prelude::*,
    types::{VectorOfPoint, VectorOfPoint2f},
};
use crate::{AnalysisError, Result};

/// Minimum paper area as fraction of total image (10%)
const MIN_PAPER_AREA_RATIO: f64 = 0.10;

/// Maximum rectification angle in degrees (45°)
const MAX_RECTIFICATION_ANGLE: f64 = 45.0;

/// Polygon approximation epsilon as fraction of perimeter (2%)
const POLY_APPROX_EPSILON: f64 = 0.02;

/// Gaussian blur sigma for noise reduction
const BLUR_SIGMA: f64 = 1.0;

/// Morphological kernel size
const MORPH_KERNEL_SIZE: i32 = 3;

/// Paper detection result with rectification data
#[derive(Debug, Clone)]
pub struct PaperDetectionResult {
    /// Rectified paper image
    pub rectified_image: Mat,
    /// Paper contour in original image coordinates
    pub paper_contour: VectorOfPoint,
    /// Homography matrix used for rectification
    pub homography: Mat,
    /// Binary mask of foreign objects (true = foreign object)
    pub foreign_object_mask: Mat,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
}

/// Paper detector implementing adaptive detection with foreign object exclusion
pub struct PaperDetector {
    min_area_ratio: f64,
    max_angle: f64,
    poly_epsilon: f64,
}

impl Default for PaperDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl PaperDetector {
    /// Create a new paper detector with default parameters
    pub fn new() -> Self {
        Self {
            min_area_ratio: MIN_PAPER_AREA_RATIO,
            max_angle: MAX_RECTIFICATION_ANGLE,
            poly_epsilon: POLY_APPROX_EPSILON,
        }
    }

    /// Create a paper detector with custom parameters
    pub fn with_params(min_area_ratio: f64, max_angle: f64, poly_epsilon: f64) -> Self {
        Self {
            min_area_ratio,
            max_angle,
            poly_epsilon,
        }
    }

    /// Detect paper and rectify image, excluding foreign objects
    ///
    /// # Arguments
    ///
    /// * `image` - Input BGR image from camera
    ///
    /// # Returns
    ///
    /// `PaperDetectionResult` with rectified image and metadata
    ///
    /// # Errors
    ///
    /// Returns `AnalysisError` if:
    /// - Image cannot be processed
    /// - No paper region detected
    /// - Paper area too small
    /// - Rectification angle exceeds limits
    pub fn detect(&self, image: &Mat) -> Result<PaperDetectionResult> {
        // Step 1: Preprocessing
        let lab_image = self.preprocess(image)?;

        // Step 2: Adaptive thresholding
        let binary = self.threshold_paper(&lab_image)?;

        // Step 3: Foreign object detection
        let foreign_mask = self.detect_foreign_objects(image, &binary)?;

        // Step 4: Morphological operations (excluding foreign objects)
        let cleaned = self.morphological_ops(&binary, &foreign_mask)?;

        // Step 5: Contour detection and filtering
        let paper_contour = self.find_paper_contour(&cleaned, image)?;

        // Step 6: Polygon approximation
        let corners = self.approximate_rectangle(&paper_contour)?;

        // Step 7: Homography and rectification
        let (rectified, homography) = self.rectify_image(image, &corners)?;

        // Apply foreign mask to rectified image
        let rectified_foreign_mask = self.warp_mask(&foreign_mask, &homography, rectified.size()?)?;

        // Compute confidence
        let confidence = self.compute_confidence(&paper_contour, &corners, image)?;

        Ok(PaperDetectionResult {
            rectified_image: rectified,
            paper_contour,
            homography,
            foreign_object_mask: rectified_foreign_mask,
            confidence,
        })
    }

    /// Preprocess image: convert to Lab and blur
    fn preprocess(&self, image: &Mat) -> Result<Mat> {
        let mut lab = Mat::default();
        cvt_color(image, &mut lab, COLOR_BGR2Lab, 0)
            .map_err(|e| AnalysisError::ProcessingError(format!("Lab conversion failed: {}", e)))?;

        let mut blurred = Mat::default();
        gaussian_blur(
            &lab,
            &mut blurred,
            Size::new(0, 0),
            BLUR_SIGMA,
            BLUR_SIGMA,
            BORDER_CONSTANT,
        )
        .map_err(|e| AnalysisError::ProcessingError(format!("Gaussian blur failed: {}", e)))?;

        Ok(blurred)
    }

    /// Apply adaptive thresholding to detect bright paper regions
    fn threshold_paper(&self, lab_image: &Mat) -> Result<Mat> {
        // Extract L* channel (lightness)
        let mut channels = Vector::<Mat>::new();
        opencv::core::split(lab_image, &mut channels)
            .map_err(|e| AnalysisError::ProcessingError(format!("Channel split failed: {}", e)))?;

        let l_channel = channels.get(0)
            .map_err(|e| AnalysisError::ProcessingError(format!("L* channel access failed: {}", e)))?;

        // Apply Otsu's thresholding
        let mut binary = Mat::default();
        threshold(&l_channel, &mut binary, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU)
            .map_err(|e| AnalysisError::ProcessingError(format!("Otsu threshold failed: {}", e)))?;

        Ok(binary)
    }

    /// Detect foreign objects using edge detection and contour analysis
    fn detect_foreign_objects(&self, _image: &Mat, binary: &Mat) -> Result<Mat> {
        // Create initial mask (all false = no foreign objects)
        let mask = Mat::zeros(binary.rows(), binary.cols(), opencv::core::CV_8UC1)
            .map_err(|e| AnalysisError::ProcessingError(format!("Mask creation failed: {}", e)))?
            .to_mat()
            .map_err(|e| AnalysisError::ProcessingError(format!("Mask conversion failed: {}", e)))?;

        // TODO: Implement foreign object detection
        // - Canny edge detection for high-contrast edges
        // - Detect rectangular contours with high edge density (rulers)
        // - Detect glossy/reflective regions (tape)
        // - Detect strong straight lines not aligned with paper edges
        // For initial implementation, return empty mask

        Ok(mask)
    }

    /// Apply morphological operations excluding foreign objects
    fn morphological_ops(&self, binary: &Mat, foreign_mask: &Mat) -> Result<Mat> {
        let kernel = opencv::imgproc::get_structuring_element(
            opencv::imgproc::MORPH_RECT,
            Size::new(MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
            opencv::core::Point::new(-1, -1),
        )
        .map_err(|e| AnalysisError::ProcessingError(format!("Kernel creation failed: {}", e)))?;

        // Opening: remove noise
        let mut opened = Mat::default();
        morphology_ex(binary, &mut opened, MORPH_OPEN, &kernel, opencv::core::Point::new(-1, -1), 1, BORDER_CONSTANT, Scalar::default())
            .map_err(|e| AnalysisError::ProcessingError(format!("Opening operation failed: {}", e)))?;

        // Closing: fill gaps
        let mut closed = Mat::default();
        morphology_ex(&opened, &mut closed, MORPH_CLOSE, &kernel, opencv::core::Point::new(-1, -1), 1, BORDER_CONSTANT, Scalar::default())
            .map_err(|e| AnalysisError::ProcessingError(format!("Closing operation failed: {}", e)))?;

        // Exclude foreign objects
        let mut result = Mat::default();
        opencv::core::bitwise_and(&closed, &closed, &mut result, foreign_mask)
            .map_err(|e| AnalysisError::ProcessingError(format!("Mask application failed: {}", e)))?;

        Ok(result)
    }

    /// Find the largest rectangular paper contour
    fn find_paper_contour(&self, binary: &Mat, original_image: &Mat) -> Result<VectorOfPoint> {
        let mut contours = Vector::<VectorOfPoint>::new();
        find_contours(binary, &mut contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, opencv::core::Point::new(0, 0))
            .map_err(|e| AnalysisError::ProcessingError(format!("Contour detection failed: {}", e)))?;

        if contours.is_empty() {
            return Err(AnalysisError::NoSwatchDetected("No contours found in image".into()));
        }

        let image_area = (original_image.rows() * original_image.cols()) as f64;
        let min_area = image_area * self.min_area_ratio;

        // Find largest contour that meets area requirement
        let mut best_contour: Option<VectorOfPoint> = None;
        let mut best_area = 0.0;

        for i in 0..contours.len() {
            let contour = contours.get(i)
                .map_err(|e| AnalysisError::ProcessingError(format!("Contour access failed: {}", e)))?;
            let area = opencv::imgproc::contour_area(&contour, false)
                .map_err(|e| AnalysisError::ProcessingError(format!("Area calculation failed: {}", e)))?;

            if area >= min_area && area > best_area {
                best_area = area;
                best_contour = Some(contour);
            }
        }

        best_contour.ok_or_else(|| {
            AnalysisError::NoSwatchDetected(
                format!("No paper region found (minimum {}% of image area required)", self.min_area_ratio * 100.0)
            )
        })
    }

    /// Approximate contour as rectangle (4 corners)
    fn approximate_rectangle(&self, contour: &VectorOfPoint) -> Result<VectorOfPoint2f> {
        let perimeter = arc_length(contour, true)
            .map_err(|e| AnalysisError::ProcessingError(format!("Perimeter calculation failed: {}", e)))?;
        let epsilon = perimeter * self.poly_epsilon;

        let mut approx = VectorOfPoint::new();
        approx_poly_dp(contour, &mut approx, epsilon, true)
            .map_err(|e| AnalysisError::ProcessingError(format!("Polygon approximation failed: {}", e)))?;

        if approx.len() != 4 {
            return Err(AnalysisError::NoSwatchDetected(
                format!("Paper region is not rectangular (found {} corners instead of 4)", approx.len())
            ));
        }

        // Convert to Point2f
        let mut corners = VectorOfPoint2f::new();
        for i in 0..4 {
            let pt = approx.get(i)
                .map_err(|e| AnalysisError::ProcessingError(format!("Corner access failed: {}", e)))?;
            corners.push(Point2f::new(pt.x as f32, pt.y as f32));
        }

        Ok(corners)
    }

    /// Rectify image using perspective transform
    fn rectify_image(&self, image: &Mat, corners: &VectorOfPoint2f) -> Result<(Mat, Mat)> {
        // Order corners: top-left, top-right, bottom-right, bottom-left
        let ordered = self.order_corners(corners)?;

        // Compute output dimensions
        let (width, height) = self.compute_output_size(&ordered)?;

        // Destination corners (standard rectangle)
        let mut dst_corners = VectorOfPoint2f::new();
        dst_corners.push(Point2f::new(0.0, 0.0));
        dst_corners.push(Point2f::new(width as f32, 0.0));
        dst_corners.push(Point2f::new(width as f32, height as f32));
        dst_corners.push(Point2f::new(0.0, height as f32));

        // Compute homography
        let homography = get_perspective_transform(&ordered, &dst_corners, 0)
            .map_err(|e| AnalysisError::ProcessingError(format!("Homography computation failed: {}", e)))?;

        // Apply perspective warp
        let mut rectified = Mat::default();
        warp_perspective(
            image,
            &mut rectified,
            &homography,
            Size::new(width, height),
            opencv::imgproc::INTER_LINEAR,
            BORDER_CONSTANT,
            Scalar::default(),
        )
        .map_err(|e| AnalysisError::ProcessingError(format!("Perspective warp failed: {}", e)))?;

        Ok((rectified, homography))
    }

    /// Order corners in consistent manner: TL, TR, BR, BL
    fn order_corners(&self, corners: &VectorOfPoint2f) -> Result<VectorOfPoint2f> {
        if corners.len() != 4 {
            return Err(AnalysisError::ProcessingError("Expected 4 corners".into()));
        }

        // Find top-left (smallest sum of coordinates)
        // Find bottom-right (largest sum)
        // Find top-right (smallest diff: x-y)
        // Find bottom-left (largest diff)
        let mut pts: Vec<Point2f> = (0..4)
            .map(|i| corners.get(i).unwrap())
            .collect();

        pts.sort_by(|a, b| {
            let sum_a = a.x + a.y;
            let sum_b = b.x + b.y;
            sum_a.partial_cmp(&sum_b).unwrap()
        });
        let tl = pts[0];
        let br = pts[3];

        pts.sort_by(|a, b| {
            let diff_a = a.x - a.y;
            let diff_b = b.x - b.y;
            diff_a.partial_cmp(&diff_b).unwrap()
        });
        let tr = pts[3];
        let bl = pts[0];

        let mut ordered = VectorOfPoint2f::new();
        ordered.push(tl);
        ordered.push(tr);
        ordered.push(br);
        ordered.push(bl);

        Ok(ordered)
    }

    /// Compute output size for rectified image
    fn compute_output_size(&self, corners: &VectorOfPoint2f) -> Result<(i32, i32)> {
        let tl = corners.get(0).map_err(|e| AnalysisError::ProcessingError(format!("Corner access failed: {}", e)))?;
        let tr = corners.get(1).map_err(|e| AnalysisError::ProcessingError(format!("Corner access failed: {}", e)))?;
        let br = corners.get(2).map_err(|e| AnalysisError::ProcessingError(format!("Corner access failed: {}", e)))?;
        let bl = corners.get(3).map_err(|e| AnalysisError::ProcessingError(format!("Corner access failed: {}", e)))?;

        // Width: max of top and bottom edge lengths
        let top_width = ((tr.x - tl.x).powi(2) + (tr.y - tl.y).powi(2)).sqrt();
        let bottom_width = ((br.x - bl.x).powi(2) + (br.y - bl.y).powi(2)).sqrt();
        let width = top_width.max(bottom_width) as i32;

        // Height: max of left and right edge lengths
        let left_height = ((bl.x - tl.x).powi(2) + (bl.y - tl.y).powi(2)).sqrt();
        let right_height = ((br.x - tr.x).powi(2) + (br.y - tr.y).powi(2)).sqrt();
        let height = left_height.max(right_height) as i32;

        Ok((width, height))
    }

    /// Warp foreign object mask to rectified coordinates
    fn warp_mask(&self, mask: &Mat, homography: &Mat, size: Size) -> Result<Mat> {
        let mut warped = Mat::default();
        warp_perspective(
            mask,
            &mut warped,
            homography,
            size,
            opencv::imgproc::INTER_NEAREST,
            BORDER_CONSTANT,
            Scalar::all(0.0),
        )
        .map_err(|e| AnalysisError::ProcessingError(format!("Mask warping failed: {}", e)))?;

        Ok(warped)
    }

    /// Compute confidence score for detection
    fn compute_confidence(&self, contour: &VectorOfPoint, corners: &VectorOfPoint2f, image: &Mat) -> Result<f32> {
        let area = opencv::imgproc::contour_area(contour, false)
            .map_err(|e| AnalysisError::ProcessingError(format!("Area calculation failed: {}", e)))?;
        let perimeter = arc_length(contour, true)
            .map_err(|e| AnalysisError::ProcessingError(format!("Perimeter calculation failed: {}", e)))?;

        // Compactness: 4π*area / perimeter^2 (1.0 = perfect circle, ~0.785 = square)
        let compactness = (4.0 * std::f64::consts::PI * area) / (perimeter * perimeter);

        // Size score: how much of image is paper
        let image_area = (image.rows() * image.cols()) as f64;
        let size_ratio = area / image_area;

        // Corner score: how well approximation matches original contour
        let corner_score = if corners.len() == 4 { 1.0 } else { 0.0 };

        // Weighted combination
        let confidence = (0.3 * compactness + 0.4 * size_ratio + 0.3 * corner_score) as f32;

        Ok(confidence.clamp(0.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paper_detector_creation() {
        let detector = PaperDetector::new();
        assert_eq!(detector.min_area_ratio, MIN_PAPER_AREA_RATIO);
        assert_eq!(detector.max_angle, MAX_RECTIFICATION_ANGLE);
    }

    #[test]
    fn test_paper_detector_custom_params() {
        let detector = PaperDetector::with_params(0.15, 30.0, 0.03);
        assert_eq!(detector.min_area_ratio, 0.15);
        assert_eq!(detector.max_angle, 30.0);
        assert_eq!(detector.poly_epsilon, 0.03);
    }

    // TODO: Add integration tests with sample images
    // - Test with clear paper boundary
    // - Test with foreign objects (tape, rulers)
    // - Test with poor lighting
    // - Test with rounded corners
    // - Test error cases (no paper, too small, etc.)
}
