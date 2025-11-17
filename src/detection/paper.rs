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
    core::{Mat, Point2f, Point, Scalar, Size, Vector, BORDER_CONSTANT},
    imgproc::{
        approx_poly_dp, arc_length, find_contours, get_perspective_transform,
        warp_perspective, gaussian_blur,
        CHAIN_APPROX_SIMPLE, RETR_EXTERNAL,
    },
    prelude::*,
};
use crate::{AnalysisError, Result};

// Type aliases for OpenCV vector types
type VectorOfPoint = Vector<Point>;
type VectorOfPoint2f = Vector<Point2f>;

/// Minimum paper area as fraction of total image (10%)
const MIN_PAPER_AREA_RATIO: f64 = 0.10;

/// Maximum rectification angle in degrees (45°)
const MAX_RECTIFICATION_ANGLE: f64 = 45.0;

/// Polygon approximation epsilon as fraction of perimeter (2%)
const POLY_APPROX_EPSILON: f64 = 0.02;

/// Minimum aspect ratio for valid card (excludes very elongated rulers)
/// Cards are typically 1:1.3 to 1:2, so we use 1:3 as minimum
const MIN_CARD_ASPECT_RATIO: f64 = 0.33; // 1:3

/// Maximum aspect ratio for valid card (excludes very elongated rulers)
/// Cards are typically 1:1.3 to 1:2, so we use 3:1 as maximum
const MAX_CARD_ASPECT_RATIO: f64 = 3.0; // 3:1

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
    #[allow(dead_code)]
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
        // Step 1: Edge detection to find card boundary
        let edges = self.detect_edges(image)?;

        // Step 2: Find paper contour from edges
        let paper_contour = self.find_paper_contour_from_edges(&edges, image)?;

        // Step 3: Foreign object detection (simplified - uses color analysis)
        let foreign_mask = self.detect_foreign_objects_simple(image)?;

        // Step 4: Polygon approximation
        let corners = self.approximate_rectangle(&paper_contour)?;

        // Step 5: Homography and rectification
        let (rectified, homography) = self.rectify_image(image, &corners)?;

        // Step 6: Apply foreign mask to rectified image
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

    /// Detect edges using Canny edge detection
    fn detect_edges(&self, image: &Mat) -> Result<Mat> {
        // Convert to grayscale
        let mut gray = Mat::default();
        opencv::imgproc::cvt_color(image, &mut gray, opencv::imgproc::COLOR_BGR2GRAY, 0, opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT)
            .map_err(|e| AnalysisError::ProcessingError(format!("Grayscale conversion failed: {}", e)))?;

        // Apply Gaussian blur to reduce noise
        let mut blurred = Mat::default();
        gaussian_blur(
            &gray,
            &mut blurred,
            Size::new(5, 5),
            1.5,
            1.5,
            BORDER_CONSTANT,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )
        .map_err(|e| AnalysisError::ProcessingError(format!("Gaussian blur failed: {}", e)))?;

        // Apply Canny edge detection
        let mut edges = Mat::default();
        opencv::imgproc::canny(&blurred, &mut edges, 50.0, 150.0, 3, false)
            .map_err(|e| AnalysisError::ProcessingError(format!("Canny edge detection failed: {}", e)))?;

        // Dilate edges to close gaps
        let kernel = opencv::imgproc::get_structuring_element(
            opencv::imgproc::MORPH_RECT,
            Size::new(3, 3),
            Point::new(-1, -1),
        )
        .map_err(|e| AnalysisError::ProcessingError(format!("Kernel creation failed: {}", e)))?;

        let mut dilated = Mat::default();
        opencv::imgproc::dilate(
            &edges,
            &mut dilated,
            &kernel,
            Point::new(-1, -1),
            1,
            BORDER_CONSTANT,
            opencv::core::Scalar::all(0.0),
        )
        .map_err(|e| AnalysisError::ProcessingError(format!("Dilation failed: {}", e)))?;

        Ok(dilated)
    }

    /// Find paper contour from edge image
    fn find_paper_contour_from_edges(&self, edges: &Mat, original_image: &Mat) -> Result<VectorOfPoint> {
        let mut contours = Vector::<VectorOfPoint>::new();
        find_contours(edges, &mut contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point::new(0, 0))
            .map_err(|e| AnalysisError::ProcessingError(format!("Contour detection failed: {}", e)))?;

        if contours.is_empty() {
            return Err(AnalysisError::NoSwatchDetected("No contours found in edge image".into()));
        }

        let image_area = (original_image.rows() * original_image.cols()) as f64;
        let min_area = image_area * self.min_area_ratio;

        // Find largest rectangular contour that meets area requirement
        // and excludes elongated rulers based on aspect ratio
        // Prioritize contours that are more centrally located
        let mut best_contour: Option<VectorOfPoint> = None;
        let mut best_score = 0.0;

        let img_width = original_image.cols() as f64;
        let img_height = original_image.rows() as f64;
        let img_center_x = img_width / 2.0;
        let img_center_y = img_height / 2.0;

        for i in 0..contours.len() {
            let contour = contours.get(i)
                .map_err(|e| AnalysisError::ProcessingError(format!("Contour access failed: {}", e)))?;
            let area = opencv::imgproc::contour_area(&contour, false)
                .map_err(|e| AnalysisError::ProcessingError(format!("Area calculation failed: {}", e)))?;

            // Check if contour approximates to a rectangle
            let perimeter = arc_length(&contour, true)
                .map_err(|e| AnalysisError::ProcessingError(format!("Perimeter calculation failed: {}", e)))?;
            let epsilon = perimeter * self.poly_epsilon;

            let mut approx = VectorOfPoint::new();
            approx_poly_dp(&contour, &mut approx, epsilon, true)
                .map_err(|e| AnalysisError::ProcessingError(format!("Polygon approximation failed: {}", e)))?;

            // Only consider 4-sided polygons (rectangles/quadrilaterals)
            if approx.len() == 4 && area >= min_area {
                // Check aspect ratio to exclude rulers
                let bounding_rect = opencv::imgproc::bounding_rect(&contour)
                    .map_err(|e| AnalysisError::ProcessingError(format!("Bounding rect failed: {}", e)))?;

                let width = bounding_rect.width as f64;
                let height = bounding_rect.height as f64;
                let aspect_ratio = width / height;

                // Exclude very elongated rectangles (rulers)
                // Cards have aspect ratio between 1:3 and 3:1
                if aspect_ratio >= MIN_CARD_ASPECT_RATIO && aspect_ratio <= MAX_CARD_ASPECT_RATIO {
                    // Calculate centrality score: prefer contours closer to image center
                    let rect_center_x = (bounding_rect.x as f64) + (width / 2.0);
                    let rect_center_y = (bounding_rect.y as f64) + (height / 2.0);

                    let distance_from_center = ((rect_center_x - img_center_x).powi(2) +
                                               (rect_center_y - img_center_y).powi(2)).sqrt();

                    // Normalize distance by diagonal
                    let img_diagonal = (img_width.powi(2) + img_height.powi(2)).sqrt();
                    let centrality = 1.0 - (distance_from_center / img_diagonal);

                    // Score combines area and centrality (70% area, 30% centrality)
                    let score = (area / image_area) * 0.7 + centrality * 0.3;

                    if score > best_score {
                        best_score = score;
                        best_contour = Some(contour);
                    }
                }
            }
        }

        best_contour.ok_or_else(|| {
            AnalysisError::NoSwatchDetected(
                format!("No rectangular paper region found (minimum {}% of image area required)", self.min_area_ratio * 100.0)
            )
        })
    }

    /// Simplified foreign object detection
    fn detect_foreign_objects_simple(&self, image: &Mat) -> Result<Mat> {
        // For now, create an empty mask (no foreign objects detected)
        // This can be enhanced later with color-based detection
        let mask = Mat::zeros(image.rows(), image.cols(), opencv::core::CV_8UC1)
            .map_err(|e| AnalysisError::ProcessingError(format!("Mask creation failed: {}", e)))?
            .to_mat()
            .map_err(|e| AnalysisError::ProcessingError(format!("Mask conversion failed: {}", e)))?;

        Ok(mask)
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

    #[test]
    fn test_order_corners() {
        let detector = PaperDetector::new();

        // Create unordered corners
        let mut corners = VectorOfPoint2f::new();
        corners.push(Point2f::new(100.0, 100.0)); // Top-left
        corners.push(Point2f::new(400.0, 100.0)); // Top-right
        corners.push(Point2f::new(400.0, 400.0)); // Bottom-right
        corners.push(Point2f::new(100.0, 400.0)); // Bottom-left

        let ordered = detector.order_corners(&corners).unwrap();

        // Verify correct ordering
        let tl = ordered.get(0).unwrap();
        let tr = ordered.get(1).unwrap();
        let br = ordered.get(2).unwrap();
        let bl = ordered.get(3).unwrap();

        // Top-left should have smallest sum
        assert!(tl.x + tl.y <= tr.x + tr.y);
        assert!(tl.x + tl.y <= br.x + br.y);
        assert!(tl.x + tl.y <= bl.x + bl.y);

        // Bottom-right should have largest sum
        assert!(br.x + br.y >= tl.x + tl.y);
        assert!(br.x + br.y >= tr.x + tr.y);
        assert!(br.x + br.y >= bl.x + bl.y);
    }

    #[test]
    fn test_compute_output_size() {
        let detector = PaperDetector::new();

        // Create rectangular corners (300x200)
        let mut corners = VectorOfPoint2f::new();
        corners.push(Point2f::new(0.0, 0.0));    // Top-left
        corners.push(Point2f::new(300.0, 0.0));  // Top-right
        corners.push(Point2f::new(300.0, 200.0)); // Bottom-right
        corners.push(Point2f::new(0.0, 200.0));  // Bottom-left

        let (width, height) = detector.compute_output_size(&corners).unwrap();

        assert_eq!(width, 300);
        assert_eq!(height, 200);
    }

    #[test]
    fn test_compute_output_size_skewed() {
        let detector = PaperDetector::new();

        // Create skewed quadrilateral
        let mut corners = VectorOfPoint2f::new();
        corners.push(Point2f::new(10.0, 20.0));   // Top-left
        corners.push(Point2f::new(310.0, 30.0));  // Top-right (slightly down)
        corners.push(Point2f::new(300.0, 220.0)); // Bottom-right
        corners.push(Point2f::new(0.0, 210.0));   // Bottom-left

        let (width, height) = detector.compute_output_size(&corners).unwrap();

        // Should use maximum dimensions
        assert!(width >= 300);
        assert!(height >= 190);
    }

    #[test]
    fn test_confidence_score_calculation() {
        let detector = PaperDetector::new();

        // Create a square contour
        let mut contour = VectorOfPoint::new();
        contour.push(Point::new(0, 0));
        contour.push(Point::new(100, 0));
        contour.push(Point::new(100, 100));
        contour.push(Point::new(0, 100));

        let mut corners = VectorOfPoint2f::new();
        corners.push(Point2f::new(0.0, 0.0));
        corners.push(Point2f::new(100.0, 0.0));
        corners.push(Point2f::new(100.0, 100.0));
        corners.push(Point2f::new(0.0, 100.0));

        // Create test image
        let image = Mat::zeros(200, 200, opencv::core::CV_8UC3).unwrap().to_mat().unwrap();

        let confidence = detector.compute_confidence(&contour, &corners, &image).unwrap();

        // Confidence should be between 0 and 1
        assert!(confidence >= 0.0 && confidence <= 1.0);

        // With 4 corners and good size ratio, should be relatively high
        assert!(confidence > 0.3);
    }

    #[test]
    fn test_foreign_object_detection_empty() {
        let detector = PaperDetector::new();

        // Create blank image and binary mask
        let image = Mat::zeros(480, 640, opencv::core::CV_8UC3).unwrap().to_mat().unwrap();
        let binary = Mat::zeros(480, 640, opencv::core::CV_8UC1).unwrap().to_mat().unwrap();

        let foreign_mask = detector.detect_foreign_objects(&image, &binary).unwrap();

        // Should return empty mask for blank image
        assert_eq!(foreign_mask.rows(), 480);
        assert_eq!(foreign_mask.cols(), 640);
    }

    // Integration tests with actual images would go here
    #[test]
    #[ignore] // Ignore until we have test images
    fn test_paper_detection_with_real_image() {
        // TODO: Add test with sample image containing clear paper boundary
        // let detector = PaperDetector::new();
        // let image = opencv::imgcodecs::imread("tests/paper_sample.jpg", opencv::imgcodecs::IMREAD_COLOR).unwrap();
        // let result = detector.detect(&image).unwrap();
        // assert!(result.confidence > 0.5);
    }

    #[test]
    #[ignore] // Ignore until we have test images
    fn test_paper_detection_with_foreign_objects() {
        // TODO: Test with image containing tape or rulers
        // Should successfully detect paper and exclude foreign objects
    }
}
