//! White balance estimation and correction
//!
//! Implements algorithms for estimating scene illuminant and applying
//! white balance correction to normalize colors to D65 standard.

use crate::error::{AnalysisError, Result};
use opencv::{core::Mat, prelude::*, core::Rect};
use palette::{Lab, Srgb, IntoColor, white_point::D65};

/// Paper band white balance estimation result
#[derive(Debug, Clone)]
pub struct PaperBandEstimation {
    /// Estimated paper color in Lab space
    pub paper_color: Lab<D65>,
    /// Number of valid pixels used for estimation
    pub pixel_count: usize,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
}

/// White balance estimator using multiple algorithms
pub struct WhiteBalanceEstimator {
    /// Whether to use gray world assumption
    #[allow(dead_code)]
    use_gray_world: bool,
    /// Whether to use learning-based estimation
    #[allow(dead_code)]
    use_learning_based: bool,
    /// Paper band width as fraction of distance to image border (default: 0.2)
    pub paper_band_width: f64,
}

impl WhiteBalanceEstimator {
    /// Create a new white balance estimator with default settings
    pub fn new() -> Self {
        Self {
            use_gray_world: true,
            use_learning_based: false, // Requires OpenCV contrib modules
            paper_band_width: 0.2, // 20% of distance to border
        }
    }

    /// Estimate paper color from sampling band around detected rectangle
    ///
    /// Implements the adaptive white point estimation from the swatch-first
    /// detection algorithm. Samples pixels in a band located midway between
    /// the detected rectangle and image borders.
    ///
    /// # Arguments
    ///
    /// * `image` - Input image in BGR format
    /// * `rect_bounds` - Bounding box of detected rectangle (paper or swatch)
    ///
    /// # Returns
    ///
    /// `PaperBandEstimation` with estimated paper color and confidence
    pub fn estimate_from_paper_band(
        &self,
        image: &Mat,
        rect_bounds: Rect,
    ) -> Result<PaperBandEstimation> {
        let img_width = image.cols() as f64;
        let img_height = image.rows() as f64;

        // Calculate midpoints between rectangle and image borders
        let band_left = (rect_bounds.x as f64 / 2.0) as i32;
        let band_right = ((rect_bounds.x + rect_bounds.width) as f64 + img_width) / 2.0;
        let band_top = (rect_bounds.y as f64 / 2.0) as i32;
        let band_bottom = ((rect_bounds.y + rect_bounds.height) as f64 + img_height) / 2.0;

        // Calculate band widths (percentage of distance)
        let left_distance = rect_bounds.x as f64;
        let right_distance = img_width - (rect_bounds.x + rect_bounds.width) as f64;
        let top_distance = rect_bounds.y as f64;
        let bottom_distance = img_height - (rect_bounds.y + rect_bounds.height) as f64;

        let left_band_width = (left_distance * self.paper_band_width) as i32;
        let right_band_width = (right_distance * self.paper_band_width) as i32;
        let top_band_height = (top_distance * self.paper_band_width) as i32;
        let bottom_band_height = (bottom_distance * self.paper_band_width) as i32;

        // Convert image to Lab for filtering
        let mut lab_image = Mat::default();
        opencv::imgproc::cvt_color(
            image,
            &mut lab_image,
            opencv::imgproc::COLOR_BGR2Lab,
            0,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )
        .map_err(|e| AnalysisError::ProcessingError(format!("Lab conversion failed: {}", e)))?;

        // Extract pixels from 4 sampling bands
        let mut paper_pixels: Vec<Lab<D65>> = Vec::new();

        // Left band
        if left_band_width > 0 {
            let x_start = (band_left - left_band_width / 2).max(0);
            let x_end = (band_left + left_band_width / 2).min(image.cols());
            self.extract_band_pixels(&lab_image, x_start, x_end, 0, image.rows(), &mut paper_pixels)?;
        }

        // Right band
        if right_band_width > 0 {
            let x_start = (band_right as i32 - right_band_width / 2).max(0);
            let x_end = (band_right as i32 + right_band_width / 2).min(image.cols());
            self.extract_band_pixels(&lab_image, x_start, x_end, 0, image.rows(), &mut paper_pixels)?;
        }

        // Top band
        if top_band_height > 0 {
            let y_start = (band_top - top_band_height / 2).max(0);
            let y_end = (band_top + top_band_height / 2).min(image.rows());
            self.extract_band_pixels(&lab_image, 0, image.cols(), y_start, y_end, &mut paper_pixels)?;
        }

        // Bottom band
        if bottom_band_height > 0 {
            let y_start = (band_bottom as i32 - bottom_band_height / 2).max(0);
            let y_end = (band_bottom as i32 + bottom_band_height / 2).min(image.rows());
            self.extract_band_pixels(&lab_image, 0, image.cols(), y_start, y_end, &mut paper_pixels)?;
        }

        if paper_pixels.is_empty() {
            return Err(AnalysisError::ProcessingError(
                "No valid pixels found in paper sampling band".into()
            ));
        }

        // Compute median Lab (robust to outliers)
        let mut l_values: Vec<f32> = paper_pixels.iter().map(|p| p.l).collect();
        let mut a_values: Vec<f32> = paper_pixels.iter().map(|p| p.a).collect();
        let mut b_values: Vec<f32> = paper_pixels.iter().map(|p| p.b).collect();

        l_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        a_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        b_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let l_median = l_values[l_values.len() / 2];
        let a_median = a_values[a_values.len() / 2];
        let b_median = b_values[b_values.len() / 2];

        let paper_color = Lab::new(l_median, a_median, b_median);

        // Compute confidence based on pixel count and color variance
        let pixel_count = paper_pixels.len();
        let chroma = (a_median * a_median + b_median * b_median).sqrt();

        // Higher confidence: more pixels, lower chroma (more neutral paper)
        let confidence = if pixel_count >= 1000 && chroma < 10.0 {
            0.9
        } else if pixel_count >= 500 && chroma < 20.0 {
            0.7
        } else if pixel_count >= 100 {
            0.5
        } else {
            0.3
        };

        Ok(PaperBandEstimation {
            paper_color,
            pixel_count,
            confidence,
        })
    }

    /// Extract and filter pixels from a rectangular region
    fn extract_band_pixels(
        &self,
        lab_image: &Mat,
        x_start: i32,
        x_end: i32,
        y_start: i32,
        y_end: i32,
        pixels: &mut Vec<Lab<D65>>,
    ) -> Result<()> {
        for row in y_start..y_end {
            for col in x_start..x_end {
                let pixel = lab_image.at_2d::<opencv::core::Vec3b>(row, col)
                    .map_err(|e| AnalysisError::ProcessingError(format!("Pixel access failed: {}", e)))?;

                // Convert OpenCV Lab [0-255] to palette Lab
                let l = (pixel[0] as f32) * 100.0 / 255.0;
                let a = (pixel[1] as f32) - 128.0;
                let b = (pixel[2] as f32) - 128.0;

                // Filter overexposed pixels (L* > 98, very close to #FFFFFF)
                if l > 98.0 {
                    continue;
                }

                // Filter shadowed pixels (L* < 40, deep shadows)
                if l < 40.0 {
                    continue;
                }

                pixels.push(Lab::new(l, a, b));
            }
        }
        Ok(())
    }

    /// Estimate paper color from paper region
    ///
    /// Uses the detected paper/background region to estimate the paper color
    /// in Lab space. The paper is assumed to be white or light neutral under
    /// the scene illuminant.
    ///
    /// # Arguments
    ///
    /// * `image` - Input image in BGR format
    /// * `paper_mask` - Binary mask indicating paper regions (inverted from paper detection)
    ///
    /// # Returns
    ///
    /// Estimated paper color in Lab space
    pub fn estimate_from_paper(
        &self,
        image: &Mat,
        paper_mask: &Mat,
    ) -> Result<Lab<D65>> {
        // Convert image to Lab
        let mut lab_image = Mat::default();
        opencv::imgproc::cvt_color(
            image,
            &mut lab_image,
            opencv::imgproc::COLOR_BGR2Lab,
            0,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )
        .map_err(|e| AnalysisError::ProcessingError(format!("Lab conversion failed: {}", e)))?;

        // Extract paper region pixels
        let mut paper_pixels: Vec<Lab<D65>> = Vec::new();
        for row in 0..image.rows() {
            for col in 0..image.cols() {
                let mask_val = *paper_mask.at_2d::<u8>(row, col)
                    .map_err(|e| AnalysisError::ProcessingError(format!("Mask access failed: {}", e)))?;

                // In paper mask, 0 = paper, 255 = non-paper (inverted)
                if mask_val == 0 {
                    let pixel = lab_image.at_2d::<opencv::core::Vec3b>(row, col)
                        .map_err(|e| AnalysisError::ProcessingError(format!("Pixel access failed: {}", e)))?;

                    // Convert OpenCV Lab to palette Lab
                    let l = (pixel[0] as f32) * 100.0 / 255.0;
                    let a = (pixel[1] as f32) - 128.0;
                    let b = (pixel[2] as f32) - 128.0;

                    paper_pixels.push(Lab::new(l, a, b));
                }
            }
        }

        if paper_pixels.is_empty() {
            return Err(AnalysisError::ProcessingError("No paper pixels found in mask".into()));
        }

        // Compute median L* and mean a*b* (robust to outliers)
        let mut l_values: Vec<f32> = paper_pixels.iter().map(|p| p.l).collect();
        l_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let l_median = l_values[l_values.len() / 2];

        let a_mean: f32 = paper_pixels.iter().map(|p| p.a).sum::<f32>() / paper_pixels.len() as f32;
        let b_mean: f32 = paper_pixels.iter().map(|p| p.b).sum::<f32>() / paper_pixels.len() as f32;

        Ok(Lab::new(l_median, a_mean, b_mean))
    }

    /// Apply gray world white balance estimation
    ///
    /// Assumes the average color of the scene should be neutral gray.
    /// Note: Currently not used in favor of paper-based estimation.
    pub fn estimate_gray_world(&self, image: &Mat) -> Result<Lab<D65>> {
        // Compute mean BGR values across entire image
        let mean = opencv::core::mean(image, &Mat::default())
            .map_err(|e| AnalysisError::ProcessingError(format!("Mean calculation failed: {}", e)))?;

        // Convert mean BGR to Lab
        let b = mean[0] as f32 / 255.0;
        let g = mean[1] as f32 / 255.0;
        let r = mean[2] as f32 / 255.0;

        let srgb = Srgb::new(r, g, b);
        let lab: Lab = srgb.into_color();

        Ok(lab)
    }

    /// Apply white balance correction to an image
    ///
    /// Corrects image colors from estimated illuminant to D65 standard.
    /// Uses a simple chromatic adaptation approach.
    ///
    /// # Arguments
    ///
    /// * `image` - Input BGR image to correct
    /// * `paper_color` - Estimated paper color under current illuminant
    ///
    /// # Returns
    ///
    /// White balance corrected BGR image
    pub fn apply_correction(&self, image: &Mat, paper_color: Lab<D65>) -> Result<Mat> {
        // Target paper color under D65 (assumed to be neutral white)
        // L* around 95 (very light), a* and b* near 0 (neutral)
        let target_paper: Lab<D65> = Lab::new(95.0, 0.0, 0.0);

        // Compute correction offsets in Lab space
        let delta_l = target_paper.l - paper_color.l;
        let delta_a = target_paper.a - paper_color.a;
        let delta_b = target_paper.b - paper_color.b;

        // Convert image to Lab
        let mut lab_image = Mat::default();
        opencv::imgproc::cvt_color(
            image,
            &mut lab_image,
            opencv::imgproc::COLOR_BGR2Lab,
            0,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )
        .map_err(|e| AnalysisError::ProcessingError(format!("Lab conversion failed: {}", e)))?;

        // Create corrected Lab image
        let mut corrected_lab = lab_image.clone();

        // Apply corrections to each pixel
        for row in 0..image.rows() {
            for col in 0..image.cols() {
                let pixel = lab_image.at_2d::<opencv::core::Vec3b>(row, col)
                    .map_err(|e| AnalysisError::ProcessingError(format!("Pixel access failed: {}", e)))?;

                // Convert from OpenCV Lab [0-255] to palette Lab
                let l = (pixel[0] as f32) * 100.0 / 255.0;
                let a = (pixel[1] as f32) - 128.0;
                let b = (pixel[2] as f32) - 128.0;

                // Apply correction
                let corrected_l = (l + delta_l).clamp(0.0, 100.0);
                let corrected_a = (a + delta_a).clamp(-128.0, 127.0);
                let corrected_b = (b + delta_b).clamp(-128.0, 127.0);

                // Convert back to OpenCV Lab [0-255]
                let opencv_l = (corrected_l * 255.0 / 100.0).round() as u8;
                let opencv_a = (corrected_a + 128.0).round() as u8;
                let opencv_b = (corrected_b + 128.0).round() as u8;

                *corrected_lab.at_2d_mut::<opencv::core::Vec3b>(row, col)
                    .map_err(|e| AnalysisError::ProcessingError(format!("Pixel write failed: {}", e)))? =
                    opencv::core::Vec3b::from([opencv_l, opencv_a, opencv_b]);
            }
        }

        // Convert back to BGR
        let mut corrected_bgr = Mat::default();
        opencv::imgproc::cvt_color(
            &corrected_lab,
            &mut corrected_bgr,
            opencv::imgproc::COLOR_Lab2BGR,
            0,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )
        .map_err(|e| AnalysisError::ProcessingError(format!("BGR conversion failed: {}", e)))?;

        Ok(corrected_bgr)
    }
}

impl Default for WhiteBalanceEstimator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_white_balance_estimator_creation() {
        let estimator = WhiteBalanceEstimator::new();
        assert!(estimator.use_gray_world);
        assert!(!estimator.use_learning_based);
    }

    #[test]
    fn test_default_implementation() {
        let estimator = WhiteBalanceEstimator::default();
        assert!(estimator.use_gray_world);
    }

    #[test]
    fn test_estimate_from_paper() {
        let estimator = WhiteBalanceEstimator::new();

        // Create a test image with uniform white color (245, 245, 245)
        let mut image = Mat::zeros(100, 100, opencv::core::CV_8UC3).unwrap().to_mat().unwrap();
        for row in 0..100 {
            for col in 0..100 {
                *image.at_2d_mut::<opencv::core::Vec3b>(row, col).unwrap() = opencv::core::Vec3b::from([245, 245, 245]);
            }
        }

        // Create paper mask (all zeros = all paper)
        let paper_mask = Mat::zeros(100, 100, opencv::core::CV_8UC1).unwrap().to_mat().unwrap();

        let paper_color = estimator.estimate_from_paper(&image, &paper_mask).unwrap();

        // Paper should be very light (high L*)
        assert!(paper_color.l > 90.0);

        // Should be close to neutral (a* and b* near 0)
        assert!(paper_color.a.abs() < 5.0);
        assert!(paper_color.b.abs() < 5.0);
    }

    #[test]
    fn test_estimate_from_paper_empty_mask() {
        let estimator = WhiteBalanceEstimator::new();

        // Create test image
        let image = Mat::zeros(100, 100, opencv::core::CV_8UC3).unwrap().to_mat().unwrap();

        // Create paper mask with all non-paper (all 255 = no paper)
        let mut paper_mask = Mat::zeros(100, 100, opencv::core::CV_8UC1).unwrap().to_mat().unwrap();
        for row in 0..100 {
            for col in 0..100 {
                *paper_mask.at_2d_mut::<u8>(row, col).unwrap() = 255;
            }
        }

        let result = estimator.estimate_from_paper(&image, &paper_mask);

        // Should fail with no paper pixels
        assert!(result.is_err());
    }

    #[test]
    fn test_gray_world_estimation() {
        let estimator = WhiteBalanceEstimator::new();

        // Create test image with neutral gray
        let mut image = Mat::zeros(100, 100, opencv::core::CV_8UC3).unwrap().to_mat().unwrap();
        for row in 0..100 {
            for col in 0..100 {
                *image.at_2d_mut::<opencv::core::Vec3b>(row, col).unwrap() = opencv::core::Vec3b::from([128, 128, 128]);
            }
        }

        let gray_color = estimator.estimate_gray_world(&image).unwrap();

        // Should be medium lightness
        assert!(gray_color.l > 40.0 && gray_color.l < 60.0);

        // Should be neutral
        assert!(gray_color.a.abs() < 2.0);
        assert!(gray_color.b.abs() < 2.0);
    }

    #[test]
    fn test_estimate_from_paper_partial_mask() {
        let estimator = WhiteBalanceEstimator::new();

        // Create test image
        let mut image = Mat::zeros(100, 100, opencv::core::CV_8UC3).unwrap().to_mat().unwrap();
        for row in 0..100 {
            for col in 0..100 {
                *image.at_2d_mut::<opencv::core::Vec3b>(row, col).unwrap() = opencv::core::Vec3b::from([240, 240, 240]);
            }
        }

        // Create paper mask with half paper, half non-paper
        let mut paper_mask = Mat::zeros(100, 100, opencv::core::CV_8UC1).unwrap().to_mat().unwrap();
        for row in 0..100 {
            for col in 50..100 {
                *paper_mask.at_2d_mut::<u8>(row, col).unwrap() = 255; // Non-paper
            }
        }

        let paper_color = estimator.estimate_from_paper(&image, &paper_mask).unwrap();

        // Should still estimate paper color correctly from available pixels
        assert!(paper_color.l > 85.0);
    }
}