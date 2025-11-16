//! White balance estimation and correction
//!
//! Implements algorithms for estimating scene illuminant and applying
//! white balance correction to normalize colors to D65 standard.

use crate::error::{AnalysisError, Result};
use crate::constants::d65;
use opencv::{core::Mat, prelude::*};
use palette::{Lab, Srgb, Xyz, IntoColor, white_point::D65};

/// White balance estimator using multiple algorithms
pub struct WhiteBalanceEstimator {
    /// Whether to use gray world assumption
    use_gray_world: bool,
    /// Whether to use learning-based estimation
    use_learning_based: bool,
}

impl WhiteBalanceEstimator {
    /// Create a new white balance estimator with default settings
    pub fn new() -> Self {
        Self {
            use_gray_world: true,
            use_learning_based: false, // Requires OpenCV contrib modules
        }
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