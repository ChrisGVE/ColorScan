//! Robust color extraction and analysis
//!
//! Extracts representative colors from ink regions with:
//! - Outlier removal via percentile filtering
//! - Transparency handling (ink-paper mixing model)
//! - Statistical aggregation with confidence scoring
//! - Validation against reasonable ink gamut bounds
//!
//! Algorithm tag: `algo-robust-color-extraction`

use opencv::{core::Mat, imgproc::cvt_color, imgproc::COLOR_BGR2Lab, prelude::*};
use palette::Lab;
use crate::{AnalysisError, Result, color::ColorConverter};

/// Minimum color difference from paper for valid ink (ΔE)
const MIN_INK_DELTA_E: f32 = 15.0;

/// Percentile range for outlier removal (15th-85th percentile)
const OUTLIER_PERCENTILE_LOW: f32 = 15.0;
const OUTLIER_PERCENTILE_HIGH: f32 = 85.0;

/// Color extraction method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractionMethod {
    /// Current method: Median L*, Mean a*/b* on 15-85 percentile (baseline)
    MedianMean,
    /// Darkest pixels: 10-25 percentile L* (concentrated/wet ink)
    Darkest,
    /// Most saturated pixels: Highest chroma (true ink color)
    MostSaturated,
    /// Mode: Most frequent color (dominant color)
    Mode,
}

/// Color analysis result with statistics
#[derive(Debug, Clone)]
pub struct ColorAnalysisResult {
    /// Representative Lab color
    pub lab: Lab,
    /// Color variance/spread
    pub variance: f32,
    /// Number of pixels analyzed
    pub pixel_count: usize,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// Extraction method used
    pub method: ExtractionMethod,
}

/// Color analyzer implementing robust color extraction
pub struct ColorAnalyzer {
    converter: ColorConverter,
    min_delta_e: f32,
    percentile_low: f32,
    percentile_high: f32,
}

impl Default for ColorAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl ColorAnalyzer {
    /// Create a new color analyzer with default parameters
    pub fn new() -> Self {
        Self {
            converter: ColorConverter::new(),
            min_delta_e: MIN_INK_DELTA_E,
            percentile_low: OUTLIER_PERCENTILE_LOW,
            percentile_high: OUTLIER_PERCENTILE_HIGH,
        }
    }

    /// Create a color analyzer with custom parameters
    pub fn with_params(min_delta_e: f32, percentile_low: f32, percentile_high: f32) -> Self {
        Self {
            converter: ColorConverter::new(),
            min_delta_e,
            percentile_low,
            percentile_high,
        }
    }

    /// Extract representative color from ink region
    ///
    /// # Arguments
    ///
    /// * `image` - BGR image of ink swatch
    /// * `mask` - Binary mask of ink region (true = ink)
    /// * `paper_color` - Estimated paper color in Lab space
    /// * `method` - Extraction method to use
    ///
    /// # Returns
    ///
    /// `ColorAnalysisResult` with representative color and confidence
    ///
    /// # Errors
    ///
    /// Returns `AnalysisError` if:
    /// - Image cannot be processed
    /// - Too few pixels in mask
    /// - Extracted color too similar to paper
    pub fn extract_color(
        &self,
        image: &Mat,
        mask: &Mat,
        paper_color: Lab,
        method: ExtractionMethod,
    ) -> Result<ColorAnalysisResult> {
        // Step 1: Extract ink pixels
        let lab_pixels = self.extract_lab_pixels(image, mask)?;

        if lab_pixels.is_empty() {
            return Err(AnalysisError::SwatchTooSmall("No pixels in swatch mask".into()));
        }

        // Step 2: Outlier removal
        let filtered_pixels = self.remove_outliers(&lab_pixels)?;

        if filtered_pixels.len() < 10 {
            return Err(AnalysisError::SwatchTooSmall(
                format!("Too few pixels after outlier removal: {}", filtered_pixels.len())
            ));
        }

        // Step 3: Transparency handling (simplified - use filtered pixels directly)
        // TODO: Implement full ink-paper mixing model when needed

        // Step 4: Statistical aggregation using specified method
        let representative_color = self.compute_representative_color(&filtered_pixels, method)?;

        // Step 5: Validation
        self.validate_ink_color(representative_color, paper_color)?;

        // Step 6: Compute variance and confidence
        let variance = self.compute_variance(&filtered_pixels, representative_color);
        let confidence = self.compute_confidence(
            &filtered_pixels,
            representative_color,
            paper_color,
            variance,
        )?;

        Ok(ColorAnalysisResult {
            lab: representative_color,
            variance,
            pixel_count: filtered_pixels.len(),
            confidence,
            method,
        })
    }

    /// Extract Lab pixels from image using mask
    fn extract_lab_pixels(&self, image: &Mat, mask: &Mat) -> Result<Vec<Lab>> {
        // Convert image to Lab
        let mut lab_image = Mat::default();
        cvt_color(image, &mut lab_image, COLOR_BGR2Lab, 0, opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT)
            .map_err(|e| AnalysisError::ProcessingError(format!("Lab conversion failed: {}", e)))?;

        let mut pixels = Vec::new();

        // Extract pixels where mask is true
        for row in 0..image.rows() {
            for col in 0..image.cols() {
                let mask_val = *mask.at_2d::<u8>(row, col)
                    .map_err(|e| AnalysisError::ProcessingError(format!("Mask access failed: {}", e)))?;

                if mask_val > 0 {
                    let pixel = lab_image.at_2d::<opencv::core::Vec3b>(row, col)
                        .map_err(|e| AnalysisError::ProcessingError(format!("Pixel access failed: {}", e)))?;

                    // OpenCV Lab: L*[0,255], a*[0,255], b*[0,255]
                    // Convert to palette Lab: L[0,100], a[-128,127], b[-128,127]
                    let l = (pixel[0] as f32) * 100.0 / 255.0;
                    let a = (pixel[1] as f32) - 128.0;
                    let b = (pixel[2] as f32) - 128.0;

                    pixels.push(Lab::new(l, a, b));
                }
            }
        }

        Ok(pixels)
    }

    /// Remove outliers using percentile filtering
    fn remove_outliers(&self, pixels: &[Lab]) -> Result<Vec<Lab>> {
        if pixels.is_empty() {
            return Ok(Vec::new());
        }

        // For each channel, compute percentile thresholds
        let mut l_values: Vec<f32> = pixels.iter().map(|p| p.l).collect();
        let mut a_values: Vec<f32> = pixels.iter().map(|p| p.a).collect();
        let mut b_values: Vec<f32> = pixels.iter().map(|p| p.b).collect();

        l_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        a_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        b_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let low_idx = ((pixels.len() as f32 * self.percentile_low / 100.0) as usize).min(pixels.len() - 1);
        let high_idx = ((pixels.len() as f32 * self.percentile_high / 100.0) as usize).min(pixels.len() - 1);

        let l_min = l_values[low_idx];
        let l_max = l_values[high_idx];
        let a_min = a_values[low_idx];
        let a_max = a_values[high_idx];
        let b_min = b_values[low_idx];
        let b_max = b_values[high_idx];

        // Keep pixels within percentile range for all channels
        let filtered: Vec<Lab> = pixels
            .iter()
            .copied()
            .filter(|p| {
                p.l >= l_min && p.l <= l_max &&
                p.a >= a_min && p.a <= a_max &&
                p.b >= b_min && p.b <= b_max
            })
            .collect();

        Ok(filtered)
    }

    /// Compute representative color using specified extraction method
    fn compute_representative_color(&self, pixels: &[Lab], method: ExtractionMethod) -> Result<Lab> {
        if pixels.is_empty() {
            return Err(AnalysisError::ProcessingError("No pixels to analyze".into()));
        }

        match method {
            ExtractionMethod::MedianMean => self.method_median_mean(pixels),
            ExtractionMethod::Darkest => self.method_darkest(pixels),
            ExtractionMethod::MostSaturated => self.method_most_saturated(pixels),
            ExtractionMethod::Mode => self.method_mode(pixels),
        }
    }

    /// Method 1: Median L*, Mean a*/b* (baseline - current method)
    fn method_median_mean(&self, pixels: &[Lab]) -> Result<Lab> {
        // Use median for L* (lightness) to handle specular effects
        let mut l_values: Vec<f32> = pixels.iter().map(|p| p.l).collect();
        l_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let l_median = l_values[l_values.len() / 2];

        // Use mean for a* and b* (chromaticity)
        let a_mean: f32 = pixels.iter().map(|p| p.a).sum::<f32>() / pixels.len() as f32;
        let b_mean: f32 = pixels.iter().map(|p| p.b).sum::<f32>() / pixels.len() as f32;

        Ok(Lab::new(l_median, a_mean, b_mean))
    }

    /// Method 2: Darkest pixels (10-25 percentile L*)
    /// Represents concentrated/wet ink - the truest ink color
    fn method_darkest(&self, pixels: &[Lab]) -> Result<Lab> {
        // Sort by lightness
        let mut sorted: Vec<Lab> = pixels.to_vec();
        sorted.sort_by(|a, b| a.l.partial_cmp(&b.l).unwrap());

        // Take 10-25 percentile (darkest quarter excluding extreme outliers)
        let low_idx = (sorted.len() as f32 * 0.10) as usize;
        let high_idx = (sorted.len() as f32 * 0.25) as usize;
        let darkest_pixels = &sorted[low_idx..high_idx];

        if darkest_pixels.is_empty() {
            return self.method_median_mean(pixels); // Fallback
        }

        // Median L*, Mean a*/b* of darkest pixels
        let mut l_values: Vec<f32> = darkest_pixels.iter().map(|p| p.l).collect();
        l_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let l_median = l_values[l_values.len() / 2];

        let a_mean: f32 = darkest_pixels.iter().map(|p| p.a).sum::<f32>() / darkest_pixels.len() as f32;
        let b_mean: f32 = darkest_pixels.iter().map(|p| p.b).sum::<f32>() / darkest_pixels.len() as f32;

        Ok(Lab::new(l_median, a_mean, b_mean))
    }

    /// Method 3: Most saturated pixels (highest chroma)
    /// Represents the truest ink color regardless of concentration
    fn method_most_saturated(&self, pixels: &[Lab]) -> Result<Lab> {
        // Compute chroma for each pixel: sqrt(a^2 + b^2)
        let mut pixels_with_chroma: Vec<(Lab, f32)> = pixels
            .iter()
            .map(|p| {
                let chroma = (p.a * p.a + p.b * p.b).sqrt();
                (*p, chroma)
            })
            .collect();

        // Sort by chroma descending
        pixels_with_chroma.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Take top 25% most saturated pixels
        let count = (pixels_with_chroma.len() as f32 * 0.25).max(1.0) as usize;
        let most_saturated: Vec<Lab> = pixels_with_chroma
            .iter()
            .take(count)
            .map(|(p, _)| *p)
            .collect();

        // Median L*, Mean a*/b* of most saturated pixels
        let mut l_values: Vec<f32> = most_saturated.iter().map(|p| p.l).collect();
        l_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let l_median = l_values[l_values.len() / 2];

        let a_mean: f32 = most_saturated.iter().map(|p| p.a).sum::<f32>() / most_saturated.len() as f32;
        let b_mean: f32 = most_saturated.iter().map(|p| p.b).sum::<f32>() / most_saturated.len() as f32;

        Ok(Lab::new(l_median, a_mean, b_mean))
    }

    /// Method 4: Mode (most frequent color)
    /// Uses 3D histogram binning to find dominant color
    fn method_mode(&self, pixels: &[Lab]) -> Result<Lab> {
        use std::collections::HashMap;

        // Quantize Lab values into bins (5 units per bin for L, 10 units for a/b)
        let quantize_l = |l: f32| ((l / 5.0).round() * 5.0) as i32;
        let quantize_ab = |v: f32| ((v / 10.0).round() * 10.0) as i32;

        let mut histogram: HashMap<(i32, i32, i32), Vec<Lab>> = HashMap::new();

        // Build histogram
        for pixel in pixels {
            let key = (
                quantize_l(pixel.l),
                quantize_ab(pixel.a),
                quantize_ab(pixel.b),
            );
            histogram.entry(key).or_insert_with(Vec::new).push(*pixel);
        }

        // Find bin with most pixels
        let most_frequent_bin = histogram
            .iter()
            .max_by_key(|(_, pixels)| pixels.len())
            .map(|(_, pixels)| pixels);

        match most_frequent_bin {
            Some(dominant_pixels) => {
                // Return mean of pixels in dominant bin
                let l_mean: f32 = dominant_pixels.iter().map(|p| p.l).sum::<f32>() / dominant_pixels.len() as f32;
                let a_mean: f32 = dominant_pixels.iter().map(|p| p.a).sum::<f32>() / dominant_pixels.len() as f32;
                let b_mean: f32 = dominant_pixels.iter().map(|p| p.b).sum::<f32>() / dominant_pixels.len() as f32;

                Ok(Lab::new(l_mean, a_mean, b_mean))
            }
            None => self.method_median_mean(pixels), // Fallback
        }
    }

    /// Validate that extracted color is sufficiently different from paper
    fn validate_ink_color(&self, ink_color: Lab, paper_color: Lab) -> Result<()> {
        let delta_e = self.converter.delta_e(ink_color, paper_color);

        if delta_e < self.min_delta_e {
            return Err(AnalysisError::ProcessingError(
                format!(
                    "Extracted color too similar to paper (ΔE = {:.1}, minimum = {:.1})",
                    delta_e, self.min_delta_e
                )
            ));
        }

        // Check for reasonable ink gamut bounds
        // L* should be in reasonable range (not pure black or white)
        if ink_color.l < 5.0 || ink_color.l > 95.0 {
            return Err(AnalysisError::ProcessingError(
                format!("Unrealistic lightness value: {:.1}", ink_color.l)
            ));
        }

        Ok(())
    }

    /// Compute color variance
    fn compute_variance(&self, pixels: &[Lab], representative: Lab) -> f32 {
        if pixels.is_empty() {
            return 0.0;
        }

        let sum_squared_diff: f32 = pixels
            .iter()
            .map(|p| {
                let delta_e = self.converter.delta_e(*p, representative);
                delta_e * delta_e
            })
            .sum();

        (sum_squared_diff / pixels.len() as f32).sqrt()
    }

    /// Compute confidence score
    fn compute_confidence(
        &self,
        pixels: &[Lab],
        representative: Lab,
        paper_color: Lab,
        variance: f32,
    ) -> Result<f32> {
        // Size score: more pixels = higher confidence
        let size_score = if pixels.len() >= 1000 {
            1.0
        } else if pixels.len() >= 100 {
            0.8
        } else {
            0.5
        };

        // Variance score: lower variance = higher confidence
        let variance_score = if variance < 5.0 {
            1.0
        } else if variance < 10.0 {
            0.7
        } else {
            0.4
        };

        // Separation score: greater distance from paper = higher confidence
        let delta_e = self.converter.delta_e(representative, paper_color);
        let separation_score = if delta_e > 30.0 {
            1.0
        } else if delta_e > 20.0 {
            0.8
        } else {
            0.6
        };

        // Weighted combination
        let confidence: f32 = 0.4 * size_score + 0.3 * variance_score + 0.3 * separation_score;

        Ok(confidence.clamp(0.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_analyzer_creation() {
        let analyzer = ColorAnalyzer::new();
        assert_eq!(analyzer.min_delta_e, MIN_INK_DELTA_E);
        assert_eq!(analyzer.percentile_low, OUTLIER_PERCENTILE_LOW);
        assert_eq!(analyzer.percentile_high, OUTLIER_PERCENTILE_HIGH);
    }

    #[test]
    fn test_color_analyzer_custom_params() {
        let analyzer = ColorAnalyzer::with_params(20.0, 10.0, 90.0);
        assert_eq!(analyzer.min_delta_e, 20.0);
        assert_eq!(analyzer.percentile_low, 10.0);
        assert_eq!(analyzer.percentile_high, 90.0);
    }

    #[test]
    fn test_remove_outliers_empty() {
        let analyzer = ColorAnalyzer::new();
        let pixels: Vec<Lab> = vec![];
        let filtered = analyzer.remove_outliers(&pixels).unwrap();
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_remove_outliers() {
        let analyzer = ColorAnalyzer::new();
        let mut pixels = vec![];

        // Add normal pixels
        for _ in 0..100 {
            pixels.push(Lab::new(50.0, 10.0, 10.0));
        }

        // Add outliers
        pixels.push(Lab::new(10.0, 10.0, 10.0)); // Very dark
        pixels.push(Lab::new(90.0, 10.0, 10.0)); // Very bright
        pixels.push(Lab::new(50.0, 100.0, 10.0)); // Extreme a*
        pixels.push(Lab::new(50.0, 10.0, 100.0)); // Extreme b*

        let filtered = analyzer.remove_outliers(&pixels).unwrap();

        // Should filter out extremes
        assert!(filtered.len() < pixels.len());
        assert!(filtered.len() >= 70); // Most normal pixels should remain
    }

    #[test]
    fn test_compute_representative_color() {
        let analyzer = ColorAnalyzer::new();
        let pixels = vec![
            Lab::new(48.0, 10.0, 10.0),
            Lab::new(50.0, 11.0, 11.0),
            Lab::new(52.0, 9.0, 9.0),
        ];

        let repr = analyzer.compute_representative_color(&pixels, ExtractionMethod::MedianMean).unwrap();

        // L* should be median (50.0)
        assert!((repr.l - 50.0).abs() < 0.1);

        // a* and b* should be mean (10.0)
        assert!((repr.a - 10.0).abs() < 0.5);
        assert!((repr.b - 10.0).abs() < 0.5);
    }

    #[test]
    fn test_validate_ink_color_success() {
        let analyzer = ColorAnalyzer::new();
        let ink_color = Lab::new(30.0, 20.0, -30.0);
        let paper_color = Lab::new(95.0, 0.0, 0.0);

        // Should succeed - sufficient color difference
        assert!(analyzer.validate_ink_color(ink_color, paper_color).is_ok());
    }

    #[test]
    fn test_validate_ink_color_too_similar() {
        let analyzer = ColorAnalyzer::new();
        let ink_color = Lab::new(93.0, 1.0, 1.0);
        let paper_color = Lab::new(95.0, 0.0, 0.0);

        // Should fail - too similar to paper
        assert!(analyzer.validate_ink_color(ink_color, paper_color).is_err());
    }

    #[test]
    fn test_validate_ink_color_unrealistic_lightness() {
        let analyzer = ColorAnalyzer::new();
        let paper_color = Lab::new(95.0, 0.0, 0.0);

        // Too dark
        let too_dark = Lab::new(2.0, 10.0, 10.0);
        assert!(analyzer.validate_ink_color(too_dark, paper_color).is_err());

        // Too bright
        let too_bright = Lab::new(98.0, 10.0, 10.0);
        assert!(analyzer.validate_ink_color(too_bright, paper_color).is_err());
    }

    #[test]
    fn test_compute_variance() {
        let analyzer = ColorAnalyzer::new();
        let representative = Lab::new(50.0, 10.0, 10.0);

        // Identical pixels - zero variance
        let identical = vec![representative; 10];
        let variance = analyzer.compute_variance(&identical, representative);
        assert!(variance < 0.1);

        // Spread pixels - higher variance
        let spread = vec![
            Lab::new(45.0, 10.0, 10.0),
            Lab::new(50.0, 10.0, 10.0),
            Lab::new(55.0, 10.0, 10.0),
        ];
        let variance2 = analyzer.compute_variance(&spread, representative);
        assert!(variance2 > 1.0);
    }

    // TODO: Add integration tests with sample images
    // - Test with uniform color swatch
    // - Test with gradient swatch
    // - Test with very light ink
    // - Test with shimmer/sheen effects
    // - Test error cases (too few pixels, invalid colors)
}
