//! Calibration constants and reference values for color analysis
//!
//! This module contains compile-time constants for color calibration,
//! based on industry standards and research findings.

/// D65 Standard Illuminant Reference
///
/// CIE Standard Illuminant D65 represents average daylight with a correlated
/// color temperature of 6504K. This is the standard reference for digital
/// images and computer displays.
pub mod d65 {
    /// D65 white point in CIE XYZ color space (array form)
    /// Source: CIE 15:2004 Colorimetry, 3rd edition
    /// Note: palette crate doesn't support const XYZ, so we use array
    pub const WHITE_POINT_XYZ: [f32; 3] = [0.95047, 1.00000, 1.08883];

    /// Correlated Color Temperature of D65 in Kelvin
    pub const CCT_KELVIN: f32 = 6504.0;

    /// D65 chromaticity coordinates
    pub const CHROMATICITY_X: f32 = 0.31271;
    pub const CHROMATICITY_Y: f32 = 0.32902;
}

/// Re-export D65 white point at top level for convenience
pub const D65_WHITE_POINT_XYZ: [f32; 3] = d65::WHITE_POINT_XYZ;

/// Color analysis thresholds and limits
pub mod thresholds {
    /// Minimum swatch area as percentage of total image for reliable analysis
    pub const MIN_SWATCH_AREA_PERCENT: f32 = 10.0;

    /// Maximum swatch area as percentage (to avoid detecting entire image)
    pub const MAX_SWATCH_AREA_PERCENT: f32 = 80.0;

    /// Color accuracy target: maximum acceptable ΔE for fountain pen differentiation
    pub const TARGET_DELTA_E: f32 = 3.0;

    /// High confidence threshold for color analysis
    pub const HIGH_CONFIDENCE_THRESHOLD: f32 = 0.8;

    /// Low confidence threshold below which results should be flagged
    pub const LOW_CONFIDENCE_THRESHOLD: f32 = 0.3;

    /// Paper detection: minimum lightness (L*) for paper regions
    pub const PAPER_MIN_LIGHTNESS: f32 = 80.0;

    /// Paper detection: maximum chroma for near-neutral regions
    pub const PAPER_MAX_CHROMA: f32 = 10.0;

    /// Swatch detection: minimum chroma difference from paper
    pub const SWATCH_MIN_CHROMA_DIFF: f32 = 15.0;
}

/// Performance targets and limits
pub mod performance {
    use std::time::Duration;

    /// Maximum analysis time for smartphone-sized images
    pub const MAX_ANALYSIS_TIME: Duration = Duration::from_millis(100);

    /// Typical smartphone image dimensions for optimization
    pub const TYPICAL_WIDTH: u32 = 4000;
    pub const TYPICAL_HEIGHT: u32 = 3000;

    /// Maximum image size to process without downscaling
    pub const MAX_PROCESSING_PIXELS: u32 = 16_000_000; // 16MP

    /// Downscale target for very large images
    pub const DOWNSCALE_TARGET_PIXELS: u32 = 8_000_000; // 8MP
}

/// Color temperature range for common lighting conditions
pub mod lighting {
    /// Minimum supported color temperature (warm incandescent)
    pub const MIN_COLOR_TEMP_K: f32 = 3000.0;

    /// Maximum supported color temperature (cool daylight)
    pub const MAX_COLOR_TEMP_K: f32 = 6500.0;

    /// Standard daylight color temperatures
    pub const DAYLIGHT_5000K: f32 = 5000.0;
    pub const DAYLIGHT_5500K: f32 = 5500.0;
    pub const DAYLIGHT_6500K: f32 = 6500.0; // D65

    /// Indoor lighting typical ranges
    pub const INCANDESCENT_TYPICAL: f32 = 3000.0;
    pub const FLUORESCENT_COOL: f32 = 4000.0;
    pub const FLUORESCENT_DAYLIGHT: f32 = 5000.0;
}

/// Image processing parameters
pub mod processing {
    /// Gaussian blur sigma for noise reduction
    pub const BLUR_SIGMA: f32 = 1.0;

    /// Morphological operation kernel size
    pub const MORPH_KERNEL_SIZE: i32 = 3;

    /// Edge detection threshold
    pub const CANNY_LOW_THRESHOLD: f64 = 50.0;
    pub const CANNY_HIGH_THRESHOLD: f64 = 150.0;

    /// Contour approximation epsilon (as fraction of perimeter)
    pub const CONTOUR_EPSILON_FACTOR: f64 = 0.02;

    /// Minimum contour area (as fraction of image area)
    pub const MIN_CONTOUR_AREA_FACTOR: f64 = 0.01;

    /// Hough transform parameters for line detection
    pub const HOUGH_RHO: f64 = 1.0;
    pub const HOUGH_THETA: f64 = std::f64::consts::PI / 180.0;
    pub const HOUGH_THRESHOLD: i32 = 100;
}

/// Statistical analysis parameters
pub mod statistics {
    /// Percentile for robust mean estimation (exclude outliers)
    pub const ROBUST_PERCENTILE_LOW: f32 = 15.0;
    pub const ROBUST_PERCENTILE_HIGH: f32 = 85.0;

    /// Minimum sample size for statistical validity
    pub const MIN_SAMPLE_SIZE: usize = 100;

    /// Outlier detection: standard deviations from mean
    pub const OUTLIER_SIGMA_THRESHOLD: f32 = 2.5;

    /// Bootstrap sampling iterations for confidence estimation
    pub const BOOTSTRAP_ITERATIONS: usize = 1000;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_d65_constants() {
        // Verify D65 white point values match CIE standards
        assert!((d65::WHITE_POINT_XYZ[0] - 0.95047).abs() < 1e-5);
        assert!((d65::WHITE_POINT_XYZ[1] - 1.00000).abs() < 1e-5);
        assert!((d65::WHITE_POINT_XYZ[2] - 1.08883).abs() < 1e-5);
    }

    #[test]
    fn test_threshold_ranges() {
        // Verify reasonable threshold ranges
        assert!(thresholds::MIN_SWATCH_AREA_PERCENT < thresholds::MAX_SWATCH_AREA_PERCENT);
        assert!(thresholds::LOW_CONFIDENCE_THRESHOLD < thresholds::HIGH_CONFIDENCE_THRESHOLD);
        assert!(lighting::MIN_COLOR_TEMP_K < lighting::MAX_COLOR_TEMP_K);
    }

    #[test]
    fn test_performance_constraints() {
        // Verify performance targets are reasonable
        assert!(performance::MAX_ANALYSIS_TIME.as_millis() > 0);
        assert!(performance::MAX_PROCESSING_PIXELS > performance::DOWNSCALE_TARGET_PIXELS);
    }
}