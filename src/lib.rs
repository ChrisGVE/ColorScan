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
    // Implementation will be added in subsequent iterations
    todo!("analyze_swatch implementation")
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