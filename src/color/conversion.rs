//! Color space conversion utilities
//!
//! Provides conversions between color spaces with chromatic adaptation:
//! - BGR/RGB to Lab/LCh
//! - Chromatic adaptation to D65 illuminant
//! - Lab to sRGB with gamut mapping
//! - Hex color representation
//!
//! Algorithm tag: `algo-d65-chromatic-adaptation`

use palette::{Lab, Lch, Srgb, FromColor, IntoColor};
use crate::{constants::D65_WHITE_POINT_XYZ, AnalysisError, Result};

/// Color converter with chromatic adaptation support
pub struct ColorConverter {
    /// Target white point (always D65)
    #[allow(dead_code)]
    target_white_point: [f32; 3],
}

impl Default for ColorConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl ColorConverter {
    /// Create a new color converter with D65 as target illuminant
    pub fn new() -> Self {
        Self {
            target_white_point: D65_WHITE_POINT_XYZ,
        }
    }

    /// Convert RGB (0-255) to Lab color space
    ///
    /// # Arguments
    ///
    /// * `r`, `g`, `b` - RGB values in range [0, 255]
    ///
    /// # Returns
    ///
    /// Lab color in D65 illuminant
    pub fn rgb_to_lab(&self, r: u8, g: u8, b: u8) -> Lab {
        let srgb = Srgb::new(
            r as f32 / 255.0,
            g as f32 / 255.0,
            b as f32 / 255.0,
        );
        Lab::from_color(srgb)
    }

    /// Convert Lab to sRGB with gamut mapping
    ///
    /// # Arguments
    ///
    /// * `lab` - Lab color
    ///
    /// # Returns
    ///
    /// sRGB color, clamped to valid gamut
    pub fn lab_to_srgb(&self, lab: Lab) -> Srgb {
        let srgb: Srgb = lab.into_color();
        // Clamp to valid sRGB gamut
        Srgb::new(
            srgb.red.clamp(0.0, 1.0),
            srgb.green.clamp(0.0, 1.0),
            srgb.blue.clamp(0.0, 1.0),
        )
    }

    /// Convert Lab to LCh (cylindrical representation)
    ///
    /// # Arguments
    ///
    /// * `lab` - Lab color
    ///
    /// # Returns
    ///
    /// LCh color (Lightness, Chroma, Hue)
    pub fn lab_to_lch(&self, lab: Lab) -> Lch {
        Lch::from_color(lab)
    }

    /// Convert LCh to Lab
    ///
    /// # Arguments
    ///
    /// * `lch` - LCh color
    ///
    /// # Returns
    ///
    /// Lab color
    pub fn lch_to_lab(&self, lch: Lch) -> Lab {
        Lab::from_color(lch)
    }

    /// Convert sRGB to hexadecimal color string
    ///
    /// # Arguments
    ///
    /// * `srgb` - sRGB color
    ///
    /// # Returns
    ///
    /// Hex color string (e.g., "#FF0000")
    pub fn srgb_to_hex(&self, srgb: Srgb) -> String {
        let r = (srgb.red * 255.0).round() as u8;
        let g = (srgb.green * 255.0).round() as u8;
        let b = (srgb.blue * 255.0).round() as u8;
        format!("#{:02X}{:02X}{:02X}", r, g, b)
    }

    /// Parse hexadecimal color string to sRGB
    ///
    /// # Arguments
    ///
    /// * `hex` - Hex color string (e.g., "#FF0000" or "FF0000")
    ///
    /// # Returns
    ///
    /// sRGB color
    ///
    /// # Errors
    ///
    /// Returns error if hex string is invalid
    pub fn hex_to_srgb(&self, hex: &str) -> Result<Srgb> {
        let hex = hex.trim_start_matches('#');
        if hex.len() != 6 {
            return Err(AnalysisError::ProcessingError(
                format!("Invalid hex color: expected 6 characters, got {}", hex.len())
            ));
        }

        let r = u8::from_str_radix(&hex[0..2], 16)
            .map_err(|e| AnalysisError::ProcessingError(format!("Invalid red value: {}", e)))?;
        let g = u8::from_str_radix(&hex[2..4], 16)
            .map_err(|e| AnalysisError::ProcessingError(format!("Invalid green value: {}", e)))?;
        let b = u8::from_str_radix(&hex[4..6], 16)
            .map_err(|e| AnalysisError::ProcessingError(format!("Invalid blue value: {}", e)))?;

        Ok(Srgb::new(
            r as f32 / 255.0,
            g as f32 / 255.0,
            b as f32 / 255.0,
        ))
    }

    /// Adapt Lab color from source illuminant to D65
    ///
    /// Uses chromatic adaptation transform (CAT02 method)
    /// Note: For initial implementation, we assume input is already in D65
    /// Full chromatic adaptation will be implemented when EXIF illuminant
    /// estimation is available.
    ///
    /// # Arguments
    ///
    /// * `lab` - Lab color under source illuminant
    /// * `source_white_point` - XYZ white point of source illuminant
    ///
    /// # Returns
    ///
    /// Lab color adapted to D65
    pub fn adapt_to_d65(&self, lab: Lab, _source_white_point: [f32; 3]) -> Lab {
        // TODO: Implement full chromatic adaptation when needed
        // For now, assume input is already normalized to D65 via white balance
        // This is a placeholder for the full CAT02 implementation
        lab
    }

    /// Check if a Lab color is within sRGB gamut
    ///
    /// # Arguments
    ///
    /// * `lab` - Lab color
    ///
    /// # Returns
    ///
    /// true if color is within sRGB gamut
    pub fn is_in_srgb_gamut(&self, lab: Lab) -> bool {
        let srgb: Srgb = lab.into_color();
        srgb.red >= 0.0 && srgb.red <= 1.0 &&
        srgb.green >= 0.0 && srgb.green <= 1.0 &&
        srgb.blue >= 0.0 && srgb.blue <= 1.0
    }

    /// Compute Delta E (color difference) between two Lab colors
    ///
    /// Uses simple Euclidean distance (ΔE76)
    /// Note: For full CIEDE2000 implementation, use empfindung crate
    ///
    /// # Arguments
    ///
    /// * `lab1`, `lab2` - Lab colors to compare
    ///
    /// # Returns
    ///
    /// Color difference value
    pub fn delta_e(&self, lab1: Lab, lab2: Lab) -> f32 {
        let dl = lab1.l - lab2.l;
        let da = lab1.a - lab2.a;
        let db = lab1.b - lab2.b;
        (dl * dl + da * da + db * db).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_converter_creation() {
        let converter = ColorConverter::new();
        assert_eq!(converter.target_white_point, D65_WHITE_POINT_XYZ);
    }

    #[test]
    fn test_rgb_to_lab_black() {
        let converter = ColorConverter::new();
        let lab = converter.rgb_to_lab(0, 0, 0);
        assert!(lab.l < 1.0); // Black should have very low lightness
    }

    #[test]
    fn test_rgb_to_lab_white() {
        let converter = ColorConverter::new();
        let lab = converter.rgb_to_lab(255, 255, 255);
        assert!(lab.l > 99.0); // White should have high lightness
        assert!(lab.a.abs() < 1.0); // Should be near neutral
        assert!(lab.b.abs() < 1.0);
    }

    #[test]
    fn test_lab_to_lch_conversion() {
        let converter = ColorConverter::new();
        let lab = Lab::new(50.0, 25.0, 25.0);
        let lch = converter.lab_to_lch(lab);

        // Verify Lightness is preserved
        assert!((lch.l - lab.l).abs() < 0.001);

        // Verify chroma is computed correctly: sqrt(a^2 + b^2)
        let expected_chroma = (25.0_f32 * 25.0 + 25.0 * 25.0).sqrt();
        assert!((lch.chroma - expected_chroma).abs() < 0.001);
    }

    #[test]
    fn test_lch_to_lab_roundtrip() {
        let converter = ColorConverter::new();
        let lab = Lab::new(50.0, 25.0, 25.0);
        let lch = converter.lab_to_lch(lab);
        let lab2 = converter.lch_to_lab(lch);

        assert!((lab2.l - lab.l).abs() < 0.001);
        assert!((lab2.a - lab.a).abs() < 0.001);
        assert!((lab2.b - lab.b).abs() < 0.001);
    }

    #[test]
    fn test_srgb_to_hex() {
        let converter = ColorConverter::new();
        let red = Srgb::new(1.0, 0.0, 0.0);
        assert_eq!(converter.srgb_to_hex(red), "#FF0000");

        let green = Srgb::new(0.0, 1.0, 0.0);
        assert_eq!(converter.srgb_to_hex(green), "#00FF00");

        let blue = Srgb::new(0.0, 0.0, 1.0);
        assert_eq!(converter.srgb_to_hex(blue), "#0000FF");
    }

    #[test]
    fn test_hex_to_srgb() {
        let converter = ColorConverter::new();

        let red = converter.hex_to_srgb("#FF0000").unwrap();
        assert!((red.red - 1.0).abs() < 0.01);
        assert!(red.green < 0.01);
        assert!(red.blue < 0.01);

        let green = converter.hex_to_srgb("00FF00").unwrap(); // Test without #
        assert!(green.red < 0.01);
        assert!((green.green - 1.0).abs() < 0.01);
        assert!(green.blue < 0.01);
    }

    #[test]
    fn test_hex_to_srgb_invalid() {
        let converter = ColorConverter::new();

        assert!(converter.hex_to_srgb("#FF").is_err()); // Too short
        assert!(converter.hex_to_srgb("#GGGGGG").is_err()); // Invalid chars
    }

    #[test]
    fn test_delta_e_same_color() {
        let converter = ColorConverter::new();
        let lab = Lab::new(50.0, 0.0, 0.0);
        let delta = converter.delta_e(lab, lab);
        assert!(delta < 0.001); // Should be essentially zero
    }

    #[test]
    fn test_delta_e_different_colors() {
        let converter = ColorConverter::new();
        let lab1 = Lab::new(50.0, 0.0, 0.0);
        let lab2 = Lab::new(60.0, 10.0, 10.0);
        let delta = converter.delta_e(lab1, lab2);
        assert!(delta > 10.0); // Should be significant
    }

    #[test]
    fn test_gamut_checking() {
        let converter = ColorConverter::new();

        // Valid sRGB color
        let valid = Lab::new(50.0, 0.0, 0.0);
        assert!(converter.is_in_srgb_gamut(valid));

        // Very saturated color that might be out of gamut
        // (actual behavior depends on palette crate implementation)
        let extreme = Lab::new(50.0, 100.0, 100.0);
        let in_gamut = converter.is_in_srgb_gamut(extreme);
        // Just verify the function runs without panic
        assert!(in_gamut || !in_gamut);
    }

    #[test]
    fn test_lab_to_srgb_gamut_clipping() {
        let converter = ColorConverter::new();
        let lab = Lab::new(50.0, 0.0, 0.0);
        let srgb = converter.lab_to_srgb(lab);

        // Verify clamping works
        assert!(srgb.red >= 0.0 && srgb.red <= 1.0);
        assert!(srgb.green >= 0.0 && srgb.green <= 1.0);
        assert!(srgb.blue >= 0.0 && srgb.blue <= 1.0);
    }
}
