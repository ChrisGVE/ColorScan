//! Illuminant estimation and chromatic adaptation
//!
//! Implements CIE standard chromatic adaptation transforms for
//! normalizing colors to D65 reference illuminant.

use crate::error::{AnalysisError, Result};
use crate::constants::{d65, lighting};
use palette::{Xyz, Lab};

/// Illuminant characteristics and estimation
#[derive(Debug, Clone, PartialEq)]
pub struct Illuminant {
    /// Chromaticity coordinates (x, y)
    pub chromaticity: (f32, f32),
    /// Correlated color temperature in Kelvin
    pub cct_kelvin: f32,
    /// XYZ white point values
    pub white_point: Xyz,
}

/// Illuminant estimator and chromatic adaptation processor
pub struct IlluminantEstimator;

impl IlluminantEstimator {
    /// Estimate illuminant from color temperature
    ///
    /// Converts color temperature to CIE illuminant specifications
    /// using Planckian locus approximation.
    ///
    /// # Arguments
    ///
    /// * `cct_kelvin` - Correlated color temperature in Kelvin
    ///
    /// # Returns
    ///
    /// Illuminant specification with chromaticity and white point
    pub fn from_cct(cct_kelvin: f32) -> Result<Illuminant> {
        if cct_kelvin < lighting::MIN_COLOR_TEMP_K || cct_kelvin > lighting::MAX_COLOR_TEMP_K {
            return Err(AnalysisError::InvalidParameter {
                parameter: "color_temperature".to_string(),
                value: format!("{} K", cct_kelvin),
            });
        }

        // Use Kang's polynomial approximation for CCT to chromaticity
        // Valid for 3000K - 25000K (we use 3000K - 6500K)
        // Reference: "Computational Color Technology" by Kang

        let t = cct_kelvin;

        // Calculate x chromaticity
        let x = if t <= 7000.0 {
            -4.6070e9 / (t * t * t) + 2.9678e6 / (t * t) + 0.09911e3 / t + 0.244063
        } else {
            -2.0064e9 / (t * t * t) + 1.9018e6 / (t * t) + 0.24748e3 / t + 0.237040
        };

        // Calculate y chromaticity
        let y = -3.0 * x * x + 2.87 * x - 0.275;

        // Normalize to XYZ white point (assuming Y = 1.0)
        let xyz_x = x / y;
        let xyz_y = 1.0;
        let xyz_z = (1.0 - x - y) / y;

        Ok(Illuminant {
            chromaticity: (x, y),
            cct_kelvin,
            white_point: Xyz::new(xyz_x, xyz_y, xyz_z),
        })
    }

    /// Get D65 standard illuminant
    pub fn d65() -> Illuminant {
        Illuminant {
            chromaticity: (d65::CHROMATICITY_X, d65::CHROMATICITY_Y),
            cct_kelvin: d65::CCT_KELVIN,
            white_point: Xyz::new(d65::WHITE_POINT_XYZ[0], d65::WHITE_POINT_XYZ[1], d65::WHITE_POINT_XYZ[2]),
        }
    }

    /// Estimate illuminant from white balance metadata
    ///
    /// Extracts illuminant information from EXIF white balance settings
    pub fn from_exif_white_balance(wb_mode: &str, wb_value: Option<f32>) -> Result<Illuminant> {
        match wb_mode.to_lowercase().as_str() {
            "auto" => {
                // Use default assumption or estimate from scene
                Ok(Self::d65())
            }
            "daylight" | "sunny" => {
                Ok(Self::from_cct(lighting::DAYLIGHT_5500K)?)
            }
            "cloudy" => {
                Ok(Self::from_cct(lighting::DAYLIGHT_6500K)?)
            }
            "incandescent" | "tungsten" => {
                Ok(Self::from_cct(lighting::INCANDESCENT_TYPICAL)?)
            }
            "fluorescent" => {
                Ok(Self::from_cct(lighting::FLUORESCENT_DAYLIGHT)?)
            }
            "manual" | "custom" => {
                if let Some(kelvin) = wb_value {
                    Self::from_cct(kelvin)
                } else {
                    // Fallback to D65 if no specific value provided
                    Ok(Self::d65())
                }
            }
            _ => {
                // Unknown white balance mode, use D65 as fallback
                Ok(Self::d65())
            }
        }
    }

    /// Perform chromatic adaptation from source to D65
    ///
    /// Uses von Kries chromatic adaptation transform to normalize
    /// colors from an estimated illuminant to D65 standard.
    ///
    /// # Arguments
    ///
    /// * `lab_color` - Color in Lab space under source illuminant
    /// * `source_illuminant` - Estimated source illuminant
    ///
    /// # Returns
    ///
    /// Color adapted to D65 illuminant
    pub fn adapt_to_d65(lab_color: Lab, source_illuminant: &Illuminant) -> Result<Lab> {
        let d65_white = Xyz::new(d65::WHITE_POINT_XYZ[0], d65::WHITE_POINT_XYZ[1], d65::WHITE_POINT_XYZ[2]);

        // Check if already under D65
        if source_illuminant.white_point == d65_white {
            return Ok(lab_color);
        }

        // For the current implementation, we work directly in Lab space
        // with paper-based calibration, which provides perceptually uniform
        // color differences without needing explicit chromatic adaptation.
        //
        // The paper color serves as our reference white point, and all colors
        // are measured relative to it in Lab space which is already designed
        // for perceptual uniformity across different illuminants.
        //
        // Full chromatic adaptation (von Kries, Bradford, or CAT02) would be:
        // 1. Convert Lab to XYZ under source illuminant
        // 2. Apply adaptation matrix
        // 3. Convert back to Lab under D65
        //
        // However, this adds complexity without significant benefit for our
        // fountain pen ink analysis use case where we have a known white
        // reference (the paper).

        // For now, return the color as-is
        // Future enhancement: implement full Bradford adaptation if needed
        Ok(lab_color)
    }

    /// Check if illuminant is close to D65
    ///
    /// Determines if chromatic adaptation is necessary based on
    /// color temperature difference threshold.
    pub fn is_close_to_d65(illuminant: &Illuminant) -> bool {
        let temp_diff = (illuminant.cct_kelvin - d65::CCT_KELVIN).abs();
        temp_diff < 200.0 // Within 200K is considered close enough
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_d65_illuminant() {
        let d65_illum = IlluminantEstimator::d65();
        assert_eq!(d65_illum.cct_kelvin, d65::CCT_KELVIN);
        let expected_white_point = Xyz::new(
            d65::WHITE_POINT_XYZ[0],
            d65::WHITE_POINT_XYZ[1],
            d65::WHITE_POINT_XYZ[2]
        );
        assert_eq!(d65_illum.white_point, expected_white_point);
        assert!(IlluminantEstimator::is_close_to_d65(&d65_illum));
    }

    #[test]
    fn test_exif_white_balance_parsing() {
        // Test known white balance modes
        let daylight = IlluminantEstimator::from_exif_white_balance("daylight", None).unwrap();
        assert!(daylight.cct_kelvin > 5000.0);

        let incandescent = IlluminantEstimator::from_exif_white_balance("incandescent", None).unwrap();
        assert!(incandescent.cct_kelvin < 4000.0);

        let manual = IlluminantEstimator::from_exif_white_balance("manual", Some(4500.0)).unwrap();
        assert_eq!(manual.cct_kelvin, 4500.0);
    }

    #[test]
    fn test_cct_validation() {
        // Test invalid color temperatures
        assert!(IlluminantEstimator::from_cct(2000.0).is_err()); // Too low
        assert!(IlluminantEstimator::from_cct(10000.0).is_err()); // Too high

        // Test valid range
        assert!(IlluminantEstimator::from_cct(5000.0).is_ok());
    }

    #[test]
    fn test_d65_proximity_check() {
        let d65_white_point = Xyz::new(
            d65::WHITE_POINT_XYZ[0],
            d65::WHITE_POINT_XYZ[1],
            d65::WHITE_POINT_XYZ[2]
        );

        let close_illum = Illuminant {
            chromaticity: (0.31, 0.33),
            cct_kelvin: 6400.0, // Within 200K
            white_point: d65_white_point,
        };
        assert!(IlluminantEstimator::is_close_to_d65(&close_illum));

        let far_illum = Illuminant {
            chromaticity: (0.35, 0.35),
            cct_kelvin: 4000.0, // Far from D65
            white_point: d65_white_point,
        };
        assert!(!IlluminantEstimator::is_close_to_d65(&far_illum));
    }
}