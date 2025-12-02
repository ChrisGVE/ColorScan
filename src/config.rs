//! Configuration structures for the inkswatch_colorscan analysis pipeline.
//!
//! This module defines all tunable parameters for color analysis,
//! organized into logical groups for preprocessing, detection, and extraction.
//!
//! # Configuration Loading
//!
//! Configuration can be loaded from JSON files or constructed programmatically:
//!
//! ```no_run
//! use inkswatch_colorscan::PipelineConfig;
//! use std::path::Path;
//!
//! // Load from file
//! let config = PipelineConfig::from_json_file(Path::new("config.json"))?;
//!
//! // Or use defaults
//! let config = PipelineConfig::default_experiment_0();
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Configuration Sections
//!
//! - [`PreprocessingConfig`]: EXIF correction, white balance settings
//! - [`PaperDetectionConfig`]: Edge detection and contour parameters
//! - [`SwatchDetectionConfig`]: Ink region isolation settings
//! - [`ColorExtractionConfig`]: Statistical extraction methods

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use palette::Lab;

/// Complete pipeline configuration for color analysis.
///
/// Contains all parameters needed to process an image from input to color result.
/// Can be serialized to/from JSON for reproducible experiments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Input directory or file path
    pub input_path: PathBuf,

    /// Output directory for artifacts
    pub output_path: PathBuf,

    /// Preprocessing configuration
    pub preprocessing: PreprocessingConfig,

    /// Paper detection configuration
    pub paper_detection: PaperDetectionConfig,

    /// Swatch detection configuration
    pub swatch_detection: SwatchDetectionConfig,

    /// Color extraction configuration
    pub color_extraction: ColorExtractionConfig,
}

/// Preprocessing parameters applied before detection.
///
/// Controls image orientation correction and white balance normalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Apply EXIF orientation correction
    pub exif_correction: bool,

    /// White balance correction settings
    pub white_balance: WhiteBalanceConfig,

    /// Use swatch-first detection mode
    /// When true, estimates WB from paper band outside detected rectangle
    /// and applies WB to full image before swatch detection.
    /// This handles cases where rectangle detection finds swatch instead of paper.
    #[serde(default)]
    pub swatch_first_mode: bool,
}

/// White balance correction parameters.
///
/// When enabled, the pipeline estimates the paper color and adjusts
/// the image to normalize to the target paper color (typically neutral white).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhiteBalanceConfig {
    /// Enable white balance correction
    pub enabled: bool,

    /// Target paper color under D65 (L*, a*, b*)
    pub target_paper: LabColor,
}

/// Lab color representation for configuration files.
///
/// Uses CIE L*a*b* color space coordinates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabColor {
    pub l: f32,
    pub a: f32,
    pub b: f32,
}

impl From<LabColor> for Lab {
    fn from(color: LabColor) -> Self {
        Lab::new(color.l, color.a, color.b)
    }
}

/// Paper detection parameters.
///
/// Controls the edge detection and contour analysis used to locate
/// the paper card in the image. Parameters affect sensitivity and
/// which rectangular regions are considered valid paper candidates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperDetectionConfig {
    /// Minimum paper area as fraction of image (0.0-1.0)
    pub min_area_ratio: f64,

    /// Maximum rectification angle in degrees
    pub max_rectification_angle: f64,

    /// Polygon approximation epsilon as fraction of perimeter
    pub poly_approx_epsilon: f64,

    /// Minimum aspect ratio for valid card
    pub min_aspect_ratio: f64,

    /// Maximum aspect ratio for valid card
    pub max_aspect_ratio: f64,

    /// Canny edge detection low threshold
    pub canny_low_threshold: f64,

    /// Canny edge detection high threshold
    pub canny_high_threshold: f64,

    /// Gaussian blur kernel size (must be odd)
    pub gaussian_blur_kernel_size: i32,

    /// Gaussian blur sigma
    pub gaussian_blur_sigma: f64,

    /// Edge dilation kernel size
    pub edge_dilation_kernel_size: i32,

    /// Centrality weight in scoring (0.0-1.0)
    pub centrality_weight: f64,
}

/// Swatch detection parameters.
///
/// Controls how ink regions are identified and isolated from the paper background.
/// Uses color difference thresholds and morphological operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwatchDetectionConfig {
    /// Minimum color difference (ΔE) to separate ink from paper
    pub min_delta_e: f64,

    /// Minimum swatch area as fraction of paper area (0.0-1.0)
    pub min_area_ratio: f64,

    /// Maximum swatch area as fraction of paper area (0.0-1.0)
    pub max_area_ratio: f64,

    /// Morphological kernel size for noise removal
    pub morph_kernel_size: i32,

    /// Boundary erosion size in pixels
    pub boundary_erosion: i32,
}

/// Color extraction parameters.
///
/// Controls how the representative ink color is computed from the
/// detected swatch pixels. Includes outlier filtering and validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorExtractionConfig {
    /// Extraction method: "MedianMean", "Darkest", "MostSaturated", "Mode"
    pub method: String,

    /// Minimum color difference from paper for valid ink (ΔE)
    pub min_ink_delta_e: f32,

    /// Low percentile for outlier removal (0.0-100.0)
    pub outlier_percentile_low: f32,

    /// High percentile for outlier removal (0.0-100.0)
    pub outlier_percentile_high: f32,

    /// Minimum pixels required after filtering
    pub min_pixels_threshold: usize,

    /// Lightness bounds for validation
    pub lightness_bounds: LightnessBounds,
}

/// Lightness validation bounds.
///
/// Defines the acceptable range of L* values for ink colors.
/// Values outside this range may indicate detection errors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightnessBounds {
    /// Minimum L* value
    pub min: f32,

    /// Maximum L* value
    pub max: f32,
}

impl PipelineConfig {
    /// Create default configuration (Experiment 0 baseline)
    pub fn default_experiment_0() -> Self {
        Self {
            input_path: PathBuf::from("validation/swatch_samples"),
            output_path: PathBuf::from("validation/Experiment 0/images"),
            preprocessing: PreprocessingConfig {
                exif_correction: true,
                white_balance: WhiteBalanceConfig {
                    enabled: true,
                    target_paper: LabColor {
                        l: 95.0,
                        a: 0.0,
                        b: 0.0,
                    },
                },
                swatch_first_mode: false, // Default to standard pipeline
            },
            paper_detection: PaperDetectionConfig {
                min_area_ratio: 0.05,
                max_rectification_angle: 45.0,
                poly_approx_epsilon: 0.02,
                min_aspect_ratio: 0.33,
                max_aspect_ratio: 3.0,
                canny_low_threshold: 30.0,
                canny_high_threshold: 90.0,
                gaussian_blur_kernel_size: 5,
                gaussian_blur_sigma: 1.5,
                edge_dilation_kernel_size: 3,
                centrality_weight: 0.3,
            },
            swatch_detection: SwatchDetectionConfig {
                min_delta_e: 15.0,
                min_area_ratio: 0.10,
                max_area_ratio: 0.90,
                morph_kernel_size: 3,
                boundary_erosion: 2,
            },
            color_extraction: ColorExtractionConfig {
                method: "MedianMean".to_string(),
                min_ink_delta_e: 15.0,
                outlier_percentile_low: 15.0,
                outlier_percentile_high: 85.0,
                min_pixels_threshold: 10,
                lightness_bounds: LightnessBounds {
                    min: 5.0,
                    max: 95.0,
                },
            },
        }
    }

    /// Load configuration from JSON file
    pub fn from_json_file(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to JSON file
    pub fn to_json_file(&self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}
