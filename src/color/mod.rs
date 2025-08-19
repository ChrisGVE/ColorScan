//! Color analysis and conversion module
//!
//! This module handles color space conversions, statistical analysis
//! of color regions, and generation of representative color values.

pub mod conversion;
pub mod analysis;

pub use conversion::ColorConverter;
pub use analysis::ColorAnalyzer;