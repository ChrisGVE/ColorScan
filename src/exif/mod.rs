//! EXIF metadata extraction module
//!
//! This module handles extraction and interpretation of EXIF metadata
//! relevant to color analysis, including white balance, color space,
//! and camera settings.

pub mod extractor;

pub use extractor::ExifExtractor;