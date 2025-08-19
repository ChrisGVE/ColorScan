//! Paper and swatch detection module
//!
//! This module handles computer vision tasks for detecting paper surfaces
//! and isolating ink swatch regions within images.

pub mod paper;
pub mod swatch;

pub use paper::PaperDetector;
pub use swatch::SwatchDetector;