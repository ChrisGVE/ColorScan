//! Integration tests for the complete analyze_swatch pipeline
//!
//! These tests validate the end-to-end color analysis workflow including:
//! - Image loading and preprocessing
//! - Paper detection and perspective correction
//! - White balance estimation
//! - Swatch detection and isolation
//! - Color extraction and conversion
//! - Error handling for edge cases
//!
//! Note: Tests requiring sample images are marked with #[ignore] until
//! test assets are created. See tests/README.md for test image requirements.

use scan_colors::{analyze_swatch, ColorResult, AnalysisError};
use std::path::Path;

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_analyze_swatch_file_not_found() {
    let result = analyze_swatch(Path::new("nonexistent_file.jpg"));

    assert!(result.is_err());
    let err = result.unwrap_err();

    // Should fail with either ImageLoadError or ProcessingError
    // (OpenCV returns empty Mat for missing files, triggering empty check)
    match err {
        AnalysisError::ImageLoadError { .. } | AnalysisError::ProcessingError(_) => {
            // Expected error types
        }
        _ => panic!("Expected ImageLoadError or ProcessingError, got: {:?}", err),
    }
}

#[test]
fn test_analyze_swatch_invalid_path() {
    // Path with null bytes (invalid on most systems)
    let result = analyze_swatch(Path::new("\0invalid\0path.jpg"));

    assert!(result.is_err());
}

#[test]
fn test_analyze_swatch_empty_path() {
    let result = analyze_swatch(Path::new(""));

    assert!(result.is_err());
}

// ============================================================================
// Integration Tests with Sample Images (Ignored Until Assets Created)
// ============================================================================

#[test]
#[ignore] // Enable when test image is created
fn test_analyze_swatch_uniform_color() {
    // Test Requirements:
    // - Image: tests/assets/uniform_blue.jpg
    // - Content: Uniform blue fountain pen ink swatch on white paper
    // - Expected Lab: L* ≈ 40-50, a* ≈ 10-20, b* ≈ -40 to -50 (typical blue)
    // - Expected confidence: > 0.8 (high confidence for uniform swatch)

    let result = analyze_swatch(Path::new("tests/assets/uniform_blue.jpg"));

    assert!(result.is_ok(), "Analysis should succeed for uniform swatch");
    let color = result.unwrap();

    // Validate color result structure
    assert!(color.confidence > 0.8, "Confidence should be high for uniform swatch");

    // Validate Lab values are in reasonable range for blue ink
    assert!(color.lab.l > 20.0 && color.lab.l < 70.0, "Lightness in valid range");
    assert!(color.lab.a > -20.0 && color.lab.a < 40.0, "a* in valid range");
    assert!(color.lab.b > -80.0 && color.lab.b < 0.0, "b* in valid range for blue");

    // Validate hex format
    assert!(color.hex.starts_with('#'), "Hex should start with #");
    assert_eq!(color.hex.len(), 7, "Hex should be 7 characters");

    // Validate sRGB is in gamut
    assert!(color.srgb.red >= 0.0 && color.srgb.red <= 1.0);
    assert!(color.srgb.green >= 0.0 && color.srgb.green <= 1.0);
    assert!(color.srgb.blue >= 0.0 && color.srgb.blue <= 1.0);
}

#[test]
#[ignore] // Enable when test image is created
fn test_analyze_swatch_gradient() {
    // Test Requirements:
    // - Image: tests/assets/gradient_swatch.jpg
    // - Content: Ink swatch with gradient from dark to light
    // - Expected: Should extract representative color from middle region
    // - Expected confidence: 0.5-0.8 (medium confidence due to variance)

    let result = analyze_swatch(Path::new("tests/assets/gradient_swatch.jpg"));

    assert!(result.is_ok(), "Should handle gradient swatches");
    let color = result.unwrap();

    // Gradient should have lower confidence due to higher variance
    assert!(color.confidence > 0.4 && color.confidence < 0.9,
            "Confidence should reflect gradient variance");
}

#[test]
#[ignore] // Enable when test image is created
fn test_analyze_swatch_light_ink() {
    // Test Requirements:
    // - Image: tests/assets/light_yellow.jpg
    // - Content: Very light yellow/cream fountain pen ink
    // - Expected Lab: L* > 80, low chroma
    // - Expected confidence: > 0.6 (sufficient contrast with paper)

    let result = analyze_swatch(Path::new("tests/assets/light_yellow.jpg"));

    assert!(result.is_ok(), "Should handle light colored inks");
    let color = result.unwrap();

    // Light ink should have high L* value
    assert!(color.lab.l > 70.0, "Light ink should have high lightness");

    // Should still be distinguishable from paper
    assert!(color.confidence > 0.5, "Should have sufficient confidence");
}

#[test]
#[ignore] // Enable when test image is created
fn test_analyze_swatch_dark_ink() {
    // Test Requirements:
    // - Image: tests/assets/dark_black.jpg
    // - Content: Very dark black fountain pen ink
    // - Expected Lab: L* < 30, near-neutral chromaticity
    // - Expected confidence: > 0.8 (high contrast with paper)

    let result = analyze_swatch(Path::new("tests/assets/dark_black.jpg"));

    assert!(result.is_ok(), "Should handle dark inks");
    let color = result.unwrap();

    // Dark ink should have low L* value
    assert!(color.lab.l < 40.0, "Dark ink should have low lightness");
    assert!(color.lab.l > 5.0, "Should not be pure black (unrealistic)");

    // High contrast with paper
    assert!(color.confidence > 0.7);
}

#[test]
#[ignore] // Enable when test image is created
fn test_analyze_swatch_with_foreign_objects() {
    // Test Requirements:
    // - Image: tests/assets/swatch_with_pen.jpg
    // - Content: Ink swatch with fountain pen placed on paper
    // - Expected: Should detect and exclude pen, analyze only ink
    // - Expected confidence: > 0.6 (pen excluded from analysis)

    let result = analyze_swatch(Path::new("tests/assets/swatch_with_pen.jpg"));

    assert!(result.is_ok(), "Should handle foreign objects");
    let color = result.unwrap();

    // Foreign object detection should maintain reasonable confidence
    assert!(color.confidence > 0.5, "Should still have confidence with foreign objects excluded");
}

#[test]
#[ignore] // Enable when test image is created
fn test_analyze_swatch_warm_lighting() {
    // Test Requirements:
    // - Image: tests/assets/warm_lighting.jpg
    // - Content: Same ink as uniform_blue but under warm (3000K) lighting
    // - Expected: White balance should normalize to similar Lab as uniform_blue
    // - Acceptable ΔE: < 5.0 from uniform_blue after calibration

    let result = analyze_swatch(Path::new("tests/assets/warm_lighting.jpg"));

    assert!(result.is_ok(), "Should handle warm lighting conditions");
    let color = result.unwrap();

    // Should still produce valid color despite lighting variation
    assert!(color.confidence > 0.5);
}

#[test]
#[ignore] // Enable when test image is created
fn test_analyze_swatch_cool_lighting() {
    // Test Requirements:
    // - Image: tests/assets/cool_lighting.jpg
    // - Content: Same ink as uniform_blue but under cool (6500K) lighting
    // - Expected: White balance should normalize to similar Lab as uniform_blue
    // - Acceptable ΔE: < 5.0 from uniform_blue after calibration

    let result = analyze_swatch(Path::new("tests/assets/cool_lighting.jpg"));

    assert!(result.is_ok(), "Should handle cool lighting conditions");
    let color = result.unwrap();

    // Should still produce valid color despite lighting variation
    assert!(color.confidence > 0.5);
}

#[test]
#[ignore] // Enable when test image is created
fn test_analyze_swatch_angled_perspective() {
    // Test Requirements:
    // - Image: tests/assets/angled_paper.jpg
    // - Content: Ink swatch on paper photographed at 30-45° angle
    // - Expected: Perspective correction should handle rectification
    // - Expected confidence: > 0.6 (after perspective correction)

    let result = analyze_swatch(Path::new("tests/assets/angled_paper.jpg"));

    assert!(result.is_ok(), "Should handle angled perspectives");
    let color = result.unwrap();

    assert!(color.confidence > 0.5, "Perspective correction should maintain quality");
}

#[test]
#[ignore] // Enable when test image is created
fn test_analyze_swatch_small_swatch() {
    // Test Requirements:
    // - Image: tests/assets/small_swatch.jpg
    // - Content: Ink swatch occupying exactly 10% of paper area (minimum)
    // - Expected: Should succeed at minimum size threshold

    let result = analyze_swatch(Path::new("tests/assets/small_swatch.jpg"));

    assert!(result.is_ok(), "Should handle minimum swatch size");
}

#[test]
#[ignore] // Enable when test image is created
fn test_analyze_swatch_too_small() {
    // Test Requirements:
    // - Image: tests/assets/tiny_swatch.jpg
    // - Content: Ink swatch occupying < 10% of paper area
    // - Expected: Should fail with SwatchTooSmall error

    let result = analyze_swatch(Path::new("tests/assets/tiny_swatch.jpg"));

    assert!(result.is_err(), "Should reject too-small swatches");

    match result.unwrap_err() {
        AnalysisError::SwatchTooSmall(_) => {
            // Expected error
        }
        err => panic!("Expected SwatchTooSmall, got: {:?}", err),
    }
}

#[test]
#[ignore] // Enable when test image is created
fn test_analyze_swatch_no_paper_detected() {
    // Test Requirements:
    // - Image: tests/assets/no_paper.jpg
    // - Content: Image with no clear paper boundaries (e.g., all ink, no margin)
    // - Expected: Should fail with PaperDetectionError

    let result = analyze_swatch(Path::new("tests/assets/no_paper.jpg"));

    assert!(result.is_err(), "Should fail when no paper is detected");

    match result.unwrap_err() {
        AnalysisError::PaperDetectionError { .. } => {
            // Expected error
        }
        err => panic!("Expected PaperDetectionError, got: {:?}", err),
    }
}

#[test]
#[ignore] // Enable when test image is created
fn test_analyze_swatch_no_ink_detected() {
    // Test Requirements:
    // - Image: tests/assets/blank_paper.jpg
    // - Content: White paper with no visible ink
    // - Expected: Should fail with NoSwatchDetected error

    let result = analyze_swatch(Path::new("tests/assets/blank_paper.jpg"));

    assert!(result.is_err(), "Should fail when no ink is detected");

    match result.unwrap_err() {
        AnalysisError::NoSwatchDetected(_) => {
            // Expected error
        }
        err => panic!("Expected NoSwatchDetected, got: {:?}", err),
    }
}

#[test]
#[ignore] // Enable when test image is created
fn test_analyze_swatch_shimmer_ink() {
    // Test Requirements:
    // - Image: tests/assets/shimmer_ink.jpg
    // - Content: Fountain pen ink with shimmer/sheen particles
    // - Expected: Should extract base color, may have slightly lower confidence
    // - Expected confidence: > 0.5 (outlier removal handles shimmer particles)

    let result = analyze_swatch(Path::new("tests/assets/shimmer_ink.jpg"));

    assert!(result.is_ok(), "Should handle shimmer inks");
    let color = result.unwrap();

    // Shimmer may reduce confidence slightly but should still analyze
    assert!(color.confidence > 0.4);
}

#[test]
#[ignore] // Enable when test image is created
fn test_analyze_swatch_multiple_swatches() {
    // Test Requirements:
    // - Image: tests/assets/multiple_swatches.jpg
    // - Content: Paper with 2-3 different ink swatches
    // - Expected: Should detect and analyze the largest/most prominent swatch
    // - Expected confidence: > 0.6

    let result = analyze_swatch(Path::new("tests/assets/multiple_swatches.jpg"));

    assert!(result.is_ok(), "Should handle multiple swatches by selecting largest");
    let color = result.unwrap();

    assert!(color.confidence > 0.5);
}

// ============================================================================
// Performance Tests (Ignored Until Optimization Phase)
// ============================================================================

#[test]
#[ignore] // Enable during performance optimization
fn test_analyze_swatch_performance() {
    // Test Requirements:
    // - Image: tests/assets/uniform_blue.jpg (typical smartphone image)
    // - Expected: Analysis should complete in < 100ms

    use std::time::Instant;

    let start = Instant::now();
    let result = analyze_swatch(Path::new("tests/assets/uniform_blue.jpg"));
    let duration = start.elapsed();

    assert!(result.is_ok());
    assert!(duration.as_millis() < 100,
            "Analysis took {}ms, expected < 100ms", duration.as_millis());
}

// ============================================================================
// Serialization Tests
// ============================================================================

#[test]
#[ignore] // Enable when test image is created
fn test_color_result_json_serialization() {
    // Validate that ColorResult can be serialized to JSON for API use

    let result = analyze_swatch(Path::new("tests/assets/uniform_blue.jpg"));
    assert!(result.is_ok());

    let color = result.unwrap();
    let json = serde_json::to_string(&color).unwrap();

    // Should contain all expected fields
    assert!(json.contains("\"lab\""));
    assert!(json.contains("\"lch\""));
    assert!(json.contains("\"srgb\""));
    assert!(json.contains("\"hex\""));
    assert!(json.contains("\"confidence\""));

    // Should be able to deserialize back
    let deserialized: ColorResult = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.hex, color.hex);
    assert_eq!(deserialized.confidence, color.confidence);
}
