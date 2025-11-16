# Integration Test Assets

This directory contains integration tests for the `analyze_swatch` pipeline. Most tests are currently marked with `#[ignore]` until test image assets are created.

## Test Image Requirements

All test images should be:
- Format: JPEG or PNG
- Resolution: Typical smartphone camera (3000x4000 or similar)
- Lighting: Well-lit but varied as specified per test
- Paper: White or off-white standard paper/cardstock

### Required Test Images

#### Basic Color Tests

**tests/assets/uniform_blue.jpg**
- Uniform blue fountain pen ink swatch
- Expected Lab: L* ≈ 40-50, a* ≈ 10-20, b* ≈ -40 to -50
- Swatch size: 15-20% of paper area
- Clean, sharp focus

**tests/assets/dark_black.jpg**
- Very dark black/near-black fountain pen ink
- Expected Lab: L* < 30, near-neutral a* and b*
- High contrast with paper

**tests/assets/light_yellow.jpg**
- Very light yellow/cream colored ink
- Expected Lab: L* > 80, low chroma
- Should still be distinguishable from paper

#### Gradient and Variance Tests

**tests/assets/gradient_swatch.jpg**
- Ink swatch with visible gradient from dark to light
- Could be created with diluted ink or edge effects
- Tests outlier removal and robust color extraction

**tests/assets/shimmer_ink.jpg**
- Fountain pen ink with shimmer/sheen particles
- Tests handling of specular highlights and particle effects

#### Lighting Condition Tests

**tests/assets/warm_lighting.jpg**
- Same blue ink as uniform_blue.jpg
- Photographed under warm (incandescent, ~3000K) lighting
- Should normalize to similar Lab values after white balance

**tests/assets/cool_lighting.jpg**
- Same blue ink as uniform_blue.jpg
- Photographed under cool (daylight, ~6500K) lighting
- Should normalize to similar Lab values after white balance

#### Perspective and Geometry Tests

**tests/assets/angled_paper.jpg**
- Ink swatch on paper photographed at 30-45° angle
- Tests perspective correction and homography

#### Size Threshold Tests

**tests/assets/small_swatch.jpg**
- Ink swatch occupying exactly ~10% of paper area (minimum threshold)
- Should succeed at boundary

**tests/assets/tiny_swatch.jpg**
- Ink swatch occupying < 10% of paper area
- Should fail with SwatchTooSmall error

#### Foreign Object Tests

**tests/assets/swatch_with_pen.jpg**
- Ink swatch with fountain pen placed on paper
- Tests foreign object detection and exclusion

**tests/assets/multiple_swatches.jpg**
- Paper with 2-3 different colored ink swatches
- Should detect and analyze largest/most prominent

#### Error Case Tests

**tests/assets/blank_paper.jpg**
- Clean white paper with no ink
- Should fail with NoSwatchDetected error

**tests/assets/no_paper.jpg**
- Image with no clear paper boundaries
- Could be all ink or tight crop with no margin
- Should fail with PaperDetectionError

## Creating Test Images

### Recommended Setup

1. **Camera**: Modern smartphone (iPhone 12+, Pixel 6+, etc.)
2. **Paper**: Standard white printer paper or index cards
3. **Ink**: Various fountain pen inks in different colors
4. **Lighting**:
   - Daylight: Natural window light (indirect)
   - Warm: Incandescent bulb (~3000K)
   - Cool: LED daylight bulb (~6500K)

### Photography Guidelines

- **Distance**: 30-40cm from paper surface
- **Focus**: Tap on ink swatch to ensure sharp focus
- **Exposure**: Avoid overexposure (paper should not be blown out)
- **Shadows**: Minimize harsh shadows
- **Background**: Place paper on neutral surface
- **Angle**: Perpendicular to paper unless testing angled perspectives

### Swatch Creation

- **Uniform swatches**: Use broad nib, apply ink evenly
- **Gradient swatches**: Start dark, lighten pressure or dilute
- **Size reference**: Measure swatch area vs paper area to ensure correct proportions

## Running Tests

```bash
# Run only non-ignored tests (error handling)
cargo test --test integration_test

# Run all tests (will skip ignored tests by default)
cargo test --test integration_test -- --include-ignored

# Run specific test
cargo test --test integration_test test_analyze_swatch_uniform_color -- --ignored
```

## Validating Test Images

Before using test images in CI/CD:

1. Run `cargo test --test integration_test -- --include-ignored`
2. Verify expected Lab values are within acceptable ranges
3. Adjust test assertions if needed based on actual ink characteristics
4. Document actual measured values in test comments

## Future Enhancements

- Add EXIF metadata to test images with known white balance settings
- Create synthetic test images using image generation
- Add color reference patches (e.g., ColorChecker) in some images
- Performance benchmark suite with various image sizes
