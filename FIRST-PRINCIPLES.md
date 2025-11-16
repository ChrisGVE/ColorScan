# FIRST-PRINCIPLES.md - Scan Colors Project

## Principle 1: Test Driven Development

**Philosophy**: Systematic TDD - write unit test immediately after each logical unit of code.

**Implementation implications**:
- Each logical unit (function, object, method) needs ≥1 unit test
- Cover edge cases and validation errors (multiple tests per unit)
- Run tests after atomic changes; amend tests only after first run
- Use LSP to identify calling/called code relationships
- Test color accuracy with known reference values (D65, ColorChecker patches)
- Benchmark performance targets (100ms analysis time)

## Principle 2: Leverage Existing Solutions

**Philosophy**: Reuse mature, well-maintained libraries rather than reinventing functionality.

**Implementation Implications**:
- Prefer established, actively maintained libraries with strong community support
- Choose mature solutions with proven track record (but not stale/unmaintained)
- Follow standard protocols and interfaces when available
- Ensure compatibility with existing toolchains and ecosystems
- Evaluate library health: recent updates, active issues/PRs, documentation quality
- Align with industry best practices and conventions
- Use industry-standard libraries: OpenCV for image processing, palette for color conversions

## Principle 3: Color Accuracy Over Speed

**Philosophy**: Prioritize accurate, perceptually meaningful color measurements over processing speed.

**Implementation implications**:
- Use CIE Lab/LCh color spaces for perceptual uniformity
- Implement CIEDE2000 (ΔE00) for color difference calculations
- Validate against ColorChecker reference patches where applicable
- Target ΔE < 3.0 accuracy for fountain pen ink differentiation
- Optimize only after correctness is verified
- Document accuracy limitations and confidence metrics

## Principle 4: D65 Standard Illuminant

**Philosophy**: Use D65 (6504K daylight) as the universal anchor point for all color calibration.

**Implementation implications**:
- CIE XYZ D65 reference: [0.95047, 1.0, 1.08883]
- All white balance corrections normalize to D65
- Support chromatic adaptation for other illuminants (3000K-6500K range)
- Use CAT02 or von Kries transform for illuminant adaptation
- Document D65 assumption in API and error messages
- Provide warnings when lighting conditions deviate significantly from D65

## Principle 5: Minimal User Intervention

**Philosophy**: Single function call from image path to calibrated color result.

**Implementation implications**:
- API: `analyze_swatch(image_path: &Path) -> Result<ColorResult, AnalysisError>`
- Library handles all image loading, EXIF extraction, and format detection
- Automatic detection of paper background and color swatch boundaries
- No manual calibration steps or user-provided reference points required
- Sensible defaults for all internal parameters
- Optional advanced API for fine-grained control if needed

## Principle 6: Real-World Robustness

**Philosophy**: Handle diverse lighting conditions, paper types, smartphone camera variations, and practical photography setups.

**Implementation implications**:
- Support white, cream, and off-white paper backgrounds
- Handle lighting temperature range: 3000K-6500K
- Adapt to smartphone camera color matrix variations
- Detect and exclude foreign objects (tape, rulers, weights) from analysis
- Assume ink swatch itself is not obstructed by transparent materials
- Detect and handle gradient saturation in cotton swab swatches
- Account for ink sheen and shimmer particles where possible
- Validate against multiple smartphone camera models
- Provide confidence scores for ambiguous conditions
- Provide user guidance for optimal photography setup

## Principle 7: Comprehensive Error Reporting

**Philosophy**: Every error provides actionable information with clear recovery hints.

**Implementation implications**:
- Custom error types with detailed context
- User-friendly error messages explaining what went wrong
- Recovery hints suggesting corrective actions
- Distinguish between recoverable and fatal errors
- Include relevant metadata (file path, detected conditions, thresholds)
- Log warnings for suboptimal but processable conditions
- Provide confidence metrics alongside results

## Principle 8: Modular Architecture

**Philosophy**: Separate concerns into distinct, testable modules with clear interfaces.

**Implementation implications**:
- Distinct modules: calibration, detection, color analysis, EXIF handling
- Each module independently testable
- Clear data flow: EXIF → calibration → detection → color analysis
- Minimal coupling between modules
- Well-defined types at module boundaries
- Support future extensions (e.g., batch processing, alternative algorithms)
