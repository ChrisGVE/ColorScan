# ALGO.md - Algorithm Repository

## Purpose

This file maintains a record of all algorithms designed for the scan_colors project.
Each algorithm is also stored in the knowledge graph with the tag listed here.

## Active Algorithms

### D65 Chromatic Adaptation

**Tag**: `algo-d65-chromatic-adaptation`
**Created**: 2025-01-19
**Last Modified**: 2025-01-19
**Purpose**: Normalize colors from any illuminant to D65 standard reference
**Performance Requirements**: O(1) operation, <1ms per color conversion
**Commit**: [TBD - initial implementation]

```
ALGORITHM: D65 Chromatic Adaptation
INPUT: Lab color under source illuminant, source illuminant XYZ white point
OUTPUT: Lab color adapted to D65 illuminant

STEPS:
1. Convert Lab to XYZ using source illuminant white point:
   - X = (L + 16) / 116 + a/500
   - Y = (L + 16) / 116  
   - Z = (L + 16) / 116 - b/200
   - Apply white point scaling: XYZ *= source_white_point

2. Apply von Kries chromatic adaptation transform:
   - Transform XYZ to cone response space (LMS) using CAT02 matrix
   - Scale cone responses: LMS_adapted = LMS * (D65_white_LMS / source_white_LMS)
   - Transform back to XYZ using inverse CAT02 matrix

3. Convert adapted XYZ back to Lab using D65 white point:
   - Normalize by D65 white point: XYZ /= D65_WHITE_POINT_XYZ
   - Apply Lab conversion with D65 illuminant

EDGE CASES:
- Skip adaptation if source illuminant is within 200K of D65
- Clip out-of-gamut colors in XYZ space before Lab conversion
- Handle zero or negative XYZ values gracefully

PERFORMANCE OPTIMIZATIONS:
- Cache transformation matrices for repeated use
- Use lookup tables for common illuminant transformations
- Apply SIMD operations for batch processing

DECISION POINTS:
- User chose D65 as universal anchor point for consistency
- CAT02 preferred over Bradford for better accuracy with digital cameras
- 200K threshold chosen to avoid unnecessary computation for small differences
```

### Paper-Based White Balance Estimation

**Tag**: `algo-paper-white-balance`
**Created**: 2025-01-19
**Last Modified**: 2025-01-19
**Purpose**: Estimate scene illuminant from paper/background regions
**Performance Requirements**: <10ms processing time, minimum 1000 pixels required
**Commit**: [TBD - initial implementation]

```
ALGORITHM: Paper-Based White Balance Estimation
INPUT: RGB image, binary mask of paper regions
OUTPUT: Estimated illuminant XYZ white point, confidence score (0.0-1.0)

STEPS:
1. Extract RGB values from paper regions using mask:
   - Apply mask to get paper-only pixels
   - Require minimum 1000 pixels for reliability
   - Remove specular highlights (top 5% of lightness values)

2. Compute robust statistics:
   - Calculate median RGB (less sensitive to outliers)
   - Compute 25th-75th percentile range for variance estimation
   - Check for neutral paper assumption (low chroma in Lab space)

3. Estimate illuminant:
   - Convert paper RGB to XYZ assuming D65 initially
   - Solve for illuminant scaling: actual_illuminant = paper_XYZ / expected_white_XYZ
   - Validate color temperature is within 3000K-6500K range
   - Check chromaticity coordinates are within reasonable bounds

4. Compute confidence score:
   - High confidence: large paper area, low color variance, neutral paper
   - Medium confidence: adequate area but higher variance
   - Low confidence: small area, high variance, or non-neutral paper

EDGE CASES:
- Fallback to EXIF-based estimation if paper detection fails
- Handle colored paper by detecting non-neutral background
- Reject estimation if color temperature is outside valid range

DECISION POINTS:
- User approved focusing on common lighting (3000K-6500K) vs full spectrum
- Median used over mean for outlier robustness per user preference
- 1000 pixel minimum chosen for statistical validity
```

### Adaptive Paper Detection

**Tag**: `algo-adaptive-paper-detection`
**Created**: 2025-01-19
**Last Modified**: 2025-01-19
**Purpose**: Detect and rectify paper/card surface in image using computer vision
**Performance Requirements**: <30ms processing time, handle 4000x3000 pixel images
**Commit**: [TBD - initial implementation]

```
ALGORITHM: Adaptive Paper Detection
INPUT: RGB image from smartphone camera
OUTPUT: Rectified image, paper contour coordinates, homography matrix

STEPS:
1. Preprocessing:
   - Convert RGB to Lab color space for better lightness analysis
   - Use L* channel for brightness-based detection
   - Apply Gaussian blur (sigma=1.0) to reduce noise

2. Adaptive thresholding:
   - Use Otsu's method for initial threshold estimation
   - Apply local adaptive threshold for uneven lighting
   - Target: detect bright regions (paper) vs darker background

3. Morphological operations:
   - Opening operation (kernel size 3x3) to remove noise
   - Closing operation to fill small gaps in paper regions

4. Contour detection and filtering:
   - Find all contours in thresholded image
   - Filter by area: minimum 10% of total image area
   - Filter by aspect ratio: approximately rectangular shapes

5. Polygon approximation:
   - Use approxPolyDP with epsilon = 2% of perimeter
   - Select contours with approximately 4 corners
   - Validate corners form near-rectangular shape (angles ~90°)

6. Homography computation and rectification:
   - Order corners (top-left, top-right, bottom-right, bottom-left)
   - Compute homography to standard rectangle
   - Apply perspective transform to rectify image
   - Limit maximum rectification angle to 45° to avoid distortion

EDGE CASES:
- Multiple paper regions: select largest rectangular contour
- Rounded corners: allow polygon approximation tolerance
- Poor lighting: use multiple threshold values and combine results
- No clear paper boundary: fallback to whole-image analysis

PERFORMANCE OPTIMIZATIONS:
- Downscale image for detection, upscale coordinates for rectification
- Use integral images for efficient region analysis
- Cache morphological kernels

DECISION POINTS:
- User approved 10% minimum area to ensure adequate swatch region
- 45° maximum rectification angle chosen to preserve image quality
- Otsu + adaptive combination chosen over single threshold method
```

### Robust Color Extraction

**Tag**: `algo-robust-color-extraction`
**Created**: 2025-01-19
**Last Modified**: 2025-01-19
**Purpose**: Extract representative color from ink region handling gradients and transparency
**Performance Requirements**: <20ms processing, handle 10-80% swatch area range
**Commit**: [TBD - initial implementation]

```
ALGORITHM: Robust Color Extraction
INPUT: RGB image, ink mask, paper color model
OUTPUT: Representative Lab color, confidence score (0.0-1.0)

STEPS:
1. Extract ink pixels:
   - Apply binary mask to isolate ink regions
   - Convert RGB to Lab color space using D65 illuminant
   - Validate swatch area is 10-80% of total image

2. Outlier removal:
   - Apply percentile filtering: keep 15th-85th percentile range
   - Remove specular highlights and deep shadows
   - Handle edge effects by excluding boundary pixels

3. Transparency handling (ink-paper mixing model):
   - For each pixel: observed_color = alpha * ink_color + (1-alpha) * paper_color
   - Estimate alpha (opacity) from color distance to paper model
   - Solve for pure ink color: ink_color = (observed - (1-alpha)*paper) / alpha
   - Weight pixels by alpha value (higher opacity = more reliable)

4. Statistical aggregation:
   - Use median for L* (lightness) to handle specular effects
   - Use weighted mean for a* and b* (chromaticity)
   - Weight by distance from swatch boundary (center pixels more reliable)
   - Apply bootstrap sampling for confidence estimation

5. Confidence scoring:
   - High confidence: large swatch, low color variance, clear boundaries
   - Medium confidence: adequate size but some variance or edge issues
   - Low confidence: small swatch, high noise, or poor isolation

6. Validation:
   - Check final color is within reasonable ink gamut bounds
   - Verify color is sufficiently different from paper (min ΔE = 15)
   - Flag unrealistic colors (e.g., impossible chroma values)

EDGE CASES:
- Gradient swatches: use robust statistics to find representative color
- Very light inks: careful transparency handling to avoid paper bias
- Shimmer/sheen: clip extreme values and focus on main color
- Small swatches: increase confidence requirements, warn if insufficient

DECISION POINTS:
- User approved 15-85 percentile range for outlier handling
- Median for lightness chosen to handle fountain pen ink sheen effects
- Bootstrap confidence estimation chosen for statistical rigor
- 10-80% area range chosen based on practical fountain pen photography
```

### CIEDE2000 Color Difference

**Tag**: `algo-ciede2000-delta-e`
**Created**: 2025-01-19
**Last Modified**: 2025-01-19
**Purpose**: Compute perceptually uniform color differences for ink differentiation
**Performance Requirements**: <0.1ms per comparison, batch operations supported
**Commit**: [TBD - initial implementation]

```
ALGORITHM: CIEDE2000 Color Difference (ΔE2000)
INPUT: Two Lab colors (Lab1, Lab2)
OUTPUT: ΔE2000 difference value (0.0 = identical, >3.0 = easily distinguishable)

IMPLEMENTATION:
- Use empfindung crate for CIEDE2000 implementation
- Provides most perceptually uniform color difference metric
- Handles edge cases (zero chroma, hue discontinuity) properly

STEPS (for reference - implemented in empfindung):
1. Compute chroma and hue for both colors
2. Calculate mean chroma and adjust a* values  
3. Compute ΔL*, Δa*, Δb* differences
4. Calculate rotation function RT for hue interaction
5. Apply weighting functions SL, SC, SH
6. Combine components with rotation term
7. Return final ΔE2000 value

PERFORMANCE OPTIMIZATIONS:
- Cache intermediate calculations for batch operations
- Use lookup tables for trigonometric functions
- SIMD operations for vector calculations when available

INTERPRETATION:
- ΔE < 1.0: Not perceptible under ideal conditions
- ΔE < 2.0: Perceptible through close examination  
- ΔE < 3.0: Perceptible at a glance (target for fountain pen differentiation)
- ΔE < 6.0: Very perceivable
- ΔE > 6.0: Different colors

DECISION POINTS:
- User approved ΔE < 3.0 as target for fountain pen ink differentiation
- CIEDE2000 chosen over ΔE94 or ΔE76 for better perceptual uniformity
- empfindung crate chosen for reliable, tested implementation
```

## Deprecated Algorithms

[None yet - initial implementation]

## Guidelines

1. **Storage**: Each algorithm must be stored in knowledge graph with exact tag
2. **Detail Level**: Include all implementation details, not summaries  
3. **Updates**: When modifying, update both this file and knowledge graph
4. **Deprecation**: Never delete - move to deprecated section with reason
5. **Git Reference**: Always include commit hash for traceability
6. **Decisions**: Document all user decisions that affected the algorithm