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

### Adaptive Paper Detection with Foreign Object Exclusion

**Tag**: `algo-adaptive-paper-detection`
**Created**: 2025-01-19
**Last Modified**: 2025-11-16
**Purpose**: Detect and rectify paper/card surface in image, excluding foreign objects (tape, rulers, weights)
**Performance Requirements**: <30ms processing time, handle 4000x3000 pixel images
**Commit**: [TBD - initial implementation]

```
ALGORITHM: Adaptive Paper Detection with Foreign Object Exclusion
INPUT: RGB image from smartphone camera
OUTPUT: Rectified image, paper contour coordinates, homography matrix, foreign object mask

STEPS:
1. Preprocessing:
   - Convert RGB to Lab color space for better lightness analysis
   - Use L* channel for brightness-based detection
   - Apply Gaussian blur (sigma=1.0) to reduce noise

2. Adaptive thresholding:
   - Use Otsu's method for initial threshold estimation
   - Apply local adaptive threshold for uneven lighting
   - Target: detect bright regions (paper) vs darker background

3. Foreign object detection and masking:
   - Detect high-contrast edges using Canny edge detection
   - Identify non-paper objects (rulers, tape, weights) by:
     - Rectangular contours with high edge density (rulers)
     - Glossy/reflective regions (transparent tape, glossy objects)
     - Strong straight lines not aligned with paper edges
   - Create binary mask excluding foreign objects from analysis
   - Assumption: ink swatch itself is not obstructed by transparent materials

4. Morphological operations:
   - Opening operation (kernel size 3x3) to remove noise
   - Closing operation to fill small gaps in paper regions
   - Exclude masked foreign object regions

5. Contour detection and filtering:
   - Find all contours in thresholded image (excluding foreign objects)
   - Filter by area: minimum 10% of total image area
   - Filter by aspect ratio: approximately rectangular shapes

6. Polygon approximation:
   - Use approxPolyDP with epsilon = 2% of perimeter
   - Select contours with approximately 4 corners
   - Validate corners form near-rectangular shape (angles ~90°)

7. Homography computation and rectification:
   - Order corners (top-left, top-right, bottom-right, bottom-left)
   - Compute homography to standard rectangle
   - Apply perspective transform to rectify image
   - Apply foreign object mask to rectified image
   - Limit maximum rectification angle to 45° to avoid distortion

EDGE CASES:
- Multiple paper regions: select largest rectangular contour
- Rounded corners: allow polygon approximation tolerance
- Poor lighting: use multiple threshold values and combine results
- No clear paper boundary: fallback to whole-image analysis
- Foreign objects at edges: exclude from paper boundary detection
- Complex holding mechanisms: detect and mask opaque objects

PERFORMANCE OPTIMIZATIONS:
- Downscale image for detection, upscale coordinates for rectification
- Use integral images for efficient region analysis
- Cache morphological kernels
- Parallel edge detection for foreign objects

DECISION POINTS:
- User approved 10% minimum area to ensure adequate swatch region
- 45° maximum rectification angle chosen to preserve image quality
- Otsu + adaptive combination chosen over single threshold method
- User requested foreign object exclusion (2025-11-16)
- Assumption: ink area not obstructed by transparent tape (user guidance)
```

### Ink Swatch Boundary Detection

**Tag**: `algo-swatch-boundary-detection`
**Created**: 2025-11-16
**Last Modified**: 2025-11-16
**Purpose**: Isolate ink swatch region from paper background, excluding foreign objects
**Performance Requirements**: <20ms processing time, handle 10-80% swatch area range
**Commit**: [TBD - initial implementation]

```
ALGORITHM: Ink Swatch Boundary Detection with Foreign Object Exclusion
INPUT: Rectified paper image, foreign object mask, estimated paper color
OUTPUT: Ink swatch binary mask, swatch boundary contour, confidence score

STEPS:
1. Color-based segmentation:
   - Convert rectified image to Lab color space
   - Compute color difference (ΔE) between each pixel and estimated paper color
   - Threshold by minimum ΔE (default: 15) to separate ink from paper
   - Apply foreign object mask to exclude non-ink, non-paper regions

2. Morphological refinement:
   - Opening operation to remove small noise regions
   - Closing operation to fill holes within swatch
   - Ensure foreign objects (tape, rulers) are excluded from swatch mask

3. Contour analysis:
   - Find contours in binary swatch mask
   - Filter by size: minimum 10%, maximum 80% of paper area
   - Select largest contour as primary swatch region
   - Validate swatch is reasonably compact (not highly fragmented)

4. Boundary refinement:
   - Apply gradient-based edge refinement for precise boundaries
   - Smooth contour to reduce noise while preserving shape
   - Erode boundary slightly (1-2 pixels) to avoid edge artifacts

5. Confidence scoring:
   - High confidence: clear color separation, compact swatch, adequate size
   - Medium confidence: adequate separation but irregular shape or size issues
   - Low confidence: poor separation, very small/large swatch, or fragmented

EDGE CASES:
- Gradient swatches: use color range instead of single threshold
- Very light inks: lower ΔE threshold, increase sensitivity
- Multiple ink regions: select largest connected component
- Foreign objects overlapping edges: exclude from boundary detection
- Paper tone variations: use local paper color estimation

PERFORMANCE OPTIMIZATIONS:
- Use integral images for efficient color distance computation
- Cache Lab conversions for repeated pixel access
- Parallel processing for independent regions

DECISION POINTS:
- User requested foreign object exclusion (2025-11-16)
- Minimum ΔE of 15 chosen for clear ink-paper separation
- 10-80% area range based on practical fountain pen photography
- Assumption: ink swatch itself not obstructed by transparent materials
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

### Swatch-First Detection with Adaptive White Point Estimation

**Tag**: `algo-swatch-first-adaptive-wb`
**Created**: 2025-11-25
**Last Modified**: 2025-11-25
**Purpose**: Unified detection strategy that works whether algorithm finds paper card or swatch, with adaptive white point estimation from paper sampling band
**Performance Requirements**: <50ms total processing time, handles 15-30% swatch area range
**Commit**: [TBD - implementation pending]

```
ALGORITHM: Swatch-First Detection with Adaptive White Point Estimation
INPUT: RGB image from smartphone camera (after EXIF orientation correction)
OUTPUT: White-balance corrected full image, swatch coordinates, swatch mask with transparency, paper band color characterization (optional)

RATIONALE:
Current paper detection algorithm reliably finds the strongest rectangular contour in the image.
For images with large swatches (15-30% of frame), this is often the swatch boundary (high-contrast
dark ink vs light paper) rather than paper card boundary (low-contrast white card vs background).
Instead of trying to fix this, we embrace it: use the detected rectangle (paper OR swatch) as
input to adaptive white point estimation from the surrounding paper region.

STEPS:
1. Initial rectangular detection (agnostic to classification):
   - Use existing Canny edge detection + contour finding algorithm
   - Apply area filtering (≥5%), aspect ratio filtering (0.33-3.0)
   - Score by 70% area + 30% centrality
   - Result: coordinates of strongest rectangle (could be paper card or swatch)
   - No need to classify whether this is "paper" or "swatch"

2. Define paper sampling band:
   - Calculate midpoint between rectangle borders and image borders:
     * band_left = (rect_left + 0) / 2
     * band_right = (rect_right + image_width) / 2
     * band_top = (rect_top + 0) / 2
     * band_bottom = (rect_bottom + image_height) / 2
   - Define band width as percentage of distance (e.g., 20% on each side)
   - This creates 4 rectangular sampling regions around the detected rectangle
   - Sampling band captures paper region whether rect is full card or just swatch

3. Adaptive white point estimation:
   - Extract all pixels within paper sampling band regions
   - Convert to Lab color space
   - Filter overexposed pixels:
     * Remove pixels with RGB values (255, 255, 255) or very close (L* > 98)
     * These are specular reflections, not paper surface
   - Filter shadowed pixels:
     * Remove pixels with L* < 40 (deep shadows not representative of paper)
   - Compute robust statistics on remaining pixels:
     * Median L*, a*, b* values (robust to outliers)
     * Check color variance to assess confidence
   - White point = median Lab of paper band
   - Confidence based on:
     * Number of valid pixels (more = higher confidence)
     * Color variance (lower = higher confidence)
     * Neutrality check (paper should have low chroma)

4. White balance correction:
   - Calculate offset from target paper color:
     * target = (L*=95, a*=0, b*=0) [neutral white]
     * offset_L = target_L - measured_L
     * offset_a = target_a - measured_a
     * offset_b = target_b - measured_b
   - Apply correction to ENTIRE original image:
     * Convert full image to Lab
     * Add offsets: L* += offset_L, a* += offset_a, b* += offset_b
     * Clip to valid ranges
     * Convert back to RGB
   - Result: full-frame corrected image (not cropped)

5. Swatch extraction:
   - Use rectangle coordinates from step 1 (already known)
   - If rectangle is significantly smaller than image (< 50% area):
     * Assume rectangle = swatch boundary, use directly
   - If rectangle is close to full image (≥ 50% area):
     * Assume rectangle = paper card
     * Run color-based swatch detection on corrected image
     * Use ΔE thresholding against corrected paper color
   - Extract swatch region pixels

6. High-luminance masking:
   - Within swatch region, identify high-luminance pixels:
     * L* > 90 in corrected image (likely paper showing through)
     * Or pixels very close to paper color (ΔE < 5)
   - Convert these pixels to alpha channel (transparency)
   - Result: swatch fragment with transparent regions for paper areas

7. Paper band color characterization (future use):
   - Extract narrow band (2-5 pixels) around final swatch boundary
   - Filter high-chroma pixels (C* > 20, likely ink traces)
   - Compute band color statistics:
     * Mean L*, a*, b*
     * Chroma and hue if non-neutral
   - Store for potential correction of non-white card influence
   - NOT applied in current implementation (white cards assumed)

EDGE CASES:
- Very large swatch (> 30% of frame):
  * Paper sampling band may be very narrow
  * Increase band width adaptively
  * Require minimum 500 pixels for reliable estimation
- Very small swatch (< 10% of frame):
  * Rectangle likely detects paper card correctly
  * Use standard swatch detection on corrected image
- Overexposed paper (flash photography):
  * High-luminance filtering removes most pixels
  * Fallback to EXIF-based white balance if < 100 valid pixels
  * Flag low confidence
- Colored backgrounds:
  * Paper sampling band may include background
  * Use centrality weighting (pixels closer to rectangle more reliable)
  * Check for bimodal distribution in band colors
- No clear rectangle detected:
  * Fallback to whole-image white balance estimation
  * Use EXIF data if available
  * Warn user about potential inaccuracy

PERFORMANCE OPTIMIZATIONS:
- Only convert paper sampling band to Lab (not full image initially)
- Use vectorized operations for filtering and statistics
- Cache Lab conversion of full image for reuse
- Parallelize band sampling (4 independent rectangular regions)

DECISION POINTS:
- User proposed swatch-first strategy after observing "bug as feature" (2025-11-25)
- User confirmed swatches are 15-17% of successful images, so area threshold won't solve issue
- User approved assumption that ink swatch not obstructed by transparent materials
- User suggested paper band characterization for future non-white card support
- Midpoint band strategy chosen to work for both paper-detected and swatch-detected cases
- 20% band width chosen as initial value (may need tuning based on experiments)

COMPARISON WITH PREVIOUS APPROACH:
Previous: Detect paper → rectify → estimate WB from rectified paper → fail if detected swatch
New: Detect rectangle → sample surrounding paper → estimate WB → extract swatch using known coords
Advantage: Works whether rectangle is paper or swatch, no cascade failure
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