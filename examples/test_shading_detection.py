#!/usr/bin/env python3
"""
Shading Detection Experiment
Tests different approaches for detecting ink shading properties.

Experimental Design:
- Two-Tone Detection: 3 detection methods × 3 extraction methods = 9 combinations
- Single-Tone Baseline: 4 methods from ExtractionMethod enum
- Total: 13 methods per sample

Detection Methods:
1. K-means clustering (k=2)
2. Histogram bimodality (L* histogram peaks)
3. Spatial gradient analysis (edge-based region detection)

Extraction Methods (applied to each detected region):
1. Gaussian blur → center pixel
2. Average of pixels in region
3. Median of pixels in region

Baseline Methods (single-tone, no shading detection):
1. MedianMean: Median L*, Mean a*/b* on 15-85 percentile
2. Darkest: 10-25 percentile L* (concentrated ink)
3. MostSaturated: Highest chroma pixels (true color)
4. Mode: Most frequent color (histogram binning)
"""
import sys
import os
from pathlib import Path
import numpy as np
import cv2
from collections import defaultdict

def load_swatch_fragment(debug_dir, sample_name):
    """Load swatch fragment and mask."""
    swatch_path = Path(debug_dir) / f"{sample_name}_swatch.png"
    mask_path = Path(debug_dir) / f"{sample_name}_mask.png"

    if not swatch_path.exists() or not mask_path.exists():
        return None, None

    swatch = cv2.imread(str(swatch_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    return swatch, mask

def extract_lab_pixels(swatch, mask, min_lightness=5.0):
    """Extract Lab pixels from masked swatch region.

    Args:
        swatch: BGR image
        mask: Binary mask
        min_lightness: Minimum L* value to include (default 5.0 filters out #000000 background fill)

    Note:
        Filters out background fill pixels (L* < min_lightness) that appear in rectangular
        swatch images where the actual ink swatch has irregular edges. These #000000 pixels
        are not ink and must be excluded from all color analysis.
    """
    # Ensure swatch and mask have same dimensions
    if swatch.shape[:2] != mask.shape[:2]:
        # Resize mask to match swatch dimensions
        mask = cv2.resize(mask, (swatch.shape[1], swatch.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Convert to Lab
    lab = cv2.cvtColor(swatch, cv2.COLOR_BGR2Lab)

    # Get pixels where mask is non-zero
    pixels = []
    for y in range(min(mask.shape[0], lab.shape[0])):
        for x in range(min(mask.shape[1], lab.shape[1])):
            if mask[y, x] > 0:
                l, a, b = lab[y, x]
                # Convert OpenCV Lab to standard Lab
                l_norm = (l / 255.0) * 100.0
                a_norm = a - 128.0
                b_norm = b - 128.0
                # Filter out background fill (L* ≈ 0)
                if l_norm >= min_lightness:
                    pixels.append([l_norm, a_norm, b_norm])

    return np.array(pixels)

def kmeans_clustering(pixels, k=2, max_iter=100):
    """Simple K-means clustering to detect tones."""
    if len(pixels) < k:
        return None, None

    # Initialize centers: pick darkest and lightest L* values
    l_values = pixels[:, 0]
    dark_idx = np.argmin(l_values)
    light_idx = np.argmax(l_values)
    centers = np.array([pixels[dark_idx], pixels[light_idx]])

    for _ in range(max_iter):
        # Assign labels based on distance to centers
        distances = np.zeros((len(pixels), k))
        for i in range(k):
            distances[:, i] = np.linalg.norm(pixels - centers[i], axis=1)

        labels = np.argmin(distances, axis=1)

        # Update centers
        new_centers = np.zeros((k, 3))
        for i in range(k):
            cluster_pixels = pixels[labels == i]
            if len(cluster_pixels) > 0:
                new_centers[i] = cluster_pixels.mean(axis=0)
            else:
                new_centers[i] = centers[i]

        # Check convergence
        if np.allclose(centers, new_centers):
            break

        centers = new_centers

    # Calculate percentage for each cluster
    unique, counts = np.unique(labels, return_counts=True)
    percentages = dict(zip(unique, (counts / len(labels)) * 100))

    # Sort by L* value (darker first, then lighter)
    sorted_indices = np.argsort(centers[:, 0])

    return centers[sorted_indices], [percentages[i] for i in sorted_indices]

def simple_find_peaks(arr, min_height):
    """Simple peak detection without scipy."""
    peaks = []
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i-1] and arr[i] > arr[i+1] and arr[i] >= min_height:
            peaks.append(i)
    return np.array(peaks)

def histogram_bimodality(pixels):
    """Detect bimodality in L* histogram.

    Always returns at least one tone. If bimodality detected, returns two tones
    with percentages that sum to 100%.
    """
    if len(pixels) == 0:
        return None, None

    l_values = pixels[:, 0]
    hist, bins = np.histogram(l_values, bins=20)

    # Find peaks with simple peak detection
    peaks = simple_find_peaks(hist, min_height=len(pixels) * 0.05)

    if len(peaks) >= 2:
        # Get two highest peaks
        peak_heights = hist[peaks]
        top_two_indices = np.argsort(peak_heights)[-2:]
        top_two = peaks[top_two_indices]

        # Sort peaks by L* position (darker first)
        if bins[top_two[0]] > bins[top_two[1]]:
            top_two = top_two[::-1]

        # Get pixels in each peak region - expand to include surrounding bins
        # This ensures percentages add to 100%
        midpoint_l = (bins[top_two[0]] + bins[top_two[1]]) / 2

        tone1_mask = l_values < midpoint_l
        tone2_mask = l_values >= midpoint_l

        if tone1_mask.sum() > 0 and tone2_mask.sum() > 0:
            tone1_center = pixels[tone1_mask].mean(axis=0)
            tone2_center = pixels[tone2_mask].mean(axis=0)

            pct1 = (tone1_mask.sum() / len(pixels)) * 100
            pct2 = (tone2_mask.sum() / len(pixels)) * 100

            return np.array([tone1_center, tone2_center]), [pct1, pct2]

    # No bimodality detected - return single tone (average of all pixels)
    single_tone = pixels.mean(axis=0)
    return np.array([single_tone]), [100.0]

def gradient_detection(pixels):
    """Detect tones using spatial gradient analysis."""
    # This is a simplified version - real implementation would use spatial coordinates
    # For now, treat similar to histogram bimodality but with gradient-weighted binning
    l_values = pixels[:, 0]

    # Use gradient in L* space to find boundaries
    hist, bins = np.histogram(l_values, bins=20)

    # Find largest gap in histogram (indicates two separate regions)
    gaps = []
    for i in range(len(hist) - 1):
        if hist[i] > len(pixels) * 0.05 and hist[i+1] < len(pixels) * 0.02:
            gaps.append((i, hist[i] - hist[i+1]))

    if not gaps:
        return None, None

    # Use largest gap as boundary
    boundary_idx = max(gaps, key=lambda x: x[1])[0]
    boundary_l = bins[boundary_idx]

    # Split pixels by boundary
    tone1_mask = l_values <= boundary_l
    tone2_mask = l_values > boundary_l

    if tone1_mask.sum() > 0 and tone2_mask.sum() > 0:
        tone1_center = pixels[tone1_mask].mean(axis=0)
        tone2_center = pixels[tone2_mask].mean(axis=0)

        pct1 = (tone1_mask.sum() / len(pixels)) * 100
        pct2 = (tone2_mask.sum() / len(pixels)) * 100

        # Sort by L*
        if tone1_center[0] < tone2_center[0]:
            return np.array([tone1_center, tone2_center]), [pct1, pct2]
        else:
            return np.array([tone2_center, tone1_center]), [pct2, pct1]

    return None, None

# Extraction methods for applying to detected regions
def extract_gaussian_blur(swatch, mask, region_mask):
    """Apply Gaussian blur and extract center pixel color."""
    # Ensure dimensions match
    if swatch.shape[:2] != region_mask.shape[:2]:
        region_mask = cv2.resize(region_mask, (swatch.shape[1], swatch.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply region mask to swatch
    masked_swatch = swatch.copy()
    masked_swatch[region_mask == 0] = 0

    # Apply strong Gaussian blur
    blurred = cv2.GaussianBlur(masked_swatch, (51, 51), 0)
    lab_blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2Lab)

    # Find center of mass of region
    M = cv2.moments(region_mask.astype(np.uint8))
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Ensure coordinates are within bounds
    cy = min(cy, lab_blurred.shape[0] - 1)
    cx = min(cx, lab_blurred.shape[1] - 1)

    # Get color at center
    l, a, b = lab_blurred[cy, cx]
    l_norm = (l / 255.0) * 100.0
    a_norm = a - 128.0
    b_norm = b - 128.0

    return np.array([l_norm, a_norm, b_norm])

def extract_average(pixels):
    """Average of all pixels in region."""
    if len(pixels) == 0:
        return None
    return pixels.mean(axis=0)

def extract_median(pixels):
    """Median of all pixels in region."""
    if len(pixels) == 0:
        return None
    return np.median(pixels, axis=0)

# Baseline single-tone methods (from ExtractionMethod enum)
def baseline_median_mean(pixels):
    """MedianMean: Median L*, Mean a*/b* on 15-85 percentile."""
    if len(pixels) == 0:
        return None

    # Filter to 15-85 percentile based on L*
    l_values = pixels[:, 0]
    p15 = np.percentile(l_values, 15)
    p85 = np.percentile(l_values, 85)

    filtered = pixels[(l_values >= p15) & (l_values <= p85)]
    if len(filtered) == 0:
        return None

    l_median = np.median(filtered[:, 0])
    a_mean = filtered[:, 1].mean()
    b_mean = filtered[:, 2].mean()

    return np.array([l_median, a_mean, b_mean])

def baseline_darkest(pixels):
    """Darkest: 10-25 percentile L* (concentrated/wet ink)."""
    if len(pixels) == 0:
        return None

    l_values = pixels[:, 0]
    p10 = np.percentile(l_values, 10)
    p25 = np.percentile(l_values, 25)

    darkest = pixels[(l_values >= p10) & (l_values <= p25)]
    if len(darkest) == 0:
        return None

    return darkest.mean(axis=0)

def baseline_most_saturated(pixels):
    """MostSaturated: Highest chroma pixels (true ink color)."""
    if len(pixels) == 0:
        return None

    # Calculate chroma
    chroma = np.sqrt(pixels[:, 1]**2 + pixels[:, 2]**2)
    p75 = np.percentile(chroma, 75)

    saturated = pixels[chroma >= p75]
    if len(saturated) == 0:
        return None

    return saturated.mean(axis=0)

def baseline_mode(pixels):
    """Mode: Most frequent color (histogram binning)."""
    if len(pixels) == 0:
        return None

    # Bin L* values into 20 bins
    l_values = pixels[:, 0]
    hist, bins = np.histogram(l_values, bins=20)

    # Find most frequent bin
    mode_idx = np.argmax(hist)

    # Get pixels in mode bin
    bin_low = bins[mode_idx]
    bin_high = bins[mode_idx + 1]
    mode_pixels = pixels[(l_values >= bin_low) & (l_values < bin_high)]

    if len(mode_pixels) == 0:
        return None

    return mode_pixels.mean(axis=0)

def lab_to_hex(lab):
    """Convert Lab to hex color string."""
    l, a, b = lab

    # Lab to XYZ (D65)
    fy = (l + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    epsilon = 0.008856
    kappa = 903.3

    xr = fx**3 if fx**3 > epsilon else (116.0 * fx - 16.0) / kappa
    yr = fy**3 if l > kappa * epsilon else l / kappa
    zr = fz**3 if fz**3 > epsilon else (116.0 * fz - 16.0) / kappa

    # D65 white point
    x = xr * 0.95047
    y = yr * 1.00000
    z = zr * 1.08883

    # XYZ to sRGB
    r_linear = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    g_linear = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    b_linear = x * 0.0556434 + y * -0.2040259 + z * 1.0572252

    # Gamma correction
    def srgb_gamma(linear):
        if linear <= 0.0031308:
            return 12.92 * linear
        else:
            return 1.055 * (linear ** (1.0 / 2.4)) - 0.055

    r = srgb_gamma(r_linear)
    g = srgb_gamma(g_linear)
    b = srgb_gamma(b_linear)

    # Clamp and convert to 0-255
    r = int(max(0, min(1, r)) * 255)
    g = int(max(0, min(1, g)) * 255)
    b = int(max(0, min(1, b)) * 255)

    return f"#{r:02x}{g:02x}{b:02x}"

def lab_to_color_name_and_base(lab):
    """Convert Lab to ISCC-NBS color name and Base color name using munsellspace crate.

    Returns: (color_name, base_color)
    For N/A results, includes hex: ("N/A (#aabbcc)", "N")
    """
    import subprocess

    l, a, b = lab

    try:
        result = subprocess.run(
            ['cargo', 'run', '--release', '--example', 'lab_to_color_name', '--',
             str(l), str(a), str(b)],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            if '|' in output:
                name, base = output.split('|', 1)
                # If name is N/A, add hex value
                if name == "N/A":
                    hex_val = lab_to_hex(lab)
                    return (f"N/A ({hex_val})", base)
                return (name, base)
            else:
                hex_val = lab_to_hex(lab)
                return (f"{output} ({hex_val})", "N")
        else:
            hex_val = lab_to_hex(lab)
            return (f"L*{l:.0f} ({hex_val})", "N")  # Fallback
    except Exception:
        hex_val = lab_to_hex(lab)
        return (f"L*{l:.0f} ({hex_val})", "N")  # Fallback

def check_same_base_color(base1, base2):
    """Check if two colors have the same base color.

    Args:
        base1: Base color name (e.g., "red", "blue", "teal", "N")
        base2: Base color name

    Returns:
        True if both have the same base color
    """
    # Neutral colors ("N") and "N/A" don't have shading
    if base1 in ("N", "N/A") or base2 in ("N", "N/A"):
        return False

    return base1 == base2

def process_sample(debug_dir, sample_name):
    """Process a single sample with all 10 methods (2 detection × 3 extraction + 4 baseline)."""
    swatch, mask = load_swatch_fragment(debug_dir, sample_name)

    if swatch is None or mask is None:
        return None

    pixels = extract_lab_pixels(swatch, mask)

    if len(pixels) == 0:
        return None

    results = []

    # === Two-Tone Detection (6 combinations: 2 detection × 3 extraction) ===

    detection_methods = {
        'K-means': kmeans_clustering,
        'Histogram': histogram_bimodality,
    }

    for detect_name, detect_func in detection_methods.items():
        # Run detection
        centers, percentages = detect_func(pixels)

        if centers is None:
            # Detection failed - should not happen anymore
            for extract_name in ['Gaussian', 'Average', 'Median']:
                results.append({
                    'detection': detect_name,
                    'extraction': extract_name,
                    'tone1': 'N/A',
                    'tone1_base': 'N',
                    'tone1_pct': 0.0,
                    'tone2': 'N/A',
                    'tone2_base': 'N',
                    'tone2_pct': 0.0,
                    'delta_l': 0.0,
                    'rel_delta_l': 0.0,
                    'same_base': False,
                    'shading': False
                })
            continue

        # Check if single-tone or two-tone detection
        if len(centers) == 1:
            # Single tone detected (no shading) - treat as baseline
            single_color = centers[0]
            color_name, color_base = lab_to_color_name_and_base(single_color)

            for extract_name in ['Gaussian', 'Average', 'Median']:
                results.append({
                    'detection': detect_name,
                    'extraction': extract_name,
                    'tone1': color_name,
                    'tone1_base': color_base,
                    'tone1_pct': 100.0,
                    'tone2': '-',
                    'tone2_base': '-',
                    'tone2_pct': 0.0,
                    'delta_l': 0.0,
                    'rel_delta_l': 0.0,
                    'same_base': False,
                    'shading': False
                })
            continue

        # Two-tone detected - proceed with extraction
        # Create region masks for extraction
        l_values = pixels[:, 0]
        midpoint_l = (centers[0][0] + centers[1][0]) / 2
        region1_pixels = pixels[l_values <= midpoint_l]
        region2_pixels = pixels[l_values > midpoint_l]

        # Apply each extraction method
        extraction_methods = [
            ('Gaussian', None),  # Use detection centers
            ('Average', extract_average),
            ('Median', extract_median),
        ]

        for extract_name, extract_func in extraction_methods:
            if extract_name == 'Gaussian':
                # Use centers from detection
                tone1_color = centers[0]
                tone2_color = centers[1]
            else:
                # Apply extraction to regions
                tone1_color = extract_func(region1_pixels) if len(region1_pixels) > 0 else centers[0]
                tone2_color = extract_func(region2_pixels) if len(region2_pixels) > 0 else centers[1]

            # Get color names and base colors
            tone1_name, tone1_base = lab_to_color_name_and_base(tone1_color)
            tone2_name, tone2_base = lab_to_color_name_and_base(tone2_color)

            # Check if same base color
            same_base = check_same_base_color(tone1_base, tone2_base)

            # Only calculate metrics if same base color (otherwise meaningless)
            if same_base:
                delta_l = abs(tone2_color[0] - tone1_color[0])
                avg_l = (tone1_color[0] + tone2_color[0]) / 2
                rel_delta_l = (delta_l / avg_l * 100) if avg_l > 0 else 0
                # Shading requires: same base + significant ΔL*
                shading = delta_l > 10
            else:
                delta_l = 0.0
                rel_delta_l = 0.0
                shading = False

            results.append({
                'detection': detect_name,
                'extraction': extract_name,
                'tone1': tone1_name,
                'tone1_base': tone1_base,
                'tone1_pct': percentages[0],
                'tone2': tone2_name,
                'tone2_base': tone2_base,
                'tone2_pct': percentages[1],
                'delta_l': delta_l,
                'rel_delta_l': rel_delta_l,
                'same_base': same_base,
                'shading': shading
            })

    # === Baseline Single-Tone Methods (4 methods) ===

    baseline_methods = [
        ('MedianMean', baseline_median_mean),
        ('Darkest', baseline_darkest),
        ('MostSaturated', baseline_most_saturated),
        ('Mode', baseline_mode),
    ]

    for method_name, method_func in baseline_methods:
        color = method_func(pixels)
        if color is not None:
            color_name, color_family = lab_to_color_name_and_base(color)
            results.append({
                'detection': method_name,
                'extraction': 'N/A',
                'tone1': color_name,
                'tone1_base': color_family,
                'tone1_pct': 100.0,
                'tone2': '-',
                'tone2_base': '-',
                'tone2_pct': 0.0,
                'delta_l': 0.0,
                'rel_delta_l': 0.0,
                'same_base': False,
                'shading': False
            })

    return results

def main():
    if len(sys.argv) < 3:
        print("Usage: test_shading_detection.py <debug_dir> <output_md>")
        sys.exit(1)

    debug_dir = sys.argv[1]
    output_file = sys.argv[2]

    # Get all samples (exclude flash)
    samples = []
    for f in Path(debug_dir).glob("*_swatch.png"):
        sample_name = f.stem.replace("_swatch", "")
        if "flash" not in sample_name.lower():
            samples.append(sample_name)

    samples.sort()

    # Process all samples
    all_results = {}
    for sample in samples:
        print(f"Processing {sample}...", file=sys.stderr)
        results = process_sample(debug_dir, sample)
        if results:
            all_results[sample] = results

    # Generate markdown report
    with open(output_file, 'w') as f:
        f.write("# Shading Detection Experiment\n\n")

        f.write("## Objective\n\n")
        f.write("Test different approaches for detecting ink shading properties by identifying distinct tones within a swatch.\n\n")

        f.write("## Experimental Design\n\n")
        f.write("**Two-Tone Detection Methods (6 combinations):**\n")
        f.write("- **Detection**: K-means, Histogram\n")
        f.write("- **Extraction**: Gaussian blur, Average, Median\n\n")

        f.write("**Baseline Single-Tone Methods (4 methods):**\n")
        f.write("1. **MedianMean**: Median L*, Mean a*/b* on 15-85 percentile\n")
        f.write("2. **Darkest**: 10-25 percentile L* (concentrated/wet ink)\n")
        f.write("3. **MostSaturated**: Highest chroma pixels (true ink color)\n")
        f.write("4. **Mode**: Most frequent color (histogram binning)\n\n")

        f.write("**Shading Criteria:**\n")
        f.write("- Both tones must have the same base color (e.g., both \"blue\", both \"red\", both \"teal\")\n")
        f.write("- ΔL* > 10 between tones (significant luminance difference)\n\n")

        f.write("**Metrics:**\n")
        f.write("- ΔL*: Absolute luminance difference\n")
        f.write("- Rel ΔL*: Relative luminance difference as % of average L*\n")
        f.write("- Base Color: ISCC-NBS base color name from munsellspace\n\n")

        f.write("## Results\n\n")

        # Create detailed table
        f.write("| Sample | Detection | Extraction | Tone 1 (Dark) | Base | % | Tone 2 (Light) | Base | % | ΔL* | Rel ΔL* | Same Base | Shading |\n")
        f.write("|--------|-----------|------------|---------------|------|---|----------------|------|---|-----|---------|-----------|----------|\n")

        for sample in samples:
            if sample not in all_results:
                continue

            results = all_results[sample]

            # Write all 10 rows for this sample
            for r in results:
                detect = r['detection']
                extract = r['extraction']
                tone1 = r['tone1']
                tone1_base = r['tone1_base']
                tone1_pct = r['tone1_pct']
                tone2 = r['tone2']
                tone2_base = r['tone2_base']
                tone2_pct = r['tone2_pct']
                delta_l = r['delta_l']
                rel_delta_l = r['rel_delta_l']
                same_base = r['same_base']
                shading = r['shading']

                # Format values
                pct1_str = f"{tone1_pct:.1f}" if tone1_pct > 0 else "-"
                pct2_str = f"{tone2_pct:.1f}" if tone2_pct > 0 else "-"
                delta_str = f"{delta_l:.1f}" if delta_l > 0 else "-"
                rel_delta_str = f"{rel_delta_l:.1f}%" if rel_delta_l > 0 else "-"
                family_str = "✓" if same_base and tone2 != "-" else ("✗" if tone2 != "-" else "-")
                shading_str = "**YES**" if shading else "no"

                f.write(f"| {sample} | {detect} | {extract} | {tone1} | {tone1_base} | {pct1_str} | {tone2} | {tone2_base} | {pct2_str} | {delta_str} | {rel_delta_str} | {family_str} | {shading_str} |\n")

            # Add spacing between samples
            f.write("|--------|-----------|------------|---------------|--------|---|----------------|--------|---|-----|---------|-------------|----------|\n")

        f.write("\n## Summary Statistics\n\n")

        # Count shading detections per detection method
        detection_shading = defaultdict(int)
        extraction_shading = defaultdict(int)
        baseline_counts = defaultdict(int)

        for sample, results in all_results.items():
            for r in results:
                if r['extraction'] == 'N/A':
                    # Baseline method
                    baseline_counts[r['detection']] += 1
                elif r['shading']:
                    # Two-tone shading detected
                    detection_shading[r['detection']] += 1
                    extraction_shading[r['extraction']] += 1

        f.write(f"**Samples analyzed:** {len(all_results)} (excluding flash samples)\n\n")

        f.write("**Two-Tone Shading Detection:**\n")
        f.write(f"- K-means: {detection_shading['K-means']} detections (across all 3 extraction methods)\n")
        f.write(f"- Histogram: {detection_shading['Histogram']} detections (across all 3 extraction methods)\n\n")

        f.write("**By Extraction Method:**\n")
        f.write(f"- Gaussian: {extraction_shading['Gaussian']} detections (across both detection methods)\n")
        f.write(f"- Average: {extraction_shading['Average']} detections (across both detection methods)\n")
        f.write(f"- Median: {extraction_shading['Median']} detections (across both detection methods)\n\n")

        f.write("**Baseline Methods:**\n")
        f.write(f"- All baseline methods process {len(all_results)} samples (single-tone only)\n\n")

        f.write("## Observations\n\n")
        f.write("*To be filled after manual review of results*\n\n")

        f.write("## Next Steps\n\n")
        f.write("Based on these results, determine:\n")
        f.write("1. Which detection method works best?\n")
        f.write("2. Which extraction method works best?\n")
        f.write("3. Does the best combination vary by ink type?\n")
        f.write("4. How do two-tone methods compare to baseline single-tone methods?\n")
        f.write("5. Should we adjust ΔL* threshold?\n")
        f.write("6. Should we adjust hue angle threshold for \"same family\"?\n")

    print(f"\nReport generated: {output_file}")

if __name__ == '__main__':
    main()
