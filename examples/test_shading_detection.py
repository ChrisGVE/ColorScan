#!/usr/bin/env python3
"""
Shading Detection Experiment
Tests different approaches for detecting ink shading properties.

Generates two result files:
- two-colors-results.md: Shows only first 2 colors with full metrics
- three-colors-results.md: Shows all 3 colors with full metrics

Experimental Design:
- Multi-Tone Detection: 2 detection methods × 4 extraction methods = 8 combinations
- Single-Tone Baseline: 4 global extraction methods
- Detection: K-means (k=3), Histogram (1-3 peaks)
- Total: 12 methods per sample

Detection Methods (can return 1-3 colors):
1. K-means clustering (k=3, may reduce to 2 or 1 after deduplication)
2. Histogram multimodality (L* histogram peaks, 1-3 modes)

Extraction Methods (applied to each detected region):
1. Gaussian blur → average of region (K-means centers)
2. Average of pixels in region
3. Median of pixels in region
4. Mode: Most frequent L* bin in region

Global Methods (no shading detection, single color):
1. Median: Pure median of L*, a*, b* (robust, no filtering)
2. Darkest: 10-25 percentile L* (concentrated ink)
3. MostSaturated: Highest chroma pixels (true color)
4. Mode: Most frequent L* bin (global histogram)

Deduplication:
- If 2+ detected colors have same ISCC-NBS name, merge regions and re-extract
- Ensures each color in results has unique name

Color Ordering:
- Colors sorted by spatial frequency (%) descending
- Color 1 = most frequent, Color 2/3 = less frequent

Shading Criteria:
- All colors must have same base color family (e.g., all "blue")
- All colors must have different ISCC-NBS names (not identical)
- Relative ΔL* > 30% between darkest and lightest tones
"""
import sys
import os
import subprocess
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

def extract_lab_pixels(swatch, mask=None):
    """Extract Lab pixels from swatch, filtering out #000000 padding.

    Args:
        swatch: BGR image (already cropped with #000000 padding around irregular edges)
        mask: Unused (kept for compatibility) - mask was already applied during swatch extraction

    Note:
        The swatch image is a cropped rectangle from the full image, with #000000 BGR pixels
        as padding where the irregular ink swatch edges don't fill the rectangle.
        We filter out pure #000000 BGR to exclude this padding while keeping all ink pixels,
        even very dark inks (which are not pure #000000).
    """
    # Convert to Lab
    lab = cv2.cvtColor(swatch, cv2.COLOR_BGR2Lab)

    # Extract pixels, filtering out #000000 BGR padding
    pixels = []
    for y in range(swatch.shape[0]):
        for x in range(swatch.shape[1]):
            b, g, r = swatch[y, x]
            # Filter out pure black padding (BGR #000000)
            # Real ink pixels, even very dark ones, are not pure #000000
            if not (b == 0 and g == 0 and r == 0):
                l, a_val, b_val = lab[y, x]
                # Convert OpenCV Lab to standard Lab
                l_norm = (l / 255.0) * 100.0
                a_norm = a_val - 128.0
                b_norm = b_val - 128.0
                pixels.append([l_norm, a_norm, b_norm])

    return np.array(pixels)

def kmeans_clustering(pixels, k=3, max_iter=100):
    """K-means clustering to detect 1-3 tones.

    Returns:
        centers: Array of cluster centers (1-3 colors)
        percentages: List of percentages for each cluster
        labels: Cluster assignment for each pixel
    """
    if len(pixels) < k:
        return None, None, None

    # Initialize centers across L* range (10th, 50th, 90th percentile)
    l_values = pixels[:, 0]
    if k == 3:
        p10_idx = np.argmin(np.abs(l_values - np.percentile(l_values, 10)))
        p50_idx = np.argmin(np.abs(l_values - np.percentile(l_values, 50)))
        p90_idx = np.argmin(np.abs(l_values - np.percentile(l_values, 90)))
        centers = np.array([pixels[p10_idx], pixels[p50_idx], pixels[p90_idx]])
    elif k == 2:
        dark_idx = np.argmin(l_values)
        light_idx = np.argmax(l_values)
        centers = np.array([pixels[dark_idx], pixels[light_idx]])
    else:
        centers = np.array([pixels[np.argmin(l_values)]])

    for _ in range(max_iter):
        # Assign labels based on distance to centers
        distances = np.zeros((len(pixels), len(centers)))
        for i in range(len(centers)):
            distances[:, i] = np.linalg.norm(pixels - centers[i], axis=1)

        labels = np.argmin(distances, axis=1)

        # Update centers
        new_centers = np.zeros((len(centers), 3))
        for i in range(len(centers)):
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

    # Return clusters with their percentages and labels
    # Don't sort yet - will be sorted by frequency later
    return centers, [percentages.get(i, 0.0) for i in range(len(centers))], labels

def simple_find_peaks(arr, min_height):
    """Simple peak detection without scipy."""
    peaks = []
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i-1] and arr[i] > arr[i+1] and arr[i] >= min_height:
            peaks.append(i)
    return np.array(peaks)

def histogram_multimodality(pixels):
    """Detect 1-3 modes in L* histogram.

    Returns:
        centers: Array of mode centers (1-3 colors)
        percentages: List of percentages for each mode
        labels: Mode assignment for each pixel
    """
    if len(pixels) == 0:
        return None, None, None

    l_values = pixels[:, 0]
    hist, bins = np.histogram(l_values, bins=20)

    # Find peaks with simple peak detection
    peaks = simple_find_peaks(hist, min_height=len(pixels) * 0.05)

    if len(peaks) >= 3:
        # Get three highest peaks
        peak_heights = hist[peaks]
        top_three_indices = np.argsort(peak_heights)[-3:]
        top_three = peaks[top_three_indices]

        # Sort peaks by L* position (darker first)
        top_three = top_three[np.argsort(bins[top_three])]

        # Divide pixels into 3 regions using midpoints
        mid1_l = (bins[top_three[0]] + bins[top_three[1]]) / 2
        mid2_l = (bins[top_three[1]] + bins[top_three[2]]) / 2

        labels = np.zeros(len(pixels), dtype=int)
        labels[l_values < mid1_l] = 0
        labels[(l_values >= mid1_l) & (l_values < mid2_l)] = 1
        labels[l_values >= mid2_l] = 2

        centers = []
        percentages = []
        for i in range(3):
            region_pixels = pixels[labels == i]
            if len(region_pixels) > 0:
                centers.append(region_pixels.mean(axis=0))
                percentages.append((len(region_pixels) / len(pixels)) * 100)
            else:
                # Empty region - shouldn't happen but handle gracefully
                centers.append(pixels.mean(axis=0))
                percentages.append(0.0)

        return np.array(centers), percentages, labels

    elif len(peaks) >= 2:
        # Get two highest peaks
        peak_heights = hist[peaks]
        top_two_indices = np.argsort(peak_heights)[-2:]
        top_two = peaks[top_two_indices]

        # Sort peaks by L* position (darker first)
        top_two = top_two[np.argsort(bins[top_two])]

        # Divide pixels into 2 regions
        midpoint_l = (bins[top_two[0]] + bins[top_two[1]]) / 2

        labels = np.zeros(len(pixels), dtype=int)
        labels[l_values < midpoint_l] = 0
        labels[l_values >= midpoint_l] = 1

        tone1_pixels = pixels[labels == 0]
        tone2_pixels = pixels[labels == 1]

        if len(tone1_pixels) > 0 and len(tone2_pixels) > 0:
            centers = np.array([tone1_pixels.mean(axis=0), tone2_pixels.mean(axis=0)])
            pct1 = (len(tone1_pixels) / len(pixels)) * 100
            pct2 = (len(tone2_pixels) / len(pixels)) * 100

            return centers, [pct1, pct2], labels

    # No multimodality detected - return single tone
    single_tone = pixels.mean(axis=0)
    labels = np.zeros(len(pixels), dtype=int)
    return np.array([single_tone]), [100.0], labels

# Extraction methods for applying to detected regions
def extract_gaussian(region_pixels):
    """Gaussian: Average of region pixels (simplified from blur approach)."""
    if len(region_pixels) == 0:
        return None
    return region_pixels.mean(axis=0)

def extract_average(region_pixels):
    """Average of all pixels in region."""
    if len(region_pixels) == 0:
        return None
    return region_pixels.mean(axis=0)

def extract_median(region_pixels):
    """Median of all pixels in region."""
    if len(region_pixels) == 0:
        return None
    return np.median(region_pixels, axis=0)

def extract_mode(region_pixels):
    """Mode: Most frequent L* bin in region."""
    if len(region_pixels) == 0:
        return None

    # Bin L* values into 20 bins
    l_values = region_pixels[:, 0]
    hist, bins = np.histogram(l_values, bins=20)

    # Find most frequent bin
    mode_idx = np.argmax(hist)

    # Get pixels in mode bin
    bin_low = bins[mode_idx]
    bin_high = bins[mode_idx + 1]
    mode_pixels = region_pixels[(l_values >= bin_low) & (l_values < bin_high)]

    if len(mode_pixels) == 0:
        return None

    return mode_pixels.mean(axis=0)

# Global baseline methods (single-tone extraction)
def baseline_median(pixels):
    """Median: Pure median of L*, a*, b* (robust, no filtering)."""
    if len(pixels) == 0:
        return None
    return np.median(pixels, axis=0)

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
    """Mode: Most frequent L* bin (global histogram)."""
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

def deduplicate_colors(centers, percentages, labels, pixels, extract_func):
    """Deduplicate colors with same ISCC-NBS name by merging regions.

    Args:
        centers: Array of detected cluster centers
        percentages: List of percentages for each cluster
        labels: Cluster assignment for each pixel
        pixels: All pixels (for re-extraction)
        extract_func: Extraction function to apply to merged region

    Returns:
        deduplicated_centers: Array of unique colors
        deduplicated_percentages: List of percentages
        deduplicated_labels: Updated labels
    """
    if len(centers) == 1:
        return centers, percentages, labels

    # Get color names for all centers
    color_info = []
    for i, center in enumerate(centers):
        name, base = lab_to_color_name_and_base(center)
        color_info.append({'idx': i, 'name': name, 'base': base, 'pct': percentages[i]})

    # Group by color name
    name_groups = defaultdict(list)
    for info in color_info:
        name_groups[info['name']].append(info)

    # If no duplicates, return as-is
    if all(len(group) == 1 for group in name_groups.values()):
        return centers, percentages, labels

    # Merge duplicate groups
    new_centers = []
    new_percentages = []
    new_labels = np.copy(labels)
    label_mapping = {}
    next_label = 0

    for name, group in name_groups.items():
        if len(group) == 1:
            # No duplication for this color
            old_idx = group[0]['idx']
            label_mapping[old_idx] = next_label
            new_centers.append(centers[old_idx])
            new_percentages.append(group[0]['pct'])
        else:
            # Multiple clusters with same name - merge them
            old_indices = [info['idx'] for info in group]
            merged_pct = sum(info['pct'] for info in group)

            # Combine regions from all duplicate clusters
            merged_mask = np.isin(labels, old_indices)
            merged_pixels = pixels[merged_mask]

            # Re-extract color from merged region
            if len(merged_pixels) > 0:
                merged_color = extract_func(merged_pixels)
                if merged_color is not None:
                    new_centers.append(merged_color)
                    new_percentages.append(merged_pct)
                else:
                    # Fallback: use average of duplicate centers
                    new_centers.append(np.mean([centers[idx] for idx in old_indices], axis=0))
                    new_percentages.append(merged_pct)
            else:
                new_centers.append(np.mean([centers[idx] for idx in old_indices], axis=0))
                new_percentages.append(merged_pct)

            # Map all old labels to new merged label
            for old_idx in old_indices:
                label_mapping[old_idx] = next_label

        next_label += 1

    # Remap labels
    for old_label, new_label in label_mapping.items():
        new_labels[labels == old_label] = new_label

    return np.array(new_centers), new_percentages, new_labels

def process_sample(debug_dir, sample_name):
    """Process a single sample with all 12 methods (2 detection × 4 extraction + 4 global)."""
    swatch, mask = load_swatch_fragment(debug_dir, sample_name)

    if swatch is None or mask is None:
        return None

    pixels = extract_lab_pixels(swatch, mask)

    if len(pixels) == 0:
        return None

    results = []

    # === Multi-Tone Detection (8 combinations: 2 detection × 4 extraction) ===

    detection_methods = {
        'K-means': kmeans_clustering,
        'Histogram': histogram_multimodality,
    }

    extraction_methods = [
        ('Gaussian', extract_gaussian),
        ('Average', extract_average),
        ('Median', extract_median),
        ('Mode', extract_mode),
    ]

    for detect_name, detect_func in detection_methods.items():
        # Run detection
        centers, percentages, labels = detect_func(pixels)

        if centers is None or len(centers) == 0:
            # Detection failed
            for extract_name, _ in extraction_methods:
                results.append({
                    'detection': detect_name,
                    'extraction': extract_name,
                    'colors': [],
                    'shading': False
                })
            continue

        # Apply each extraction method
        for extract_name, extract_func in extraction_methods:
            # Extract colors for each region
            extracted_centers = []
            for i in range(len(centers)):
                region_pixels = pixels[labels == i]
                if len(region_pixels) > 0:
                    color = extract_func(region_pixels)
                    if color is not None:
                        extracted_centers.append(color)
                    else:
                        extracted_centers.append(centers[i])
                else:
                    extracted_centers.append(centers[i])

            extracted_centers = np.array(extracted_centers)

            # Deduplicate colors with same ISCC-NBS name
            dedup_centers, dedup_percentages, dedup_labels = deduplicate_colors(
                extracted_centers, percentages, labels, pixels, extract_func
            )

            # Sort by frequency (descending)
            sorted_indices = np.argsort(dedup_percentages)[::-1]
            sorted_centers = dedup_centers[sorted_indices]
            sorted_percentages = [dedup_percentages[i] for i in sorted_indices]

            # Get color names and base colors
            color_data = []
            for center, pct in zip(sorted_centers, sorted_percentages):
                name, base = lab_to_color_name_and_base(center)
                color_data.append({
                    'name': name,
                    'base': base,
                    'pct': pct,
                    'lab': center
                })

            # Calculate metrics
            delta_l = 0.0
            rel_delta_l = 0.0
            same_base = False
            shading = False

            if len(color_data) >= 2:
                # All colors must have same base color family
                bases = [c['base'] for c in color_data]
                same_base = all(b == bases[0] for b in bases) and bases[0] not in ('N', 'N/A')

                # All colors must have different names
                names = [c['name'] for c in color_data]
                different_names = len(set(names)) == len(names)

                # Calculate Delta L* and Rel Delta L*
                l_values = [c['lab'][0] for c in color_data]
                min_l = min(l_values)
                max_l = max(l_values)
                delta_l = max_l - min_l
                avg_l = (min_l + max_l) / 2
                rel_delta_l = (delta_l / avg_l * 100) if avg_l > 0 else 0

                # Shading if: same base + different names + relative ΔL* > 30%
                if same_base and different_names:
                    shading = rel_delta_l > 30

            results.append({
                'detection': detect_name,
                'extraction': extract_name,
                'colors': color_data,
                'delta_l': delta_l,
                'rel_delta_l': rel_delta_l,
                'same_base': same_base,
                'shading': shading
            })

    # === Global Single-Tone Methods (4 methods) ===

    global_methods = [
        ('Median', baseline_median),
        ('Darkest', baseline_darkest),
        ('MostSaturated', baseline_most_saturated),
        ('Mode', baseline_mode),
    ]

    for method_name, method_func in global_methods:
        color = method_func(pixels)
        if color is not None:
            color_name, color_base = lab_to_color_name_and_base(color)
            results.append({
                'detection': 'Global',
                'extraction': method_name,
                'colors': [{
                    'name': color_name,
                    'base': color_base,
                    'pct': 100.0,
                    'lab': color
                }],
                'delta_l': 0.0,
                'rel_delta_l': 0.0,
                'same_base': False,
                'shading': False
            })

    return results

def generate_report(all_results, samples, output_file, num_colors):
    """Generate markdown report for 2-color or 3-color results.

    Args:
        all_results: Dictionary of sample results
        samples: List of sample names
        output_file: Output markdown file path
        num_colors: 2 or 3 (number of color columns to show)
    """
    with open(output_file, 'w') as f:
        f.write("# Shading Detection Experiment\n\n")

        f.write("## Objective\n\n")
        f.write("Test different approaches for detecting ink shading properties by identifying distinct tones within a swatch.\n\n")

        f.write("## Experimental Design\n\n")
        f.write("**Multi-Tone Detection Methods (8 combinations):**\n")
        f.write("- **Detection**: K-means (k=3), Histogram (1-3 peaks)\n")
        f.write("- **Extraction**: Gaussian, Average, Median, Mode\n\n")

        f.write("**Global Single-Tone Methods (4 methods):**\n")
        f.write("1. **Median**: Pure median of L*, a*, b* (robust, no filtering)\n")
        f.write("2. **Darkest**: 10-25 percentile L* (concentrated/wet ink)\n")
        f.write("3. **MostSaturated**: Highest chroma pixels (true ink color)\n")
        f.write("4. **Mode**: Most frequent L* bin (global histogram)\n\n")

        f.write("**Shading Criteria:**\n")
        f.write("- All colors must have the same base color family (e.g., all \"blue\")\n")
        f.write("- All colors must have different ISCC-NBS names (no duplicates)\n")
        f.write("- Relative ΔL* > 30% between darkest and lightest tones\n\n")

        f.write("**Color Ordering:**\n")
        f.write("- Colors sorted by spatial frequency (%) descending\n")
        f.write("- Color 1 = most frequent, Color 2/3 = less frequent\n\n")

        f.write("**Metrics:**\n")
        f.write("- ΔL*: Absolute luminance difference between darkest and lightest\n")
        f.write("- Rel ΔL*: Relative luminance difference as % of average L*\n")
        f.write("- Same Base: Whether all colors share the same base color family\n")
        f.write("- Base Color: ISCC-NBS base color name from munsellspace\n\n")

        f.write("## Results\n\n")

        # Create table header based on num_colors
        if num_colors == 2:
            f.write("| Sample | Detection | Extraction | Color 1 | Base | % | Color 2 | Base | % | ΔL* | Rel ΔL* | Same Base | Shading |\n")
            f.write("|--------|-----------|------------|---------|------|---|---------|------|---|-----|---------|-----------|----------|\n")
        else:  # 3 colors
            f.write("| Sample | Detection | Extraction | Color 1 | Base | % | Color 2 | Base | % | Color 3 | Base | % | ΔL* | Rel ΔL* | Same Base | Shading |\n")
            f.write("|--------|-----------|------------|---------|------|---|---------|------|---|---------|------|---|-----|---------|-----------|----------|\n")

        for sample in samples:
            if sample not in all_results:
                continue

            results = all_results[sample]

            # Write all rows for this sample
            for r in results:
                detect = r['detection']
                extract = r['extraction']
                colors = r['colors'][:]  # Copy to avoid modifying original
                delta_l = r['delta_l']
                rel_delta_l = r['rel_delta_l']
                same_base = r['same_base']
                shading = r['shading']

                # Pad to required number of colors
                while len(colors) < num_colors:
                    colors.append({'name': '-', 'base': '-', 'pct': 0.0})

                # Format metric strings
                delta_str = f"{delta_l:.1f}" if delta_l > 0 else "-"
                rel_delta_str = f"{rel_delta_l:.1f}%" if rel_delta_l > 0 else "-"
                same_base_str = "✓" if same_base and len(r['colors']) >= 2 else ("✗" if len(r['colors']) >= 2 else "-")
                shading_str = "**YES**" if shading else "no"

                if num_colors == 2:
                    c1, c2 = colors[0], colors[1]
                    pct1_str = f"{c1['pct']:.1f}" if c1['pct'] > 0 else "-"
                    pct2_str = f"{c2['pct']:.1f}" if c2['pct'] > 0 else "-"
                    f.write(f"| {sample} | {detect} | {extract} | {c1['name']} | {c1['base']} | {pct1_str} | {c2['name']} | {c2['base']} | {pct2_str} | {delta_str} | {rel_delta_str} | {same_base_str} | {shading_str} |\n")
                else:  # 3 colors
                    c1, c2, c3 = colors[0], colors[1], colors[2]
                    pct1_str = f"{c1['pct']:.1f}" if c1['pct'] > 0 else "-"
                    pct2_str = f"{c2['pct']:.1f}" if c2['pct'] > 0 else "-"
                    pct3_str = f"{c3['pct']:.1f}" if c3['pct'] > 0 else "-"
                    f.write(f"| {sample} | {detect} | {extract} | {c1['name']} | {c1['base']} | {pct1_str} | {c2['name']} | {c2['base']} | {pct2_str} | {c3['name']} | {c3['base']} | {pct3_str} | {delta_str} | {rel_delta_str} | {same_base_str} | {shading_str} |\n")

            # Add spacing between samples (blank line with column bars)
            if num_colors == 2:
                f.write("| | | | | | | | | | | | | |\n")
            else:
                f.write("| | | | | | | | | | | | | | | | |\n")

        f.write("\n## Summary Statistics\n\n")

        # Count shading detections
        detection_shading = defaultdict(int)
        extraction_shading = defaultdict(int)

        for sample, results in all_results.items():
            for r in results:
                if r['detection'] != 'Global' and r['shading']:
                    detection_shading[r['detection']] += 1
                    extraction_shading[r['extraction']] += 1

        f.write(f"**Samples analyzed:** {len(all_results)} (excluding flash samples)\n\n")

        f.write("**Multi-Tone Shading Detection:**\n")
        f.write(f"- K-means: {detection_shading['K-means']} detections (across all 4 extraction methods)\n")
        f.write(f"- Histogram: {detection_shading['Histogram']} detections (across all 4 extraction methods)\n\n")

        f.write("**By Extraction Method:**\n")
        f.write(f"- Gaussian: {extraction_shading['Gaussian']} detections (across both detection methods)\n")
        f.write(f"- Average: {extraction_shading['Average']} detections (across both detection methods)\n")
        f.write(f"- Median: {extraction_shading['Median']} detections (across both detection methods)\n")
        f.write(f"- Mode: {extraction_shading['Mode']} detections (across both detection methods)\n\n")

        f.write("**Global Methods:**\n")
        f.write(f"- All global methods process {len(all_results)} samples (single-tone only)\n\n")

        f.write("## Observations\n\n")
        f.write("*To be filled after manual review of results*\n\n")

        f.write("## Next Steps\n\n")
        f.write("Based on these results, determine:\n")
        f.write("1. Which detection method works best?\n")
        f.write("2. Which extraction method works best?\n")
        f.write("3. Does the best combination vary by ink type?\n")
        f.write("4. How do multi-tone methods compare to global single-tone methods?\n")
        f.write("5. Should we adjust the 30% relative ΔL* threshold?\n")

def process_samples_with_cli(config_path):
    """Process all samples using Rust batch CLI with JSON configuration.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        List of sample names that were processed
    """
    config_path = Path(config_path)

    if not config_path.exists():
        print(f"Error: Config file {config_path} not found", file=sys.stderr)
        return []

    print(f"\nProcessing samples using config: {config_path}", file=sys.stderr)

    try:
        result = subprocess.run(
            ["cargo", "run", "--release", "--example", "cli_batch", "--", str(config_path)],
            capture_output=False,  # Show progress to user
            text=True,
            timeout=1800  # 30 minutes for batch processing (199 samples)
        )

        if result.returncode != 0:
            print(f"Error: Batch processing failed with return code {result.returncode}", file=sys.stderr)
            return []

    except subprocess.TimeoutExpired:
        print(f"Error: Batch processing timed out", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Error running batch CLI: {e}", file=sys.stderr)
        return []

    # Load config to find output directory and get sample list
    import json
    with open(config_path) as f:
        config = json.load(f)

    output_path = Path(config['output_path'])

    # Find all processed samples by looking at generated swatch images
    samples = []
    for swatch_file in sorted(output_path.glob("*_swatch.png")):
        sample_name = swatch_file.stem.replace("_swatch", "")
        samples.append(sample_name)

    print(f"✓ Processed {len(samples)} samples", file=sys.stderr)
    return samples

def main():
    if len(sys.argv) < 2:
        print("Usage: test_shading_detection.py <config.json>")
        print("Example: test_shading_detection.py \"validation/Experiment 0/config.json\"")
        sys.exit(1)

    config_path = sys.argv[1]

    # Process samples with Rust batch CLI (generates debug images)
    samples = process_samples_with_cli(config_path)

    if not samples:
        print("Error: No samples processed", file=sys.stderr)
        sys.exit(1)

    # Load config to get output directory
    import json
    with open(config_path) as f:
        config = json.load(f)

    output_dir = Path(config['output_path'])

    # Analyze swatch images for shading detection
    print("\nAnalyzing shading...", file=sys.stderr)
    all_results = {}
    for sample in samples:
        results = process_sample(str(output_dir), sample)
        if results:
            all_results[sample] = results

    # Generate both reports
    print("Generating reports...", file=sys.stderr)
    generate_report(all_results, samples, output_dir / "two-colors-results.md", num_colors=2)
    generate_report(all_results, samples, output_dir / "three-colors-results.md", num_colors=3)

    print(f"\n✓ Experiment complete!")
    print(f"  Two-color results: {output_dir / 'two-colors-results.md'}")
    print(f"  Three-color results: {output_dir / 'three-colors-results.md'}")
    print(f"  Images: {output_dir}/")

if __name__ == '__main__':
    main()
