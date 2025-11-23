#!/usr/bin/env python3
"""
Shading Detection Experiment
Tests different approaches for detecting ink shading properties.
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

def extract_lab_pixels(swatch, mask):
    """Extract Lab pixels from masked swatch region."""
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
    """Detect bimodality in L* histogram."""
    l_values = pixels[:, 0]
    hist, bins = np.histogram(l_values, bins=20)

    # Find peaks with simple peak detection
    peaks = simple_find_peaks(hist, min_height=len(pixels) * 0.05)

    if len(peaks) >= 2:
        # Get two highest peaks
        peak_heights = hist[peaks]
        top_two = peaks[np.argsort(peak_heights)[-2:]]

        # Get pixels in each peak region
        bin_width = bins[1] - bins[0]
        tone1_mask = (l_values >= bins[top_two[0]]) & (l_values < bins[top_two[0] + 1])
        tone2_mask = (l_values >= bins[top_two[1]]) & (l_values < bins[top_two[1] + 1])

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

def gaussian_blur_method(swatch, mask):
    """Apply Gaussian blur and extract representative color."""
    # Ensure swatch and mask have same dimensions
    if swatch.shape[:2] != mask.shape[:2]:
        # Resize mask to match swatch dimensions
        mask = cv2.resize(mask, (swatch.shape[1], swatch.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply strong Gaussian blur
    blurred = cv2.GaussianBlur(swatch, (51, 51), 0)
    lab_blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2Lab)

    # Find center of mass of mask
    M = cv2.moments(mask)
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

def average_method(pixels):
    """Simple average of all pixels."""
    return pixels.mean(axis=0)

def median_method(pixels):
    """Median of all pixels."""
    return np.median(pixels, axis=0)

def lab_to_color_name(lab):
    """Convert Lab to rough color name (simplified)."""
    l, a, b = lab

    if l < 30:
        return f"dark L*{l:.0f}"
    elif l > 70:
        return f"light L*{l:.0f}"
    else:
        return f"moderate L*{l:.0f}"

def check_same_munsell_family(lab1, lab2):
    """Simplified check if colors are in same family based on hue angle."""
    import math

    # Calculate hue angles
    h1 = math.atan2(lab1[2], lab1[1]) * 180 / math.pi
    h2 = math.atan2(lab2[2], lab2[1]) * 180 / math.pi

    # Normalize to 0-360
    h1 = h1 if h1 >= 0 else h1 + 360
    h2 = h2 if h2 >= 0 else h2 + 360

    # Check if within ~60 degrees (same general hue family)
    hue_diff = min(abs(h1 - h2), 360 - abs(h1 - h2))

    return hue_diff < 60

def process_sample(debug_dir, sample_name):
    """Process a single sample with all methods."""
    swatch, mask = load_swatch_fragment(debug_dir, sample_name)

    if swatch is None or mask is None:
        return None

    pixels = extract_lab_pixels(swatch, mask)

    if len(pixels) == 0:
        return None

    results = {}

    # Method 1: K-means clustering
    centers, percentages = kmeans_clustering(pixels, k=2)
    if centers is not None:
        delta_l = abs(centers[1][0] - centers[0][0])
        same_family = check_same_munsell_family(centers[0], centers[1])
        shading = same_family and delta_l > 10

        results['kmeans'] = {
            'tone1': lab_to_color_name(centers[0]),
            'tone1_pct': percentages[0],
            'tone2': lab_to_color_name(centers[1]),
            'tone2_pct': percentages[1],
            'delta_l': delta_l,
            'same_family': same_family,
            'shading': shading
        }

    # Method 2: Histogram bimodality
    centers, percentages = histogram_bimodality(pixels)
    if centers is not None:
        delta_l = abs(centers[1][0] - centers[0][0])
        same_family = check_same_munsell_family(centers[0], centers[1])
        shading = same_family and delta_l > 10

        results['histogram'] = {
            'tone1': lab_to_color_name(centers[0]),
            'tone1_pct': percentages[0],
            'tone2': lab_to_color_name(centers[1]),
            'tone2_pct': percentages[1],
            'delta_l': delta_l,
            'same_family': same_family,
            'shading': shading
        }

    # Method 3: Average (baseline - no shading detection)
    avg = average_method(pixels)
    results['average'] = {
        'tone1': lab_to_color_name(avg),
        'tone1_pct': 100.0,
        'tone2': '-',
        'tone2_pct': 0.0,
        'delta_l': 0.0,
        'same_family': True,
        'shading': False
    }

    # Method 4: Median (baseline - no shading detection)
    med = median_method(pixels)
    results['median'] = {
        'tone1': lab_to_color_name(med),
        'tone1_pct': 100.0,
        'tone2': '-',
        'tone2_pct': 0.0,
        'delta_l': 0.0,
        'same_family': True,
        'shading': False
    }

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
        f.write("Test different methods for detecting ink shading properties by identifying distinct tones within a swatch.\n\n")

        f.write("## Methodology\n\n")
        f.write("**Tone Detection Methods:**\n")
        f.write("1. **K-means Clustering**: Cluster pixels into 2 groups based on Lab values\n")
        f.write("2. **Histogram Bimodality**: Detect peaks in L* histogram to identify distinct tones\n")
        f.write("3. **Average**: Baseline - simple average of all pixels (no shading detection)\n")
        f.write("4. **Median**: Baseline - median of all pixels (no shading detection)\n\n")

        f.write("**Shading Criteria:**\n")
        f.write("- Both tones must be in same Munsell hue family (hue angle difference < 60°)\n")
        f.write("- ΔL* > 10 between tones (significant luminance difference)\n\n")

        f.write("**Output for Each Method:**\n")
        f.write("- Tone 1 (darker): Color description, coverage %\n")
        f.write("- Tone 2 (lighter): Color description, coverage %\n")
        f.write("- ΔL*: Luminance difference\n")
        f.write("- Shading: Yes/No based on criteria\n\n")

        f.write("## Results\n\n")

        # Create detailed table
        f.write("| Sample | Method | Tone 1 (Dark) | % | Tone 2 (Light) | % | ΔL* | Same Family | Shading |\n")
        f.write("|--------|--------|---------------|---|----------------|---|-----|-------------|----------|\n")

        for sample in samples:
            if sample not in all_results:
                continue

            results = all_results[sample]

            # K-means
            if 'kmeans' in results:
                r = results['kmeans']
                f.write(f"| {sample} | K-means | {r['tone1']} | {r['tone1_pct']:.1f} | {r['tone2']} | {r['tone2_pct']:.1f} | {r['delta_l']:.1f} | {'✓' if r['same_family'] else '✗'} | {'**YES**' if r['shading'] else 'no'} |\n")

            # Histogram
            if 'histogram' in results:
                r = results['histogram']
                f.write(f"| {sample} | Histogram | {r['tone1']} | {r['tone1_pct']:.1f} | {r['tone2']} | {r['tone2_pct']:.1f} | {r['delta_l']:.1f} | {'✓' if r['same_family'] else '✗'} | {'**YES**' if r['shading'] else 'no'} |\n")

            # Average
            r = results['average']
            f.write(f"| {sample} | Average | {r['tone1']} | {r['tone1_pct']:.0f} | - | - | - | - | no |\n")

            # Median
            r = results['median']
            f.write(f"| {sample} | Median | {r['tone1']} | {r['tone1_pct']:.0f} | - | - | - | - | no |\n")

            # Add spacing between samples
            f.write("|--------|--------|---------------|---|----------------|---|-----|-------------|----------|\n")

        f.write("\n## Summary Statistics\n\n")

        # Count shading detections per method
        shading_counts = defaultdict(int)
        for sample, results in all_results.items():
            for method, r in results.items():
                if method in ['kmeans', 'histogram'] and r['shading']:
                    shading_counts[method] += 1

        f.write(f"**Samples analyzed:** {len(all_results)} (excluding flash samples)\n\n")
        f.write("**Shading detected by method:**\n")
        f.write(f"- K-means: {shading_counts['kmeans']} samples\n")
        f.write(f"- Histogram: {shading_counts['histogram']} samples\n\n")

        f.write("## Observations\n\n")
        f.write("*To be filled after manual review of results*\n\n")

        f.write("## Next Steps\n\n")
        f.write("Based on these results, determine:\n")
        f.write("1. Which tone detection method works best?\n")
        f.write("2. Should we adjust ΔL* threshold?\n")
        f.write("3. Should we adjust hue angle threshold for \"same family\"?\n")
        f.write("4. Which blur method should we use for final color extraction?\n")

    print(f"\nReport generated: {output_file}")

if __name__ == '__main__':
    main()
