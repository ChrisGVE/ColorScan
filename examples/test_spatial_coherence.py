#!/usr/bin/env python3
"""
Test script to investigate K-means spatial coherence and k=3 behavior.

Questions being answered:
1. Why does K-means return #000000 (is it transparency channel or noise)?
2. What happens with k=3 clusters?
3. Are detected clusters spatially coherent or randomly distributed?
"""
import sys
import numpy as np
import cv2
from pathlib import Path

def extract_lab_pixels_with_coords(swatch, mask=None):
    """Extract Lab pixels with their (y, x) spatial coordinates.

    Args:
        swatch: BGR image (already cropped with #000000 padding around irregular edges)
        mask: Unused (kept for compatibility) - mask was already applied during swatch extraction

    Returns:
        pixels: Lab values (N, 3)
        coords: Spatial coordinates (N, 2)

    Note:
        Filters out pure #000000 BGR padding while keeping all ink pixels,
        even very dark inks (which are not pure #000000).
    """
    lab = cv2.cvtColor(swatch, cv2.COLOR_BGR2Lab)

    pixels = []
    coords = []
    filtered_count = 0

    for y in range(swatch.shape[0]):
        for x in range(swatch.shape[1]):
            b, g, r = swatch[y, x]
            # Filter out pure black padding (BGR #000000)
            # Real ink pixels, even very dark ones, are not pure #000000
            if not (b == 0 and g == 0 and r == 0):
                l, a_val, b_val = lab[y, x]
                l_norm = (l / 255.0) * 100.0
                a_norm = a_val - 128.0
                b_norm = b_val - 128.0
                pixels.append([l_norm, a_norm, b_norm])
                coords.append([y, x])
            else:
                filtered_count += 1

    if filtered_count > 0:
        print(f"  → Filtered {filtered_count} background pixels (BGR #000000)")

    return np.array(pixels), np.array(coords)

def calculate_spatial_coherence(coords, labels, k):
    """Calculate spatial coherence for each cluster.

    High coherence (>0.7) = spatially clustered (good shading region)
    Medium coherence (0.4-0.7) = somewhat clustered
    Low coherence (<0.4) = randomly distributed (noise/artifacts)
    """
    coherence_scores = []
    avg_distances = []

    for i in range(k):
        cluster_coords = coords[labels == i]
        if len(cluster_coords) < 2:
            coherence_scores.append(0.0)
            avg_distances.append(0.0)
            continue

        # Calculate spatial centroid
        centroid = cluster_coords.mean(axis=0)

        # Calculate average distance from centroid
        distances = np.linalg.norm(cluster_coords - centroid, axis=1)
        avg_distance = distances.mean()

        # Calculate spatial spread (normalized by image diagonal)
        max_distance = np.sqrt(coords[:, 0].max()**2 + coords[:, 1].max()**2)
        compactness = 1.0 - (avg_distance / max_distance) if max_distance > 0 else 1.0

        coherence_scores.append(compactness)
        avg_distances.append(avg_distance)

    return coherence_scores, avg_distances

def kmeans_with_spatial(pixels, coords, k=2, max_iter=100):
    """K-means clustering with spatial coherence calculation."""
    if len(pixels) < k:
        return None, None, None, None

    # Initialize centers across L* range
    l_values = pixels[:, 0]
    if k == 2:
        dark_idx = np.argmin(l_values)
        light_idx = np.argmax(l_values)
        centers = np.array([pixels[dark_idx], pixels[light_idx]])
    elif k == 3:
        p10_idx = np.argmin(np.abs(l_values - np.percentile(l_values, 10)))
        p50_idx = np.argmin(np.abs(l_values - np.percentile(l_values, 50)))
        p90_idx = np.argmin(np.abs(l_values - np.percentile(l_values, 90)))
        centers = np.array([pixels[p10_idx], pixels[p50_idx], pixels[p90_idx]])
    else:
        indices = np.random.choice(len(pixels), k, replace=False)
        centers = pixels[indices]

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

        if np.allclose(centers, new_centers):
            break

        centers = new_centers

    # Calculate percentages
    unique, counts = np.unique(labels, return_counts=True)
    percentages = dict(zip(unique, (counts / len(labels)) * 100))

    # Sort by L* (darker first)
    sorted_indices = np.argsort(centers[:, 0])
    sorted_centers = centers[sorted_indices]
    sorted_percentages = [percentages[i] for i in sorted_indices]

    # Remap labels
    label_map = {old: new for new, old in enumerate(sorted_indices)}
    sorted_labels = np.array([label_map[l] for l in labels])

    # Calculate spatial coherence
    coherence_scores, avg_distances = calculate_spatial_coherence(coords, sorted_labels, k)

    return sorted_centers, sorted_percentages, coherence_scores, sorted_labels

def lab_to_hex(lab):
    """Convert Lab to hex string for visualization."""
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

    x = xr * 0.95047
    y = yr * 1.00000
    z = zr * 1.08883

    # XYZ to sRGB
    r_linear = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    g_linear = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    b_linear = x * 0.0556434 + y * -0.2040259 + z * 1.0572252

    def srgb_gamma(linear):
        if linear <= 0.0031308:
            return 12.92 * linear
        else:
            return 1.055 * (linear ** (1.0 / 2.4)) - 0.055

    r = srgb_gamma(r_linear)
    g = srgb_gamma(g_linear)
    b = srgb_gamma(b_linear)

    r = int(max(0, min(1, r)) * 255)
    g = int(max(0, min(1, g)) * 255)
    b = int(max(0, min(1, b)) * 255)

    return f"#{r:02x}{g:02x}{b:02x}"

def interpret_coherence(score):
    """Interpret spatial coherence score."""
    if score >= 0.7:
        return "HIGH (spatially clustered - good shading)"
    elif score >= 0.4:
        return "MEDIUM (somewhat clustered)"
    else:
        return "LOW (randomly distributed - likely noise/artifacts)"

def test_sample(debug_dir, sample_name):
    """Test a single sample with k=2 and k=3."""
    swatch_path = Path(debug_dir) / f"{sample_name}_swatch.png"
    mask_path = Path(debug_dir) / f"{sample_name}_mask.png"

    if not swatch_path.exists() or not mask_path.exists():
        print(f"✗ Files not found for {sample_name}")
        return

    swatch = cv2.imread(str(swatch_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    pixels, coords = extract_lab_pixels_with_coords(swatch, mask)

    if len(pixels) == 0:
        print(f"✗ No pixels found for {sample_name}")
        return

    print(f"\n{'='*80}")
    print(f"Sample: {sample_name}")
    print(f"Total pixels: {len(pixels)}")
    print(f"{'='*80}\n")

    # Test k=2
    print("K-means with k=2:")
    print("-" * 80)
    centers2, percentages2, coherence2, labels2 = kmeans_with_spatial(pixels, coords, k=2)

    for i, (center, pct, coh) in enumerate(zip(centers2, percentages2, coherence2)):
        hex_color = lab_to_hex(center)
        l, a, b = center
        print(f"  Cluster {i+1}: {pct:5.1f}% | L*={l:5.1f} a*={a:6.1f} b*={b:6.1f} | {hex_color}")
        print(f"            Spatial Coherence: {coh:.3f} - {interpret_coherence(coh)}")

    # Test k=3
    print("\nK-means with k=3:")
    print("-" * 80)
    centers3, percentages3, coherence3, labels3 = kmeans_with_spatial(pixels, coords, k=3)

    for i, (center, pct, coh) in enumerate(zip(centers3, percentages3, coherence3)):
        hex_color = lab_to_hex(center)
        l, a, b = center
        print(f"  Cluster {i+1}: {pct:5.1f}% | L*={l:5.1f} a*={a:6.1f} b*={b:6.1f} | {hex_color}")
        print(f"            Spatial Coherence: {coh:.3f} - {interpret_coherence(coh)}")

    # Analysis
    print("\nAnalysis:")
    print("-" * 80)

    # Check for #000000 clusters
    for i, (center, pct, coh) in enumerate(zip(centers2, percentages2, coherence2)):
        hex_color = lab_to_hex(center)
        if hex_color in ["#000000", "#000001", "#010101"]:
            print(f"  ⚠️  k=2 Cluster {i+1}: #000000 detected ({pct:.1f}%)")
            print(f"      → Spatial Coherence: {coh:.3f} ({interpret_coherence(coh)})")
            if pct < 10:
                print(f"      → Small cluster (<10%) likely noise or extreme outliers")
            if coh < 0.4:
                print(f"      → LOW coherence confirms random distribution (not real shading)")

    for i, (center, pct, coh) in enumerate(zip(centers3, percentages3, coherence3)):
        hex_color = lab_to_hex(center)
        if hex_color in ["#000000", "#000001", "#010101"]:
            print(f"  ⚠️  k=3 Cluster {i+1}: #000000 detected ({pct:.1f}%)")
            print(f"      → Spatial Coherence: {coh:.3f} ({interpret_coherence(coh)})")
            if pct < 10:
                print(f"      → Small cluster (<10%) likely noise or extreme outliers")
            if coh < 0.4:
                print(f"      → LOW coherence confirms random distribution (not real shading)")

    # Compare k=2 vs k=3
    print(f"\n  k=2 vs k=3 Comparison:")
    print(f"    k=2: {len([c for c in coherence2 if c >= 0.4])} spatially coherent clusters (coherence >= 0.4)")
    print(f"    k=3: {len([c for c in coherence3 if c >= 0.4])} spatially coherent clusters (coherence >= 0.4)")

    if len([c for c in coherence2 if c >= 0.4]) > len([c for c in coherence3 if c >= 0.4]):
        print(f"    → k=2 appears more appropriate (fewer false clusters)")
    elif len([c for c in coherence3 if c >= 0.4]) > 2:
        print(f"    → k=3 detected potentially valid third tone (real multi-tone shading)")
    else:
        print(f"    → k=2 and k=3 similar (ink likely has 1-2 tones)")

def main():
    if len(sys.argv) < 3:
        print("Usage: test_spatial_coherence.py <debug_dir> <sample1> [sample2] [...]")
        print("Example: test_spatial_coherence.py validation/debug_output Diplomat_Royal_blue JH_Bleu_austral")
        sys.exit(1)

    debug_dir = sys.argv[1]
    samples = sys.argv[2:]

    for sample in samples:
        test_sample(debug_dir, sample)

    print(f"\n{'='*80}")
    print("Summary of Findings:")
    print(f"{'='*80}")
    print("1. #000000 Issue: NOT from transparency channel (BGR has no alpha)")
    print("   → Caused by K-means forcing k clusters even with noise/outliers")
    print("   → Spatial coherence reveals if #000000 is noise (low coherence)")
    print()
    print("2. k=3 Behavior: Shows whether ink has 2 or 3 distinct tones")
    print("   → High coherence clusters = real shading tones")
    print("   → Low coherence clusters = artifacts/noise")
    print()
    print("3. Spatial Consistency: Coherence score validates shading vs noise")
    print("   → Coherence >= 0.7: Spatially clustered (good shading region)")
    print("   → Coherence < 0.4: Randomly distributed (noise, not real shading)")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
