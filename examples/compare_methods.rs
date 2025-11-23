use scan_colors::{analyze_swatch_with_method, analyze_swatch_debug, color::ExtractionMethod};
use std::path::{Path, PathBuf};
use std::{env, fs};
use opencv::imgcodecs;
use opencv::core::Vector;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: compare_methods <directory> [output_dir] [experiment_name]");
        eprintln!("Example: cargo run --release --example compare_methods validation/local_test_samples validation/debug_output experiment_1");
        std::process::exit(1);
    }

    let dir_path = &args[1];
    let output_dir = if args.len() > 2 {
        &args[2]
    } else {
        "validation/debug_output"
    };
    let experiment_name = if args.len() > 3 {
        &args[3]
    } else {
        "exp1"
    };

    // Collect all image files
    let mut image_files: Vec<PathBuf> = Vec::new();
    match fs::read_dir(dir_path) {
        Ok(entries) => {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    let ext_str = ext.to_string_lossy().to_lowercase();
                    if ext_str == "jpg" || ext_str == "jpeg" || ext_str == "png" {
                        image_files.push(path);
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Error reading directory: {}", e);
            std::process::exit(1);
        }
    }

    if image_files.is_empty() {
        eprintln!("No JPEG or PNG images found in directory: {}", dir_path);
        std::process::exit(1);
    }

    image_files.sort();

    // Create output directory
    fs::create_dir_all(output_dir).expect("Failed to create output directory");

    // Methods to test
    let methods = [
        (ExtractionMethod::MedianMean, "Method1_MedianMean"),
        (ExtractionMethod::Darkest, "Method2_Darkest"),
        (ExtractionMethod::MostSaturated, "Method3_MostSaturated"),
        (ExtractionMethod::Mode, "Method4_Mode"),
    ];

    // Print table header with experiment name
    println!("# Experiment: {}", experiment_name);
    println!();
    println!("| Sample | Method1 (MedianMean) | Method2 (Darkest) | Method3 (MostSaturated) | Method4 (Mode) |");
    println!("|--------|---------------------|-------------------|------------------------|----------------|");

    // Process each image with all methods
    for image_path in &image_files {
        let image_name = image_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        // First, process with debug output to save images (using default MedianMean method)
        if let Ok((_, debug_output)) = analyze_swatch_debug(image_path) {
            save_debug_output(&debug_output, Path::new(output_dir), image_path);
        } else {
            eprintln!("Warning: Failed to generate debug output for {}", image_name);
        }

        // Now analyze with each method and collect results
        let mut results = Vec::new();
        for (method, _) in &methods {
            match analyze_swatch_with_method(image_path, *method) {
                Ok(result) => {
                    results.push(result.color_name.clone());
                }
                Err(e) => {
                    results.push(format!("ERROR: {}", e));
                }
            }
        }

        // Print one row per image with all methods
        println!(
            "| {} | {} | {} | {} | {} |",
            image_name,
            results[0],
            results[1],
            results[2],
            results[3]
        );
    }
}

fn save_debug_output(debug: &scan_colors::DebugOutput, output_dir: &Path, input_path: &Path) {
    // Generate base filename from input
    let base_name = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");

    // Save original image (before white balance correction)
    let original_path = output_dir.join(format!("{}_original.png", base_name));
    let _ = imgcodecs::imwrite(original_path.to_str().unwrap(), &debug.original_image, &Vector::new());

    // Save corrected image (white balance applied)
    let corrected_path = output_dir.join(format!("{}_corrected.png", base_name));
    let _ = imgcodecs::imwrite(corrected_path.to_str().unwrap(), &debug.corrected_image, &Vector::new());

    // Save swatch fragment
    let swatch_path = output_dir.join(format!("{}_swatch.png", base_name));
    let _ = imgcodecs::imwrite(swatch_path.to_str().unwrap(), &debug.swatch_fragment, &Vector::new());

    // Save swatch mask
    let mask_path = output_dir.join(format!("{}_mask.png", base_name));
    let _ = imgcodecs::imwrite(mask_path.to_str().unwrap(), &debug.swatch_mask, &Vector::new());
}
