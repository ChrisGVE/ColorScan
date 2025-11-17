use scan_colors::{analyze_swatch_with_method, color::ExtractionMethod};
use std::path::PathBuf;
use std::{env, fs};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: compare_methods <directory>");
        eprintln!("Example: cargo run --release --example compare_methods validation/local_test_samples");
        std::process::exit(1);
    }

    let dir_path = &args[1];

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

    // Methods to test
    let methods = [
        (ExtractionMethod::MedianMean, "MedianMean"),
        (ExtractionMethod::Darkest, "Darkest"),
        (ExtractionMethod::MostSaturated, "MostSaturated"),
        (ExtractionMethod::Mode, "Mode"),
    ];

    // Print table header
    println!("| Image | Method | Color Name | Confidence |");
    println!("|-------|--------|------------|------------|");

    // Process each image with each method
    for image_path in &image_files {
        let image_name = image_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        for (method, method_name) in &methods {
            match analyze_swatch_with_method(image_path, *method) {
                Ok(result) => {
                    println!(
                        "| {} | {} | {} | {:.0}% |",
                        image_name,
                        method_name,
                        result.color_name,
                        result.confidence * 100.0
                    );
                }
                Err(e) => {
                    println!(
                        "| {} | {} | ERROR: {} | - |",
                        image_name, method_name, e
                    );
                }
            }
        }
    }
}
