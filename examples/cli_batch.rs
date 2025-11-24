//! Batch CLI for scan_colors with JSON configuration
//!
//! Processes all images in a directory using a JSON configuration file

use scan_colors::{analyze_swatch_debug, PipelineConfig};
use std::{env, path::{Path, PathBuf}, process, fs};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_help(&args[0]);
        process::exit(1);
    }

    let config_path = Path::new(&args[1]);

    if !config_path.exists() {
        eprintln!("Error: Config file '{}' does not exist", config_path.display());
        process::exit(1);
    }

    // Load configuration
    let config = match PipelineConfig::from_json_file(config_path) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("Error loading config file: {}", e);
            process::exit(1);
        }
    };

    eprintln!("Loaded configuration from {}", config_path.display());
    eprintln!("Input path: {}", config.input_path.display());
    eprintln!("Output path: {}", config.output_path.display());
    eprintln!();

    // Create output directory
    if let Err(e) = fs::create_dir_all(&config.output_path) {
        eprintln!("Error creating output directory: {}", e);
        process::exit(1);
    }

    // Find all image files in input directory
    let image_files = match find_image_files(&config.input_path) {
        Ok(files) => files,
        Err(e) => {
            eprintln!("Error finding image files: {}", e);
            process::exit(1);
        }
    };

    if image_files.is_empty() {
        eprintln!("No image files found in {}", config.input_path.display());
        process::exit(1);
    }

    eprintln!("Found {} image files to process", image_files.len());
    eprintln!();

    // Process each image
    let mut success_count = 0;
    let mut error_count = 0;

    for (i, image_path) in image_files.iter().enumerate() {
        let filename = image_path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        eprint!("[{}/{}] Processing {}... ", i + 1, image_files.len(), filename);

        match analyze_swatch_debug(image_path) {
            Ok((result, debug_output)) => {
                // Save debug artifacts
                let base_name = image_path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("output");

                if let Err(e) = save_debug_output(&debug_output, &config.output_path, base_name) {
                    eprintln!("Warning saving artifacts: {}", e);
                }

                eprintln!("✓");
                success_count += 1;

                // Optionally print result summary
                if env::var("VERBOSE").is_ok() {
                    eprintln!("  → Hex: {}, Munsell: {}, Confidence: {:.1}%",
                             result.hex, result.munsell, result.confidence * 100.0);
                }
            }
            Err(error) => {
                eprintln!("✗ {}", error);
                error_count += 1;
            }
        }
    }

    eprintln!();
    eprintln!("Batch processing complete:");
    eprintln!("  Success: {}", success_count);
    eprintln!("  Errors: {}", error_count);
    eprintln!("  Artifacts saved to: {}", config.output_path.display());

    if error_count > 0 {
        process::exit(1);
    }
}

fn print_help(program_name: &str) {
    eprintln!("Usage: {} <config.json>", program_name);
    eprintln!();
    eprintln!("Batch process fountain pen ink swatches using JSON configuration.");
    eprintln!();
    eprintln!("Arguments:");
    eprintln!("  config.json    JSON file containing pipeline configuration");
    eprintln!();
    eprintln!("Environment:");
    eprintln!("  VERBOSE=1      Print detailed results for each image");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  {} experiment_0.json", program_name);
    eprintln!("  VERBOSE=1 {} experiment_0.json", program_name);
}

fn find_image_files(dir: &Path) -> Result<Vec<PathBuf>, std::io::Error> {
    let mut files = Vec::new();

    if dir.is_file() {
        // Single file provided
        files.push(dir.to_path_buf());
        return Ok(files);
    }

    // Directory provided - scan for image files
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_str().unwrap_or("").to_lowercase();
                if matches!(ext_str.as_str(), "jpg" | "jpeg" | "png" | "heic") {
                    // Skip flash images
                    if let Some(name) = path.file_name() {
                        if let Some(name_str) = name.to_str() {
                            if !name_str.to_lowercase().contains("flash") {
                                files.push(path);
                            }
                        }
                    }
                }
            }
        }
    }

    files.sort();
    Ok(files)
}

fn save_debug_output(
    debug: &scan_colors::DebugOutput,
    output_dir: &Path,
    base_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use opencv::imgcodecs;
    use opencv::core::Vector;

    // Save original image (before white balance correction)
    let original_path = output_dir.join(format!("{}_original.png", base_name));
    imgcodecs::imwrite(
        original_path.to_str().ok_or("Invalid path")?,
        &debug.original_image,
        &Vector::new()
    )?;

    // Save corrected image (white balance applied)
    let corrected_path = output_dir.join(format!("{}_original_corrected.png", base_name));
    imgcodecs::imwrite(
        corrected_path.to_str().ok_or("Invalid path")?,
        &debug.corrected_image,
        &Vector::new()
    )?;

    // Save swatch fragment
    let swatch_path = output_dir.join(format!("{}_swatch.png", base_name));
    imgcodecs::imwrite(
        swatch_path.to_str().ok_or("Invalid path")?,
        &debug.swatch_fragment,
        &Vector::new()
    )?;

    // Save swatch mask
    let mask_path = output_dir.join(format!("{}_mask.png", base_name));
    imgcodecs::imwrite(
        mask_path.to_str().ok_or("Invalid path")?,
        &debug.swatch_mask,
        &Vector::new()
    )?;

    Ok(())
}
