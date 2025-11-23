//! Command-line interface for scan_colors
//!
//! Basic CLI tool for testing color analysis functionality

use scan_colors::{analyze_swatch, analyze_swatch_debug, ColorResult};
use std::{env, path::{Path, PathBuf}, process};
use serde_json;

fn main() {
    let args: Vec<String> = env::args().collect();

    let mut debug_mode = false;
    let mut debug_output_dir = None;
    let mut image_path_arg = None;

    // Parse arguments
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--debug" => {
                debug_mode = true;
                // Check if next arg is a directory (not an image file)
                if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    let next_arg = &args[i + 1];
                    let is_image = next_arg.ends_with(".jpg") || next_arg.ends_with(".jpeg") ||
                                 next_arg.ends_with(".heic") || next_arg.ends_with(".png") ||
                                 next_arg.ends_with(".JPEG") || next_arg.ends_with(".JPG") ||
                                 next_arg.ends_with(".HEIC") || next_arg.ends_with(".PNG");

                    if !is_image {
                        debug_output_dir = Some(PathBuf::from(next_arg));
                        i += 1;
                    } else {
                        debug_output_dir = Some(PathBuf::from("validation"));
                    }
                } else {
                    debug_output_dir = Some(PathBuf::from("validation"));
                }
            }
            "--help" | "-h" => {
                print_help(&args[0]);
                process::exit(0);
            }
            arg if !arg.starts_with("--") => {
                if image_path_arg.is_none() {
                    image_path_arg = Some(arg.to_string());
                } else {
                    eprintln!("Error: Multiple image paths provided");
                    process::exit(1);
                }
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
                eprintln!("Use --help for usage information");
                process::exit(1);
            }
        }
        i += 1;
    }

    let image_path_str = match image_path_arg {
        Some(path) => path,
        None => {
            print_help(&args[0]);
            process::exit(1);
        }
    };

    let image_path = Path::new(&image_path_str);

    if !image_path.exists() {
        eprintln!("Error: File '{}' does not exist", image_path.display());
        process::exit(1);
    }

    if debug_mode {
        let output_dir = debug_output_dir.unwrap_or_else(|| PathBuf::from("validation"));
        match analyze_swatch_debug(image_path) {
            Ok((result, debug_output)) => {
                print_result(&result);
                save_debug_output(&debug_output, &output_dir, image_path);
            }
            Err(error) => {
                eprintln!("Analysis failed: {}", error);
                if error.is_recoverable() {
                    eprintln!("Suggestion: {}", error.user_message());
                }
                process::exit(1);
            }
        }
    } else {
        match analyze_swatch(image_path) {
            Ok(result) => {
                print_result(&result);
            }
            Err(error) => {
                eprintln!("Analysis failed: {}", error);
                if error.is_recoverable() {
                    eprintln!("Suggestion: {}", error.user_message());
                }
                process::exit(1);
            }
        }
    }
}

fn print_help(program_name: &str) {
    eprintln!("Usage: {} [OPTIONS] <image_path>", program_name);
    eprintln!();
    eprintln!("Analyze fountain pen ink color from an image file.");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --debug [DIR]    Save debug images (corrected image, swatch fragment)");
    eprintln!("                   Optional: specify output directory (default: validation/)");
    eprintln!("  --help, -h       Show this help message");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  {} photo.jpg", program_name);
    eprintln!("  {} --debug photo.jpg", program_name);
    eprintln!("  {} --debug output/ swatch.heic", program_name);
}

fn save_debug_output(debug: &scan_colors::DebugOutput, output_dir: &Path, input_path: &Path) {
    use opencv::imgcodecs;
    use opencv::core::Vector;
    use std::fs;

    // Create output directory if it doesn't exist
    if let Err(e) = fs::create_dir_all(output_dir) {
        eprintln!("Warning: Failed to create output directory: {}", e);
        return;
    }

    // Generate base filename from input
    let base_name = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");

    // Save original image (before white balance correction)
    let original_path = output_dir.join(format!("{}_original.png", base_name));
    match imgcodecs::imwrite(original_path.to_str().unwrap(), &debug.original_image, &Vector::new()) {
        Ok(_) => eprintln!("Debug: Saved original image to {}", original_path.display()),
        Err(e) => eprintln!("Warning: Failed to save original image: {}", e),
    }

    // Save corrected image (white balance applied)
    let corrected_path = output_dir.join(format!("{}_corrected.png", base_name));
    match imgcodecs::imwrite(corrected_path.to_str().unwrap(), &debug.corrected_image, &Vector::new()) {
        Ok(_) => eprintln!("Debug: Saved corrected image to {}", corrected_path.display()),
        Err(e) => eprintln!("Warning: Failed to save corrected image: {}", e),
    }

    // Save swatch fragment
    let swatch_path = output_dir.join(format!("{}_swatch.png", base_name));
    match imgcodecs::imwrite(swatch_path.to_str().unwrap(), &debug.swatch_fragment, &Vector::new()) {
        Ok(_) => eprintln!("Debug: Saved swatch fragment to {}", swatch_path.display()),
        Err(e) => eprintln!("Warning: Failed to save swatch fragment: {}", e),
    }

    // Save swatch mask
    let mask_path = output_dir.join(format!("{}_mask.png", base_name));
    match imgcodecs::imwrite(mask_path.to_str().unwrap(), &debug.swatch_mask, &Vector::new()) {
        Ok(_) => eprintln!("Debug: Saved swatch mask to {}", mask_path.display()),
        Err(e) => eprintln!("Warning: Failed to save swatch mask: {}", e),
    }
}

fn print_result(result: &ColorResult) {
    // Print JSON to stdout for programmatic use
    match serde_json::to_string_pretty(result) {
        Ok(json) => println!("{}", json),
        Err(e) => {
            eprintln!("Error serializing result: {}", e);
            
            // Fallback to manual formatting
            println!("{{");
            println!("  \"lab\": [{{\"l\": {}, \"a\": {}, \"b\": {}}}],", 
                     result.lab.l, result.lab.a, result.lab.b);
            println!("  \"lch\": [{{\"l\": {}, \"c\": {}, \"h\": {}}}],", 
                     result.lch.l, result.lch.chroma, result.lch.hue.into_positive_degrees());
            println!("  \"srgb\": [{{\"r\": {}, \"g\": {}, \"b\": {}}}],", 
                     result.srgb.red, result.srgb.green, result.srgb.blue);
            println!("  \"hex\": \"{}\",", result.hex);
            println!("  \"confidence\": {}", result.confidence);
            println!("}}");
        }
    }
    
    // Print summary to stderr for human reading
    eprintln!();
    eprintln!("Color Analysis Summary:");
    eprintln!("  Hex Color: {}", result.hex);
    eprintln!("  Munsell: {}", result.munsell);
    eprintln!("  Lab Values: L*={:.1}, a*={:.1}, b*={:.1}",
              result.lab.l, result.lab.a, result.lab.b);
    eprintln!("  LCh Values: L*={:.1}, C*={:.1}, h°={:.1}",
              result.lch.l, result.lch.chroma, result.lch.hue.into_positive_degrees());
    eprintln!("  Confidence: {:.1}%", result.confidence * 100.0);

    if result.confidence < 0.5 {
        eprintln!("  Warning: Low confidence result. Consider better lighting or larger swatch.");
    }
}