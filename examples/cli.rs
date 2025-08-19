//! Command-line interface for scan_colors
//!
//! Basic CLI tool for testing color analysis functionality

use scan_colors::{analyze_swatch, ColorResult, AnalysisError};
use std::{env, path::Path, process};
use serde_json;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() != 2 {
        eprintln!("Usage: {} <image_path>", args[0]);
        eprintln!();
        eprintln!("Analyze fountain pen ink color from an image file.");
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  {} photo.jpg", args[0]);
        eprintln!("  {} /path/to/swatch.heic", args[0]);
        process::exit(1);
    }

    let image_path = Path::new(&args[1]);
    
    if !image_path.exists() {
        eprintln!("Error: File '{}' does not exist", image_path.display());
        process::exit(1);
    }

    match analyze_swatch(image_path) {
        Ok(result) => {
            print_result(&result);
        }
        Err(error) => {
            eprintln!("Analysis failed: {}", error);
            
            // Print user-friendly message for common errors
            if error.is_recoverable() {
                eprintln!("Suggestion: {}", error.user_message());
            }
            
            process::exit(1);
        }
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
    eprintln!("  Lab Values: L*={:.1}, a*={:.1}, b*={:.1}", 
              result.lab.l, result.lab.a, result.lab.b);
    eprintln!("  LCh Values: L*={:.1}, C*={:.1}, h°={:.1}", 
              result.lch.l, result.lch.chroma, result.lch.hue.into_positive_degrees());
    eprintln!("  Confidence: {:.1}%", result.confidence * 100.0);
    
    if result.confidence < 0.5 {
        eprintln!("  Warning: Low confidence result. Consider better lighting or larger swatch.");
    }
}