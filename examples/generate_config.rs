//! Generate default Experiment 0 configuration file
//!
//! Creates a JSON config with all default parameters

use inkswatch_colorscan::PipelineConfig;
use std::{env, path::Path, process};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <output_config.json>", args[0]);
        eprintln!();
        eprintln!("Example:");
        eprintln!("  {} validation/Experiment\\ 0/config.json", args[0]);
        process::exit(1);
    }

    let output_path = Path::new(&args[1]);

    // Create parent directory if needed
    if let Some(parent) = output_path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            eprintln!("Error creating directory: {}", e);
            process::exit(1);
        }
    }

    // Generate default config
    let config = PipelineConfig::default_experiment_0();

    // Save to file
    match config.to_json_file(output_path) {
        Ok(_) => {
            eprintln!("Configuration saved to {}", output_path.display());
            eprintln!();
            eprintln!("Config summary:");
            eprintln!("  Input:  {}", config.input_path.display());
            eprintln!("  Output: {}", config.output_path.display());
            eprintln!("  Paper detection: Canny ({:.0}, {:.0}), min area {:.0}%",
                     config.paper_detection.canny_low_threshold,
                     config.paper_detection.canny_high_threshold,
                     config.paper_detection.min_area_ratio * 100.0);
            eprintln!("  Swatch detection: ΔE ≥ {:.0}, area {:.0}%-{:.0}%",
                     config.swatch_detection.min_delta_e,
                     config.swatch_detection.min_area_ratio * 100.0,
                     config.swatch_detection.max_area_ratio * 100.0);
            eprintln!("  Color extraction: {}, outliers {:.0}%-{:.0}%",
                     config.color_extraction.method,
                     config.color_extraction.outlier_percentile_low,
                     config.color_extraction.outlier_percentile_high);
        }
        Err(e) => {
            eprintln!("Error saving config: {}", e);
            process::exit(1);
        }
    }
}
