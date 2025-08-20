# Scan Colors

A Rust crate for analyzing fountain pen ink colors from digital photographs with calibrated color measurement.

## Features

- **Accurate Color Measurement**: Uses CIE Lab and LCh color spaces for perceptually uniform analysis
- **D65 Calibration**: Normalizes colors to industry-standard D65 illuminant
- **Smartphone Compatible**: Optimized for modern smartphone camera images
- **EXIF Aware**: Leverages camera metadata for improved white balance
- **Fast Processing**: Target 100ms analysis time for typical images
- **Confidence Scoring**: Provides reliability metrics for color measurements

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
scan_colors = "0.1"
```

### Library Usage

```rust
use scan_colors::{analyze_swatch, ColorResult};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let result = analyze_swatch(Path::new("fountain_pen_swatch.jpg"))?;
    
    println!("Detected color: {}", result.hex);
    println!("Lab values: L*={:.1}, a*={:.1}, b*={:.1}", 
             result.lab.l, result.lab.a, result.lab.b);
    println!("Confidence: {:.1}%", result.confidence * 100.0);
    
    Ok(())
}
```

### Command Line Usage

```bash
# Analyze a single image
cargo run --example cli photo.jpg

# Output includes JSON for programmatic use
cargo run --example cli swatch.heic > result.json
```

## How It Works

1. **Paper Detection**: Locates and rectifies the paper/card surface using computer vision
2. **White Balance**: Estimates illuminant from paper region and applies D65 normalization
3. **Swatch Isolation**: Detects ink regions and separates from background
4. **Color Extraction**: Performs robust statistical analysis handling gradients and transparency
5. **Calibration**: Applies research-based corrections for accurate color representation

## Technical Details

### Color Spaces

- **Input Processing**: sRGB from camera with EXIF-based corrections
- **Analysis**: CIE Lab for perceptual uniformity and color difference calculations
- **Output**: Lab, LCh, sRGB, and hex representations

### Calibration Standards

- **Reference Illuminant**: D65 (6504K) - industry standard for digital images
- **Color Accuracy**: Target ΔE < 3.0 for fountain pen ink differentiation
- **White Balance**: Supports temperature range 3000K-6500K

### Performance

- **Target Speed**: 100ms analysis time for smartphone images
- **Memory Efficient**: Small footprint except during image processing
- **Format Support**: JPEG, PNG, HEIC, and other common formats

## Requirements

### System Dependencies

- **OpenCV**: Required for computer vision operations
  ```bash
  # macOS
  brew install opencv
  
  # Ubuntu/Debian
  sudo apt install libopencv-dev
  
  # Windows
  # See opencv-rust documentation
  ```

### Rust Version

- Minimum supported Rust version: 1.70.0
- Uses 2021 edition features

## Development Setup

### Environment Variables

The project automatically detects OpenCV through pkg-config. No additional environment variables are typically needed.

### Compilation Requirements

The following system components are required for building:

1. **OpenCV 4.x** - Computer vision library
   - Tested with OpenCV 4.12.0
   - Must be discoverable via `pkg-config --exists opencv4`
   - Required modules: core, imgproc, imgcodecs, photo

2. **pkg-config** - Build configuration discovery
   - Used for OpenCV flags: `pkg-config --cflags --libs opencv4`
   - Handles include paths and linking automatically

3. **C++ Compiler** - For OpenCV Rust bindings
   - Clang (preferred) or GCC
   - Required for bindgen code generation

### Verified Configurations

**macOS (Homebrew)**
```bash
brew install opencv pkg-config
# OpenCV 4.12.0 confirmed working
# Located at: /usr/local/opt/opencv/
```

**Build Verification**
```bash
# Test OpenCV detection
pkg-config --exists opencv4 && echo "✓ OpenCV found"
pkg-config --modversion opencv4  # Should show 4.x.x

# Test compilation
cargo check  # Should compile opencv crate successfully
```

### Troubleshooting

If compilation fails with OpenCV errors:

1. Verify OpenCV installation: `pkg-config --exists opencv4`
2. Check modules are available: `pkg-config --libs opencv4 | grep core`
3. Ensure pkg-config is in PATH
4. On macOS, verify Homebrew paths are correct

The opencv Rust crate (v0.95.1) uses buildtime-bindgen and should automatically:
- Detect system OpenCV via pkg-config
- Generate appropriate Rust bindings
- Handle linking flags

## Image Guidelines

For best results:

- **Swatch Size**: Ink area should occupy 10-15% of total image
- **Background**: Place swatch on white or light-colored paper
- **Lighting**: Use even lighting, avoid harsh shadows
- **Focus**: Ensure ink area is sharp and well-focused
- **Distance**: Close enough to see ink texture, far enough to include paper margin

## Error Handling

The library provides detailed error information:

```rust
match analyze_swatch(path) {
    Ok(result) => println!("Color: {}", result.hex),
    Err(error) => {
        eprintln!("Analysis failed: {}", error);
        
        // User-friendly error messages
        if error.is_recoverable() {
            eprintln!("Try: {}", error.user_message());
        }
    }
}
```

## Development Status

**Current Version**: 0.1.0 (Initial Development)

### Implemented
- [x] Project structure and dependencies
- [x] Core types and error handling
- [x] D65 calibration constants
- [x] Module architecture
- [x] CLI interface

### In Progress
- [ ] EXIF metadata extraction
- [ ] Paper detection algorithms
- [ ] White balance estimation
- [ ] Swatch isolation
- [ ] Color analysis pipeline

### Planned
- [ ] Performance optimization
- [ ] Comprehensive testing
- [ ] Documentation examples
- [ ] Camera profile integration

## Contributing

This project is in early development. Contributions welcome!

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Research References

- CIE 15:2004 Colorimetry, 3rd edition
- Adobe DNG Specification
- "Color Science" by Wyszecki & Stiles
- CIEDE2000 color difference formula
- Smartphone camera calibration research (2024)

## Acknowledgments

Built on excellent Rust ecosystem crates:
- [image](https://crates.io/crates/image) - Image processing
- [opencv](https://crates.io/crates/opencv) - Computer vision
- [palette](https://crates.io/crates/palette) - Color science
- [empfindung](https://crates.io/crates/empfindung) - Color differences