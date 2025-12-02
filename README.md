# InkSwatch ColorScan

A Rust crate for analyzing fountain pen ink colors from digital photographs with calibrated color measurement.

## Features

- **Color Measurement**: CIE Lab and LCh color spaces for perceptually uniform analysis
- **D65 Calibration**: Normalizes colors to D65 illuminant
- **Munsell & ISCC-NBS Names**: Converts colors to Munsell notation and descriptive color names
- **HEIC/HEIF Support**: Loads iPhone photos via libheif (optional dependency)
- **Multiple Image Formats**: Supports JPEG, PNG, TIFF, WebP, HEIC, AVIF, and others
- **Smartphone Compatible**: Handles varying lighting and camera characteristics
- **EXIF Orientation**: Automatic image rotation based on EXIF metadata
- **White Balance**: Paper band sampling for color correction
- **JSON Configuration**: Pipeline parameters configurable via JSON
- **Confidence Scoring**: Provides reliability metrics for measurements

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
inkswatch_colorscan = "0.1"
```

### Library Usage

```rust
use inkswatch_colorscan::{analyze_swatch, ColorResult};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let result = analyze_swatch(Path::new("fountain_pen_swatch.jpg"))?;

    println!("Hex: {}", result.hex);                    // #3B5998
    println!("Munsell: {}", result.munsell);            // 7.5PB 4/8
    println!("Color name: {}", result.color_name);      // dark purplish blue
    println!("Base color: {}", result.base_color);      // blue
    println!("Lab: L*={:.1}, a*={:.1}, b*={:.1}",
             result.lab.l, result.lab.a, result.lab.b); // L*=40.2, a*=5.1, b*=-32.4
    println!("Confidence: {:.1}%", result.confidence * 100.0);

    Ok(())
}
```

### With Configuration

```rust
use inkswatch_colorscan::{analyze_swatch_debug_with_config, PipelineConfig};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load configuration from JSON
    let config = PipelineConfig::from_json_file(Path::new("config.json"))?;

    // Analyze with debug output (intermediate images)
    let (result, debug) = analyze_swatch_debug_with_config(
        Path::new("swatch.heic"),
        &config
    )?;

    // Access debug images: debug.original_image, debug.corrected_image,
    // debug.swatch_fragment, debug.swatch_mask

    Ok(())
}
```

### Command Line Usage

```bash
# Analyze a single image
cargo run --example cli photo.jpg

# Analyze with JSON output
cargo run --example cli swatch.heic --json

# Batch processing with configuration file
cargo run --example cli_batch config.json

# Batch with CSV output
cargo run --example cli_batch config.json --csv results.csv
```

**Batch Configuration** (`config.json`):
```json
{
  "input_path": "/path/to/photos/",
  "output_path": "/path/to/output/images/",
  "file_pattern": "*.heic",
  "white_balance": { "enabled": true },
  "swatch_detection": { "luminance_threshold": 0.85 }
}
```

**Note**: The CLI tools are **development and testing utilities only**:
- Dynamically linked to system OpenCV for fast compile times
- Not intended for distribution to end users
- Used for validating crate functionality and iterating on algorithms

For production applications, see the Deployment section below.

## How It Works

1. **Paper Detection**: Locates and rectifies the paper/card surface using edge detection
2. **White Balance**: Estimates illuminant from paper region and applies D65 normalization
3. **Swatch Isolation**: Detects ink regions and separates from background
4. **Color Extraction**: Computes representative color using statistical methods
5. **Color Naming**: Converts Lab values to Munsell notation and ISCC-NBS color names

## Technical Details

### Color Spaces

- **Input Processing**: sRGB from camera with EXIF-based corrections
- **Analysis**: CIE Lab for perceptual uniformity and color difference calculations
- **Output**: Lab, LCh, sRGB, and hex representations

### Calibration Standards

- **Reference Illuminant**: D65 (6504K) standard illuminant
- **Color Difference**: Uses CIEDE2000 for perceptual color comparisons
- **White Balance**: Designed for indoor lighting (3000K-6500K range)

### Performance

- **Processing Time**: Typically under 100ms per image on modern hardware
- **Format Support**: JPEG, PNG, TIFF, WebP, HEIC/HEIF, AVIF, BMP, GIF, ICO, OpenEXR, PNM, QOI, TGA, and others

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

- **libheif** (Optional): Required for HEIC/HEIF image support (iPhone photos)
  ```bash
  # macOS
  brew install libheif

  # Ubuntu/Debian
  sudo apt install libheif-dev

  # Windows
  # See libheif documentation
  ```
  Note: Without libheif, the crate still supports 20+ other formats including JPEG, PNG, TIFF, and WebP.

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

2. **libheif** (Optional) - HEIC/HEIF image format support
   - Required only for iPhone HEIC photo support
   - Must be discoverable via `pkg-config --exists libheif`
   - Without this, all other formats still work

3. **pkg-config** - Build configuration discovery
   - Used for OpenCV flags: `pkg-config --cflags --libs opencv4`
   - Handles include paths and linking automatically

4. **C++ Compiler** - For OpenCV Rust bindings
   - Clang (preferred) or GCC
   - Required for bindgen code generation

### Verified Configurations

**macOS (Homebrew)**
```bash
brew install opencv pkg-config libheif
# OpenCV 4.12.0 confirmed working
# libheif 1.19+ for HEIC support
```

**Build Verification**
```bash
# Test OpenCV detection
pkg-config --exists opencv4 && echo "OpenCV found"
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

## Deployment Options

### For Library Users (Recommended)

When using `inkswatch_colorscan` as a library dependency in your application, you have several OpenCV deployment options:

**1. Dynamic Linking** (Development/Testing)
- Default configuration - links to system-installed OpenCV
- Fast compile times, smaller binaries
- Users must have OpenCV installed on their system
- Current crate configuration

**2. Static Linking** (Standalone Applications)
```toml
[dependencies]
opencv = { version = "0.95", features = ["static"] }
```
- Embeds OpenCV into your application binary
- Single self-contained executable (~50-100MB)
- No runtime OpenCV dependency required
- Ideal for CLI tools distributed to end users

**3. Bundled Libraries** (Desktop Applications)
- Ship OpenCV dynamic libraries alongside your application
- Smaller main binary, libraries loaded at runtime
- Common for macOS .app bundles, Windows installers
- Platform-specific packaging required

**4. vcpkg Integration** (Reproducible Builds)
```bash
vcpkg install opencv4
export VCPKG_ROOT=/path/to/vcpkg
cargo build
```
- Builds OpenCV as part of your build process
- Reproducible across development environments
- opencv-rust has built-in vcpkg support

**Recommendation**: Document OpenCV as a system dependency and let consuming applications choose their deployment strategy based on their distribution requirements.

## Development Status

**Current Version**: 0.1.0 (Early Development)

This crate is functional but still under active development. The API may change between versions.

### Implemented
- Project structure and dependencies
- Core types and error handling
- D65 calibration constants and color spaces
- CLI interface (single and batch processing)
- Image loader with multiple format support
- HEIC/HEIF support (requires libheif)
- EXIF orientation handling
- Paper/swatch detection
- White balance estimation
- Color analysis pipeline
- Munsell notation conversion
- ISCC-NBS color naming
- JSON configuration system
- Unit tests

### Known Limitations
- Detection may fail on complex backgrounds or unusual lighting
- White balance assumes neutral paper background
- Not tested with all smartphone camera models

### Planned
- Multi-tone shading detection
- API refinement
- Additional camera profile support

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

This crate uses the following dependencies:
- [image](https://crates.io/crates/image) - Image decoding
- [libheif-rs](https://crates.io/crates/libheif-rs) - HEIC/HEIF support
- [opencv](https://crates.io/crates/opencv) - Computer vision
- [palette](https://crates.io/crates/palette) - Color space conversions
- [empfindung](https://crates.io/crates/empfindung) - CIEDE2000 color differences
- [serde](https://crates.io/crates/serde) - Serialization