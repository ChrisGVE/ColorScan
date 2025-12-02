# Test Assets

This directory contains test images for validating the `inkswatch_colorscan` library.

## Directory Structure

```
assets/
├── strip_geolocation.zsh    # Script to remove GPS data from images
├── local_test_samples/       # Private test images (gitignored - your personal photos)
├── test_samples/             # Public test images (committed - for CI/CD workflows)
├── references/               # Reference materials
└── README.md                 # This file
```

### Directory Purpose

- **`local_test_samples/`**: Your personal test images with potentially sensitive data
  - ✅ Gitignored - safe to add your own photos
  - ⚠️ May contain geolocation data - use `strip_geolocation.zsh` before sharing
  - 💡 Used for local development and testing

- **`test_samples/`**: Curated public test images for CI/CD
  - ✅ Committed to repository
  - ✅ GPS-stripped and privacy-safe
  - ✅ Used by integration tests and CI workflows
  - 💡 Should match test specifications in `tests/README.md`

## Adding New Test Images

### For Local Testing (Private Images)

When adding images to `local_test_samples/`:

1. **Add your images** directly - they're gitignored
   ```bash
   cp ~/Photos/my_swatch.jpg assets/local_test_samples/
   ```

2. **Navigate to the directory**
   ```bash
   cd assets/local_test_samples
   ```

### For CI/CD (Public Images)

When adding images to `test_samples/` for continuous integration:

1. **Start in `local_test_samples/`** (process there first)
   ```bash
   cp ~/Photos/curated_swatch.jpg assets/local_test_samples/
   cd assets/local_test_samples
   ```

2. **Strip GPS data** before moving to public directory
   ```bash
   ../strip_geolocation.zsh
   ```

3. **Verify removal**
   ```bash
   exiftool -gps:all curated_swatch.jpg
   # Should return nothing
   ```

4. **Move to public directory**
   ```bash
   mv curated_swatch.jpg ../test_samples/
   ```

5. **Commit to repository**
   ```bash
   cd ../..
   git add assets/test_samples/curated_swatch.jpg
   git commit -m "test: add curated_swatch.jpg test image"
   ```

## Script Usage

### Basic Usage
```bash
# From test_samples directory
../strip_geolocation.zsh
```

### Options
```bash
# Dry run - check what would be removed without making changes
../strip_geolocation.zsh --dry-run

# Show help
../strip_geolocation.zsh --help
```

## What Gets Removed

The script removes all geolocation data:
- GPS coordinates (latitude, longitude, altitude)
- GPS direction and bearing
- GPS speed and timestamp
- Location names and place information

## What Gets Preserved

Camera metadata useful for color analysis testing:
- Camera make and model (e.g., "Apple iPhone 16 Pro Max")
- White balance settings (e.g., "Auto")
- Color space information (e.g., "Uncalibrated")
- Exposure settings (ISO, shutter speed, aperture)
- Date/time stamps
- Lens information

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- HEIC (.heic)
- PNG (.png)
- TIFF (.tiff)
- RAW formats (.dng, .cr2, .nef)

## Why Remove GPS Data?

Geolocation data can reveal:
- Your home address
- Workplace location
- Travel patterns
- Time zones

**Always remove GPS data before committing images to public repositories.**

## Integration Tests

Test images in `test_samples/` are used by integration tests in `tests/integration_test.rs`. See `tests/README.md` for test image requirements and specifications.

## Manual Verification

To check specific GPS data in an image:
```bash
# Show all GPS tags
exiftool -gps:all image.jpg

# Show GPS coordinates only
exiftool -GPSLatitude -GPSLongitude image.jpg

# Show all metadata
exiftool image.jpg
```

## Security Best Practices

1. **Never commit unprocessed images** with GPS data
2. **Use `local_test_samples/`** for private images (gitignored)
3. **Always run the strip script** before moving to `test_samples/`
4. **Use `--dry-run`** first to verify what will be removed
5. **The `local_test_samples/` directory is gitignored** to prevent accidental commits
6. **Only commit GPS-stripped images** to `test_samples/` for CI/CD
7. **Consider using a pre-commit hook** to verify GPS data is removed

## Example Workflow

### Local Development Testing

```bash
# 1. Add new test images to local directory
cp ~/Downloads/ink_swatch_*.jpg assets/local_test_samples/

# 2. Test with local images
cargo test --test integration_test -- --ignored

# 3. No git tracking - your images stay private
```

### Preparing Images for CI/CD

```bash
# 1. Add curated images to local directory first
cp ~/Downloads/ink_swatch_blue.jpg assets/local_test_samples/

# 2. Remove GPS data
cd assets/local_test_samples
../strip_geolocation.zsh

# 3. Verify GPS removal
exiftool -gps:all ink_swatch_blue.jpg

# 4. Move to public directory and rename appropriately
mv ink_swatch_blue.jpg ../test_samples/uniform_blue.jpg

# 5. Commit for CI/CD use
cd ../..
git add assets/test_samples/uniform_blue.jpg
git commit -m "test: add uniform blue ink test image"

# 6. CI workflows can now use this image
```

## Troubleshooting

**Script fails with "exiftool not found"**
```bash
brew install exiftool
```

**Want to process a single file?**
```bash
exiftool -gps:all= -overwrite_original image.jpg
```

**Want to keep backup copies?**
```bash
# Remove -overwrite_original flag (creates .jpg_original files)
exiftool -gps:all= image.jpg
```

**Want to see what metadata exists?**
```bash
# Show all EXIF data
exiftool -a image.jpg

# Show specific tags
exiftool -Make -Model -WhiteBalance -ISO image.jpg
```
