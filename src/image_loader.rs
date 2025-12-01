//! Unified image loading with support for multiple formats
//!
//! This module provides a single entry point for loading images from various formats
//! including JPEG, PNG, TIFF, WebP, GIF, BMP, and HEIC/HEIF.
//!
//! ## Supported Formats
//!
//! Standard formats (via `image` crate):
//! - JPEG, PNG, GIF, WebP, TIFF, BMP, ICO, TGA, EXR, PNM, QOI, DDS, HDR
//!
//! Apple formats (via `libheif-rs`):
//! - HEIC, HEIF
//!
//! ## Design
//!
//! The loader converts all images to OpenCV Mat in BGR format for consistent
//! downstream processing. EXIF orientation is NOT applied here - that's handled
//! separately in the pipeline to allow access to the original orientation data.

use crate::error::{AnalysisError, Result};
use opencv::core::Mat;
use std::path::Path;

/// Supported image formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    /// JPEG image
    Jpeg,
    /// PNG image
    Png,
    /// GIF image (first frame only)
    Gif,
    /// WebP image
    WebP,
    /// TIFF image
    Tiff,
    /// BMP image
    Bmp,
    /// ICO image
    Ico,
    /// TGA image
    Tga,
    /// OpenEXR image
    Exr,
    /// PNM image (PBM, PGM, PPM)
    Pnm,
    /// QOI image
    Qoi,
    /// DDS image
    Dds,
    /// HDR image
    Hdr,
    /// HEIC/HEIF image (Apple)
    Heic,
    /// AVIF image
    Avif,
}

impl ImageFormat {
    /// Detect format from file extension
    pub fn from_extension(path: &Path) -> Option<ImageFormat> {
        let ext = path.extension()?.to_str()?.to_lowercase();
        match ext.as_str() {
            "jpg" | "jpeg" => Some(ImageFormat::Jpeg),
            "png" => Some(ImageFormat::Png),
            "gif" => Some(ImageFormat::Gif),
            "webp" => Some(ImageFormat::WebP),
            "tiff" | "tif" => Some(ImageFormat::Tiff),
            "bmp" => Some(ImageFormat::Bmp),
            "ico" => Some(ImageFormat::Ico),
            "tga" => Some(ImageFormat::Tga),
            "exr" => Some(ImageFormat::Exr),
            "pbm" | "pgm" | "ppm" | "pnm" => Some(ImageFormat::Pnm),
            "qoi" => Some(ImageFormat::Qoi),
            "dds" => Some(ImageFormat::Dds),
            "hdr" => Some(ImageFormat::Hdr),
            "heic" | "heif" => Some(ImageFormat::Heic),
            "avif" => Some(ImageFormat::Avif),
            _ => None,
        }
    }

    /// Check if format requires libheif
    pub fn requires_heif(&self) -> bool {
        matches!(self, ImageFormat::Heic)
    }
}

/// Load an image from disk and convert to OpenCV Mat (BGR format)
///
/// This function automatically detects the image format and uses the appropriate
/// decoder. For HEIC/HEIF files, it uses libheif. For all other formats, it uses
/// the `image` crate.
///
/// # Arguments
///
/// * `path` - Path to the image file
///
/// # Returns
///
/// OpenCV Mat in BGR format (standard OpenCV color format)
///
/// # Errors
///
/// Returns `AnalysisError::ImageLoad` if:
/// - File cannot be opened
/// - Format is not supported
/// - Decoding fails
///
/// # Example
///
/// ```rust,no_run
/// use scan_colors::image_loader::load_image;
/// use std::path::Path;
///
/// let mat = load_image(Path::new("photo.jpg"))?;
/// println!("Loaded image: {}x{}", mat.cols(), mat.rows());
/// # Ok::<(), scan_colors::AnalysisError>(())
/// ```
pub fn load_image(path: &Path) -> Result<Mat> {
    // Detect format from extension
    let format = ImageFormat::from_extension(path).ok_or_else(|| {
        AnalysisError::ProcessingError(format!(
            "Unknown image format for file: {}",
            path.display()
        ))
    })?;

    if format.requires_heif() {
        load_heic(path)
    } else {
        load_standard(path)
    }
}

/// Load image using the `image` crate (standard formats)
fn load_standard(path: &Path) -> Result<Mat> {
    use image::ImageReader;
    use image::DynamicImage;

    // Open and decode image
    let reader = ImageReader::open(path).map_err(|e| {
        AnalysisError::image_load(
            &format!("Failed to open image file: {}", path.display()),
            e,
        )
    })?;

    let img: DynamicImage = reader.decode().map_err(|e| {
        AnalysisError::image_load(
            &format!("Failed to decode image: {}", path.display()),
            e,
        )
    })?;

    // Convert to RGB8
    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();

    // Convert RGB to BGR for OpenCV
    rgb_to_bgr_mat(&rgb_img.into_raw(), width as i32, height as i32)
}

/// Load HEIC/HEIF image using libheif
fn load_heic(path: &Path) -> Result<Mat> {
    use libheif_rs::{HeifContext, RgbChroma, ColorSpace, LibHeif};

    // Create LibHeif instance
    let lib_heif = LibHeif::new();

    // Load HEIF container
    let ctx = HeifContext::read_from_file(path.to_str().ok_or_else(|| {
        AnalysisError::ProcessingError("Invalid file path encoding".into())
    })?)
    .map_err(|e| {
        AnalysisError::image_load(
            &format!("Failed to read HEIC file: {}", path.display()),
            e,
        )
    })?;

    // Get primary image handle
    let handle = ctx.primary_image_handle().map_err(|e| {
        AnalysisError::image_load("Failed to get primary image handle", e)
    })?;

    // Decode to RGB using LibHeif instance
    let image = lib_heif
        .decode(&handle, ColorSpace::Rgb(RgbChroma::Rgb), None)
        .map_err(|e| {
            AnalysisError::image_load("Failed to decode HEIC image", e)
        })?;

    // Get RGB plane
    let planes = image.planes();
    let rgb_plane = planes.interleaved.ok_or_else(|| {
        AnalysisError::ProcessingError("HEIC image has no interleaved RGB data".into())
    })?;

    let width = handle.width() as i32;
    let height = handle.height() as i32;
    let stride = rgb_plane.stride;
    let data = rgb_plane.data;

    // Handle potential stride padding (copy row by row)
    let expected_row_bytes = (width * 3) as usize;
    let rgb_data: Vec<u8> = if stride as usize == expected_row_bytes {
        data.to_vec()
    } else {
        // Copy row by row, skipping padding
        let mut result = Vec::with_capacity((width * height * 3) as usize);
        for row in 0..height as usize {
            let row_start = row * stride as usize;
            let row_end = row_start + expected_row_bytes;
            result.extend_from_slice(&data[row_start..row_end]);
        }
        result
    };

    // Convert RGB to BGR for OpenCV
    rgb_to_bgr_mat(&rgb_data, width, height)
}

/// Convert RGB byte buffer to OpenCV BGR Mat
fn rgb_to_bgr_mat(rgb_data: &[u8], width: i32, height: i32) -> Result<Mat> {
    use opencv::core::{Mat, CV_8UC3, Vec3b};
    use opencv::prelude::{MatExprTraitConst, MatTrait};

    // Create OpenCV Mat
    let mut mat = Mat::zeros(height, width, CV_8UC3)
        .map_err(|e| AnalysisError::ProcessingError(format!("Failed to create Mat: {}", e)))?
        .to_mat()
        .map_err(|e| AnalysisError::ProcessingError(format!("Mat conversion failed: {}", e)))?;

    // Copy data with RGB to BGR conversion
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 3) as usize;
            let r = rgb_data[idx];
            let g = rgb_data[idx + 1];
            let b = rgb_data[idx + 2];

            // BGR order for OpenCV
            let pixel = mat.at_2d_mut::<Vec3b>(y, x).map_err(|e| {
                AnalysisError::ProcessingError(format!("Failed to access pixel: {}", e))
            })?;
            pixel[0] = b;
            pixel[1] = g;
            pixel[2] = r;
        }
    }

    Ok(mat)
}

/// Get list of all supported file extensions
pub fn supported_extensions() -> &'static [&'static str] {
    &[
        "jpg", "jpeg", "png", "gif", "webp", "tiff", "tif", "bmp", "ico",
        "tga", "exr", "pbm", "pgm", "ppm", "pnm", "qoi", "dds", "hdr",
        "heic", "heif", "avif",
    ]
}

/// Check if a file extension is supported
pub fn is_supported_extension(ext: &str) -> bool {
    let ext_lower = ext.to_lowercase();
    supported_extensions().contains(&ext_lower.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_detection() {
        assert_eq!(
            ImageFormat::from_extension(Path::new("photo.jpg")),
            Some(ImageFormat::Jpeg)
        );
        assert_eq!(
            ImageFormat::from_extension(Path::new("photo.JPEG")),
            Some(ImageFormat::Jpeg)
        );
        assert_eq!(
            ImageFormat::from_extension(Path::new("photo.png")),
            Some(ImageFormat::Png)
        );
        assert_eq!(
            ImageFormat::from_extension(Path::new("photo.heic")),
            Some(ImageFormat::Heic)
        );
        assert_eq!(
            ImageFormat::from_extension(Path::new("photo.HEIF")),
            Some(ImageFormat::Heic)
        );
        assert_eq!(
            ImageFormat::from_extension(Path::new("photo.webp")),
            Some(ImageFormat::WebP)
        );
        assert_eq!(
            ImageFormat::from_extension(Path::new("photo.xyz")),
            None
        );
    }

    #[test]
    fn test_heif_requirement() {
        assert!(ImageFormat::Heic.requires_heif());
        assert!(!ImageFormat::Jpeg.requires_heif());
        assert!(!ImageFormat::Png.requires_heif());
    }

    #[test]
    fn test_supported_extensions() {
        assert!(is_supported_extension("jpg"));
        assert!(is_supported_extension("JPEG"));
        assert!(is_supported_extension("png"));
        assert!(is_supported_extension("heic"));
        assert!(is_supported_extension("HEIF"));
        assert!(!is_supported_extension("xyz"));
        assert!(!is_supported_extension("doc"));
    }

    #[test]
    fn test_rgb_to_bgr_conversion() {
        use opencv::prelude::MatTraitConst;

        // Test 2x2 image: red, green, blue, white
        let rgb_data = vec![
            255, 0, 0,    // Red
            0, 255, 0,    // Green
            0, 0, 255,    // Blue
            255, 255, 255 // White
        ];

        let mat = rgb_to_bgr_mat(&rgb_data, 2, 2).unwrap();

        // Check BGR values
        let pixel_0_0: &opencv::core::Vec3b = mat.at_2d(0, 0).unwrap();
        assert_eq!(pixel_0_0[0], 0);   // B
        assert_eq!(pixel_0_0[1], 0);   // G
        assert_eq!(pixel_0_0[2], 255); // R

        let pixel_0_1: &opencv::core::Vec3b = mat.at_2d(0, 1).unwrap();
        assert_eq!(pixel_0_1[0], 0);   // B
        assert_eq!(pixel_0_1[1], 255); // G
        assert_eq!(pixel_0_1[2], 0);   // R

        let pixel_1_0: &opencv::core::Vec3b = mat.at_2d(1, 0).unwrap();
        assert_eq!(pixel_1_0[0], 255); // B
        assert_eq!(pixel_1_0[1], 0);   // G
        assert_eq!(pixel_1_0[2], 0);   // R

        let pixel_1_1: &opencv::core::Vec3b = mat.at_2d(1, 1).unwrap();
        assert_eq!(pixel_1_1[0], 255); // B
        assert_eq!(pixel_1_1[1], 255); // G
        assert_eq!(pixel_1_1[2], 255); // R
    }
}
