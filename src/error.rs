//! Error types for the scan_colors library

use thiserror::Error;

/// Result type alias for scan_colors operations
pub type Result<T> = std::result::Result<T, AnalysisError>;

/// Comprehensive error types for color analysis operations
#[derive(Error, Debug)]
pub enum AnalysisError {
    /// Image file could not be loaded or decoded
    #[error("Failed to load image: {message}")]
    ImageLoadError {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// EXIF metadata extraction failed
    #[error("EXIF processing error: {message}")]
    ExifError {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Paper/card surface could not be detected
    #[error("Paper detection failed: {reason}")]
    PaperDetectionError { reason: String },

    /// Ink swatch area could not be identified
    #[error("Swatch detection failed: {reason}")]
    SwatchDetectionError { reason: String },

    /// Swatch area too small for reliable color analysis
    #[error("Swatch area insufficient: {area_percentage:.1}% of image (minimum {minimum:.1}%)")]
    InsufficientSwatchArea {
        area_percentage: f32,
        minimum: f32,
    },

    /// White balance estimation failed
    #[error("White balance estimation failed: {reason}")]
    WhiteBalanceError { reason: String },

    /// Color space conversion error
    #[error("Color conversion error: {message}")]
    ColorConversionError { message: String },

    /// Generic processing error
    #[error("Processing error: {message}")]
    ProcessingError { message: String },

    /// Invalid input parameters
    #[error("Invalid parameter: {parameter} = {value}")]
    InvalidParameter { parameter: String, value: String },

    /// OpenCV operation failed
    #[error("OpenCV error: {operation}")]
    OpenCvError {
        operation: String,
        #[source]
        source: Option<opencv::Error>,
    },

    /// Performance constraint violation
    #[error("Performance constraint violated: {operation} took {duration_ms}ms (limit: {limit_ms}ms)")]
    PerformanceError {
        operation: String,
        duration_ms: u64,
        limit_ms: u64,
    },
}

impl AnalysisError {
    /// Create an image load error with context
    pub fn image_load<E>(message: impl Into<String>, source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::ImageLoadError {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    /// Create an EXIF processing error with context
    pub fn exif<E>(message: impl Into<String>, source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::ExifError {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    /// Create an OpenCV error with context
    pub fn opencv(operation: impl Into<String>, source: opencv::Error) -> Self {
        Self::OpenCvError {
            operation: operation.into(),
            source: Some(source),
        }
    }

    /// Check if this error indicates a recoverable condition
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            AnalysisError::WhiteBalanceError { .. }
                | AnalysisError::InsufficientSwatchArea { .. }
                | AnalysisError::PerformanceError { .. }
        )
    }

    /// Get user-friendly error description for application display
    pub fn user_message(&self) -> String {
        match self {
            AnalysisError::ImageLoadError { .. } => {
                "Could not load the image. Please check the file format and try again.".to_string()
            }
            AnalysisError::PaperDetectionError { .. } => {
                "Could not detect paper in the image. Please ensure the swatch is on a clear background.".to_string()
            }
            AnalysisError::SwatchDetectionError { .. } => {
                "Could not detect ink swatch. Please ensure the ink area is clearly visible.".to_string()
            }
            AnalysisError::InsufficientSwatchArea { area_percentage, minimum } => {
                format!(
                    "Ink swatch is too small ({:.1}% of image). Please use a larger swatch (minimum {:.1}%).",
                    area_percentage, minimum
                )
            }
            AnalysisError::WhiteBalanceError { .. } => {
                "Could not determine proper lighting conditions. Please ensure adequate lighting.".to_string()
            }
            _ => "Color analysis failed. Please try with a different image.".to_string(),
        }
    }
}