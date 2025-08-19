//! White balance and color calibration module
//!
//! This module handles illuminant estimation and chromatic adaptation
//! to ensure consistent color measurement across different lighting conditions.

pub mod white_balance;
pub mod illuminant;

pub use white_balance::WhiteBalanceEstimator;
pub use illuminant::IlluminantEstimator;