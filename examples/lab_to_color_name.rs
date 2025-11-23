#!/usr/bin/env cargo
//! Simple helper to convert Lab values to ISCC-NBS color names using munsellspace
//! Usage: cargo run --example lab_to_color_name -- <L> <a> <b>

use munsellspace::{MunsellConverter, IsccNbsClassifier};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() != 4 {
        eprintln!("Usage: {} <L> <a> <b>", args[0]);
        eprintln!("Example: {} 45.0 -5.2 -15.8", args[0]);
        std::process::exit(1);
    }

    let l: f32 = args[1].parse().expect("Invalid L value");
    let a: f32 = args[2].parse().expect("Invalid a value");
    let b: f32 = args[3].parse().expect("Invalid b value");

    // Convert Lab to sRGB (munsellspace expects sRGB)
    // Lab -> XYZ -> sRGB conversion
    let (x, y, z) = lab_to_xyz(l, a, b);
    let (r, g, b_val) = xyz_to_srgb(x, y, z);

    // Clamp to valid sRGB range and convert to u8
    let r = (r.max(0.0).min(1.0) * 255.0).round() as u8;
    let g = (g.max(0.0).min(1.0) * 255.0).round() as u8;
    let b_val = (b_val.max(0.0).min(1.0) * 255.0).round() as u8;

    // Convert to Munsell using munsellspace
    let munsell_result = MunsellConverter::new()
        .and_then(|converter| converter.srgb_to_munsell([r, g, b_val]));

    match munsell_result {
        Ok(munsell_color) => {
            // Try to get ISCC-NBS classification
            let color_name = match (&munsell_color.hue, munsell_color.chroma) {
                (Some(hue), Some(chroma)) => {
                    IsccNbsClassifier::new()
                        .ok()
                        .and_then(|classifier| {
                            classifier.classify_munsell(hue.as_str(), munsell_color.value, chroma).ok()
                        })
                        .flatten()
                        .map(|metadata| metadata.iscc_nbs_descriptor())
                        .unwrap_or_else(|| "N/A".to_string())
                }
                _ => "N/A".to_string(),
            };
            println!("{}", color_name);
        }
        Err(_) => {
            println!("N/A");
        }
    }
}

fn lab_to_xyz(l: f32, a: f32, b: f32) -> (f32, f32, f32) {
    // D65 illuminant
    let x_n = 0.95047;
    let y_n = 1.0;
    let z_n = 1.08883;

    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b / 200.0;

    let epsilon = 0.008856;
    let kappa = 903.3;

    let xr = if fx.powi(3) > epsilon {
        fx.powi(3)
    } else {
        (116.0 * fx - 16.0) / kappa
    };

    let yr = if l > kappa * epsilon {
        fy.powi(3)
    } else {
        l / kappa
    };

    let zr = if fz.powi(3) > epsilon {
        fz.powi(3)
    } else {
        (116.0 * fz - 16.0) / kappa
    };

    let x = xr * x_n;
    let y = yr * y_n;
    let z = zr * z_n;

    (x, y, z)
}

fn xyz_to_srgb(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    // D65 XYZ to sRGB transformation matrix
    let r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314;
    let g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560;
    let b = x * 0.0556434 + y * -0.2040259 + z * 1.0572252;

    // Apply gamma correction
    let r = srgb_gamma(r);
    let g = srgb_gamma(g);
    let b = srgb_gamma(b);

    (r, g, b)
}

fn srgb_gamma(linear: f32) -> f32 {
    if linear <= 0.0031308 {
        12.92 * linear
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}
