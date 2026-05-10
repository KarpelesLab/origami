//! Unit conversions used at the boundary between CHARMM-style parameters
//! (kcal/mol, degrees) and origami's preferred SI-ish units (kJ/mol, radians).

/// Convert kcal/mol → kJ/mol. (Exact factor: 4.184 by definition.)
pub const KCAL_PER_MOL_TO_KJ_PER_MOL: f64 = 4.184;

#[inline]
pub fn kcal_to_kj(value_kcal: f64) -> f64 {
    value_kcal * KCAL_PER_MOL_TO_KJ_PER_MOL
}

#[inline]
pub fn deg_to_rad(deg: f64) -> f64 {
    deg * std::f64::consts::PI / 180.0
}
