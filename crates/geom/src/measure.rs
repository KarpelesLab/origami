use crate::Vec3;

pub fn distance(a: Vec3, b: Vec3) -> f64 {
    (b - a).norm()
}

/// Bond angle ∠ABC at vertex B, in radians (range [0, π]).
pub fn angle(a: Vec3, b: Vec3, c: Vec3) -> f64 {
    let ba = (a - b).normalize();
    let bc = (c - b).normalize();
    ba.dot(&bc).clamp(-1.0, 1.0).acos()
}

/// Dihedral angle A-B-C-D in radians (range [-π, π]).
/// IUPAC sign convention: looking down B→C, positive = D rotates
/// counter-clockwise from A (right-handed).
pub fn dihedral(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> f64 {
    // Praxeolitic / IUPAC formulation: b0 points A→B (i.e. negated).
    let b0 = a - b;
    let b1 = (c - b).normalize();
    let b2 = d - c;
    // Project b0 and b2 onto the plane perpendicular to b1.
    let v = b0 - b1 * b0.dot(&b1);
    let w = b2 - b1 * b2.dot(&b1);
    let x = v.dot(&w);
    let y = b1.cross(&v).dot(&w);
    y.atan2(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn distance_basic() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(3.0, 4.0, 0.0);
        assert_relative_eq!(distance(a, b), 5.0, epsilon = 1e-12);
    }

    #[test]
    fn right_angle() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 1.0, 0.0);
        assert_relative_eq!(angle(a, b, c), PI / 2.0, epsilon = 1e-12);
    }

    #[test]
    fn dihedral_trans_is_pi() {
        // Planar trans configuration: A-B-C-D in same plane, D on opposite
        // side of BC from A.
        let a = Vec3::new(0.0, 1.0, 0.0);
        let b = Vec3::new(0.0, 0.0, 0.0);
        let c = Vec3::new(1.0, 0.0, 0.0);
        let d = Vec3::new(1.0, -1.0, 0.0);
        assert_relative_eq!(dihedral(a, b, c, d).abs(), PI, epsilon = 1e-12);
    }

    #[test]
    fn dihedral_cis_is_zero() {
        // Planar cis: D on same side of BC as A.
        let a = Vec3::new(0.0, 1.0, 0.0);
        let b = Vec3::new(0.0, 0.0, 0.0);
        let c = Vec3::new(1.0, 0.0, 0.0);
        let d = Vec3::new(1.0, 1.0, 0.0);
        assert_relative_eq!(dihedral(a, b, c, d), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn dihedral_perpendicular_is_half_pi() {
        let a = Vec3::new(0.0, 1.0, 0.0);
        let b = Vec3::new(0.0, 0.0, 0.0);
        let c = Vec3::new(1.0, 0.0, 0.0);
        let d = Vec3::new(1.0, 0.0, 1.0);
        assert_relative_eq!(dihedral(a, b, c, d), PI / 2.0, epsilon = 1e-12);
    }
}
