use crate::Vec3;

/// Place a fourth atom D given three placed atoms A, B, C and the internal
/// coordinates: bond length |CD|, bond angle ∠BCD, and dihedral A-B-C-D.
///
/// All angles in radians. Length in Å.
///
/// This is the standard NeRF (Natural Extension Reference Frame) construction,
/// formulated for stability when A, B, C are not collinear. We build a local
/// frame at C: x̂ along BC (so D departs from C in the −x̂ direction), ŷ in the
/// plane of A-B-C, ẑ = x̂ × ŷ. Then D's offset in this frame is:
///   D − C = R · ( −L cos(θ),  L sin(θ) cos(φ),  L sin(θ) sin(φ) )
/// where θ is the bond angle, φ is the dihedral, R is the local-to-global
/// rotation, and L is the bond length.
pub fn place_atom(a: Vec3, b: Vec3, c: Vec3, length: f64, angle: f64, dihedral: f64) -> Vec3 {
    let bc = (c - b).normalize();
    let ba = a - b;
    // n = unit normal to the A-B-C plane.
    let n = bc.cross(&ba).normalize();
    // The local frame at C:
    //   x̂ = bc (along B → C)
    //   ẑ = n (out of plane)
    //   ŷ = ẑ × x̂ (in plane, perpendicular to bc)
    let x_hat = bc;
    let z_hat = n;
    let y_hat = z_hat.cross(&x_hat);
    let (sin_t, cos_t) = angle.sin_cos();
    let (sin_d, cos_d) = dihedral.sin_cos();
    let offset = -x_hat * (length * cos_t)
        + y_hat * (length * sin_t * cos_d)
        + z_hat * (length * sin_t * sin_d);
    c + offset
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    fn deg(x: f64) -> f64 {
        x * PI / 180.0
    }

    #[test]
    fn placing_a_fourth_atom_reproduces_inputs() {
        // Lay down three reference atoms in space, place a fourth, then
        // measure the resulting bond length, angle, and dihedral. They should
        // match the inputs exactly.
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.5, 0.0, 0.0);
        let c = Vec3::new(2.5, 1.2, 0.0);

        let length = 1.4;
        let angle = deg(110.0);
        let dihedral = deg(60.0);

        let d = place_atom(a, b, c, length, angle, dihedral);

        assert_relative_eq!((d - c).norm(), length, epsilon = 1e-9);

        let cb = (b - c).normalize();
        let cd = (d - c).normalize();
        let measured_angle = cb.dot(&cd).acos();
        assert_relative_eq!(measured_angle, angle, epsilon = 1e-9);

        let measured_dihedral = crate::measure::dihedral(a, b, c, d);
        assert_relative_eq!(measured_dihedral, dihedral, epsilon = 1e-9);
    }

    #[test]
    fn dihedral_zero_places_atom_in_abc_plane_cis() {
        // φ = 0 should put D on the same side of the BC line as A,
        // i.e. in the same plane as A-B-C.
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.5, 0.0, 0.0);
        let c = Vec3::new(2.5, 1.2, 0.0);
        let d = place_atom(a, b, c, 1.4, deg(110.0), 0.0);
        // All four should share z = 0.
        assert_relative_eq!(d.z, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn extending_a_straight_chain_with_180_dihedrals() {
        // A perfectly extended chain with bond angle 109.47° (tetrahedral)
        // and dihedral 180° should zigzag in a single plane.
        let mut atoms = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.5, 0.0, 0.0),
            Vec3::new(2.0, 1.41, 0.0),
        ];
        for _ in 0..5 {
            let n = atoms.len();
            let next = place_atom(atoms[n - 3], atoms[n - 2], atoms[n - 1], 1.5, deg(109.47), deg(180.0));
            atoms.push(next);
        }
        // All atoms share z = 0 (the chain stays in the original plane).
        for atom in &atoms {
            assert_relative_eq!(atom.z, 0.0, epsilon = 1e-9);
        }
    }
}
