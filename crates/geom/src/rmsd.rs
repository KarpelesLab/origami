//! Optimal-rotation root-mean-square deviation (Kabsch algorithm).
//!
//! For two equal-length sets of points P and Q, find the rotation R that
//! minimises Σ |R p_i − q_i|² and return the resulting RMS deviation.
//! Standard procedure:
//!
//!   1. Centre both point sets at their centroids.
//!   2. Build the cross-covariance matrix H = Pᵀ Q.
//!   3. SVD: H = U Σ Vᵀ.
//!   4. R = V D Uᵀ where D = diag(1, 1, det(VUᵀ)) — the determinant
//!      term flips one axis if the SVD produced a reflection.
//!   5. Apply R to P and compute RMSD against Q.
//!
//! Output is in the same length units as the input (Å here).
//!
//! For protein structure comparison, the convention is to use only the
//! Cα atoms — that's what `rmsd_ca` does, picking the "CA" atom from
//! each residue. The two structures must have identical residue
//! sequences for the alignment to make sense.

use crate::structure::Structure;
use crate::Vec3;
use nalgebra::Matrix3;

/// Cα RMSD between two structures whose residue sequences agree atom-
/// for-atom. Both must contain a "CA" atom in every residue.
///
/// Returns `None` if the two structures have different residue counts
/// or any residue is missing its Cα.
pub fn rmsd_ca(a: &Structure, b: &Structure) -> Option<f64> {
    if a.residues.len() != b.residues.len() || a.residues.is_empty() {
        return None;
    }
    let mut p: Vec<Vec3> = Vec::with_capacity(a.residues.len());
    let mut q: Vec<Vec3> = Vec::with_capacity(b.residues.len());
    for (ra, rb) in a.residues.iter().zip(b.residues.iter()) {
        p.push(ra.position("CA")?);
        q.push(rb.position("CA")?);
    }
    Some(rmsd_points(&p, &q))
}

/// Optimal-rotation RMSD between two equal-length point sets. Translates
/// both sets to their centroids, then solves the Kabsch SVD for the
/// rotation, then returns the RMS distance.
///
/// Panics if `p.len() != q.len()` or if either is empty.
pub fn rmsd_points(p: &[Vec3], q: &[Vec3]) -> f64 {
    assert_eq!(p.len(), q.len(), "rmsd_points: mismatched lengths");
    assert!(!p.is_empty(), "rmsd_points: empty input");
    let n = p.len() as f64;
    let cp: Vec3 = p.iter().copied().sum::<Vec3>() / n;
    let cq: Vec3 = q.iter().copied().sum::<Vec3>() / n;

    // Cross-covariance H = Σ (p_i − cp)(q_i − cq)ᵀ.
    let mut h = Matrix3::zeros();
    for (pi, qi) in p.iter().zip(q.iter()) {
        let pd = pi - cp;
        let qd = qi - cq;
        h.m11 += pd.x * qd.x;
        h.m12 += pd.x * qd.y;
        h.m13 += pd.x * qd.z;
        h.m21 += pd.y * qd.x;
        h.m22 += pd.y * qd.y;
        h.m23 += pd.y * qd.z;
        h.m31 += pd.z * qd.x;
        h.m32 += pd.z * qd.y;
        h.m33 += pd.z * qd.z;
    }

    // R = V · diag(1, 1, det(V Uᵀ)) · Uᵀ.
    let svd = h.svd(true, true);
    let u = svd.u.expect("SVD U should be present");
    let v_t = svd.v_t.expect("SVD V^T should be present");
    let v: Matrix3<f64> = v_t.transpose();
    let det: f64 = (v * u.transpose()).determinant();
    let d = det.signum();
    let mut diag = Matrix3::<f64>::identity();
    diag.m33 = d;
    let rot = v * diag * u.transpose();

    let mut sq_sum = 0.0;
    for (pi, qi) in p.iter().zip(q.iter()) {
        let pd = pi - cp;
        let qd = qi - cq;
        let rotated = rot * pd;
        let diff = rotated - qd;
        sq_sum += diff.norm_squared();
    }
    (sq_sum / n).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn identical_points_rmsd_zero() {
        let p = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let q = p.clone();
        let r = rmsd_points(&p, &q);
        assert!(r < 1e-9, "expected ~0, got {r}");
    }

    #[test]
    fn pure_translation_is_invariant() {
        let p = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let shift = Vec3::new(7.3, -2.1, 11.0);
        let q: Vec<Vec3> = p.iter().map(|v| v + shift).collect();
        let r = rmsd_points(&p, &q);
        assert!(r < 1e-9, "translation should give RMSD=0, got {r}");
    }

    #[test]
    fn pure_rotation_is_invariant() {
        let p = vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
        ];
        // 90° rotation around z.
        let q: Vec<Vec3> = p.iter().map(|v| Vec3::new(-v.y, v.x, v.z)).collect();
        let r = rmsd_points(&p, &q);
        assert!(r < 1e-9, "pure rotation should give RMSD=0, got {r}");
    }

    #[test]
    fn known_displacement_gives_expected_rmsd() {
        // Two-point system where the second point is displaced by 2 Å.
        // After optimal alignment, half the displacement is absorbed by
        // translation: RMSD = sqrt((1² + 1²)/2) = 1.0.
        let p = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0)];
        let q = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)];
        let r = rmsd_points(&p, &q);
        assert_relative_eq!(r, 1.0, epsilon = 1e-9);
    }
}
