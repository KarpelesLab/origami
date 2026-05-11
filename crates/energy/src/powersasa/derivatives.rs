//! Analytical derivatives of the spherical Gauss-Bonnet area function
//! (Klenin et al. 2011 §3). Replaces the numerical central-difference
//! scheme in `forces_sasa.rs` once complete.
//!
//! # Math
//!
//! For atom *i* with accessible region M_i bounded by arcs and vertices,
//!
//!   A_i = R_i² · [ 2π · χ_i  −  Σ_arcs cos(α_K) θ_arc  −  Σ_vertices ε_v ]
//!
//! where:
//!   - χ_i = 2c_i − L_i  (locally integer-valued → no smooth contribution to dA/dr)
//!   - α_K = cone half-angle of cap K (the cap from neighbour atom k)
//!   - θ_arc = signed central angle of the arc, measured around cap K's axis
//!   - ε_v   = exterior angle between incoming and outgoing tangents at vertex v
//!
//! Each arc lives on cap K (atom k) and is bounded by two vertices:
//! V_s = (cap_K ∩ cap_L) for atom l, and V_e = (cap_K ∩ cap_M) for atom m.
//! Thus the arc's contribution depends smoothly on positions of *up to four*
//! atoms: {i, k, l, m}.
//!
//! Each vertex V lives at the intersection of two cap circles, so it
//! depends on three atoms: {i, k_a, k_b} for caps K_a and K_b.
//!
//! By the product rule:
//!
//!   ∂(cos α_K · θ_arc)/∂r_x = θ_arc · (∂cos α_K / ∂r_x)
//!                           + cos α_K · (∂θ_arc / ∂r_x)
//!
//! - **Cone angle.** cos α_K = (d² + R_i² − R_k²) / (2 d R_i), d = |r_k − r_i|.
//!   Depends only on r_i and r_k. Derivation in [`cos_alpha_grad`].
//!
//! - **Arc angle.** θ_arc is the signed angle around ω_K from V_s to V_e.
//!   As atoms move, three things change: (1) the cap K circle itself
//!   (ω_K and α_K change with r_i, r_k); (2) V_s moves with r_i, r_k, r_l;
//!   (3) V_e moves with r_i, r_k, r_m. The total derivative is the sum of
//!   the partial derivatives along each motion mode. (TODO.)
//!
//! - **Exterior angle.** At vertex V = cap_K ∩ cap_L,
//!     t_in  = V × ω_K  (tangent on cap K at V, going CW around ω_K)
//!     t_out = V × ω_L  (tangent on cap L at V, going CW around ω_L)
//!     ε = atan2(V · (t_in × t_out), t_in · t_out)
//!   Depends on r_i, r_k, r_l (via V and the two cone axes). (TODO.)
//!
//! # Implementation status
//!
//! - [x] `cos_alpha_grad`: ∂cos α / ∂r_i, ∂cos α / ∂r_k closed form
//! - [ ] `theta_grad`: ∂θ_arc / ∂r_{i,k,l,m}
//! - [ ] `epsilon_grad`: ∂ε / ∂r_{i,k,l}
//! - [ ] full `add_sasa_forces_analytical` replacing the numerical version
//!
//! Each is staged with its own finite-difference acceptance test (see
//! `tests` module) so the work can land incrementally and we never trust
//! an analytical derivative we haven't cross-checked against ε-perturbation.

use geom::Vec3;

/// Apply the Jacobian `∂ω_K/∂r_k` to a vector `v`, where
///
///   ω_K = (r_k − r_i) / |r_k − r_i|
///
/// Writing d = |r_k − r_i| and u = ω_K:
///
///   ∂u/∂r_k = (1/d)(I − u uᵀ)
///   (∂u/∂r_k) · v = (v − (u·v) u) / d
///   (∂u/∂r_i) · v = −(∂u/∂r_k) · v
///
/// Returns `(jvec_i, jvec_k)` = `((∂ω_K/∂r_i)·v, (∂ω_K/∂r_k)·v)`.
pub fn cap_axis_jvp(p_i: Vec3, p_k: Vec3, v: Vec3) -> (Vec3, Vec3) {
    let r_vec = p_k - p_i;
    let d2 = r_vec.norm_squared();
    let d = d2.sqrt();
    if d < 1e-12 {
        return (Vec3::zeros(), Vec3::zeros());
    }
    let u = r_vec / d;
    let projected = v - u * u.dot(&v);
    let g_k = projected / d;
    (-g_k, g_k)
}

/// Gradient of `cos α_K` w.r.t. positions of atoms i and k, where the cap
/// is defined by:
///
///   cos α_K = (d² + R_i² − R_k²) / (2 d R_i),     d = |r_k − r_i|
///
/// Returns `(∂cos α / ∂r_i, ∂cos α / ∂r_k)`. By translational invariance,
/// `∂/∂r_i = −∂/∂r_k`. The chain rule on d gives:
///
///   ∂(cos α)/∂d = (d² − R_i² + R_k²) / (2 d² R_i)
///   ∂d/∂r_k = (r_k − r_i)/d = u_ik
///
///   ∂cos α / ∂r_k = ((d² − R_i² + R_k²) / (2 d² R_i)) · u_ik
///   ∂cos α / ∂r_i = − ∂cos α / ∂r_k
pub fn cos_alpha_grad(
    p_i: Vec3,
    p_k: Vec3,
    radius_i: f64,
    radius_k: f64,
) -> (Vec3, Vec3) {
    let r_vec = p_k - p_i;
    let d2 = r_vec.norm_squared();
    let d = d2.sqrt();
    if d < 1e-12 {
        return (Vec3::zeros(), Vec3::zeros());
    }
    let u_ik = r_vec / d;
    // ∂cos α / ∂d = (d² − R_i² + R_k²) / (2 d² R_i).
    let dcos_dd = (d2 - radius_i * radius_i + radius_k * radius_k) / (2.0 * d2 * radius_i);
    let g_k = u_ik * dcos_dd;
    (-g_k, g_k)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Central-difference check on `cos_alpha_grad`. Build cap geometry
    /// from two atom positions/radii, compute cos α analytically, then
    /// perturb each atom by ±ε and compare to central-difference values.
    #[test]
    fn cos_alpha_grad_matches_finite_difference() {
        // Atom i: position (0, 0, 0), radius 3.0; atom k: (2.5, 1.0, 0.3),
        // radius 2.5. Generic separation, no symmetry traps.
        let p_i = Vec3::new(0.0, 0.0, 0.0);
        let p_k = Vec3::new(2.5, 1.0, 0.3);
        let r_i = 3.0;
        let r_k = 2.5;

        let (g_i, g_k) = cos_alpha_grad(p_i, p_k, r_i, r_k);

        let cos_alpha = |pi: Vec3, pk: Vec3| {
            let d = (pk - pi).norm();
            (d * d + r_i * r_i - r_k * r_k) / (2.0 * d * r_i)
        };

        let eps = 1e-6;
        for axis in 0..3 {
            // ∂/∂r_i[axis]
            let mut pi_p = p_i;
            let mut pi_m = p_i;
            pi_p[axis] += eps;
            pi_m[axis] -= eps;
            let numeric_i = (cos_alpha(pi_p, p_k) - cos_alpha(pi_m, p_k)) / (2.0 * eps);
            assert!(
                (g_i[axis] - numeric_i).abs() < 1e-7,
                "g_i[{}]: analytical={} numeric={}",
                axis, g_i[axis], numeric_i,
            );
            // ∂/∂r_k[axis]
            let mut pk_p = p_k;
            let mut pk_m = p_k;
            pk_p[axis] += eps;
            pk_m[axis] -= eps;
            let numeric_k = (cos_alpha(p_i, pk_p) - cos_alpha(p_i, pk_m)) / (2.0 * eps);
            assert!(
                (g_k[axis] - numeric_k).abs() < 1e-7,
                "g_k[{}]: analytical={} numeric={}",
                axis, g_k[axis], numeric_k,
            );
        }
    }

    /// Translational invariance: shifting both atoms by the same vector
    /// leaves cos α unchanged, so g_i + g_k = 0.
    #[test]
    fn cos_alpha_grad_translational_invariance() {
        let p_i = Vec3::new(1.5, -0.7, 2.1);
        let p_k = Vec3::new(3.0, 0.4, 1.8);
        let (g_i, g_k) = cos_alpha_grad(p_i, p_k, 2.8, 2.6);
        let sum = g_i + g_k;
        assert!(
            sum.norm() < 1e-12,
            "expected g_i + g_k = 0, got {:?} (norm {})",
            sum,
            sum.norm()
        );
    }

    /// Central-difference check on `cap_axis_jvp`. We pick a random `v`
    /// and verify that the analytical Jacobian-vector product agrees
    /// with the numerical one.
    #[test]
    fn cap_axis_jvp_matches_finite_difference() {
        let p_i = Vec3::new(-0.5, 1.2, 0.8);
        let p_k = Vec3::new(2.7, -0.3, 1.6);
        let v = Vec3::new(0.4, -0.9, 0.2);

        let axis_of = |pi: Vec3, pk: Vec3| {
            let r = pk - pi;
            r / r.norm()
        };

        let (g_i, g_k) = cap_axis_jvp(p_i, p_k, v);

        // Directional finite-difference: (∂ω/∂r_i) · v ≈
        //   (ω(p_i + ε v, p_k) − ω(p_i − ε v, p_k)) / (2 ε).
        let eps = 1e-6;
        let numeric_i =
            (axis_of(p_i + v * eps, p_k) - axis_of(p_i - v * eps, p_k)) / (2.0 * eps);
        let numeric_k =
            (axis_of(p_i, p_k + v * eps) - axis_of(p_i, p_k - v * eps)) / (2.0 * eps);

        assert!(
            (g_i - numeric_i).norm() < 1e-7,
            "cap_axis_jvp_i: analytical {:?} vs numeric {:?}",
            g_i,
            numeric_i
        );
        assert!(
            (g_k - numeric_k).norm() < 1e-7,
            "cap_axis_jvp_k: analytical {:?} vs numeric {:?}",
            g_k,
            numeric_k
        );
    }
}
