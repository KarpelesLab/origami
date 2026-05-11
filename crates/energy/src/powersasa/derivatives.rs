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

use super::geometry::SmallCircle;

/// One of the two intersection points of two small-circle caps on the
/// unit sphere. The sign disambiguates which of the two roots:
/// `RootSign::Plus` is `base + offset`, `Minus` is `base − offset`, in
/// the parameterisation `intersect_circles` uses (see geometry.rs).
///
/// We need this in the analytical-derivative path because the
/// topology-caching strategy stores *which* of the two intersection
/// points a vertex is, so it can recompute the vertex position from
/// updated cap parameters without re-running `find_boundary`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootSign {
    Plus,
    Minus,
}

impl RootSign {
    pub fn as_f64(self) -> f64 {
        match self {
            RootSign::Plus => 1.0,
            RootSign::Minus => -1.0,
        }
    }
}

/// Compute the vertex point at the intersection of two small-circle
/// caps on the unit sphere, given a sign choice. Returns `None` if the
/// caps are disjoint or coincident — the caller should treat that as a
/// topology break (the cached identity no longer applies and a fresh
/// boundary recomputation is needed).
///
/// Mirrors `intersect_circles` but takes the sign explicitly so it can
/// be cached. Used by the topology-cached SASA force path.
pub fn vertex_point(c1: SmallCircle, c2: SmallCircle, sign: RootSign) -> Option<Vec3> {
    let sigma = c1.axis.dot(&c2.axis);
    let denom = 1.0 - sigma * sigma;
    if denom.abs() < 1e-12 {
        return None;
    }
    let a = (c1.cos_alpha - sigma * c2.cos_alpha) / denom;
    let b = (c2.cos_alpha - sigma * c1.cos_alpha) / denom;
    let c_sq = 1.0 - a * a - b * b - 2.0 * a * b * sigma;
    if c_sq <= 0.0 {
        return None;
    }
    let base = c1.axis * a + c2.axis * b;
    let normal_cross = c1.axis.cross(&c2.axis);
    let n_norm = normal_cross.norm();
    if n_norm < 1e-12 {
        return None;
    }
    let offset = normal_cross / n_norm * c_sq.sqrt();
    Some(base + offset * sign.as_f64())
}

/// Identify which sign of `vertex_point` matches a known reference
/// position (typically a vertex extracted from an unperturbed
/// `find_boundary` result). Returns `None` if the caps don't intersect.
pub fn identify_root_sign(c1: SmallCircle, c2: SmallCircle, reference: Vec3) -> Option<RootSign> {
    let v_plus = vertex_point(c1, c2, RootSign::Plus)?;
    let v_minus = vertex_point(c1, c2, RootSign::Minus)?;
    let d_plus = (v_plus - reference).norm_squared();
    let d_minus = (v_minus - reference).norm_squared();
    Some(if d_plus <= d_minus {
        RootSign::Plus
    } else {
        RootSign::Minus
    })
}

/// Directional derivative of `cos α_K` along `(dr_i, dr_k)`:
///
///   dcos α = ∂cos α/∂r_i · dr_i + ∂cos α/∂r_k · dr_k
///
/// Used as a primitive by `vertex_point_jvp` and `arc_theta_jvp`.
pub fn cos_alpha_directional(
    p_i: Vec3,
    p_k: Vec3,
    radius_i: f64,
    radius_k: f64,
    dr_i: Vec3,
    dr_k: Vec3,
) -> f64 {
    let (g_i, g_k) = cos_alpha_grad(p_i, p_k, radius_i, radius_k);
    g_i.dot(&dr_i) + g_k.dot(&dr_k)
}

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

/// Directional derivative of the vertex point
/// `V = base + sign·offset` at the intersection of cap K (atom k) and
/// cap L (atom l) on atom i's unit sphere, given small position
/// displacements `dr_i, dr_k, dr_l`.
///
/// Chain rule through the intermediate parameters per `intersect_circles`:
///
///   σ      = ω_K · ω_L                            → dσ
///   denom  = 1 − σ²                               → ddenom = −2 σ dσ
///   a      = (cos α_K − σ cos α_L) / denom        → da
///   b      = (cos α_L − σ cos α_K) / denom        → db
///   c²     = 1 − a² − b² − 2ab σ                  → dc² (signed sqrt for c)
///   base   = a ω_K + b ω_L                        → dbase
///   m      = ω_K × ω_L,   |m|² = denom            → dm, d|m|
///   n_perp = m / |m|                              → dn_perp
///   V      = base + sign·c·n_perp                 → dV
///
/// Returns `None` when the caps no longer intersect under the
/// reference inputs (denom too small, c² ≤ 0, or |m| too small).
#[allow(clippy::too_many_arguments)]
pub fn vertex_point_jvp(
    p_i: Vec3,
    p_k: Vec3,
    p_l: Vec3,
    radius_i: f64,
    radius_k: f64,
    radius_l: f64,
    sign: RootSign,
    dr_i: Vec3,
    dr_k: Vec3,
    dr_l: Vec3,
) -> Option<Vec3> {
    // ω_K = (r_k − r_i) / |r_k − r_i|
    let rk_vec = p_k - p_i;
    let d_ik = rk_vec.norm();
    if d_ik < 1e-12 {
        return None;
    }
    let omega_k = rk_vec / d_ik;
    // ω_L
    let rl_vec = p_l - p_i;
    let d_il = rl_vec.norm();
    if d_il < 1e-12 {
        return None;
    }
    let omega_l = rl_vec / d_il;
    // cos α values
    let cos_alpha_k =
        (d_ik * d_ik + radius_i * radius_i - radius_k * radius_k) / (2.0 * d_ik * radius_i);
    let cos_alpha_l =
        (d_il * d_il + radius_i * radius_i - radius_l * radius_l) / (2.0 * d_il * radius_i);

    let sigma = omega_k.dot(&omega_l);
    let denom = 1.0 - sigma * sigma;
    if denom.abs() < 1e-12 {
        return None;
    }

    let a = (cos_alpha_k - sigma * cos_alpha_l) / denom;
    let b = (cos_alpha_l - sigma * cos_alpha_k) / denom;
    let c_sq = 1.0 - a * a - b * b - 2.0 * a * b * sigma;
    if c_sq <= 0.0 {
        return None;
    }
    let c = c_sq.sqrt();
    let sign_f = sign.as_f64();

    let m = omega_k.cross(&omega_l);
    let m_norm_sq = m.norm_squared();
    let m_norm = m_norm_sq.sqrt();
    if m_norm < 1e-12 {
        return None;
    }
    let n_perp = m / m_norm;

    // === Directional derivatives at the reference point ===

    // dω_K via the JVP form (v − (u·v) u) / d, with v = dr_k − dr_i
    // (ω_K = (r_k − r_i)/d_ik depends only on the difference).
    let d_omega_k = {
        let dv = dr_k - dr_i;
        (dv - omega_k * omega_k.dot(&dv)) / d_ik
    };
    let d_omega_l = {
        let dv = dr_l - dr_i;
        (dv - omega_l * omega_l.dot(&dv)) / d_il
    };

    // dcos α_K = cos_alpha_directional(p_i, p_k, r_i, r_k, dr_i, dr_k)
    let dcos_k = cos_alpha_directional(p_i, p_k, radius_i, radius_k, dr_i, dr_k);
    let dcos_l = cos_alpha_directional(p_i, p_l, radius_i, radius_l, dr_i, dr_l);

    // dσ = dω_K · ω_L + ω_K · dω_L
    let d_sigma = d_omega_k.dot(&omega_l) + omega_k.dot(&d_omega_l);
    let d_denom = -2.0 * sigma * d_sigma;

    // da, db via quotient rule.
    // a = num_a / denom, num_a = cos α_K − σ cos α_L.
    let num_a = cos_alpha_k - sigma * cos_alpha_l;
    let d_num_a = dcos_k - d_sigma * cos_alpha_l - sigma * dcos_l;
    let d_a = (d_num_a * denom - num_a * d_denom) / (denom * denom);

    let num_b = cos_alpha_l - sigma * cos_alpha_k;
    let d_num_b = dcos_l - d_sigma * cos_alpha_k - sigma * dcos_k;
    let d_b = (d_num_b * denom - num_b * d_denom) / (denom * denom);

    // dc² = -2a·da − 2b·db − 2·(da·b + a·db)·σ − 2ab·dσ
    let dc_sq = -2.0 * a * d_a - 2.0 * b * d_b - 2.0 * (d_a * b + a * d_b) * sigma
        - 2.0 * a * b * d_sigma;
    let d_c = dc_sq / (2.0 * c);

    // dbase = da·ω_K + a·dω_K + db·ω_L + b·dω_L
    let d_base = omega_k * d_a + d_omega_k * a + omega_l * d_b + d_omega_l * b;

    // dm = dω_K × ω_L + ω_K × dω_L; dn_perp is the tangential component
    // of dm/|m| (the radial component cancels in the unit-vector
    // derivative): dn_perp = (dm − n_perp · (n_perp · dm)) / |m|.
    let dm = d_omega_k.cross(&omega_l) + omega_k.cross(&d_omega_l);
    let dn_perp = (dm - n_perp * n_perp.dot(&dm)) / m_norm;

    // dV = dbase + sign · (dc · n_perp + c · dn_perp)
    let d_v = d_base + (n_perp * d_c + dn_perp * c) * sign_f;
    Some(d_v)
}

/// Directional derivative of the signed arc angle θ on cap K's circle
/// from V_start (= cap_K ∩ cap_L, atom l) to V_end (= cap_K ∩ cap_M,
/// atom m).
///
/// θ is the same quantity that `signed_arc_angle` (in `geometry.rs`)
/// returns at the reference configuration:
///
///   proj_s = V_s − ω_K (V_s · ω_K),  u_s = proj_s / |proj_s|
///   proj_e = V_e − ω_K (V_e · ω_K),  u_e = proj_e / |proj_e|
///   cos θ  = u_s · u_e
///   sin θ  = ω_K · (u_s × u_e)
///   θ      = atan2(sin θ, cos θ)
///
/// dθ = cos θ · dsin θ − sin θ · dcos θ.
///
/// All five inputs r_i, r_k, r_l, r_m, and the cap K axis change as
/// atoms move — the derivative routes through vertex_point_jvp for
/// V_s and V_e, and through cap_axis_jvp-style logic for ω_K.
#[allow(clippy::too_many_arguments)]
pub fn arc_theta_jvp(
    p_i: Vec3,
    p_k: Vec3,
    p_l: Vec3,
    p_m: Vec3,
    radius_i: f64,
    radius_k: f64,
    radius_l: f64,
    radius_m: f64,
    sign_s: RootSign,
    sign_e: RootSign,
    dr_i: Vec3,
    dr_k: Vec3,
    dr_l: Vec3,
    dr_m: Vec3,
) -> Option<f64> {
    // ω_K and dω_K (cap K depends only on r_i, r_k).
    let rk_vec = p_k - p_i;
    let d_ik = rk_vec.norm();
    if d_ik < 1e-12 {
        return None;
    }
    let omega_k = rk_vec / d_ik;
    let d_omega_k = {
        let dv = dr_k - dr_i;
        (dv - omega_k * omega_k.dot(&dv)) / d_ik
    };

    // V_s and dV_s (start vertex; intersection with cap L → depends on r_i, r_k, r_l).
    let cos_alpha_k =
        (d_ik * d_ik + radius_i * radius_i - radius_k * radius_k) / (2.0 * d_ik * radius_i);
    let circle_k = SmallCircle::new(omega_k, cos_alpha_k);

    let circle_l = {
        let rl_vec = p_l - p_i;
        let d_il = rl_vec.norm();
        if d_il < 1e-12 {
            return None;
        }
        let omega_l = rl_vec / d_il;
        let cos_alpha_l =
            (d_il * d_il + radius_i * radius_i - radius_l * radius_l) / (2.0 * d_il * radius_i);
        SmallCircle::new(omega_l, cos_alpha_l)
    };
    let circle_m = {
        let rm_vec = p_m - p_i;
        let d_im = rm_vec.norm();
        if d_im < 1e-12 {
            return None;
        }
        let omega_m = rm_vec / d_im;
        let cos_alpha_m =
            (d_im * d_im + radius_i * radius_i - radius_m * radius_m) / (2.0 * d_im * radius_i);
        SmallCircle::new(omega_m, cos_alpha_m)
    };

    let v_s = vertex_point(circle_k, circle_l, sign_s)?;
    let v_e = vertex_point(circle_k, circle_m, sign_e)?;
    let d_v_s = vertex_point_jvp(
        p_i, p_k, p_l, radius_i, radius_k, radius_l, sign_s, dr_i, dr_k, dr_l,
    )?;
    let d_v_e = vertex_point_jvp(
        p_i, p_k, p_m, radius_i, radius_k, radius_m, sign_e, dr_i, dr_k, dr_m,
    )?;

    // proj = V − ω_K (V·ω_K); u = proj/|proj|.
    let do_proj = |v: Vec3, dv: Vec3| {
        // proj = v − ω_K · (v · ω_K)
        let v_dot_omega = v.dot(&omega_k);
        let proj = v - omega_k * v_dot_omega;
        // d(v · ω_K) = dv · ω_K + v · dω_K
        let d_v_dot_omega = dv.dot(&omega_k) + v.dot(&d_omega_k);
        let d_proj = dv - d_omega_k * v_dot_omega - omega_k * d_v_dot_omega;
        let n = proj.norm();
        if n < 1e-12 {
            return None;
        }
        let u = proj / n;
        // du = (dproj − u·(u·dproj)) / |proj|
        let du = (d_proj - u * u.dot(&d_proj)) / n;
        Some((u, du))
    };
    let (u_s, du_s) = do_proj(v_s, d_v_s)?;
    let (u_e, du_e) = do_proj(v_e, d_v_e)?;

    // cos θ = u_s · u_e; dcos θ = du_s · u_e + u_s · du_e.
    let cos_theta = u_s.dot(&u_e);
    let d_cos_theta = du_s.dot(&u_e) + u_s.dot(&du_e);
    // sin θ = ω_K · (u_s × u_e);
    // dsin θ = dω_K · (u_s × u_e) + ω_K · (du_s × u_e + u_s × du_e).
    let us_cross_ue = u_s.cross(&u_e);
    let sin_theta = omega_k.dot(&us_cross_ue);
    let d_us_cross_ue = du_s.cross(&u_e) + u_s.cross(&du_e);
    let d_sin_theta = d_omega_k.dot(&us_cross_ue) + omega_k.dot(&d_us_cross_ue);

    // θ = atan2(sin, cos) ⇒ dθ = cos · dsin − sin · dcos
    //   (using sin² + cos² = 1 here since u_s, u_e are unit vectors and θ is well-defined).
    Some(cos_theta * d_sin_theta - sin_theta * d_cos_theta)
}

/// Directional derivative of the exterior angle ε at a vertex
/// V = cap_K ∩ cap_L on atom i's unit sphere.
///
/// ε is the same quantity the area code in `area.rs` accumulates:
///
///   t_in  = (V × ω_K) / |V × ω_K|     (tangent on cap K at V)
///   t_out = (V × ω_L) / |V × ω_L|     (tangent on cap L at V)
///   cos ε = t_in · t_out
///   sin ε = V · (t_in × t_out)
///   ε     = atan2(sin ε, cos ε)
///
/// Both t_in and t_out lie in the tangent plane at V, so
/// `t_in × t_out` is parallel to V and the (sin² + cos²) of the atan2
/// inputs is exactly 1 — meaning dε = cos ε · dsin ε − sin ε · dcos ε
/// with no normalisation.
///
/// Depends smoothly on (r_i, r_k, r_l) via V (vertex_point_jvp), ω_K
/// (cap-axis JVP), and ω_L (cap-axis JVP). Same chain rule pattern as
/// `arc_theta_jvp`, with two unit-vector normalisations.
#[allow(clippy::too_many_arguments)]
pub fn vertex_epsilon_jvp(
    p_i: Vec3,
    p_k: Vec3,
    p_l: Vec3,
    radius_i: f64,
    radius_k: f64,
    radius_l: f64,
    sign: RootSign,
    incoming_is_k: bool,
    dr_i: Vec3,
    dr_k: Vec3,
    dr_l: Vec3,
) -> Option<f64> {
    let rk_vec = p_k - p_i;
    let d_ik = rk_vec.norm();
    if d_ik < 1e-12 {
        return None;
    }
    let omega_k = rk_vec / d_ik;
    let d_omega_k = {
        let dv = dr_k - dr_i;
        (dv - omega_k * omega_k.dot(&dv)) / d_ik
    };
    let rl_vec = p_l - p_i;
    let d_il = rl_vec.norm();
    if d_il < 1e-12 {
        return None;
    }
    let omega_l = rl_vec / d_il;
    let d_omega_l = {
        let dv = dr_l - dr_i;
        (dv - omega_l * omega_l.dot(&dv)) / d_il
    };

    // V = vertex of (cap_K ∩ cap_L) with given sign.
    let cos_alpha_k =
        (d_ik * d_ik + radius_i * radius_i - radius_k * radius_k) / (2.0 * d_ik * radius_i);
    let cos_alpha_l =
        (d_il * d_il + radius_i * radius_i - radius_l * radius_l) / (2.0 * d_il * radius_i);
    let circle_k = SmallCircle::new(omega_k, cos_alpha_k);
    let circle_l = SmallCircle::new(omega_l, cos_alpha_l);
    let v = vertex_point(circle_k, circle_l, sign)?;
    let dv = vertex_point_jvp(
        p_i, p_k, p_l, radius_i, radius_k, radius_l, sign, dr_i, dr_k, dr_l,
    )?;

    // Tangents on caps K and L at V.
    // Convention: `incoming_is_k` selects which cap's tangent is t_in.
    let (omega_in, d_omega_in, omega_out, d_omega_out) = if incoming_is_k {
        (omega_k, d_omega_k, omega_l, d_omega_l)
    } else {
        (omega_l, d_omega_l, omega_k, d_omega_k)
    };

    // (V × ω) / |V × ω| and its directional derivative.
    let make_tangent = |v: Vec3, dv: Vec3, omega: Vec3, d_omega: Vec3| {
        let cross = v.cross(&omega);
        let n = cross.norm();
        if n < 1e-12 {
            return None;
        }
        let u = cross / n;
        let d_cross = dv.cross(&omega) + v.cross(&d_omega);
        // du = (dcross − u (u · dcross)) / |cross|
        let du = (d_cross - u * u.dot(&d_cross)) / n;
        Some((u, du))
    };
    let (t_in, dt_in) = make_tangent(v, dv, omega_in, d_omega_in)?;
    let (t_out, dt_out) = make_tangent(v, dv, omega_out, d_omega_out)?;

    let cos_eps = t_in.dot(&t_out);
    let d_cos_eps = dt_in.dot(&t_out) + t_in.dot(&dt_out);

    let tin_cross_tout = t_in.cross(&t_out);
    let sin_eps = v.dot(&tin_cross_tout);
    let d_tin_cross_tout = dt_in.cross(&t_out) + t_in.cross(&dt_out);
    let d_sin_eps = dv.dot(&tin_cross_tout) + v.dot(&d_tin_cross_tout);

    Some(cos_eps * d_sin_eps - sin_eps * d_cos_eps)
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

    /// `vertex_point_jvp` matches the central-difference derivative
    /// of the vertex point along a directional perturbation
    /// `(dr_i, dr_k, dr_l)`. We pick a generic three-atom configuration
    /// and a random perturbation direction, then compare the analytical
    /// JVP to `(V(r + εd) − V(r − εd)) / (2ε)` for each atom independently.
    #[test]
    fn vertex_point_jvp_matches_finite_difference() {
        let p_i = Vec3::new(0.0, 0.0, 0.0);
        let p_k = Vec3::new(2.5, 0.4, 0.7);
        let p_l = Vec3::new(0.8, 2.3, -0.5);
        let r_i = 3.0;
        let r_k = 2.8;
        let r_l = 2.6;
        // Random-ish perturbation directions.
        let dr_i = Vec3::new(0.31, -0.42, 0.17);
        let dr_k = Vec3::new(-0.23, 0.18, 0.27);
        let dr_l = Vec3::new(0.11, 0.34, -0.29);

        let eps = 1e-6;
        let circles = |pi: Vec3, pk: Vec3, pl: Vec3| {
            let d_ik = (pk - pi).norm();
            let omega_k = (pk - pi) / d_ik;
            let cos_alpha_k = (d_ik * d_ik + r_i * r_i - r_k * r_k) / (2.0 * d_ik * r_i);
            let d_il = (pl - pi).norm();
            let omega_l = (pl - pi) / d_il;
            let cos_alpha_l = (d_il * d_il + r_i * r_i - r_l * r_l) / (2.0 * d_il * r_i);
            (
                SmallCircle::new(omega_k, cos_alpha_k),
                SmallCircle::new(omega_l, cos_alpha_l),
            )
        };

        for sign in [RootSign::Plus, RootSign::Minus] {
            // For each of the three "atom" directions independently,
            // plus a combined perturbation:
            for (label, di, dk, dl) in [
                ("d/dr_i", dr_i, Vec3::zeros(), Vec3::zeros()),
                ("d/dr_k", Vec3::zeros(), dr_k, Vec3::zeros()),
                ("d/dr_l", Vec3::zeros(), Vec3::zeros(), dr_l),
                ("combined", dr_i, dr_k, dr_l),
            ] {
                let analytical =
                    vertex_point_jvp(p_i, p_k, p_l, r_i, r_k, r_l, sign, di, dk, dl).expect("jvp");

                let (c1p, c2p) = circles(p_i + di * eps, p_k + dk * eps, p_l + dl * eps);
                let (c1m, c2m) = circles(p_i - di * eps, p_k - dk * eps, p_l - dl * eps);
                let v_p = vertex_point(c1p, c2p, sign).expect("V+");
                let v_m = vertex_point(c1m, c2m, sign).expect("V-");
                let numeric = (v_p - v_m) / (2.0 * eps);

                let err = (analytical - numeric).norm();
                assert!(
                    err < 1e-6,
                    "{} {:?}: analytical {:?} vs numeric {:?} (|err| = {})",
                    label,
                    sign,
                    analytical,
                    numeric,
                    err,
                );
            }
        }
    }

    /// `arc_theta_jvp` matches a central-difference of
    /// `signed_arc_angle` for each atom perturbation direction.
    #[test]
    fn arc_theta_jvp_matches_finite_difference() {
        use crate::powersasa::geometry::signed_arc_angle;

        // Configuration: atom i at origin, three neighbour atoms k/l/m
        // forming a 3D triangle that produces a non-degenerate boundary.
        let p_i = Vec3::new(0.0, 0.0, 0.0);
        let p_k = Vec3::new(2.5, 0.4, 0.7);
        let p_l = Vec3::new(0.8, 2.3, -0.5);
        let p_m = Vec3::new(-0.6, 0.9, 2.4);
        let r_i = 3.0;
        let r_k = 2.8;
        let r_l = 2.6;
        let r_m = 2.7;

        // Sign choices for the two endpoint vertices. We test both for
        // V_s and pick a working V_e for each. With 4 axis-aligned and
        // 1 combined perturbation, that's 5×4 = 20 finite-difference
        // checks per (s, e) sign pair.
        let circles_of = |pi: Vec3, pk: Vec3, pl: Vec3, pm: Vec3| {
            let make_cap = |p_a: Vec3, p_b: Vec3, r_a: f64, r_b: f64| {
                let d = (p_b - p_a).norm();
                let omega = (p_b - p_a) / d;
                let cos_a = (d * d + r_a * r_a - r_b * r_b) / (2.0 * d * r_a);
                SmallCircle::new(omega, cos_a)
            };
            (
                make_cap(pi, pk, r_i, r_k),
                make_cap(pi, pl, r_i, r_l),
                make_cap(pi, pm, r_i, r_m),
            )
        };

        let dr_i = Vec3::new(0.31, -0.42, 0.17);
        let dr_k = Vec3::new(-0.23, 0.18, 0.27);
        let dr_l = Vec3::new(0.11, 0.34, -0.29);
        let dr_m = Vec3::new(0.07, -0.13, 0.21);
        let eps = 1e-6;

        for sign_s in [RootSign::Plus, RootSign::Minus] {
            for sign_e in [RootSign::Plus, RootSign::Minus] {
                for (label, di, dk, dl, dm) in [
                    ("d/dr_i", dr_i, Vec3::zeros(), Vec3::zeros(), Vec3::zeros()),
                    ("d/dr_k", Vec3::zeros(), dr_k, Vec3::zeros(), Vec3::zeros()),
                    ("d/dr_l", Vec3::zeros(), Vec3::zeros(), dr_l, Vec3::zeros()),
                    ("d/dr_m", Vec3::zeros(), Vec3::zeros(), Vec3::zeros(), dr_m),
                    ("combined", dr_i, dr_k, dr_l, dr_m),
                ] {
                    let analytical = arc_theta_jvp(
                        p_i, p_k, p_l, p_m, r_i, r_k, r_l, r_m, sign_s, sign_e, di, dk, dl, dm,
                    )
                    .expect("arc_theta_jvp");

                    let theta_at = |pi: Vec3, pk: Vec3, pl: Vec3, pm: Vec3| {
                        let (ck, cl, cm) = circles_of(pi, pk, pl, pm);
                        let vs = vertex_point(ck, cl, sign_s).expect("V_s");
                        let ve = vertex_point(ck, cm, sign_e).expect("V_e");
                        signed_arc_angle(vs, ve, ck)
                    };
                    let t_p = theta_at(p_i + di * eps, p_k + dk * eps, p_l + dl * eps, p_m + dm * eps);
                    let t_m = theta_at(p_i - di * eps, p_k - dk * eps, p_l - dl * eps, p_m - dm * eps);
                    let numeric = (t_p - t_m) / (2.0 * eps);

                    let err = (analytical - numeric).abs();
                    assert!(
                        err < 1e-5,
                        "{} sign_s={:?} sign_e={:?}: analytical {} vs numeric {} (err {})",
                        label,
                        sign_s,
                        sign_e,
                        analytical,
                        numeric,
                        err,
                    );
                }
            }
        }
    }

    /// `vertex_epsilon_jvp` matches a central-difference of the same
    /// ε formula used in `area.rs`.
    #[test]
    fn vertex_epsilon_jvp_matches_finite_difference() {
        let p_i = Vec3::new(0.0, 0.0, 0.0);
        let p_k = Vec3::new(2.5, 0.4, 0.7);
        let p_l = Vec3::new(0.8, 2.3, -0.5);
        let r_i = 3.0;
        let r_k = 2.8;
        let r_l = 2.6;
        let dr_i = Vec3::new(0.31, -0.42, 0.17);
        let dr_k = Vec3::new(-0.23, 0.18, 0.27);
        let dr_l = Vec3::new(0.11, 0.34, -0.29);
        let eps = 1e-6;

        // Forward formula matching area.rs.
        let epsilon_at = |pi: Vec3, pk: Vec3, pl: Vec3, sign: RootSign, incoming_is_k: bool| {
            let d_ik = (pk - pi).norm();
            let omega_k = (pk - pi) / d_ik;
            let cos_a_k = (d_ik * d_ik + r_i * r_i - r_k * r_k) / (2.0 * d_ik * r_i);
            let d_il = (pl - pi).norm();
            let omega_l = (pl - pi) / d_il;
            let cos_a_l = (d_il * d_il + r_i * r_i - r_l * r_l) / (2.0 * d_il * r_i);
            let ck = SmallCircle::new(omega_k, cos_a_k);
            let cl = SmallCircle::new(omega_l, cos_a_l);
            let v = vertex_point(ck, cl, sign).expect("V");
            let (omega_in, omega_out) = if incoming_is_k {
                (omega_k, omega_l)
            } else {
                (omega_l, omega_k)
            };
            let t_in = v.cross(&omega_in).normalize();
            let t_out = v.cross(&omega_out).normalize();
            let cos_eps = t_in.dot(&t_out).clamp(-1.0, 1.0);
            let sin_eps = v.dot(&t_in.cross(&t_out));
            sin_eps.atan2(cos_eps)
        };

        for sign in [RootSign::Plus, RootSign::Minus] {
            for incoming_is_k in [true, false] {
                for (label, di, dk, dl) in [
                    ("d/dr_i", dr_i, Vec3::zeros(), Vec3::zeros()),
                    ("d/dr_k", Vec3::zeros(), dr_k, Vec3::zeros()),
                    ("d/dr_l", Vec3::zeros(), Vec3::zeros(), dr_l),
                    ("combined", dr_i, dr_k, dr_l),
                ] {
                    let analytical = vertex_epsilon_jvp(
                        p_i, p_k, p_l, r_i, r_k, r_l, sign, incoming_is_k, di, dk, dl,
                    )
                    .expect("vertex_epsilon_jvp");

                    let e_p = epsilon_at(
                        p_i + di * eps,
                        p_k + dk * eps,
                        p_l + dl * eps,
                        sign,
                        incoming_is_k,
                    );
                    let e_m = epsilon_at(
                        p_i - di * eps,
                        p_k - dk * eps,
                        p_l - dl * eps,
                        sign,
                        incoming_is_k,
                    );
                    let numeric = (e_p - e_m) / (2.0 * eps);

                    let err = (analytical - numeric).abs();
                    assert!(
                        err < 1e-5,
                        "{} sign={:?} in_k={}: analytical {} vs numeric {} (err {})",
                        label,
                        sign,
                        incoming_is_k,
                        analytical,
                        numeric,
                        err,
                    );
                }
            }
        }
    }

    /// `vertex_point` matches the two roots that `intersect_circles`
    /// returns, and `identify_root_sign` picks the right one when
    /// given a reference position.
    #[test]
    fn vertex_point_matches_intersect_circles() {
        use crate::powersasa::geometry::{intersect_circles, CircleIntersection};
        // Two caps at a generic configuration.
        let c1 = SmallCircle::new(Vec3::new(0.3, 0.5, 1.0), 0.4);
        let c2 = SmallCircle::new(Vec3::new(1.0, 0.2, -0.3), 0.3);
        let (p, q) = match intersect_circles(c1, c2) {
            CircleIntersection::Two(p, q) => (p, q),
            other => panic!("expected Two, got {:?}", other),
        };
        let v_plus = vertex_point(c1, c2, RootSign::Plus).expect("Plus");
        let v_minus = vertex_point(c1, c2, RootSign::Minus).expect("Minus");
        // The unordered pair {p, q} should equal {v_plus, v_minus}.
        let matches_plus = (v_plus - p).norm() < 1e-12 || (v_plus - q).norm() < 1e-12;
        let matches_minus = (v_minus - p).norm() < 1e-12 || (v_minus - q).norm() < 1e-12;
        assert!(matches_plus && matches_minus);
        assert!((v_plus - v_minus).norm() > 1e-6, "plus and minus must differ");

        // identify_root_sign reproduces both halves.
        assert_eq!(identify_root_sign(c1, c2, v_plus), Some(RootSign::Plus));
        assert_eq!(identify_root_sign(c1, c2, v_minus), Some(RootSign::Minus));
    }
}
