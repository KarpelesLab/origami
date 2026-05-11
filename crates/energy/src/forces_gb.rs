//! GB OBC II forces under the frozen-radii approximation.
//!
//! Treats the effective Born radii Rᵢ as constants over a single force
//! evaluation; the minimization driver recomputes Rᵢ before each step.
//! Under that approximation only the pair sum
//!     E = −(1 − 1/εw) × kCoul × Σᵢ<ⱼ qᵢqⱼ / fGB(rᵢⱼ, Rᵢ, Rⱼ)
//! contributes to ∂E/∂r (the self-energy is r-independent under frozen
//! radii).
//!
//! The pair derivative:
//!   fGB² = r² + RᵢRⱼ × exp(−r²/(4 RᵢRⱼ))
//!   dfGB²/dr = 2r − (r/2) × exp(−r²/(4 RᵢRⱼ))
//!   dfGB/dr  = (1/(2 fGB)) × dfGB²/dr
//! and F_i = +(1−1/εw) × kCoul × qᵢqⱼ / fGB² × dfGB/dr × (r_j − r_i)/r.

use chem::ForceField;
use geom::{Structure, Vec3};

use crate::gb::compute_born_inputs;
use crate::units::kcal_to_kj;

const COULOMB_CONST_KCAL_A_PER_E2: f64 = 332.0637;
const EPSILON_WATER: f64 = 78.5;
const EPSILON_SOLUTE: f64 = 1.0;

pub fn add_gb_forces(structure: &Structure, ff: &ForceField, forces: &mut [Vec3]) {
    let inputs = compute_born_inputs(structure, ff);
    let n = inputs.positions.len();
    let prefactor_kcal = (1.0 / EPSILON_SOLUTE - 1.0 / EPSILON_WATER) * COULOMB_CONST_KCAL_A_PER_E2;
    let positions = &inputs.positions;
    let charges = &inputs.charges;
    let radii = &inputs.effective_radii;
    // Outer-loop skip on q_i = 0 removes ~half the iterations cheaply
    // (CHARMM gives many hydrocarbons q ≈ 0 on the heavy atom). The
    // pre-converted Coulomb prefactor avoids re-multiplying by the
    // kcal→kJ factor inside the hot loop.
    let prefactor_kj = kcal_to_kj(prefactor_kcal);
    for i in 0..n {
        let qi = charges[i];
        if qi == 0.0 {
            continue;
        }
        let pi = positions[i];
        let ri = radii[i];
        for j in (i + 1)..n {
            let qj = charges[j];
            if qj == 0.0 {
                continue;
            }
            let qq = qi * qj;
            let r_ij_vec = positions[j] - pi;
            let r2 = r_ij_vec.norm_squared();
            if r2 < 1e-18 {
                continue;
            }
            let r = r2.sqrt();
            let rij_prod = ri * radii[j];
            let exp_val = (-r2 / (4.0 * rij_prod)).exp();
            let f_gb_sq = r2 + rij_prod * exp_val;
            let f_gb = f_gb_sq.sqrt();
            let d_fgb_sq_dr = 2.0 * r - 0.5 * r * exp_val;
            let d_fgb_dr = d_fgb_sq_dr / (2.0 * f_gb);
            // F_i = +(1−1/εw) k qq / fGB² × dfGB/dr × (r_j − r_i)/r
            let coeff = prefactor_kj * qq / f_gb_sq * d_fgb_dr / r;
            let f = r_ij_vec * coeff;
            forces[i] += f;
            forces[j] -= f;
        }
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::gb::gb_energy;
    use chem::{standard_ff, AminoAcid};
    use geom::build_extended_chain;

    #[test]
    fn gb_force_finite_difference() {
        // Use a Lys (positive charge) + Glu (negative charge) chain where GB
        // forces are non-trivial.
        let s = build_extended_chain(&[AminoAcid::Lys, AminoAcid::Glu]).unwrap();
        let ff = standard_ff();
        let n = s.atom_count();
        let mut forces = vec![Vec3::zeros(); n];
        add_gb_forces(&s, ff, &mut forces);
        // Compare a few atoms against finite differences of *frozen-radii*
        // GB energy. To get frozen radii in the central-difference, compute
        // radii from the unperturbed structure and use them for both ± sides.
        let inputs = compute_born_inputs(&s, ff);
        let prefactor_kj = kcal_to_kj(
            -0.5 * (1.0 / EPSILON_SOLUTE - 1.0 / EPSILON_WATER) * COULOMB_CONST_KCAL_A_PER_E2,
        );
        let frozen_radii = inputs.effective_radii.clone();
        let charges = inputs.charges.clone();

        let energy_with_radii = |s: &geom::Structure| -> f64 {
            let mut positions: Vec<Vec3> = Vec::new();
            for r in &s.residues {
                for a in &r.atoms {
                    positions.push(a.position);
                }
            }
            let mut self_e = 0.0;
            for i in 0..n {
                let q2 = charges[i] * charges[i];
                self_e += prefactor_kj * q2 / frozen_radii[i];
            }
            let mut pair_e = 0.0;
            for i in 0..n {
                for j in (i + 1)..n {
                    let qq = charges[i] * charges[j];
                    if qq == 0.0 { continue; }
                    let r2 = (positions[i] - positions[j]).norm_squared();
                    let rprod = frozen_radii[i] * frozen_radii[j];
                    let f_gb = (r2 + rprod * (-r2 / (4.0 * rprod)).exp()).sqrt();
                    pair_e += 2.0 * prefactor_kj * qq / f_gb;
                }
            }
            self_e + pair_e
        };

        let eps = 1e-5;
        for i in 0..n.min(10) {
            for axis in 0..3 {
                let mut s_plus = s.clone();
                let mut s_minus = s.clone();
                bump(&mut s_plus, i, axis, eps);
                bump(&mut s_minus, i, axis, -eps);
                let e_plus = energy_with_radii(&s_plus);
                let e_minus = energy_with_radii(&s_minus);
                let numeric = -(e_plus - e_minus) / (2.0 * eps);
                let an = forces[i][axis];
                assert!(
                    (an - numeric).abs() < 1e-1,
                    "atom {} axis {}: analytical={:.4}, numeric={:.4}", i, axis, an, numeric
                );
            }
        }
    }

    fn bump(s: &mut geom::Structure, atom_idx: usize, axis: usize, eps: f64) {
        let mut count = 0usize;
        for residue in &mut s.residues {
            for atom in &mut residue.atoms {
                if count == atom_idx {
                    atom.position[axis] += eps;
                    return;
                }
                count += 1;
            }
        }
    }

    #[test]
    fn gb_force_finite_for_chain() {
        let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala, AminoAcid::Ala]).unwrap();
        let ff = standard_ff();
        let mut forces = vec![Vec3::zeros(); s.atom_count()];
        add_gb_forces(&s, ff, &mut forces);
        for f in &forces {
            assert!(f.norm().is_finite());
        }
        let _ = gb_energy(&s, ff); // sanity: energy still works
    }
}
