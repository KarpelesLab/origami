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
use geom::{CellList, Structure, Vec3};

use crate::gb::compute_born_inputs;
use crate::nonbonded::DEFAULT_CUTOFF_A;
use crate::units::kcal_to_kj;

const COULOMB_CONST_KCAL_A_PER_E2: f64 = 332.0637;
const EPSILON_WATER: f64 = 78.5;
const EPSILON_SOLUTE: f64 = 1.0;

/// Default cutoff for the GB pair sum. At large r the GB pair term
/// asymptotes to the screened Coulomb form −(1 − 1/εw)·k·qᵢqⱼ/r — the
/// missing piece needed to cancel the bare Coulomb contribution we'd
/// otherwise be double-counting against the LJ+Coulomb pair sum (also
/// cut off at this distance). Using the same cutoff for both keeps
/// the two electrostatic terms self-consistent.
const GB_DEFAULT_CUTOFF_A: f64 = DEFAULT_CUTOFF_A;

pub fn add_gb_forces(structure: &Structure, ff: &ForceField, forces: &mut [Vec3]) {
    add_gb_forces_with_cutoff(structure, ff, forces, GB_DEFAULT_CUTOFF_A);
}

pub fn add_gb_forces_with_cutoff(
    structure: &Structure,
    ff: &ForceField,
    forces: &mut [Vec3],
    cutoff_a: f64,
) {
    let inputs = compute_born_inputs(structure, ff);
    let n = inputs.positions.len();
    let positions = &inputs.positions;
    let charges = &inputs.charges;
    let radii = &inputs.effective_radii;
    let prefactor_kcal = (1.0 / EPSILON_SOLUTE - 1.0 / EPSILON_WATER) * COULOMB_CONST_KCAL_A_PER_E2;
    let prefactor_kj = kcal_to_kj(prefactor_kcal);

    // For small N (< CELL_LIST_THRESHOLD) the cell-list construction
    // and iteration cost exceeds the savings from skipping out-of-
    // cutoff pairs — a simple O(N²) sweep with an early-exit on
    // r² > cutoff² wins. For larger N the asymptotic O(N×neighbours)
    // cell-list wins. Threshold tuned on Apple Silicon for Trp-cage
    // vs villin sizes.
    const CELL_LIST_THRESHOLD: usize = 600;
    let cutoff_sq = cutoff_a * cutoff_a;

    if n < CELL_LIST_THRESHOLD {
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
                let r_ij_vec = positions[j] - pi;
                let r2 = r_ij_vec.norm_squared();
                if r2 > cutoff_sq || r2 < 1e-18 {
                    continue;
                }
                let r = r2.sqrt();
                let rij_prod = ri * radii[j];
                let exp_val = (-r2 / (4.0 * rij_prod)).exp();
                let f_gb_sq = r2 + rij_prod * exp_val;
                let f_gb = f_gb_sq.sqrt();
                let d_fgb_sq_dr = 2.0 * r - 0.5 * r * exp_val;
                let d_fgb_dr = d_fgb_sq_dr / (2.0 * f_gb);
                let coeff = prefactor_kj * (qi * qj) / f_gb_sq * d_fgb_dr / r;
                let f = r_ij_vec * coeff;
                forces[i] += f;
                forces[j] -= f;
            }
        }
    } else {
        let cl = CellList::build(positions, cutoff_a);
        for (i, j, r) in cl.iter_pairs_within(positions, cutoff_a) {
            let qi = charges[i];
            let qj = charges[j];
            if qi == 0.0 || qj == 0.0 || r < 1e-9 {
                continue;
            }
            let r_ij_vec = positions[j] - positions[i];
            let r2 = r * r;
            let rij_prod = radii[i] * radii[j];
            let exp_val = (-r2 / (4.0 * rij_prod)).exp();
            let f_gb_sq = r2 + rij_prod * exp_val;
            let f_gb = f_gb_sq.sqrt();
            let d_fgb_sq_dr = 2.0 * r - 0.5 * r * exp_val;
            let d_fgb_dr = d_fgb_sq_dr / (2.0 * f_gb);
            let coeff = prefactor_kj * (qi * qj) / f_gb_sq * d_fgb_dr / r;
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

        // Reference energy honours the same cutoff the force code uses,
        // so the finite-difference comparison is apples-to-apples.
        let cutoff_sq = GB_DEFAULT_CUTOFF_A * GB_DEFAULT_CUTOFF_A;
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
                    if r2 > cutoff_sq { continue; }
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
