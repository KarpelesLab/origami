//! Generalized-Born implicit solvent — OBC II (Onufriev, Bashford, Case 2004).
//!
//! Two stages:
//! 1. **Effective Born radii**. For each atom i, sum a pairwise descreening
//!    integral over all other atoms, then map to an effective radius via
//!    the OBC II tanh transformation.
//! 2. **Pair energy**. Sum over all atom pairs (including self-pairs i = j)
//!    of `−½ (1 − 1/εw) qᵢqⱼ / fGB(rᵢⱼ, Rᵢ, Rⱼ)`.
//!
//! Reference: Onufriev, Bashford, Case (2004) "Exploring protein native
//! states and large-scale conformational changes with a modified
//! generalized born model" Proteins 55(2):383–394.

use chem::{AminoAcid, AtomType, Element, ForceField};
use geom::{CellList, Structure, Vec3};

use crate::units::kcal_to_kj;

/// CHARMM Coulomb constant — same as in `nonbonded`.
const COULOMB_CONST_KCAL_A_PER_E2: f64 = 332.0637;

/// Water dielectric constant at 298 K (standard convention).
const EPSILON_WATER: f64 = 78.5;
/// Vacuum dielectric (we model the solute as ε = 1).
const EPSILON_SOLUTE: f64 = 1.0;

/// OBC II constants (Onufriev-Bashford-Case 2004, table I).
const OBC_ALPHA: f64 = 1.0;
const OBC_BETA: f64 = 0.8;
const OBC_GAMMA: f64 = 4.85;
/// Born radius offset (Å). Subtracted from ρ to give ρ̃.
const OBC_OFFSET: f64 = 0.09;

/// Cutoff distance for the descreening sum (Å). Beyond this, atoms
/// contribute negligibly to the effective Born radius.
const BORN_RADIUS_CUTOFF_A: f64 = 20.0;

#[derive(Debug, Default, Clone, Copy)]
pub struct GbBreakdown {
    pub gb_kj_mol: f64,
    /// Per-atom self-energy contribution (i = j): −½(1−1/εw) Σ qᵢ²/Rᵢ.
    pub self_kj_mol: f64,
    /// Cross-pair contribution (i ≠ j).
    pub pair_kj_mol: f64,
    /// Atoms whose effective Born radius came out unphysical (clamped to a
    /// minimum). A non-zero count usually means the descreening integration
    /// produced a near-zero or negative inverse radius for an atom buried
    /// under many neighbours. The clamp prevents NaN downstream.
    pub clamped_count: usize,
}

/// Default per-element intrinsic Born radius (Å). These are Bondi vdW radii
/// — reasonable defaults when CHARMM doesn't supply atom-specific values.
fn intrinsic_radius(element: Element) -> f64 {
    match element {
        Element::H => 1.20,
        Element::C => 1.70,
        Element::N => 1.55,
        Element::O => 1.50,
        Element::S => 1.80,
    }
}

/// HCT scaling factor for the descreening integral. AMBER's mbondi2 set.
fn hct_scale(element: Element) -> f64 {
    match element {
        Element::H => 0.85,
        Element::C => 0.72,
        Element::N => 0.79,
        Element::O => 0.85,
        Element::S => 0.96,
    }
}

#[allow(dead_code)]
fn _atom_type_unused(_t: AtomType) {} // silence unused import warning if AtomType ends up unused

/// Factored-out per-atom data that both the energy and the force code need.
pub struct BornInputs {
    pub positions: Vec<Vec3>,
    pub charges: Vec<f64>,
    pub effective_radii: Vec<f64>,
    pub clamped_count: usize,
}

/// Compute effective Born radii for the given structure (the radii are what
/// the M4 force code treats as constants under the frozen-radii
/// approximation). Returns positions/charges as well so callers don't have
/// to re-flatten the structure.
pub fn compute_born_inputs(structure: &Structure, ff: &ForceField) -> BornInputs {
    let mut positions: Vec<Vec3> = Vec::with_capacity(structure.atom_count());
    let mut charges: Vec<f64> = Vec::with_capacity(structure.atom_count());
    let mut rho: Vec<f64> = Vec::with_capacity(structure.atom_count());
    let mut rho_tilde: Vec<f64> = Vec::with_capacity(structure.atom_count());
    let mut scale: Vec<f64> = Vec::with_capacity(structure.atom_count());

    for residue in &structure.residues {
        for atom in &residue.atoms {
            positions.push(atom.position);
            charges.push(charge_for(ff, residue.aa, atom.name));
            let r = intrinsic_radius(atom.element);
            rho.push(r);
            rho_tilde.push(r - OBC_OFFSET);
            scale.push(hct_scale(atom.element));
        }
    }

    let n = positions.len();
    let cl = CellList::build(&positions, BORN_RADIUS_CUTOFF_A);
    let mut integral = vec![0.0_f64; n];
    for (i, j, r) in cl.iter_pairs_within(&positions, BORN_RADIUS_CUTOFF_A) {
        let h_ij = pairwise_descreening(r, rho_tilde[i], scale[j] * rho_tilde[j]);
        let h_ji = pairwise_descreening(r, rho_tilde[j], scale[i] * rho_tilde[i]);
        integral[i] += h_ij;
        integral[j] += h_ji;
    }

    let mut effective: Vec<f64> = Vec::with_capacity(n);
    let mut clamped = 0usize;
    for i in 0..n {
        let psi = integral[i] * rho_tilde[i];
        let tanh_arg = OBC_ALPHA * psi - OBC_BETA * psi * psi + OBC_GAMMA * psi * psi * psi;
        let inv = 1.0 / rho_tilde[i] - tanh_arg.tanh() / rho[i];
        let r_eff = if inv <= 0.0 || !inv.is_finite() {
            clamped += 1;
            rho_tilde[i].max(0.5)
        } else {
            (1.0 / inv).max(rho_tilde[i].max(0.5))
        };
        effective.push(r_eff);
    }

    BornInputs { positions, charges, effective_radii: effective, clamped_count: clamped }
}

pub fn gb_energy(structure: &Structure, ff: &ForceField) -> GbBreakdown {
    let BornInputs { positions, charges, effective_radii: effective, clamped_count: clamped } =
        compute_born_inputs(structure, ff);
    let n = positions.len();

    // Stage 3: GB pair-energy sum (including self-terms).
    let prefactor = -0.5 * (1.0 / EPSILON_SOLUTE - 1.0 / EPSILON_WATER) * COULOMB_CONST_KCAL_A_PER_E2;
    let mut self_kcal = 0.0;
    for i in 0..n {
        if charges[i] != 0.0 {
            let q2 = charges[i] * charges[i];
            self_kcal += prefactor * q2 / effective[i];
        }
    }
    let mut pair_kcal = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let qq = charges[i] * charges[j];
            if qq == 0.0 {
                continue;
            }
            let r2 = (positions[i] - positions[j]).norm_squared();
            let r_eff_prod = effective[i] * effective[j];
            let f_gb = (r2 + r_eff_prod * (-r2 / (4.0 * r_eff_prod)).exp()).sqrt();
            // Factor of 2 because the symmetric pair sum cancels the leading ½.
            pair_kcal += 2.0 * prefactor * qq / f_gb;
        }
    }

    let gb_kcal = self_kcal + pair_kcal;
    GbBreakdown {
        gb_kj_mol: kcal_to_kj(gb_kcal),
        self_kj_mol: kcal_to_kj(self_kcal),
        pair_kj_mol: kcal_to_kj(pair_kcal),
        clamped_count: clamped,
    }
}

fn charge_for(ff: &ForceField, aa: AminoAcid, atom_name: &str) -> f64 {
    ff.partial_charge(aa, atom_name).unwrap_or(0.0)
}

/// Pairwise descreening integral (HCT/OBC form).
/// `r` = inter-atomic distance; `rho_i_tilde` = atom i's reduced radius;
/// `s_rho_j_tilde` = atom j's scaled reduced radius.
fn pairwise_descreening(r: f64, rho_i_tilde: f64, s_rho_j_tilde: f64) -> f64 {
    // Atom j fully inside atom i: no descreening contribution.
    if r + s_rho_j_tilde <= rho_i_tilde {
        return 0.0;
    }
    // Standard non-overlapping form (atom j outside atom i).
    let l = if r - s_rho_j_tilde < rho_i_tilde {
        rho_i_tilde
    } else {
        r - s_rho_j_tilde
    };
    let u = r + s_rho_j_tilde;
    if u <= 0.0 || l <= 0.0 {
        return 0.0;
    }
    let inv_l = 1.0 / l;
    let inv_u = 1.0 / u;
    let term1 = 0.5 * (inv_l - inv_u);
    let term2 = (r / 4.0) * (inv_u * inv_u - inv_l * inv_l);
    let term3 = (1.0 / (2.0 * r)) * (l / u).ln();
    let term4 = (s_rho_j_tilde * s_rho_j_tilde - r * r) / (4.0 * r) * (inv_u * inv_u - inv_l * inv_l);
    term1 + term2 + term3 + term4
}

#[cfg(test)]
mod tests {
    use super::*;
    use chem::{standard_ff, AminoAcid};
    use geom::build_extended_chain;

    #[test]
    fn isolated_chain_self_energy_finite() {
        // Even with no charges (test passes regardless), the GB calc shouldn't NaN.
        let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala, AminoAcid::Ala]).unwrap();
        let ff = standard_ff();
        let br = gb_energy(&s, ff);
        assert!(br.gb_kj_mol.is_finite(), "GB total {} not finite", br.gb_kj_mol);
        assert!(br.self_kj_mol <= 0.0, "self energy should be non-positive (favourable)");
        assert_eq!(br.clamped_count, 0, "no atoms should clamp for a small extended chain");
    }

    #[test]
    fn polar_chain_has_negative_total_gb() {
        // A chain with charged residues (Lys-Glu) should have favourable
        // (negative) GB solvation — the formula's prefactor is negative,
        // and self-energies dominate for unscreened isolated charges.
        let s = build_extended_chain(&[AminoAcid::Lys, AminoAcid::Glu]).unwrap();
        let ff = standard_ff();
        let br = gb_energy(&s, ff);
        assert!(br.gb_kj_mol < 0.0, "GB total should be negative, got {}", br.gb_kj_mol);
    }

    #[test]
    fn pairwise_descreening_zero_for_buried_atom() {
        // If atom j is fully inside atom i, no descreening.
        let h = pairwise_descreening(0.5, 2.0, 0.5);
        assert_eq!(h, 0.0);
    }
}
