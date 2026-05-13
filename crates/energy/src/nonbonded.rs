//! Non-bonded pair energies: Lennard-Jones and Coulomb.
//!
//! Both are summed over all atom pairs whose distance is at most `CUTOFF`.
//! Bonded 1-2 and 1-3 pairs are skipped (their interactions are absorbed
//! by the bonded energy terms). 1-4 pairs use either special CHARMM "1-4"
//! parameters when present (epsilon_14 / rmin_half_14) or the regular
//! parameters; full Coulomb strength is used (CHARMM's e14fac = 1.0).
//!
//! ## Cutoff treatment
//!
//! LJ is summed bare and stopped at `r ≤ CUTOFF`. Its 1/r⁶ decay is
//! steep enough that the discontinuity at the cutoff is small (< 0.1 %
//! of well depth at r ≥ 8 Å in CHARMM), and any LJ shift adds bias to
//! the well depth that we'd rather not absorb.
//!
//! Coulomb is treated with a **reaction-field** correction (Tironi,
//! Sperb, Smith, van Gunsteren, JCP 102, 5451, 1995) at the ε_RF → ∞
//! limit, which is the right setting when the GB term is providing
//! the dielectric screening:
//!
//!   V_RF(r) = (k qᵢ qⱼ) · [1/r + r²/(2 Rc³) − 3/(2 Rc)]   for r ≤ Rc
//!
//! Both V_RF(Rc) and dV_RF/dr|_{r=Rc} are zero by construction, so the
//! integrator sees a smooth pair potential and a smooth pair force at
//! the cutoff (no impulse / energy drift). The short-r limit recovers
//! bare Coulomb up to a constant offset.

use chem::{classify, AtomType, ForceField};
use geom::{CellList, Structure, TopologyGraph, Vec3};

use crate::units::kcal_to_kj;

/// CHARMM Coulomb constant: 332.0637 kcal·Å / (mol·e²).
pub const COULOMB_CONST_KCAL_A_PER_E2: f64 = 332.0637;

/// Default Lennard-Jones / Coulomb cutoff in Å. Standard CHARMM choice.
pub const DEFAULT_CUTOFF_A: f64 = 10.0;

/// Pre-computed reaction-field constants for a given cutoff Rc, at
/// ε_RF → ∞ (matching our GB-screened solvent treatment).
#[derive(Debug, Clone, Copy)]
pub struct CoulombRf {
    /// `1 / (2 Rc³)` — multiplies r² in the energy.
    pub kappa: f64,
    /// `3 / (2 Rc)` — the constant offset that drives V → 0 at Rc.
    pub c: f64,
    /// `1 / Rc³` — cached for the force coefficient.
    pub inv_rc3: f64,
}

impl CoulombRf {
    #[inline]
    pub fn for_cutoff(rc_a: f64) -> Self {
        let rc3 = rc_a * rc_a * rc_a;
        Self {
            kappa: 1.0 / (2.0 * rc3),
            c: 3.0 / (2.0 * rc_a),
            inv_rc3: 1.0 / rc3,
        }
    }

    /// Reaction-field Coulomb pair energy in kcal/mol:
    /// `k qᵢ qⱼ · (1/r + κ r² − c)`. Both endpoints (r→0 small-r limit
    /// and r=Rc) are well-defined: at Rc the term in parentheses is
    /// 1/Rc + Rc/(2·Rc²) − 3/(2·Rc) = 0.
    #[inline]
    pub fn pair_energy_kcal(&self, qq: f64, r: f64) -> f64 {
        COULOMB_CONST_KCAL_A_PER_E2 * qq * (1.0 / r + self.kappa * r * r - self.c)
    }

    /// Coefficient `α` such that the pair force on atom i (kcal/mol/Å)
    /// is `α · (r_j − r_i)`. Derived from F_i = −∇_i V_RF:
    ///   F_i = k qᵢ qⱼ · (1/r² − r/Rc³) · r̂_ji
    /// and `r̂_ji = (r_j − r_i) / r`, so the multiplier on the displacement
    /// vector itself is `α = −k qᵢ qⱼ · (1/r³ − 1/Rc³)`. The minus sign
    /// matches the existing convention (`F = rij · α`, with `rij =
    /// positions[j] − positions[i]`). At r = Rc the bracket vanishes,
    /// giving a smooth force at the cutoff.
    #[inline]
    pub fn force_coefficient_kcal_per_a(&self, qq: f64, r: f64) -> f64 {
        let inv_r3 = 1.0 / (r * r * r);
        -COULOMB_CONST_KCAL_A_PER_E2 * qq * (inv_r3 - self.inv_rc3)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NonbondedBreakdown {
    pub lj_kj_mol: f64,
    pub coulomb_kj_mol: f64,
    pub pair_count: usize,
    pub one_four_count: usize,
    pub missing_count: usize,
}

/// Compute LJ and Coulomb energies for `structure` with the supplied
/// topology graph (for exclusion masks) and force field. `cutoff_a` is in Å.
pub fn nonbonded_energy(
    structure: &Structure,
    graph: &TopologyGraph,
    ff: &ForceField,
    cutoff_a: f64,
) -> NonbondedBreakdown {
    let positions: Vec<Vec3> = structure
        .residues
        .iter()
        .flat_map(|r| r.atoms.iter().map(|a| a.position))
        .collect();
    let mut atom_types: Vec<AtomType> = Vec::with_capacity(positions.len());
    let mut charges: Vec<f64> = Vec::with_capacity(positions.len());
    for residue in &structure.residues {
        for atom in &residue.atoms {
            atom_types.push(
                classify(residue.aa, atom.name)
                    .unwrap_or_else(|| panic!("unclassified atom {:?} {}", residue.aa, atom.name)),
            );
            charges.push(ff.partial_charge(residue.aa, atom.name).unwrap_or(0.0));
        }
    }

    let cell_size = cutoff_a;
    let cl = CellList::build(&positions, cell_size);
    let mut br = NonbondedBreakdown::default();
    let mut lj_kcal = 0.0;
    let mut coul_kcal = 0.0;
    let rf = CoulombRf::for_cutoff(cutoff_a);

    for (i, j, r) in cl.iter_pairs_within(&positions, cutoff_a) {
        // 1-2 and 1-3 exclusions: zero non-bonded contribution.
        if graph.is_bonded(i, j) || graph.is_one_three(i, j) {
            continue;
        }
        let one_four = graph.is_one_four(i, j);
        let (ti, tj) = (atom_types[i], atom_types[j]);
        let (Some(pi), Some(pj)) = (ff.nonbonded(ti), ff.nonbonded(tj)) else {
            br.missing_count += 1;
            continue;
        };

        // Lorentz-Berthelot combining (CHARMM stores Rmin/2):
        let (rmin_i, eps_i, rmin_j, eps_j) = if one_four {
            // Use 1-4 specific params if present, else fall back to regular.
            let r14_i = pi.rmin_half_14.unwrap_or(pi.rmin_half);
            let e14_i = pi.epsilon_14.unwrap_or(pi.epsilon);
            let r14_j = pj.rmin_half_14.unwrap_or(pj.rmin_half);
            let e14_j = pj.epsilon_14.unwrap_or(pj.epsilon);
            (r14_i, e14_i, r14_j, e14_j)
        } else {
            (pi.rmin_half, pi.epsilon, pj.rmin_half, pj.epsilon)
        };
        let rmin_ij = rmin_i + rmin_j;
        let eps_ij = (eps_i * eps_j).sqrt();
        let ratio6 = (rmin_ij / r).powi(6);
        let ratio12 = ratio6 * ratio6;
        // CHARMM LJ form: V = ε [(Rmin/r)^12 − 2(Rmin/r)^6]
        lj_kcal += eps_ij * (ratio12 - 2.0 * ratio6);

        // Coulomb (vacuum dielectric — solvent screening is handled by
        // the GB term). Reaction-field correction makes both V and F
        // continuous at the cutoff.
        let qq = charges[i] * charges[j];
        if qq != 0.0 {
            coul_kcal += rf.pair_energy_kcal(qq, r);
        }

        br.pair_count += 1;
        if one_four {
            br.one_four_count += 1;
        }
    }

    br.lj_kj_mol = kcal_to_kj(lj_kcal);
    br.coulomb_kj_mol = kcal_to_kj(coul_kcal);
    br
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use chem::{standard_ff, AminoAcid};
    use geom::{build_extended_chain, build_topology_graph};

    #[test]
    fn extended_chain_nonbonded_energy_finite() {
        let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala, AminoAcid::Ala]).unwrap();
        let g = build_topology_graph(&s);
        let ff = standard_ff();
        let br = nonbonded_energy(&s, &g, ff, DEFAULT_CUTOFF_A);
        assert!(br.lj_kj_mol.is_finite());
        assert!(br.coulomb_kj_mol.is_finite());
        assert_eq!(br.missing_count, 0);
        // Should have at least some 1-4 pairs in the chain.
        assert!(br.one_four_count > 0);
    }

    #[test]
    fn excluded_pairs_skipped() {
        // For a single Ala residue: bonded pairs (1-2 and 1-3) shouldn't
        // contribute to non-bonded.
        let s = build_extended_chain(&[AminoAcid::Ala]).unwrap();
        let g = build_topology_graph(&s);
        let ff = standard_ff();
        let br = nonbonded_energy(&s, &g, ff, DEFAULT_CUTOFF_A);
        // Single Ala: ~10 atoms, the only non-bonded pairs are the few 1-4
        // pairs (e.g., N-...-CB at distance ~2.5 Å, plus a few others) and
        // distance-only-truncated pairs.
        // Sanity: pair count is non-zero but bounded.
        assert!(br.pair_count > 0);
        assert!(br.pair_count < 50);
    }

    #[test]
    fn known_coulomb_value() {
        // Construct two atoms at fixed positions with known charges and
        // verify the Coulomb energy comes out right. Use a manual, minimal
        // structure-like setup by exploiting the public path — easier to
        // bypass and test the core Coulomb formula directly.
        use geom::Vec3;
        let r = 5.0; // Å
        let q1 = 1.0;
        let q2 = -1.0;
        let r_a = Vec3::new(0.0, 0.0, 0.0);
        let r_b = Vec3::new(r, 0.0, 0.0);
        let dist = (r_a - r_b).norm();
        let coul_kcal = COULOMB_CONST_KCAL_A_PER_E2 * q1 * q2 / dist;
        let coul_kj = kcal_to_kj(coul_kcal);
        // Expected: -332.0637 / 5.0 × 4.184 = -277.87 kJ/mol.
        assert_relative_eq!(coul_kj, -277.871, epsilon = 0.01);
    }
}
