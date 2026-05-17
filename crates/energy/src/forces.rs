//! Force aggregator — combine bonded + LJ + Coulomb + GB (and optionally
//! SASA) into one `Vec<Vec3>` of forces (kJ/mol/Å) ready for the optimiser
//! or integrator.

use chem::ForceField;
use geom::{Structure, TopologyGraph, Vec3};

use crate::forces_bonded::{
    add_angle_forces, add_bond_forces, add_dihedral_forces, add_improper_forces, build_atom_types,
};
use crate::forces_gb::add_gb_forces;
use crate::forces_nonbonded::add_nonbonded_forces;
use crate::forces_sasa::add_sasa_forces;
use crate::nonbonded::DEFAULT_CUTOFF_A;

/// Compute the total atomic force vector (without SASA forces — preserves
/// the M4 force-aggregator behaviour). Length equals `structure.atom_count()`.
pub fn total_force(
    structure: &Structure,
    graph: &TopologyGraph,
    ff: &ForceField,
) -> Vec<Vec3> {
    total_force_with_cutoff(structure, graph, ff, DEFAULT_CUTOFF_A)
}

pub fn total_force_with_cutoff(
    structure: &Structure,
    graph: &TopologyGraph,
    ff: &ForceField,
    cutoff_a: f64,
) -> Vec<Vec3> {
    total_force_with_options(structure, graph, ff, cutoff_a, false)
}

/// Compute the total atomic force vector with optional SASA contribution.
/// PSA.2 — when `include_sasa` is true, `add_sasa_forces` is called and
/// the result includes the hydrophobic gradient. SASA forces are slow
/// (numerical central differencing), so this is opt-in.
pub fn total_force_with_options(
    structure: &Structure,
    graph: &TopologyGraph,
    ff: &ForceField,
    cutoff_a: f64,
    include_sasa: bool,
) -> Vec<Vec3> {
    let n = structure.atom_count();
    let mut forces = vec![Vec3::zeros(); n];
    let atom_types = build_atom_types(structure);
    let positions: Vec<Vec3> = structure
        .residues
        .iter()
        .flat_map(|r| r.atoms.iter().map(|a| a.position))
        .collect();
    add_bond_forces(&positions, graph, ff, &atom_types, &mut forces);
    add_angle_forces(&positions, graph, ff, &atom_types, &mut forces);
    add_dihedral_forces(&positions, graph, ff, &atom_types, &mut forces);
    add_improper_forces(&positions, graph, ff, &atom_types, &mut forces);
    add_nonbonded_forces(structure, graph, ff, cutoff_a, &mut forces);
    add_gb_forces(structure, ff, &mut forces);
    if include_sasa {
        add_sasa_forces(structure, ff, &mut forces);
    }
    forces
}

/// SoA-aware force aggregator. Same physics as `total_force_with_options`
/// but the nonbonded LJ+Coulomb pair sum is computed through the SoA
/// kernel that reads from `scratch`. The caller owns `scratch` and
/// `forces` and reuses them across steps so the per-step allocation cost
/// drops to zero.
///
/// `scratch.rebuild_params` and `scratch.rebuild_exclusions` must already
/// have run (e.g. via `ForceScratch::new`). Positions are re-synced
/// inside this call from `structure`, so the caller does not need to
/// pre-sync.
///
/// `forces` is overwritten — caller does not need to zero it.
pub fn total_force_with_scratch(
    structure: &Structure,
    graph: &TopologyGraph,
    ff: &ForceField,
    cutoff_a: f64,
    include_sasa: bool,
    scratch: &mut crate::scratch::ForceScratch,
    forces: &mut Vec<Vec3>,
) {
    let n = structure.atom_count();
    debug_assert_eq!(scratch.n, n, "scratch sized for a different structure");
    if forces.len() != n {
        forces.clear();
        forces.resize(n, Vec3::zeros());
    } else {
        forces.iter_mut().for_each(|f| *f = Vec3::zeros());
    }

    let atom_types = build_atom_types(structure);
    let positions: Vec<Vec3> = structure
        .residues
        .iter()
        .flat_map(|r| r.atoms.iter().map(|a| a.position))
        .collect();
    add_bond_forces(&positions, graph, ff, &atom_types, forces);
    add_angle_forces(&positions, graph, ff, &atom_types, forces);
    add_dihedral_forces(&positions, graph, ff, &atom_types, forces);
    add_improper_forces(&positions, graph, ff, &atom_types, forces);

    // SoA nonbonded pair loop: sync + zero scratch, run, accumulate
    // back into the AoS output buffer.
    scratch.sync_positions(structure);
    scratch.zero_forces();
    crate::forces_nonbonded::add_nonbonded_forces_soa(scratch, cutoff_a);
    // SoA GB pair force — also goes through the scratch (its Born-
    // radius compute and pair loop both use the cached Verlet pair
    // list and the SoA position arrays).
    crate::forces_gb::add_gb_forces_soa(
        scratch,
        structure,
        ff,
        crate::forces_gb::GB_DEFAULT_CUTOFF_A_PUB,
    );
    scratch.accumulate_into(forces);

    if include_sasa {
        crate::powersasa::analytical::add_sasa_forces_analytical_with_scratch(
            structure, ff, scratch, forces,
        );
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::bonded::bonded_energy;
    use crate::gb::compute_born_inputs;
    use crate::nonbonded::nonbonded_energy;
    use crate::units::kcal_to_kj;
    use chem::{standard_ff, AminoAcid};
    use geom::{build_extended_chain, build_topology_graph};

    fn bump(s: &mut Structure, atom_idx: usize, axis: usize, eps: f64) {
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

    /// Total energy without SASA, evaluated under frozen Born radii so the
    /// numerical gradient matches the M4 force convention exactly.
    fn total_energy_for_force_check(
        structure: &Structure,
        graph: &TopologyGraph,
        ff: &ForceField,
        frozen_charges: &[f64],
        frozen_radii: &[f64],
    ) -> f64 {
        let bonded = bonded_energy(structure, graph, ff);
        let nb = nonbonded_energy(structure, graph, ff, DEFAULT_CUTOFF_A);
        // GB pair sum + self with frozen radii.
        let n = frozen_radii.len();
        // Solute ε = 1, water ε = 78.5 (matching gb.rs).
        let prefactor_kj = kcal_to_kj(-0.5 * (1.0 - 1.0_f64 / 78.5) * 332.0637);
        let mut positions: Vec<Vec3> = Vec::with_capacity(n);
        for r in &structure.residues {
            for a in &r.atoms {
                positions.push(a.position);
            }
        }
        let mut self_e = 0.0;
        for i in 0..n {
            let q2 = frozen_charges[i] * frozen_charges[i];
            self_e += prefactor_kj * q2 / frozen_radii[i];
        }
        // The reference energy honours the same GB cutoff the force
        // code uses (see forces_gb::GB_DEFAULT_CUTOFF_A); otherwise
        // numeric vs analytical gradients would differ by the
        // truncated long-range tail.
        let gb_cutoff_sq = DEFAULT_CUTOFF_A * DEFAULT_CUTOFF_A;
        let mut pair_e = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let qq = frozen_charges[i] * frozen_charges[j];
                if qq == 0.0 { continue; }
                let r2 = (positions[i] - positions[j]).norm_squared();
                if r2 > gb_cutoff_sq { continue; }
                let rprod = frozen_radii[i] * frozen_radii[j];
                let f_gb = (r2 + rprod * (-r2 / (4.0 * rprod)).exp()).sqrt();
                pair_e += 2.0 * prefactor_kj * qq / f_gb;
            }
        }
        bonded.total_kj_mol() + nb.lj_kj_mol + nb.coulomb_kj_mol + self_e + pair_e
    }

    #[test]
    fn total_force_finite_difference() {
        let s = build_extended_chain(&[
            AminoAcid::Lys, AminoAcid::Ala, AminoAcid::Glu,
        ]).unwrap();
        let g = build_topology_graph(&s);
        let ff = standard_ff();
        let analytical = total_force(&s, &g, ff);
        let inputs = compute_born_inputs(&s, ff);
        let frozen_charges = inputs.charges.clone();
        let frozen_radii = inputs.effective_radii.clone();

        let n = s.atom_count();
        let eps = 1e-5;
        let mut max_err: f64 = 0.0;
        let mut max_label = String::new();
        for i in 0..n.min(15) {
            for axis in 0..3 {
                let mut s_plus = s.clone();
                let mut s_minus = s.clone();
                bump(&mut s_plus, i, axis, eps);
                bump(&mut s_minus, i, axis, -eps);
                let e_plus = total_energy_for_force_check(&s_plus, &g, ff, &frozen_charges, &frozen_radii);
                let e_minus = total_energy_for_force_check(&s_minus, &g, ff, &frozen_charges, &frozen_radii);
                let numeric = -(e_plus - e_minus) / (2.0 * eps);
                let an = analytical[i][axis];
                let err = (an - numeric).abs();
                if err > max_err {
                    max_err = err;
                    max_label = format!("atom {} axis {}: analytical={:.4}, numeric={:.4}", i, axis, an, numeric);
                }
                assert!(
                    err < 1.0,
                    "{}", max_label
                );
            }
        }
        eprintln!("max force discrepancy in total_force test: {} ({})", max_err, max_label);
    }

    #[test]
    fn total_force_with_scratch_matches_aos() {
        let s = build_extended_chain(&[
            AminoAcid::Lys, AminoAcid::Ala, AminoAcid::Glu, AminoAcid::Phe,
        ]).unwrap();
        let g = build_topology_graph(&s);
        let ff = standard_ff();
        let aos = total_force_with_options(&s, &g, ff, DEFAULT_CUTOFF_A, false);

        let mut scratch = crate::scratch::ForceScratch::new(&s, &g, ff);
        let mut soa = Vec::new();
        super::total_force_with_scratch(&s, &g, ff, DEFAULT_CUTOFF_A, false, &mut scratch, &mut soa);

        assert_eq!(aos.len(), soa.len());
        for (i, (a, b)) in aos.iter().zip(soa.iter()).enumerate() {
            assert!((a.x - b.x).abs() < 1e-9, "atom {i} x: AoS={:.6e} SoA={:.6e}", a.x, b.x);
            assert!((a.y - b.y).abs() < 1e-9, "atom {i} y: AoS={:.6e} SoA={:.6e}", a.y, b.y);
            assert!((a.z - b.z).abs() < 1e-9, "atom {i} z: AoS={:.6e} SoA={:.6e}", a.z, b.z);
        }
    }
}
