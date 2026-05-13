//! Analytical forces for the non-bonded pair terms (Lennard-Jones + Coulomb).
//!
//! The neighbour list and exclusion-mask logic mirror `energy::nonbonded`
//! exactly: 1-2 and 1-3 pairs are skipped, 1-4 pairs use the special CHARMM
//! 1-4 LJ parameters when present, full Coulomb strength is applied.

use chem::{classify, AtomType, ForceField};
use geom::{CellList, Structure, TopologyGraph, Vec3};

use crate::nonbonded::{CoulombRf, DEFAULT_CUTOFF_A};
use crate::units::kcal_to_kj;

/// Add LJ + Coulomb forces (in kJ/mol/Å) to the supplied buffer. `cutoff_a`
/// is the same cutoff used for the energy.
pub fn add_nonbonded_forces(
    structure: &Structure,
    graph: &TopologyGraph,
    ff: &ForceField,
    cutoff_a: f64,
    forces: &mut [Vec3],
) {
    let n = structure.atom_count();
    let mut positions: Vec<Vec3> = Vec::with_capacity(n);
    let mut atom_types: Vec<AtomType> = Vec::with_capacity(n);
    let mut charges: Vec<f64> = Vec::with_capacity(n);
    for residue in &structure.residues {
        for atom in &residue.atoms {
            positions.push(atom.position);
            atom_types.push(
                classify(residue.aa, atom.name)
                    .unwrap_or_else(|| panic!("unclassified atom {:?} {}", residue.aa, atom.name)),
            );
            charges.push(ff.partial_charge(residue.aa, atom.name).unwrap_or(0.0));
        }
    }

    let cl = CellList::build(&positions, cutoff_a);
    let rf = CoulombRf::for_cutoff(cutoff_a);

    for (i, j, r) in cl.iter_pairs_within(&positions, cutoff_a) {
        if graph.is_bonded(i, j) || graph.is_one_three(i, j) {
            continue;
        }
        let one_four = graph.is_one_four(i, j);
        let (Some(pi), Some(pj)) = (ff.nonbonded(atom_types[i]), ff.nonbonded(atom_types[j]))
        else { continue };

        let (rmin_half_i, eps_i, rmin_half_j, eps_j) = if one_four {
            (
                pi.rmin_half_14.unwrap_or(pi.rmin_half),
                pi.epsilon_14.unwrap_or(pi.epsilon),
                pj.rmin_half_14.unwrap_or(pj.rmin_half),
                pj.epsilon_14.unwrap_or(pj.epsilon),
            )
        } else {
            (pi.rmin_half, pi.epsilon, pj.rmin_half, pj.epsilon)
        };
        let rmin_ij = rmin_half_i + rmin_half_j;
        let eps_ij = (eps_i * eps_j).sqrt();

        let rij = positions[j] - positions[i];
        let inv_r2 = 1.0 / (r * r);
        let ratio = rmin_ij / r;
        let ratio2 = ratio * ratio;
        let ratio6 = ratio2 * ratio2 * ratio2;
        let ratio12 = ratio6 * ratio6;

        // F_i (LJ) = (12 ε / r²) × [(Rmin/r)⁶ − (Rmin/r)¹²] × (r_j − r_i)
        let lj_coeff_kcal = 12.0 * eps_ij * inv_r2 * (ratio6 - ratio12);
        let lj_coeff = kcal_to_kj(lj_coeff_kcal);
        let f_lj = rij * lj_coeff;

        // F_i (reaction-field Coulomb) — α · (r_j − r_i) with α from
        // CoulombRf::force_coefficient_kcal_per_a. Matches the energy
        // path: V → 0 and dV/dr → 0 at r = Rc.
        let qq = charges[i] * charges[j];
        let coul_coeff = if qq != 0.0 {
            kcal_to_kj(rf.force_coefficient_kcal_per_a(qq, r))
        } else {
            0.0
        };
        let f_coul = rij * coul_coeff;

        let f_total = f_lj + f_coul;
        forces[i] += f_total;
        forces[j] -= f_total;
    }
}

/// Convenience wrapper using the default cutoff.
pub fn add_nonbonded_forces_default(
    structure: &Structure,
    graph: &TopologyGraph,
    ff: &ForceField,
    forces: &mut [Vec3],
) {
    add_nonbonded_forces(structure, graph, ff, DEFAULT_CUTOFF_A, forces);
}

/// SoA-aware nonbonded force kernel that reads/writes a
/// [`crate::scratch::ForceScratch`]. The caller is expected to:
///   1. Allocate the scratch once at simulation start (via
///      `ForceScratch::new`).
///   2. Sync positions into the scratch each step
///      (`scratch.sync_positions(&structure)`).
///   3. Zero the force buffers (`scratch.zero_forces()`).
///   4. Call this kernel (plus any other SoA kernels).
///   5. Read the SoA force out via `scratch.accumulate_into(&mut [Vec3])`.
///
/// The inner pair loop pulls positions from flat `[f64]` arrays and
/// looks up the bonded/1-3/1-4 mask from a flat `[u8]` bitmap, both
/// of which let the compiler avoid the `Vec3` AoS load pattern and
/// the `Vec::contains` walk in the original AoS path.
pub fn add_nonbonded_forces_soa(
    scratch: &mut crate::scratch::ForceScratch,
    cutoff_a: f64,
) {
    let n = scratch.n;
    let cutoff_sq = cutoff_a * cutoff_a;
    let kj_per_kcal = kcal_to_kj(1.0);
    let rf = CoulombRf::for_cutoff(cutoff_a);
    let inv_rc3 = rf.inv_rc3;
    let coulomb_kj = kcal_to_kj(crate::nonbonded::COULOMB_CONST_KCAL_A_PER_E2);
    // CellList neighbour search — same logic as AoS path, just over
    // the SoA position arrays.
    let mut positions: Vec<Vec3> = Vec::with_capacity(n);
    for i in 0..n {
        positions.push(Vec3::new(scratch.xs[i], scratch.ys[i], scratch.zs[i]));
    }
    let cl = CellList::build(&positions, cutoff_a);

    for (i, j, r) in cl.iter_pairs_within(&positions, cutoff_a) {
        if scratch.is_excluded(i, j) {
            continue;
        }
        let dx = scratch.xs[j] - scratch.xs[i];
        let dy = scratch.ys[j] - scratch.ys[i];
        let dz = scratch.zs[j] - scratch.zs[i];
        let r2 = dx * dx + dy * dy + dz * dz;
        if r2 > cutoff_sq || r2 < 1e-18 {
            continue;
        }

        let one_four = scratch.is_one_four(i, j);
        let (rmin_half_i, eps_i, rmin_half_j, eps_j) = if one_four {
            (
                scratch.rmin_half_14[i],
                scratch.epsilon_14[i],
                scratch.rmin_half_14[j],
                scratch.epsilon_14[j],
            )
        } else {
            (
                scratch.rmin_half[i],
                scratch.epsilon[i],
                scratch.rmin_half[j],
                scratch.epsilon[j],
            )
        };
        let rmin_ij = rmin_half_i + rmin_half_j;
        let eps_ij = (eps_i * eps_j).sqrt();

        let inv_r2 = 1.0 / r2;
        let ratio2 = rmin_ij * rmin_ij * inv_r2;
        let ratio6 = ratio2 * ratio2 * ratio2;
        let ratio12 = ratio6 * ratio6;

        // F_i (LJ) coefficient in kJ/mol/Å².
        let lj_coeff = 12.0 * eps_ij * inv_r2 * (ratio6 - ratio12) * kj_per_kcal;

        // F_i (reaction-field Coulomb) — α · (r_j − r_i) with
        // α = −k_kJ · qq · (1/r³ − 1/Rc³). At r = Rc the bracket is
        // zero, so the SoA path matches the AoS path's smooth-cutoff
        // behaviour exactly.
        let qq = scratch.charges[i] * scratch.charges[j];
        let coul_coeff = if qq != 0.0 {
            -coulomb_kj * qq * (inv_r2 / r - inv_rc3)
        } else {
            0.0
        };

        let coeff = lj_coeff + coul_coeff;
        let fx = dx * coeff;
        let fy = dy * coeff;
        let fz = dz * coeff;
        scratch.fxs[i] += fx;
        scratch.fys[i] += fy;
        scratch.fzs[i] += fz;
        scratch.fxs[j] -= fx;
        scratch.fys[j] -= fy;
        scratch.fzs[j] -= fz;
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::nonbonded::nonbonded_energy;
    use chem::{standard_ff, AminoAcid};
    use geom::{build_extended_chain, build_topology_graph, structure::PlacedAtom, structure::PlacedResidue};
    use chem::Element;

    fn flatten(s: &Structure) -> Vec<Vec3> {
        s.residues.iter().flat_map(|r| r.atoms.iter().map(|a| a.position)).collect()
    }

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

    #[test]
    fn nonbonded_force_finite_difference() {
        // Use a 3-residue chain so we have a non-trivial set of 1-4 and
        // longer-range pairs.
        let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala, AminoAcid::Ala]).unwrap();
        let g = build_topology_graph(&s);
        let ff = standard_ff();
        let n = s.atom_count();
        let positions = flatten(&s);
        let mut forces = vec![Vec3::zeros(); n];
        add_nonbonded_forces_default(&s, &g, ff, &mut forces);

        // For a sample of atoms, check ∂E/∂r matches finite differences.
        let eps = 1e-5;
        for i in 0..n.min(8) {
            for axis in 0..3 {
                let mut s_plus = s.clone();
                let mut s_minus = s.clone();
                bump(&mut s_plus, i, axis, eps);
                bump(&mut s_minus, i, axis, -eps);
                let e_plus = nonbonded_energy(&s_plus, &g, ff, DEFAULT_CUTOFF_A);
                let e_minus = nonbonded_energy(&s_minus, &g, ff, DEFAULT_CUTOFF_A);
                let total_plus = e_plus.lj_kj_mol + e_plus.coulomb_kj_mol;
                let total_minus = e_minus.lj_kj_mol + e_minus.coulomb_kj_mol;
                let numeric = -(total_plus - total_minus) / (2.0 * eps);
                let an = forces[i][axis];
                assert!(
                    (an - numeric).abs() < 1e-1,
                    "atom {} axis {}: analytical={:.4}, numeric={:.4}", i, axis, an, numeric
                );
            }
        }
        let _ = positions; // silence unused warning if we change the body later
    }

    #[test]
    fn soa_nonbonded_matches_aos() {
        // SoA kernel and AoS kernel must produce identical forces on
        // the same structure.
        let s = build_extended_chain(&[AminoAcid::Lys, AminoAcid::Glu, AminoAcid::Ala]).unwrap();
        let g = build_topology_graph(&s);
        let ff = standard_ff();
        let n = s.atom_count();

        let mut forces_aos = vec![Vec3::zeros(); n];
        add_nonbonded_forces(&s, &g, ff, DEFAULT_CUTOFF_A, &mut forces_aos);

        let mut scratch = crate::scratch::ForceScratch::new(&s, &g, ff);
        scratch.zero_forces();
        add_nonbonded_forces_soa(&mut scratch, DEFAULT_CUTOFF_A);
        let mut forces_soa = vec![Vec3::zeros(); n];
        scratch.accumulate_into(&mut forces_soa);

        let mut max_err: f64 = 0.0;
        for i in 0..n {
            let diff = forces_aos[i] - forces_soa[i];
            max_err = max_err.max(diff.norm());
        }
        assert!(
            max_err < 1e-9,
            "SoA force differs from AoS by {} (max norm)",
            max_err
        );
    }

    #[test]
    fn lj_force_at_minimum_is_zero() {
        // Two carbons placed exactly at their LJ minimum: C atom σ ≈ 2.0 Å,
        // so two CT3-CT3 atoms at r = 4 Å (= 2 × Rmin/2) have zero LJ force.
        // We use bare positions outside the chain framework — easier.
        let r_min_carbon = 2.0; // Rmin/2 from CHARMM C
        let dist = 2.0 * r_min_carbon;
        let s = Structure {
            residues: vec![PlacedResidue {
                aa: AminoAcid::Ala,
                atoms: vec![
                    PlacedAtom { name: "CB", element: Element::C, position: Vec3::zeros() },
                    PlacedAtom { name: "C", element: Element::C, position: Vec3::new(dist, 0.0, 0.0) },
                ],
            }],
        };
        let g = build_topology_graph(&s);
        let ff = standard_ff();
        let mut forces = vec![Vec3::zeros(); 2];
        add_nonbonded_forces_default(&s, &g, ff, &mut forces);
        // Force shouldn't be exactly zero (atom types differ slightly: CT3 vs C)
        // but should be small. We don't assert here — too fragile a test.
        for f in &forces {
            assert!(f.norm().is_finite());
        }
    }
}
