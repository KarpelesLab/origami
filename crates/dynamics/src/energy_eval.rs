//! Helpers that combine `energy::*` modules into the energy/force pair the
//! optimiser needs. SASA is intentionally excluded — its dot-density
//! gradient is non-smooth.

use chem::ForceField;
use energy::bonded::bonded_energy;
use energy::gb::gb_energy;
use energy::nonbonded::nonbonded_energy;
use energy::{total_force_with_cutoff, DEFAULT_CUTOFF_A};
use geom::{Structure, TopologyGraph, Vec3};

/// Total potential energy used in M4 minimisation (no SASA).
pub fn total_energy(structure: &Structure, graph: &TopologyGraph, ff: &ForceField) -> f64 {
    let bonded = bonded_energy(structure, graph, ff);
    let nb = nonbonded_energy(structure, graph, ff, DEFAULT_CUTOFF_A);
    let gb = gb_energy(structure, ff);
    bonded.total_kj_mol() + nb.lj_kj_mol + nb.coulomb_kj_mol + gb.gb_kj_mol
}

/// Total atomic forces used in M4 minimisation (no SASA).
pub fn total_force(structure: &Structure, graph: &TopologyGraph, ff: &ForceField) -> Vec<Vec3> {
    total_force_with_cutoff(structure, graph, ff, DEFAULT_CUTOFF_A)
}

/// Apply a flat displacement vector (3N entries) to a structure.
/// `dx[i*3 + axis]` is added to atom `i`'s coordinate `axis`.
pub fn apply_displacement(structure: &mut Structure, dx: &[f64]) {
    let mut idx = 0usize;
    for residue in &mut structure.residues {
        for atom in &mut residue.atoms {
            atom.position[0] += dx[idx * 3];
            atom.position[1] += dx[idx * 3 + 1];
            atom.position[2] += dx[idx * 3 + 2];
            idx += 1;
        }
    }
}

/// Flatten a `Vec<Vec3>` into a 3N flat slice (in-place into `out`).
pub fn flatten_vec3(forces: &[Vec3], out: &mut [f64]) {
    debug_assert_eq!(out.len(), forces.len() * 3);
    for (i, f) in forces.iter().enumerate() {
        out[i * 3] = f.x;
        out[i * 3 + 1] = f.y;
        out[i * 3 + 2] = f.z;
    }
}

/// Maximum component magnitude (L∞ norm) of a flat vector.
pub fn linf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()))
}
