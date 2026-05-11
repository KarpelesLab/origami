//! Helpers that combine `energy::*` modules into the energy/force pair the
//! optimiser needs. SASA is intentionally excluded — its dot-density
//! gradient is non-smooth.

use chem::ForceField;
use energy::bonded::bonded_energy;
use energy::gb::gb_energy;
use energy::nonbonded::nonbonded_energy;
use energy::powersasa::powersasa_energy;
use energy::{total_force_with_options, DEFAULT_CUTOFF_A};
use geom::{Structure, TopologyGraph, Vec3};

/// Total potential energy used in M4 minimisation (no SASA). For
/// compatibility with existing tests that lock in numerical results.
pub fn total_energy(structure: &Structure, graph: &TopologyGraph, ff: &ForceField) -> f64 {
    total_energy_with_options(structure, graph, ff, false)
}

/// Total potential energy with optional SASA term (PSA.2).
pub fn total_energy_with_options(
    structure: &Structure,
    graph: &TopologyGraph,
    ff: &ForceField,
    include_sasa: bool,
) -> f64 {
    let bonded = bonded_energy(structure, graph, ff);
    let nb = nonbonded_energy(structure, graph, ff, DEFAULT_CUTOFF_A);
    let gb = gb_energy(structure, ff);
    let mut total = bonded.total_kj_mol() + nb.lj_kj_mol + nb.coulomb_kj_mol + gb.gb_kj_mol;
    if include_sasa {
        total += powersasa_energy(structure, ff).sasa_kj_mol;
    }
    total
}

/// Total atomic forces used in M4 minimisation (no SASA, preserves
/// historical numerical baselines).
pub fn total_force(structure: &Structure, graph: &TopologyGraph, ff: &ForceField) -> Vec<Vec3> {
    total_force_with_options(structure, graph, ff, DEFAULT_CUTOFF_A, false)
}

/// Total atomic forces with optional SASA term (PSA.2).
pub fn total_force_opts(
    structure: &Structure,
    graph: &TopologyGraph,
    ff: &ForceField,
    include_sasa: bool,
) -> Vec<Vec3> {
    total_force_with_options(structure, graph, ff, DEFAULT_CUTOFF_A, include_sasa)
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
