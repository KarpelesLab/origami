//! Per-term timing of the force aggregator. Loads a PDB and calls each
//! force-computing routine independently a fixed number of times,
//! reporting average wall-time per call. Identifies the dominant cost
//! before deciding what to vectorise / parallelise / GPU-offload.

use std::time::Instant;

use chem::standard_ff;
use energy::forces_bonded::{
    add_angle_forces, add_bond_forces, add_dihedral_forces, add_improper_forces, build_atom_types,
};
use energy::forces_gb::add_gb_forces;
use energy::forces_nonbonded::add_nonbonded_forces;
use energy::forces_sasa::add_sasa_forces;
use energy::nonbonded::DEFAULT_CUTOFF_A;
use geom::{build_topology_graph, Vec3};
use io::read_pdb;

fn timeit<F: FnMut()>(label: &str, n_iter: usize, mut f: F) {
    // Warm up.
    f();
    let t0 = Instant::now();
    for _ in 0..n_iter {
        f();
    }
    let elapsed = t0.elapsed();
    let per = elapsed / n_iter as u32;
    println!(
        "  {:32}  {:>8.3} ms/call  ({:>4} iter, total {:>6.2} s)",
        label,
        per.as_secs_f64() * 1000.0,
        n_iter,
        elapsed.as_secs_f64(),
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "crates/io/tests/fixtures/1L2Y_model1.pdb".to_string());
    let n_iter: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let pdb = std::fs::read_to_string(&path).expect("read pdb");
    let s = read_pdb(pdb.as_bytes()).expect("parse pdb");
    let graph = build_topology_graph(&s);
    let ff = standard_ff();
    let n = s.atom_count();
    let atom_types = build_atom_types(&s);
    let positions: Vec<Vec3> = s
        .residues
        .iter()
        .flat_map(|r| r.atoms.iter().map(|a| a.position))
        .collect();

    println!(
        "Benchmarking force terms: {} residues / {} atoms, {} iterations each",
        s.residues.len(),
        n,
        n_iter
    );
    println!();

    let mut forces = vec![Vec3::zeros(); n];

    timeit("bond force", n_iter, || {
        forces.iter_mut().for_each(|f| *f = Vec3::zeros());
        add_bond_forces(&positions, &graph, ff, &atom_types, &mut forces);
    });
    timeit("angle force", n_iter, || {
        forces.iter_mut().for_each(|f| *f = Vec3::zeros());
        add_angle_forces(&positions, &graph, ff, &atom_types, &mut forces);
    });
    timeit("dihedral force", n_iter, || {
        forces.iter_mut().for_each(|f| *f = Vec3::zeros());
        add_dihedral_forces(&positions, &graph, ff, &atom_types, &mut forces);
    });
    timeit("improper force", n_iter, || {
        forces.iter_mut().for_each(|f| *f = Vec3::zeros());
        add_improper_forces(&positions, &graph, ff, &atom_types, &mut forces);
    });
    timeit("nonbonded (LJ+Coulomb) force", n_iter, || {
        forces.iter_mut().for_each(|f| *f = Vec3::zeros());
        add_nonbonded_forces(&s, &graph, ff, DEFAULT_CUTOFF_A, &mut forces);
    });
    timeit("GB force", n_iter, || {
        forces.iter_mut().for_each(|f| *f = Vec3::zeros());
        add_gb_forces(&s, ff, &mut forces);
    });
    timeit("SASA force (analytical)", (n_iter / 20).max(1), || {
        forces.iter_mut().for_each(|f| *f = Vec3::zeros());
        add_sasa_forces(&s, ff, &mut forces);
    });

    // Combined.
    println!();
    timeit("total_force (no SASA)", n_iter, || {
        forces = energy::total_force_with_options(&s, &graph, ff, DEFAULT_CUTOFF_A, false);
    });
}
