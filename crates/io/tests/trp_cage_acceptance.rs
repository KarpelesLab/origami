//! M3 acceptance test: the native Trp-cage structure (1L2Y MODEL 1) has
//! a lower total potential energy than the same sequence built as a
//! fully-extended chain. This validates that our hand-assembled force
//! field has the right qualitative behaviour: folding is favourable.

use chem::{standard_ff, AminoAcid};
use energy::{
    bonded::bonded_energy, gb_energy, nonbonded_energy, sasa_energy, DEFAULT_CUTOFF_A,
};
use geom::{build_extended_chain, build_topology_graph};
use io::read_pdb;

fn total_energy_kj_mol(structure: &geom::Structure) -> f64 {
    let g = build_topology_graph(structure);
    let ff = standard_ff();
    let bonded = bonded_energy(structure, &g, ff);
    let nb = nonbonded_energy(structure, &g, ff, DEFAULT_CUTOFF_A);
    let gb = gb_energy(structure, ff);
    let sasa = sasa_energy(structure, ff);
    bonded.total_kj_mol() + nb.lj_kj_mol + nb.coulomb_kj_mol + gb.gb_kj_mol + sasa.sasa_kj_mol
}

#[test]
fn native_trp_cage_lower_energy_than_extended() {
    let pdb = include_str!("fixtures/1L2Y_model1.pdb");
    let native = read_pdb(pdb.as_bytes()).expect("parse 1L2Y");
    assert_eq!(native.residues.len(), 20);
    let native_seq: String = native.residues.iter().map(|r| r.aa.one_letter()).collect();
    assert_eq!(native_seq, "NLYIQWLKDGGPSSGRPPPS");

    let extended = build_extended_chain(
        &native_seq.chars().filter_map(AminoAcid::from_one_letter).collect::<Vec<_>>(),
    )
    .expect("build extended");

    let e_native = total_energy_kj_mol(&native);
    let e_extended = total_energy_kj_mol(&extended);

    let delta = e_extended - e_native;
    assert!(
        e_native < e_extended,
        "native energy {} kJ/mol should be < extended energy {} kJ/mol",
        e_native, e_extended,
    );
    // Sanity: the gap should be substantial (not just a fluke of one term).
    assert!(
        delta > 1000.0,
        "expected a large favourability gap, got Δ = {} kJ/mol",
        delta,
    );
}
