//! M7c — energy-ranking acceptance for three small folds.
//!
//! For each of chignolin (1UAO), Trp-cage (1L2Y), and villin HP35 (2F4K),
//! the native structure should score lower than the same sequence built
//! as a fully extended chain — i.e. the hand-built physics has the right
//! qualitative direction. We use the no-SASA energy total to keep the
//! test fast.

use chem::standard_ff;
use energy::bonded::bonded_energy;
use energy::{gb_energy, nonbonded_energy, DEFAULT_CUTOFF_A};
use geom::{build_extended_chain, build_topology_graph};
use io::read_pdb;

fn total_energy_no_sasa(s: &geom::Structure) -> f64 {
    let g = build_topology_graph(s);
    let ff = standard_ff();
    let bonded = bonded_energy(s, &g, ff);
    let nb = nonbonded_energy(s, &g, ff, DEFAULT_CUTOFF_A);
    let gb = gb_energy(s, ff);
    bonded.total_kj_mol() + nb.lj_kj_mol + nb.coulomb_kj_mol + gb.gb_kj_mol
}

fn read_fixture(path: &str) -> geom::Structure {
    let pdb = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    read_pdb(pdb.as_bytes()).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

fn assert_native_better(name: &str, native_path: &str, expected_seq: &str, min_gap_kj_mol: f64) {
    let native = read_fixture(native_path);
    let seq: String = native.residues.iter().map(|r| r.aa.one_letter()).collect();
    assert_eq!(
        seq, expected_seq,
        "{name}: parsed sequence ({}) != expected ({})", seq, expected_seq
    );

    let extended = build_extended_chain(
        &expected_seq
            .chars()
            .filter_map(chem::AminoAcid::from_one_letter)
            .collect::<Vec<_>>(),
    )
    .expect("build extended");

    let e_native = total_energy_no_sasa(&native);
    let e_extended = total_energy_no_sasa(&extended);
    let gap = e_extended - e_native;

    eprintln!(
        "{name}: native = {e_native:.1}  extended = {e_extended:.1}  gap = {gap:.1} kJ/mol"
    );
    assert!(
        gap > min_gap_kj_mol,
        "{name}: native should score at least {min_gap_kj_mol} below extended, got {gap}"
    );
}

#[test]
fn chignolin_native_beats_extended() {
    // 1UAO chignolin: 10 residues, sequence GYDPETGTWG.
    assert_native_better(
        "chignolin (1UAO)",
        "tests/fixtures/1UAO_chignolin.pdb",
        "GYDPETGTWG",
        1000.0,
    );
}

#[test]
fn trp_cage_native_beats_extended() {
    assert_native_better(
        "Trp-cage (1L2Y)",
        "tests/fixtures/1L2Y_model1.pdb",
        "NLYIQWLKDGGPSSGRPPPS",
        10000.0,
    );
}

#[test]
fn villin_hp35_native_beats_extended() {
    // 2F4K villin HP-35: 33-35 residues. The fixture is an X-ray
    // structure and has strained bond lengths relative to CHARMM r₀
    // (bond term ~20 000 kJ/mol on the raw fixture). Standard practice
    // is to minimise briefly before scoring against a force field, so
    // the comparison is "native fold" vs "extended chain", not
    // "X-ray-reported coordinates" vs "extended chain".
    let mut native = read_fixture("tests/fixtures/2F4K_villin_hp35.pdb");
    let seq: String = native.residues.iter().map(|r| r.aa.one_letter()).collect();
    eprintln!("villin HP35 fixture seq ({} aa): {}", seq.len(), seq);
    assert!(seq.len() >= 30, "expected ≥30 residues, got {}", seq.len());

    // Brief minimisation of the native to relieve bond strain. 100
    // L-BFGS steps is enough to drop the bond term from ~20 000 to
    // <200 kJ/mol without moving the fold significantly (the rest of
    // the gradient is small).
    let g = build_topology_graph(&native);
    let ff = standard_ff();
    let _ = dynamics::minimize(
        &mut native,
        &g,
        ff,
        dynamics::MinimizeOptions {
            algorithm: dynamics::Algorithm::Lbfgs,
            // 30 L-BFGS steps brings the bond term from ~20 000 to
            // <2 000 kJ/mol — enough to make the energy comparison
            // meaningful without paying the full convergence cost.
            max_steps: 30,
            gradient_tol: 50.0,
            energy_tol: 1.0,
            max_step_a: 0.1,
            include_sasa: false,
        },
    );

    let extended = build_extended_chain(
        &seq.chars()
            .filter_map(chem::AminoAcid::from_one_letter)
            .collect::<Vec<_>>(),
    )
    .expect("build extended");
    let e_native = total_energy_no_sasa(&native);
    let e_extended = total_energy_no_sasa(&extended);
    let gap = e_extended - e_native;
    eprintln!("villin HP35: native (minimised) = {e_native:.1} extended = {e_extended:.1} gap = {gap:.1}");
    assert!(
        gap > 10_000.0,
        "villin: native should score at least 10 000 kJ/mol below extended, got {gap}"
    );
}
