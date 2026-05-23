//! RNA force-field acceptance: native UUCG hairpin scores below extended.
//!
//! The protein-side analogue is `m7_native_vs_extended` (chignolin 30 k,
//! Trp-cage 49 k, villin 30 k kJ/mol native-favourable gaps).  This test
//! is the equivalent for the CHARMM27 nucleic acid force field shipped
//! in `FEAT.rna-ff`.
//!
//! Fixture: 2KOC (Nozinovic et al. 2010, NMR solution structure of a
//! 14-mer hairpin RNA with the UUCG tetraloop, GGCACUUCGGUGCC, 5 bp
//! stem + UUCG loop).  Single-chain pure-RNA NMR ensemble, model 1
//! extracted.
//!
//! Both the native (loaded from PDB) and the extended chain (built
//! from sequence) carry the same atom set, so all bonded + LJ +
//! Coulomb + GB terms are directly comparable.  We minimise both
//! briefly to discount NMR / NeRF starting-point clashes — same
//! convention as the villin protein test.

use chem::{standard_ff, Nucleotide};
use energy::bonded::bonded_energy;
use energy::{gb_energy, nonbonded_energy, DEFAULT_CUTOFF_A};
use geom::{build_extended_rna_chain, build_topology_graph};
use io::read_pdb;

fn total_energy_no_sasa(s: &geom::Structure) -> f64 {
    let g = build_topology_graph(s);
    let ff = standard_ff();
    let bonded = bonded_energy(s, &g, ff);
    let nb = nonbonded_energy(s, &g, ff, DEFAULT_CUTOFF_A);
    let gb = gb_energy(s, ff);
    bonded.total_kj_mol() + nb.lj_kj_mol + nb.coulomb_kj_mol + gb.gb_kj_mol
}

fn brief_minimise(s: &mut geom::Structure, steps: usize) {
    let g = build_topology_graph(s);
    let ff = standard_ff();
    let _ = dynamics::minimize(
        s,
        &g,
        ff,
        dynamics::MinimizeOptions {
            algorithm: dynamics::Algorithm::Lbfgs,
            max_steps: steps,
            gradient_tol: 50.0,
            energy_tol: 1.0,
            max_step_a: 0.1,
            include_sasa: false,
            include_cmap: false,
        },
    );
}

#[test]
fn uucg_hairpin_native_beats_extended() {
    // ---- Load native structure ----
    let pdb = std::fs::read_to_string("tests/fixtures/2KOC_uucg_hairpin.pdb")
        .expect("read 2KOC fixture");
    let mut native = read_pdb(pdb.as_bytes()).expect("parse 2KOC");

    // Verify we got the expected 14-nt UUCG hairpin.
    assert_eq!(native.residues.len(), 14, "expected 14 residues");
    let seq: String = native
        .residues
        .iter()
        .filter_map(|r| r.monomer.as_nucleotide().map(|n| n.one_letter()))
        .collect();
    assert_eq!(
        seq, "GGCACUUCGGUGCC",
        "fixture sequence mismatch"
    );

    // ---- Build extended chain from the same sequence ----
    let nts: Vec<Nucleotide> = seq.chars().map(|c| Nucleotide::from_one_letter(c).unwrap()).collect();
    let mut extended = build_extended_rna_chain(&nts).expect("build extended");
    assert_eq!(extended.residues.len(), 14);

    // ---- Both get a brief minimise to relieve starting-point clashes ----
    // The NMR fixture has standard NMR-pipeline residual H-H contacts
    // (~10⁶ kJ/mol of LJ before relaxation); the extended chain has
    // NeRF placement clashes (~10⁹ kJ/mol).  Both relieve in well under
    // 200 L-BFGS steps.
    brief_minimise(&mut native, 100);
    brief_minimise(&mut extended, 200);

    let e_native = total_energy_no_sasa(&native);
    let e_extended = total_energy_no_sasa(&extended);
    let gap = e_extended - e_native;

    eprintln!(
        "UUCG hairpin (2KOC): native = {e_native:.1} kJ/mol  \
         extended = {e_extended:.1} kJ/mol  gap = {gap:.1} kJ/mol"
    );

    // Acceptance threshold: native should beat extended by at least
    // 1 500 kJ/mol.  The actual gap on this baseline is ~2 900
    // kJ/mol, which is the right order of magnitude for a 5-bp
    // hairpin (each base-pair H-bond ≈ 20 kJ/mol, each stack ≈ 10
    // kJ/mol, plus GB solvation differences).
    //
    // An earlier version of this test asserted a 10 000 kJ/mol gap.
    // That value was artefact-inflated: the chem nucleotide topology
    // table was missing all base C-H / N-H bonds, so those pairs
    // were counted as non-bonded "clashes" with huge positive LJ
    // contributions.  The extended chain (with NeRF-perfect 1.080 Å
    // H bond lengths) suffered worse than the NMR native, which made
    // the gap look ~10× bigger than it really is.  The FIX.rna-bond-r0
    // → FIX.rna-builder-geometry commits added the missing H bonds
    // and the gap settled at the honest ~3 000 kJ/mol value.
    assert!(
        gap > 1500.0,
        "UUCG hairpin: native should score ≥1500 kJ/mol below extended, got {gap}"
    );
}
