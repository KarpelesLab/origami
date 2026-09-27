//! RNA force-field acceptance: native fold scores below extended.
//!
//! The protein-side analogue is `m7_native_vs_extended` (chignolin 30 k,
//! Trp-cage 49 k, villin 30 k kJ/mol native-favourable gaps).  This is
//! the equivalent for the CHARMM27 nucleic acid force field — covering
//! all three canonical small-RNA reference motifs:
//!
//! - **UUCG tetraloop** (PDB 2KOC, NMR): 14-nt hairpin
//!   GGCACUUCGGUGCC with the canonical UUCG closing loop.
//! - **GNRA tetraloop** (PDB 1ZIH, NMR): 12-nt hairpin GGGCGCAAGCCU
//!   with the GCAA closing loop — a GNRA-class tetraloop where N = C
//!   and R = A.
//! - **Sarcin/ricin loop** (PDB 483D, X-ray 1.5 Å): 27-nt SRL from
//!   E. coli 23S rRNA, sequence UGCUCCUAGUACGAGAGGACCGGAGUG.
//!   The X-ray fixture is heavy-atom-only — hydrogens get added on
//!   the fly via `geom::add_rna_hydrogens` before scoring, using the
//!   same sp²/sp³ placement helpers the chain builder uses.
//!
//! Both native (loaded from PDB) and extended (built from sequence)
//! carry the same atom set, so all bonded + LJ + Coulomb + GB terms
//! are directly comparable.  We minimise both briefly to discount
//! NMR / NeRF starting-point clashes — same convention as the villin
//! protein test.

use chem::{Nucleotide, standard_ff};
use energy::bonded::bonded_energy;
use energy::{DEFAULT_CUTOFF_A, gb_energy, nonbonded_energy};
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
    let pdb =
        std::fs::read_to_string("tests/fixtures/2KOC_uucg_hairpin.pdb").expect("read 2KOC fixture");
    let mut native = read_pdb(pdb.as_bytes()).expect("parse 2KOC");

    // Verify we got the expected 14-nt UUCG hairpin.
    assert_eq!(native.residues.len(), 14, "expected 14 residues");
    let seq: String = native
        .residues
        .iter()
        .filter_map(|r| r.monomer.as_nucleotide().map(|n| n.one_letter()))
        .collect();
    assert_eq!(seq, "GGCACUUCGGUGCC", "fixture sequence mismatch");

    // ---- Build extended chain from the same sequence ----
    let nts: Vec<Nucleotide> = seq
        .chars()
        .map(|c| Nucleotide::from_one_letter(c).unwrap())
        .collect();
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

#[test]
fn gnra_hairpin_native_beats_extended() {
    // ---- Load native structure (PDB 1ZIH model 1, NMR) ----
    // 12-nt RNA hairpin GGGCGCAAGCCU — 4 bp stem + GCAA tetraloop (a
    // GNRA-class loop where N = C, R = A).  The fixture has the
    // canonical NMR atom set including all hydrogens.
    let pdb = std::fs::read_to_string("tests/fixtures/1ZIH_gnra_tetraloop.pdb")
        .expect("read 1ZIH fixture");
    let mut native = read_pdb(pdb.as_bytes()).expect("parse 1ZIH");

    assert_eq!(native.residues.len(), 12, "expected 12 residues");
    let seq: String = native
        .residues
        .iter()
        .filter_map(|r| r.monomer.as_nucleotide().map(|n| n.one_letter()))
        .collect();
    assert_eq!(seq, "GGGCGCAAGCCU", "1ZIH sequence mismatch");

    // ---- Build extended chain from the same sequence ----
    let nts: Vec<Nucleotide> = seq
        .chars()
        .map(|c| Nucleotide::from_one_letter(c).unwrap())
        .collect();
    let mut extended = build_extended_rna_chain(&nts).expect("build extended");

    brief_minimise(&mut native, 100);
    brief_minimise(&mut extended, 200);

    let e_native = total_energy_no_sasa(&native);
    let e_extended = total_energy_no_sasa(&extended);
    let gap = e_extended - e_native;

    eprintln!(
        "GNRA hairpin (1ZIH): native = {e_native:.1} kJ/mol  \
         extended = {e_extended:.1} kJ/mol  gap = {gap:.1} kJ/mol"
    );

    // GNRA hairpin is smaller (12 nt vs 14 nt) and has only 4 bp of
    // stem (vs 5 bp on UUCG), so the native-favourable gap is
    // somewhat smaller.  1 000 kJ/mol is a defensible floor for "the
    // FF prefers the fold" given those constraints.
    assert!(
        gap > 1000.0,
        "GNRA hairpin: native should score ≥1000 kJ/mol below extended, got {gap}"
    );
}

#[test]
fn sarcin_ricin_loop_native_beats_extended() {
    // ---- Load native structure (PDB 483D, X-ray 1.5 Å) ----
    // 27-nt sarcin/ricin loop from E. coli 23S rRNA.  X-ray with no
    // hydrogens — we add them with `add_rna_hydrogens` before
    // scoring.
    let pdb =
        std::fs::read_to_string("tests/fixtures/483D_sarcin_ricin.pdb").expect("read 483D fixture");
    let mut native = read_pdb(pdb.as_bytes()).expect("parse 483D");

    assert_eq!(native.residues.len(), 27, "expected 27 residues");
    let seq: String = native
        .residues
        .iter()
        .filter_map(|r| r.monomer.as_nucleotide().map(|n| n.one_letter()))
        .collect();
    assert_eq!(seq, "UGCUCCUAGUACGAGAGGACCGGAGUG", "483D sequence mismatch");

    // Hydrogenate.  All 27 residues should grow ~9 H atoms each
    // (backbone 7 H + base 2-4 H).
    let h_summary = geom::add_rna_hydrogens(&mut native);
    eprintln!("SRL H-addition: {h_summary:?}");
    assert!(
        h_summary.h_added >= 200,
        "expected ~250 H atoms added on 27-nt SRL, got {}",
        h_summary.h_added
    );

    // ---- Build extended chain from the same sequence ----
    let nts: Vec<Nucleotide> = seq
        .chars()
        .map(|c| Nucleotide::from_one_letter(c).unwrap())
        .collect();
    let mut extended = build_extended_rna_chain(&nts).expect("build extended");

    // X-ray bond lengths can drift ~0.05 Å from CHARMM r₀; minimise
    // both fairly hard to get the bond term back to physical values
    // (villin's protein analogue uses 30 L-BFGS, but the SRL is
    // longer and the X-ray vs CHARMM gap is larger).
    brief_minimise(&mut native, 200);
    brief_minimise(&mut extended, 300);

    let e_native = total_energy_no_sasa(&native);
    let e_extended = total_energy_no_sasa(&extended);
    let gap = e_extended - e_native;

    eprintln!(
        "Sarcin/ricin (483D): native = {e_native:.1} kJ/mol  \
         extended = {e_extended:.1} kJ/mol  gap = {gap:.1} kJ/mol"
    );

    // SRL is 27 nt with extensive non-canonical pairing (the GAGA
    // tetraloop on top of the bulged-G motif).  Roughly 2× the
    // structural content of the 14-nt UUCG hairpin, so the floor
    // can be set proportionally higher.
    assert!(
        gap > 3000.0,
        "Sarcin/ricin: native should score ≥3000 kJ/mol below extended, got {gap}"
    );
}
