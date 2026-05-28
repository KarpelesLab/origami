//! Native-stability MD on RNA fixtures — the RNA analogue of
//! `m7_native_stability` for Trp-cage.
//!
//! For each native fold, run a short Langevin trajectory at 310 K
//! and verify that the backbone P-RMSD stays bounded.  If the force
//! field is sane, 2 ps of dynamics won't unfold the structure.  This
//! is the flip side of the energy-ranking test in
//! `crates/io/tests/rna_native_vs_extended.rs`: ranking confirms the
//! *values*, stability confirms the *gradients* point the same way.
//!
//! References:
//!   * UUCG tetraloop — PDB 2KOC NMR model 1 (Nozinovic et al. 2010),
//!     14 nt hairpin GGCACUUCGGUGCC with a CUUCGG closing loop.
//!   * GNRA tetraloop — PDB 1ZIH NMR model 1 (Jucker et al. 1996),
//!     12 nt hairpin GGGCGCAAGCCU with a GCAA closing loop.
//!   * Sarcin/ricin loop — PDB 483D X-ray 1.5 Å (Correll et al. 1999),
//!     27 nt SRL from E. coli 23S rRNA.  Heavy-atom-only crystal
//!     structure; hydrogens added via `geom::add_rna_hydrogens`
//!     before MD.

use chem::standard_ff;
use dynamics::{minimize, run_langevin, Algorithm, LangevinOptions, MinimizeOptions};
use geom::{build_topology_graph, rmsd_p};
use io::read_pdb;

fn read_fixture(path: &str) -> geom::Structure {
    let pdb = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    read_pdb(pdb.as_bytes()).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

fn brief_minimise(s: &mut geom::Structure, steps: usize) {
    let g = build_topology_graph(s);
    let ff = standard_ff();
    let _ = minimize(s, &g, ff, MinimizeOptions {
        algorithm: Algorithm::Lbfgs,
        max_steps: steps,
        gradient_tol: 50.0,
        energy_tol: 1.0,
        max_step_a: 0.1,
        include_sasa: false,
        include_cmap: false,
    });
}

fn run_short_md(s: &mut geom::Structure, seed: u64, steps: usize) -> dynamics::LangevinSummary {
    let g = build_topology_graph(s);
    let ff = standard_ff();
    let opts = LangevinOptions {
        dt_fs: 1.0,
        temperature_k: 310.0,
        friction_ps_inv: 2.0,
        steps,
        save_every: 0,
        seed,
        randomise_initial_velocities: true,
        include_sasa: false,
        include_cmap: false,
        constrain_h_bonds: false,
        use_gpu: false,
        use_gpu_integrator: false,
    };
    run_langevin(s, &g, ff, opts, |_| {})
}

#[test]
fn uucg_hairpin_stays_near_native_during_2ps_md() {
    let mut s = read_fixture("../io/tests/fixtures/2KOC_uucg_hairpin.pdb");
    // The raw NMR fixture has residual H-H contacts that explode LJ
    // on the first BAOAB step (the integrator can't survive 10⁶
    // kJ/mol forces in one fs).  100 L-BFGS steps drops the bond +
    // LJ terms to physical values without moving the fold; same
    // pre-conditioning the M3/M7 protein tests use.
    brief_minimise(&mut s, 100);
    let initial = s.clone();
    let summary = run_short_md(&mut s, 7, 2000);
    assert!(!summary.diverged, "UUCG trajectory diverged");
    let rmsd = rmsd_p(&initial, &s).expect("rmsd_p UUCG");
    eprintln!("UUCG (2KOC) native MD 2 ps: P-RMSD = {rmsd:.3} Å");
    // The protein Trp-cage analogue uses 3.5 Å as the ceiling.  RNA
    // backbones are stiffer per nucleotide but the hairpin loop has
    // more conformational freedom.  3.5 Å keeps the same physical
    // meaning ("the fold didn't fall apart").
    assert!(rmsd < 3.5,
        "UUCG hairpin P-RMSD {rmsd} > 3.5 Å — force field may not retain the fold");
}

#[test]
fn gnra_hairpin_stays_near_native_during_2ps_md() {
    let mut s = read_fixture("../io/tests/fixtures/1ZIH_gnra_tetraloop.pdb");
    brief_minimise(&mut s, 100);
    let initial = s.clone();
    let summary = run_short_md(&mut s, 11, 2000);
    assert!(!summary.diverged, "GNRA trajectory diverged");
    let rmsd = rmsd_p(&initial, &s).expect("rmsd_p GNRA");
    eprintln!("GNRA (1ZIH) native MD 2 ps: P-RMSD = {rmsd:.3} Å");
    assert!(rmsd < 3.5,
        "GNRA hairpin P-RMSD {rmsd} > 3.5 Å — force field may not retain the fold");
}

#[test]
fn sarcin_ricin_stays_near_native_during_2ps_md() {
    // X-ray structure with no hydrogens — `add_rna_hydrogens` fills
    // them in.
    let mut s = read_fixture("../io/tests/fixtures/483D_sarcin_ricin.pdb");
    let _ = geom::add_rna_hydrogens(&mut s);
    // SRL X-ray bond lengths drift further from CHARMM r₀ than NMR
    // ensembles do (typical 0.03-0.05 Å vs ~0.01 Å), so the brief
    // minimisation needs more steps to land the bond + LJ terms at
    // physical values before BAOAB can take 1 fs steps without
    // exploding.
    brief_minimise(&mut s, 300);
    let initial = s.clone();
    let summary = run_short_md(&mut s, 13, 2000);
    assert!(!summary.diverged, "SRL trajectory diverged");
    let rmsd = rmsd_p(&initial, &s).expect("rmsd_p SRL");
    eprintln!("SRL (483D) native MD 2 ps: P-RMSD = {rmsd:.3} Å");
    // SRL is 27 nt with extensive tertiary contacts and bulged-G
    // motif — slightly more conformational freedom than a hairpin.
    // 4 Å is the same physical "fold intact" bar as the tetraloops
    // get with their 3.5 Å bound, just scaled.
    assert!(rmsd < 4.0,
        "SRL P-RMSD {rmsd} > 4 Å — force field may not retain the fold");
}
