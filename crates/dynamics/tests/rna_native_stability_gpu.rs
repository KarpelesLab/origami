//! GPU mirror of `rna_native_stability.rs` — runs the same UUCG /
//! GNRA / SRL trajectories on the [`FullGpuIntegrator`] instead of
//! the CPU Langevin path.
//!
//! End-to-end check that the GPU stack (bonded + LJ/Coulomb + GB +
//! BAOAB + Verlet neighbour lists + Morton sort) produces a
//! physically reasonable trajectory on real RNA fixtures, not just
//! the synthetic chains the existing GPU tests use.  Also confirms
//! `add_rna_hydrogens` produces structures the GPU integrator
//! happily ingests (it goes through the same `FullGpuIntegrator::new`
//! path that classifies atoms by their `Monomer`).
//!
//! Same 2 ps timescale as the CPU tests so the bar is directly
//! comparable — RMSD numbers should land in the same ballpark
//! (within f32-vs-f64 noise on cancellation-heavy electrostatic
//! atoms; see `FEAT.gpu.26` commit message).

use chem::standard_ff;
use dynamics::full_gpu_integrator::FullGpuIntegrator;
use dynamics::{minimize, Algorithm, MinimizeOptions};
use geom::{build_topology_graph, rmsd_p, Vec3};
use gpu::GpuContext;
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

fn run_gpu_md(s: &mut geom::Structure, n_steps: usize, seed: u64) -> bool {
    let g = build_topology_graph(s);
    let ff = standard_ff();
    let n = s.atom_count();
    let mut integ = match FullGpuIntegrator::new(
        s, &g, ff,
        /*dt_fs=*/ 1.0,
        /*gamma_ps_inv=*/ 2.0,
        /*temperature_k=*/ 310.0,
        seed,
    ) {
        Ok(x) => x,
        Err(e) => panic!("FullGpuIntegrator::new failed: {e}"),
    };
    let velocities = vec![Vec3::zeros(); n];
    integ.upload_initial_state(s, &velocities);
    integ.step_batch(n_steps);
    integ.download_positions_into(s);
    for r in &s.residues {
        for a in &r.atoms {
            if !(a.position.x.is_finite() && a.position.y.is_finite() && a.position.z.is_finite()) {
                return false;
            }
        }
    }
    true
}

#[test]
fn uucg_hairpin_stays_near_native_during_2ps_gpu_md() {
    if GpuContext::get().is_err() { eprintln!("GPU unavailable, skipping"); return; }
    let mut s = read_fixture("../io/tests/fixtures/2KOC_uucg_hairpin.pdb");
    brief_minimise(&mut s, 100);
    let initial = s.clone();
    assert!(run_gpu_md(&mut s, 2000, 7), "UUCG GPU trajectory diverged");
    let rmsd = rmsd_p(&initial, &s).expect("rmsd_p UUCG GPU");
    eprintln!("UUCG (2KOC) GPU MD 2 ps: P-RMSD = {rmsd:.3} Å");
    assert!(rmsd < 3.5,
        "UUCG GPU hairpin P-RMSD {rmsd} > 3.5 Å — GPU stack may not retain the fold");
}

#[test]
fn gnra_hairpin_stays_near_native_during_2ps_gpu_md() {
    if GpuContext::get().is_err() { return; }
    let mut s = read_fixture("../io/tests/fixtures/1ZIH_gnra_tetraloop.pdb");
    brief_minimise(&mut s, 100);
    let initial = s.clone();
    assert!(run_gpu_md(&mut s, 2000, 11), "GNRA GPU trajectory diverged");
    let rmsd = rmsd_p(&initial, &s).expect("rmsd_p GNRA GPU");
    eprintln!("GNRA (1ZIH) GPU MD 2 ps: P-RMSD = {rmsd:.3} Å");
    assert!(rmsd < 3.5,
        "GNRA GPU hairpin P-RMSD {rmsd} > 3.5 Å");
}

#[test]
fn sarcin_ricin_stays_near_native_during_2ps_gpu_md() {
    if GpuContext::get().is_err() { return; }
    let mut s = read_fixture("../io/tests/fixtures/483D_sarcin_ricin.pdb");
    let _ = geom::add_rna_hydrogens(&mut s);
    brief_minimise(&mut s, 300);
    let initial = s.clone();
    assert!(run_gpu_md(&mut s, 2000, 13), "SRL GPU trajectory diverged");
    let rmsd = rmsd_p(&initial, &s).expect("rmsd_p SRL GPU");
    eprintln!("SRL (483D) GPU MD 2 ps: P-RMSD = {rmsd:.3} Å");
    assert!(rmsd < 4.0,
        "SRL GPU P-RMSD {rmsd} > 4 Å");
}
