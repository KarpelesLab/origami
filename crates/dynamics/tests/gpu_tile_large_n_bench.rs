//! Find the crossover where the tile kernel beats the Verlet
//! kernel.  At our standard bench sizes (≤5840 atoms) Verlet wins
//! by 2-3× because the kernel is compute-bound and tile does more
//! distance tests.  At larger N memory bandwidth should dominate
//! and tile's shared-memory cooperative-load advantage should
//! take over.
//!
//! This test builds extended chains up to ~16k atoms (using the
//! synthetic AGLEK block — atom counts ≈ residues × 14.6) and
//! reports per-step times for both kernels.
//!
//! Marked `#[ignore]` — slow even in release.  Run with:
//!
//! ```text
//! cargo test --release -p dynamics --test gpu_tile_large_n_bench -- --ignored --nocapture
//! ```

use std::time::Instant;

use chem::{standard_ff, AminoAcid};
use dynamics::{full_gpu_integrator::FullGpuIntegrator, minimize, Algorithm, MinimizeOptions};
use geom::{build_extended_chain, build_topology_graph, Vec3};
use gpu::GpuContext;

fn build_chain(n_residues: usize) -> geom::Structure {
    let block = [AminoAcid::Ala, AminoAcid::Gly, AminoAcid::Leu, AminoAcid::Glu, AminoAcid::Lys];
    let seq: Vec<AminoAcid> = block.iter().cloned().cycle().take(n_residues).collect();
    build_extended_chain(&seq).expect("build")
}

fn relax_min(s: &mut geom::Structure, g: &geom::TopologyGraph, ff: &chem::ForceField) {
    let opts = MinimizeOptions {
        algorithm: Algorithm::Lbfgs,
        max_steps: 50,
        gradient_tol: 1.0,
        energy_tol: 0.01,
        max_step_a: 0.1,
        include_sasa: false,
        include_cmap: false,
    };
    let _ = minimize(s, g, ff, opts);
}

fn time_path(
    s: &geom::Structure,
    g: &geom::TopologyGraph,
    ff: &chem::ForceField,
    use_tile: bool,
    warmup: usize,
    timed: usize,
) -> Result<f64, String> {
    let n = s.atom_count();
    let mut integ = FullGpuIntegrator::new(s, g, ff, 1.0, 2.0, 310.0, 1)
        .map_err(|e| format!("GPU init: {e}"))?;
    if use_tile {
        integ.enable_tile_nb_mode(s, g, ff);
    }
    let velocities = vec![Vec3::zeros(); n];
    integ.upload_initial_state(s, &velocities);
    integ.step_batch(warmup);
    let t0 = Instant::now();
    integ.step_batch(timed);
    Ok(t0.elapsed().as_secs_f64() * 1000.0 / timed as f64)
}

#[test]
#[ignore]
fn tile_vs_verlet_at_large_n() {
    if GpuContext::get().is_err() {
        eprintln!("GPU unavailable, skipping");
        return;
    }
    let warmup = 10;
    let timed = 30;
    eprintln!("\n=== Tile-vs-Verlet bench at large N ===");
    eprintln!("(short trajectories: {warmup} warmup + {timed} timed steps each)");
    eprintln!();
    // Aim for ~3000, 6000, 10000, 16000 atoms.
    for n_residues in [200usize, 400, 700, 1100] {
        let mut s = build_chain(n_residues);
        let g = build_topology_graph(&s);
        let ff = standard_ff();
        let n_atoms = s.atom_count();
        eprintln!("{n_residues} residues = {n_atoms} atoms");
        let t0 = Instant::now();
        relax_min(&mut s, &g, ff);
        eprintln!("  minimised in {:.1} s", t0.elapsed().as_secs_f64());
        let verlet_ms = match time_path(&s, &g, ff, false, warmup, timed) {
            Ok(ms) => ms,
            Err(e) => { eprintln!("  Verlet: {e}"); continue; }
        };
        let tile_ms = match time_path(&s, &g, ff, true, warmup, timed) {
            Ok(ms) => ms,
            Err(e) => { eprintln!("  Tile: {e}"); continue; }
        };
        let ratio = verlet_ms / tile_ms;
        let winner = if ratio > 1.0 { "TILE" } else { "VERLET" };
        eprintln!(
            "  Verlet {:.3} ms/step | Tile {:.3} ms/step | {winner} wins {:.2}×",
            verlet_ms, tile_ms,
            if ratio > 1.0 { ratio } else { 1.0 / ratio }
        );
    }
}
