//! Profile each phase of the GPU integrator's per-batch cost.
//!
//! Hypothesis from the SHAKE bench: bare `step_batch(N)` is much
//! faster than the same workload through `run_langevin` with
//! callback orchestration.  This test breaks down where the time
//! actually goes.
//!
//! Marked `#[ignore]`; run explicitly:
//!
//! ```text
//! cargo test --release -p dynamics --test gpu_overhead_profile -- --ignored --nocapture
//! ```

use std::time::Instant;

use chem::{AminoAcid, standard_ff};
use dynamics::{LangevinOptions, full_gpu_integrator::FullGpuIntegrator, run_langevin};
use geom::{Vec3, build_extended_chain, build_topology_graph};
use gpu::GpuContext;

fn build_chain(n_residues: usize) -> geom::Structure {
    let mut seq = Vec::with_capacity(n_residues);
    let block = [
        AminoAcid::Ala,
        AminoAcid::Gly,
        AminoAcid::Leu,
        AminoAcid::Glu,
        AminoAcid::Lys,
    ];
    for i in 0..n_residues {
        seq.push(block[i % block.len()]);
    }
    build_extended_chain(&seq).expect("build")
}

#[test]
#[ignore]
fn profile_run_langevin_vs_bare_step_batch() {
    if GpuContext::get().is_err() {
        eprintln!("GPU unavailable, skipping");
        return;
    }
    // Big-N system where the overhead matters most.
    let s = build_chain(400); // ~5840 atoms
    let g = build_topology_graph(&s);
    let ff = standard_ff();
    let n = s.atom_count();
    eprintln!("\n=== Profile target: {n} atoms ===");

    // ---- Phase 1: construction ----
    let t0 = Instant::now();
    let mut full = FullGpuIntegrator::new(&s, &g, ff, 1.0, 2.0, 310.0, 1).expect("construct");
    let construct_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "Construction (FullGpuIntegrator::new): {:.2} ms",
        construct_ms
    );

    // ---- Phase 2: upload_initial_state (includes refresh_neighbour_lists) ----
    let velocities = vec![Vec3::zeros(); n];
    let t0 = Instant::now();
    full.upload_initial_state(&s, &velocities);
    let upload_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "upload_initial_state (incl. initial nb/gb build): {:.2} ms",
        upload_ms
    );

    // ---- Phase 3: warmup ----
    full.step_batch(20);

    // ---- Phase 4: time individual step_batch(N) calls ----
    for batch in [1, 5, 25, 50, 100] {
        let t0 = Instant::now();
        full.step_batch(batch);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "step_batch({batch}): {:.2} ms total = {:.3} ms/step",
            ms,
            ms / batch as f64
        );
    }

    // ---- Phase 5: time download_positions_into in isolation ----
    let mut s_copy = s.clone();
    let mut total_ms = 0.0;
    let n_iters = 20;
    for _ in 0..n_iters {
        let t0 = Instant::now();
        full.download_positions_into(&mut s_copy);
        total_ms += t0.elapsed().as_secs_f64() * 1000.0;
    }
    eprintln!(
        "download_positions_into (avg over {n_iters} calls): {:.3} ms",
        total_ms / n_iters as f64
    );

    let mut total_ms = 0.0;
    for _ in 0..n_iters {
        let t0 = Instant::now();
        let _v = full.download_velocities();
        total_ms += t0.elapsed().as_secs_f64() * 1000.0;
    }
    eprintln!(
        "download_velocities (avg over {n_iters} calls): {:.3} ms",
        total_ms / n_iters as f64
    );

    // ---- Phase 6: time the per-save_every pattern ----
    //
    // Mimic run_langevin's per-batch sequence: step_batch(25),
    // download_positions_into, download_velocities, compute KE.
    let mut s_copy = s.clone();
    let mut total_ms = 0.0;
    let batches = 4;
    for _ in 0..batches {
        let t0 = Instant::now();
        full.step_batch(25);
        full.download_positions_into(&mut s_copy);
        let _v = full.download_velocities();
        total_ms += t0.elapsed().as_secs_f64() * 1000.0;
    }
    eprintln!(
        "Per-batch loop pattern (step_batch(25) + 2 downloads, {batches}×): \
         total {:.2} ms = {:.2} ms/batch = {:.3} ms/step",
        total_ms,
        total_ms / batches as f64,
        total_ms / (batches as f64 * 25.0),
    );

    // ---- Phase 7: full run_langevin for comparison ----
    let mut s_run = s.clone();
    let opts = LangevinOptions {
        dt_fs: 1.0,
        temperature_k: 310.0,
        friction_ps_inv: 2.0,
        steps: 100,
        save_every: 25,
        seed: 1,
        randomise_initial_velocities: true,
        include_sasa: false,
        include_cmap: false,
        constrain_h_bonds: false,
        use_gpu: false,
        use_gpu_integrator: true,
    };
    let t0 = Instant::now();
    let _ = run_langevin(&mut s_run, &g, ff, opts, |_| {});
    let run_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "Full run_langevin(steps=100, save_every=25): {:.2} ms = {:.3} ms/step",
        run_ms,
        run_ms / 100.0
    );
}
