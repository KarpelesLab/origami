//! Bench: GPU integrator throughput on RNA chains of varying length.
//! Mirrors `gpu_sasa_bench` for proteins.  Confirms the GPU stack
//! scales the same way for RNA as it does for proteins, and that the
//! phosphodiester backbone (more atoms per residue, more bonded terms)
//! doesn't introduce a per-residue penalty beyond the count itself.
//!
//! Run:
//! ```text
//! cargo test --release -p dynamics --test rna_gpu_bench -- --ignored --nocapture
//! ```

use std::time::Instant;

use chem::{standard_ff, Nucleotide};
use dynamics::full_gpu_integrator::FullGpuIntegrator;
use dynamics::{minimize, Algorithm, MinimizeOptions};
use geom::{build_extended_rna_chain, build_topology_graph, Vec3};
use gpu::GpuContext;

fn build_rna(n: usize) -> geom::Structure {
    let block = [
        Nucleotide::Adenine,
        Nucleotide::Uracil,
        Nucleotide::Guanine,
        Nucleotide::Cytosine,
    ];
    let seq: Vec<Nucleotide> = block.iter().cloned().cycle().take(n).collect();
    build_extended_rna_chain(&seq).expect("build")
}

#[test]
#[ignore]
fn bench_gpu_integrator_on_rna() {
    if GpuContext::get().is_err() {
        eprintln!("GPU unavailable, skipping");
        return;
    }
    eprintln!("\n=== GPU integrator on RNA (extended chain, dt=1 fs, T=310 K) ===\n");

    let warmup = 10usize;
    let timed = 50usize;
    for n_residues in [4usize, 10, 25, 50, 100] {
        let mut s = build_rna(n_residues);
        let g = build_topology_graph(&s);
        let ff = standard_ff();
        // Cheap minimisation so the first step doesn't see extended-NeRF
        // strain.  Real production workloads would minimise harder.
        minimize(
            &mut s,
            &g,
            ff,
            MinimizeOptions {
                algorithm: Algorithm::Lbfgs,
                max_steps: 200,
                ..MinimizeOptions::default()
            },
        );
        let n = s.atom_count();

        let mut integ = match FullGpuIntegrator::new(&s, &g, ff, 1.0, 2.0, 310.0, 1) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("{n_residues:3} residues: GPU init failed: {e}");
                continue;
            }
        };
        let velocities = vec![Vec3::zeros(); n];
        integ.upload_initial_state(&s, &velocities);

        integ.step_batch(warmup);
        let t0 = Instant::now();
        integ.step_batch(timed);
        let ms_per_step = t0.elapsed().as_secs_f64() * 1000.0 / timed as f64;

        // Sanity: no divergence.
        integ.download_positions_into(&mut s);
        let mut any_nan = false;
        for r in &s.residues {
            for a in &r.atoms {
                if !(a.position.x.is_finite()
                    && a.position.y.is_finite()
                    && a.position.z.is_finite())
                {
                    any_nan = true;
                }
            }
        }
        assert!(!any_nan, "RNA trajectory diverged at {n_residues} residues");

        eprintln!("{n_residues:3} residues / {n:5} atoms  |  {ms_per_step:7.3} ms/step",);
    }
}
