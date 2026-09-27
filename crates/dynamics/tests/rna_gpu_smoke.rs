//! Smoke test: does the GPU integrator construct and run on an RNA
//! chain at all?  The GPU pipelines were designed against protein
//! topologies; this test probes whether the same code paths cope
//! with RNA's atom types (P, On3, Cn8b, …) and the phosphodiester
//! bonded tuples coming out of CHARMM27.
//!
//! Acceptance: the integrator builds, advances 50 steps, and the
//! positions / velocities stay finite.  No comparison to the CPU
//! path yet — that comes in the next test.

use chem::{Nucleotide, standard_ff};
use dynamics::full_gpu_integrator::FullGpuIntegrator;
use dynamics::{Algorithm, MinimizeOptions, minimize};
use geom::{Vec3, build_extended_rna_chain, build_topology_graph};
use gpu::GpuContext;

#[test]
fn gpu_integrator_runs_on_short_rna_chain() {
    if GpuContext::get().is_err() {
        eprintln!("GPU unavailable, skipping");
        return;
    }
    let mut s = build_extended_rna_chain(&[Nucleotide::Uracil, Nucleotide::Adenine]).unwrap();
    let g = build_topology_graph(&s);
    let ff = standard_ff();

    // Pre-minimise: extended-NeRF RNA starts with high bond energies
    // (same pattern as the CPU `rna_chain_langevin_runs_without_explosion`
    // test).  Without this the first GPU step gets megaton forces.
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
    let mut integ = FullGpuIntegrator::new(
        &s, &g, ff, /*dt_fs=*/ 1.0, /*gamma_ps_inv=*/ 2.0, /*temperature_k=*/ 310.0,
        /*seed=*/ 17,
    )
    .expect("FullGpuIntegrator::new on RNA");

    let velocities = vec![Vec3::zeros(); n];
    integ.upload_initial_state(&s, &velocities);

    integ.step_batch(50);
    integ.download_positions_into(&mut s);

    for r in &s.residues {
        for a in &r.atoms {
            assert!(
                a.position.x.is_finite() && a.position.y.is_finite() && a.position.z.is_finite(),
                "non-finite atom {} after GPU integrator on RNA: {:?}",
                a.name,
                a.position
            );
        }
    }

    let vels = integ.download_velocities();
    let max_v = vels.iter().map(|v| v.norm()).fold(0.0_f64, f64::max);
    eprintln!("UA dimer ({n} atoms) GPU-Langevin: max |v| after 50 steps = {max_v:.4} Å/fs");
    assert!(
        max_v < 1.0,
        "RNA velocity blew up on GPU: max {max_v} Å/fs (should be < 1 Å/fs at 310 K)"
    );
}
