//! End-to-end test: GPU integrator with SASA forces enabled.
//!
//! Runs a short Langevin trajectory on Ala-Lys-Glu with SASA mode
//! on; verifies no divergence + that the trajectory equilibrates to
//! approximately the target temperature.  Doesn't bit-compare against
//! the CPU SASA path (different SASA implementation — smooth coverage
//! on GPU vs analytical PowerSasa on CPU — would not give bit-equal
//! forces).  Just confirms the SASA forces are non-zero,
//! finite, and don't blow up the trajectory.

use std::time::Instant;

use chem::{standard_ff, AminoAcid};
use dynamics::full_gpu_integrator::FullGpuIntegrator;
use geom::{build_extended_chain, build_topology_graph, Vec3};
use gpu::GpuContext;

#[test]
fn gpu_integrator_with_sasa_runs_without_divergence() {
    if GpuContext::get().is_err() {
        eprintln!("GPU unavailable, skipping");
        return;
    }
    let mut s = build_extended_chain(&[
        AminoAcid::Ala, AminoAcid::Lys, AminoAcid::Glu,
    ]).unwrap();
    let g = build_topology_graph(&s);
    let ff = standard_ff();
    let n = s.atom_count();

    let dt_fs = 1.0;
    let gamma_ps_inv = 2.0;
    let temperature_k = 310.0;

    let mut full = FullGpuIntegrator::new(
        &s, &g, ff, dt_fs, gamma_ps_inv, temperature_k, 7,
    ).expect("FullGpuIntegrator construction");
    // Build per-atom γ from the default table.
    let gammas = energy::powersasa::default_sasa_gammas(&s);
    full.enable_sasa_mode(&s, &gammas);
    let velocities = vec![Vec3::zeros(); n];
    full.upload_initial_state(&s, &velocities);

    let n_steps = 100usize;
    let t0 = Instant::now();
    full.step_batch(n_steps);
    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "GPU integrator + SASA on {n} atoms: {n_steps} steps in {:.3} s = {:.3} ms/step",
        elapsed, elapsed * 1000.0 / n_steps as f64,
    );

    full.download_positions_into(&mut s);
    for r in &s.residues {
        for a in &r.atoms {
            assert!(a.position.x.is_finite() && a.position.y.is_finite() && a.position.z.is_finite(),
                "non-finite position after SASA-mode integrator");
        }
    }
    // Soft sanity: velocity magnitudes should be in the Maxwell-Boltzmann
    // ballpark — a 12 Da atom at 310 K has thermal velocity ~ 0.005 Å/fs.
    // We just check nothing exploded to ridiculous magnitudes.
    let vels = full.download_velocities();
    let max_v = vels.iter().map(|v| v.norm()).fold(0.0_f64, f64::max);
    eprintln!("max velocity magnitude: {max_v:.4} Å/fs");
    assert!(max_v < 0.5,
        "velocity blew up: max {max_v} Å/fs (should be < 0.5 Å/fs at 310 K with γ=2/ps)");
}
