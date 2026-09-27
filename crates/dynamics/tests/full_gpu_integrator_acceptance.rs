//! Acceptance test for `LangevinOptions::use_gpu_integrator`.
//!
//! Builds Ala-Lys-Glu, runs the same Langevin trajectory twice — once
//! purely on CPU, once with the full GPU integrator — and verifies
//! both produce physically reasonable equilibrium statistics.  Doesn't
//! assert bit-equality (different RNG between GPU xoshiro128++ and
//! CPU xoshiro256++, plus f32 round-off, makes any longer-than-few-step
//! comparison diverge chaotically) — only that the GPU integrator hits
//! a similar mean temperature and doesn't explode.

use chem::{standard_ff, AminoAcid};
use dynamics::{run_langevin, LangevinOptions};
use geom::{build_extended_chain, build_topology_graph};
use gpu::GpuContext;

#[test]
fn gpu_integrator_matches_cpu_temperature_on_ala_lys_glu() {
    if GpuContext::get().is_err() {
        eprintln!("GPU unavailable, skipping");
        return;
    }
    let make_struct =
        || build_extended_chain(&[AminoAcid::Ala, AminoAcid::Lys, AminoAcid::Glu]).unwrap();

    let template = LangevinOptions {
        dt_fs: 1.0,
        temperature_k: 310.0,
        friction_ps_inv: 2.0,
        steps: 1000, // long enough to clear burn-in
        save_every: 100,
        seed: 17,
        randomise_initial_velocities: true,
        include_sasa: false,
        include_cmap: false,
        constrain_h_bonds: false,
        use_gpu: false,
        use_gpu_integrator: false,
    };

    let mut s_cpu = make_struct();
    let g_cpu = build_topology_graph(&s_cpu);
    let ff = standard_ff();
    let sum_cpu = run_langevin(&mut s_cpu, &g_cpu, ff, template, |_| {});

    let mut s_gpu = make_struct();
    let g_gpu = build_topology_graph(&s_gpu);
    let mut gpu_opts = template;
    gpu_opts.use_gpu_integrator = true;
    let sum_gpu = run_langevin(&mut s_gpu, &g_gpu, ff, gpu_opts, |_| {});

    assert!(!sum_cpu.diverged, "CPU run diverged");
    assert!(!sum_gpu.diverged, "GPU integrator run diverged");

    eprintln!(
        "CPU integrator: T_mean = {:.1} K, equipartition = {:.3}",
        sum_cpu.temperature_mean_k, sum_cpu.equipartition_ratio
    );
    eprintln!(
        "GPU integrator: T_mean = {:.1} K, equipartition = {:.3}",
        sum_gpu.temperature_mean_k, sum_gpu.equipartition_ratio
    );

    // Both should be in spec for a 1000-step trajectory after the
    // ~100-step burn-in.  Loose ±100 K — accepts the natural
    // single-trajectory variance at 47 atoms.
    assert!(
        (sum_cpu.temperature_mean_k - 310.0).abs() < 100.0,
        "CPU T_mean {:.1} far from target",
        sum_cpu.temperature_mean_k
    );
    assert!(
        (sum_gpu.temperature_mean_k - 310.0).abs() < 100.0,
        "GPU T_mean {:.1} far from target",
        sum_gpu.temperature_mean_k
    );

    // No atom catastrophe — final positions should be finite.
    for r in &s_gpu.residues {
        for a in &r.atoms {
            assert!(
                a.position.x.is_finite() && a.position.y.is_finite() && a.position.z.is_finite(),
                "non-finite final position"
            );
        }
    }
}
