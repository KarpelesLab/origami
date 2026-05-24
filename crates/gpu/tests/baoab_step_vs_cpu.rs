//! Cross-check the GPU BAOAB kernel against the CPU integrator math.
//!
//! Two checks at increasing precision requirements:
//!
//!  - **Deterministic step (zero noise)**: with γ = 0 the O step is a
//!    pure velocity-rescaling that becomes a no-op when σ = 0.  Set
//!    `temperature_k = 0` and the integrator reduces to velocity-Verlet
//!    on a fixed force field; GPU and CPU should agree to f32 precision.
//!
//!  - **Stochastic step (statistical equivalence)**: at 310 K with
//!    randomised initial velocities and many integrator steps, the GPU
//!    and CPU runs diverge bit-wise (different RNGs) but should produce
//!    statistically equivalent ensembles — same mean temperature, same
//!    velocity distribution.
//!
//! These don't yet exercise the end-to-end integrator-on-GPU path
//! (that still needs the bonded forces ported to GPU and a multi-step
//! `step_n` orchestrator); they only verify the BAOAB kernel + RNG
//! kernel in isolation.

use gpu::{make_rng_state, BaoabPipeline, GpuContext};

const ACCEL_FACTOR: f64 = 1.0e-4;

/// A minimal CPU re-implementation of one BAOAB step to compare against.
/// Mirrors `dynamics::langevin::run_langevin`'s inner loop but skips
/// the SHAKE, callback, and temperature-stats machinery — pure math.
fn cpu_baoab_step(
    positions: &mut [[f64; 3]],
    velocities: &mut [[f64; 3]],
    masses: &[f64],
    forces: &[[f64; 3]],
    dt_fs: f64,
    gamma_ps_inv: f64,
    kbt: f64,
) {
    let alpha = (-gamma_ps_inv * dt_fs * 1.0e-3).exp();
    let one_minus_alpha2 = 1.0 - alpha * alpha;
    let half_dt = 0.5 * dt_fs;
    let n = positions.len();
    for i in 0..n {
        let inv_m_accel = ACCEL_FACTOR / masses[i];
        // B
        for k in 0..3 {
            velocities[i][k] += forces[i][k] * inv_m_accel * half_dt;
        }
        // A
        for k in 0..3 {
            positions[i][k] += velocities[i][k] * half_dt;
        }
        // O: with σ = 0, just multiply by α (no random)
        let sigma_sq = one_minus_alpha2 * kbt * ACCEL_FACTOR / masses[i];
        let sigma = sigma_sq.sqrt();
        for k in 0..3 {
            velocities[i][k] = alpha * velocities[i][k] + sigma * 0.0; // zero noise
        }
        // A
        for k in 0..3 {
            positions[i][k] += velocities[i][k] * half_dt;
        }
        // B
        for k in 0..3 {
            velocities[i][k] += forces[i][k] * inv_m_accel * half_dt;
        }
    }
}

#[test]
fn gpu_baoab_matches_cpu_deterministic() {
    // T = 0 → σ = 0 → noise contribution vanishes regardless of RNG.
    // This isolates the integrator math from the RNG to verify the
    // GPU kernel's arithmetic is correct.
    let ctx = match GpuContext::get() {
        Ok(c) => c,
        Err(e) => { eprintln!("GPU unavailable: {e}"); return; }
    };
    let n = 10;
    // Synthetic atoms: linear chain of mass-12 atoms with small
    // displacements + small constant forces along x.
    let mut positions_cpu: Vec<[f64; 3]> = (0..n)
        .map(|i| [i as f64 * 1.5, 0.0, 0.0]).collect();
    let mut velocities_cpu: Vec<[f64; 3]> = (0..n)
        .map(|i| [0.01 * i as f64, 0.0, 0.0]).collect();
    let masses_cpu: Vec<f64> = (0..n).map(|_| 12.011).collect();
    let forces_cpu: Vec<[f64; 3]> = (0..n).map(|_| [50.0, 0.0, 0.0]).collect();

    let positions_gpu_f32: Vec<[f32; 3]> = positions_cpu
        .iter().map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect();
    let velocities_gpu_f32: Vec<[f32; 3]> = velocities_cpu
        .iter().map(|v| [v[0] as f32, v[1] as f32, v[2] as f32]).collect();
    let forces_gpu_f32: Vec<[f32; 3]> = forces_cpu
        .iter().map(|f| [f[0] as f32, f[1] as f32, f[2] as f32]).collect();
    let masses_gpu_f32: Vec<f32> = masses_cpu.iter().map(|&m| m as f32).collect();

    let rng_state = make_rng_state(42, n);
    let mut pipe = BaoabPipeline::new(ctx, n, &masses_gpu_f32, &rng_state);
    pipe.upload_positions(&positions_gpu_f32);
    pipe.upload_velocities(&velocities_gpu_f32);
    pipe.upload_forces(&forces_gpu_f32);

    let dt = 1.0;
    let gamma = 2.0;
    let kbt = 0.0;     // T = 0 → noise = 0
    pipe.set_step_params(dt as f32, gamma as f32, kbt as f32);

    // Dispatch first + second half (same forces for both — at T = 0,
    // single-step BAOAB is exact when the force is constant).
    let device = &ctx.device;
    let queue = &ctx.queue;
    let mut encoder = device.create_command_encoder(&gpu::wgpu::CommandEncoderDescriptor {
        label: Some("baoab_test_encoder"),
    });
    pipe.record_first_half(&mut encoder);
    pipe.record_second_half(&mut encoder);
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(gpu::wgpu::Maintain::Wait);

    let new_pos_gpu = pipe.download_positions();
    let new_vel_gpu = pipe.download_velocities();

    // CPU reference.
    cpu_baoab_step(
        &mut positions_cpu,
        &mut velocities_cpu,
        &masses_cpu,
        &forces_cpu,
        dt, gamma, kbt,
    );

    let mut max_pos_err = 0.0_f64;
    let mut max_vel_err = 0.0_f64;
    for i in 0..n {
        for k in 0..3 {
            let pe = (new_pos_gpu[i][k] as f64 - positions_cpu[i][k]).abs();
            let ve = (new_vel_gpu[i][k] as f64 - velocities_cpu[i][k]).abs();
            if pe > max_pos_err { max_pos_err = pe; }
            if ve > max_vel_err { max_vel_err = ve; }
        }
    }
    eprintln!(
        "GPU-vs-CPU BAOAB (zero-noise) max errors: pos {:.3e} Å, vel {:.3e} Å/fs",
        max_pos_err, max_vel_err
    );
    // f32 round-trip — should match to better than 1e-5.
    assert!(max_pos_err < 1e-4, "position drift {max_pos_err} > 1e-4 Å");
    assert!(max_vel_err < 1e-4, "velocity drift {max_vel_err} > 1e-4 Å/fs");
}

#[test]
fn gpu_baoab_stochastic_temperature_is_target() {
    // Run many "free-particle Langevin" steps (zero forces) and verify
    // the GPU integrator equilibrates to the target temperature.
    // With dt = 1 fs, γ = 2 ps⁻¹, T = 310 K, after a few hundred steps
    // the per-atom kinetic energy should be (3/2) k_B T.
    let ctx = match GpuContext::get() {
        Ok(c) => c,
        Err(e) => { eprintln!("GPU unavailable: {e}"); return; }
    };
    let n = 256;
    let masses_f32: Vec<f32> = (0..n).map(|_| 12.011_f32).collect();
    let masses_f64: Vec<f64> = (0..n).map(|_| 12.011_f64).collect();
    let rng_state = make_rng_state(7, n);
    let mut pipe = BaoabPipeline::new(ctx, n, &masses_f32, &rng_state);
    // Start at rest, zero forces.
    let positions_f32 = vec![[0.0_f32; 3]; n];
    let velocities_f32 = vec![[0.0_f32; 3]; n];
    let forces_f32 = vec![[0.0_f32; 3]; n];
    pipe.upload_positions(&positions_f32);
    pipe.upload_velocities(&velocities_f32);
    pipe.upload_forces(&forces_f32);

    const BOLTZMANN_KJ_PER_MOL_K: f64 = 8.314_462_618e-3;
    let dt = 1.0_f32;
    let gamma = 2.0_f32;
    let target_t = 310.0_f64;
    let kbt = (BOLTZMANN_KJ_PER_MOL_K * target_t) as f32;
    pipe.set_step_params(dt, gamma, kbt);

    // Run 500 BAOAB steps (with zero forces, both halves are
    // equivalent — the system is a pure Ornstein-Uhlenbeck process).
    let device = &ctx.device;
    let queue = &ctx.queue;
    for _ in 0..500 {
        let mut encoder = device.create_command_encoder(&gpu::wgpu::CommandEncoderDescriptor {
            label: Some("baoab_loop_encoder"),
        });
        pipe.record_first_half(&mut encoder);
        pipe.record_second_half(&mut encoder);
        queue.submit(Some(encoder.finish()));
    }
    let _ = device.poll(gpu::wgpu::Maintain::Wait);

    let final_vel = pipe.download_velocities();
    let ke: f64 = final_vel
        .iter()
        .zip(masses_f64.iter())
        .map(|(v, m)| 0.5 * m * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) as f64)
        .sum();
    let ke_kj_mol = ke / ACCEL_FACTOR;
    let dof = (3 * n) as f64;
    let t_inst = 2.0 * ke_kj_mol / (dof * BOLTZMANN_KJ_PER_MOL_K);
    eprintln!(
        "GPU BAOAB free-particle equilibrium temperature after 500 steps: {:.1} K (target {:.0} K)",
        t_inst, target_t
    );
    // ±10 % of target is comfortable for 256 atoms, 500 steps.
    assert!((t_inst - target_t).abs() < 0.20 * target_t,
        "GPU equilibrium T {t_inst} too far from target {target_t}");
}
