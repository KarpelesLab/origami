//! Measure the actual GPU-vs-CPU crossover on real proteins.
//!
//! Runs a short Langevin trajectory on each fixture twice — once
//! purely on the CPU, once with `use_gpu = true` — and reports the
//! wall-clock per step for the force evaluations.  Bonded forces still
//! run on the CPU in both cases, so what we're measuring is the
//! difference in the LJ + Coulomb + GB pair-loop cost.
//!
//! Marked `#[ignore]` so it doesn't slow the normal test suite — run
//! explicitly with:
//!
//! ```text
//! cargo test --release -p dynamics --test gpu_vs_cpu_bench -- --ignored --nocapture
//! ```
//!
//! Release build matters: debug builds run the CPU SoA pair loop ~20×
//! slower, which artificially favours the GPU.  The crossover number
//! we care about is from optimised CPU code.

use std::time::Instant;

use chem::{classify_atom, standard_ff, AminoAcid, AtomType, Element};
use dynamics::shake::build_h_bond_constraints;
use dynamics::full_gpu_integrator::FullGpuIntegrator;
use dynamics::{minimize, run_langevin, Algorithm, LangevinOptions, MinimizeOptions};
use geom::{build_extended_chain, build_topology_graph, Vec3};
use gpu::{build_per_x_shake_data, GpuContext, ShakeConstraint};
use io::read_pdb;

fn read_fixture(path: &str) -> geom::Structure {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    read_pdb(bytes.as_slice()).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

/// Brief L-BFGS minimisation to relax crystal-structure strain.
/// Without this, MD on a raw PDB explodes within a few fs from
/// off-equilibrium bond lengths (typical real-crystal bond strain
/// generates 10 000+ K kinetic energy as soon as Langevin starts
/// distributing it across DoF).
fn relax(s: &mut geom::Structure, g: &geom::TopologyGraph, ff: &chem::ForceField) {
    let opts = MinimizeOptions {
        algorithm: Algorithm::Lbfgs,
        max_steps: 100,
        gradient_tol: 1.0,
        energy_tol: 0.01,
        max_step_a: 0.1,
        include_sasa: false,
        include_cmap: false,
    };
    let _ = minimize(s, g, ff, opts);
}

fn bench_one(path: &str, label: &str, warmup: usize, timed: usize) {
    // Quick CPU-availability sanity, plus produce both topologies up front.
    let mut s = read_fixture(path);
    let n = s.atom_count();
    let g = build_topology_graph(&s);
    let ff = standard_ff();
    eprintln!("\n=== {label} — {n} atoms ({} residues) ===", s.residues.len());
    // Brief minimisation to absorb crystal-strain bond strain.  Without
    // this, raw-PDB structures often have 0.05 Å bond offsets from r₀
    // which Langevin redistributes into 1000+ K kinetic energy.
    let t0 = Instant::now();
    relax(&mut s, &g, ff);
    eprintln!("  relaxed in {:.2} s", t0.elapsed().as_secs_f64());

    let template = LangevinOptions {
        dt_fs: 1.0,
        temperature_k: 310.0,
        friction_ps_inv: 2.0,
        steps: warmup + timed,
        save_every: 0, // no callback overhead
        seed: 1,
        randomise_initial_velocities: true,
        include_sasa: false,
        include_cmap: false,
        constrain_h_bonds: false,
        use_gpu: false,
        use_gpu_integrator: false,
    };

    // CPU: do a short warmup run (lets the JIT / branch predictors
    // settle) then a timed run.  Two separate trajectories so the
    // warmup doesn't pollute the position state.
    let mut s_cpu = s.clone();
    let mut cpu_opts = template;
    cpu_opts.steps = warmup;
    run_langevin(&mut s_cpu, &g, ff, cpu_opts, |_| {});

    let mut s_cpu = s.clone();
    cpu_opts.steps = timed;
    let t0 = Instant::now();
    let sum_cpu = run_langevin(&mut s_cpu, &g, ff, cpu_opts, |_| {});
    let cpu_secs = t0.elapsed().as_secs_f64();
    let cpu_per_step_ms = cpu_secs * 1000.0 / timed as f64;
    eprintln!(
        "  CPU: {timed} steps in {:.2} s — {:.3} ms/step  (T_mean {:.1} K, diverged={})",
        cpu_secs, cpu_per_step_ms, sum_cpu.temperature_mean_k, sum_cpu.diverged
    );

    // GPU — pair forces only.
    let mut s_gpu = s.clone();
    let mut gpu_opts = template;
    gpu_opts.use_gpu = true;
    gpu_opts.steps = warmup;
    run_langevin(&mut s_gpu, &g, ff, gpu_opts, |_| {});
    let mut s_gpu = s.clone();
    gpu_opts.steps = timed;
    let t0 = Instant::now();
    let sum_gpu = run_langevin(&mut s_gpu, &g, ff, gpu_opts, |_| {});
    let gpu_secs = t0.elapsed().as_secs_f64();
    let gpu_per_step_ms = gpu_secs * 1000.0 / timed as f64;
    eprintln!(
        "  GPU (pair only): {timed} steps in {:.2} s — {:.3} ms/step  (T_mean {:.1} K)",
        gpu_secs, gpu_per_step_ms, sum_gpu.temperature_mean_k,
    );

    // Full GPU integrator (bonded + pair + BAOAB all on device).
    let mut s_gi = s.clone();
    let mut gi_opts = template;
    gi_opts.use_gpu = false;
    gi_opts.use_gpu_integrator = true;
    gi_opts.steps = warmup;
    run_langevin(&mut s_gi, &g, ff, gi_opts, |_| {});
    let mut s_gi = s.clone();
    gi_opts.steps = timed;
    // Save_every of 0 means no callback overhead — the integrator
    // can run a full `timed` batch without intermediate sync.  Pick a
    // realistic save cadence instead to see how the per-batch sync
    // cost shapes per-step time.
    gi_opts.save_every = (timed / 4).max(10);
    let t0 = Instant::now();
    let sum_gi = run_langevin(&mut s_gi, &g, ff, gi_opts, |_| {});
    let gi_secs = t0.elapsed().as_secs_f64();
    let gi_per_step_ms = gi_secs * 1000.0 / timed as f64;
    eprintln!(
        "  GPU (full integrator, save_every={}): {timed} steps in {:.2} s — {:.3} ms/step  (T_mean {:.1} K)",
        gi_opts.save_every, gi_secs, gi_per_step_ms, sum_gi.temperature_mean_k,
    );

    let speedup_pair = cpu_per_step_ms / gpu_per_step_ms;
    let speedup_full = cpu_per_step_ms / gi_per_step_ms;
    eprintln!(
        "  speedup vs CPU: pair-only {:.2}×, full-integrator {:.2}×",
        speedup_pair, speedup_full
    );

    // SHAKE-mode integrator at dt = 2 fs.  Walltime metric is
    // ms/simulated-fs (= ms/step / 2), which is what matters for
    // trajectories of fixed simulated length.
    bench_shake_arm(&s, &g, ff, label, warmup, timed, cpu_per_step_ms);
    bench_tile_arm(&s, &g, ff, label, warmup, timed, cpu_per_step_ms);
}

fn bench_tile_arm(
    s: &geom::Structure,
    g: &geom::TopologyGraph,
    ff: &chem::ForceField,
    _label: &str,
    warmup: usize,
    timed: usize,
    cpu_ms: f64,
) {
    let n = s.atom_count();
    let mut tile = match FullGpuIntegrator::new(s, g, ff, 1.0, 2.0, 310.0, 1) {
        Ok(f) => f,
        Err(e) => { eprintln!("  TILE arm: GPU unavailable ({e})"); return; }
    };
    tile.enable_tile_nb_mode(s, g, ff);
    let velocities = vec![Vec3::zeros(); n];
    tile.upload_initial_state(s, &velocities);
    tile.step_batch(warmup);
    let t0 = Instant::now();
    tile.step_batch(timed);
    let secs = t0.elapsed().as_secs_f64();
    let ms_per_step = secs * 1000.0 / timed as f64;
    eprintln!(
        "  GPU (TILE nb, dt=1 fs, bare step_batch): {timed} steps in {:.2} s — {:.3} ms/step  (vs Verlet GPU full: {:.2}×, vs CPU: {:.2}×)",
        secs, ms_per_step,
        // The earlier "GPU full integrator" line ran the Verlet path
        // through run_langevin — different overheads.  The tile is a
        // bare step_batch, so the comparable baseline is the bare
        // step_batch we ran in bench_shake_arm above (no SHAKE).  We
        // don't have it inline here — just report ms vs CPU.
        cpu_ms / ms_per_step,
        cpu_ms / ms_per_step,
    );
}

fn bench_shake_arm(
    s: &geom::Structure,
    g: &geom::TopologyGraph,
    ff: &chem::ForceField,
    _label: &str,
    warmup: usize,
    timed: usize,
    cpu_ms_per_fs_dt1: f64,
) {
    let n = s.atom_count();
    let atom_types: Vec<AtomType> = s.residues.iter()
        .flat_map(|r| r.atoms.iter()
            .map(|a| classify_atom(r.monomer, a.name).unwrap()))
        .collect();
    let cpu_constraints = build_h_bond_constraints(s, g, ff, &atom_types);
    let atoms_flat: Vec<Element> = s.residues.iter()
        .flat_map(|r| r.atoms.iter().map(|a| a.element)).collect();
    let gpu_constraints: Vec<ShakeConstraint> = cpu_constraints.iter().map(|c| {
        let (x, h) = if atoms_flat[c.i] == Element::H {
            (c.j as u32, c.i as u32)
        } else {
            (c.i as u32, c.j as u32)
        };
        ShakeConstraint { x_atom: x, h_atom: h, d_sq: c.d_sq as f32 }
    }).collect();
    let masses_f32: Vec<f32> = s.residues.iter()
        .flat_map(|r| r.atoms.iter().map(|a| a.element.mass_da() as f32))
        .collect();
    let shake_data = build_per_x_shake_data(n, &gpu_constraints, &masses_f32);

    // ---- Fair baseline: time the SAME `step_batch(timed)` call on
    // a non-SHAKE FullGpuIntegrator (dt = 1 fs).  Strips out
    // run_langevin's callback / save-every / construction overhead so
    // we're comparing apples-to-apples.
    let mut no_shake = match FullGpuIntegrator::new(s, g, ff, 1.0, 2.0, 310.0, 1) {
        Ok(f) => f,
        Err(e) => { eprintln!("  SHAKE arm: GPU unavailable ({e})"); return; }
    };
    let velocities = vec![Vec3::zeros(); n];
    no_shake.upload_initial_state(s, &velocities);
    no_shake.step_batch(warmup);
    let t0 = Instant::now();
    no_shake.step_batch(timed);
    let no_shake_secs = t0.elapsed().as_secs_f64();
    let no_shake_ms_per_step = no_shake_secs * 1000.0 / timed as f64;
    eprintln!(
        "  GPU (no SHAKE, dt=1 fs, bare step_batch): {timed} steps in {:.2} s — {:.3} ms/step = {:.3} ms/fs",
        no_shake_secs, no_shake_ms_per_step, no_shake_ms_per_step / 1.0
    );

    let mut full = match FullGpuIntegrator::new(s, g, ff, 2.0, 2.0, 310.0, 1) {
        Ok(f) => f,
        Err(e) => { eprintln!("  SHAKE arm: GPU unavailable ({e})"); return; }
    };
    full.enable_shake(&shake_data, 64, 1e-6);
    full.upload_initial_state(s, &velocities);
    full.step_batch_shake(warmup);
    let t0 = Instant::now();
    full.step_batch_shake(timed);
    let secs = t0.elapsed().as_secs_f64();
    let ms_per_step = secs * 1000.0 / timed as f64;
    let ms_per_fs = ms_per_step / 2.0;  // dt = 2 fs
    eprintln!(
        "  GPU (SHAKE, dt=2 fs, bare step_batch_shake): {timed} steps in {:.2} s — {:.3} ms/step = {:.3} ms/fs",
        secs, ms_per_step, ms_per_fs
    );
    eprintln!(
        "  per-fs comparison: no-SHAKE {:.3} ms/fs vs SHAKE {:.3} ms/fs — SHAKE speedup {:.2}× (also vs CPU: {:.2}×)",
        no_shake_ms_per_step,
        ms_per_fs,
        no_shake_ms_per_step / ms_per_fs,
        cpu_ms_per_fs_dt1 / ms_per_fs
    );
}

/// Variant of [`bench_one`] that works directly on a built (in-memory)
/// `Structure` — used for the synthetic 2-k+ atom probes where there's
/// no PDB fixture and reading from disk would be silly.
fn bench_built(mut s: geom::Structure, label: &str, warmup: usize, timed: usize) {
    let n = s.atom_count();
    let g = build_topology_graph(&s);
    let ff = standard_ff();
    eprintln!("\n=== {label} — {n} atoms ({} residues) ===", s.residues.len());
    let t0 = Instant::now();
    relax(&mut s, &g, ff);
    eprintln!("  relaxed in {:.2} s", t0.elapsed().as_secs_f64());

    let template = LangevinOptions {
        dt_fs: 1.0,
        temperature_k: 310.0,
        friction_ps_inv: 2.0,
        steps: warmup + timed,
        save_every: 0,
        seed: 1,
        randomise_initial_velocities: true,
        include_sasa: false,
        include_cmap: false,
        constrain_h_bonds: false,
        use_gpu: false,
        use_gpu_integrator: false,
    };
    let mut s_cpu = s.clone();
    let mut cpu_opts = template;
    cpu_opts.steps = warmup;
    run_langevin(&mut s_cpu, &g, ff, cpu_opts, |_| {});
    let mut s_cpu = s.clone();
    cpu_opts.steps = timed;
    let t0 = Instant::now();
    let sum_cpu = run_langevin(&mut s_cpu, &g, ff, cpu_opts, |_| {});
    let cpu_secs = t0.elapsed().as_secs_f64();
    let cpu_ms = cpu_secs * 1000.0 / timed as f64;
    eprintln!(
        "  CPU: {timed} steps in {:.2} s — {:.3} ms/step  (T_mean {:.1} K)",
        cpu_secs, cpu_ms, sum_cpu.temperature_mean_k
    );

    let mut s_gpu = s.clone();
    let mut gpu_opts = template;
    gpu_opts.use_gpu = true;
    gpu_opts.steps = warmup;
    run_langevin(&mut s_gpu, &g, ff, gpu_opts, |_| {});
    let mut s_gpu = s.clone();
    gpu_opts.steps = timed;
    let t0 = Instant::now();
    let sum_gpu = run_langevin(&mut s_gpu, &g, ff, gpu_opts, |_| {});
    let gpu_secs = t0.elapsed().as_secs_f64();
    let gpu_ms = gpu_secs * 1000.0 / timed as f64;
    eprintln!(
        "  GPU (pair only): {timed} steps in {:.2} s — {:.3} ms/step  (T_mean {:.1} K)",
        gpu_secs, gpu_ms, sum_gpu.temperature_mean_k
    );

    // Full GPU integrator (bonded + pair + BAOAB on device).
    let mut s_gi = s.clone();
    let mut gi_opts = template;
    gi_opts.use_gpu = false;
    gi_opts.use_gpu_integrator = true;
    gi_opts.steps = warmup;
    run_langevin(&mut s_gi, &g, ff, gi_opts, |_| {});
    let mut s_gi = s.clone();
    gi_opts.steps = timed;
    gi_opts.save_every = (timed / 4).max(10);
    let t0 = Instant::now();
    let sum_gi = run_langevin(&mut s_gi, &g, ff, gi_opts, |_| {});
    let gi_secs = t0.elapsed().as_secs_f64();
    let gi_ms = gi_secs * 1000.0 / timed as f64;
    eprintln!(
        "  GPU (full integrator, save_every={}): {timed} steps in {:.2} s — {:.3} ms/step  (T_mean {:.1} K)",
        gi_opts.save_every, gi_secs, gi_ms, sum_gi.temperature_mean_k,
    );

    let speedup_pair = cpu_ms / gpu_ms;
    let speedup_full = cpu_ms / gi_ms;
    eprintln!(
        "  speedup vs CPU: pair-only {:.2}×, full-integrator {:.2}×",
        speedup_pair, speedup_full
    );
    bench_shake_arm(&s, &g, ff, label, warmup, timed, cpu_ms);
    bench_tile_arm(&s, &g, ff, label, warmup, timed, cpu_ms);
}

#[test]
#[ignore]
fn bench_gpu_vs_cpu_across_sizes() {
    if GpuContext::get().is_err() {
        eprintln!("GPU unavailable, skipping bench");
        return;
    }
    // 20 warmup steps + 100 timed steps balances accuracy against
    // bench-test wall-time.  Insulin alone takes ~20 s on CPU per
    // 100-step trajectory, so larger samples bloat the bench.
    let warmup = 20;
    let timed = 100;
    bench_one("../io/tests/fixtures/1L2Y_model1.pdb", "Trp-cage 1L2Y", warmup, timed);
    bench_one("../io/tests/fixtures/1CRN_crambin.pdb", "Crambin 1CRN", warmup, timed);
    bench_one("../io/tests/fixtures/2F4K_villin_hp35.pdb", "Villin HP-35 2F4K", warmup, timed);
    bench_one("../io/tests/fixtures/2HIU_insulin.pdb", "Insulin 2HIU (first MODEL)", warmup, timed);

    // Synthetic above-crossover probes.  Built poly-(AGLEK)ₙ extended
    // chains so the sequence variety exercises every relevant atom
    // type and per-atom charge.  These never get to fold — we just
    // need realistic per-step pair-loop work at known size.
    let block: Vec<AminoAcid> = vec![
        AminoAcid::Ala, AminoAcid::Gly, AminoAcid::Leu, AminoAcid::Glu, AminoAcid::Lys,
    ];
    for (n_reps, label) in [(20, "~1300 atoms"), (40, "~2600 atoms"), (80, "~5300 atoms")] {
        let seq: Vec<AminoAcid> = block.iter().cloned().cycle().take(n_reps * 5).collect();
        let s = build_extended_chain(&seq).expect("build extended chain");
        let prefixed = format!("Built {n_reps}×AGLEK ({label})");
        bench_built(s, &prefixed, warmup, timed);
    }
}

/// Diagnostic: villin only, with per-step progress reporting, to
/// identify if/where the GPU path stalls.
#[test]
#[ignore]
fn bench_villin_diagnostic() {
    if GpuContext::get().is_err() {
        eprintln!("GPU unavailable, skipping bench");
        return;
    }
    let s = read_fixture("../io/tests/fixtures/2F4K_villin_hp35.pdb");
    let n = s.atom_count();
    let g = build_topology_graph(&s);
    let ff = standard_ff();
    eprintln!("villin: {n} atoms");

    let mut s_gpu = s.clone();
    let opts = LangevinOptions {
        dt_fs: 1.0,
        temperature_k: 310.0,
        friction_ps_inv: 2.0,
        steps: 20,
        save_every: 1,
        seed: 1,
        randomise_initial_velocities: true,
        include_sasa: false,
        include_cmap: false,
        constrain_h_bonds: false,
        use_gpu: true,
        use_gpu_integrator: false,
    };
    let t0 = Instant::now();
    run_langevin(&mut s_gpu, &g, ff, opts, |frame| {
        eprintln!(
            "  step {} t_inst {:.1} K  ({:.2}s elapsed)",
            frame.step,
            frame.instantaneous_temperature_k,
            t0.elapsed().as_secs_f64()
        );
    });
}
