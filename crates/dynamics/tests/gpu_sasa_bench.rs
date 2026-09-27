//! Bench: GPU integrator with vs without SASA forces enabled.
//! Measures the per-step cost of the smooth-coverage SASA path
//! when wired into the full integrator.
//!
//! Run with:
//! ```text
//! cargo test --release -p dynamics --test gpu_sasa_bench -- --ignored --nocapture
//! ```

use std::time::Instant;

use chem::{AminoAcid, standard_ff};
use dynamics::full_gpu_integrator::FullGpuIntegrator;
use geom::{Vec3, build_extended_chain, build_topology_graph};
use gpu::GpuContext;

fn build_chain(n_residues: usize) -> geom::Structure {
    let block = [
        AminoAcid::Ala,
        AminoAcid::Gly,
        AminoAcid::Leu,
        AminoAcid::Glu,
        AminoAcid::Lys,
    ];
    let seq: Vec<AminoAcid> = block.iter().cloned().cycle().take(n_residues).collect();
    build_extended_chain(&seq).expect("build")
}

fn time_path(
    s: &geom::Structure,
    g: &geom::TopologyGraph,
    ff: &chem::ForceField,
    use_sasa: bool,
    warmup: usize,
    timed: usize,
) -> Result<f64, String> {
    let n = s.atom_count();
    let mut integ = FullGpuIntegrator::new(s, g, ff, 1.0, 2.0, 310.0, 1)
        .map_err(|e| format!("GPU init: {e}"))?;
    if use_sasa {
        let gammas = energy::powersasa::default_sasa_gammas(s);
        integ.enable_sasa_mode(s, &gammas);
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
fn bench_gpu_integrator_with_sasa() {
    if GpuContext::get().is_err() {
        eprintln!("GPU unavailable, skipping");
        return;
    }
    eprintln!("\n=== GPU integrator: no-SASA vs SASA-enabled, dt=1 fs ===");
    eprintln!();
    let warmup = 10;
    let timed = 50;
    for n_residues in [3usize, 20, 40, 100, 200] {
        let s = if n_residues == 3 {
            build_extended_chain(&[AminoAcid::Ala, AminoAcid::Lys, AminoAcid::Glu]).unwrap()
        } else {
            build_chain(n_residues)
        };
        let g = build_topology_graph(&s);
        let ff = standard_ff();
        let n = s.atom_count();
        let no_sasa = time_path(&s, &g, ff, false, warmup, timed).unwrap_or(f64::NAN);
        let sasa = time_path(&s, &g, ff, true, warmup, timed).unwrap_or(f64::NAN);
        eprintln!(
            "{n_residues:3} residues / {n:5} atoms  |  no SASA: {:6.3} ms/step  |  with SASA: {:7.3} ms/step  |  SASA overhead: +{:5.3} ms/step ({:.1}×)",
            no_sasa,
            sasa,
            sasa - no_sasa,
            sasa / no_sasa,
        );
    }
}
