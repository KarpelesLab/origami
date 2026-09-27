//! Bench GPU dot-density SASA against the CPU implementations.
//!
//! Run with:
//!
//! ```text
//! cargo test --release -p gpu --test sasa_bench -- --ignored --nocapture
//! ```
//!
//! Compares:
//!   - GPU dot-density (256 dots) — this commit's new path
//!   - CPU dot-density (256 dots) — `energy::sasa::sasa_per_atom_with_dots`
//!   - CPU analytical exact-SASA — `energy::powersasa::powersasa_energy`

use std::time::Instant;

use chem::{AminoAcid, Element, standard_ff};
use energy::powersasa::powersasa_energy;
use energy::sasa::sasa_per_atom_with_dots;
use geom::{Vec3, build_extended_chain};
use gpu::{GpuContext, SASA_N_DOTS, SasaPipeline, SasaSetup};

const PROBE_RADIUS_A: f64 = 1.4;

fn vdw_radius(e: Element) -> f64 {
    match e {
        Element::H => 1.20,
        Element::C => 1.70,
        Element::N => 1.55,
        Element::O => 1.50,
        Element::P => 1.80,
        Element::S => 1.80,
    }
}

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

fn build_csr(positions_f64: &[Vec3], radii_f64: &[f64]) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let n = positions_f64.len();
    let mut counts = vec![0u32; n];
    let mut per_atom_nbrs: Vec<Vec<u32>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = (positions_f64[i] - positions_f64[j]).norm();
            if d <= radii_f64[i] + radii_f64[j] {
                per_atom_nbrs[i].push(j as u32);
            }
        }
    }
    let mut starts = vec![0u32; n];
    let mut total = 0u32;
    for i in 0..n {
        starts[i] = total;
        counts[i] = per_atom_nbrs[i].len() as u32;
        total += counts[i];
    }
    let mut indices_flat: Vec<u32> = Vec::with_capacity(total as usize);
    for v in &per_atom_nbrs {
        indices_flat.extend_from_slice(v);
    }
    (counts, starts, indices_flat)
}

fn time_n(closure: &mut dyn FnMut(), n_reps: usize) -> f64 {
    // One warmup.
    closure();
    let t0 = Instant::now();
    for _ in 0..n_reps {
        closure();
    }
    t0.elapsed().as_secs_f64() * 1000.0 / n_reps as f64
}

#[test]
#[ignore]
fn bench_gpu_sasa_vs_cpu() {
    let ctx = match GpuContext::get() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("GPU unavailable: {e}");
            return;
        }
    };
    eprintln!("\n=== GPU dot-density SASA bench (N_DOTS = {SASA_N_DOTS}) ===");
    eprintln!();
    for n_residues in [3usize, 20, 40, 100, 200, 400] {
        let s = if n_residues == 3 {
            build_extended_chain(&[AminoAcid::Ala, AminoAcid::Lys, AminoAcid::Glu]).unwrap()
        } else {
            build_chain(n_residues)
        };
        let n = s.atom_count();
        let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n);
        let mut positions_f64: Vec<Vec3> = Vec::with_capacity(n);
        let mut radii_f32: Vec<f32> = Vec::with_capacity(n);
        let mut radii_f64: Vec<f64> = Vec::with_capacity(n);
        for r in &s.residues {
            for a in &r.atoms {
                positions.push([
                    a.position.x as f32,
                    a.position.y as f32,
                    a.position.z as f32,
                ]);
                positions_f64.push(a.position);
                let exp = vdw_radius(a.element) + PROBE_RADIUS_A;
                radii_f32.push(exp as f32);
                radii_f64.push(exp);
            }
        }
        let (counts, starts, indices) = build_csr(&positions_f64, &radii_f64);

        // GPU bench.
        let mut pipe = SasaPipeline::new(
            ctx,
            n,
            SasaSetup {
                radii: &radii_f32,
                initial_indices_capacity: indices.len().max(64),
            },
        );
        pipe.update_positions(&positions);
        pipe.update_neighbours(&counts, &starts, &indices);
        let gpu_ms = time_n(
            &mut || {
                let _ = pipe.compute_area();
            },
            10,
        );

        // CPU dot-density bench.
        let cpu_dot_ms = time_n(
            &mut || {
                let _ = sasa_per_atom_with_dots(&s, SASA_N_DOTS);
            },
            5,
        );

        // CPU analytical bench (also produces a per-atom area, but
        // via the exact topology — much more accurate, generally
        // slower per call).
        let ff = standard_ff();
        let cpu_ana_ms = time_n(
            &mut || {
                let _ = powersasa_energy(&s, ff);
            },
            3,
        );

        eprintln!(
            "{n_residues:3} residues / {n:5} atoms  |  GPU dot: {:7.3} ms  |  CPU dot: {:7.3} ms  |  CPU analytical: {:7.3} ms  |  GPU vs CPU dot: {:.1}×  |  GPU vs CPU analytical: {:.1}×",
            gpu_ms,
            cpu_dot_ms,
            cpu_ana_ms,
            cpu_dot_ms / gpu_ms,
            cpu_ana_ms / gpu_ms,
        );
    }
}
