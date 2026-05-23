//! Rough CPU-vs-GPU LJ benchmark.  Runs over a built protein chain
//! that's large enough to be meaningful (Trp-cage extended at 300
//! atoms, then a synthetic 1000- and 3000-atom chain by repeating
//! the sequence).  Reports wall-time and a per-step extrapolation.
//!
//! Run with `cargo test -p gpu --release --test lj_bench -- --ignored
//! --nocapture`.  Ignored because timing tests are flaky in CI.

use chem::{classify_atom, standard_ff, AminoAcid, AtomType};
use geom::{build_extended_chain, build_topology_graph};
use gpu::{lj::lj_force_gpu, lj::LjInput, GpuContext};
use std::time::Instant;

const KCAL_TO_KJ: f32 = 4.184;
const CUTOFF_A: f32 = 10.0;

fn cpu_lj_forces(
    positions: &[[f32; 3]],
    type_index: &[u32],
    lj_params: &[[f32; 2]],
    exclusions: &[u32],
    cutoff: f32,
) -> Vec<[f32; 3]> {
    let n = positions.len();
    let mut out = vec![[0.0_f32; 3]; n];
    let cutoff_sq = cutoff * cutoff;
    for i in 0..n {
        let pi = positions[i];
        let (eps_i, rmin_half_i) = (lj_params[type_index[i] as usize][0], lj_params[type_index[i] as usize][1]);
        for j in 0..n {
            if i == j { continue; }
            let bit = i * n + j;
            if exclusions[bit / 32] & (1u32 << (bit % 32)) != 0 {
                continue;
            }
            let dx = [
                positions[j][0] - pi[0],
                positions[j][1] - pi[1],
                positions[j][2] - pi[2],
            ];
            let r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
            if r2 > cutoff_sq || r2 < 1e-12 {
                continue;
            }
            let (eps_j, rmin_half_j) = (lj_params[type_index[j] as usize][0], lj_params[type_index[j] as usize][1]);
            let eps = (eps_i * eps_j).sqrt();
            let rmin = rmin_half_i + rmin_half_j;
            let r = r2.sqrt();
            let ratio = rmin / r;
            let r6 = (ratio * ratio).powi(3);
            let r12 = r6 * r6;
            let coeff = 12.0 * eps / r2 * (r6 - r12);
            out[i][0] += dx[0] * coeff;
            out[i][1] += dx[1] * coeff;
            out[i][2] += dx[2] * coeff;
        }
    }
    out
}

#[test]
#[ignore]
fn bench_lj_at_three_sizes() {
    let ctx = match GpuContext::get() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("GPU unavailable, skipping: {e}");
            return;
        }
    };

    // Trp-cage sequence repeated 1, 3, 10 times → 300, 900, 3000 atoms.
    let base: Vec<AminoAcid> = "NLYIQWLKDGGPSSGRPPPS"
        .chars()
        .filter_map(AminoAcid::from_one_letter)
        .collect();
    eprintln!("\n  N atoms | CPU (ms) | GPU (ms) | speedup");
    eprintln!("  --------+----------+----------+--------");
    for repeats in [1, 3, 10] {
        let mut seq = Vec::with_capacity(base.len() * repeats);
        for _ in 0..repeats { seq.extend(base.iter().copied()); }
        let s = build_extended_chain(&seq).unwrap();
        let g = build_topology_graph(&s);
        let ff = standard_ff();
        let n = s.atom_count();
        let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n);
        let mut atom_types: Vec<AtomType> = Vec::with_capacity(n);
        for r in &s.residues {
            for a in &r.atoms {
                positions.push([a.position.x as f32, a.position.y as f32, a.position.z as f32]);
                atom_types.push(classify_atom(r.monomer, a.name).unwrap());
            }
        }
        let mut unique = atom_types.clone();
        unique.sort();
        unique.dedup();
        let type_index: Vec<u32> = atom_types.iter().map(|t| unique.iter().position(|x| x == t).unwrap() as u32).collect();
        let lj_params: Vec<[f32; 2]> = unique.iter().map(|t| {
            let p = ff.nonbonded(*t).unwrap();
            [(p.epsilon as f32) * KCAL_TO_KJ, p.rmin_half as f32]
        }).collect();
        let mut exclusions = vec![0u32; (n * n).div_ceil(32)];
        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                if g.is_bonded(i, j) || g.is_one_three(i, j) || g.is_one_four(i, j) {
                    let bit = i * n + j;
                    exclusions[bit / 32] |= 1u32 << (bit % 32);
                }
            }
        }
        // Warm up.
        let _ = lj_force_gpu(ctx, LjInput { positions: &positions, type_index: &type_index, lj_params: &lj_params, exclusions: &exclusions, cutoff_a: CUTOFF_A });
        let _ = cpu_lj_forces(&positions, &type_index, &lj_params, &exclusions, CUTOFF_A);
        // Time.
        let iters = if n < 500 { 50 } else { 10 };
        let t0 = Instant::now();
        for _ in 0..iters { let _ = cpu_lj_forces(&positions, &type_index, &lj_params, &exclusions, CUTOFF_A); }
        let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        let t0 = Instant::now();
        for _ in 0..iters { let _ = lj_force_gpu(ctx, LjInput { positions: &positions, type_index: &type_index, lj_params: &lj_params, exclusions: &exclusions, cutoff_a: CUTOFF_A }); }
        let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        let speedup = cpu_ms / gpu_ms;
        eprintln!("  {:>7} | {:>8.2} | {:>8.2} | {:.2}×", n, cpu_ms, gpu_ms, speedup);
    }
}
