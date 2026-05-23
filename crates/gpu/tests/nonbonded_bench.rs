//! Per-step timing of the stateful [`NonbondedPipeline`] vs the
//! one-shot `nonbonded_force_gpu` and a naive O(N²) CPU baseline.
//! The pipeline is constructed once (paid as overhead), then
//! `update_positions + compute` is repeated `iters` times — the
//! pattern the integrator will use.
//!
//! Run with `cargo test -p gpu --release --test nonbonded_bench --
//! --ignored --nocapture`.

use chem::{classify_atom, standard_ff, AminoAcid, AtomType};
use geom::{build_extended_chain, build_topology_graph};
use gpu::{nonbonded_force_gpu, GpuContext, NonbondedPipeline, NonbondedSetup};
use std::time::Instant;

const KCAL_TO_KJ: f32 = 4.184;
const COULOMB_K_KJ: f32 = 1389.354_55;
const CUTOFF_A: f32 = 10.0;

fn cpu_nonbonded(
    positions: &[[f32; 3]],
    type_index: &[u32],
    lj_params: &[[f32; 2]],
    charges: &[f32],
    exclusions: &[u32],
    cutoff: f32,
) -> Vec<[f32; 3]> {
    let n = positions.len();
    let mut out = vec![[0.0_f32; 3]; n];
    let cutoff_sq = cutoff * cutoff;
    let inv_rc3 = 1.0 / (cutoff * cutoff * cutoff);
    for i in 0..n {
        let pi = positions[i];
        let qi = charges[i];
        let (eps_i, rmin_half_i) = (lj_params[type_index[i] as usize][0], lj_params[type_index[i] as usize][1]);
        for j in 0..n {
            if i == j { continue; }
            let bit = i * n + j;
            if exclusions[bit / 32] & (1u32 << (bit % 32)) != 0 { continue; }
            let dx = [positions[j][0]-pi[0], positions[j][1]-pi[1], positions[j][2]-pi[2]];
            let r2 = dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2];
            if r2 > cutoff_sq || r2 < 1e-12 { continue; }
            let (eps_j, rmin_half_j) = (lj_params[type_index[j] as usize][0], lj_params[type_index[j] as usize][1]);
            let eps = (eps_i * eps_j).sqrt();
            let rmin = rmin_half_i + rmin_half_j;
            let r = r2.sqrt();
            let inv_r2 = 1.0 / r2;
            let ratio = rmin / r;
            let r6 = (ratio * ratio).powi(3);
            let r12 = r6 * r6;
            let lj_coeff = 12.0 * eps * inv_r2 * (r6 - r12);
            let qq = qi * charges[j];
            let coul_coeff = -COULOMB_K_KJ * qq * (inv_r2 / r - inv_rc3);
            let c = lj_coeff + coul_coeff;
            out[i][0] += dx[0] * c;
            out[i][1] += dx[1] * c;
            out[i][2] += dx[2] * c;
        }
    }
    out
}

#[test]
#[ignore]
fn bench_nonbonded_at_three_sizes() {
    let ctx = match GpuContext::get() {
        Ok(c) => c,
        Err(e) => { eprintln!("GPU unavailable: {e}"); return; }
    };

    let base: Vec<AminoAcid> = "NLYIQWLKDGGPSSGRPPPS"
        .chars().filter_map(AminoAcid::from_one_letter).collect();
    eprintln!("\n  N atoms | CPU (ms) | GPU one-shot (ms) | GPU pipeline (ms) | speedup vs CPU");
    eprintln!("  --------+----------+-------------------+-------------------+----------------");
    for repeats in [1, 3, 10] {
        let mut seq = Vec::with_capacity(base.len() * repeats);
        for _ in 0..repeats { seq.extend(base.iter().copied()); }
        let s = build_extended_chain(&seq).unwrap();
        let g = build_topology_graph(&s);
        let ff = standard_ff();
        let n = s.atom_count();
        let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n);
        let mut atom_types: Vec<AtomType> = Vec::with_capacity(n);
        let mut charges: Vec<f32> = Vec::with_capacity(n);
        for r in &s.residues {
            for a in &r.atoms {
                positions.push([a.position.x as f32, a.position.y as f32, a.position.z as f32]);
                atom_types.push(classify_atom(r.monomer, a.name).unwrap());
                charges.push(ff.partial_charge_for(r.monomer, a.name).unwrap_or(0.0) as f32);
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
        let lj_params_14: Vec<[f32; 2]> = unique.iter().map(|t| {
            let p = ff.nonbonded(*t).unwrap();
            let eps_14 = p.epsilon_14.unwrap_or(p.epsilon);
            let rmin_half_14 = p.rmin_half_14.unwrap_or(p.rmin_half);
            [(eps_14 as f32) * KCAL_TO_KJ, rmin_half_14 as f32]
        }).collect();
        let mut exclusions = vec![0u32; (n * n).div_ceil(32)];
        let mut one_four = vec![0u32; (n * n).div_ceil(32)];
        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                let bit = i * n + j;
                if g.is_bonded(i, j) || g.is_one_three(i, j) {
                    exclusions[bit / 32] |= 1u32 << (bit % 32);
                } else if g.is_one_four(i, j) {
                    one_four[bit / 32] |= 1u32 << (bit % 32);
                }
            }
        }
        let setup = || NonbondedSetup {
            type_index: &type_index,
            lj_params: &lj_params,
            lj_params_14: &lj_params_14,
            charges: &charges,
            exclusions: &exclusions,
            one_four_mask: &one_four,
            cutoff_a: CUTOFF_A,
        };

        // Warm up.
        let _ = nonbonded_force_gpu(ctx, &positions, setup());
        let _ = cpu_nonbonded(&positions, &type_index, &lj_params, &charges, &exclusions, CUTOFF_A);

        let iters = if n < 500 { 100 } else { 20 };

        // CPU baseline.
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = cpu_nonbonded(&positions, &type_index, &lj_params, &charges, &exclusions, CUTOFF_A);
        }
        let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // GPU one-shot — pays setup every iteration.
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = nonbonded_force_gpu(ctx, &positions, setup());
        }
        let gpu_oneshot_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        // GPU stateful pipeline — setup once, dispatch many.
        let mut pipe = NonbondedPipeline::new(ctx, n, setup());
        let t0 = Instant::now();
        for _ in 0..iters {
            pipe.update_positions(&positions);
            let _ = pipe.compute();
        }
        let gpu_pipe_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        let speedup = cpu_ms / gpu_pipe_ms;
        eprintln!(
            "  {:>7} | {:>8.2} | {:>17.2} | {:>17.2} | {:.2}×",
            n, cpu_ms, gpu_oneshot_ms, gpu_pipe_ms, speedup
        );
    }
}
