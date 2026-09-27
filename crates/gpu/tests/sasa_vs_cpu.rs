//! Validate the GPU dot-density SASA kernel against the CPU
//! `sasa_per_atom_with_dots` (same algorithm, same dot count, same
//! Fibonacci pattern → must agree to f32 round-trip noise).

use chem::{standard_ff, AminoAcid, Element};
use energy::sasa::sasa_per_atom_with_dots;
use geom::{build_extended_chain, Vec3};
use gpu::{GpuContext, SasaPipeline, SasaSetup, SASA_N_DOTS};

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

#[test]
fn gpu_sasa_matches_cpu_on_ala_lys_glu() {
    let ctx = match GpuContext::get() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("GPU unavailable: {e}");
            return;
        }
    };
    let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Lys, AminoAcid::Glu]).unwrap();
    let n = s.atom_count();

    // Per-atom expanded radii.
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

    // Build the SASA neighbour list (every j whose expanded sphere
    // overlaps i's).  No skin since we're not running dynamics —
    // just a one-shot SASA evaluation.
    let mut counts = vec![0u32; n];
    let mut starts = vec![0u32; n];
    let mut indices_flat: Vec<u32> = Vec::new();
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
    let mut total = 0u32;
    for i in 0..n {
        starts[i] = total;
        counts[i] = per_atom_nbrs[i].len() as u32;
        total += counts[i];
    }
    indices_flat.reserve(total as usize);
    for v in &per_atom_nbrs {
        indices_flat.extend_from_slice(v);
    }

    let mut pipe = SasaPipeline::new(
        ctx,
        n,
        SasaSetup {
            radii: &radii_f32,
            initial_indices_capacity: indices_flat.len().max(64),
        },
    );
    pipe.update_positions(&positions);
    pipe.update_neighbours(&counts, &starts, &indices_flat);
    let gpu_areas = pipe.compute_area();

    // CPU reference at the same N_dots.
    let cpu_areas = sasa_per_atom_with_dots(&s, SASA_N_DOTS);

    assert_eq!(gpu_areas.len(), cpu_areas.len());
    let mut max_err = 0.0_f64;
    let mut argmax = String::new();
    for i in 0..n {
        let g = gpu_areas[i] as f64;
        let c = cpu_areas[i];
        let err = (g - c).abs();
        if err > max_err {
            max_err = err;
            argmax = format!("atom {i}: cpu={c:.3} Å²  gpu={g:.3} Å²  err={err:.4}");
        }
    }
    eprintln!(
        "max GPU-vs-CPU per-atom area discrepancy on Ala-Lys-Glu ({n} atoms, N_DOTS={SASA_N_DOTS}): \
         {max_err:.3e} Å²  ({argmax})"
    );
    // Per-atom tolerance: with 256 dots and the dot-density method,
    // each dot represents ~0.4 Å² (depending on atom radius).  f32
    // precision at the per-atom-pair boundary occasionally flips a
    // single dot's accessibility verdict — 1-3 dots per atom can
    // legitimately differ between f32 and f64 runs.  Allow up to
    // 3 % relative error per atom (≈5 dots out of 256) or 2 Å²
    // absolute, whichever is larger.
    let cpu_at_argmax = cpu_areas.iter().cloned().fold(0.0_f64, f64::max);
    let rel_tol = 0.03 * cpu_at_argmax.max(2.0).max(1.0);
    assert!(
        max_err < rel_tol.max(2.0),
        "GPU SASA area diverges from CPU past tolerance ({argmax})"
    );

    // Totals — same precision argument applies, but averaged over N
    // atoms the noise should cancel substantially.
    let gpu_total: f64 = gpu_areas.iter().map(|&a| a as f64).sum();
    let cpu_total: f64 = cpu_areas.iter().sum();
    let total_err = (gpu_total - cpu_total).abs();
    let total_rel = total_err / cpu_total;
    eprintln!(
        "total area — CPU: {cpu_total:.2} Å²  GPU: {gpu_total:.2} Å²  rel err {:.3}%",
        100.0 * total_rel
    );
    assert!(
        total_rel < 0.02,
        "total area differs by > 2 %: {cpu_total} vs {gpu_total}"
    );
}
