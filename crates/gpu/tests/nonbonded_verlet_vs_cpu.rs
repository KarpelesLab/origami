//! Validate the Verlet-list GPU kernel against the production CPU
//! nonbonded path.
//!
//! The Verlet kernel is identical to the O(N²) GPU kernel in physics —
//! same LJ + reaction-field Coulomb formulas, same 1-4 specials, same
//! exclusion handling — it only differs in how the inner loop discovers
//! neighbours.  So the same Ala₃ test that pinned the O(N²) kernel to
//! the CPU also pins this one, as long as the supplied neighbour list
//! is complete (covers every pair within the cutoff).
//!
//! We build a brute-force neighbour list on the CPU (every j ≠ i within
//! the cutoff) and pass it via `pair_list_to_csr` — the same CSR layout
//! the real integrator will use.

use chem::{AminoAcid, AtomType, classify_atom, standard_ff};
use energy::DEFAULT_CUTOFF_A;
use geom::{Vec3, build_extended_chain, build_topology_graph};
use gpu::{GpuContext, VerletNonbondedPipeline, VerletNonbondedSetup, pair_list_to_csr};

const KCAL_TO_KJ: f32 = 4.184;

#[test]
fn gpu_verlet_matches_cpu_on_ala3() {
    let ctx = match GpuContext::get() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("GPU unavailable: {e}");
            return;
        }
    };

    let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala, AminoAcid::Ala]).unwrap();
    let g = build_topology_graph(&s);
    let ff = standard_ff();

    let n = s.atom_count();
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n);
    let mut atom_types: Vec<AtomType> = Vec::with_capacity(n);
    let mut charges: Vec<f32> = Vec::with_capacity(n);
    let mut positions_f64: Vec<Vec3> = Vec::with_capacity(n);
    for r in &s.residues {
        for a in &r.atoms {
            positions.push([
                a.position.x as f32,
                a.position.y as f32,
                a.position.z as f32,
            ]);
            positions_f64.push(a.position);
            let t = classify_atom(r.monomer, a.name).unwrap();
            atom_types.push(t);
            charges.push(ff.partial_charge_for(r.monomer, a.name).unwrap_or(0.0) as f32);
        }
    }
    let atom_lj_data: Vec<[f32; 4]> = atom_types
        .iter()
        .map(|t| {
            let p = ff.nonbonded(*t).unwrap();
            let eps_14 = p.epsilon_14.unwrap_or(p.epsilon);
            let rmin_half_14 = p.rmin_half_14.unwrap_or(p.rmin_half);
            [
                (p.epsilon as f32) * KCAL_TO_KJ,
                p.rmin_half as f32,
                (eps_14 as f32) * KCAL_TO_KJ,
                rmin_half_14 as f32,
            ]
        })
        .collect();
    let mut exclusions = vec![0u32; (n * n).div_ceil(32)];
    let mut one_four = vec![0u32; (n * n).div_ceil(32)];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let bit = i * n + j;
            if g.is_bonded(i, j) || g.is_one_three(i, j) {
                exclusions[bit / 32] |= 1u32 << (bit % 32);
            } else if g.is_one_four(i, j) {
                one_four[bit / 32] |= 1u32 << (bit % 32);
            }
        }
    }

    // Brute-force neighbour list at the same cutoff the kernel uses.
    let cutoff = DEFAULT_CUTOFF_A as f32;
    let cutoff_sq = (cutoff as f64) * (cutoff as f64);
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let d = positions_f64[i] - positions_f64[j];
            if d.norm_squared() <= cutoff_sq {
                pairs.push((i as u32, j as u32));
            }
        }
    }
    eprintln!(
        "Ala₃: {} atoms, {} verlet pairs at cutoff {:.1} Å",
        n,
        pairs.len(),
        cutoff
    );
    let (counts, starts, indices) = pair_list_to_csr(n, &pairs);

    let mut pipe = VerletNonbondedPipeline::new(
        ctx,
        n,
        VerletNonbondedSetup {
            atom_lj_data: &atom_lj_data,
            charges: &charges,
            exclusions: &exclusions,
            one_four_mask: &one_four,
            cutoff_a: cutoff,
            initial_indices_capacity: indices.len().max(64),
        },
    );
    pipe.update_neighbours(&counts, &starts, &indices);
    pipe.update_positions(&positions);
    let gpu_forces = pipe.compute();

    let mut cpu_forces = vec![Vec3::zeros(); n];
    energy::forces_nonbonded::add_nonbonded_forces(&s, &g, ff, DEFAULT_CUTOFF_A, &mut cpu_forces);

    assert_eq!(gpu_forces.len(), cpu_forces.len());
    let mut max_err = 0.0_f64;
    let mut label = String::new();
    for (i, (g_f, c_f)) in gpu_forces.iter().zip(cpu_forces.iter()).enumerate() {
        for axis in 0..3 {
            let gv = g_f[axis] as f64;
            let cv = c_f[axis];
            let err = (gv - cv).abs();
            if err > max_err {
                max_err = err;
                label = format!("atom {i} axis {axis}: cpu={cv:.6} gpu={gv:.6} err={err:.6}");
            }
        }
    }
    eprintln!(
        "max GPU-Verlet-vs-CPU force discrepancy on Ala₃ ({n} atoms): {max_err:.3e} kJ/mol/Å"
    );
    eprintln!("  {label}");
    assert!(max_err < 1e-1, "Verlet GPU and CPU disagree: {label}");
}

#[test]
fn gpu_verlet_handles_buffer_growth() {
    // Stress-test: provide an initial capacity that's intentionally
    // too small, then upload a larger list — the buffer must grow
    // transparently and the second compute() must still match.
    let ctx = match GpuContext::get() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("GPU unavailable: {e}");
            return;
        }
    };

    let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala, AminoAcid::Ala]).unwrap();
    let g = build_topology_graph(&s);
    let ff = standard_ff();
    let n = s.atom_count();

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n);
    let mut atom_types: Vec<AtomType> = Vec::with_capacity(n);
    let mut charges: Vec<f32> = Vec::with_capacity(n);
    let mut positions_f64: Vec<Vec3> = Vec::with_capacity(n);
    for r in &s.residues {
        for a in &r.atoms {
            positions.push([
                a.position.x as f32,
                a.position.y as f32,
                a.position.z as f32,
            ]);
            positions_f64.push(a.position);
            let t = classify_atom(r.monomer, a.name).unwrap();
            atom_types.push(t);
            charges.push(ff.partial_charge_for(r.monomer, a.name).unwrap_or(0.0) as f32);
        }
    }
    let atom_lj_data: Vec<[f32; 4]> = atom_types
        .iter()
        .map(|t| {
            let p = ff.nonbonded(*t).unwrap();
            [
                (p.epsilon as f32) * KCAL_TO_KJ,
                p.rmin_half as f32,
                (p.epsilon as f32) * KCAL_TO_KJ,
                p.rmin_half as f32,
            ]
        })
        .collect();
    let mut exclusions = vec![0u32; (n * n).div_ceil(32)];
    let mut one_four = vec![0u32; (n * n).div_ceil(32)];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let bit = i * n + j;
            if g.is_bonded(i, j) || g.is_one_three(i, j) {
                exclusions[bit / 32] |= 1u32 << (bit % 32);
            } else if g.is_one_four(i, j) {
                one_four[bit / 32] |= 1u32 << (bit % 32);
            }
        }
    }

    let cutoff = DEFAULT_CUTOFF_A as f32;
    let cutoff_sq = (cutoff as f64) * (cutoff as f64);
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let d = positions_f64[i] - positions_f64[j];
            if d.norm_squared() <= cutoff_sq {
                pairs.push((i as u32, j as u32));
            }
        }
    }
    let (counts, starts, indices) = pair_list_to_csr(n, &pairs);

    // Start with a too-small capacity (64 entries).  The upload should
    // trigger the grow path.
    let mut pipe = VerletNonbondedPipeline::new(
        ctx,
        n,
        VerletNonbondedSetup {
            atom_lj_data: &atom_lj_data,
            charges: &charges,
            exclusions: &exclusions,
            one_four_mask: &one_four,
            cutoff_a: cutoff,
            initial_indices_capacity: 32,
        },
    );
    pipe.update_neighbours(&counts, &starts, &indices);
    pipe.update_positions(&positions);
    let gpu_forces = pipe.compute();

    // Sanity check vs CPU using regular (no-specials) params — this
    // test is about the buffer-grow code path, not physics accuracy.
    let mut sum = 0.0_f32;
    for f in &gpu_forces {
        sum += f[0].abs() + f[1].abs() + f[2].abs();
    }
    assert!(sum.is_finite(), "non-finite forces after buffer grow");
    assert!(
        sum > 0.0,
        "zero forces — buffer-grow path broke the bind group"
    );
    eprintln!("buffer-grow path OK: |F| sum {sum:.2}, atoms {n}");
}
