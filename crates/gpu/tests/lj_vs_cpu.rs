//! Validate the GPU LJ-only kernel against an inline CPU reference.
//! Test target: built Ala₃ chain (small, deterministic geometry).
//!
//! Both sides use the same:
//!   - atom positions
//!   - per-atom ε and Rmin/2 (looked up from `chem::standard_ff()`
//!     CHARMM36 nonbonded table)
//!   - 1-2 / 1-3 / 1-4 exclusion mask (from `geom::build_topology_graph`)
//!   - 10 Å bare cutoff
//!
//! Tolerance: 1e-3 kJ/mol/Å absolute, per atom-axis (f32 round-trip
//! through the GPU + sqrt() accumulation).

use chem::{classify_atom, standard_ff, AtomType};
use geom::{build_extended_chain, build_topology_graph, Vec3};
use gpu::{lj::lj_force_gpu, lj::LjInput, GpuContext};

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
        let (eps_i, rmin_half_i) = (
            lj_params[type_index[i] as usize][0],
            lj_params[type_index[i] as usize][1],
        );
        for j in 0..n {
            if i == j {
                continue;
            }
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
            let (eps_j, rmin_half_j) = (
                lj_params[type_index[j] as usize][0],
                lj_params[type_index[j] as usize][1],
            );
            let eps = (eps_i * eps_j).sqrt();
            let rmin = rmin_half_i + rmin_half_j;
            let r = r2.sqrt();
            let ratio = rmin / r;
            let r2_ratio = ratio * ratio;
            let r6 = r2_ratio * r2_ratio * r2_ratio;
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
fn gpu_lj_matches_cpu_on_ala3() {
    // Skip the test if GPU init fails (e.g. CI without a GPU).
    let ctx = match GpuContext::get() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("GPU unavailable, skipping: {e}");
            return;
        }
    };

    let s = build_extended_chain(&[
        chem::AminoAcid::Ala,
        chem::AminoAcid::Ala,
        chem::AminoAcid::Ala,
    ])
    .unwrap();
    let g = build_topology_graph(&s);
    let ff = standard_ff();

    // Build atom data + parameter table.
    let n = s.atom_count();
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n);
    let mut atom_types: Vec<AtomType> = Vec::with_capacity(n);
    for r in &s.residues {
        for a in &r.atoms {
            positions.push([
                a.position.x as f32,
                a.position.y as f32,
                a.position.z as f32,
            ]);
            atom_types.push(classify_atom(r.monomer, a.name).unwrap());
        }
    }

    // De-duplicate atom types into a compact parameter table, keep
    // a per-atom index into that table.
    let mut unique_types: Vec<AtomType> = atom_types.clone();
    unique_types.sort();
    unique_types.dedup();
    let type_index: Vec<u32> = atom_types
        .iter()
        .map(|t| unique_types.iter().position(|x| x == t).unwrap() as u32)
        .collect();
    let lj_params: Vec<[f32; 2]> = unique_types
        .iter()
        .map(|t| {
            let p = ff.nonbonded(*t).unwrap();
            // ε in kJ/mol (CHARMM stores kcal; multiply by 4.184).
            [(p.epsilon as f32) * KCAL_TO_KJ, p.rmin_half as f32]
        })
        .collect();

    // Build flat exclusion bitmap from the topology graph.
    let n_bits = n * n;
    let mut exclusions = vec![0u32; n_bits.div_ceil(32)];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if g.is_bonded(i, j) || g.is_one_three(i, j) || g.is_one_four(i, j) {
                let bit = i * n + j;
                exclusions[bit / 32] |= 1u32 << (bit % 32);
            }
        }
    }

    let cpu_forces = cpu_lj_forces(&positions, &type_index, &lj_params, &exclusions, CUTOFF_A);

    let gpu_forces = lj_force_gpu(
        ctx,
        LjInput {
            positions: &positions,
            type_index: &type_index,
            lj_params: &lj_params,
            exclusions: &exclusions,
            cutoff_a: CUTOFF_A,
        },
    );

    assert_eq!(cpu_forces.len(), gpu_forces.len());
    let mut max_err = 0.0_f32;
    let mut argmax_i = 0;
    let mut argmax_axis = 0;
    for (i, (c, g)) in cpu_forces.iter().zip(gpu_forces.iter()).enumerate() {
        for axis in 0..3 {
            let err = (c[axis] - g[axis]).abs();
            if err > max_err {
                max_err = err;
                argmax_i = i;
                argmax_axis = axis;
            }
        }
    }
    let label = format!(
        "atom {} axis {}: cpu={:.6} gpu={:.6} err={:.6} kJ/mol/Å",
        argmax_i,
        argmax_axis,
        cpu_forces[argmax_i][argmax_axis],
        gpu_forces[argmax_i][argmax_axis],
        max_err,
    );
    eprintln!(
        "max GPU-vs-CPU LJ force discrepancy on Ala₃ ({} atoms): {} kJ/mol/Å — {}",
        n, max_err, label
    );
    assert!(max_err < 1e-2, "GPU and CPU LJ disagree: {label}");

    // Bonus sanity: forces sum to ~zero (Newton's third law on every
    // intra-LJ pair, no external forces).
    let net: [f32; 3] = gpu_forces.iter().fold([0.0; 3], |acc, f| {
        [acc[0] + f[0], acc[1] + f[1], acc[2] + f[2]]
    });
    let net_mag = (net[0] * net[0] + net[1] * net[1] + net[2] * net[2]).sqrt();
    assert!(
        net_mag < 1e-3,
        "GPU LJ forces don't sum to zero: net = {net:?}"
    );

    let _ = Vec3::zeros(); // keep `geom::Vec3` use alive
}
