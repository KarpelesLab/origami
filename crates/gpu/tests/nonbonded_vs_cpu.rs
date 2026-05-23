//! Validate the GPU LJ + reaction-field Coulomb kernel against an
//! inline CPU reference that uses the *same* exclusion mask the GPU
//! uses (1-2, 1-3, AND 1-4 all excluded) and the *same* parameters
//! (no CHARMM ε_14 / Rmin_14 specials).  Both restrictions are
//! limitations of the current GPU kernel; both will be lifted in a
//! follow-up that adds the 1-4 special LJ path to the shader.
//!
//! The production CPU path in `energy::forces_nonbonded` doesn't
//! match either restriction — it *includes* 1-4 pairs with the
//! CHARMM specials.  Test against the production path is deferred
//! until the GPU 1-4 path lands.

use chem::{classify_atom, standard_ff, AtomType};
use energy::DEFAULT_CUTOFF_A;
use geom::{build_extended_chain, build_topology_graph};
use gpu::{nonbonded_force_gpu, GpuContext, NonbondedSetup};

const KCAL_TO_KJ: f32 = 4.184;
const COULOMB_K_KJ: f32 = 1389.354_55; // 332.0637 × 4.184

fn cpu_nonbonded_reference(
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
            let inv_r2 = 1.0 / r2;
            let ratio = rmin / r;
            let r6 = (ratio * ratio).powi(3);
            let r12 = r6 * r6;
            let lj_coeff = 12.0 * eps * inv_r2 * (r6 - r12);
            let qq = qi * charges[j];
            let coul_coeff = -COULOMB_K_KJ * qq * (inv_r2 / r - inv_rc3);
            let coeff = lj_coeff + coul_coeff;
            out[i][0] += dx[0] * coeff;
            out[i][1] += dx[1] * coeff;
            out[i][2] += dx[2] * coeff;
        }
    }
    out
}

#[test]
fn gpu_nonbonded_matches_cpu_on_ala3() {
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
    ]).unwrap();
    let g = build_topology_graph(&s);
    let ff = standard_ff();

    let n = s.atom_count();
    // GPU input.
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n);
    let mut atom_types: Vec<AtomType> = Vec::with_capacity(n);
    let mut charges: Vec<f32> = Vec::with_capacity(n);
    for r in &s.residues {
        for a in &r.atoms {
            positions.push([a.position.x as f32, a.position.y as f32, a.position.z as f32]);
            let t = classify_atom(r.monomer, a.name).unwrap();
            atom_types.push(t);
            let q = ff.partial_charge_for(r.monomer, a.name).unwrap_or(0.0) as f32;
            charges.push(q);
        }
    }
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
            [(p.epsilon as f32) * KCAL_TO_KJ, p.rmin_half as f32]
        })
        .collect();
    // Exclude 1-2, 1-3, and 1-4 — matches the current GPU kernel
    // limitation (no 1-4 special params yet).  Once the GPU 1-4
    // path lands, this can drop the 1-4 exclusion and compare
    // against the production CPU `add_nonbonded_forces`.
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

    let gpu_forces = nonbonded_force_gpu(
        ctx,
        &positions,
        NonbondedSetup {
            type_index: &type_index,
            lj_params: &lj_params,
            charges: &charges,
            exclusions: &exclusions,
            cutoff_a: DEFAULT_CUTOFF_A as f32,
        },
    );

    // CPU reference matching the GPU's exclusions + parameter table.
    let cpu_forces = cpu_nonbonded_reference(
        &positions,
        &type_index,
        &lj_params,
        &charges,
        &exclusions,
        DEFAULT_CUTOFF_A as f32,
    );

    assert_eq!(gpu_forces.len(), cpu_forces.len());
    let mut max_err = 0.0_f32;
    let mut argmax_label = String::new();
    for (i, (g_f, c_f)) in gpu_forces.iter().zip(cpu_forces.iter()).enumerate() {
        for axis in 0..3 {
            let err = (g_f[axis] - c_f[axis]).abs();
            if err > max_err {
                max_err = err;
                argmax_label = format!(
                    "atom {i} axis {axis}: cpu={:.6} gpu={:.6} err={err:.6}",
                    c_f[axis], g_f[axis]
                );
            }
        }
    }
    eprintln!(
        "max GPU-vs-CPU (LJ+Coulomb) force discrepancy on Ala₃ ({} atoms): {} kJ/mol/Å",
        n, max_err
    );
    eprintln!("  {argmax_label}");
    assert!(max_err < 1e-3, "GPU and CPU nonbonded disagree: {argmax_label}");
}
