//! Validate the GPU LJ + reaction-field Coulomb kernel against the
//! production CPU AoS path in `energy::forces_nonbonded::add_nonbonded_forces`.
//!
//! Now that the GPU kernel supports CHARMM `ε_14 / Rmin/2_14` specials
//! on 1-4 pairs, the comparison is direct: same exclusion convention
//! (only 1-2 and 1-3 excluded; 1-4 pairs counted with specials), same
//! parameter table.  Production-CPU vs GPU agreement.

use chem::{AtomType, classify_atom, standard_ff};
use energy::DEFAULT_CUTOFF_A;
use geom::{Vec3, build_extended_chain, build_topology_graph};
use gpu::{GpuContext, NonbondedSetup, nonbonded_force_gpu};

const KCAL_TO_KJ: f32 = 4.184;

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
    ])
    .unwrap();
    let g = build_topology_graph(&s);
    let ff = standard_ff();

    let n = s.atom_count();
    // GPU input.
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n);
    let mut atom_types: Vec<AtomType> = Vec::with_capacity(n);
    let mut charges: Vec<f32> = Vec::with_capacity(n);
    for r in &s.residues {
        for a in &r.atoms {
            positions.push([
                a.position.x as f32,
                a.position.y as f32,
                a.position.z as f32,
            ]);
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
    let lj_params_14: Vec<[f32; 2]> = unique_types
        .iter()
        .map(|t| {
            let p = ff.nonbonded(*t).unwrap();
            // CHARMM convention: ε_14 / Rmin/2_14 fall back to the
            // regular values if no specials are present.  Match the
            // CPU code in `forces_nonbonded::add_nonbonded_forces`.
            let eps_14 = p.epsilon_14.unwrap_or(p.epsilon);
            let rmin_half_14 = p.rmin_half_14.unwrap_or(p.rmin_half);
            [(eps_14 as f32) * KCAL_TO_KJ, rmin_half_14 as f32]
        })
        .collect();
    // Exclude 1-2 and 1-3; 1-4 pairs contribute with specials.
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

    let gpu_forces = nonbonded_force_gpu(
        ctx,
        &positions,
        NonbondedSetup {
            type_index: &type_index,
            lj_params: &lj_params,
            lj_params_14: &lj_params_14,
            charges: &charges,
            exclusions: &exclusions,
            one_four_mask: &one_four,
            cutoff_a: DEFAULT_CUTOFF_A as f32,
        },
    );

    // Production-CPU reference.
    let mut cpu_forces_v3 = vec![Vec3::zeros(); n];
    energy::forces_nonbonded::add_nonbonded_forces(
        &s,
        &g,
        ff,
        DEFAULT_CUTOFF_A,
        &mut cpu_forces_v3,
    );

    assert_eq!(gpu_forces.len(), cpu_forces_v3.len());
    let mut max_err = 0.0_f64;
    let mut argmax_label = String::new();
    for (i, (g_f, c_f)) in gpu_forces.iter().zip(cpu_forces_v3.iter()).enumerate() {
        for axis in 0..3 {
            let gv = g_f[axis] as f64;
            let cv = c_f[axis];
            let err = (gv - cv).abs();
            if err > max_err {
                max_err = err;
                argmax_label =
                    format!("atom {i} axis {axis}: cpu={cv:.6} gpu={gv:.6} err={err:.6}");
            }
        }
    }
    eprintln!(
        "max GPU-vs-CPU (LJ+Coulomb, with 1-4 specials) force discrepancy on Ala₃ ({} atoms): {:.3e} kJ/mol/Å",
        n, max_err
    );
    eprintln!("  {argmax_label}");
    // Loose tolerance: f32 round-trip + Coulomb's 1/r² gradient
    // sensitivity at close pairs accumulates.
    assert!(
        max_err < 1e-1,
        "GPU and CPU nonbonded disagree: {argmax_label}"
    );
}
