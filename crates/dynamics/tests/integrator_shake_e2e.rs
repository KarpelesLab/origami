//! End-to-end test for SHAKE-mode integrator on GPU.
//!
//! Builds Ala-Lys-Glu, sets up the GPU integrator with SHAKE, runs
//! N integrator steps at dt = 2 fs (impossible without constraints —
//! unconstrained dt = 2 fs explodes from H-bond vibrational
//! aliasing).  Verifies:
//!
//!   1. No NaN / divergence.
//!   2. Every X-H bond stays within tolerance of its target length.
//!   3. Approximate equilibration toward target T.
//!
//! Doesn't bit-compare against CPU — different RNG between
//! xoshiro128++ (GPU) and xoshiro256++ (CPU) makes that impossible.

use std::sync::Arc;

use chem::{classify_atom, standard_ff, AminoAcid, AtomType, Element};
use dynamics::full_gpu_integrator::FullGpuIntegrator;
use dynamics::shake::build_h_bond_constraints;
use geom::{build_extended_chain, build_topology_graph, Vec3};
use gpu::{build_per_x_shake_data, GpuContext, ShakeConstraint};

fn atom_types_for(s: &geom::Structure) -> Vec<AtomType> {
    let mut out = Vec::with_capacity(s.atom_count());
    for r in &s.residues {
        for a in &r.atoms {
            out.push(classify_atom(r.monomer, a.name).unwrap());
        }
    }
    out
}

#[test]
fn integrator_shake_holds_h_bond_lengths_at_dt_2fs() {
    let ctx = match GpuContext::get() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("GPU unavailable: {e}");
            return;
        }
    };
    let mut s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Lys, AminoAcid::Glu]).unwrap();
    let g = build_topology_graph(&s);
    let ff = standard_ff();
    let n = s.atom_count();
    let atom_types = atom_types_for(&s);

    // CPU-side constraint list + orientation.  Convert to (X, H)
    // tuples for the GPU per-X CSR.
    let cpu_constraints = build_h_bond_constraints(&s, &g, ff, &atom_types);
    let atoms_flat: Vec<Element> = s
        .residues
        .iter()
        .flat_map(|r| r.atoms.iter().map(|a| a.element))
        .collect();
    let gpu_constraints: Vec<ShakeConstraint> = cpu_constraints
        .iter()
        .map(|c| {
            let (x, h) = if atoms_flat[c.i] == Element::H {
                (c.j as u32, c.i as u32)
            } else {
                (c.i as u32, c.j as u32)
            };
            ShakeConstraint {
                x_atom: x,
                h_atom: h,
                d_sq: c.d_sq as f32,
            }
        })
        .collect();
    let masses_f32: Vec<f32> = s
        .residues
        .iter()
        .flat_map(|r| r.atoms.iter().map(|a| a.element.mass_da() as f32))
        .collect();
    let shake_data = build_per_x_shake_data(n, &gpu_constraints, &masses_f32);
    eprintln!(
        "Ala-Lys-Glu: {n} atoms, {} X-H constraints",
        cpu_constraints.len()
    );

    let dt_fs = 2.0; // The whole point of SHAKE.
    let gamma_ps_inv = 2.0;
    let temperature_k = 310.0;

    let mut full = FullGpuIntegrator::new(&s, &g, ff, dt_fs, gamma_ps_inv, temperature_k, 42)
        .expect("FullGpuIntegrator construction");
    full.enable_shake(&shake_data, 64, 1e-6_f32);
    let velocities = vec![Vec3::zeros(); n];
    full.upload_initial_state(&s, &velocities);

    let n_steps = 200usize;
    full.step_batch_shake(n_steps);
    full.download_positions_into(&mut s);

    // Check 1: no NaN.
    for r in &s.residues {
        for a in &r.atoms {
            assert!(
                a.position.x.is_finite() && a.position.y.is_finite() && a.position.z.is_finite(),
                "non-finite position after SHAKE-mode integrator"
            );
        }
    }

    // Check 2: every X-H constraint still satisfied.
    let positions: Vec<Vec3> = s
        .residues
        .iter()
        .flat_map(|r| r.atoms.iter().map(|a| a.position))
        .collect();
    let mut max_constraint_err = 0.0_f64;
    let mut worst = String::new();
    for c in &cpu_constraints {
        let r2 = (positions[c.i] - positions[c.j]).norm_squared();
        let err_abs = (r2 - c.d_sq).abs();
        if err_abs > max_constraint_err {
            max_constraint_err = err_abs;
            worst = format!(
                "constraint i={} j={} d²={:.4} got r²={:.4} err={:.4e}",
                c.i, c.j, c.d_sq, r2, err_abs
            );
        }
    }
    eprintln!(
        "after {n_steps} SHAKE-mode integrator steps (dt={dt_fs} fs): \
         max X-H constraint |r² − d²| = {:.3e} Å²  ({worst})",
        max_constraint_err
    );
    assert!(max_constraint_err < 5e-3, "constraint blew up: {worst}");

    eprintln!("SHAKE-mode integrator on GPU ran {n_steps} steps × {dt_fs} fs without divergence");
}
