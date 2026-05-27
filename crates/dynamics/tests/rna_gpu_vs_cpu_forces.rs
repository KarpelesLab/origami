//! Cross-check the GPU integrator's per-step forces on an RNA chain
//! against the CPU force aggregator.
//!
//! Trick: with `γ = 0`, `T = 0 K`, and zero initial velocities, BAOAB
//! reduces to one velocity-Verlet step.  After exactly one step the
//! displacement is
//!     Δr_i = (F_i / m_i) × (dt² / 2) × ACCEL_FACTOR
//! so reading positions back and undoing the kinematics gives per-atom
//! forces — no need for a debug force-readback API.
//!
//! Bug this test catches (FEAT.gpu.26, originally FIX.gpu.dihedral-max-terms):
//! CHARMM27 nucleic-acid dihedrals around the phosphodiester backbone
//! (e.g. CN7-CN7-ON2-Pn for the α / ζ torsions) carry **up to 5
//! periodic terms** — the protein file caps at 3, so the GPU's fixed
//! 4-term `DihedralTerm` struct silently dropped the 5th term, giving
//! ~10 kJ/mol/Å systematic force error on the O3' / P backbone atoms.
//! Fix: split >4-term dihedrals into multiple GPU records sharing the
//! same atom tuple.  After the fix the residual error is f32
//! cancellation noise on charged atoms (mainly P, q ≈ +1.5 e).

use chem::{standard_ff, Nucleotide};
use dynamics::full_gpu_integrator::FullGpuIntegrator;
use dynamics::{minimize, Algorithm, MinimizeOptions};
use energy::total_force;
use geom::{build_extended_rna_chain, build_topology_graph, Vec3};
use gpu::GpuContext;

const DT_FS: f64 = 0.5;
const ACCEL_FACTOR: f64 = 1.0e-4;

#[test]
fn gpu_force_eval_matches_cpu_on_rna_chain() {
    if GpuContext::get().is_err() {
        eprintln!("GPU unavailable, skipping");
        return;
    }

    let mut s = build_extended_rna_chain(&[
        Nucleotide::Uracil,
        Nucleotide::Cytosine,
        Nucleotide::Adenine,
        Nucleotide::Guanine,
    ]).unwrap();
    let g = build_topology_graph(&s);
    let ff = standard_ff();

    minimize(&mut s, &g, ff, MinimizeOptions {
        algorithm: Algorithm::Lbfgs,
        max_steps: 1000,
        gradient_tol: 0.1,
        ..MinimizeOptions::default()
    });
    let n = s.atom_count();

    // CPU reference.
    let cpu_forces: Vec<Vec3> = total_force(&s, &g, ff);

    // GPU: one BAOAB step at γ=0, T=0, v0=0 → pure velocity-Verlet.
    let mut integ = FullGpuIntegrator::new(
        &s, &g, ff, DT_FS, /*gamma_ps_inv=*/ 0.0,
        /*temperature_k=*/ 0.0, /*seed=*/ 31,
    ).expect("FullGpuIntegrator::new on RNA");
    let velocities = vec![Vec3::zeros(); n];
    integ.upload_initial_state(&s, &velocities);
    let before = s.clone();
    integ.step_batch(1);
    let mut after = s.clone();
    integ.download_positions_into(&mut after);

    let dt_sq = DT_FS * DT_FS;
    let masses: Vec<f64> = s.residues.iter().flat_map(|r| r.atoms.iter())
        .map(|a| a.element.mass_da() as f64).collect();

    let mut max_abs = 0.0_f64;
    let mut worst = 0usize;
    let mut total_cpu_mag = 0.0_f64;
    let mut b_iter = before.residues.iter().flat_map(|r| r.atoms.iter());
    let mut a_iter = after.residues.iter().flat_map(|r| r.atoms.iter());
    let mut gpu_forces = Vec::with_capacity(n);
    for i in 0..n {
        let p0 = b_iter.next().unwrap().position;
        let p1 = a_iter.next().unwrap().position;
        let scale = 2.0 * masses[i] / (dt_sq * ACCEL_FACTOR);
        let f_gpu = (p1 - p0) * scale;
        gpu_forces.push(f_gpu);
        total_cpu_mag += cpu_forces[i].norm();
        let diff = (cpu_forces[i] - f_gpu).norm();
        if diff > max_abs { max_abs = diff; worst = i; }
    }

    // Resolve the worst atom for the diagnostic message.
    let (worst_res, worst_name) = {
        let mut k = 0usize;
        let mut hit = (0usize, "?");
        for (ri, r) in s.residues.iter().enumerate() {
            for a in &r.atoms {
                if k == worst { hit = (ri, a.name); }
                k += 1;
            }
        }
        hit
    };
    eprintln!(
        "UCAG ({n} atoms): mean |F_cpu|={:.2} kJ/mol/Å, max |ΔF|={max_abs:.3e} on \
         atom {worst} ({} of res {worst_res})",
        total_cpu_mag / n as f64, worst_name,
    );

    // Phosphorus has q = +1.5 e in CHARMM27 NA — the largest partial
    // charge of any atom in the system.  Its GB (~70) and nonbonded
    // (~30) terms partly cancel down to a few kJ/mol/Å of net force, so
    // single-precision noise on the dominant terms shows up as a few
    // kJ/mol/Å absolute error.  Pre-FEAT.gpu.26 this test saw
    // ~10 kJ/mol/Å on backbone O3' atoms due to truncated dihedrals;
    // post-fix the floor is ~2 kJ/mol/Å on the phosphorus.
    assert!(max_abs < 3.0,
        "GPU and CPU forces diverge on RNA: max |ΔF|={max_abs} on atom {worst} \
         ({worst_name} of res {worst_res})");
}
