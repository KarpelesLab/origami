//! Validate the GPU smooth-coverage SASA force kernel against
//! central differences of the smooth-coverage AREA from the same
//! kernel.  This pins the analytical gradient down to its own
//! definition — both numerator and denominator come from the GPU,
//! so the test catches sign errors / factor mistakes in the kernel
//! even though the smooth area itself is an approximation of the
//! true SASA.

use chem::{AminoAcid, Element, standard_ff};
use geom::{Vec3, build_extended_chain};
use gpu::{GpuContext, SASA_SMOOTH_DEFAULT_SIGMA_A, SasaSmoothPipeline, SasaSmoothSetup};

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

fn build_csr(positions_f64: &[Vec3], radii_f64: &[f64]) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let n = positions_f64.len();
    let mut counts = vec![0u32; n];
    let mut per_atom_nbrs: Vec<Vec<u32>> = vec![Vec::new(); n];
    // Use a small skin (1 Å) so dots that drift just past a
    // neighbour's centre still see the smooth boundary.
    let skin = 1.0;
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let d = (positions_f64[i] - positions_f64[j]).norm();
            if d <= radii_f64[i] + radii_f64[j] + skin {
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

fn area_total_with_gammas(areas: &[f32], gammas: &[f32]) -> f64 {
    let mut total = 0.0_f64;
    for (a, g) in areas.iter().zip(gammas) {
        total += (*a as f64) * (*g as f64);
    }
    total
}

#[test]
fn gpu_sasa_smooth_force_matches_central_difference() {
    let ctx = match GpuContext::get() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("GPU unavailable: {e}");
            return;
        }
    };
    let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Lys, AminoAcid::Glu]).unwrap();
    let ff = standard_ff();
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
    let gammas_f64 = energy::powersasa::default_sasa_gammas(&s);
    let gammas: Vec<f32> = gammas_f64.iter().map(|&g| g as f32).collect();
    let (counts, starts, indices) = build_csr(&positions_f64, &radii_f64);

    let mut pipe = SasaSmoothPipeline::new(
        ctx,
        n,
        SasaSmoothSetup {
            radii: &radii_f32,
            gammas: &gammas,
            sigma_a: SASA_SMOOTH_DEFAULT_SIGMA_A,
            initial_indices_capacity: indices.len().max(64),
        },
    );
    pipe.update_neighbours(&counts, &starts, &indices);

    // Analytical forces.
    pipe.update_positions(&positions);
    pipe.clear_forces();
    let analytical = pipe.compute_forces();

    // Central differences of E = Σ γ_i × A_i at each atom × axis.
    // (Same definition of A that the analytical kernel
    // differentiates against — so the analytical and numerical
    // gradients must agree to f32 noise.)
    let eps = 5e-4_f32;
    let mut max_err = 0.0_f64;
    let mut argmax = String::new();
    // Spot-check a handful of atoms × axes rather than the full N×3
    // grid — the central-difference loop is slow (2 GPU dispatches
    // per axis × per atom).  10 atoms × 3 axes = 60 GPU evals, still
    // a few seconds.
    let atoms_to_test: Vec<usize> = (0..n).step_by(n / 10).take(10).collect();
    for &atom in &atoms_to_test {
        for axis in 0..3 {
            let mut pos_plus = positions.clone();
            let mut pos_minus = positions.clone();
            pos_plus[atom][axis] += eps;
            pos_minus[atom][axis] -= eps;
            pipe.update_positions(&pos_plus);
            let area_plus = pipe.compute_area();
            pipe.update_positions(&pos_minus);
            let area_minus = pipe.compute_area();
            let e_plus = area_total_with_gammas(&area_plus, &gammas);
            let e_minus = area_total_with_gammas(&area_minus, &gammas);
            let numeric = -(e_plus - e_minus) / (2.0 * eps as f64);
            let analytic = analytical[atom][axis] as f64;
            let err = (numeric - analytic).abs();
            if err > max_err {
                max_err = err;
                argmax = format!(
                    "atom {atom} axis {axis}: numeric={numeric:.4} analytic={analytic:.4} err={err:.4e}"
                );
            }
        }
    }
    eprintln!(
        "max GPU SASA-smooth analytical-vs-central-diff force discrepancy: {:.3e} kJ/mol/Å — {argmax}",
        max_err
    );
    // Tolerance: f32 round-off + ε² truncation in central differences
    // dominate; 1 kJ/mol/Å is generous.  A real bug (sign, factor)
    // would land orders of magnitude above this.
    assert!(
        max_err < 1.0,
        "analytical gradient disagrees with central differences: {argmax}"
    );
    let _ = ff;
}
