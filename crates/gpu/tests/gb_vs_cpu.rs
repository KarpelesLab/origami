//! Validate the GPU GB Born-radii + pair-force kernels against the
//! production CPU path `energy::forces_gb::add_gb_forces`.
//!
//! Test target: Ala-Lys-Glu chain — has charged residues so GB
//! contributions are non-trivial.  Same atom set, same parameters
//! on both sides; the CPU code happens to do its own Born-radius
//! recompute via `compute_born_inputs`, the GPU does the same via
//! the gb_born kernel.

use chem::{AminoAcid, Element, standard_ff};
use energy::forces_gb::GB_DEFAULT_CUTOFF_A_PUB;
use energy::gb::{BORN_RADIUS_CUTOFF_A_PUB, OBC_OFFSET_PUB, hct_scale_pub, intrinsic_radius_pub};
use geom::{Vec3, build_extended_chain, build_topology_graph};
use gpu::{GbPipeline, GbSetup, GpuContext, pair_list_to_csr};

#[test]
fn gpu_gb_matches_cpu_on_ala_lys_glu() {
    let ctx = match GpuContext::get() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("GPU unavailable: {e}");
            return;
        }
    };

    let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Lys, AminoAcid::Glu]).unwrap();
    let _g = build_topology_graph(&s);
    let ff = standard_ff();

    let n = s.atom_count();
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n);
    let mut rho: Vec<f32> = Vec::with_capacity(n);
    let mut rho_tilde: Vec<f32> = Vec::with_capacity(n);
    let mut scale: Vec<f32> = Vec::with_capacity(n);
    let mut charges: Vec<f32> = Vec::with_capacity(n);
    for r in &s.residues {
        for a in &r.atoms {
            positions.push([
                a.position.x as f32,
                a.position.y as f32,
                a.position.z as f32,
            ]);
            let r0 = intrinsic_radius_pub(a.element);
            rho.push(r0 as f32);
            rho_tilde.push((r0 - OBC_OFFSET_PUB) as f32);
            scale.push(hct_scale_pub(a.element) as f32);
            charges.push(ff.partial_charge_for(r.monomer, a.name).unwrap_or(0.0) as f32);
        }
    }

    // Build a brute-force neighbour list at the Born-radius cutoff —
    // every i < j pair with |r| < 20 Å.  The GPU kernels walk this
    // CSR list; the pair-force kernel filters by 10 Å internally.
    let born_cutoff = BORN_RADIUS_CUTOFF_A_PUB;
    let born_cutoff_sq = born_cutoff * born_cutoff;
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = positions[i][0] as f64 - positions[j][0] as f64;
            let dy = positions[i][1] as f64 - positions[j][1] as f64;
            let dz = positions[i][2] as f64 - positions[j][2] as f64;
            if dx * dx + dy * dy + dz * dz <= born_cutoff_sq {
                pairs.push((i as u32, j as u32));
            }
        }
    }
    let (counts, starts, indices) = pair_list_to_csr(n, &pairs);

    let mut pipe = GbPipeline::new(
        ctx,
        n,
        GbSetup {
            rho: &rho,
            rho_tilde: &rho_tilde,
            scale: &scale,
            charges: &charges,
            cutoff_a: BORN_RADIUS_CUTOFF_A_PUB as f32,
            pair_cutoff_a: GB_DEFAULT_CUTOFF_A_PUB as f32,
            initial_indices_capacity: indices.len().max(64),
        },
    );
    pipe.update_neighbours(&counts, &starts, &indices);
    pipe.update_positions(&positions);
    let gpu_forces = pipe.compute_forces();

    // CPU production reference.
    let mut cpu_forces = vec![Vec3::zeros(); n];
    energy::forces_gb::add_gb_forces(&s, ff, &mut cpu_forces);

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
    let _ = Element::C;
    eprintln!(
        "max GPU-vs-CPU GB force discrepancy on Ala-Lys-Glu ({n} atoms): {max_err:.3e} kJ/mol/Å"
    );
    eprintln!("  {label}");
    // GB has more accumulation than LJ+Coulomb (every atom contributes
    // to every other atom's Born radius, then those radii feed into
    // pair forces).  f32 noise floor is higher here — 1.0 kJ/mol/Å
    // covers it generously while still catching real bugs (CPU
    // values are 10+ kJ/mol/Å for nearby charged atoms).
    assert!(max_err < 1.0, "GPU and CPU GB disagree: {label}");
}
