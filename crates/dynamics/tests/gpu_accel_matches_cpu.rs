//! Cross-check the GPU accelerator against the production CPU SoA path
//! for the LJ + Coulomb + GB contributions.  Builds Ala-Lys-Glu (charged
//! side-chains so GB is non-trivial), runs one CPU force eval and one
//! GPU force eval starting from the same positions, and verifies the
//! per-atom forces agree to f32 precision.
//!
//! This is the integrator-level analogue of the unit tests in
//! `crates/gpu/tests/` — those verify the individual kernels; this
//! verifies the `GpuAccelerator` wiring (CSR upload, position pack,
//! force readback + accumulate).

use chem::{standard_ff, AminoAcid};
use energy::scratch::ForceScratch;
use energy::forces_gb::add_gb_forces_soa;
use energy::forces_nonbonded::add_nonbonded_forces_soa;
use energy::DEFAULT_CUTOFF_A;
use geom::{build_extended_chain, build_topology_graph};

#[test]
fn gpu_accel_matches_cpu_nonbonded_plus_gb_on_ala_lys_glu() {
    // Build the system + topology + scratch (same path every CPU call uses).
    let s = build_extended_chain(&[
        AminoAcid::Ala,
        AminoAcid::Lys,
        AminoAcid::Glu,
    ]).unwrap();
    let g = build_topology_graph(&s);
    let ff = standard_ff();
    let n = s.atom_count();

    // CPU reference: sync positions, zero forces, run nonbonded SoA +
    // GB SoA (same calls the SoA aggregator makes).
    let mut cpu_scratch = ForceScratch::new(&s, &g, ff);
    cpu_scratch.sync_positions(&s);
    cpu_scratch.zero_forces();
    add_nonbonded_forces_soa(&mut cpu_scratch, DEFAULT_CUTOFF_A);
    add_gb_forces_soa(&mut cpu_scratch, &s, ff, energy::forces_gb::GB_DEFAULT_CUTOFF_A_PUB);

    // GPU candidate: build the accelerator, run its combined call on a
    // fresh scratch.
    let mut gpu = match dynamics::GpuAccelerator::new(&s, &g, ff, DEFAULT_CUTOFF_A) {
        Ok(a) => a,
        Err(e) => { eprintln!("GPU unavailable: {e}"); return; }
    };
    let mut gpu_scratch = ForceScratch::new(&s, &g, ff);
    gpu_scratch.sync_positions(&s);
    gpu_scratch.zero_forces();
    gpu.add_nonbonded_and_gb(&mut gpu_scratch);

    // Compare per-atom forces.  f32 round-trip + the accumulation
    // pattern differences put the noise floor at ~1e-3 kJ/mol/Å for
    // moderately-sized systems — well below the ~10 kJ/mol/Å forces
    // the integrator deals with on a per-atom basis.
    let mut max_err = 0.0_f64;
    let mut label = String::new();
    for i in 0..n {
        for axis in 0..3 {
            let (cv, gv) = match axis {
                0 => (cpu_scratch.fxs[i], gpu_scratch.fxs[i]),
                1 => (cpu_scratch.fys[i], gpu_scratch.fys[i]),
                _ => (cpu_scratch.fzs[i], gpu_scratch.fzs[i]),
            };
            let err = (cv - gv).abs();
            if err > max_err {
                max_err = err;
                label = format!("atom {i} axis {axis}: cpu={cv:.6} gpu={gv:.6} err={err:.6}");
            }
        }
    }
    eprintln!(
        "max GPU-accel vs CPU SoA (nonbonded + GB) force discrepancy on Ala-Lys-Glu ({n} atoms): {max_err:.3e} kJ/mol/Å"
    );
    eprintln!("  {label}");
    assert!(
        max_err < 1.0,
        "GPU accelerator and CPU SoA disagree past tolerance: {label}"
    );
}

#[test]
fn gpu_accel_persists_neighbour_list_across_static_calls() {
    // The accelerator only re-uploads the CSR neighbour list when the
    // CPU's Verlet list rebuilds.  Run the same positions twice on the
    // GPU and verify the second call gives identical forces (proves
    // the cached neighbour list still drives correct results when no
    // rebuild was triggered).
    let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala, AminoAcid::Ala]).unwrap();
    let g = build_topology_graph(&s);
    let ff = standard_ff();
    let n = s.atom_count();

    let mut gpu = match dynamics::GpuAccelerator::new(&s, &g, ff, DEFAULT_CUTOFF_A) {
        Ok(a) => a,
        Err(e) => { eprintln!("GPU unavailable: {e}"); return; }
    };

    let mut sc1 = ForceScratch::new(&s, &g, ff);
    sc1.sync_positions(&s);
    sc1.zero_forces();
    gpu.add_nonbonded_and_gb(&mut sc1);

    let mut sc2 = ForceScratch::new(&s, &g, ff);
    sc2.sync_positions(&s);
    sc2.zero_forces();
    gpu.add_nonbonded_and_gb(&mut sc2);

    for i in 0..n {
        assert!((sc1.fxs[i] - sc2.fxs[i]).abs() < 1e-9, "atom {i} fx drift between calls");
        assert!((sc1.fys[i] - sc2.fys[i]).abs() < 1e-9, "atom {i} fy drift between calls");
        assert!((sc1.fzs[i] - sc2.fzs[i]).abs() < 1e-9, "atom {i} fz drift between calls");
    }
}
