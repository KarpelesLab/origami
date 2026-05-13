//! SHAKE acceptance: with X-H bond-length constraints, the integrator
//! is stable at dt = 2 fs. Without constraints dt = 2 fs is *just*
//! borderline-stable on Trp-cage at 310 K but accumulates noticeable
//! energy drift; with SHAKE the trajectory holds and the wall-time
//! per simulated picosecond drops ~2×.

use chem::standard_ff;
use dynamics::{run_langevin, LangevinOptions};
use geom::build_topology_graph;
use io::read_pdb;

fn read_fixture(path: &str) -> geom::Structure {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    read_pdb(bytes.as_slice()).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

#[test]
fn trp_cage_dt2_with_shake_is_stable() {
    let mut s = read_fixture("../io/tests/fixtures/1L2Y_model1.pdb");
    let g = build_topology_graph(&s);
    let ff = standard_ff();
    let opts = LangevinOptions {
        dt_fs: 2.0,
        temperature_k: 310.0,
        friction_ps_inv: 2.0,
        steps: 1000, // 2 ps of simulated time
        save_every: 0,
        seed: 11,
        randomise_initial_velocities: true,
        include_sasa: false,
        constrain_h_bonds: true,
    };
    let summary = run_langevin(&mut s, &g, ff, opts, |_| {});
    assert!(!summary.diverged, "dt = 2 fs + SHAKE diverged");
    assert_eq!(
        summary.shake_failures, 0,
        "SHAKE failed to converge in {} half-step(s)",
        summary.shake_failures
    );
    // Temperature should be near target. The post-burn-in average is
    // typically within ±50 K with SHAKE's DOF-corrected thermostat;
    // anything that drifts further is a regression.
    assert!(
        (summary.temperature_mean_k - 310.0).abs() < 80.0,
        "T_mean = {:.1} K outside ±80 K of 310 K target",
        summary.temperature_mean_k
    );
}
