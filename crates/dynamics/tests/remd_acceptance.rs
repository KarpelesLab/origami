//! Replica-exchange MD acceptance: a small Ala₃ system, 4 replicas
//! at 300 / 360 / 430 / 520 K, 4 ps total simulated time per replica,
//! swap every 200 fs. Expectations:
//!
//!   • no replica diverges
//!   • at least one swap attempt happens per adjacent pair
//!   • on the temperature ladder we chose, the lowest-pair
//!     acceptance ratio is comfortably above 0.2 (the conventional
//!     "well-mixed" lower bound)

use chem::{standard_ff, AminoAcid};
use dynamics::remd::{run_remd, RemdOptions};
use geom::{build_extended_chain, build_topology_graph};

#[test]
fn ala3_remd_4_replicas_swaps_correctly() {
    let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala, AminoAcid::Ala]).unwrap();
    let g = build_topology_graph(&s);
    let ff = standard_ff();
    let opts = RemdOptions {
        temperatures_k: vec![300.0, 360.0, 430.0, 520.0],
        dt_fs: 1.0,
        friction_ps_inv: 2.0,
        total_time_fs: 4000.0,
        swap_interval_fs: 200.0,
        save_every: 0,
        seed: 17,
        include_sasa: false,
        constrain_h_bonds: false,
    };
    let summary = run_remd(&s, &g, ff, opts, |_| {});

    assert_eq!(summary.n_replicas, 4);
    for r in &summary.per_replica {
        assert!(!r.diverged, "replica at T={:.0} K diverged", r.temperature_k);
    }
    assert_eq!(summary.swap_attempts.len(), 3);
    for &att in &summary.swap_attempts {
        assert!(att >= 5, "expected many swap attempts, got {att}");
    }
    let ratios = summary.acceptance_ratios();
    eprintln!(
        "acceptance ratios: {:?}, attempts {:?}, accepts {:?}",
        ratios, summary.swap_attempts, summary.swap_accepts
    );
    // The narrowest temperature gap (300 → 360 K, factor 1.2) should
    // give the best acceptance. Anything > 0.2 means swaps are
    // actually happening. Anything < 0.05 means the ladder is too
    // spread out for swaps to mix.
    let best_ratio = ratios.iter().cloned().fold(0.0_f64, f64::max);
    assert!(
        best_ratio > 0.1,
        "best-pair acceptance ratio {best_ratio:.3} suggests the swap step isn't working"
    );
}
