//! PSA.2c — smoke test that minimisation with SASA forces enabled still
//! converges to a sensible energy. The SASA term is small relative to
//! bonded/LJ/Coulomb/GB, so adding it shouldn't change the answer much —
//! we just check that the gradient remains well-behaved (no NaN, no
//! explosion, energy still drops).

use chem::{standard_ff, AminoAcid};
use dynamics::{minimize, Algorithm, MinimizeOptions};
use geom::{build_extended_chain, build_topology_graph};

#[test]
fn ala3_minimises_with_sasa_forces() {
    let mut s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala, AminoAcid::Ala]).unwrap();
    let g = build_topology_graph(&s);
    let ff = standard_ff();

    // Few steps — the numerical SASA gradient is slow (~1 s/step on
    // Ala₃, ~10 s/step on Trp-cage). Replacing it with analytical Klenin §3
    // derivatives is the PSA.2-followup. We just need to confirm the
    // gradient is well-behaved, so 10 steps is enough.
    let opts = MinimizeOptions {
        algorithm: Algorithm::Lbfgs,
        max_steps: 10,
        gradient_tol: 5.0,
        max_step_a: 0.1,
        include_sasa: true,
        ..Default::default()
    };
    let result = minimize(&mut s, &g, ff, opts);

    assert!(
        result.final_energy < result.initial_energy,
        "energy didn't drop with SASA: {} → {}",
        result.initial_energy,
        result.final_energy
    );
    assert!(
        result.final_energy.is_finite(),
        "final energy not finite: {}",
        result.final_energy
    );
    for r in &s.residues {
        for a in &r.atoms {
            assert!(a.position.x.is_finite());
            assert!(a.position.y.is_finite());
            assert!(a.position.z.is_finite());
        }
    }
}
