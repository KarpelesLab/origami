//! End-to-end stability check on the A-form RNA builder.
//!
//! The geometric tests in `crates/geom/tests/rna_a_form_geometry.rs`
//! prove the builder *produces* a right-handed helix with C3'-endo
//! pucker and anti glycosidic; this test proves the CHARMM27 force
//! field *keeps* it that way under Langevin dynamics.  If the builder
//! were off-canonical enough to land outside the FF's A-form basin,
//! the helix would lose rise/twist or unwind during MD.
//!
//! Runs a 2 ps Langevin trajectory on a 12-nt A-form chain and
//! checks:
//!   1. No divergence.
//!   2. Backbone P-RMSD vs the pre-MD reference stays under 3.5 Å
//!      (helix shape preserved as a whole).
//!   3. Per-step rise stays positive and per-step twist stays
//!      positive (helix didn't flip to left-handed or extend out).

use chem::{Nucleotide, standard_ff};
use dynamics::{Algorithm, LangevinOptions, MinimizeOptions, minimize, run_langevin};
use geom::{Vec3, build_a_form_rna_chain, build_topology_graph, rmsd_p};

fn fit_helix_axis(p: &[Vec3]) -> (Vec3, Vec3) {
    // Power iteration on the (3×3) variance matrix Σ (p_i - c)(p_i - c)ᵀ.
    // Implemented in plain f64 so the test doesn't need a nalgebra
    // dependency.
    let centroid: Vec3 = p.iter().sum::<Vec3>() / (p.len() as f64);
    let mut m = [[0.0_f64; 3]; 3];
    for v in p {
        let c = v - centroid;
        let cs = [c.x, c.y, c.z];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] += cs[i] * cs[j];
            }
        }
    }
    let mut axis = Vec3::new(1.0, 1.0, 1.0).normalize();
    for _ in 0..60 {
        let v = [axis.x, axis.y, axis.z];
        let next = Vec3::new(
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        );
        axis = next.normalize();
    }
    let mean_rise: f64 = (1..p.len())
        .map(|i| (p[i] - p[i - 1]).dot(&axis))
        .sum::<f64>()
        / (p.len() - 1) as f64;
    if mean_rise < 0.0 {
        axis = -axis;
    }
    (centroid, axis)
}

fn helix_rise_twist(p: &[Vec3]) -> (f64, f64) {
    let (centroid, axis) = fit_helix_axis(p);
    let project = |v: Vec3| v - axis * v.dot(&axis);
    let rises: Vec<f64> = (1..p.len()).map(|i| (p[i] - p[i - 1]).dot(&axis)).collect();
    let twists: Vec<f64> = (1..p.len())
        .map(|i| {
            let a = project(p[i - 1] - centroid);
            let b = project(p[i] - centroid);
            let cos_t = (a.dot(&b) / (a.norm() * b.norm())).clamp(-1.0, 1.0);
            let sign = a.cross(&b).dot(&axis).signum();
            sign * cos_t.acos().to_degrees()
        })
        .collect();
    let mean_rise = rises.iter().sum::<f64>() / rises.len() as f64;
    let mean_twist = twists.iter().sum::<f64>() / twists.len() as f64;
    (mean_rise, mean_twist)
}

#[test]
fn a_form_rna_stays_helical_under_langevin() {
    let seq: Vec<_> = [
        Nucleotide::Guanine,
        Nucleotide::Cytosine,
        Nucleotide::Adenine,
        Nucleotide::Uracil,
    ]
    .iter()
    .cloned()
    .cycle()
    .take(12)
    .collect();
    let mut s = build_a_form_rna_chain(&seq).unwrap();
    let g = build_topology_graph(&s);
    let ff = standard_ff();

    // Brief minimisation so the integrator doesn't see whatever
    // residual strain the builder left.  The A-form builder gets
    // very close to a CHARMM27 minimum so 100 steps is plenty.
    let _ = minimize(
        &mut s,
        &g,
        ff,
        MinimizeOptions {
            algorithm: Algorithm::Lbfgs,
            max_steps: 100,
            gradient_tol: 50.0,
            energy_tol: 1.0,
            max_step_a: 0.1,
            include_sasa: false,
            include_cmap: false,
        },
    );

    // Pre-MD reference geometry.
    let initial = s.clone();
    let p_initial: Vec<_> = initial
        .residues
        .iter()
        .filter_map(|r| r.position("P"))
        .collect();
    let (rise_initial, twist_initial) = helix_rise_twist(&p_initial);
    eprintln!("Pre-MD A-form: rise = {rise_initial:.2} Å/nt, twist = {twist_initial:.1}°/nt");
    assert!(
        rise_initial > 0.0 && twist_initial > 0.0,
        "pre-MD A-form should already be right-handed positive-rise"
    );

    // 2 ps Langevin at 310 K.
    let g = build_topology_graph(&s);
    let opts = LangevinOptions {
        dt_fs: 1.0,
        temperature_k: 310.0,
        friction_ps_inv: 2.0,
        steps: 2000,
        save_every: 0,
        seed: 23,
        randomise_initial_velocities: true,
        include_sasa: false,
        include_cmap: false,
        constrain_h_bonds: false,
        use_gpu: false,
        use_gpu_integrator: false,
    };
    let summary = run_langevin(&mut s, &g, ff, opts, |_| {});
    assert!(!summary.diverged, "A-form trajectory diverged");

    // Post-MD geometry.
    let p_final: Vec<_> = s.residues.iter().filter_map(|r| r.position("P")).collect();
    let (rise_final, twist_final) = helix_rise_twist(&p_final);
    let rmsd = rmsd_p(&initial, &s).expect("rmsd_p A-form");
    eprintln!(
        "Post-MD A-form: rise = {rise_final:.2} Å/nt, twist = {twist_final:.1}°/nt, \
         P-RMSD vs initial = {rmsd:.3} Å"
    );

    // ---- Acceptance bars ----
    // 1. Backbone shape preserved (same 3.5 Å threshold as the
    //    tetraloop tests).
    assert!(
        rmsd < 3.5,
        "A-form P-RMSD {rmsd} > 3.5 Å after 2 ps — builder may be off-canonical"
    );
    // 2. Still right-handed (positive twist).
    assert!(
        twist_final > 0.0,
        "A-form helix flipped to left-handed: twist {twist_final}°"
    );
    // 3. Still helical (positive rise, not stretched flat).
    assert!(
        rise_final > 0.5,
        "A-form helix lost rise: {rise_final} Å/nt (was {rise_initial})"
    );
}
