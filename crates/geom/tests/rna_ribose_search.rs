//! Grid-search the three ribose-branch torsions
//! (`O4_TORS`, `C2_TORS`, `C1_TORS`) to find a set that closes the
//! C1'-O4' ring at ≈ 1.414 Å for a given backbone γ.  The current
//! values in `rna_ic` were fitted against γ = 54° (gauche, A-form);
//! this search lets us re-fit if we change γ to make the extended
//! chain straighter.
//!
//! Run with `cargo test -p geom --release --test rna_ribose_search
//! -- --ignored --nocapture`.  Marked `#[ignore]` because it's a
//! ~10-second exploratory search, not an acceptance test.

use chem::{Element, Nucleotide};
use geom::nerf::place_atom;
use geom::structure::Monomer;
use geom::Vec3;
use std::f64::consts::PI;

#[test]
#[ignore]
fn grid_search_branch_torsions_for_ring_closure() {
    // Replay the existing builder's sugar+phosphate placement for a
    // single residue with configurable backbone + branch torsions,
    // then measure the implicit C1'-O4' closure distance.

    let deg = |d: f64| d * PI / 180.0;

    // ---- backbone constants (must match `rna_ic`) ----
    let p_o5: f64 = 1.600;
    let o5_c5: f64 = 1.440;
    let c5_c4: f64 = 1.512;
    let c4_c3: f64 = 1.529;
    let c4_o4: f64 = 1.480;
    let c3_c2: f64 = 1.460;
    let c2_c1: f64 = 1.450;
    let p_o5_c5 = deg(120.9);
    let o5_c5_c4 = deg(110.2);
    let c5_c4_c3 = deg(115.0);
    let c5_c4_o4 = deg(109.5);
    let c4_c3_c2 = deg(102.5);
    let c3_c2_c1 = deg(101.5);
    let beta = deg(178.0);

    // ---- candidate γ values to search ----
    // The current builder uses γ = 54° (A-form gauche).  γ = 180°
    // gives a properly extended chain but breaks the closure; we
    // sweep a range around it.
    for &gamma_deg in &[180.0_f64, 170.0, 150.0, 120.0, 60.0, 54.0] {
        let gamma = deg(gamma_deg);

        // Place P, O5', C5' anchored at origin (first residue).
        let p = Vec3::zeros();
        let o5 = Vec3::new(p_o5, 0.0, 0.0);
        let a = PI - p_o5_c5;
        let c5 = Vec3::new(o5.x + o5_c5 * a.cos(), o5_c5 * a.sin(), 0.0);

        // Walk the main backbone path with the candidate γ.
        let c4 = place_atom(p, o5, c5, c5_c4, o5_c5_c4, beta);
        let c3 = place_atom(o5, c5, c4, c4_c3, c5_c4_c3, gamma);

        // Sweep branch torsions in 3° steps.
        let step = 3.0;
        let mut best: Option<(f64, f64, f64, f64)> = None;
        let mut t = -180.0_f64;
        let candidates = {
            let mut v = Vec::new();
            while t <= 180.0 {
                v.push(t);
                t += step;
            }
            v
        };
        for &o4_t in &candidates {
            let o4 = place_atom(o5, c5, c4, c4_o4, c5_c4_o4, deg(o4_t));
            for &c2_t in &candidates {
                let c2 = place_atom(c5, c4, c3, c3_c2, c4_c3_c2, deg(c2_t));
                for &c1_t in &candidates {
                    let c1 = place_atom(c4, c3, c2, c2_c1, c3_c2_c1, deg(c1_t));
                    let dist = (c1 - o4).norm();
                    let err = (dist - 1.414).abs();
                    if best.map_or(true, |b: (f64, f64, f64, f64)| err < b.3) {
                        best = Some((o4_t, c2_t, c1_t, err));
                    }
                }
            }
        }
        let (a_o4, a_c2, a_c1, err) = best.unwrap();
        eprintln!(
            "γ = {:>5.0}°: best branch (O4_TORS, C2_TORS, C1_TORS) = ({:>5.0}°, {:>5.0}°, {:>5.0}°), |C1'-O4' - 1.414| = {:.4} Å",
            gamma_deg, a_o4, a_c2, a_c1, err
        );
    }
    // Confirm that geom::Vec3 etc. are still wired (defensive: this
    // test is mostly diagnostic but should compile under workspace
    // changes).  Avoids dead-code warnings.
    let _ = (Element::C, Monomer::Rna(Nucleotide::Adenine));
}

#[test]
#[ignore]
fn grid_search_o2_torsion_for_no_o2_c1_clash() {
    // With the new γ=180° backbone + branch torsions, the existing
    // O2_TORS = 48° puts O2' very close to C1' (~0.6 Å clash). The
    // ribose hydroxyl O2' is placed off C2' independently of the
    // ring closure, so we can re-tune it without breaking the ring.
    // Search for the value that maximises |O2'-C1'| distance.
    let deg = |d: f64| d * PI / 180.0;
    let p_o5: f64 = 1.600;
    let o5_c5: f64 = 1.440;
    let c5_c4: f64 = 1.512;
    let c4_c3: f64 = 1.529;
    let c4_o4: f64 = 1.480;
    let c3_c2: f64 = 1.460;
    let c2_c1: f64 = 1.450;
    let c2_o2: f64 = 1.400;
    let p_o5_c5 = deg(120.9);
    let o5_c5_c4 = deg(110.2);
    let c5_c4_c3 = deg(115.0);
    let c5_c4_o4 = deg(109.5);
    let c4_c3_c2 = deg(102.5);
    let c3_c2_c1 = deg(101.5);
    let c3_c2_o2 = deg(110.7);
    let beta = deg(178.0);
    let gamma = deg(180.0);
    let o4_t = deg(111.0);
    let c2_t = deg(-159.0);
    let c1_t = deg(24.0);

    let p = Vec3::zeros();
    let o5 = Vec3::new(p_o5, 0.0, 0.0);
    let a = PI - p_o5_c5;
    let c5 = Vec3::new(o5.x + o5_c5 * a.cos(), o5_c5 * a.sin(), 0.0);
    let c4 = place_atom(p, o5, c5, c5_c4, o5_c5_c4, beta);
    let c3 = place_atom(o5, c5, c4, c4_c3, c5_c4_c3, gamma);
    let _o4 = place_atom(o5, c5, c4, c4_o4, c5_c4_o4, o4_t);
    let c2 = place_atom(c5, c4, c3, c3_c2, c4_c3_c2, c2_t);
    let c1 = place_atom(c4, c3, c2, c2_c1, c3_c2_c1, c1_t);

    let mut best: Option<(f64, f64)> = None;
    let step = 1.0_f64;
    let mut t = -180.0_f64;
    while t <= 180.0 {
        let o2 = place_atom(c4, c3, c2, c2_o2, c3_c2_o2, deg(t));
        let d = (o2 - c1).norm();
        if best.map_or(true, |b| d > b.1) {
            best = Some((t, d));
        }
        t += step;
    }
    let (o2_t, dist) = best.unwrap();
    eprintln!(
        "Best O2_TORS = {:.0}° gives |O2'-C1'| = {:.3} Å (was clashing at 0.61 Å with O2_TORS=48°)",
        o2_t, dist
    );
    let _ = (Element::C, Monomer::Rna(Nucleotide::Adenine));
}
