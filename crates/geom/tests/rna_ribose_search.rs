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
fn grid_search_op_torsions_for_no_h3_op_clash() {
    // The current OP1_TORS = 120°, OP2_TORS = -120° produce an
    // inter-residue H3'(i) ↔ OP2(i+1) close-contact at 1.21 Å
    // (~13 kJ/mol per clash × 13 junctions in a 14-mer chain).
    // Build a 2-nucleotide chain, sweep (OP1_TORS, OP2_TORS),
    // measure |H3'(0) - OP2(1)| AND |H3'(0) - OP1(1)|, find the
    // pair that maximises the *minimum* of the two distances.
    use chem::Nucleotide;
    use geom::build_extended_rna_chain;
    use std::f64::consts::PI;

    // We have to rebuild the chain via the public builder because
    // the H3'/H placement isn't exposed in `rna_ic`.  Instead,
    // we work around: sweep the constants by patching the test's
    // own runtime placement (read-only on the public builder).
    //
    // Simpler approach: just iterate over candidate (op1, op2)
    // pairs by temporarily setting them via a thread-local? — no,
    // overkill.  Cleanest is to do the placement manually here,
    // matching the builder logic, and report the best torsions.

    let deg = |d: f64| d * PI / 180.0;
    let p_o5: f64 = 1.600;
    let p_op: f64 = 1.480;
    let o5_c5: f64 = 1.440;
    let c5_c4: f64 = 1.512;
    let c4_c3: f64 = 1.529;
    let c3_o3: f64 = 1.433;
    let p_o5_c5 = deg(120.9);
    let o5_c5_c4 = deg(110.2);
    let c5_c4_c3 = deg(115.0);
    let c4_c3_o3 = deg(110.6);
    let o5_p_op = deg(108.0);
    let o3_p_o5 = deg(104.0);
    let c3_o3_p = deg(119.7);
    let beta = deg(178.0);
    let gamma = deg(180.0);
    let delta = deg(82.0);
    let epsilon = deg(-153.0);
    let zeta = deg(-71.0);
    // Place residue 0 sugar/phosphate scaffold up to O3'.
    let p0 = Vec3::zeros();
    let o5_0 = Vec3::new(p_o5, 0.0, 0.0);
    let ang = PI - p_o5_c5;
    let c5_0 = Vec3::new(o5_0.x + o5_c5 * ang.cos(), o5_c5 * ang.sin(), 0.0);
    let c4_0 = place_atom(p0, o5_0, c5_0, c5_c4, o5_c5_c4, beta);
    let c3_0 = place_atom(o5_0, c5_0, c4_0, c4_c3, c5_c4_c3, gamma);
    let o3_0 = place_atom(c5_0, c4_0, c3_0, c3_o3, c4_c3_o3, delta);
    // H3' of residue 0 — placed by the builder's sp³ one-H helper.
    // Replicate by computing the negated-sum of unit vectors from
    // C3' to its three placed heavy neighbours (C4', O3', C2').
    // We don't have C2' yet here; pull it via the same branch
    // torsion as the builder.
    let c4_c3_c2 = deg(102.5);
    let c3_c2 = 1.460;
    let c2_t = deg(-159.0);
    let c2_0 = place_atom(c5_0, c4_0, c3_0, c3_c2, c4_c3_c2, c2_t);
    // sp³ one-H placement for H3':
    let d1 = (c4_0 - c3_0).normalize();
    let d2 = (o3_0 - c3_0).normalize();
    let d3 = (c2_0 - c3_0).normalize();
    let h3p_dir = -(d1 + d2 + d3).normalize();
    let h3_0 = c3_0 + h3p_dir * 1.090;

    // Place residue 1's P (inter-residue, off residue 0's O3') and
    // its OP1/OP2 with candidate dihedrals.
    let p1 = place_atom(c4_0, c3_0, o3_0, /*O3-P*/ 1.600, c3_o3_p, epsilon);
    let o5_1 = place_atom(c3_0, o3_0, p1, p_o5, o3_p_o5, zeta);
    // Build a manual scan of (op1, op2).
    let step = 5.0_f64;
    let mut best: Option<(f64, f64, f64, f64)> = None;
    let mut t1 = -180.0_f64;
    while t1 <= 180.0 {
        let op1 = place_atom(c5_0, o5_1, p1, p_op, o5_p_op, deg(t1));
        let mut t2 = -180.0_f64;
        while t2 <= 180.0 {
            // Avoid putting OP1 and OP2 on top of each other (skip
            // pairs where they overlap).
            let op2 = place_atom(c5_0, o5_1, p1, p_op, o5_p_op, deg(t2));
            if (op1 - op2).norm() < 1.5 {
                t2 += step;
                continue;
            }
            let d_op1 = (h3_0 - op1).norm();
            let d_op2 = (h3_0 - op2).norm();
            let min_d = d_op1.min(d_op2);
            if best.map_or(true, |b: (f64, f64, f64, f64)| min_d > b.2) {
                best = Some((t1, t2, min_d, d_op1.max(d_op2)));
            }
            t2 += step;
        }
        t1 += step;
    }
    let (t1, t2, min_d, max_d) = best.unwrap();
    eprintln!(
        "Best (OP1_TORS, OP2_TORS) = ({:.0}°, {:.0}°): min |H3'(0) - OP*(1)| = {:.3} Å, max = {:.3} Å",
        t1, t2, min_d, max_d
    );
    let _ = build_extended_rna_chain;
    let _ = Nucleotide::Adenine;
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
