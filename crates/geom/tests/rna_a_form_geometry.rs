//! Acceptance tests for the A-form RNA builder.  Confirms the
//! starting geometry is helical, right-handed, with the canonical
//! C3'-endo sugar pucker, no clashes, and clean ribose ring closure.
//! Numbers are *not* required to match canonical A-form exactly —
//! we just want a defensible helical starting point for dynamics.

use chem::Nucleotide;
use geom::{build_a_form_rna_chain, measure, Vec3};

/// Fit the helix axis to a sequence of P positions by power iteration
/// on the principal eigenvector of Σ (p_i - centroid)(p_i - centroid)ᵀ.
/// For a real helix the axial direction dominates the variance.  Sign
/// is then flipped so the rise per residue is positive.
fn fit_helix_axis(p: &[Vec3]) -> (Vec3, Vec3) {
    let centroid = p.iter().sum::<Vec3>() / (p.len() as f64);
    let mut m = nalgebra::Matrix3::zeros();
    for v in p {
        let c = v - centroid;
        m += c * c.transpose();
    }
    let mut axis = Vec3::new(1.0, 1.0, 1.0).normalize();
    for _ in 0..60 {
        axis = (m * axis).normalize();
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

#[test]
fn a_form_helix_is_right_handed_with_positive_rise() {
    // 12-nt chain — long enough to fit a stable axis.
    let block = [
        Nucleotide::Adenine,
        Nucleotide::Uracil,
        Nucleotide::Guanine,
        Nucleotide::Cytosine,
    ];
    let seq: Vec<_> = block.iter().cloned().cycle().take(12).collect();
    let s = build_a_form_rna_chain(&seq).unwrap();

    let p: Vec<_> = s
        .residues
        .iter()
        .map(|r| r.position("P").unwrap())
        .collect();
    let (centroid, axis) = fit_helix_axis(&p);

    // Rise per residue along axis.
    let rises: Vec<f64> = (1..p.len()).map(|i| (p[i] - p[i - 1]).dot(&axis)).collect();
    let mean_rise: f64 = rises.iter().sum::<f64>() / rises.len() as f64;

    // Twist per residue around axis (signed; right-handed = positive).
    let project = |v: Vec3| v - axis * v.dot(&axis);
    let twists: Vec<f64> = (1..p.len())
        .map(|i| {
            let a = project(p[i - 1] - centroid);
            let b = project(p[i] - centroid);
            let cos_t = (a.dot(&b) / (a.norm() * b.norm())).clamp(-1.0, 1.0);
            let sign = a.cross(&b).dot(&axis).signum();
            sign * cos_t.acos().to_degrees()
        })
        .collect();
    let mean_twist: f64 = twists.iter().sum::<f64>() / twists.len() as f64;

    eprintln!("A-form 12-nt: mean rise = {mean_rise:.2} Å/nt, mean twist = {mean_twist:.1}°/nt");

    // Acceptance: rise within ±1 Å of canonical 2.81; twist
    // right-handed (positive) and within ±20° of canonical 32.7°.
    assert!(
        (mean_rise - 2.81).abs() < 1.0,
        "A-form rise off canonical: {mean_rise} Å/nt (canonical 2.81)"
    );
    assert!(
        mean_twist > 10.0 && mean_twist < 60.0,
        "A-form not right-handed in the ballpark: {mean_twist}°/nt (canonical +32.7°)"
    );
}

#[test]
fn a_form_sugar_pucker_is_c3_endo() {
    // C3'-endo pucker is captured by δ = C5'-C4'-C3'-O3' ≈ 82° ± 15°.
    // (C2'-endo is δ ≈ 145°.)  Check the canonical δ on every residue.
    let s = build_a_form_rna_chain(&[Nucleotide::Adenine, Nucleotide::Uracil, Nucleotide::Guanine])
        .unwrap();
    for (i, r) in s.residues.iter().enumerate() {
        let c5 = r.position("C5'").unwrap();
        let c4 = r.position("C4'").unwrap();
        let c3 = r.position("C3'").unwrap();
        let o3 = r.position("O3'").unwrap();
        let delta = measure::dihedral(c5, c4, c3, o3).to_degrees();
        assert!(
            (delta - 82.0).abs() < 15.0,
            "residue {i}: δ = {delta:.1}° (expected ~82° for C3'-endo)"
        );
    }
}

#[test]
fn a_form_anti_glycosidic() {
    // χ = O4'-C1'-N1/N9-C2/C4 is the canonical glycosidic torsion;
    // anti is around -120° to -180°.  Our builder uses C3'-C2'-C1'-N
    // = -160° internally, which corresponds to an anti χ.
    let s = build_a_form_rna_chain(&[Nucleotide::Adenine, Nucleotide::Cytosine]).unwrap();
    for r in &s.residues {
        let o4 = r.position("O4'").unwrap();
        let c1 = r.position("C1'").unwrap();
        let n = match r.monomer {
            geom::structure::Monomer::Rna(chem::Nucleotide::Adenine)
            | geom::structure::Monomer::Rna(chem::Nucleotide::Guanine) => r.position("N9").unwrap(),
            _ => r.position("N1").unwrap(),
        };
        let c2_or_c4 = match r.monomer {
            geom::structure::Monomer::Rna(chem::Nucleotide::Adenine)
            | geom::structure::Monomer::Rna(chem::Nucleotide::Guanine) => r.position("C4").unwrap(),
            _ => r.position("C2").unwrap(),
        };
        let chi = measure::dihedral(o4, c1, n, c2_or_c4).to_degrees();
        assert!(
            chi.abs() > 90.0,
            "χ not anti: {chi:.1}° on a {:?}",
            r.monomer
        );
    }
}

#[test]
fn a_form_chain_has_no_steric_clashes() {
    // No two non-bonded atoms (across the whole chain) should be
    // closer than the typical hydrogen-bond / 1-4 contact floor of
    // 1.3 Å.  Excludes 1-2 and 1-3 pairs from the topology graph;
    // 1-4 pairs use scaled LJ in CHARMM so a close 1-4 contact is
    // allowed (and the test that LJ doesn't explode is
    // `rna_a_form_energy_below_extended`).
    let block = [
        Nucleotide::Adenine,
        Nucleotide::Uracil,
        Nucleotide::Guanine,
        Nucleotide::Cytosine,
    ];
    let seq: Vec<_> = block.iter().cloned().cycle().take(8).collect();
    let s = build_a_form_rna_chain(&seq).unwrap();
    let g = geom::build_topology_graph(&s);
    let bonded: std::collections::HashSet<(usize, usize)> = g
        .bonds
        .iter()
        .flat_map(|b| [(b.a, b.b), (b.b, b.a)])
        .chain(g.angles.iter().flat_map(|a| [(a.a, a.c), (a.c, a.a)]))
        .collect();

    let atoms: Vec<_> = s.iter_atoms().collect();
    for i in 0..atoms.len() {
        for j in (i + 1)..atoms.len() {
            if bonded.contains(&(i, j)) {
                continue;
            }
            let (ri, ai) = atoms[i];
            let (rj, aj) = atoms[j];
            let d = (ai.position - aj.position).norm();
            assert!(
                d > 1.3,
                "close non-bonded contact: {ri}/{} ↔ {rj}/{} = {d:.2} Å",
                ai.name,
                aj.name
            );
        }
    }
}

#[test]
fn a_form_ribose_ring_closes_cleanly() {
    // C1'-O4' is the implicit ring-closure bond.  CHARMM r₀ is 1.414 Å;
    // the NeRF builder doesn't enforce it directly so the error here
    // measures how well our torsion set lands on the ring.
    let s = build_a_form_rna_chain(&[
        Nucleotide::Adenine,
        Nucleotide::Uracil,
        Nucleotide::Guanine,
        Nucleotide::Cytosine,
    ])
    .unwrap();
    for (i, r) in s.residues.iter().enumerate() {
        let c1 = r.position("C1'").unwrap();
        let o4 = r.position("O4'").unwrap();
        let d = (c1 - o4).norm();
        assert!(
            (d - 1.414).abs() < 0.05,
            "residue {i}: ring-closure |C1'-O4'| = {d:.3} Å (want 1.414)"
        );
    }
}
