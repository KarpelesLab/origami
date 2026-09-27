//! Diagnostic: build an A-form RNA helix, report rise/twist per
//! nucleotide, clash atoms, and ring-closure error.  Used while
//! tuning the A-form torsion set; marked `#[ignore]` because it's
//! diagnostic, not an acceptance test.
//!
//! Run with:
//! ```text
//! cargo test --release -p geom --test rna_a_form_probe -- --ignored --nocapture
//! ```

use chem::Nucleotide;
use geom::{build_a_form_rna_chain, build_extended_rna_chain, measure};

#[test]
#[ignore]
fn probe_chi_offset() {
    for (label, s) in [
        (
            "extended",
            build_extended_rna_chain(&[Nucleotide::Adenine]).unwrap(),
        ),
        (
            "A-form  ",
            build_a_form_rna_chain(&[Nucleotide::Adenine]).unwrap(),
        ),
    ] {
        let r = &s.residues[0];
        let o4 = r.position("O4'").unwrap();
        let c1 = r.position("C1'").unwrap();
        let n9 = r.position("N9").unwrap();
        let c4 = r.position("C4").unwrap();
        let c3 = r.position("C3'").unwrap();
        let c2 = r.position("C2'").unwrap();
        let chi_canon = measure::dihedral(o4, c1, n9, c4).to_degrees();
        let chi_internal = measure::dihedral(c3, c2, c1, n9).to_degrees();
        eprintln!("{label}: internal C3'-C2'-C1'-N9 = {chi_internal:.1}°, canonical O4'-C1'-N9-C4 = {chi_canon:.1}°");
    }
}

#[test]
#[ignore]
fn probe_a_form_helix() {
    use geom::build_rna_chain_with_torsions;
    use std::f64::consts::PI;
    let deg = |d: f64| d * PI / 180.0;
    // 10-residue chain so helix params resolve cleanly.
    let block = [
        Nucleotide::Adenine,
        Nucleotide::Uracil,
        Nucleotide::Guanine,
        Nucleotide::Cytosine,
    ];
    let seq: Vec<_> = block.iter().cloned().cycle().take(10).collect();

    // Several candidate torsion sets to compare.
    let a_form = geom::builder::rna_ic::RnaTorsionSet::a_form();
    let saenger = geom::builder::rna_ic::RnaTorsionSet {
        alpha: deg(-50.0),
        beta: deg(172.0),
        gamma: deg(41.0),
        delta: deg(79.0),
        epsilon: deg(-146.0),
        zeta: deg(-78.0),
        ..a_form
    };
    let amber_nab = geom::builder::rna_ic::RnaTorsionSet {
        alpha: deg(-75.0),
        beta: deg(175.0),
        gamma: deg(47.0),
        delta: deg(79.0),
        epsilon: deg(-147.0),
        zeta: deg(-75.0),
        ..a_form
    };

    for (label, s) in [
        ("extended", build_extended_rna_chain(&seq).unwrap()),
        ("A (Olson)", build_a_form_rna_chain(&seq).unwrap()),
        (
            "A (Saenger)",
            build_rna_chain_with_torsions(&seq, saenger).unwrap(),
        ),
        (
            "A (AMBER NAB)",
            build_rna_chain_with_torsions(&seq, amber_nab).unwrap(),
        ),
    ] {
        eprintln!("\n=== {label} 10-nt chain ===");
        let p: Vec<_> = s
            .residues
            .iter()
            .map(|r| r.position("P").unwrap())
            .collect();

        // ---- helix-axis fit via best-fit line through P atoms ----
        // Crude PCA: axis = principal eigenvector of the covariance
        // matrix of the centred P positions.  For a true helix the
        // axial direction has by far the highest variance over the
        // span.
        let centroid: geom::Vec3 = p.iter().sum::<geom::Vec3>() / (p.len() as f64);
        let centred: Vec<_> = p.iter().map(|x| x - centroid).collect();
        // Power iteration on M = Σ p_i p_iᵀ (3×3 outer-product sum).
        let mut m = nalgebra::Matrix3::zeros();
        for v in &centred {
            m += v * v.transpose();
        }
        let mut axis = geom::Vec3::new(1.0, 1.0, 1.0).normalize();
        for _ in 0..50 {
            axis = (m * axis).normalize();
        }

        // ---- helix radius (mean P distance from axis line through centroid) ----
        let radii: Vec<f64> = centred
            .iter()
            .map(|v| (v - axis * v.dot(&axis)).norm())
            .collect();
        let mean_radius = radii.iter().sum::<f64>() / radii.len() as f64;

        // ---- rise per residue (along axis) ----
        let mut rises: Vec<f64> = (1..p.len()).map(|i| (p[i] - p[i - 1]).dot(&axis)).collect();
        // Sign: align so rise is positive.
        let mean_rise_signed = rises.iter().sum::<f64>() / rises.len() as f64;
        if mean_rise_signed < 0.0 {
            axis = -axis;
            rises = rises.iter().map(|r| -r).collect();
        }
        let mean_rise = rises.iter().sum::<f64>() / rises.len() as f64;

        // ---- twist per residue (around axis) ----
        // Project P_i and P_{i+1} into plane normal to axis, compute
        // signed angle.
        let project = |v: geom::Vec3| v - axis * v.dot(&axis);
        let mut twists: Vec<f64> = Vec::new();
        for i in 1..p.len() {
            let a = project(centred[i - 1]);
            let b = project(centred[i]);
            let cross = a.cross(&b);
            let sign = cross.dot(&axis).signum();
            let cos_t = (a.dot(&b) / (a.norm() * b.norm())).clamp(-1.0, 1.0);
            twists.push(sign * cos_t.acos().to_degrees());
        }
        let mean_twist = twists.iter().sum::<f64>() / twists.len() as f64;

        eprintln!(
            "  mean P-P (along bond) = {:.2} Å",
            (1..p.len()).map(|i| (p[i] - p[i - 1]).norm()).sum::<f64>() / (p.len() - 1) as f64
        );
        eprintln!("  helix radius (P to axis) = {mean_radius:.2} Å  (canonical A-form: ~9.4 Å)");
        eprintln!("  rise per residue (axial) = {mean_rise:.2} Å    (canonical A-form: 2.81 Å)");
        eprintln!(
            "  twist per residue (around axis) = {mean_twist:.1}°  (canonical A-form: +32.7°)"
        );

        // ---- ring closure ----
        let mut max_ring_err = 0.0_f64;
        for r in &s.residues {
            let c1 = r.position("C1'").unwrap();
            let o4 = r.position("O4'").unwrap();
            max_ring_err = max_ring_err.max(((c1 - o4).norm() - 1.414).abs());
        }
        eprintln!("  max ring-closure error |C1'-O4' - 1.414| = {max_ring_err:.4} Å");

        // ---- nearest non-bonded pair across all atoms ----
        let atoms: Vec<_> = s.iter_atoms().collect();
        let mut close_pairs: Vec<(f64, String, String)> = Vec::new();
        for i in 0..atoms.len() {
            for j in (i + 1)..atoms.len() {
                let (ri, ai) = atoms[i];
                let (rj, aj) = atoms[j];
                let d = (ai.position - aj.position).norm();
                if d < 1.5 {
                    close_pairs.push((d, format!("{ri}/{}", ai.name), format!("{rj}/{}", aj.name)));
                }
            }
        }
        close_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        eprintln!("  closest 5 pairs (any atoms):");
        for (d, a, b) in close_pairs.iter().take(5) {
            eprintln!("    {a} ↔ {b} = {d:.2} Å");
        }
    }
}
