//! Grid-search over A-form backbone torsions to find the combination
//! that produces a canonical right-handed A-form helix:
//! rise ≈ 2.81 Å/nt, twist ≈ +32.7°/nt, P-from-axis radius ≈ 9.4 Å.
//!
//! `#[ignore]` because it's exploratory (sweeping a 2D grid takes a
//! few seconds and is run by hand when re-tuning, not on every CI
//! pass).  The winning torsion set is hard-coded back into
//! `RnaTorsionSet::a_form()`.
//!
//! Sweep variables:
//!   γ — O5'-C5'-C4'-C3'  (g+ near 54°)
//!   ε — C4'-C3'-O3'-P    (around -150°)
//! These two dihedrals dominate the inter-residue rotation around
//! the helix axis.  α / β / δ / ζ / χ are kept at the published Olson
//! 2009 means; the ribose-branch torsions O4_TORS / C2_TORS / C1_TORS
//! shift with γ via the Δγ-rotation-around-C5'-C4' rule the builder
//! comment explains.

use chem::{Nucleotide, standard_ff};
use geom::builder::rna_ic::RnaTorsionSet;
use geom::{Vec3, build_rna_chain_with_torsions, build_topology_graph};
use std::f64::consts::PI;

fn deg(d: f64) -> f64 {
    d * PI / 180.0
}

fn fit_helix_axis(p: &[Vec3]) -> (Vec3, Vec3) {
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
    for _ in 0..80 {
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

fn helix_metrics(s: &geom::Structure) -> (f64, f64, f64, f64) {
    // Returns (mean_rise, mean_twist_deg, mean_radius, max_ring_err).
    let p: Vec<Vec3> = s.residues.iter().filter_map(|r| r.position("P")).collect();
    let (centroid, axis) = fit_helix_axis(&p);
    let rises: Vec<f64> = (1..p.len()).map(|i| (p[i] - p[i - 1]).dot(&axis)).collect();
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
    let mean_rise = rises.iter().sum::<f64>() / rises.len() as f64;
    let mean_twist = twists.iter().sum::<f64>() / twists.len() as f64;
    let radii: Vec<f64> = p
        .iter()
        .map(|q| (q - centroid - axis * ((q - centroid).dot(&axis))).norm())
        .collect();
    let mean_radius = radii.iter().sum::<f64>() / radii.len() as f64;
    let mut max_ring_err = 0.0_f64;
    for r in &s.residues {
        if let (Some(c1), Some(o4)) = (r.position("C1'"), r.position("O4'")) {
            max_ring_err = max_ring_err.max(((c1 - o4).norm() - 1.414).abs());
        }
    }
    (mean_rise, mean_twist, mean_radius, max_ring_err)
}

#[test]
#[ignore]
fn grid_search_gamma_epsilon_for_canonical_a_form() {
    let block = [
        Nucleotide::Adenine,
        Nucleotide::Uracil,
        Nucleotide::Guanine,
        Nucleotide::Cytosine,
    ];
    let seq: Vec<_> = block.iter().cloned().cycle().take(12).collect();

    // Canonical targets.
    let r_target = 2.81; // Å/nt rise
    let t_target = 32.7; // °/nt twist
    let rad_target = 9.4; // Å helix radius

    // Score: weighted sum of squared deviations.  Weights tuned by
    // hand so the three axes contribute comparable magnitudes near
    // the current builder's miss.
    let score = |r: f64, t: f64, rad: f64| -> f64 {
        let w_r = 1.0; // 0.1 Å miss → 0.01
        let w_t = 0.01; // 5°  miss → 0.25
        let w_rad = 0.1; // 1 Å miss → 0.1
        let dr = r - r_target;
        let dt = t - t_target;
        let drad = rad - rad_target;
        w_r * dr * dr + w_t * dt * dt + w_rad * drad * drad
    };

    let base = RnaTorsionSet::a_form();
    let gamma_range: Vec<f64> = (40..=65).step_by(3).map(|x| x as f64).collect();
    let eps_range: Vec<f64> = (-175..=-135).step_by(5).map(|x| x as f64).collect();
    let alpha_range: Vec<f64> = (-90..=-40).step_by(5).map(|x| x as f64).collect();
    let zeta_range: Vec<f64> = (-95..=-45).step_by(5).map(|x| x as f64).collect();

    // Score each torsion-set by geometry AND energy.  The previous
    // (γ=54° baseline) A-form was 250 kJ/mol below extended — any
    // candidate we adopt should keep that.  Combined score:
    //   geom_score + 0.001 * (E_torsion_set - E_baseline)
    // so a 1000 kJ/mol energy regression costs the same as a 1 unit
    // jump in geom_score (≈ rise off by 1 Å).
    let ff = standard_ff();
    let total_energy = |s: &geom::Structure| -> f64 {
        use energy::bonded::bonded_energy;
        use energy::{DEFAULT_CUTOFF_A, gb_energy, nonbonded_energy};
        let g = build_topology_graph(s);
        let b = bonded_energy(s, &g, ff);
        let nb = nonbonded_energy(s, &g, ff, DEFAULT_CUTOFF_A);
        let gb = gb_energy(s, ff);
        b.total_kj_mol() + nb.lj_kj_mol + nb.coulomb_kj_mol + gb.gb_kj_mol
    };

    // (combined, α, γ, ε, ζ, rise, twist, radius, geom_score, energy)
    type Row = (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64);
    let mut all: Vec<Row> = Vec::new();
    let mut tried = 0usize;
    let mut rejected_ring = 0usize;
    for &a_deg in &alpha_range {
        for &g_deg in &gamma_range {
            for &e_deg in &eps_range {
                for &z_deg in &zeta_range {
                    tried += 1;
                    let tors = RnaTorsionSet {
                        alpha: deg(a_deg),
                        gamma: deg(g_deg),
                        epsilon: deg(e_deg),
                        zeta: deg(z_deg),
                        o4_tors: deg(g_deg - 69.0),
                        ..base
                    };
                    let Ok(s) = build_rna_chain_with_torsions(&seq, tors) else {
                        continue;
                    };
                    let (r, t, rad, ring) = helix_metrics(&s);
                    if ring > 0.05 {
                        rejected_ring += 1;
                        continue;
                    }
                    let geom_sc = score(r, t, rad);
                    let e = total_energy(&s);
                    // Combined: 1000 kJ/mol energy penalty = 1.0 on the
                    // geom_score scale.  Use the previous A-form energy
                    // (~7900 kJ/mol on UCAG, scales linearly with chain
                    // length so 12 × 3.0 ≈ 24000 here) as a soft anchor.
                    let combined = geom_sc + (e.max(0.0)) / 100_000.0;
                    all.push((combined, a_deg, g_deg, e_deg, z_deg, r, t, rad, geom_sc, e));
                }
            }
        }
    }
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    eprintln!(
        "Grid swept {tried} 4D points ({rejected_ring} rejected for ring closure).\n\
         Top 20 (α, γ, ε, ζ, rise, twist, radius, geom_sc, E):"
    );
    for &(_c, a, g, e, z, r, t, rad, gs, en) in all.iter().take(20) {
        eprintln!(
            "  α={a:>4.0}° γ={g:>3.0}° ε={e:>5.0}° ζ={z:>4.0}°  \
             rise={r:>4.2}  twist={t:>5.1}  radius={rad:>4.2}  gs={gs:.2}  E={en:>8.0}"
        );
    }

    // Also report the geometry-only winner separately for comparison.
    let mut by_geom: Vec<_> = all.clone();
    by_geom.sort_by(|a, b| a.8.partial_cmp(&b.8).unwrap());
    eprintln!("\nGeometry-only winner:");
    let g_only = by_geom[0];
    eprintln!(
        "  α={:.0}° γ={:.0}° ε={:.0}° ζ={:.0}°  rise={:.2} twist={:.1} radius={:.2} E={:.0}",
        g_only.1, g_only.2, g_only.3, g_only.4, g_only.5, g_only.6, g_only.7, g_only.9
    );

    // Energy-only winner.
    let mut by_e: Vec<_> = all.clone();
    by_e.sort_by(|a, b| a.9.partial_cmp(&b.9).unwrap());
    let e_only = by_e[0];
    eprintln!("Energy-only winner:");
    eprintln!(
        "  α={:.0}° γ={:.0}° ε={:.0}° ζ={:.0}°  rise={:.2} twist={:.1} radius={:.2} E={:.0}",
        e_only.1, e_only.2, e_only.3, e_only.4, e_only.5, e_only.6, e_only.7, e_only.9
    );
}
