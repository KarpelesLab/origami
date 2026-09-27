//! Diagnostic: list the worst LJ clashes (closest atom-atom contacts)
//! in a built extended RNA chain.  Helps identify whether the NeRF
//! builder is placing atoms with systematic clashes (e.g. consecutive
//! phosphates too close, or base atoms clashing with the adjacent
//! sugar).
//!
//! Run with: `cargo test -p io --release --test rna_clash_audit -- --nocapture`

use chem::{Nucleotide, classify_atom, standard_ff};
use geom::{build_extended_rna_chain, build_topology_graph};

#[test]
fn audit_extended_chain_lj_clashes() {
    let nts: Vec<Nucleotide> = "GGCACUUCGGUGCC"
        .chars()
        .map(|c| Nucleotide::from_one_letter(c).unwrap())
        .collect();
    let s = build_extended_rna_chain(&nts).expect("build extended");
    let graph = build_topology_graph(&s);
    let ff = standard_ff();

    // Per-atom positions, types, and residue/name labels for reporting.
    let mut positions: Vec<geom::Vec3> = Vec::new();
    let mut atom_types: Vec<chem::AtomType> = Vec::new();
    let mut labels: Vec<String> = Vec::new();
    for (ri, r) in s.residues.iter().enumerate() {
        for atom in &r.atoms {
            positions.push(atom.position);
            atom_types.push(classify_atom(r.monomer, atom.name).unwrap());
            labels.push(format!(
                "{}{}:{}",
                r.monomer.as_nucleotide().unwrap().one_letter(),
                ri,
                atom.name
            ));
        }
    }

    // Pairwise scan; skip 1-2, 1-3, 1-4 (the exclusion mask), report
    // the closest contacts where the centre-to-centre distance is
    // smaller than 0.85 × the LJ contact distance σ.
    let n = positions.len();
    let kcal_to_kj = 4.184;
    #[derive(Clone)]
    struct Clash {
        i: usize,
        j: usize,
        r: f64,
        sigma: f64,
        energy_kj: f64,
    }
    let mut clashes: Vec<Clash> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if graph.is_bonded(i, j) || graph.is_one_three(i, j) || graph.is_one_four(i, j) {
                continue;
            }
            let r = (positions[i] - positions[j]).norm();
            if r < 1e-6 {
                continue;
            }
            let (Some(pi), Some(pj)) = (ff.nonbonded(atom_types[i]), ff.nonbonded(atom_types[j]))
            else {
                continue;
            };
            // CHARMM Rmin = sum of half-Rmin's; LJ σ = Rmin / 2^(1/6).
            let rmin = pi.rmin_half + pj.rmin_half;
            let sigma = rmin / 2.0_f64.powf(1.0 / 6.0);
            if r >= 0.85 * sigma {
                continue;
            }
            // Energy = ε [ (Rmin/r)^12 − 2 (Rmin/r)^6 ] using CHARMM
            // Lorentz-Berthelot ε = sqrt(εi εj).
            let eps = (pi.epsilon * pj.epsilon).sqrt();
            let rratio = rmin / r;
            let r6 = rratio.powi(6);
            let r12 = r6 * r6;
            let e_kcal = eps * (r12 - 2.0 * r6);
            clashes.push(Clash {
                i,
                j,
                r,
                sigma,
                energy_kj: e_kcal * kcal_to_kj,
            });
        }
    }
    clashes.sort_by(|a, b| b.energy_kj.partial_cmp(&a.energy_kj).unwrap());

    let total_kj: f64 = clashes.iter().map(|c| c.energy_kj).sum();
    eprintln!(
        "\n=== LJ clash audit on extended GGCACUUCGGUGCC ===\nClashes (r < 0.85 σ, non-1-2/1-3/1-4): {}\nTotal LJ from these: {:.3e} kJ/mol",
        clashes.len(),
        total_kj
    );
    eprintln!("\nTop 20 individual clashes:");
    for c in clashes.iter().take(20) {
        eprintln!(
            "  {} ↔ {}: r={:.3} Å (σ={:.3} Å, r/σ={:.2}) → {:.2e} kJ/mol",
            labels[c.i],
            labels[c.j],
            c.r,
            c.sigma,
            c.r / c.sigma,
            c.energy_kj,
        );
    }
}
