//! Diagnostic: list the worst-offending RNA bonds in a built extended
//! chain.  Reports each (atom-type-pair, count, mean |r − r₀|,
//! total energy contribution).  Lets us see which `rna_ic` builder
//! constants are systematically off from CHARMM27's r₀.
//!
//! Run with: `cargo test -p io --release --test rna_bond_audit -- --nocapture`

use chem::{Nucleotide, classify_atom, standard_ff};
use geom::{build_extended_rna_chain, build_topology_graph};
use std::collections::BTreeMap;

#[test]
fn audit_extended_chain_bond_lengths() {
    // Use the same 14-mer sequence as the UUCG acceptance test —
    // covers G/C/A/U and includes inter-residue phosphodiesters.
    let nts: Vec<Nucleotide> = "GGCACUUCGGUGCC"
        .chars()
        .map(|c| Nucleotide::from_one_letter(c).unwrap())
        .collect();
    let s = build_extended_rna_chain(&nts).expect("build extended");
    let graph = build_topology_graph(&s);
    let ff = standard_ff();

    // Flatten atom positions + atom types.
    let positions: Vec<geom::Vec3> = s
        .residues
        .iter()
        .flat_map(|r| r.atoms.iter().map(|a| a.position))
        .collect();
    let atom_types: Vec<chem::AtomType> = s
        .residues
        .iter()
        .flat_map(|r| {
            r.atoms
                .iter()
                .map(move |a| classify_atom(r.monomer, a.name).unwrap())
        })
        .collect();

    #[derive(Default)]
    struct Stats {
        count: usize,
        sum_deviation: f64,
        sum_energy_kj: f64,
    }
    let mut per_pair: BTreeMap<(chem::AtomType, chem::AtomType), Stats> = BTreeMap::new();
    let kcal_to_kj = 4.184;

    let mut total_kj = 0.0;
    let mut missing = 0usize;
    let mut worst_individual: Vec<(f64, String)> = Vec::new();

    for b in &graph.bonds {
        let (ta, tb) = (atom_types[b.a], atom_types[b.b]);
        let p = match ff.bond(ta, tb) {
            Some(p) => p,
            None => {
                missing += 1;
                continue;
            }
        };
        let r = (positions[b.a] - positions[b.b]).norm();
        let dev = r - p.r0;
        // CHARMM convention: V = K (r − r₀)² (no ½ prefactor).
        let energy_kj = p.k * dev * dev * kcal_to_kj;
        total_kj += energy_kj;
        let key = if (ta as u8) <= (tb as u8) {
            (ta, tb)
        } else {
            (tb, ta)
        };
        let s = per_pair.entry(key).or_default();
        s.count += 1;
        s.sum_deviation += dev.abs();
        s.sum_energy_kj += energy_kj;
        // Track top-10 individually for spot-checks.
        let label = format!(
            "{:?}-{:?}: r={:.3} r0={:.3} k={:.1} → {:.2} kJ/mol",
            ta, tb, r, p.r0, p.k, energy_kj
        );
        worst_individual.push((energy_kj, label));
    }
    worst_individual.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    eprintln!(
        "\n=== RNA bond audit on extended GGCACUUCGGUGCC ({} bonds, missing FF: {}) ===",
        graph.bonds.len(),
        missing
    );
    eprintln!("Total bond energy: {:.1} kJ/mol", total_kj);
    eprintln!("\nPer (atom-type-pair) summary, sorted by total contribution:");
    eprintln!(
        "{:>4} {:>20} {:>10} {:>10} {:>12}",
        "n", "pair", "mean |Δr|", "total kJ", "per-bond"
    );
    let mut rows: Vec<_> = per_pair.iter().collect();
    rows.sort_by(|a, b| b.1.sum_energy_kj.partial_cmp(&a.1.sum_energy_kj).unwrap());
    for (pair, stats) in rows.iter().take(20) {
        eprintln!(
            "{:>4} {:>9?}-{:>9?} {:>10.3} {:>10.1} {:>12.1}",
            stats.count,
            pair.0,
            pair.1,
            stats.sum_deviation / stats.count as f64,
            stats.sum_energy_kj,
            stats.sum_energy_kj / stats.count as f64,
        );
    }
    eprintln!("\nTop 10 individual offenders:");
    for (_, label) in worst_individual.iter().take(10) {
        eprintln!("  {}", label);
    }
}
