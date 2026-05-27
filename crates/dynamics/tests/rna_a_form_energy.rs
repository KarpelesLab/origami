//! Acceptance test: the A-form RNA builder produces a starting
//! geometry with substantially lower CHARMM27 energy than the
//! extended-NeRF builder.  Locks in the win the A-form torsion set
//! gives us as an RNA dynamics starting point.

use chem::{standard_ff, Nucleotide};
use dynamics::energy_eval::total_energy;
use energy::bonded::bonded_energy;
use energy::gb::gb_energy;
use energy::nonbonded::{nonbonded_energy, DEFAULT_CUTOFF_A};
use geom::{build_a_form_rna_chain, build_extended_rna_chain, build_topology_graph};

fn breakdown(label: &str, s: &geom::Structure, g: &geom::TopologyGraph, ff: &chem::ForceField) {
    let b = bonded_energy(s, g, ff);
    let nb = nonbonded_energy(s, g, ff, DEFAULT_CUTOFF_A);
    let gb = gb_energy(s, ff);
    eprintln!(
        "{label}: bond={:.0} angle={:.0} dih={:.0} imp={:.0} LJ={:.0} Coul={:.0} GB={:.0}",
        b.bond_kj_mol, b.angle_kj_mol, b.dihedral_kj_mol, b.improper_kj_mol,
        nb.lj_kj_mol, nb.coulomb_kj_mol, gb.gb_kj_mol,
    );
}

#[test]
fn a_form_rna_energy_below_extended() {
    let seq = [
        Nucleotide::Uracil, Nucleotide::Cytosine,
        Nucleotide::Adenine, Nucleotide::Guanine,
    ];

    let s_ext = build_extended_rna_chain(&seq).unwrap();
    let s_a = build_a_form_rna_chain(&seq).unwrap();
    let g_ext = build_topology_graph(&s_ext);
    let g_a = build_topology_graph(&s_a);
    let ff = standard_ff();

    let e_ext = total_energy(&s_ext, &g_ext, ff);
    let e_a = total_energy(&s_a, &g_a, ff);

    eprintln!("UCAG extended-NeRF energy: {e_ext:.1} kJ/mol");
    eprintln!("UCAG A-form        energy: {e_a:.1} kJ/mol");
    breakdown("  extended", &s_ext, &g_ext, ff);
    breakdown("  A-form  ", &s_a, &g_a, ff);

    // Dump the closest non-bonded pairs in A-form to localise clashes.
    let atoms: Vec<_> = s_a.iter_atoms().collect();
    let bonded: std::collections::HashSet<(usize, usize)> = g_a.bonds.iter()
        .flat_map(|b| [(b.a, b.b), (b.b, b.a)])
        .chain(g_a.angles.iter().flat_map(|a| [(a.a, a.c), (a.c, a.a)]))
        .collect();
    let mut close: Vec<(f64, String, String)> = Vec::new();
    for i in 0..atoms.len() {
        for j in (i + 1)..atoms.len() {
            if bonded.contains(&(i, j)) { continue; }
            let (ri, ai) = atoms[i];
            let (rj, aj) = atoms[j];
            let d = (ai.position - aj.position).norm();
            if d < 1.5 {
                close.push((d,
                    format!("{ri}/{}", ai.name),
                    format!("{rj}/{}", aj.name),
                ));
            }
        }
    }
    close.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    eprintln!("A-form: closest non-bonded pairs (1-2/1-3 excluded):");
    for (d, a, b) in close.iter().take(10) {
        eprintln!("    {a} ↔ {b} = {d:.2} Å");
    }
    eprintln!("Δ (A-form - extended)     : {:.1} kJ/mol", e_a - e_ext);

    assert!(e_a.is_finite() && e_ext.is_finite());
    assert!(
        e_a < e_ext,
        "A-form should start lower than extended: extended={e_ext}, A-form={e_a}",
    );
}
