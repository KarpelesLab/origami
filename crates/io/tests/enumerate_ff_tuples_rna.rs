//! Acceptance: every bonded tuple in a built RNA chain has CHARMM27
//! nucleic-acid parameters, and every placed atom has a partial charge.
//!
//! Mirror of `enumerate_ff_tuples.rs` but for the RNA side of the FF
//! data layer. If this fails, the corresponding bonded term would be
//! silently skipped by the energy code once it learns to walk
//! `Monomer::Rna` residues.

use std::collections::BTreeSet;

use chem::{AtomType, Nucleotide, classify_rna, standard_ff};
use geom::{build_extended_rna_chain, build_topology_graph};

#[test]
fn every_rna_force_field_tuple_has_parameters() {
    // One of every canonical ribonucleotide.
    let seq = [
        Nucleotide::Adenine,
        Nucleotide::Uracil,
        Nucleotide::Guanine,
        Nucleotide::Cytosine,
    ];
    let s = build_extended_rna_chain(&seq).unwrap();
    let g = build_topology_graph(&s);
    let ff = standard_ff();

    // Per-atom CHARMM27 type vector keyed by global atom index.
    let mut atom_types: Vec<AtomType> = Vec::with_capacity(s.atom_count());
    for residue in &s.residues {
        let nt = residue.monomer.as_nucleotide().expect("RNA residue");
        for atom in &residue.atoms {
            atom_types.push(
                classify_rna(nt, atom.name)
                    .unwrap_or_else(|| panic!("classify_rna missing {nt:?} {}", atom.name)),
            );
        }
    }

    let mut missing_bonds: Vec<(AtomType, AtomType)> = Vec::new();
    for b in &g.bonds {
        let (ta, tb) = (atom_types[b.a], atom_types[b.b]);
        if ff.bond(ta, tb).is_none() {
            missing_bonds.push((ta, tb));
        }
    }
    let mut missing_angles: Vec<(AtomType, AtomType, AtomType)> = Vec::new();
    for ang in &g.angles {
        let (ta, tb, tc) = (atom_types[ang.a], atom_types[ang.b], atom_types[ang.c]);
        if ff.angle(ta, tb, tc).is_none() {
            missing_angles.push((ta, tb, tc));
        }
    }
    let mut missing_dihedrals: Vec<(AtomType, AtomType, AtomType, AtomType)> = Vec::new();
    for d in &g.dihedrals {
        let (ta, tb, tc, td) = (
            atom_types[d.a],
            atom_types[d.b],
            atom_types[d.c],
            atom_types[d.d],
        );
        if ff.dihedral(ta, tb, tc, td).is_none() {
            missing_dihedrals.push((ta, tb, tc, td));
        }
    }
    let mut missing_impropers: Vec<(AtomType, AtomType, AtomType, AtomType)> = Vec::new();
    for imp in &g.impropers {
        let (ta, tb, tc, td) = (
            atom_types[imp.a],
            atom_types[imp.b],
            atom_types[imp.c],
            atom_types[imp.d],
        );
        if ff.improper(ta, tb, tc, td).is_none() {
            missing_impropers.push((ta, tb, tc, td));
        }
    }
    let mut missing_nonbonded: BTreeSet<AtomType> = BTreeSet::new();
    for &t in &atom_types {
        if ff.nonbonded(t).is_none() {
            missing_nonbonded.insert(t);
        }
    }
    let mut missing_charges: Vec<(Nucleotide, String)> = Vec::new();
    for residue in &s.residues {
        let nt = residue.monomer.as_nucleotide().unwrap();
        for atom in &residue.atoms {
            if ff.partial_charge_rna(nt, atom.name).is_none() {
                missing_charges.push((nt, atom.name.to_string()));
            }
        }
    }
    missing_bonds.sort();
    missing_bonds.dedup();
    missing_angles.sort();
    missing_angles.dedup();
    missing_dihedrals.sort();
    missing_dihedrals.dedup();
    missing_impropers.sort();
    missing_impropers.dedup();
    missing_charges.sort();
    missing_charges.dedup();

    if !missing_bonds.is_empty()
        || !missing_angles.is_empty()
        || !missing_dihedrals.is_empty()
        || !missing_impropers.is_empty()
        || !missing_nonbonded.is_empty()
        || !missing_charges.is_empty()
    {
        eprintln!("\n--- MISSING RNA force-field parameters ---");
        for x in &missing_bonds {
            eprintln!("  bond     {:?} - {:?}", x.0, x.1);
        }
        for x in &missing_angles {
            eprintln!("  angle    {:?} - {:?} - {:?}", x.0, x.1, x.2);
        }
        for x in &missing_dihedrals {
            eprintln!("  dihedral {:?} - {:?} - {:?} - {:?}", x.0, x.1, x.2, x.3);
        }
        for x in &missing_impropers {
            eprintln!("  improper {:?} - {:?} - {:?} - {:?}", x.0, x.1, x.2, x.3);
        }
        for t in &missing_nonbonded {
            eprintln!("  nonbond  {:?}", t);
        }
        for x in &missing_charges {
            eprintln!("  charge   {:?} {}", x.0, x.1);
        }
        panic!(
            "missing RNA parameters: {} bonds, {} angles, {} dihedrals, \
             {} impropers, {} nonbonded, {} charges",
            missing_bonds.len(),
            missing_angles.len(),
            missing_dihedrals.len(),
            missing_impropers.len(),
            missing_nonbonded.len(),
            missing_charges.len(),
        );
    }
}
