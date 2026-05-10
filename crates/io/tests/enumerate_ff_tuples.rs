//! Diagnostic: list every distinct atom-type tuple that appears in the
//! bonded graph of a chain containing one of every amino acid.
//!
//! The output drives the size of the M3c CHARMM36 parameter tables — we
//! only need to populate parameters for tuples that actually occur.

use std::collections::BTreeSet;

use chem::{classify, standard_ff, AminoAcid, AtomType};
use geom::{build_extended_chain, build_topology_graph};

/// Run with: `cargo test -p io --test enumerate_ff_tuples -- --nocapture`
#[test]
fn enumerate_required_force_field_tuples() {
    let seq: Vec<AminoAcid> = AminoAcid::ALL.to_vec();
    let s = build_extended_chain(&seq).unwrap();
    let g = build_topology_graph(&s);

    // Compute a flat parallel atom-type vector keyed by global index.
    let mut atom_types: Vec<AtomType> = Vec::with_capacity(s.atom_count());
    for residue in &s.residues {
        for atom in &residue.atoms {
            let t = classify(residue.aa, atom.name)
                .unwrap_or_else(|| panic!("classifier missing {:?} {:?}", residue.aa, atom.name));
            atom_types.push(t);
        }
    }

    let canonical_pair = |a: AtomType, b: AtomType| if (a as u8) <= (b as u8) { (a, b) } else { (b, a) };
    let canonical_triple = |a: AtomType, b: AtomType, c: AtomType| {
        // Central atom b is fixed; canonicalise a/c so the smaller-discriminant comes first.
        if (a as u8) <= (c as u8) { (a, b, c) } else { (c, b, a) }
    };
    let canonical_quad = |a: AtomType, b: AtomType, c: AtomType, d: AtomType| {
        // Central pair (b,c) determines orientation. If (b,c) > (c,b) by canonical
        // ordering on (b<c), reverse.
        if (b as u8, a as u8) <= (c as u8, d as u8) {
            (a, b, c, d)
        } else {
            (d, c, b, a)
        }
    };

    let mut bonds: BTreeSet<(AtomType, AtomType)> = BTreeSet::new();
    for b in &g.bonds {
        bonds.insert(canonical_pair(atom_types[b.a], atom_types[b.b]));
    }

    let mut angles: BTreeSet<(AtomType, AtomType, AtomType)> = BTreeSet::new();
    for ang in &g.angles {
        angles.insert(canonical_triple(
            atom_types[ang.a], atom_types[ang.b], atom_types[ang.c],
        ));
    }

    let mut dihedrals: BTreeSet<(AtomType, AtomType, AtomType, AtomType)> = BTreeSet::new();
    for d in &g.dihedrals {
        dihedrals.insert(canonical_quad(
            atom_types[d.a], atom_types[d.b], atom_types[d.c], atom_types[d.d],
        ));
    }

    let mut impropers: BTreeSet<(AtomType, AtomType, AtomType, AtomType)> = BTreeSet::new();
    for imp in &g.impropers {
        // Impropers have a fixed central; canonicalise off-centres a, c, d.
        let mut off = [atom_types[imp.a], atom_types[imp.c], atom_types[imp.d]];
        off.sort_by_key(|t| *t as u8);
        impropers.insert((off[0], atom_types[imp.b], off[1], off[2]));
    }

    println!("\n========================================================");
    println!("Force-field parameter tuples needed for M3c:");
    println!("  atom types in use: {}", count_distinct_types(&atom_types));
    println!("  unique bond pairs:     {}", bonds.len());
    println!("  unique angle triples:  {}", angles.len());
    println!("  unique dihedral quads: {}", dihedrals.len());
    println!("  unique improper quads: {}", impropers.len());
    println!("========================================================\n");

    println!("BOND TUPLES ({}):", bonds.len());
    for (a, b) in &bonds {
        println!("  {:?} - {:?}", a, b);
    }

    println!("\nANGLE TUPLES ({}):", angles.len());
    for (a, b, c) in &angles {
        println!("  {:?} - {:?} - {:?}", a, b, c);
    }

    println!("\nDIHEDRAL TUPLES ({}):", dihedrals.len());
    for (a, b, c, d) in &dihedrals {
        println!("  {:?} - {:?} - {:?} - {:?}", a, b, c, d);
    }

    println!("\nIMPROPER TUPLES ({}):", impropers.len());
    for (a, b, c, d) in &impropers {
        println!("  {:?} - {:?} - {:?} - {:?}", a, b, c, d);
    }
}

fn count_distinct_types(types: &[AtomType]) -> usize {
    let mut s = BTreeSet::new();
    for &t in types {
        s.insert(t as u8);
    }
    s.len()
}

/// Acceptance check: for every distinct atom-type tuple appearing in the
/// bonded graph of all 20 residues, the CHARMM36m parameter file provides
/// a parameter (or wildcard fallback). If this fails, M3d's bonded energy
/// term will silently skip those tuples — exactly the kind of thing we
/// want to catch up front.
#[test]
fn every_force_field_tuple_has_parameters() {
    let seq: Vec<AminoAcid> = AminoAcid::ALL.to_vec();
    let s = build_extended_chain(&seq).unwrap();
    let g = build_topology_graph(&s);
    let ff = standard_ff();

    let mut atom_types: Vec<AtomType> = Vec::with_capacity(s.atom_count());
    for residue in &s.residues {
        for atom in &residue.atoms {
            atom_types.push(classify(residue.aa, atom.name).unwrap());
        }
    }

    let mut missing_bonds = Vec::new();
    for b in &g.bonds {
        let (ta, tb) = (atom_types[b.a], atom_types[b.b]);
        if ff.bond(ta, tb).is_none() {
            missing_bonds.push((ta, tb));
        }
    }
    missing_bonds.sort();
    missing_bonds.dedup();

    let mut missing_angles = Vec::new();
    for ang in &g.angles {
        let (ta, tb, tc) = (atom_types[ang.a], atom_types[ang.b], atom_types[ang.c]);
        if ff.angle(ta, tb, tc).is_none() {
            missing_angles.push((ta, tb, tc));
        }
    }
    missing_angles.sort();
    missing_angles.dedup();

    let mut missing_dihedrals = Vec::new();
    for d in &g.dihedrals {
        let (ta, tb, tc, td) = (
            atom_types[d.a], atom_types[d.b], atom_types[d.c], atom_types[d.d]
        );
        if ff.dihedral(ta, tb, tc, td).is_none() {
            missing_dihedrals.push((ta, tb, tc, td));
        }
    }
    missing_dihedrals.sort();
    missing_dihedrals.dedup();

    let mut missing_impropers = Vec::new();
    for imp in &g.impropers {
        let (ta, tb, tc, td) = (
            atom_types[imp.a], atom_types[imp.b], atom_types[imp.c], atom_types[imp.d]
        );
        if ff.improper(ta, tb, tc, td).is_none() {
            missing_impropers.push((ta, tb, tc, td));
        }
    }
    missing_impropers.sort();
    missing_impropers.dedup();

    let mut missing_nonbonded = Vec::new();
    for &t in &atom_types {
        if ff.nonbonded(t).is_none() {
            missing_nonbonded.push(t);
        }
    }
    missing_nonbonded.sort();
    missing_nonbonded.dedup();

    // Partial charges: every (residue, atom-name) pair should have one.
    let mut missing_charges: Vec<(AminoAcid, String)> = Vec::new();
    for residue in &s.residues {
        for atom in &residue.atoms {
            if ff.partial_charge(residue.aa, atom.name).is_none() {
                missing_charges.push((residue.aa, atom.name.to_string()));
            }
        }
    }
    missing_charges.sort();
    missing_charges.dedup();

    if !missing_bonds.is_empty()
        || !missing_angles.is_empty()
        || !missing_dihedrals.is_empty()
        || !missing_impropers.is_empty()
        || !missing_nonbonded.is_empty()
        || !missing_charges.is_empty()
    {
        eprintln!("\n--- MISSING force-field parameters ---");
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
        for x in &missing_nonbonded {
            eprintln!("  nonbond  {:?}", x);
        }
        for x in &missing_charges {
            eprintln!("  charge   {:?} {}", x.0, x.1);
        }
        panic!(
            "Missing: {} bonds, {} angles, {} dihedrals, {} impropers, {} nonbonded, {} charges",
            missing_bonds.len(), missing_angles.len(),
            missing_dihedrals.len(), missing_impropers.len(),
            missing_nonbonded.len(), missing_charges.len()
        );
    }
}
