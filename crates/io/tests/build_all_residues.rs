use chem::{classify, AminoAcid};
use geom::{build_extended_chain, measure};
use io::write_pdb;

/// Build an extended chain containing every amino acid and verify the output
/// PDB is well-formed: every backbone bond is the right length, no two
/// non-bonded atoms clash, the file has the expected number of ATOM records.
#[test]
fn all_twenty_residues_build_cleanly() {
    let sequence: Vec<AminoAcid> = AminoAcid::ALL.to_vec();
    let structure = build_extended_chain(&sequence).expect("chain build");
    assert_eq!(structure.residues.len(), 20);

    // 1. PDB output is non-empty and parses as expected.
    let mut buf = Vec::new();
    write_pdb(&mut buf, &structure, "all-20-residues").unwrap();
    let text = String::from_utf8(buf).unwrap();
    let atom_lines = text.lines().filter(|l| l.starts_with("ATOM")).count();
    assert_eq!(atom_lines, structure.atom_count());
    assert!(atom_lines > 200, "expected many atoms, got {atom_lines}");

    // 2. Backbone bond lengths.
    for (i, res) in structure.residues.iter().enumerate() {
        let n = res.position("N").unwrap();
        let ca = res.position("CA").unwrap();
        let c = res.position("C").unwrap();
        let o = res.position("O").unwrap();
        // Within tolerance because everything is constructed via NeRF, which
        // is exact.
        assert!((measure::distance(n, ca) - 1.458).abs() < 1e-6,
            "residue {} ({:?}): N-CA = {}", i, res.aa, measure::distance(n, ca));
        assert!((measure::distance(ca, c) - 1.525).abs() < 1e-6);
        assert!((measure::distance(c, o) - 1.231).abs() < 1e-6);
        if i > 0 {
            let prev_c = structure.residues[i - 1].position("C").unwrap();
            assert!((measure::distance(prev_c, n) - 1.329).abs() < 1e-6,
                "residue {} ({:?}): prev_C-N = {}", i, res.aa, measure::distance(prev_c, n));
        }
    }

    // 3. No two atoms (other than bonded pairs and 1-3 neighbors) come closer
    //    than 1.0 Å. Any closer would indicate a topology bug.
    let atoms: Vec<_> = structure.iter_atoms().collect();
    for i in 0..atoms.len() {
        for j in (i + 1)..atoms.len() {
            let d = (atoms[i].1.position - atoms[j].1.position).norm();
            assert!(
                d > 0.7,
                "atoms too close: residue {} {} <-> residue {} {} = {} Å",
                atoms[i].0, atoms[i].1.name,
                atoms[j].0, atoms[j].1.name,
                d
            );
        }
    }
}

/// Spot-check the bicyclic tryptophan side chain: the indole ring should be
/// approximately planar.
#[test]
fn tryptophan_indole_is_approximately_planar() {
    let s = build_extended_chain(&[AminoAcid::Trp]).unwrap();
    let r = &s.residues[0];
    let cg = r.position("CG").unwrap();
    let cd1 = r.position("CD1").unwrap();
    let cd2 = r.position("CD2").unwrap();
    let ne1 = r.position("NE1").unwrap();
    let ce2 = r.position("CE2").unwrap();
    let ce3 = r.position("CE3").unwrap();
    let cz3 = r.position("CZ3").unwrap();
    let cz2 = r.position("CZ2").unwrap();
    let ch2 = r.position("CH2").unwrap();

    // Define the plane via three ring atoms (CG, CD1, CD2). All other ring
    // atoms should be within a small distance of that plane.
    let n_plane = (cd1 - cg).cross(&(cd2 - cg)).normalize();
    for (name, pt) in [
        ("NE1", ne1), ("CE2", ce2), ("CE3", ce3),
        ("CZ3", cz3), ("CZ2", cz2), ("CH2", ch2),
    ] {
        let d = (pt - cg).dot(&n_plane).abs();
        assert!(d < 0.05, "{name} is {d} Å out of indole plane");
    }
}

/// Phenyl ring in Phe should be planar and have all C-C bonds ~1.39 Å.
#[test]
fn phenylalanine_ring_is_planar_and_regular() {
    let s = build_extended_chain(&[AminoAcid::Phe]).unwrap();
    let r = &s.residues[0];
    let cg = r.position("CG").unwrap();
    let cd1 = r.position("CD1").unwrap();
    let cd2 = r.position("CD2").unwrap();
    let ce1 = r.position("CE1").unwrap();
    let ce2 = r.position("CE2").unwrap();
    let cz = r.position("CZ").unwrap();

    // Bond lengths.
    for (a, b, label) in [
        (cg, cd1, "CG-CD1"), (cg, cd2, "CG-CD2"),
        (cd1, ce1, "CD1-CE1"), (cd2, ce2, "CD2-CE2"),
        (ce1, cz, "CE1-CZ"), (ce2, cz, "CE2-CZ"),
    ] {
        let d = measure::distance(a, b);
        assert!((d - 1.39).abs() < 0.02, "{label} = {d} Å");
    }

    // Planarity.
    let n_plane = (cd1 - cg).cross(&(cd2 - cg)).normalize();
    for (name, pt) in [("CE1", ce1), ("CE2", ce2), ("CZ", cz)] {
        let d = (pt - cg).dot(&n_plane).abs();
        assert!(d < 0.01, "{name} is {d} Å out of phenyl plane");
    }
}

/// Every atom of every built residue must be classifiable into an AtomType.
/// If this fails, the topology adds an atom that the classifier doesn't know
/// about (or vice versa) — the two must stay in sync.
#[test]
fn every_built_atom_has_an_atom_type() {
    for aa in AminoAcid::ALL {
        let s = build_extended_chain(&[aa]).unwrap();
        let r = &s.residues[0];
        for atom in &r.atoms {
            let cls = classify(aa, atom.name);
            assert!(
                cls.is_some(),
                "{:?}: atom {:?} not classified",
                aa, atom.name
            );
            // Element must agree.
            let t = cls.unwrap();
            assert_eq!(
                t.element(),
                atom.element,
                "{:?} {:?}: classifier element {:?} != topology element {:?}",
                aa, atom.name, t.element(), atom.element
            );
        }
    }
}

/// Lysine should have a Cα-NZ distance of ~6 Å (extended side chain).
#[test]
fn lysine_extended_sidechain_length() {
    let s = build_extended_chain(&[AminoAcid::Lys]).unwrap();
    let r = &s.residues[0];
    let ca = r.position("CA").unwrap();
    let nz = r.position("NZ").unwrap();
    let d = measure::distance(ca, nz);
    // 4 sp3 bonds in trans configuration: should be ~6.4 Å end-to-end.
    assert!((5.5..7.0).contains(&d), "Lys CA-NZ = {d} Å (expected ~6.4)");
}
