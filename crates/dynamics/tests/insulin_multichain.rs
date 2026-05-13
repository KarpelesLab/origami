//! Insulin (PDB 2HIU, 51 residues, two chains A+B) is the smallest
//! standard test for multi-chain support and inter-chain disulfide
//! bonds. Three checks:
//!
//!   1. The PDB reader picks up both chains. Earlier single-chain
//!      code stopped at the first TER record and only returned chain A.
//!
//!   2. `geom::build_topology_graph` detects all three disulfide
//!      bridges, two of which are inter-chain (A7-B7 and A20-B19) and
//!      so depend on the geometric detection working across the chain
//!      boundary.
//!
//!   3. `build_topology_graph` does NOT auto-bond the last residue of
//!      chain A to the first residue of chain B with a phantom peptide
//!      bond. With one, the bond term would explode (the C-N
//!      separation across chains is far larger than 1.33 Å) and MD
//!      would diverge in a few fs.

use chem::AminoAcid;
use geom::build_topology_graph;
use io::read_pdb;

fn read_fixture(path: &str) -> geom::Structure {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    read_pdb(bytes.as_slice()).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

#[test]
fn insulin_has_two_chains() {
    let s = read_fixture("../io/tests/fixtures/2HIU_insulin.pdb");
    // Chain A is residues 1-21, chain B is residues 22-51 (residues are
    // stored in PDB order; insulin's chain B starts after chain A's
    // TER record).
    let chain_a_count = s.residues.iter().filter(|r| r.chain == 'A').count();
    let chain_b_count = s.residues.iter().filter(|r| r.chain == 'B').count();
    assert_eq!(chain_a_count, 21, "expected 21 residues in chain A");
    assert_eq!(chain_b_count, 30, "expected 30 residues in chain B");
}

#[test]
fn insulin_three_disulfides_detected() {
    let s = read_fixture("../io/tests/fixtures/2HIU_insulin.pdb");
    let g = build_topology_graph(&s);

    // Find SG indices per cysteine in residue order.
    let mut sg_indices: Vec<(usize, char, usize)> = Vec::new();
    let mut atom_idx = 0usize;
    for (ri, r) in s.residues.iter().enumerate() {
        let mut sg = None;
        for a in &r.atoms {
            if a.name == "SG" {
                sg = Some(atom_idx);
            }
            atom_idx += 1;
        }
        if r.monomer.as_amino_acid() == Some(AminoAcid::Cys) {
            if let Some(idx) = sg {
                sg_indices.push((ri + 1, r.chain, idx));
            }
        }
    }
    assert_eq!(sg_indices.len(), 6, "insulin has 6 cysteines");

    let mut bridges: Vec<(usize, usize)> = Vec::new();
    for (i, (ri, _, si)) in sg_indices.iter().enumerate() {
        for (rj, _, sj) in sg_indices.iter().skip(i + 1) {
            if g.is_bonded(*si, *sj) {
                bridges.push((*ri, *rj));
            }
        }
    }
    bridges.sort();
    // In our flat residue numbering: A6=6, A11=11, A7=7, A20=20,
    // B7 = 21+7 = 28, B19 = 21+19 = 40.
    let expected: Vec<(usize, usize)> = vec![(6, 11), (7, 28), (20, 40)];
    assert_eq!(
        bridges, expected,
        "expected insulin's three disulfides A6-A11, A7-B7, A20-B19; \
         detected {bridges:?}"
    );
}

#[test]
fn no_phantom_peptide_bond_between_chains() {
    // A21's C and B1's N are separated in the native fold by more than
    // 1.33 Å (the peptide-bond r₀). If `build_topology_graph` mistook
    // them for the same chain it would add a phantom C-N bond and the
    // harmonic bond force would explode.
    let s = read_fixture("../io/tests/fixtures/2HIU_insulin.pdb");
    let g = build_topology_graph(&s);

    // Find chain-A's last residue (residue 20 in 0-indexed; A21 in
    // PDB numbering) and chain-B's first residue (residue 21 in
    // 0-indexed; B1 in PDB numbering).
    let mut chain_a_last = None;
    let mut chain_b_first = None;
    let mut atom_idx = 0usize;
    for (ri, r) in s.residues.iter().enumerate() {
        if r.chain == 'A' {
            chain_a_last = Some((ri, atom_idx));
        }
        if r.chain == 'B' && chain_b_first.is_none() {
            chain_b_first = Some((ri, atom_idx));
        }
        atom_idx += r.atoms.len();
    }
    let (a_last, a_atom_start) = chain_a_last.expect("chain A present");
    let (b_first, b_atom_start) = chain_b_first.expect("chain B present");

    // Find A21's C (last backbone carbonyl) and B1's N.
    let a21_c = (0..s.residues[a_last].atoms.len())
        .find(|&i| s.residues[a_last].atoms[i].name == "C")
        .map(|i| a_atom_start + i)
        .expect("A21 has C atom");
    let b1_n = (0..s.residues[b_first].atoms.len())
        .find(|&i| s.residues[b_first].atoms[i].name == "N")
        .map(|i| b_atom_start + i)
        .expect("B1 has N atom");

    assert!(
        !g.is_bonded(a21_c, b1_n),
        "phantom peptide bond between chain A and chain B"
    );
}
