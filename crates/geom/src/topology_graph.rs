//! Bonded-connectivity graph for a built `Structure`.
//!
//! The bond graph is the source of truth for the bonded force-field terms
//! (bond stretching, angle bending, dihedral torsion, improper) and for the
//! 1-2 / 1-3 / 1-4 exclusion masks used by the non-bonded code.
//!
//! Bonds come from three sources:
//! 1. The standard backbone within each residue (N-CA, CA-C, C-O, etc.).
//! 2. The peptide bond between consecutive residues (C(i)-N(i+1)).
//! 3. Each side-chain atom's `bond_to` parent in the chem topology table.
//!
//! Plus two special cases:
//! - The Pro ring closure: an additional N-Cδ bond not encoded as `bond_to`
//!   in the side-chain table.
//! - The peptide bond N-H amide hydrogen, which is in the residue's atom
//!   list but isn't placed off the side chain.
//!
//! Once the bond graph is known, angles / dihedrals / impropers are derived
//! by enumerating connected paths.

use std::collections::{HashMap, HashSet};

use chem::AminoAcid;

use crate::structure::Structure;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Bond {
    pub a: usize,
    pub b: usize,
}

impl Bond {
    fn new(a: usize, b: usize) -> Self {
        if a < b { Bond { a, b } } else { Bond { a: b, b: a } }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Angle {
    pub a: usize,
    pub b: usize, // central atom
    pub c: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Dihedral {
    pub a: usize,
    pub b: usize,
    pub c: usize,
    pub d: usize,
}

/// An improper torsion. `b` is the central sp² atom; the planarity is
/// enforced on the dihedral a-b-c-d via the harmonic ω restraint.
#[derive(Debug, Clone, Copy)]
pub struct Improper {
    pub a: usize,
    pub b: usize, // central
    pub c: usize,
    pub d: usize,
}

#[derive(Debug, Clone)]
pub struct TopologyGraph {
    pub bonds: Vec<Bond>,
    pub angles: Vec<Angle>,
    pub dihedrals: Vec<Dihedral>,
    pub impropers: Vec<Improper>,
    /// Adjacency list: for each global atom index, the indices of all atoms
    /// directly bonded to it. Useful for excluded-pair masks and neighbour
    /// walks.
    pub bonded_to: Vec<Vec<usize>>,
}

impl TopologyGraph {
    /// 1-2 exclusion: are atoms `a` and `b` directly bonded?
    pub fn is_bonded(&self, a: usize, b: usize) -> bool {
        self.bonded_to[a].contains(&b)
    }

    /// 1-3 exclusion: do `a` and `b` share a common bonded neighbour?
    pub fn is_one_three(&self, a: usize, b: usize) -> bool {
        if a == b {
            return false;
        }
        for &n in &self.bonded_to[a] {
            if self.bonded_to[n].contains(&b) {
                return true;
            }
        }
        false
    }

    /// 1-4 connection: connected by exactly three bonds (used to apply
    /// scaled non-bonded interactions).
    pub fn is_one_four(&self, a: usize, b: usize) -> bool {
        if a == b || self.is_bonded(a, b) || self.is_one_three(a, b) {
            return false;
        }
        for &n1 in &self.bonded_to[a] {
            for &n2 in &self.bonded_to[n1] {
                if self.bonded_to[n2].contains(&b) {
                    return true;
                }
            }
        }
        false
    }
}

/// Bonds that close rings, not encoded in the linear `bond_to` parent
/// chain that the side-chain templates use. Each entry is `(atom_a, atom_b)`.
fn ring_closure_bonds(aa: AminoAcid) -> &'static [(&'static str, &'static str)] {
    match aa {
        // Proline: 5-ring N-CA-CB-CG-CD-N. CD's bond_to is CG; we add N-CD.
        AminoAcid::Pro => &[("N", "CD")],
        // Phenyl ring closes at the para position.
        AminoAcid::Phe => &[("CE2", "CZ")],
        // Tyrosine: same phenyl ring as Phe (OH bonds to CZ via topology).
        AminoAcid::Tyr => &[("CE2", "CZ")],
        // Histidine imidazole 5-ring closes between CE1 and NE2.
        AminoAcid::His => &[("CE1", "NE2")],
        // Tryptophan indole: 5-ring closure CE2-CD2; 6-ring closure CH2-CZ3.
        AminoAcid::Trp => &[("CE2", "CD2"), ("CH2", "CZ3")],
        _ => &[],
    }
}

/// Build the bonded-connectivity graph for a Structure.
pub fn build_topology_graph(structure: &Structure) -> TopologyGraph {
    // Map (residue_index, atom_name) → global atom index.
    let mut atom_idx: HashMap<(usize, &str), usize> = HashMap::new();
    let mut total = 0;
    for (ri, res) in structure.residues.iter().enumerate() {
        for atom in &res.atoms {
            atom_idx.insert((ri, atom.name), total);
            total += 1;
        }
    }
    let lookup = |ri: usize, name: &str| -> Option<usize> {
        atom_idx.get(&(ri, name)).copied()
    };

    // Collect bonds as a deduplicated set, then sort for stable iteration.
    let mut bonds: HashSet<Bond> = HashSet::new();
    let add_bond = |bonds: &mut HashSet<Bond>, a: usize, b: usize| {
        if a != b {
            bonds.insert(Bond::new(a, b));
        }
    };

    for (ri, res) in structure.residues.iter().enumerate() {
        let aa = res.aa;

        // ---- Backbone bonds ----
        let n = lookup(ri, "N");
        let ca = lookup(ri, "CA");
        let c = lookup(ri, "C");
        let o = lookup(ri, "O");
        if let (Some(n), Some(ca)) = (n, ca) {
            add_bond(&mut bonds, n, ca);
        }
        if let (Some(ca), Some(c)) = (ca, c) {
            add_bond(&mut bonds, ca, c);
        }
        if let (Some(c), Some(o)) = (c, o) {
            add_bond(&mut bonds, c, o);
        }
        if let (Some(n), Some(h)) = (n, lookup(ri, "H")) {
            add_bond(&mut bonds, n, h);
        }
        if aa == AminoAcid::Gly {
            for ha in ["HA2", "HA3"] {
                if let (Some(ca), Some(hi)) = (ca, lookup(ri, ha)) {
                    add_bond(&mut bonds, ca, hi);
                }
            }
        } else if let (Some(ca), Some(ha)) = (ca, lookup(ri, "HA")) {
            add_bond(&mut bonds, ca, ha);
        }

        // ---- Inter-residue peptide bond C(i-1) -- N(i) ----
        if ri > 0 {
            let prev_c = lookup(ri - 1, "C");
            if let (Some(prev_c), Some(n)) = (prev_c, n) {
                add_bond(&mut bonds, prev_c, n);
            }
        }

        // ---- Side-chain bonds (each side-chain atom declares its parent) ----
        for sc in aa.topology().sidechain {
            let child = lookup(ri, sc.name);
            let parent = lookup(ri, sc.bond_to);
            if let (Some(child), Some(parent)) = (child, parent) {
                add_bond(&mut bonds, parent, child);
            }
        }

        // ---- Ring-closure bonds (not in side-chain `bond_to` table) ----
        for (name_a, name_b) in ring_closure_bonds(aa) {
            if let (Some(a), Some(b)) = (lookup(ri, name_a), lookup(ri, name_b)) {
                add_bond(&mut bonds, a, b);
            }
        }
    }

    // Sort for stable iteration.
    let mut bonds: Vec<Bond> = bonds.into_iter().collect();
    bonds.sort_by_key(|b| (b.a, b.b));

    // Build adjacency list.
    let mut bonded_to: Vec<Vec<usize>> = vec![Vec::new(); total];
    for b in &bonds {
        bonded_to[b.a].push(b.b);
        bonded_to[b.b].push(b.a);
    }
    for nb in &mut bonded_to {
        nb.sort();
    }

    // Angles: every (a, b, c) where a-b and b-c are bonds, a < c, a != c.
    let mut angles: Vec<Angle> = Vec::new();
    for (b, neigh) in bonded_to.iter().enumerate() {
        for i in 0..neigh.len() {
            for j in (i + 1)..neigh.len() {
                angles.push(Angle { a: neigh[i], b, c: neigh[j] });
            }
        }
    }

    // Proper dihedrals: every (a, b, c, d) where a-b, b-c, c-d are bonds and
    // a, b, c, d are all distinct. Canonicalise so that (b, c) < (c, b) — i.e.
    // store with b < c (or b == c is impossible for distinct atoms).
    let mut dihedral_seen: HashSet<(usize, usize, usize, usize)> = HashSet::new();
    let mut dihedrals: Vec<Dihedral> = Vec::new();
    for bond in &bonds {
        let (b, c) = (bond.a, bond.b);
        for &a in &bonded_to[b] {
            if a == c {
                continue;
            }
            for &d in &bonded_to[c] {
                if d == b || d == a {
                    continue;
                }
                // Canonical order: smaller central pair comes first.
                let key = if b < c { (a, b, c, d) } else { (d, c, b, a) };
                if dihedral_seen.insert(key) {
                    dihedrals.push(Dihedral { a: key.0, b: key.1, c: key.2, d: key.3 });
                }
            }
        }
    }

    // Impropers: enforce sp² planarity at known centers. We list them
    // per-residue based on chemistry. The convention used for the harmonic
    // improper ω: dihedral measured around the central atom, with ω₀ = 0
    // (planar) for sp² centers. Order is (substituent_a, central, sub_b, sub_c).
    let mut impropers: Vec<Improper> = Vec::new();
    for (ri, res) in structure.residues.iter().enumerate() {
        let aa = res.aa;

        // Backbone peptide bond: C(i) is sp²; bonded to CA, O, N(i+1).
        let prev_atoms = if ri + 1 < structure.residues.len() {
            (lookup(ri, "CA"), lookup(ri, "C"), lookup(ri, "O"), lookup(ri + 1, "N"))
        } else {
            (None, None, None, None)
        };
        if let (Some(ca), Some(c), Some(o), Some(next_n)) = prev_atoms {
            impropers.push(Improper { a: ca, b: c, c: o, d: next_n });
        }

        // Aromatic / sp² side-chain centres.
        match aa {
            AminoAcid::Asn => {
                if let (Some(cb), Some(cg), Some(od1), Some(nd2)) = (
                    lookup(ri, "CB"),
                    lookup(ri, "CG"),
                    lookup(ri, "OD1"),
                    lookup(ri, "ND2"),
                ) {
                    impropers.push(Improper { a: cb, b: cg, c: od1, d: nd2 });
                }
            }
            AminoAcid::Gln => {
                if let (Some(cg), Some(cd), Some(oe1), Some(ne2)) = (
                    lookup(ri, "CG"),
                    lookup(ri, "CD"),
                    lookup(ri, "OE1"),
                    lookup(ri, "NE2"),
                ) {
                    impropers.push(Improper { a: cg, b: cd, c: oe1, d: ne2 });
                }
            }
            AminoAcid::Asp => {
                if let (Some(cb), Some(cg), Some(od1), Some(od2)) = (
                    lookup(ri, "CB"),
                    lookup(ri, "CG"),
                    lookup(ri, "OD1"),
                    lookup(ri, "OD2"),
                ) {
                    impropers.push(Improper { a: cb, b: cg, c: od1, d: od2 });
                }
            }
            AminoAcid::Glu => {
                if let (Some(cg), Some(cd), Some(oe1), Some(oe2)) = (
                    lookup(ri, "CG"),
                    lookup(ri, "CD"),
                    lookup(ri, "OE1"),
                    lookup(ri, "OE2"),
                ) {
                    impropers.push(Improper { a: cg, b: cd, c: oe1, d: oe2 });
                }
            }
            AminoAcid::Arg => {
                // Guanidinium centre CZ is sp², bonded to NE, NH1, NH2.
                if let (Some(ne), Some(cz), Some(nh1), Some(nh2)) = (
                    lookup(ri, "NE"),
                    lookup(ri, "CZ"),
                    lookup(ri, "NH1"),
                    lookup(ri, "NH2"),
                ) {
                    impropers.push(Improper { a: ne, b: cz, c: nh1, d: nh2 });
                }
            }
            // Aromatic rings (Phe, Tyr, Trp, His) get an improper at every
            // ring atom that has a substituent off-plane. The internal ring
            // atoms are kept planar by the dihedral periodic terms; no extra
            // impropers needed at this level. (CHARMM36 itself omits aromatic
            // impropers for this reason.)
            _ => {}
        }
    }

    TopologyGraph { bonds, angles, dihedrals, impropers, bonded_to }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_extended_chain;
    use chem::AminoAcid;

    #[test]
    fn alanine_bond_count() {
        // Alanine: backbone 5 bonds (N-CA, CA-C, C-O, N-H, CA-HA)
        //          side chain 4 bonds (CA-CB, CB-HB1, CB-HB2, CB-HB3)
        // total = 9
        let s = build_extended_chain(&[AminoAcid::Ala]).unwrap();
        let g = build_topology_graph(&s);
        assert_eq!(g.bonds.len(), 9);
    }

    #[test]
    fn glycine_bond_count() {
        // Glycine: backbone N-CA, CA-C, C-O, N-H, CA-HA2, CA-HA3 = 6 bonds.
        // No side chain.
        let s = build_extended_chain(&[AminoAcid::Gly]).unwrap();
        let g = build_topology_graph(&s);
        assert_eq!(g.bonds.len(), 6);
    }

    #[test]
    fn proline_ring_closes() {
        // Pro: residue 2 of Ala-Pro should have a bond N-CD even though CD's
        // bond_to is CG in the side-chain table.
        let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Pro]).unwrap();
        let g = build_topology_graph(&s);
        // Find indices.
        let mut total = 0;
        let mut pro_n = None;
        let mut pro_cd = None;
        for (ri, res) in s.residues.iter().enumerate() {
            for atom in &res.atoms {
                if ri == 1 && atom.name == "N" { pro_n = Some(total); }
                if ri == 1 && atom.name == "CD" { pro_cd = Some(total); }
                total += 1;
            }
        }
        let pro_n = pro_n.unwrap();
        let pro_cd = pro_cd.unwrap();
        assert!(g.is_bonded(pro_n, pro_cd), "Pro ring N-Cδ bond not present");
    }

    #[test]
    fn peptide_bond_between_residues() {
        let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala]).unwrap();
        let g = build_topology_graph(&s);
        // Atom 2 in residue 0 is C; atom 0 in residue 1 is N (backbone first).
        // Find via lookup.
        let mut total = 0;
        let mut res0_c = None;
        let mut res1_n = None;
        for (ri, res) in s.residues.iter().enumerate() {
            for atom in &res.atoms {
                if ri == 0 && atom.name == "C" { res0_c = Some(total); }
                if ri == 1 && atom.name == "N" { res1_n = Some(total); }
                total += 1;
            }
        }
        assert!(g.is_bonded(res0_c.unwrap(), res1_n.unwrap()));
    }

    #[test]
    fn one_three_and_one_four_exclusions() {
        // In Ala: N-CA-C-O is a chain. N-C is 1-3, N-O is 1-4.
        let s = build_extended_chain(&[AminoAcid::Ala]).unwrap();
        let g = build_topology_graph(&s);
        let names: Vec<&str> = s.residues[0].atoms.iter().map(|a| a.name).collect();
        let idx = |n: &str| names.iter().position(|x| *x == n).unwrap();
        let n = idx("N");
        let ca = idx("CA");
        let c = idx("C");
        let o = idx("O");
        assert!(g.is_bonded(n, ca));
        assert!(g.is_bonded(ca, c));
        assert!(g.is_bonded(c, o));
        assert!(g.is_one_three(n, c));
        assert!(g.is_one_four(n, o));
        assert!(!g.is_bonded(n, c));
        assert!(!g.is_bonded(n, o));
    }

    #[test]
    fn phenylalanine_aromatic_ring_topology() {
        let s = build_extended_chain(&[AminoAcid::Phe]).unwrap();
        let g = build_topology_graph(&s);
        let names: Vec<&str> = s.residues[0].atoms.iter().map(|a| a.name).collect();
        let idx = |n: &str| names.iter().position(|x| *x == n).unwrap();
        // 6-ring connectivity: CG-CD1-CE1-CZ-CE2-CD2-CG
        let cg = idx("CG");
        let cd1 = idx("CD1");
        let cd2 = idx("CD2");
        let ce1 = idx("CE1");
        let ce2 = idx("CE2");
        let cz = idx("CZ");
        assert!(g.is_bonded(cg, cd1));
        assert!(g.is_bonded(cg, cd2));
        assert!(g.is_bonded(cd1, ce1));
        assert!(g.is_bonded(cd2, ce2));
        assert!(g.is_bonded(ce1, cz));
        assert!(g.is_bonded(ce2, cz));
    }

    #[test]
    fn angles_and_dihedrals_grow_with_chain() {
        let one = build_topology_graph(&build_extended_chain(&[AminoAcid::Ala]).unwrap());
        let two = build_topology_graph(&build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala]).unwrap());
        // More residues = more bonds, angles, dihedrals.
        assert!(two.bonds.len() > one.bonds.len());
        assert!(two.angles.len() > one.angles.len());
        assert!(two.dihedrals.len() > one.dihedrals.len());
    }

    #[test]
    fn peptide_bond_improper_present() {
        // Ala-Ala: residue 0's C should have an improper around it (CA, C, O, next_N).
        let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala]).unwrap();
        let g = build_topology_graph(&s);
        assert!(!g.impropers.is_empty(), "expected at least the peptide-bond improper");
    }
}
