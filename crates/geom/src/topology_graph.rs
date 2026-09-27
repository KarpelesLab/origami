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

use chem::{AminoAcid, Nucleotide};

use crate::structure::{Monomer, Structure};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Bond {
    pub a: usize,
    pub b: usize,
}

impl Bond {
    fn new(a: usize, b: usize) -> Self {
        if a < b {
            Bond { a, b }
        } else {
            Bond { a: b, b: a }
        }
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

/// An improper torsion. **`a` is the central sp² atom** (matching CHARMM's
/// convention where the central atom is listed first); `b`, `c`, `d` are
/// its three substituents. The improper energy is `K (ω − ω₀)²` where ω is
/// the standard IUPAC dihedral a-b-c-d. With central first, planar
/// geometry gives ω = 0.
#[derive(Debug, Clone, Copy)]
pub struct Improper {
    pub a: usize, // central
    pub b: usize,
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
    let lookup = |ri: usize, name: &str| -> Option<usize> { atom_idx.get(&(ri, name)).copied() };

    // Collect bonds as a deduplicated set, then sort for stable iteration.
    let mut bonds: HashSet<Bond> = HashSet::new();
    let add_bond = |bonds: &mut HashSet<Bond>, a: usize, b: usize| {
        if a != b {
            bonds.insert(Bond::new(a, b));
        }
    };

    for (ri, res) in structure.residues.iter().enumerate() {
        match res.monomer {
            Monomer::Protein(aa) => {
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
                // Only auto-bond within the same chain. Multi-chain proteins
                // (insulin, antibodies) have a TER record between chains in
                // the PDB; without this check the last residue of chain A
                // would get a phantom peptide bond to the first residue of
                // chain B, which would distort everything downstream. The
                // previous residue must also be protein — a peptide bond
                // never connects to an RNA residue, and the chain check
                // alone wouldn't catch a hybrid chain.
                if ri > 0
                    && structure.residues[ri - 1].chain == res.chain
                    && structure.residues[ri - 1].monomer.is_protein()
                {
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
            Monomer::Rna(nt) => {
                // ---- Backbone + base intra-residue bonds ----
                // `Nucleotide::topology()` returns the (child, parent)
                // bond list for sugar+phosphate and for the attached
                // base; the ribose ring-closure C1'-O4' is included
                // in the backbone list.
                let topo = nt.topology();
                for &(child, parent) in topo.backbone {
                    if let (Some(c), Some(p)) = (lookup(ri, child), lookup(ri, parent)) {
                        add_bond(&mut bonds, c, p);
                    }
                }
                for &(child, parent) in topo.base {
                    if let (Some(c), Some(p)) = (lookup(ri, child), lookup(ri, parent)) {
                        add_bond(&mut bonds, c, p);
                    }
                }

                // ---- Inter-residue phosphodiester O3'(i-1) -- P(i) ----
                // RNA polymerises 5'→3': the 5'-phosphate of residue i
                // bonds to the 3'-hydroxyl-O of residue i-1. Same
                // chain-boundary + same-monomer-kind safeguard as the
                // peptide bond above.
                if ri > 0
                    && structure.residues[ri - 1].chain == res.chain
                    && structure.residues[ri - 1].monomer.is_rna()
                {
                    if let (Some(prev_o3), Some(p)) = (lookup(ri - 1, "O3'"), lookup(ri, "P")) {
                        add_bond(&mut bonds, prev_o3, p);
                    }
                }
            }
        }
    }

    // ---- Disulfide bridges ----
    // CYS-CYS SG atoms within ~2.5 Å form an S-S covalent bond. This is
    // detected geometrically rather than declared, because the input
    // structure (PDB file or built chain) is the source of truth for
    // which cysteines are oxidised. Threshold 2.5 Å is generous: native
    // disulfides sit at ~2.05 Å, and we don't want to catch sub-vdW-
    // contact spectator pairs (vdW S radius is 1.8 Å, so any pair with
    // SG–SG > ~3.6 Å is definitively not bonded).
    let mut cys_sg: Vec<(usize, crate::Vec3)> = Vec::new();
    for (ri, res) in structure.residues.iter().enumerate() {
        if res.monomer.as_amino_acid() != Some(AminoAcid::Cys) {
            continue;
        }
        if let (Some(idx), Some(pos)) = (lookup(ri, "SG"), res.position("SG")) {
            cys_sg.push((idx, pos));
        }
    }
    const DISULFIDE_MAX_A: f64 = 2.5;
    const DISULFIDE_MAX_SQ: f64 = DISULFIDE_MAX_A * DISULFIDE_MAX_A;
    for i in 0..cys_sg.len() {
        for j in (i + 1)..cys_sg.len() {
            let (idx_i, pi) = cys_sg[i];
            let (idx_j, pj) = cys_sg[j];
            if (pi - pj).norm_squared() <= DISULFIDE_MAX_SQ {
                add_bond(&mut bonds, idx_i, idx_j);
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
                angles.push(Angle {
                    a: neigh[i],
                    b,
                    c: neigh[j],
                });
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
                    dihedrals.push(Dihedral {
                        a: key.0,
                        b: key.1,
                        c: key.2,
                        d: key.3,
                    });
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
        // RNA: planarity of exocyclic substituents on sp² ring carbons.
        // In-ring atoms are kept planar by the dihedral periodic terms
        // (same convention as Phe/Tyr/Trp/His), so only the carbonyls
        // and exocyclic amines get an improper. Central atom = the sp²
        // ring carbon with the off-plane substituent.
        if let Some(nt) = res.monomer.as_nucleotide() {
            let push_rna_improper = |impropers: &mut Vec<Improper>,
                                     center: &str,
                                     sub_a: &str,
                                     sub_b: &str,
                                     sub_c: &str| {
                if let (Some(ca), Some(a), Some(b), Some(c)) = (
                    lookup(ri, center),
                    lookup(ri, sub_a),
                    lookup(ri, sub_b),
                    lookup(ri, sub_c),
                ) {
                    impropers.push(Improper {
                        a: ca,
                        b: a,
                        c: b,
                        d: c,
                    });
                }
            };
            // The CHARMM27 .rtf adds two flavours of base improper:
            //   (i)  at the sp² ring carbon, between the ring substituent
            //        and the exocyclic group (carbonyl O or amine N) —
            //        keeps the exocyclic substituent coplanar with the
            //        ring (e.g. `C6 N1 C5 N6` on adenine);
            //   (ii) at the exocyclic sp² amine N, between the parent
            //        ring carbon and the two amine hydrogens — keeps the
            //        amine NH₂ coplanar with the ring (e.g. `N6 C6 H61 H62`
            //        on adenine).
            // Uracil has no exocyclic amines, only carbonyl oxygens; its
            // IMPR list omits the type-(ii) impropers.
            match nt {
                Nucleotide::Adenine => {
                    // (i)  C6: ring C5, ring N1, exocyclic N6 (amino).
                    push_rna_improper(&mut impropers, "C6", "C5", "N1", "N6");
                    // (ii) N6 amine: parent C6, H61, H62.
                    push_rna_improper(&mut impropers, "N6", "C6", "H61", "H62");
                }
                Nucleotide::Guanine => {
                    // (i)  C6: ring C5, ring N1, exocyclic O6 (carbonyl).
                    push_rna_improper(&mut impropers, "C6", "C5", "N1", "O6");
                    // (i)  C2: ring N1, ring N3, exocyclic N2 (amino).
                    push_rna_improper(&mut impropers, "C2", "N1", "N3", "N2");
                    // (ii) N2 amine: parent C2, H21, H22.
                    push_rna_improper(&mut impropers, "N2", "C2", "H21", "H22");
                }
                Nucleotide::Cytosine => {
                    // (i)  C2: ring N1, ring N3, exocyclic O2 (carbonyl).
                    push_rna_improper(&mut impropers, "C2", "N1", "N3", "O2");
                    // (i)  C4: ring N3, ring C5, exocyclic N4 (amino).
                    push_rna_improper(&mut impropers, "C4", "N3", "C5", "N4");
                    // (ii) N4 amine: parent C4, H41, H42.
                    push_rna_improper(&mut impropers, "N4", "C4", "H41", "H42");
                }
                Nucleotide::Uracil => {
                    // C2: ring N1, ring N3, exocyclic O2 (carbonyl).
                    push_rna_improper(&mut impropers, "C2", "N1", "N3", "O2");
                    // C4: ring N3, ring C5, exocyclic O4 (carbonyl).
                    push_rna_improper(&mut impropers, "C4", "N3", "C5", "O4");
                    // (Uracil has no exocyclic amine; no type-(ii) improper.)
                }
            }
            continue;
        }

        let aa = match res.monomer.as_amino_acid() {
            Some(a) => a,
            None => continue,
        };

        // Backbone peptide bond: C(i) is sp²; bonded to CA, O, N(i+1).
        let prev_atoms = if ri + 1 < structure.residues.len() {
            (
                lookup(ri, "CA"),
                lookup(ri, "C"),
                lookup(ri, "O"),
                lookup(ri + 1, "N"),
            )
        } else {
            (None, None, None, None)
        };
        if let (Some(ca), Some(c), Some(o), Some(next_n)) = prev_atoms {
            // Central = C (sp² peptide-bond carbon).
            impropers.push(Improper {
                a: c,
                b: ca,
                c: o,
                d: next_n,
            });
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
                    // Central = CG (sp² amide C).
                    impropers.push(Improper {
                        a: cg,
                        b: cb,
                        c: od1,
                        d: nd2,
                    });
                }
            }
            AminoAcid::Gln => {
                if let (Some(cg), Some(cd), Some(oe1), Some(ne2)) = (
                    lookup(ri, "CG"),
                    lookup(ri, "CD"),
                    lookup(ri, "OE1"),
                    lookup(ri, "NE2"),
                ) {
                    // Central = CD.
                    impropers.push(Improper {
                        a: cd,
                        b: cg,
                        c: oe1,
                        d: ne2,
                    });
                }
            }
            AminoAcid::Asp => {
                if let (Some(cb), Some(cg), Some(od1), Some(od2)) = (
                    lookup(ri, "CB"),
                    lookup(ri, "CG"),
                    lookup(ri, "OD1"),
                    lookup(ri, "OD2"),
                ) {
                    // Central = CG (sp² carboxyl C).
                    impropers.push(Improper {
                        a: cg,
                        b: cb,
                        c: od1,
                        d: od2,
                    });
                }
            }
            AminoAcid::Glu => {
                if let (Some(cg), Some(cd), Some(oe1), Some(oe2)) = (
                    lookup(ri, "CG"),
                    lookup(ri, "CD"),
                    lookup(ri, "OE1"),
                    lookup(ri, "OE2"),
                ) {
                    // Central = CD.
                    impropers.push(Improper {
                        a: cd,
                        b: cg,
                        c: oe1,
                        d: oe2,
                    });
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
                    // Central = CZ.
                    impropers.push(Improper {
                        a: cz,
                        b: ne,
                        c: nh1,
                        d: nh2,
                    });
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

    TopologyGraph {
        bonds,
        angles,
        dihedrals,
        impropers,
        bonded_to,
    }
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
                if ri == 1 && atom.name == "N" {
                    pro_n = Some(total);
                }
                if ri == 1 && atom.name == "CD" {
                    pro_cd = Some(total);
                }
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
                if ri == 0 && atom.name == "C" {
                    res0_c = Some(total);
                }
                if ri == 1 && atom.name == "N" {
                    res1_n = Some(total);
                }
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
        let two =
            build_topology_graph(&build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala]).unwrap());
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
        assert!(
            !g.impropers.is_empty(),
            "expected at least the peptide-bond improper"
        );
    }

    #[test]
    fn no_disulfide_in_extended_cys_chain() {
        // Two cysteines built as an extended chain have their SG atoms
        // well separated (~5+ Å), so no S-S bond should be detected.
        let s = build_extended_chain(&[AminoAcid::Cys, AminoAcid::Cys]).unwrap();
        let g = build_topology_graph(&s);
        let sg_indices: Vec<usize> = collect_sg_indices(&s);
        assert_eq!(sg_indices.len(), 2);
        assert!(
            !g.is_bonded(sg_indices[0], sg_indices[1]),
            "extended Cys-Cys should not auto-bond"
        );
    }

    #[test]
    fn disulfide_detected_when_sg_atoms_close() {
        // Build Cys-Cys then nudge the second SG to within 2.05 Å of
        // the first SG. The graph builder should pick this up as an
        // S-S bridge.
        let mut s = build_extended_chain(&[AminoAcid::Cys, AminoAcid::Cys]).unwrap();
        let sg0 = s.residues[0]
            .atoms
            .iter()
            .find(|a| a.name == "SG")
            .unwrap()
            .position;
        // Place residue 1's SG at sg0 + 2.029 Å along x — the CHARMM
        // SM-SM equilibrium distance.
        for atom in &mut s.residues[1].atoms {
            if atom.name == "SG" {
                atom.position = sg0 + crate::Vec3::new(2.029, 0.0, 0.0);
            }
        }
        let g = build_topology_graph(&s);
        let sg_indices: Vec<usize> = collect_sg_indices(&s);
        assert_eq!(sg_indices.len(), 2);
        assert!(
            g.is_bonded(sg_indices[0], sg_indices[1]),
            "Cys-Cys at SG-SG = 2.029 Å should be auto-bonded"
        );
    }

    fn collect_sg_indices(s: &crate::structure::Structure) -> Vec<usize> {
        let mut out = Vec::new();
        let mut idx = 0;
        for r in &s.residues {
            for a in &r.atoms {
                if a.name == "SG" {
                    out.push(idx);
                }
                idx += 1;
            }
        }
        out
    }

    // ---- RNA topology graph tests ----

    fn rna_atom_index(
        s: &crate::structure::Structure,
        residue: usize,
        name: &str,
    ) -> Option<usize> {
        let mut idx = 0;
        for (ri, r) in s.residues.iter().enumerate() {
            for a in &r.atoms {
                if ri == residue && a.name == name {
                    return Some(idx);
                }
                idx += 1;
            }
        }
        None
    }

    #[test]
    fn rna_adenine_glycosidic_and_ribose_ring_bonds() {
        use crate::build_extended_rna_chain;
        let s = build_extended_rna_chain(&[chem::Nucleotide::Adenine]).unwrap();
        let g = build_topology_graph(&s);
        let n9 = rna_atom_index(&s, 0, "N9").unwrap();
        let c1 = rna_atom_index(&s, 0, "C1'").unwrap();
        let o4 = rna_atom_index(&s, 0, "O4'").unwrap();
        // Glycosidic N9-C1' bond ties the base to the sugar.
        assert!(g.is_bonded(n9, c1), "RNA adenine N9-C1' bond missing");
        // Ribose ring closure C1'-O4'.
        assert!(g.is_bonded(c1, o4), "RNA ribose ring closure missing");
    }

    #[test]
    fn rna_phosphodiester_bond_between_residues() {
        use crate::build_extended_rna_chain;
        let s = build_extended_rna_chain(&[chem::Nucleotide::Adenine, chem::Nucleotide::Uracil])
            .unwrap();
        let g = build_topology_graph(&s);
        let o3_prev = rna_atom_index(&s, 0, "O3'").unwrap();
        let p_curr = rna_atom_index(&s, 1, "P").unwrap();
        assert!(
            g.is_bonded(o3_prev, p_curr),
            "RNA phosphodiester O3'(0)-P(1) bond missing"
        );
    }

    #[test]
    fn rna_dinucleotide_has_no_phantom_protein_bonds() {
        use crate::build_extended_rna_chain;
        let s = build_extended_rna_chain(&[chem::Nucleotide::Cytosine, chem::Nucleotide::Guanine])
            .unwrap();
        let g = build_topology_graph(&s);
        // No protein-style peptide bond should be detected — the
        // residues don't have C or N backbone atoms, but the
        // chain-boundary check is the real guard for hybrid chains.
        // Sanity: the only inter-residue bond should be the phosphodiester.
        let prev_c1 = rna_atom_index(&s, 0, "C1'").unwrap();
        let curr_c1 = rna_atom_index(&s, 1, "C1'").unwrap();
        assert!(!g.is_bonded(prev_c1, curr_c1));
    }

    #[test]
    fn rna_base_impropers_via_synthetic_structure() {
        // The current RNA chain builder places only the sugar/phosphate
        // scaffold + the glycosidic N (N9/N1) — the full base ring
        // atoms aren't NeRF'd yet, so we drive this test directly off
        // a hand-built Structure that includes every named base atom.
        // This locks in that the topology graph enumerates the correct
        // sp² impropers for all four nucleobases.
        use crate::structure::{Monomer, PlacedAtom, PlacedResidue, Structure};
        use chem::Nucleotide;

        let make_residue = |nt: Nucleotide| -> PlacedResidue {
            let mut atoms: Vec<PlacedAtom> = Vec::new();
            for (name, el) in nt.all_atoms() {
                atoms.push(PlacedAtom {
                    name,
                    element: el,
                    position: crate::Vec3::zeros(),
                });
            }
            PlacedResidue {
                monomer: Monomer::Rna(nt),
                atoms,
                chain: 'A',
            }
        };

        // Adenine: C6 ring improper + N6 amine improper.
        {
            let mut s = Structure::new();
            s.residues.push(make_residue(Nucleotide::Adenine));
            let g = build_topology_graph(&s);
            let c6 = rna_atom_index(&s, 0, "C6").unwrap();
            let c5 = rna_atom_index(&s, 0, "C5").unwrap();
            let n1 = rna_atom_index(&s, 0, "N1").unwrap();
            let n6 = rna_atom_index(&s, 0, "N6").unwrap();
            let h61 = rna_atom_index(&s, 0, "H61").unwrap();
            let h62 = rna_atom_index(&s, 0, "H62").unwrap();
            assert!(
                has_improper(&g, c6, [c5, n1, n6]),
                "Adenine C6 improper missing"
            );
            assert!(
                has_improper(&g, n6, [c6, h61, h62]),
                "Adenine N6 amine improper missing"
            );
        }

        // Guanine: C6 + C2 ring impropers + N2 amine improper.
        {
            let mut s = Structure::new();
            s.residues.push(make_residue(Nucleotide::Guanine));
            let g = build_topology_graph(&s);
            let c6 = rna_atom_index(&s, 0, "C6").unwrap();
            let c5 = rna_atom_index(&s, 0, "C5").unwrap();
            let n1 = rna_atom_index(&s, 0, "N1").unwrap();
            let o6 = rna_atom_index(&s, 0, "O6").unwrap();
            let c2 = rna_atom_index(&s, 0, "C2").unwrap();
            let n3 = rna_atom_index(&s, 0, "N3").unwrap();
            let n2 = rna_atom_index(&s, 0, "N2").unwrap();
            let h21 = rna_atom_index(&s, 0, "H21").unwrap();
            let h22 = rna_atom_index(&s, 0, "H22").unwrap();
            assert!(
                has_improper(&g, c6, [c5, n1, o6]),
                "Guanine C6 improper missing"
            );
            assert!(
                has_improper(&g, c2, [n1, n3, n2]),
                "Guanine C2 improper missing"
            );
            assert!(
                has_improper(&g, n2, [c2, h21, h22]),
                "Guanine N2 amine improper missing"
            );
        }

        // Cytosine: C2 + C4 ring impropers + N4 amine improper.
        {
            let mut s = Structure::new();
            s.residues.push(make_residue(Nucleotide::Cytosine));
            let g = build_topology_graph(&s);
            let c2 = rna_atom_index(&s, 0, "C2").unwrap();
            let n1 = rna_atom_index(&s, 0, "N1").unwrap();
            let n3 = rna_atom_index(&s, 0, "N3").unwrap();
            let o2 = rna_atom_index(&s, 0, "O2").unwrap();
            let c4 = rna_atom_index(&s, 0, "C4").unwrap();
            let c5 = rna_atom_index(&s, 0, "C5").unwrap();
            let n4 = rna_atom_index(&s, 0, "N4").unwrap();
            let h41 = rna_atom_index(&s, 0, "H41").unwrap();
            let h42 = rna_atom_index(&s, 0, "H42").unwrap();
            assert!(
                has_improper(&g, c2, [n1, n3, o2]),
                "Cytosine C2 improper missing"
            );
            assert!(
                has_improper(&g, c4, [n3, c5, n4]),
                "Cytosine C4 improper missing"
            );
            assert!(
                has_improper(&g, n4, [c4, h41, h42]),
                "Cytosine N4 amine improper missing"
            );
        }

        // Uracil: impropers at C2 ({N1, N3, O2}) and C4 ({N3, C5, O4}).
        {
            let mut s = Structure::new();
            s.residues.push(make_residue(Nucleotide::Uracil));
            let g = build_topology_graph(&s);
            let c2 = rna_atom_index(&s, 0, "C2").unwrap();
            let n1 = rna_atom_index(&s, 0, "N1").unwrap();
            let n3 = rna_atom_index(&s, 0, "N3").unwrap();
            let o2 = rna_atom_index(&s, 0, "O2").unwrap();
            let c4 = rna_atom_index(&s, 0, "C4").unwrap();
            let c5 = rna_atom_index(&s, 0, "C5").unwrap();
            let o4 = rna_atom_index(&s, 0, "O4").unwrap();
            assert!(
                has_improper(&g, c2, [n1, n3, o2]),
                "Uracil C2 improper missing"
            );
            assert!(
                has_improper(&g, c4, [n3, c5, o4]),
                "Uracil C4 improper missing"
            );
        }
    }

    /// Check whether `g.impropers` contains an entry centred on `center`
    /// with the three substituents matching `subs` (in any order).
    fn has_improper(g: &TopologyGraph, center: usize, mut subs: [usize; 3]) -> bool {
        subs.sort();
        g.impropers.iter().any(|imp| {
            if imp.a != center {
                return false;
            }
            let mut got = [imp.b, imp.c, imp.d];
            got.sort();
            got == subs
        })
    }

    #[test]
    fn rna_topology_graph_on_built_chain_has_expected_impropers() {
        // End-to-end: build a 4-residue RNA chain, run the topology
        // graph, and confirm the per-base impropers (now that base
        // atoms are NeRF'd by the builder) show up against the real
        // placed atoms.
        use crate::build_extended_rna_chain;
        let s = build_extended_rna_chain(&[
            chem::Nucleotide::Adenine,
            chem::Nucleotide::Uracil,
            chem::Nucleotide::Guanine,
            chem::Nucleotide::Cytosine,
        ])
        .unwrap();
        let g = build_topology_graph(&s);
        // Adenine on residue 0 — C6 improper.
        let c6 = rna_atom_index(&s, 0, "C6").unwrap();
        let c5 = rna_atom_index(&s, 0, "C5").unwrap();
        let n1 = rna_atom_index(&s, 0, "N1").unwrap();
        let n6 = rna_atom_index(&s, 0, "N6").unwrap();
        assert!(has_improper(&g, c6, [c5, n1, n6]));

        // Uracil on residue 1 — C4 carbonyl improper.
        let c4_u = rna_atom_index(&s, 1, "C4").unwrap();
        let n3_u = rna_atom_index(&s, 1, "N3").unwrap();
        let c5_u = rna_atom_index(&s, 1, "C5").unwrap();
        let o4_u = rna_atom_index(&s, 1, "O4").unwrap();
        assert!(has_improper(&g, c4_u, [n3_u, c5_u, o4_u]));
    }
}
