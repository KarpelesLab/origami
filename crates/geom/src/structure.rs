use chem::{AminoAcid, Element, Nucleotide};

use crate::Vec3;

/// Re-exported from `chem::Monomer` — the canonical location for the
/// polymer-monomer enum. Existing callers reach it via
/// `geom::structure::Monomer` and continue to work unchanged.
pub use chem::Monomer;

#[derive(Debug, Clone)]
pub struct PlacedAtom {
    pub name: &'static str,
    pub element: Element,
    pub position: Vec3,
}

#[derive(Debug, Clone)]
pub struct PlacedResidue {
    /// What polymer monomer this residue is (protein amino acid or
    /// RNA ribonucleotide). The protein-oriented code paths reach in
    /// via `aa()` which `.expect`s a protein residue; explicitly
    /// RNA-aware code matches on `monomer` directly.
    pub monomer: Monomer,
    pub atoms: Vec<PlacedAtom>,
    /// Chain identifier from the source PDB (single ASCII byte), or
    /// `'A'` by default for synthesised single-chain structures. Used
    /// by `geom::topology_graph::build_topology_graph` to skip the
    /// peptide-bond auto-detection at chain boundaries so multi-chain
    /// proteins (insulin, antibodies) don't get a phantom C(i-1)–N(i)
    /// bond across the chain break.
    pub chain: char,
}

impl PlacedResidue {
    pub fn position(&self, atom_name: &str) -> Option<Vec3> {
        self.atoms
            .iter()
            .find(|a| a.name == atom_name)
            .map(|a| a.position)
    }

    /// Returns the amino acid this residue carries. Panics if the
    /// residue is an RNA nucleotide instead — the protein-only call
    /// sites use this everywhere `residue.aa` was a field access
    /// before the Monomer refactor; new code should pattern match on
    /// `monomer` instead.
    pub fn aa(&self) -> AminoAcid {
        self.monomer
            .as_amino_acid()
            .expect("residue is not a protein amino acid")
    }

    /// Backward-compat for `residue.aa().three_letter()` etc. through
    /// any path that doesn't need to handle nucleotides. New code can
    /// use `monomer.as_amino_acid()` if it wants Option semantics.
    pub fn nucleotide(&self) -> Option<Nucleotide> {
        self.monomer.as_nucleotide()
    }
}

#[derive(Debug, Clone, Default)]
pub struct Structure {
    pub residues: Vec<PlacedResidue>,
}

impl Structure {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn atom_count(&self) -> usize {
        self.residues.iter().map(|r| r.atoms.len()).sum()
    }

    /// Iterate over all (residue_index, atom) pairs in chain order.
    pub fn iter_atoms(&self) -> impl Iterator<Item = (usize, &PlacedAtom)> + '_ {
        self.residues
            .iter()
            .enumerate()
            .flat_map(|(i, r)| r.atoms.iter().map(move |a| (i, a)))
    }
}
