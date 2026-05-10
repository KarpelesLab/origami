use chem::{AminoAcid, Element};

use crate::Vec3;

#[derive(Debug, Clone)]
pub struct PlacedAtom {
    pub name: &'static str,
    pub element: Element,
    pub position: Vec3,
}

#[derive(Debug, Clone)]
pub struct PlacedResidue {
    pub aa: AminoAcid,
    pub atoms: Vec<PlacedAtom>,
}

impl PlacedResidue {
    pub fn position(&self, atom_name: &str) -> Option<Vec3> {
        self.atoms
            .iter()
            .find(|a| a.name == atom_name)
            .map(|a| a.position)
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
