//! `Monomer`: a polymer monomer that's either a protein amino acid or
//! an RNA ribonucleotide.
//!
//! Lives in `chem` (not `geom`) because it's a pure data enum over
//! the two chem-level residue types — no geometry inside. Generic code
//! that needs to dispatch on residue kind (atom typing, force-field
//! lookup, PDB residue naming) can take a `Monomer` directly without
//! depending on `geom`.

use crate::amino_acid::AminoAcid;
use crate::nucleotide::Nucleotide;

/// What kind of polymer monomer a residue is.  Currently protein amino
/// acids and RNA ribonucleotides; the enum is the integration point
/// for the long-horizon ribosome work — once full RNA dynamics is in
/// place, a Structure can hold mixed chains (rRNA + ribosomal
/// proteins) without changing the surrounding code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Monomer {
    Protein(AminoAcid),
    Rna(Nucleotide),
}

impl Monomer {
    pub fn as_amino_acid(self) -> Option<AminoAcid> {
        match self {
            Self::Protein(a) => Some(a),
            _ => None,
        }
    }
    pub fn as_nucleotide(self) -> Option<Nucleotide> {
        match self {
            Self::Rna(n) => Some(n),
            _ => None,
        }
    }
    pub fn is_protein(self) -> bool {
        matches!(self, Self::Protein(_))
    }
    pub fn is_rna(self) -> bool {
        matches!(self, Self::Rna(_))
    }

    /// Three-letter residue name in the form a PDB ATOM record uses
    /// (right-justified, three characters). Proteins return their
    /// canonical three-letter code (`ALA`, `ARG`, …). RNA returns the
    /// single-letter PDB v3.3 code right-padded with spaces to three
    /// characters (`A  `, `U  `, `G  `, `C  `).
    pub fn pdb_residue_name(self) -> String {
        match self {
            Self::Protein(aa) => aa.three_letter().to_uppercase(),
            Self::Rna(nt) => format!("{}  ", nt.one_letter()),
        }
    }
}

impl From<AminoAcid> for Monomer {
    fn from(a: AminoAcid) -> Self {
        Self::Protein(a)
    }
}
impl From<Nucleotide> for Monomer {
    fn from(n: Nucleotide) -> Self {
        Self::Rna(n)
    }
}
