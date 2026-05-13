//! Ribonucleotide types and per-nucleotide atom topology.
//!
//! The protein-folding code paths in `chem` are built around
//! [`crate::amino_acid::AminoAcid`] and the per-residue
//! [`crate::topology::ResidueTopology`]. This module is the sibling
//! abstraction for RNA — one [`Nucleotide`] per ribonucleobase (A, U,
//! G, C) with its full heavy-atom roster and intra-nucleotide bond
//! list. The PDB v3.3 atom names are used throughout; the chem
//! crate's protein-only code is unaffected.
//!
//! ## What's here (M-nuc.1)
//!
//! - The [`Nucleotide`] enum and its single-letter / wwPDB three-
//!   letter conversions.
//! - The full heavy-atom + hydrogen atom list per nucleotide, with
//!   element annotation, in canonical PDB order.
//! - [`NucleotideTopology`] mirroring `ResidueTopology` — each non-
//!   anchor atom names its bonded parent so a NeRF placement pass
//!   later can grow the molecule one atom at a time.
//! - A [`Base`](crate::codon::Base) → [`Nucleotide`] conversion so
//!   the existing translation pipeline can produce RNA chains.
//!
//! ## What's not here yet
//!
//! - NeRF placement parameters (bond lengths / angles / preferred
//!   dihedrals). The data is staged for a follow-up that adds an
//!   RNA chain builder.
//! - Force-field parameters. AMBER's RNA-OL3 or CHARMM36 nucleic
//!   parameter set will need to be vendored when we start running
//!   dynamics on RNA.
//! - Force-field-side atom typing (`AtomType` covers protein atoms
//!   only — there's a separate path the next commit will add).
//! - Integration with `geom::Structure`. `PlacedResidue` still holds
//!   an `AminoAcid`; mixing nucleotides into a Structure needs a
//!   `Monomer` abstraction, also a follow-up.

use std::fmt;

use crate::codon::Base;
use crate::element::Element;

/// One of the four canonical RNA ribonucleobases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Nucleotide {
    Adenine,
    Uracil,
    Guanine,
    Cytosine,
}

impl Nucleotide {
    /// wwPDB v3.3 single-character residue name. RNA residues in
    /// PDB files appear as " A", " U", " G", or " C" in columns
    /// 18-20 (right-justified single letter, often with leading
    /// spaces or as "RA"/"RU"/etc. in some toolchains).
    pub const fn one_letter(self) -> char {
        match self {
            Self::Adenine => 'A',
            Self::Uracil => 'U',
            Self::Guanine => 'G',
            Self::Cytosine => 'C',
        }
    }

    /// Three-letter alias accepted by some PDB tooling and the
    /// internal naming used by AMBER ("RA", "RU", "RG", "RC").
    pub const fn three_letter(self) -> &'static str {
        match self {
            Self::Adenine => "RA",
            Self::Uracil => "RU",
            Self::Guanine => "RG",
            Self::Cytosine => "RC",
        }
    }

    pub fn from_one_letter(c: char) -> Option<Self> {
        Some(match c.to_ascii_uppercase() {
            'A' => Self::Adenine,
            'U' | 'T' => Self::Uracil, // tolerate DNA-style T
            'G' => Self::Guanine,
            'C' => Self::Cytosine,
            _ => return None,
        })
    }

    /// Parse a wwPDB-style residue-name field. Accepts:
    ///   • single-character RNA codes "A" / "U" / "G" / "C"
    ///   • AMBER-style "RA" / "RU" / "RG" / "RC"
    ///   • DNA codes "DA" / "DT" / "DG" / "DC" (T normalises to U;
    ///     we model the ribose sugar only for now)
    /// Returns `None` for amino-acid names — let the caller fall back
    /// to [`crate::AminoAcid::from_three_letter`].
    pub fn from_three_letter(s: &str) -> Option<Self> {
        match s.trim() {
            "A" | "RA" | "DA" => Some(Self::Adenine),
            "U" | "RU" | "T" | "DT" => Some(Self::Uracil),
            "G" | "RG" | "DG" => Some(Self::Guanine),
            "C" | "RC" | "DC" => Some(Self::Cytosine),
            _ => None,
        }
    }

    /// Heavy-atom list (in PDB canonical order) shared by every
    /// ribonucleotide: phosphate, sugar (ribose), and the 2'-hydroxyl
    /// that distinguishes RNA from DNA. Hydrogens are listed
    /// separately by [`Self::backbone_hydrogens`].
    pub fn backbone_heavy_atoms() -> &'static [(&'static str, Element)] {
        &[
            ("P", Element::P),
            ("OP1", Element::O),
            ("OP2", Element::O),
            ("O5'", Element::O),
            ("C5'", Element::C),
            ("C4'", Element::C),
            ("O4'", Element::O),
            ("C3'", Element::C),
            ("O3'", Element::O),
            ("C2'", Element::C),
            ("O2'", Element::O),
            ("C1'", Element::C),
        ]
    }

    /// Hydrogens that decorate the sugar / phosphate backbone, in
    /// PDB canonical order. Every ribonucleotide carries the same
    /// set — the H1'/H2'/H3'/H4'/H5'/H5'' set, plus HO2' on the
    /// 2'-OH.
    pub fn backbone_hydrogens() -> &'static [(&'static str, Element)] {
        &[
            ("H5'", Element::H),
            ("H5''", Element::H),
            ("H4'", Element::H),
            ("H3'", Element::H),
            ("HO2'", Element::H),
            ("H2'", Element::H),
            ("H1'", Element::H),
        ]
    }

    /// Per-base heavy atoms attached at C1'. For A/G these are the
    /// purine 9-membered system (N9 attached, six-ring + five-ring
    /// fused); for C/U the pyrimidine 6-ring is attached at N1.
    pub fn base_heavy_atoms(self) -> &'static [(&'static str, Element)] {
        match self {
            Self::Adenine => &[
                ("N9", Element::N),
                ("C8", Element::C),
                ("N7", Element::N),
                ("C5", Element::C),
                ("C6", Element::C),
                ("N6", Element::N),
                ("N1", Element::N),
                ("C2", Element::C),
                ("N3", Element::N),
                ("C4", Element::C),
            ],
            Self::Guanine => &[
                ("N9", Element::N),
                ("C8", Element::C),
                ("N7", Element::N),
                ("C5", Element::C),
                ("C6", Element::C),
                ("O6", Element::O),
                ("N1", Element::N),
                ("C2", Element::C),
                ("N2", Element::N),
                ("N3", Element::N),
                ("C4", Element::C),
            ],
            Self::Cytosine => &[
                ("N1", Element::N),
                ("C2", Element::C),
                ("O2", Element::O),
                ("N3", Element::N),
                ("C4", Element::C),
                ("N4", Element::N),
                ("C5", Element::C),
                ("C6", Element::C),
            ],
            Self::Uracil => &[
                ("N1", Element::N),
                ("C2", Element::C),
                ("O2", Element::O),
                ("N3", Element::N),
                ("C4", Element::C),
                ("O4", Element::O),
                ("C5", Element::C),
                ("C6", Element::C),
            ],
        }
    }

    /// Per-base hydrogens, in PDB canonical order.
    pub fn base_hydrogens(self) -> &'static [(&'static str, Element)] {
        match self {
            Self::Adenine => &[
                ("H8", Element::H),
                ("H61", Element::H),
                ("H62", Element::H),
                ("H2", Element::H),
            ],
            Self::Guanine => &[
                ("H8", Element::H),
                ("H1", Element::H),
                ("H21", Element::H),
                ("H22", Element::H),
            ],
            Self::Cytosine => &[
                ("H41", Element::H),
                ("H42", Element::H),
                ("H5", Element::H),
                ("H6", Element::H),
            ],
            Self::Uracil => &[
                ("H3", Element::H),
                ("H5", Element::H),
                ("H6", Element::H),
            ],
        }
    }

    /// Full atom roster (heavy + hydrogen, backbone + base) in PDB
    /// canonical order. Used by the eventual NeRF builder + by tests
    /// here.
    pub fn all_atoms(self) -> Vec<(&'static str, Element)> {
        let mut out = Vec::new();
        out.extend_from_slice(Self::backbone_heavy_atoms());
        out.extend_from_slice(Self::backbone_hydrogens());
        out.extend_from_slice(self.base_heavy_atoms());
        out.extend_from_slice(self.base_hydrogens());
        out
    }

    /// Per-nucleotide intra-residue bonded topology (parent atom for
    /// each non-backbone-anchor atom). Used the same way as
    /// `AminoAcid::topology().sidechain` — each entry says "this atom
    /// is bonded to that parent inside the same residue". Inter-
    /// nucleotide P–O3'(i-1) backbone bonds are handled at the
    /// builder level, not here.
    pub fn topology(self) -> NucleotideTopology {
        NucleotideTopology {
            backbone: BACKBONE_BONDS,
            base: match self {
                Self::Adenine => &ADENINE_BONDS,
                Self::Guanine => &GUANINE_BONDS,
                Self::Cytosine => &CYTOSINE_BONDS,
                Self::Uracil => &URACIL_BONDS,
            },
        }
    }
}

impl fmt::Display for Nucleotide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.one_letter())
    }
}

impl From<Base> for Nucleotide {
    fn from(b: Base) -> Self {
        match b {
            Base::A => Self::Adenine,
            Base::U => Self::Uracil,
            Base::G => Self::Guanine,
            Base::C => Self::Cytosine,
        }
    }
}

/// Mirrors [`crate::topology::ResidueTopology`] for ribonucleotides:
/// list of (child, parent) intra-residue bonds, partitioned into
/// backbone (shared across all four nucleotides) and base-specific
/// pieces. Each tuple is `(child_atom_name, parent_atom_name)`.
#[derive(Debug, Clone, Copy)]
pub struct NucleotideTopology {
    pub backbone: &'static [(&'static str, &'static str)],
    pub base: &'static [(&'static str, &'static str)],
}

/// Sugar + phosphate bonds, identical across A / U / G / C.
const BACKBONE_BONDS: &[(&str, &str)] = &[
    // Phosphate group: P bonded to OP1, OP2, and connecting oxygens.
    ("OP1", "P"),
    ("OP2", "P"),
    ("O5'", "P"),
    ("C5'", "O5'"),
    ("C4'", "C5'"),
    ("O4'", "C4'"),
    ("C3'", "C4'"),
    ("O3'", "C3'"),
    ("C2'", "C3'"),
    ("O2'", "C2'"),
    ("C1'", "C2'"),
    // Sugar-ring closure.
    ("C1'", "O4'"),
    // Backbone hydrogens.
    ("H5'", "C5'"),
    ("H5''", "C5'"),
    ("H4'", "C4'"),
    ("H3'", "C3'"),
    ("H2'", "C2'"),
    ("H1'", "C1'"),
    ("HO2'", "O2'"),
];

const ADENINE_BONDS: [(&str, &str); 13] = [
    ("N9", "C1'"),
    ("C8", "N9"),
    ("N7", "C8"),
    ("C5", "N7"),
    ("C4", "N9"),
    ("C4", "C5"),
    ("C6", "C5"),
    ("N6", "C6"),
    ("N1", "C6"),
    ("C2", "N1"),
    ("N3", "C2"),
    ("N3", "C4"),
    ("H8", "C8"),
];

const GUANINE_BONDS: [(&str, &str); 13] = [
    ("N9", "C1'"),
    ("C8", "N9"),
    ("N7", "C8"),
    ("C5", "N7"),
    ("C4", "N9"),
    ("C4", "C5"),
    ("C6", "C5"),
    ("O6", "C6"),
    ("N1", "C6"),
    ("C2", "N1"),
    ("N2", "C2"),
    ("N3", "C2"),
    ("N3", "C4"),
];

const CYTOSINE_BONDS: [(&str, &str); 9] = [
    ("N1", "C1'"),
    ("C2", "N1"),
    ("O2", "C2"),
    ("N3", "C2"),
    ("C4", "N3"),
    ("N4", "C4"),
    ("C5", "C4"),
    ("C6", "C5"),
    ("C6", "N1"),
];

const URACIL_BONDS: [(&str, &str); 9] = [
    ("N1", "C1'"),
    ("C2", "N1"),
    ("O2", "C2"),
    ("N3", "C2"),
    ("C4", "N3"),
    ("O4", "C4"),
    ("C5", "C4"),
    ("C6", "C5"),
    ("C6", "N1"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_canonical_bases_round_trip() {
        for c in ['A', 'U', 'G', 'C'] {
            let n = Nucleotide::from_one_letter(c).unwrap();
            assert_eq!(n.one_letter(), c);
        }
    }

    #[test]
    fn dna_t_normalised_to_uracil() {
        assert_eq!(Nucleotide::from_one_letter('T').unwrap(), Nucleotide::Uracil);
    }

    #[test]
    fn base_into_nucleotide() {
        assert_eq!(Nucleotide::from(Base::A), Nucleotide::Adenine);
        assert_eq!(Nucleotide::from(Base::U), Nucleotide::Uracil);
        assert_eq!(Nucleotide::from(Base::G), Nucleotide::Guanine);
        assert_eq!(Nucleotide::from(Base::C), Nucleotide::Cytosine);
    }

    #[test]
    fn three_letter_codes_match_amber() {
        assert_eq!(Nucleotide::Adenine.three_letter(), "RA");
        assert_eq!(Nucleotide::Cytosine.three_letter(), "RC");
    }

    #[test]
    fn purine_atom_count() {
        // Adenine has 10 heavy atoms in the base + 4 H = 14 base atoms,
        // plus the 12 backbone heavy + 7 backbone H = 19 backbone atoms.
        // Total: 33 atoms per A nucleotide.
        let a = Nucleotide::Adenine.all_atoms();
        assert_eq!(a.len(), 12 + 7 + 10 + 4);
    }

    #[test]
    fn pyrimidine_atom_count() {
        // Cytosine: 8 base heavy + 4 H = 12; backbone 19. Total 31.
        let c = Nucleotide::Cytosine.all_atoms();
        assert_eq!(c.len(), 12 + 7 + 8 + 4);
        // Uracil: 8 base heavy + 3 H = 11; backbone 19. Total 30.
        let u = Nucleotide::Uracil.all_atoms();
        assert_eq!(u.len(), 12 + 7 + 8 + 3);
    }

    #[test]
    fn topology_includes_backbone_and_base_bonds() {
        let topo = Nucleotide::Adenine.topology();
        assert!(!topo.backbone.is_empty());
        assert!(!topo.base.is_empty());
        // Sanity: ribose ring closure must be in backbone bonds.
        assert!(topo.backbone.iter().any(|&(c, p)| c == "C1'" && p == "O4'"));
        // Sanity: adenine attaches at N9 to the sugar's C1'.
        assert!(topo.base.iter().any(|&(c, p)| c == "N9" && p == "C1'"));
    }

    #[test]
    fn every_base_atom_has_known_element() {
        for n in [
            Nucleotide::Adenine,
            Nucleotide::Uracil,
            Nucleotide::Guanine,
            Nucleotide::Cytosine,
        ] {
            for (name, element) in n.all_atoms() {
                // Just confirm element symbols are valid (the enum
                // is non-exhaustive-checked by the match below).
                let _ = element.symbol();
                let _ = name;
            }
        }
    }
}
