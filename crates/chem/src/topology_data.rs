//! Per-residue side-chain topology tables.
//!
//! Atom naming follows IUPAC/PDB conventions. Bond lengths and angles are
//! standard chemistry values (sp³ tetrahedral 109.47°, sp² trigonal 120°,
//! aromatic C–C 1.39 Å, sp³ C–C 1.53 Å, C–H sp³ 1.09 Å, etc.). Default χ
//! angles are taken from the canonical staggered minima of the side-chain
//! torsion potential: trans (180°), gauche+ (+60°), gauche− (−60°).
//!
//! Each side-chain atom is placed via NeRF using three parent atoms:
//! `dihedral_to`, `angle_at`, `bond_to` (corresponding to a, b, c in the
//! NeRF formulation). Parents must be backbone atoms (N, CA, C, O — the
//! backbone is placed first by the chain builder) or earlier side-chain
//! atoms in the same residue.

use std::f64::consts::PI;

use crate::amino_acid::AminoAcid;
use crate::element::Element;
use crate::topology::{angle, bond, DihedralValue, ResidueTopology, SidechainAtom};

const fn d(deg: f64) -> f64 {
    deg * PI / 180.0
}

const TRANS: f64 = PI;
#[allow(dead_code)]
const GAUCHE_PLUS: f64 = PI / 3.0;
#[allow(dead_code)]
const GAUCHE_MINUS: f64 = -PI / 3.0;

// ---------------------------------------------------------------------------
// Side-chain tables.
//
// Convention: every side-chain table starts with the Cβ atom, placed off Cα
// using parents [C, N, CA] with the standard L-amino-acid dihedral
// C-N-CA-CB = -122.55° (matching CHARMM internal coordinates). Subsequent
// atoms reference Cβ and earlier side-chain atoms.
// ---------------------------------------------------------------------------

const CB_DIHEDRAL: f64 = d(-122.55);
const CB_ANGLE: f64 = d(110.5);

/// Cβ placement is identical for every non-Gly residue.
const fn cb_atom() -> SidechainAtom {
    SidechainAtom {
        name: "CB",
        element: Element::C,
        bond_to: "CA",
        angle_at: "N",
        dihedral_to: "C",
        bond_length_a: bond::C_C,
        bond_angle_rad: CB_ANGLE,
        dihedral: DihedralValue::Fixed(CB_DIHEDRAL),
    }
}

// ===========================================================================
// Glycine — no side chain.
// ===========================================================================

const GLY: ResidueTopology = ResidueTopology {
    sidechain: &[],
    default_chi_rad: &[],
    has_amide_h: true,
    is_glycine: true,
};

// ===========================================================================
// Alanine — methyl side chain.
// ===========================================================================

const ALA: ResidueTopology = ResidueTopology {
    sidechain: &[
        cb_atom(),
        // Three β-hydrogens at staggered positions around the Cα–Cβ axis.
        // Reference dihedral: N–CA–CB–HB. χ doesn't apply (no χ for Ala).
        SidechainAtom {
            name: "HB1",
            element: Element::H,
            bond_to: "CB",
            angle_at: "CA",
            dihedral_to: "N",
            bond_length_a: bond::C_H_SP3,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Fixed(d(60.0)),
        },
        SidechainAtom {
            name: "HB2",
            element: Element::H,
            bond_to: "CB",
            angle_at: "CA",
            dihedral_to: "N",
            bond_length_a: bond::C_H_SP3,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Fixed(d(180.0)),
        },
        SidechainAtom {
            name: "HB3",
            element: Element::H,
            bond_to: "CB",
            angle_at: "CA",
            dihedral_to: "N",
            bond_length_a: bond::C_H_SP3,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Fixed(d(-60.0)),
        },
    ],
    default_chi_rad: &[],
    has_amide_h: true,
    is_glycine: false,
};

// ===========================================================================
// Valine — Cβ branches into two methyls (CG1, CG2) plus one HB.
// χ₁ = N–CA–CB–CG1.
// ===========================================================================

const VAL: ResidueTopology = ResidueTopology {
    sidechain: &[
        cb_atom(),
        SidechainAtom {
            name: "HB",
            element: Element::H,
            bond_to: "CB",
            angle_at: "CA",
            dihedral_to: "N",
            bond_length_a: bond::C_H_SP3,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::ChiPlus(1, d(120.0)),
        },
        SidechainAtom {
            name: "CG1",
            element: Element::C,
            bond_to: "CB",
            angle_at: "CA",
            dihedral_to: "N",
            bond_length_a: bond::C_C,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Chi(1),
        },
        SidechainAtom {
            name: "CG2",
            element: Element::C,
            bond_to: "CB",
            angle_at: "CA",
            dihedral_to: "N",
            bond_length_a: bond::C_C,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::ChiPlus(1, d(-120.0)),
        },
        // Methyl hydrogens on CG1, staggered around CB-CG1 axis.
        methyl_h("HG11", "CG1", "CB", "CA", d(60.0)),
        methyl_h("HG12", "CG1", "CB", "CA", d(180.0)),
        methyl_h("HG13", "CG1", "CB", "CA", d(-60.0)),
        // Methyl hydrogens on CG2.
        methyl_h("HG21", "CG2", "CB", "CA", d(60.0)),
        methyl_h("HG22", "CG2", "CB", "CA", d(180.0)),
        methyl_h("HG23", "CG2", "CB", "CA", d(-60.0)),
    ],
    default_chi_rad: &[TRANS],
    has_amide_h: true,
    is_glycine: false,
};

// ===========================================================================
// Leucine — Cβ-CγH-CδMe2.  χ₁ = N-CA-CB-CG, χ₂ = CA-CB-CG-CD1.
// ===========================================================================

const LEU: ResidueTopology = ResidueTopology {
    sidechain: &[
        cb_atom(),
        ch2_chi("HB2", "CB", "CA", "N", 1, 1),
        ch2_chi("HB3", "CB", "CA", "N", 1, -1),
        SidechainAtom {
            name: "CG",
            element: Element::C,
            bond_to: "CB",
            angle_at: "CA",
            dihedral_to: "N",
            bond_length_a: bond::C_C,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Chi(1),
        },
        SidechainAtom {
            name: "HG",
            element: Element::H,
            bond_to: "CG",
            angle_at: "CB",
            dihedral_to: "CA",
            bond_length_a: bond::C_H_SP3,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::ChiPlus(2, d(120.0)),
        },
        SidechainAtom {
            name: "CD1",
            element: Element::C,
            bond_to: "CG",
            angle_at: "CB",
            dihedral_to: "CA",
            bond_length_a: bond::C_C,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Chi(2),
        },
        SidechainAtom {
            name: "CD2",
            element: Element::C,
            bond_to: "CG",
            angle_at: "CB",
            dihedral_to: "CA",
            bond_length_a: bond::C_C,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::ChiPlus(2, d(-120.0)),
        },
        methyl_h("HD11", "CD1", "CG", "CB", d(60.0)),
        methyl_h("HD12", "CD1", "CG", "CB", d(180.0)),
        methyl_h("HD13", "CD1", "CG", "CB", d(-60.0)),
        methyl_h("HD21", "CD2", "CG", "CB", d(60.0)),
        methyl_h("HD22", "CD2", "CG", "CB", d(180.0)),
        methyl_h("HD23", "CD2", "CG", "CB", d(-60.0)),
    ],
    default_chi_rad: &[TRANS, TRANS],
    has_amide_h: true,
    is_glycine: false,
};

// ===========================================================================
// Isoleucine — Cβ-CH(CγMe)(CγEt). χ₁=N-CA-CB-CG1, χ₂=CA-CB-CG1-CD1.
// ===========================================================================

const ILE: ResidueTopology = ResidueTopology {
    sidechain: &[
        cb_atom(),
        SidechainAtom {
            name: "HB",
            element: Element::H,
            bond_to: "CB",
            angle_at: "CA",
            dihedral_to: "N",
            bond_length_a: bond::C_H_SP3,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::ChiPlus(1, d(120.0)),
        },
        SidechainAtom {
            name: "CG1",
            element: Element::C,
            bond_to: "CB",
            angle_at: "CA",
            dihedral_to: "N",
            bond_length_a: bond::C_C,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Chi(1),
        },
        SidechainAtom {
            name: "CG2",
            element: Element::C,
            bond_to: "CB",
            angle_at: "CA",
            dihedral_to: "N",
            bond_length_a: bond::C_C,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::ChiPlus(1, d(-120.0)),
        },
        // CG1 has 2 H's (not methyl, since CG1 is bonded to CD1).
        ch2_chi("HG12", "CG1", "CB", "CA", 2, 1),
        ch2_chi("HG13", "CG1", "CB", "CA", 2, -1),
        // CG2 is a methyl.
        methyl_h("HG21", "CG2", "CB", "CA", d(60.0)),
        methyl_h("HG22", "CG2", "CB", "CA", d(180.0)),
        methyl_h("HG23", "CG2", "CB", "CA", d(-60.0)),
        SidechainAtom {
            name: "CD1",
            element: Element::C,
            bond_to: "CG1",
            angle_at: "CB",
            dihedral_to: "CA",
            bond_length_a: bond::C_C,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Chi(2),
        },
        methyl_h("HD11", "CD1", "CG1", "CB", d(60.0)),
        methyl_h("HD12", "CD1", "CG1", "CB", d(180.0)),
        methyl_h("HD13", "CD1", "CG1", "CB", d(-60.0)),
    ],
    default_chi_rad: &[TRANS, TRANS],
    has_amide_h: true,
    is_glycine: false,
};

// ===========================================================================
// Methionine — CB-CG-SD-CE.  χ₁,₂,₃ along the chain; χ₃ has 3-fold S-C.
// ===========================================================================

const MET: ResidueTopology = ResidueTopology {
    sidechain: &[
        cb_atom(),
        ch2_chi("HB2", "CB", "CA", "N", 1, 1),
        ch2_chi("HB3", "CB", "CA", "N", 1, -1),
        SidechainAtom {
            name: "CG",
            element: Element::C,
            bond_to: "CB",
            angle_at: "CA",
            dihedral_to: "N",
            bond_length_a: bond::C_C,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Chi(1),
        },
        ch2_chi("HG2", "CG", "CB", "CA", 2, 1),
        ch2_chi("HG3", "CG", "CB", "CA", 2, -1),
        SidechainAtom {
            name: "SD",
            element: Element::S,
            bond_to: "CG",
            angle_at: "CB",
            dihedral_to: "CA",
            bond_length_a: bond::C_S,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Chi(2),
        },
        SidechainAtom {
            name: "CE",
            element: Element::C,
            bond_to: "SD",
            angle_at: "CG",
            dihedral_to: "CB",
            bond_length_a: bond::C_S,
            bond_angle_rad: d(100.9), // C-S-C angle in thioethers
            dihedral: DihedralValue::Chi(3),
        },
        methyl_h("HE1", "CE", "SD", "CG", d(60.0)),
        methyl_h("HE2", "CE", "SD", "CG", d(180.0)),
        methyl_h("HE3", "CE", "SD", "CG", d(-60.0)),
    ],
    default_chi_rad: &[TRANS, TRANS, TRANS],
    has_amide_h: true,
    is_glycine: false,
};

// ===========================================================================
// Proline — five-membered ring N-CA-CB-CG-CD-N. Special: no amide H, and
// the CD–N bond is implicit (we do NOT enforce ring closure at build time;
// the chain is placed with idealized chi₁ ≈ -25° and accepts a small ring
// strain that minimization will resolve).
// ===========================================================================

const PRO: ResidueTopology = ResidueTopology {
    sidechain: &[
        cb_atom(),
        ch2_chi("HB2", "CB", "CA", "N", 1, 1),
        ch2_chi("HB3", "CB", "CA", "N", 1, -1),
        SidechainAtom {
            name: "CG",
            element: Element::C,
            bond_to: "CB",
            angle_at: "CA",
            dihedral_to: "N",
            bond_length_a: bond::C_C,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Chi(1),
        },
        ch2_chi("HG2", "CG", "CB", "CA", 2, 1),
        ch2_chi("HG3", "CG", "CB", "CA", 2, -1),
        SidechainAtom {
            name: "CD",
            element: Element::C,
            bond_to: "CG",
            angle_at: "CB",
            dihedral_to: "CA",
            bond_length_a: bond::C_C,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Chi(2),
        },
        beta_h("HD2", "CD", "CG", "CB", d(60.0)),
        beta_h("HD3", "CD", "CG", "CB", d(-60.0)),
    ],
    // Ring-pucker default: Cγ-endo (most common in α-helical Pro).
    default_chi_rad: &[d(-25.0), d(35.0)],
    has_amide_h: false, // N is bonded to Cδ, not H
    is_glycine: false,
};

// ===========================================================================
// Serine — Cβ-CH₂-OH.  χ₁ = N-CA-CB-OG.
// ===========================================================================

const SER: ResidueTopology = ResidueTopology {
    sidechain: &[
        cb_atom(),
        ch2_chi("HB2", "CB", "CA", "N", 1, 1),
        ch2_chi("HB3", "CB", "CA", "N", 1, -1),
        SidechainAtom {
            name: "OG",
            element: Element::O,
            bond_to: "CB",
            angle_at: "CA",
            dihedral_to: "N",
            bond_length_a: bond::C_O,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Chi(1),
        },
        SidechainAtom {
            name: "HG",
            element: Element::H,
            bond_to: "OG",
            angle_at: "CB",
            dihedral_to: "CA",
            bond_length_a: bond::O_H,
            bond_angle_rad: d(108.0),
            dihedral: DihedralValue::Fixed(d(180.0)),
        },
    ],
    default_chi_rad: &[TRANS],
    has_amide_h: true,
    is_glycine: false,
};

// ===========================================================================
// Threonine — Cβ-CH(OH)(CH₃).  χ₁ = N-CA-CB-OG1.
// ===========================================================================

const THR: ResidueTopology = ResidueTopology {
    sidechain: &[
        cb_atom(),
        SidechainAtom {
            name: "HB",
            element: Element::H,
            bond_to: "CB",
            angle_at: "CA",
            dihedral_to: "N",
            bond_length_a: bond::C_H_SP3,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::ChiPlus(1, d(120.0)),
        },
        SidechainAtom {
            name: "OG1",
            element: Element::O,
            bond_to: "CB",
            angle_at: "CA",
            dihedral_to: "N",
            bond_length_a: bond::C_O,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Chi(1),
        },
        SidechainAtom {
            name: "HG1",
            element: Element::H,
            bond_to: "OG1",
            angle_at: "CB",
            dihedral_to: "CA",
            bond_length_a: bond::O_H,
            bond_angle_rad: d(108.0),
            dihedral: DihedralValue::Fixed(d(180.0)),
        },
        SidechainAtom {
            name: "CG2",
            element: Element::C,
            bond_to: "CB",
            angle_at: "CA",
            dihedral_to: "N",
            bond_length_a: bond::C_C,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::ChiPlus(1, d(-120.0)),
        },
        methyl_h("HG21", "CG2", "CB", "CA", d(60.0)),
        methyl_h("HG22", "CG2", "CB", "CA", d(180.0)),
        methyl_h("HG23", "CG2", "CB", "CA", d(-60.0)),
    ],
    default_chi_rad: &[TRANS],
    has_amide_h: true,
    is_glycine: false,
};

// ===========================================================================
// Cysteine — Cβ-CH₂-SH.  χ₁ = N-CA-CB-SG.
// ===========================================================================

const CYS: ResidueTopology = ResidueTopology {
    sidechain: &[
        cb_atom(),
        ch2_chi("HB2", "CB", "CA", "N", 1, 1),
        ch2_chi("HB3", "CB", "CA", "N", 1, -1),
        SidechainAtom {
            name: "SG",
            element: Element::S,
            bond_to: "CB",
            angle_at: "CA",
            dihedral_to: "N",
            bond_length_a: bond::C_S,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Chi(1),
        },
        SidechainAtom {
            name: "HG",
            element: Element::H,
            bond_to: "SG",
            angle_at: "CB",
            dihedral_to: "CA",
            bond_length_a: bond::S_H,
            bond_angle_rad: d(96.0),
            dihedral: DihedralValue::Fixed(d(180.0)),
        },
    ],
    default_chi_rad: &[TRANS],
    has_amide_h: true,
    is_glycine: false,
};

// ===========================================================================
// Asparagine — Cβ-CH₂-CONH₂.  χ₁,χ₂.
// ===========================================================================

const ASN: ResidueTopology = ResidueTopology {
    sidechain: &[
        cb_atom(),
        ch2_chi("HB2", "CB", "CA", "N", 1, 1),
        ch2_chi("HB3", "CB", "CA", "N", 1, -1),
        SidechainAtom {
            name: "CG",
            element: Element::C,
            bond_to: "CB",
            angle_at: "CA",
            dihedral_to: "N",
            bond_length_a: 1.516,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Chi(1),
        },
        // sp2 CG: bonded to CB, OD1, ND2 with 120° angles.
        SidechainAtom {
            name: "OD1",
            element: Element::O,
            bond_to: "CG",
            angle_at: "CB",
            dihedral_to: "CA",
            bond_length_a: 1.231,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Chi(2),
        },
        SidechainAtom {
            name: "ND2",
            element: Element::N,
            bond_to: "CG",
            angle_at: "CB",
            dihedral_to: "CA",
            bond_length_a: 1.328,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::ChiPlus(2, d(180.0)),
        },
        // Two amide H's on ND2, planar (sp2 N).
        SidechainAtom {
            name: "HD21",
            element: Element::H,
            bond_to: "ND2",
            angle_at: "CG",
            dihedral_to: "CB",
            bond_length_a: bond::N_H,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(0.0)), // cis to OD1 (Z-configuration)
        },
        SidechainAtom {
            name: "HD22",
            element: Element::H,
            bond_to: "ND2",
            angle_at: "CG",
            dihedral_to: "CB",
            bond_length_a: bond::N_H,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(180.0)),
        },
    ],
    default_chi_rad: &[TRANS, d(-90.0)],
    has_amide_h: true,
    is_glycine: false,
};

// ===========================================================================
// Glutamine — Cβ-CH₂-CH₂-CONH₂.  χ₁,χ₂,χ₃.
// ===========================================================================

const GLN: ResidueTopology = ResidueTopology {
    sidechain: &[
        cb_atom(),
        ch2_chi("HB2", "CB", "CA", "N", 1, 1),
        ch2_chi("HB3", "CB", "CA", "N", 1, -1),
        SidechainAtom {
            name: "CG",
            element: Element::C,
            bond_to: "CB",
            angle_at: "CA",
            dihedral_to: "N",
            bond_length_a: bond::C_C,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Chi(1),
        },
        ch2_chi("HG2", "CG", "CB", "CA", 2, 1),
        ch2_chi("HG3", "CG", "CB", "CA", 2, -1),
        SidechainAtom {
            name: "CD",
            element: Element::C,
            bond_to: "CG",
            angle_at: "CB",
            dihedral_to: "CA",
            bond_length_a: 1.516,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Chi(2),
        },
        SidechainAtom {
            name: "OE1",
            element: Element::O,
            bond_to: "CD",
            angle_at: "CG",
            dihedral_to: "CB",
            bond_length_a: 1.231,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Chi(3),
        },
        SidechainAtom {
            name: "NE2",
            element: Element::N,
            bond_to: "CD",
            angle_at: "CG",
            dihedral_to: "CB",
            bond_length_a: 1.328,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::ChiPlus(3, d(180.0)),
        },
        SidechainAtom {
            name: "HE21",
            element: Element::H,
            bond_to: "NE2",
            angle_at: "CD",
            dihedral_to: "CG",
            bond_length_a: bond::N_H,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(0.0)),
        },
        SidechainAtom {
            name: "HE22",
            element: Element::H,
            bond_to: "NE2",
            angle_at: "CD",
            dihedral_to: "CG",
            bond_length_a: bond::N_H,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(180.0)),
        },
    ],
    default_chi_rad: &[TRANS, TRANS, d(-90.0)],
    has_amide_h: true,
    is_glycine: false,
};

// ===========================================================================
// Aspartate — Cβ-CH₂-COO⁻.  χ₁,χ₂.  CG is sp2 carboxyl.
// ===========================================================================

const ASP: ResidueTopology = ResidueTopology {
    sidechain: &[
        cb_atom(),
        ch2_chi("HB2", "CB", "CA", "N", 1, 1),
        ch2_chi("HB3", "CB", "CA", "N", 1, -1),
        SidechainAtom {
            name: "CG",
            element: Element::C,
            bond_to: "CB",
            angle_at: "CA",
            dihedral_to: "N",
            bond_length_a: 1.516,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Chi(1),
        },
        SidechainAtom {
            name: "OD1",
            element: Element::O,
            bond_to: "CG",
            angle_at: "CB",
            dihedral_to: "CA",
            bond_length_a: bond::C_O_CARBOXYL,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Chi(2),
        },
        SidechainAtom {
            name: "OD2",
            element: Element::O,
            bond_to: "CG",
            angle_at: "CB",
            dihedral_to: "CA",
            bond_length_a: bond::C_O_CARBOXYL,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::ChiPlus(2, d(180.0)),
        },
    ],
    default_chi_rad: &[TRANS, d(-90.0)],
    has_amide_h: true,
    is_glycine: false,
};

// ===========================================================================
// Glutamate — Cβ-CH₂-CH₂-COO⁻.  χ₁,χ₂,χ₃.
// ===========================================================================

const GLU: ResidueTopology = ResidueTopology {
    sidechain: &[
        cb_atom(),
        ch2_chi("HB2", "CB", "CA", "N", 1, 1),
        ch2_chi("HB3", "CB", "CA", "N", 1, -1),
        SidechainAtom {
            name: "CG",
            element: Element::C,
            bond_to: "CB",
            angle_at: "CA",
            dihedral_to: "N",
            bond_length_a: bond::C_C,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Chi(1),
        },
        ch2_chi("HG2", "CG", "CB", "CA", 2, 1),
        ch2_chi("HG3", "CG", "CB", "CA", 2, -1),
        SidechainAtom {
            name: "CD",
            element: Element::C,
            bond_to: "CG",
            angle_at: "CB",
            dihedral_to: "CA",
            bond_length_a: 1.516,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Chi(2),
        },
        SidechainAtom {
            name: "OE1",
            element: Element::O,
            bond_to: "CD",
            angle_at: "CG",
            dihedral_to: "CB",
            bond_length_a: bond::C_O_CARBOXYL,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Chi(3),
        },
        SidechainAtom {
            name: "OE2",
            element: Element::O,
            bond_to: "CD",
            angle_at: "CG",
            dihedral_to: "CB",
            bond_length_a: bond::C_O_CARBOXYL,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::ChiPlus(3, d(180.0)),
        },
    ],
    default_chi_rad: &[TRANS, TRANS, d(-90.0)],
    has_amide_h: true,
    is_glycine: false,
};

// ===========================================================================
// Lysine — Cβ-CH₂-CH₂-CH₂-CH₂-NH₃⁺.
// ===========================================================================

const LYS: ResidueTopology = ResidueTopology {
    sidechain: &[
        cb_atom(),
        ch2_chi("HB2", "CB", "CA", "N", 1, 1),
        ch2_chi("HB3", "CB", "CA", "N", 1, -1),
        SidechainAtom {
            name: "CG",
            element: Element::C,
            bond_to: "CB",
            angle_at: "CA",
            dihedral_to: "N",
            bond_length_a: bond::C_C,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Chi(1),
        },
        ch2_chi("HG2", "CG", "CB", "CA", 2, 1),
        ch2_chi("HG3", "CG", "CB", "CA", 2, -1),
        SidechainAtom {
            name: "CD",
            element: Element::C,
            bond_to: "CG",
            angle_at: "CB",
            dihedral_to: "CA",
            bond_length_a: bond::C_C,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Chi(2),
        },
        ch2_chi("HD2", "CD", "CG", "CB", 3, 1),
        ch2_chi("HD3", "CD", "CG", "CB", 3, -1),
        SidechainAtom {
            name: "CE",
            element: Element::C,
            bond_to: "CD",
            angle_at: "CG",
            dihedral_to: "CB",
            bond_length_a: bond::C_C,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Chi(3),
        },
        ch2_chi("HE2", "CE", "CD", "CG", 4, 1),
        ch2_chi("HE3", "CE", "CD", "CG", 4, -1),
        SidechainAtom {
            name: "NZ",
            element: Element::N,
            bond_to: "CE",
            angle_at: "CD",
            dihedral_to: "CG",
            bond_length_a: bond::C_N,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Chi(4),
        },
        // Three NH₃⁺ hydrogens, staggered.
        SidechainAtom {
            name: "HZ1",
            element: Element::H,
            bond_to: "NZ",
            angle_at: "CE",
            dihedral_to: "CD",
            bond_length_a: bond::N_H,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Fixed(d(60.0)),
        },
        SidechainAtom {
            name: "HZ2",
            element: Element::H,
            bond_to: "NZ",
            angle_at: "CE",
            dihedral_to: "CD",
            bond_length_a: bond::N_H,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Fixed(d(180.0)),
        },
        SidechainAtom {
            name: "HZ3",
            element: Element::H,
            bond_to: "NZ",
            angle_at: "CE",
            dihedral_to: "CD",
            bond_length_a: bond::N_H,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Fixed(d(-60.0)),
        },
    ],
    default_chi_rad: &[TRANS, TRANS, TRANS, TRANS],
    has_amide_h: true,
    is_glycine: false,
};

// ===========================================================================
// Arginine — Cβ-CH₂-CH₂-CH₂-NH-C(NH₂)₂⁺ (guanidinium, planar).
// ===========================================================================

const ARG: ResidueTopology = ResidueTopology {
    sidechain: &[
        cb_atom(),
        ch2_chi("HB2", "CB", "CA", "N", 1, 1),
        ch2_chi("HB3", "CB", "CA", "N", 1, -1),
        SidechainAtom {
            name: "CG",
            element: Element::C,
            bond_to: "CB",
            angle_at: "CA",
            dihedral_to: "N",
            bond_length_a: bond::C_C,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Chi(1),
        },
        ch2_chi("HG2", "CG", "CB", "CA", 2, 1),
        ch2_chi("HG3", "CG", "CB", "CA", 2, -1),
        SidechainAtom {
            name: "CD",
            element: Element::C,
            bond_to: "CG",
            angle_at: "CB",
            dihedral_to: "CA",
            bond_length_a: bond::C_C,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Chi(2),
        },
        ch2_chi("HD2", "CD", "CG", "CB", 3, 1),
        ch2_chi("HD3", "CD", "CG", "CB", 3, -1),
        SidechainAtom {
            name: "NE",
            element: Element::N,
            bond_to: "CD",
            angle_at: "CG",
            dihedral_to: "CB",
            bond_length_a: bond::C_N,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Chi(3),
        },
        SidechainAtom {
            name: "HE",
            element: Element::H,
            bond_to: "NE",
            angle_at: "CD",
            dihedral_to: "CG",
            bond_length_a: bond::N_H,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(0.0)),
        },
        // Guanidinium — sp2 planar.
        SidechainAtom {
            name: "CZ",
            element: Element::C,
            bond_to: "NE",
            angle_at: "CD",
            dihedral_to: "CG",
            bond_length_a: bond::C_N_AROMATIC,
            bond_angle_rad: d(123.5), // C-N-C in guanidinium
            dihedral: DihedralValue::Chi(4),
        },
        SidechainAtom {
            name: "NH1",
            element: Element::N,
            bond_to: "CZ",
            angle_at: "NE",
            dihedral_to: "CD",
            bond_length_a: bond::C_N_AROMATIC,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(0.0)),
        },
        SidechainAtom {
            name: "NH2",
            element: Element::N,
            bond_to: "CZ",
            angle_at: "NE",
            dihedral_to: "CD",
            bond_length_a: bond::C_N_AROMATIC,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(180.0)),
        },
        SidechainAtom {
            name: "HH11",
            element: Element::H,
            bond_to: "NH1",
            angle_at: "CZ",
            dihedral_to: "NE",
            bond_length_a: bond::N_H,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(0.0)),
        },
        SidechainAtom {
            name: "HH12",
            element: Element::H,
            bond_to: "NH1",
            angle_at: "CZ",
            dihedral_to: "NE",
            bond_length_a: bond::N_H,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(180.0)),
        },
        SidechainAtom {
            name: "HH21",
            element: Element::H,
            bond_to: "NH2",
            angle_at: "CZ",
            dihedral_to: "NE",
            bond_length_a: bond::N_H,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(0.0)),
        },
        SidechainAtom {
            name: "HH22",
            element: Element::H,
            bond_to: "NH2",
            angle_at: "CZ",
            dihedral_to: "NE",
            bond_length_a: bond::N_H,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(180.0)),
        },
    ],
    default_chi_rad: &[TRANS, TRANS, TRANS, TRANS],
    has_amide_h: true,
    is_glycine: false,
};

// ===========================================================================
// Histidine — Cβ-CH₂-imidazole(5-ring). Default tautomer: HD1 on ND1 (Hδ).
// ===========================================================================

const HIS: ResidueTopology = ResidueTopology {
    sidechain: &[
        cb_atom(),
        ch2_chi("HB2", "CB", "CA", "N", 1, 1),
        ch2_chi("HB3", "CB", "CA", "N", 1, -1),
        SidechainAtom {
            name: "CG",
            element: Element::C,
            bond_to: "CB",
            angle_at: "CA",
            dihedral_to: "N",
            bond_length_a: 1.500,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Chi(1),
        },
        // Imidazole 5-ring: CG–ND1–CE1–NE2–CD2–CG.  Internal angles ≈108°.
        SidechainAtom {
            name: "ND1",
            element: Element::N,
            bond_to: "CG",
            angle_at: "CB",
            dihedral_to: "CA",
            bond_length_a: 1.378,
            bond_angle_rad: d(122.0),
            dihedral: DihedralValue::Chi(2),
        },
        SidechainAtom {
            name: "CD2",
            element: Element::C,
            bond_to: "CG",
            angle_at: "CB",
            dihedral_to: "CA",
            bond_length_a: 1.354,
            bond_angle_rad: d(130.6),
            dihedral: DihedralValue::ChiPlus(2, d(180.0)),
        },
        SidechainAtom {
            name: "CE1",
            element: Element::C,
            bond_to: "ND1",
            angle_at: "CG",
            dihedral_to: "CB",
            bond_length_a: 1.336,
            bond_angle_rad: d(108.4),
            dihedral: DihedralValue::Fixed(d(180.0)),
        },
        SidechainAtom {
            name: "NE2",
            element: Element::N,
            bond_to: "CD2",
            angle_at: "CG",
            dihedral_to: "CB",
            bond_length_a: 1.378,
            bond_angle_rad: d(108.4),
            dihedral: DihedralValue::Fixed(d(180.0)),
        },
        SidechainAtom {
            name: "HD1",
            element: Element::H,
            bond_to: "ND1",
            angle_at: "CG",
            dihedral_to: "CB",
            bond_length_a: bond::N_H,
            bond_angle_rad: d(126.0),
            dihedral: DihedralValue::Fixed(d(0.0)),
        },
        SidechainAtom {
            name: "HD2",
            element: Element::H,
            bond_to: "CD2",
            angle_at: "CG",
            dihedral_to: "CB",
            bond_length_a: bond::C_H_SP2,
            bond_angle_rad: d(126.0),
            dihedral: DihedralValue::Fixed(d(0.0)),
        },
        SidechainAtom {
            name: "HE1",
            element: Element::H,
            bond_to: "CE1",
            angle_at: "ND1",
            dihedral_to: "CG",
            bond_length_a: bond::C_H_SP2,
            bond_angle_rad: d(125.8),
            dihedral: DihedralValue::Fixed(d(180.0)),
        },
    ],
    default_chi_rad: &[TRANS, d(-90.0)],
    has_amide_h: true,
    is_glycine: false,
};

// ===========================================================================
// Phenylalanine — Cβ-CH₂-phenyl (6-ring, planar).
// ===========================================================================

const PHE: ResidueTopology = ResidueTopology {
    sidechain: &[
        cb_atom(),
        ch2_chi("HB2", "CB", "CA", "N", 1, 1),
        ch2_chi("HB3", "CB", "CA", "N", 1, -1),
        SidechainAtom {
            name: "CG",
            element: Element::C,
            bond_to: "CB",
            angle_at: "CA",
            dihedral_to: "N",
            bond_length_a: 1.510,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Chi(1),
        },
        // Phenyl ring atoms: CD1, CD2 branch from CG (sp2 planar);
        // CE1 from CD1, CE2 from CD2; CZ closes the para position.
        SidechainAtom {
            name: "CD1",
            element: Element::C,
            bond_to: "CG",
            angle_at: "CB",
            dihedral_to: "CA",
            bond_length_a: bond::C_C_AROMATIC,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Chi(2),
        },
        SidechainAtom {
            name: "CD2",
            element: Element::C,
            bond_to: "CG",
            angle_at: "CB",
            dihedral_to: "CA",
            bond_length_a: bond::C_C_AROMATIC,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::ChiPlus(2, d(180.0)),
        },
        SidechainAtom {
            name: "CE1",
            element: Element::C,
            bond_to: "CD1",
            angle_at: "CG",
            dihedral_to: "CB",
            bond_length_a: bond::C_C_AROMATIC,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(180.0)),
        },
        SidechainAtom {
            name: "CE2",
            element: Element::C,
            bond_to: "CD2",
            angle_at: "CG",
            dihedral_to: "CB",
            bond_length_a: bond::C_C_AROMATIC,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(180.0)),
        },
        SidechainAtom {
            name: "CZ",
            element: Element::C,
            bond_to: "CE1",
            angle_at: "CD1",
            dihedral_to: "CG",
            bond_length_a: bond::C_C_AROMATIC,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(0.0)),
        },
        // Aromatic ring hydrogens.
        SidechainAtom {
            name: "HD1",
            element: Element::H,
            bond_to: "CD1",
            angle_at: "CG",
            dihedral_to: "CB",
            bond_length_a: bond::C_H_SP2,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(0.0)),
        },
        SidechainAtom {
            name: "HD2",
            element: Element::H,
            bond_to: "CD2",
            angle_at: "CG",
            dihedral_to: "CB",
            bond_length_a: bond::C_H_SP2,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(0.0)),
        },
        SidechainAtom {
            name: "HE1",
            element: Element::H,
            bond_to: "CE1",
            angle_at: "CD1",
            dihedral_to: "CG",
            bond_length_a: bond::C_H_SP2,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(180.0)),
        },
        SidechainAtom {
            name: "HE2",
            element: Element::H,
            bond_to: "CE2",
            angle_at: "CD2",
            dihedral_to: "CG",
            bond_length_a: bond::C_H_SP2,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(180.0)),
        },
        SidechainAtom {
            name: "HZ",
            element: Element::H,
            bond_to: "CZ",
            angle_at: "CE1",
            dihedral_to: "CD1",
            bond_length_a: bond::C_H_SP2,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(180.0)),
        },
    ],
    default_chi_rad: &[TRANS, d(90.0)],
    has_amide_h: true,
    is_glycine: false,
};

// ===========================================================================
// Tyrosine — Phe + para-OH (replaces HZ with OH-CZ).
// ===========================================================================

const TYR: ResidueTopology = ResidueTopology {
    sidechain: &[
        cb_atom(),
        ch2_chi("HB2", "CB", "CA", "N", 1, 1),
        ch2_chi("HB3", "CB", "CA", "N", 1, -1),
        SidechainAtom {
            name: "CG",
            element: Element::C,
            bond_to: "CB",
            angle_at: "CA",
            dihedral_to: "N",
            bond_length_a: 1.510,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Chi(1),
        },
        SidechainAtom {
            name: "CD1",
            element: Element::C,
            bond_to: "CG",
            angle_at: "CB",
            dihedral_to: "CA",
            bond_length_a: bond::C_C_AROMATIC,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Chi(2),
        },
        SidechainAtom {
            name: "CD2",
            element: Element::C,
            bond_to: "CG",
            angle_at: "CB",
            dihedral_to: "CA",
            bond_length_a: bond::C_C_AROMATIC,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::ChiPlus(2, d(180.0)),
        },
        SidechainAtom {
            name: "CE1",
            element: Element::C,
            bond_to: "CD1",
            angle_at: "CG",
            dihedral_to: "CB",
            bond_length_a: bond::C_C_AROMATIC,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(180.0)),
        },
        SidechainAtom {
            name: "CE2",
            element: Element::C,
            bond_to: "CD2",
            angle_at: "CG",
            dihedral_to: "CB",
            bond_length_a: bond::C_C_AROMATIC,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(180.0)),
        },
        SidechainAtom {
            name: "CZ",
            element: Element::C,
            bond_to: "CE1",
            angle_at: "CD1",
            dihedral_to: "CG",
            bond_length_a: bond::C_C_AROMATIC,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(0.0)),
        },
        // Phenol OH.
        SidechainAtom {
            name: "OH",
            element: Element::O,
            bond_to: "CZ",
            angle_at: "CE1",
            dihedral_to: "CD1",
            bond_length_a: 1.378,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(180.0)),
        },
        SidechainAtom {
            name: "HH",
            element: Element::H,
            bond_to: "OH",
            angle_at: "CZ",
            dihedral_to: "CE1",
            bond_length_a: bond::O_H,
            bond_angle_rad: d(108.0),
            dihedral: DihedralValue::Fixed(d(180.0)),
        },
        // Ring hydrogens.
        SidechainAtom {
            name: "HD1",
            element: Element::H,
            bond_to: "CD1",
            angle_at: "CG",
            dihedral_to: "CB",
            bond_length_a: bond::C_H_SP2,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(0.0)),
        },
        SidechainAtom {
            name: "HD2",
            element: Element::H,
            bond_to: "CD2",
            angle_at: "CG",
            dihedral_to: "CB",
            bond_length_a: bond::C_H_SP2,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(0.0)),
        },
        SidechainAtom {
            name: "HE1",
            element: Element::H,
            bond_to: "CE1",
            angle_at: "CD1",
            dihedral_to: "CG",
            bond_length_a: bond::C_H_SP2,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(180.0)),
        },
        SidechainAtom {
            name: "HE2",
            element: Element::H,
            bond_to: "CE2",
            angle_at: "CD2",
            dihedral_to: "CG",
            bond_length_a: bond::C_H_SP2,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(180.0)),
        },
    ],
    default_chi_rad: &[TRANS, d(90.0)],
    has_amide_h: true,
    is_glycine: false,
};

// ===========================================================================
// Tryptophan — Cβ-CH₂-indole.  Bicyclic 5+6 fused aromatic system.
// ===========================================================================

const TRP: ResidueTopology = ResidueTopology {
    sidechain: &[
        cb_atom(),
        ch2_chi("HB2", "CB", "CA", "N", 1, 1),
        ch2_chi("HB3", "CB", "CA", "N", 1, -1),
        SidechainAtom {
            name: "CG",
            element: Element::C,
            bond_to: "CB",
            angle_at: "CA",
            dihedral_to: "N",
            bond_length_a: 1.498,
            bond_angle_rad: angle::TETRAHEDRAL,
            dihedral: DihedralValue::Chi(1),
        },
        // 5-ring: CG-CD1-NE1-CE2-CD2-CG. Branch CD1 and CD2 off CG (sp2).
        SidechainAtom {
            name: "CD1",
            element: Element::C,
            bond_to: "CG",
            angle_at: "CB",
            dihedral_to: "CA",
            bond_length_a: 1.365,
            bond_angle_rad: d(127.0),
            dihedral: DihedralValue::Chi(2),
        },
        SidechainAtom {
            name: "CD2",
            element: Element::C,
            bond_to: "CG",
            angle_at: "CB",
            dihedral_to: "CA",
            bond_length_a: 1.433,
            bond_angle_rad: d(126.5),
            dihedral: DihedralValue::ChiPlus(2, d(180.0)),
        },
        SidechainAtom {
            name: "NE1",
            element: Element::N,
            bond_to: "CD1",
            angle_at: "CG",
            dihedral_to: "CB",
            bond_length_a: 1.374,
            bond_angle_rad: d(110.0),
            dihedral: DihedralValue::Fixed(d(180.0)),
        },
        SidechainAtom {
            name: "CE2",
            element: Element::C,
            bond_to: "NE1",
            angle_at: "CD1",
            dihedral_to: "CG",
            bond_length_a: 1.376,
            bond_angle_rad: d(109.0),
            dihedral: DihedralValue::Fixed(d(0.0)),
        },
        // 6-ring: CD2-CE3-CZ3-CH2-CZ2-CE2 (fused at CD2-CE2 edge).
        SidechainAtom {
            name: "CE3",
            element: Element::C,
            bond_to: "CD2",
            angle_at: "CG",
            dihedral_to: "CB",
            bond_length_a: 1.398,
            bond_angle_rad: d(133.9),
            dihedral: DihedralValue::Fixed(d(0.0)),
        },
        SidechainAtom {
            name: "CZ2",
            element: Element::C,
            bond_to: "CE2",
            angle_at: "NE1",
            dihedral_to: "CD1",
            bond_length_a: 1.394,
            bond_angle_rad: d(132.8),
            dihedral: DihedralValue::Fixed(d(180.0)),
        },
        SidechainAtom {
            name: "CZ3",
            element: Element::C,
            bond_to: "CE3",
            angle_at: "CD2",
            dihedral_to: "CG",
            bond_length_a: 1.382,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(0.0)),
        },
        SidechainAtom {
            name: "CH2",
            element: Element::C,
            bond_to: "CZ2",
            angle_at: "CE2",
            dihedral_to: "NE1",
            bond_length_a: 1.368,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(0.0)),
        },
        // Hydrogens.
        SidechainAtom {
            name: "HD1",
            element: Element::H,
            bond_to: "CD1",
            angle_at: "CG",
            dihedral_to: "CB",
            bond_length_a: bond::C_H_SP2,
            bond_angle_rad: d(125.0),
            dihedral: DihedralValue::Fixed(d(0.0)),
        },
        SidechainAtom {
            name: "HE1",
            element: Element::H,
            bond_to: "NE1",
            angle_at: "CD1",
            dihedral_to: "CG",
            bond_length_a: bond::N_H,
            bond_angle_rad: d(125.0),
            dihedral: DihedralValue::Fixed(d(180.0)),
        },
        SidechainAtom {
            name: "HE3",
            element: Element::H,
            bond_to: "CE3",
            angle_at: "CD2",
            dihedral_to: "CG",
            bond_length_a: bond::C_H_SP2,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(180.0)),
        },
        SidechainAtom {
            name: "HZ2",
            element: Element::H,
            bond_to: "CZ2",
            angle_at: "CE2",
            dihedral_to: "NE1",
            bond_length_a: bond::C_H_SP2,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(180.0)),
        },
        SidechainAtom {
            name: "HZ3",
            element: Element::H,
            bond_to: "CZ3",
            angle_at: "CE3",
            dihedral_to: "CD2",
            bond_length_a: bond::C_H_SP2,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(180.0)),
        },
        SidechainAtom {
            name: "HH2",
            element: Element::H,
            bond_to: "CH2",
            angle_at: "CZ2",
            dihedral_to: "CE2",
            bond_length_a: bond::C_H_SP2,
            bond_angle_rad: angle::TRIGONAL,
            dihedral: DihedralValue::Fixed(d(180.0)),
        },
    ],
    default_chi_rad: &[TRANS, d(90.0)],
    has_amide_h: true,
    is_glycine: false,
};

// ===========================================================================
// Helpers for repetitive atom patterns.
// ===========================================================================

const fn methyl_h(
    name: &'static str,
    parent_c: &'static str,
    parent_b: &'static str,
    parent_a: &'static str,
    dih: f64,
) -> SidechainAtom {
    SidechainAtom {
        name,
        element: Element::H,
        bond_to: parent_c,
        angle_at: parent_b,
        dihedral_to: parent_a,
        bond_length_a: bond::C_H_SP3,
        bond_angle_rad: angle::TETRAHEDRAL,
        dihedral: DihedralValue::Fixed(dih),
    }
}

/// One of two methylene hydrogens, with its dihedral pinned to a *fixed*
/// value. Use this only for terminal CH₂ groups where the carbon's other
/// substituent is not under χ control (e.g. Pro's CD, where the closing N is
/// fixed by ring topology).
const fn beta_h(
    name: &'static str,
    parent_c: &'static str,
    parent_b: &'static str,
    parent_a: &'static str,
    dih: f64,
) -> SidechainAtom {
    SidechainAtom {
        name,
        element: Element::H,
        bond_to: parent_c,
        angle_at: parent_b,
        dihedral_to: parent_a,
        bond_length_a: bond::C_H_SP3,
        bond_angle_rad: angle::TETRAHEDRAL,
        dihedral: DihedralValue::Fixed(dih),
    }
}

/// One of two methylene hydrogens whose position must track a χ-controlled
/// neighbour, so the four sp³ bonds at the carbon stay tetrahedrally
/// separated. Place at χ_n + 120° with sign = +1, or χ_n − 120° with sign = -1.
const fn ch2_chi(
    name: &'static str,
    parent_c: &'static str,
    parent_b: &'static str,
    parent_a: &'static str,
    chi_n: u8,
    sign: i32,
) -> SidechainAtom {
    let off = if sign >= 0 { d(120.0) } else { d(-120.0) };
    SidechainAtom {
        name,
        element: Element::H,
        bond_to: parent_c,
        angle_at: parent_b,
        dihedral_to: parent_a,
        bond_length_a: bond::C_H_SP3,
        bond_angle_rad: angle::TETRAHEDRAL,
        dihedral: DihedralValue::ChiPlus(chi_n, off),
    }
}

// ===========================================================================
// Public entry point.
// ===========================================================================

impl AminoAcid {
    pub const fn topology(self) -> &'static ResidueTopology {
        match self {
            AminoAcid::Gly => &GLY,
            AminoAcid::Ala => &ALA,
            AminoAcid::Val => &VAL,
            AminoAcid::Leu => &LEU,
            AminoAcid::Ile => &ILE,
            AminoAcid::Met => &MET,
            AminoAcid::Pro => &PRO,
            AminoAcid::Ser => &SER,
            AminoAcid::Thr => &THR,
            AminoAcid::Cys => &CYS,
            AminoAcid::Asn => &ASN,
            AminoAcid::Gln => &GLN,
            AminoAcid::Asp => &ASP,
            AminoAcid::Glu => &GLU,
            AminoAcid::Lys => &LYS,
            AminoAcid::Arg => &ARG,
            AminoAcid::His => &HIS,
            AminoAcid::Phe => &PHE,
            AminoAcid::Tyr => &TYR,
            AminoAcid::Trp => &TRP,
        }
    }

    /// Use the caller-supplied dihedrals to resolve a `DihedralValue` for this
    /// residue's side-chain placement.
    pub fn resolve_dihedral(&self, dv: DihedralValue, chi_rad: &[f64]) -> f64 {
        match dv {
            DihedralValue::Fixed(x) => x,
            DihedralValue::Chi(n) => chi_rad[(n - 1) as usize],
            DihedralValue::ChiPlus(n, off) => chi_rad[(n - 1) as usize] + off,
        }
    }
}
