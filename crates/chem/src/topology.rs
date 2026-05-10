//! Side-chain topology: how each amino-acid's side chain is built relative
//! to its backbone.
//!
//! Backbone (N, Cα, C, O, plus HN amide hydrogen and HA α-hydrogens) is
//! placed by the chain builder using standard peptide geometry — these atoms
//! are uniform across all residues (Gly has 2 HAs and no side chain; Pro
//! has no HN). Everything beyond Cα is described here.
//!
//! Each side-chain atom specifies a NeRF placement: it bonds to one parent
//! atom and uses two more (already-placed) atoms to define the bond angle
//! and dihedral. Dihedrals can be a fixed value, a side-chain χ angle, or a
//! χ ± fixed offset (used for branching from sp³ centers, e.g. Val's two
//! γ-methyls).

use crate::element::Element;

#[derive(Debug, Clone, Copy)]
pub struct SidechainAtom {
    pub name: &'static str,
    pub element: Element,
    /// Atom this is bonded to (NeRF parent c).
    pub bond_to: &'static str,
    /// Defines the bond angle ∠b-c-new (NeRF parent b).
    pub angle_at: &'static str,
    /// Defines the dihedral a-b-c-new (NeRF parent a).
    pub dihedral_to: &'static str,
    pub bond_length_a: f64,
    /// Bond angle ∠b-c-new in radians.
    pub bond_angle_rad: f64,
    pub dihedral: DihedralValue,
}

#[derive(Debug, Clone, Copy)]
pub enum DihedralValue {
    /// A constant dihedral (radians).
    Fixed(f64),
    /// Side-chain χ_n angle (1-indexed).
    Chi(u8),
    /// χ_n + offset (radians). Used for atoms branching off the same parent
    /// as a χ-controlled atom (e.g. Val's HB and CG2 are at χ₁ ± 120° from CG1).
    ChiPlus(u8, f64),
}

#[derive(Debug, Clone, Copy)]
pub struct ResidueTopology {
    pub sidechain: &'static [SidechainAtom],
    /// Default χ angles in radians (length = number of χ angles for this residue).
    pub default_chi_rad: &'static [f64],
    /// True for all residues except Pro (whose N is bonded to Cδ instead of H).
    pub has_amide_h: bool,
    /// True only for Gly (which has no Cβ; its α has two hydrogens).
    pub is_glycine: bool,
}

/// Helpers for the common bond lengths used across residues.
pub mod bond {
    pub const C_C: f64 = 1.530; // sp3 C-C single bond
    pub const C_C_AROMATIC: f64 = 1.390; // aromatic C-C
    pub const C_N: f64 = 1.470; // sp3 C-N
    pub const C_N_AROMATIC: f64 = 1.340; // aromatic / planar C-N
    pub const C_O: f64 = 1.420; // C-O alcohol
    pub const C_O_CARBOXYL: f64 = 1.250; // carboxylate (avg of C=O and C-O−)
    pub const C_S: f64 = 1.810; // C-S
    pub const S_H: f64 = 1.340;
    pub const O_H: f64 = 0.960;
    pub const N_H: f64 = 1.010;
    pub const C_H_SP3: f64 = 1.090;
    pub const C_H_SP2: f64 = 1.080; // aromatic C-H
}

pub mod angle {
    use std::f64::consts::PI;
    pub const TETRAHEDRAL: f64 = 109.471_220_634_490_69 * PI / 180.0;
    pub const TRIGONAL: f64 = 120.0 * PI / 180.0;
    pub const PEPTIDE_C_N_CA: f64 = 121.7 * PI / 180.0;
    pub const N_CA_C: f64 = 111.2 * PI / 180.0;
    pub const CA_C_O: f64 = 120.8 * PI / 180.0;
    pub const CA_C_N: f64 = 116.2 * PI / 180.0;
}
