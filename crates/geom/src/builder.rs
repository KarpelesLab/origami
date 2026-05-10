//! Build an all-atom 3D structure from an amino-acid sequence.
//!
//! The backbone (N, HN, CA, HA(s), C, O) is placed by this module using
//! standard peptide geometry — these atoms are uniform across all residues
//! (Gly has two HAs and no side chain; Pro has no amide H). Side-chain
//! atoms are placed using each residue's [`ResidueTopology`] from the
//! `chem` crate.

use std::f64::consts::PI;

use chem::topology::angle as bb_angle;
use chem::{AminoAcid, Element};
use thiserror::Error;

use crate::nerf::place_atom;
use crate::structure::{PlacedAtom, PlacedResidue, Structure};
use crate::Vec3;

/// Default backbone torsions for an extended (β-strand-like) chain.
pub const DEFAULT_PHI: f64 = -120.0 * PI / 180.0;
pub const DEFAULT_PSI: f64 = 140.0 * PI / 180.0;
pub const DEFAULT_OMEGA: f64 = PI; // trans peptide bond

/// Standard L-amino-acid Cβ dihedral C-N-CA-CB. HA sits opposite at the
/// negative of this value to keep the chirality correct.
const CB_DIHEDRAL_RAD: f64 = -122.55 * PI / 180.0;

const PEPTIDE_C_N: f64 = 1.329;
const N_CA: f64 = 1.458;
const CA_C: f64 = 1.525;
const C_O: f64 = 1.231;
const N_H_AMIDE: f64 = 1.010;
const CA_HA: f64 = 1.090;

const C_N_CA_ANGLE: f64 = 121.7 * PI / 180.0;
const CA_C_O_ANGLE: f64 = 120.8 * PI / 180.0;
const HN_BOND_ANGLE: f64 = 119.0 * PI / 180.0; // ∠C(i-1)-N(i)-H
const HA_BOND_ANGLE: f64 = 109.0 * PI / 180.0; // ∠N-CA-HA

#[derive(Debug, Error)]
pub enum BuildError {
    #[error("residue {0} references atom {1:?} which has not been placed")]
    MissingAtom(usize, String),
    #[error("empty sequence")]
    Empty,
}

/// Build an extended chain using the default backbone torsions and each
/// residue's default χ rotamer.
pub fn build_extended_chain(sequence: &[AminoAcid]) -> Result<Structure, BuildError> {
    build_chain(sequence, DEFAULT_PHI, DEFAULT_PSI, DEFAULT_OMEGA)
}

/// Build a chain with caller-specified uniform backbone torsions.
pub fn build_chain(
    sequence: &[AminoAcid],
    phi: f64,
    psi: f64,
    omega: f64,
) -> Result<Structure, BuildError> {
    if sequence.is_empty() {
        return Err(BuildError::Empty);
    }
    let mut structure = Structure::new();
    for (idx, &aa) in sequence.iter().enumerate() {
        let residue = build_residue(&structure, idx, aa, phi, psi, omega)?;
        structure.residues.push(residue);
    }
    Ok(structure)
}

fn build_residue(
    structure: &Structure,
    idx: usize,
    aa: AminoAcid,
    phi: f64,
    psi: f64,
    omega: f64,
) -> Result<PlacedResidue, BuildError> {
    let topo = aa.topology();
    let mut residue = PlacedResidue { aa, atoms: Vec::new() };

    // ---------------- Backbone ----------------
    let (n_pos, ca_pos, c_pos) = if idx == 0 {
        // Anchor the first residue.
        let n = Vec3::zeros();
        let ca = Vec3::new(N_CA, 0.0, 0.0);
        // Place C in the xy-plane at angle ∠N-CA-C = 111.2°.
        let n_ca_c = bb_angle::N_CA_C;
        let dx = -CA_C * n_ca_c.cos(); // points from CA away from N along x
        let dy = CA_C * n_ca_c.sin(); // and "up" in y
        let c = Vec3::new(ca.x + dx, ca.y + dy, 0.0);
        residue.atoms.push(PlacedAtom { name: "N", element: Element::N, position: n });
        residue.atoms.push(PlacedAtom { name: "CA", element: Element::C, position: ca });
        residue.atoms.push(PlacedAtom { name: "C", element: Element::C, position: c });
        (n, ca, c)
    } else {
        let prev = &structure.residues[idx - 1];
        let prev_n = prev.position("N").unwrap();
        let prev_ca = prev.position("CA").unwrap();
        let prev_c = prev.position("C").unwrap();
        // N(i): bond to prev C, angle at prev CA, dihedral N(i-1)-CA(i-1)-C(i-1)-N(i) = ψ(i-1).
        let n = place_atom(prev_n, prev_ca, prev_c, PEPTIDE_C_N, bb_angle::CA_C_N, psi);
        // CA(i): bond to N(i), angle at prev C, dihedral ω(i).
        let ca = place_atom(prev_ca, prev_c, n, N_CA, C_N_CA_ANGLE, omega);
        // C(i): bond to CA(i), angle at N(i), dihedral φ(i).
        let c = place_atom(prev_c, n, ca, CA_C, bb_angle::N_CA_C, phi);
        residue.atoms.push(PlacedAtom { name: "N", element: Element::N, position: n });
        residue.atoms.push(PlacedAtom { name: "CA", element: Element::C, position: ca });
        residue.atoms.push(PlacedAtom { name: "C", element: Element::C, position: c });
        (n, ca, c)
    };

    // O(i): N-CA-C-O dihedral = ψ + 180° (O is trans to N(i+1) across C).
    let o = place_atom(n_pos, ca_pos, c_pos, C_O, CA_C_O_ANGLE, psi + PI);
    residue.atoms.push(PlacedAtom { name: "O", element: Element::O, position: o });

    // HN amide hydrogen — skipped for Pro and for the N-terminus.
    if topo.has_amide_h && idx > 0 {
        let prev = &structure.residues[idx - 1];
        let prev_c = prev.position("C").unwrap();
        let prev_o = prev.position("O").unwrap();
        // dihedral O(i-1)-C(i-1)-N(i)-H = 180° (H trans to O across C-N peptide bond).
        let h = place_atom(prev_o, prev_c, n_pos, N_H_AMIDE, HN_BOND_ANGLE, PI);
        residue.atoms.push(PlacedAtom { name: "H", element: Element::H, position: h });
    } else if topo.has_amide_h && idx == 0 {
        // First residue: place a single representative HN at the standard
        // angle. Real N-terminus has NH3⁺ (3 H's); a more complete treatment
        // is deferred.
        // dihedral C-N-CA where CA-N-H = 120° and dihedral C(i)-CA(i)-N(i)-H is set so H is opposite to C in the N–CA bond.
        let h_dihedral = PI; // H trans to C across N
        let h = place_atom(c_pos, ca_pos, n_pos, N_H_AMIDE, HN_BOND_ANGLE, h_dihedral);
        residue.atoms.push(PlacedAtom { name: "H", element: Element::H, position: h });
    }

    // HA: opposite side from CB.
    if topo.is_glycine {
        // Two HAs at ±122.55°.
        let ha2 = place_atom(c_pos, n_pos, ca_pos, CA_HA, HA_BOND_ANGLE, -CB_DIHEDRAL_RAD);
        let ha3 = place_atom(c_pos, n_pos, ca_pos, CA_HA, HA_BOND_ANGLE, CB_DIHEDRAL_RAD);
        residue.atoms.push(PlacedAtom { name: "HA2", element: Element::H, position: ha2 });
        residue.atoms.push(PlacedAtom { name: "HA3", element: Element::H, position: ha3 });
    } else {
        let ha = place_atom(c_pos, n_pos, ca_pos, CA_HA, HA_BOND_ANGLE, -CB_DIHEDRAL_RAD);
        residue.atoms.push(PlacedAtom { name: "HA", element: Element::H, position: ha });
    }

    // ---------------- Side chain ----------------
    let chi = topo.default_chi_rad;
    for atom_template in topo.sidechain {
        let parent_a = lookup(&residue, structure, idx, atom_template.dihedral_to)?;
        let parent_b = lookup(&residue, structure, idx, atom_template.angle_at)?;
        let parent_c = lookup(&residue, structure, idx, atom_template.bond_to)?;
        let dihedral = aa.resolve_dihedral(atom_template.dihedral, chi);
        let pos = place_atom(
            parent_a,
            parent_b,
            parent_c,
            atom_template.bond_length_a,
            atom_template.bond_angle_rad,
            dihedral,
        );
        residue.atoms.push(PlacedAtom {
            name: atom_template.name,
            element: atom_template.element,
            position: pos,
        });
    }

    Ok(residue)
}

fn lookup(
    current: &PlacedResidue,
    structure: &Structure,
    idx: usize,
    name: &str,
) -> Result<Vec3, BuildError> {
    if let Some(p) = current.position(name) {
        return Ok(p);
    }
    // Allow references to the previous residue's atoms if needed (currently
    // the side-chain templates only reference same-residue atoms, but this
    // future-proofs for cross-residue refs e.g. disulfides or ring closures).
    if idx > 0 {
        if let Some(p) = structure.residues[idx - 1].position(name) {
            return Ok(p);
        }
    }
    Err(BuildError::MissingAtom(idx, name.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use chem::topology::bond;

    use crate::measure;

    #[test]
    fn single_residue_builds() {
        let s = build_extended_chain(&[AminoAcid::Ala]).unwrap();
        assert_eq!(s.residues.len(), 1);
        // Ala backbone (N, CA, C, O, H, HA) = 6 atoms + side chain (CB, HB1, HB2, HB3) = 4 → 10 atoms.
        assert_eq!(s.residues[0].atoms.len(), 10);
    }

    #[test]
    fn glycine_has_two_ha() {
        let s = build_extended_chain(&[AminoAcid::Gly]).unwrap();
        let r = &s.residues[0];
        assert!(r.position("HA2").is_some());
        assert!(r.position("HA3").is_some());
        assert!(r.position("HA").is_none());
        assert!(r.position("CB").is_none());
    }

    #[test]
    fn proline_has_no_amide_h() {
        // Pro at position 1 (not first) should have no H atom.
        let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Pro]).unwrap();
        let pro = &s.residues[1];
        assert!(pro.position("H").is_none());
        assert!(pro.position("CD").is_some());
    }

    #[test]
    fn backbone_bond_lengths_are_correct() {
        let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala, AminoAcid::Ala]).unwrap();
        for i in 0..3 {
            let r = &s.residues[i];
            let n = r.position("N").unwrap();
            let ca = r.position("CA").unwrap();
            let c = r.position("C").unwrap();
            let o = r.position("O").unwrap();
            assert_relative_eq!(measure::distance(n, ca), N_CA, epsilon = 1e-9);
            assert_relative_eq!(measure::distance(ca, c), CA_C, epsilon = 1e-9);
            assert_relative_eq!(measure::distance(c, o), C_O, epsilon = 1e-9);
            if i > 0 {
                let prev_c = s.residues[i - 1].position("C").unwrap();
                assert_relative_eq!(measure::distance(prev_c, n), PEPTIDE_C_N, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn side_chain_bond_lengths_are_correct() {
        let s = build_extended_chain(&[AminoAcid::Leu]).unwrap();
        let r = &s.residues[0];
        let ca = r.position("CA").unwrap();
        let cb = r.position("CB").unwrap();
        let cg = r.position("CG").unwrap();
        let cd1 = r.position("CD1").unwrap();
        let cd2 = r.position("CD2").unwrap();
        assert_relative_eq!(measure::distance(ca, cb), bond::C_C, epsilon = 1e-9);
        assert_relative_eq!(measure::distance(cb, cg), bond::C_C, epsilon = 1e-9);
        assert_relative_eq!(measure::distance(cg, cd1), bond::C_C, epsilon = 1e-9);
        assert_relative_eq!(measure::distance(cg, cd2), bond::C_C, epsilon = 1e-9);
    }

    #[test]
    fn no_clashes_in_extended_chain() {
        // Extended Ala-Ala-Ala — no two atoms (other than bonded pairs)
        // should be closer than ~1.0 Å.
        let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala, AminoAcid::Ala]).unwrap();
        let atoms: Vec<&PlacedAtom> = s.iter_atoms().map(|(_, a)| a).collect();
        for i in 0..atoms.len() {
            for j in (i + 1)..atoms.len() {
                let d = (atoms[i].position - atoms[j].position).norm();
                assert!(
                    d > 0.7,
                    "clash between {} and {}: {} Å",
                    atoms[i].name,
                    atoms[j].name,
                    d
                );
            }
        }
    }
}
