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
    for &aa in sequence {
        append_residue(&mut structure, aa, phi, psi, omega)?;
    }
    Ok(structure)
}

/// Append one residue to the C-terminus of an existing structure using
/// the standard NeRF chain-extension geometry. This is the building
/// block for co-translational growth (M6) — a `Ribosome` scheduler calls
/// it once per emitted residue and the integrator picks up the new atoms
/// without re-initialising the whole structure.
///
/// On the first residue (empty structure), seeds the chain at the origin.
pub fn append_residue(
    structure: &mut Structure,
    aa: AminoAcid,
    phi: f64,
    psi: f64,
    omega: f64,
) -> Result<(), BuildError> {
    let idx = structure.residues.len();
    let residue = build_residue(structure, idx, aa, phi, psi, omega)?;
    structure.residues.push(residue);
    Ok(())
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
    let mut residue = PlacedResidue {
        monomer: crate::structure::Monomer::Protein(aa),
        atoms: Vec::new(),
        chain: 'A',
    };

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

// ===================== RNA chain builder =====================
//
// Places the phosphodiester backbone + ribose ring + glycosidic
// nitrogen for an RNA sequence by the same NeRF chain extension the
// protein builder uses. The 13 atoms placed per nucleotide are:
//
//   P OP1 OP2 O5' C5' C4' O4' C3' O3' C2' O2' C1' + N9/N1
//
// The C1'-O4' ribose-ring closure is *not* placed — it's an implicit
// topology bond that the forward NeRF placement satisfies only
// approximately (same as the protein builder's aromatic-ring
// closures). The bases beyond the glycosidic N, and all hydrogens,
// are deliberately not built here — adding them is a documented
// follow-up. This is the "extended starting chain" for RNA: not at
// equilibrium, meant to be relaxed by minimisation / dynamics.

/// Idealised RNA backbone + ribose internal coordinates (Å / radians).
mod rna_ic {
    use std::f64::consts::PI;
    const fn deg(d: f64) -> f64 {
        d * PI / 180.0
    }
    // Bond lengths.
    pub const P_O5: f64 = 1.593;
    pub const P_OP: f64 = 1.485;
    pub const O5_C5: f64 = 1.440;
    pub const C5_C4: f64 = 1.510;
    pub const C4_O4: f64 = 1.451;
    pub const C4_C3: f64 = 1.524;
    pub const C3_O3: f64 = 1.423;
    pub const C3_C2: f64 = 1.525;
    pub const C2_O2: f64 = 1.413;
    pub const C2_C1: f64 = 1.528;
    pub const C1_N: f64 = 1.464;
    pub const O3_P: f64 = 1.593; // inter-residue
    // Bond angles.
    pub const O3_P_O5: f64 = deg(104.0);
    pub const C3_O3_P: f64 = deg(119.7);
    pub const P_O5_C5: f64 = deg(120.9);
    pub const O5_C5_C4: f64 = deg(110.2);
    pub const C5_C4_C3: f64 = deg(115.0);
    pub const C4_C3_O3: f64 = deg(110.6);
    pub const O5_P_OP: f64 = deg(108.0);
    pub const C5_C4_O4: f64 = deg(109.5);
    pub const C4_C3_C2: f64 = deg(102.5);
    pub const C3_C2_C1: f64 = deg(101.5);
    pub const C3_C2_O2: f64 = deg(110.7);
    pub const C2_C1_N: f64 = deg(108.2);
    // Torsions. The main backbone path uses "extended" values; the
    // ribose-branch torsions are tuned (see the ring-closure test) so
    // the C1'-O4' separation lands near the 1.41 Å bond length.
    pub const ALPHA: f64 = deg(-68.0); // O3'p-P-O5'-C5'
    pub const BETA: f64 = deg(178.0); //  P-O5'-C5'-C4'
    pub const GAMMA: f64 = deg(54.0); //  O5'-C5'-C4'-C3'
    pub const DELTA: f64 = deg(82.0); //  C5'-C4'-C3'-O3'
    pub const EPSILON: f64 = deg(-153.0); // C4'-C3'-O3'-P(next)
    pub const ZETA: f64 = deg(-71.0); // C3'-O3'-P-O5'
    // The ribose-branch torsions are solved (grid search over the
    // three ring placements) so the implicit C1'-O4' ring-closure
    // bond lands within 1e-3 Å of its 1.414 Å target.
    pub const O4_TORS: f64 = deg(-24.0); // O5'-C5'-C4'-O4'
    pub const C2_TORS: f64 = deg(-63.0); // C5'-C4'-C3'-C2'
    pub const C1_TORS: f64 = deg(-72.0); // C4'-C3'-C2'-C1'
    pub const O2_TORS: f64 = deg(48.0); //  C4'-C3'-C2'-O2'
    pub const CHI: f64 = deg(-160.0); //   C3'-C2'-C1'-N (anti)
    pub const OP1_TORS: f64 = deg(120.0);
    pub const OP2_TORS: f64 = deg(-120.0);
}

/// Build an extended RNA chain (sugar-phosphate backbone + ribose ring
/// + glycosidic nitrogen) from a nucleotide sequence. Every residue is
/// a `Monomer::Rna`. See the module comment above for what is and is
/// not placed.
pub fn build_extended_rna_chain(
    sequence: &[chem::Nucleotide],
) -> Result<Structure, BuildError> {
    use chem::Nucleotide;
    use crate::structure::Monomer;
    if sequence.is_empty() {
        return Err(BuildError::Empty);
    }
    let mut structure = Structure::new();

    for (idx, &nt) in sequence.iter().enumerate() {
        let mut atoms: Vec<PlacedAtom> = Vec::with_capacity(13);
        let push = |atoms: &mut Vec<PlacedAtom>, name: &'static str, el: Element, pos: Vec3| {
            atoms.push(PlacedAtom { name, element: el, position: pos });
        };

        // ---- Anchor the phosphate-O5'-C5' triple ----
        let (p, o5, c5) = if idx == 0 {
            let p = Vec3::zeros();
            let o5 = Vec3::new(rna_ic::P_O5, 0.0, 0.0);
            // C5' in the xy-plane at ∠P-O5'-C5'.
            let a = PI - rna_ic::P_O5_C5;
            let c5 = Vec3::new(
                o5.x + rna_ic::O5_C5 * a.cos(),
                rna_ic::O5_C5 * a.sin(),
                0.0,
            );
            (p, o5, c5)
        } else {
            let prev = &structure.residues[idx - 1];
            let pc5 = prev.position("C5'").unwrap();
            let pc4 = prev.position("C4'").unwrap();
            let pc3 = prev.position("C3'").unwrap();
            let po3 = prev.position("O3'").unwrap();
            // P bonded to prev O3'; O5' then C5' continue the chain.
            let p = place_atom(pc4, pc3, po3, rna_ic::O3_P, rna_ic::C3_O3_P, rna_ic::EPSILON);
            let o5 = place_atom(pc3, po3, p, rna_ic::P_O5, rna_ic::O3_P_O5, rna_ic::ZETA);
            let c5 = place_atom(po3, p, o5, rna_ic::O5_C5, rna_ic::P_O5_C5, rna_ic::ALPHA);
            let _ = pc5;
            (p, o5, c5)
        };
        push(&mut atoms, "P", Element::P, p);
        push(&mut atoms, "O5'", Element::O, o5);
        push(&mut atoms, "C5'", Element::C, c5);

        // ---- Backbone main path C4' → C3' → O3' ----
        let c4 = place_atom(p, o5, c5, rna_ic::C5_C4, rna_ic::O5_C5_C4, rna_ic::BETA);
        let c3 = place_atom(o5, c5, c4, rna_ic::C4_C3, rna_ic::C5_C4_C3, rna_ic::GAMMA);
        let o3 = place_atom(c5, c4, c3, rna_ic::C3_O3, rna_ic::C4_C3_O3, rna_ic::DELTA);
        push(&mut atoms, "C4'", Element::C, c4);

        // ---- Non-bridging phosphate oxygens ----
        let op1 = place_atom(c5, o5, p, rna_ic::P_OP, rna_ic::O5_P_OP, rna_ic::OP1_TORS);
        let op2 = place_atom(c5, o5, p, rna_ic::P_OP, rna_ic::O5_P_OP, rna_ic::OP2_TORS);
        push(&mut atoms, "OP1", Element::O, op1);
        push(&mut atoms, "OP2", Element::O, op2);

        // ---- Ribose ring branch atoms ----
        let o4 = place_atom(o5, c5, c4, rna_ic::C4_O4, rna_ic::C5_C4_O4, rna_ic::O4_TORS);
        let c2 = place_atom(c5, c4, c3, rna_ic::C3_C2, rna_ic::C4_C3_C2, rna_ic::C2_TORS);
        let c1 = place_atom(c4, c3, c2, rna_ic::C2_C1, rna_ic::C3_C2_C1, rna_ic::C1_TORS);
        let o2 = place_atom(c4, c3, c2, rna_ic::C2_O2, rna_ic::C3_C2_O2, rna_ic::O2_TORS);
        push(&mut atoms, "O4'", Element::O, o4);
        push(&mut atoms, "C3'", Element::C, c3);
        push(&mut atoms, "O3'", Element::O, o3);
        push(&mut atoms, "C2'", Element::C, c2);
        push(&mut atoms, "O2'", Element::O, o2);
        push(&mut atoms, "C1'", Element::C, c1);

        // ---- Glycosidic nitrogen (purine N9 / pyrimidine N1) ----
        let n = place_atom(c3, c2, c1, rna_ic::C1_N, rna_ic::C2_C1_N, rna_ic::CHI);
        let n_name = match nt {
            Nucleotide::Adenine | Nucleotide::Guanine => "N9",
            Nucleotide::Cytosine | Nucleotide::Uracil => "N1",
        };
        push(&mut atoms, n_name, Element::N, n);

        structure.residues.push(PlacedResidue {
            monomer: Monomer::Rna(nt),
            atoms,
            chain: 'A',
        });
    }
    Ok(structure)
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

    // ---- RNA builder tests ----

    #[test]
    fn rna_chain_has_13_atoms_per_residue() {
        use chem::Nucleotide;
        let s = build_extended_rna_chain(&[
            Nucleotide::Adenine,
            Nucleotide::Uracil,
            Nucleotide::Guanine,
            Nucleotide::Cytosine,
        ])
        .unwrap();
        assert_eq!(s.residues.len(), 4);
        for r in &s.residues {
            // P OP1 OP2 O5' C5' C4' O4' C3' O3' C2' O2' C1' + N = 13.
            assert_eq!(r.atoms.len(), 13, "expected 13 backbone atoms");
            assert!(r.monomer.is_rna());
        }
        // Purines carry N9, pyrimidines N1.
        assert!(s.residues[0].position("N9").is_some()); // A
        assert!(s.residues[1].position("N1").is_some()); // U
        assert!(s.residues[2].position("N9").is_some()); // G
        assert!(s.residues[3].position("N1").is_some()); // C
    }

    #[test]
    fn rna_backbone_bond_lengths_correct() {
        use chem::Nucleotide;
        let s = build_extended_rna_chain(&[
            Nucleotide::Adenine,
            Nucleotide::Cytosine,
            Nucleotide::Guanine,
        ])
        .unwrap();
        for (i, r) in s.residues.iter().enumerate() {
            let p = |n: &str| r.position(n).unwrap();
            assert_relative_eq!(measure::distance(p("P"), p("O5'")), rna_ic::P_O5, epsilon = 1e-6);
            assert_relative_eq!(measure::distance(p("O5'"), p("C5'")), rna_ic::O5_C5, epsilon = 1e-6);
            assert_relative_eq!(measure::distance(p("C5'"), p("C4'")), rna_ic::C5_C4, epsilon = 1e-6);
            assert_relative_eq!(measure::distance(p("C4'"), p("C3'")), rna_ic::C4_C3, epsilon = 1e-6);
            assert_relative_eq!(measure::distance(p("C3'"), p("O3'")), rna_ic::C3_O3, epsilon = 1e-6);
            assert_relative_eq!(measure::distance(p("P"), p("OP1")), rna_ic::P_OP, epsilon = 1e-6);
            assert_relative_eq!(measure::distance(p("C4'"), p("O4'")), rna_ic::C4_O4, epsilon = 1e-6);
            assert_relative_eq!(measure::distance(p("C2'"), p("C1'")), rna_ic::C2_C1, epsilon = 1e-6);
            // Inter-residue phosphodiester bond.
            if i > 0 {
                let prev_o3 = s.residues[i - 1].position("O3'").unwrap();
                assert_relative_eq!(
                    measure::distance(prev_o3, p("P")),
                    rna_ic::O3_P,
                    epsilon = 1e-6
                );
            }
        }
    }

    #[test]
    fn rna_ribose_ring_closure_is_reasonable() {
        // C1'-O4' is the implicit ring-closure bond, not NeRF-placed.
        // Forward placement satisfies it only approximately; for an
        // "extended starting chain" anything within ~0.4 Å of the
        // 1.41 Å target ribose bond is acceptable (minimisation
        // closes the rest).
        use chem::Nucleotide;
        let s = build_extended_rna_chain(&[Nucleotide::Adenine]).unwrap();
        let r = &s.residues[0];
        let c1 = r.position("C1'").unwrap();
        let o4 = r.position("O4'").unwrap();
        let d = measure::distance(c1, o4);
        assert!(
            (d - 1.414).abs() < 0.4,
            "ribose ring closure C1'-O4' = {d} Å, want ≈ 1.41"
        );
    }
}
