//! Build an all-atom 3D structure from an amino-acid sequence.
//!
//! The backbone (N, HN, CA, HA(s), C, O) is placed by this module using
//! standard peptide geometry — these atoms are uniform across all residues
//! (Gly has two HAs and no side chain; Pro has no amide H). Side-chain
//! atoms are placed using each residue's [`ResidueTopology`] from the
//! `chem` crate.

use std::f64::consts::PI;

use chem::topology::angle as bb_angle;
use chem::{classify_rna, standard_ff, AminoAcid, Element, ForceField};
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
    // Bond lengths (matched to CHARMM27 par_all27_na.prm r₀ values so
    // the NeRF starting structure sits at the FF equilibrium).  CHARMM
    // atom-type pair in comment.
    pub const P_O5: f64 = 1.600; // ON2-P
    pub const P_OP: f64 = 1.480; // ON3-P
    pub const O5_C5: f64 = 1.440; // CN8B-ON2
    pub const C5_C4: f64 = 1.512; // CN7-CN8B
    pub const C4_O4: f64 = 1.480; // CN7-ON6B (furanose ring O)
    pub const C4_C3: f64 = 1.529; // CN7-CN7
    pub const C3_O3: f64 = 1.433; // CN7-ON2
    pub const C3_C2: f64 = 1.460; // CN7-CN7B (RNA-specific short bond)
    pub const C2_O2: f64 = 1.400; // CN7B-ON5 (2'-hydroxyl)
    pub const C2_C1: f64 = 1.450; // CN7B-CN7B (RNA-specific short bond)
    pub const C1_N: f64 = 1.456; // CN7B-NN2 / NN2B glycosidic
    pub const O3_P: f64 = 1.600; // ON2-P inter-residue
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
    // Backbone torsions: γ = 180° (trans) gives a properly extended
    // single-strand chain with bases spaced apart, instead of the
    // 54° (A-form gauche) value which placed consecutive bases on
    // top of each other (~10⁹ kJ/mol LJ clash, see the FIX.rna-bond-r0
    // commit message).  Other torsions kept at canonical RNA values.
    pub const ALPHA: f64 = deg(-68.0); // O3'p-P-O5'-C5'
    pub const BETA: f64 = deg(178.0); //  P-O5'-C5'-C4'
    pub const GAMMA: f64 = deg(180.0); // O5'-C5'-C4'-C3' (trans, extended)
    pub const DELTA: f64 = deg(82.0); //  C5'-C4'-C3'-O3'
    pub const EPSILON: f64 = deg(-153.0); // C4'-C3'-O3'-P(next)
    pub const ZETA: f64 = deg(-71.0); // C3'-O3'-P-O5'
    // Ribose-branch torsions solved via grid search (see
    // crates/geom/tests/rna_ribose_search.rs) for the γ=180° backbone
    // + the CHARMM-r₀ bond lengths above.  Closes the implicit
    // C1'-O4' ring-closure bond at 1.414 Å within search resolution.
    pub const O4_TORS: f64 = deg(111.0); // O5'-C5'-C4'-O4'
    pub const C2_TORS: f64 = deg(-159.0); // C5'-C4'-C3'-C2'
    pub const C1_TORS: f64 = deg(24.0); // C4'-C3'-C2'-C1'
    // O2_TORS retuned for the new C1' position so the 2'-hydroxyl
    // doesn't clash with C1' (was 0.61 Å with the old O2_TORS=48°).
    pub const O2_TORS: f64 = deg(-156.0); // C4'-C3'-C2'-O2'
    pub const CHI: f64 = deg(-160.0); //   C3'-C2'-C1'-N (anti)
    // Phosphate non-bridging O placements — rotated to keep them
    // clear of the previous residue's H3' (~3.9 Å apart vs ~1.2 Å
    // at the prior 120°/−120° symmetric placement, see the
    // `grid_search_op_torsions_for_no_h3_op_clash` ignored test).
    pub const OP1_TORS: f64 = deg(-170.0);
    pub const OP2_TORS: f64 = deg(125.0);

    // ---- Base ring geometry (canonical idealised bases) ----
    // Bond lengths and angles taken from standard nucleobase
    // crystallographic averages (Saenger 1984, ch. 4 and the AMBER
    // OL3 / CHARMM27 reference geometries which agree to <0.01 Å on
    // bond lengths and <0.5° on angles).

    // Purine (A / G) common ring lengths.  All `_BASE` to disambiguate
    // from the sugar/phosphate `C5_C4` etc. above.  The closure-bond
    // constants (N9-C8, N3-C4) are referenced only from the ring-
    // closure tests below — they aren't NeRF-placed, only used as
    // the canonical target the closure should land near.
    #[allow(dead_code)] pub const N9_C8_BASE: f64 = 1.371;
    pub const C8_N7_BASE: f64 = 1.305;
    pub const N7_C5_BASE: f64 = 1.388;
    pub const C5_C4_BASE: f64 = 1.409; // shared 5-ring / 6-ring edge
    pub const N9_C4_PUR: f64 = 1.380; // 5-ring closure
    pub const C5_C6_BASE: f64 = 1.404;
    pub const C6_N1_PUR: f64 = 1.346;
    pub const N1_C2_PUR: f64 = 1.353;
    pub const C2_N3_PUR: f64 = 1.337;
    #[allow(dead_code)] pub const N3_C4_PUR: f64 = 1.346; // 6-ring closure (target)
    // Purine 5-ring interior angles (sum = 540°). The N9 vertex angle
    // and the second N3-C4 ring-closure length are kept for the
    // closure-bond test (they aren't directly placed).
    #[allow(dead_code)] pub const ANG_C8_N9_C4: f64 = deg(105.8);
    pub const ANG_N9_C8_N7: f64 = deg(113.6);
    pub const ANG_C8_N7_C5: f64 = deg(103.7);
    pub const ANG_N7_C5_C4: f64 = deg(110.7);
    pub const ANG_C5_C4_N9: f64 = deg(106.2);
    // Purine 6-ring interior angles (sum = 720°).
    pub const ANG_C4_C5_C6: f64 = deg(117.2);
    pub const ANG_C5_C6_N1_PUR: f64 = deg(117.7);
    pub const ANG_C6_N1_C2_PUR: f64 = deg(117.8);
    pub const ANG_N1_C2_N3_PUR: f64 = deg(128.0);
    // Purine χ (anti) — sets the base orientation around C1'-N9.
    // Anchored as the dihedral C2'-C1'-N9-C4 used to NeRF-place C4.
    pub const PURINE_CHI_C4: f64 = deg(-120.0);
    // C1'-N9-C4 sp² angle = 360° - 105.8° (interior) - 126.4° = 127.8°
    // for symmetric placement (we use the 126.4° value which gives
    // C1'-N9-C8 = 127.8° on the other branch).
    pub const ANG_C1P_N9_C4: f64 = deg(126.4);

    // Adenine-specific.
    pub const C6_N6: f64 = 1.337;
    pub const ANG_C5_C6_N6: f64 = deg(123.5);
    // Guanine-specific.
    pub const C6_O6: f64 = 1.237; // carbonyl
    pub const C2_N2: f64 = 1.341; // exocyclic amine
    pub const ANG_C5_C6_O6: f64 = deg(128.5);
    pub const ANG_N1_C2_N2: f64 = deg(116.0);

    // Pyrimidine (C / U) common ring lengths.
    pub const N1_C2_PYR: f64 = 1.349;
    pub const C2_N3_PYR: f64 = 1.353;
    pub const N3_C4_PYR: f64 = 1.330;
    pub const C4_C5_PYR: f64 = 1.426;
    pub const C5_C6_PYR: f64 = 1.337;
    #[allow(dead_code)] pub const C6_N1_PYR: f64 = 1.367; // 6-ring closure (implicit)
    // Pyrimidine 6-ring interior angles (sum = 720°).
    pub const ANG_N1_C2_N3_PYR: f64 = deg(120.4);
    pub const ANG_C2_N3_C4_PYR: f64 = deg(119.6);
    pub const ANG_N3_C4_C5_PYR: f64 = deg(121.8);
    pub const ANG_C4_C5_C6_PYR: f64 = deg(117.4);
    pub const ANG_C5_C6_N1_PYR: f64 = deg(120.5);
    #[allow(dead_code)] pub const ANG_C2_N1_C6_PYR: f64 = deg(120.3); // closure
    // Pyrimidine χ (anti) — dihedral C2'-C1'-N1-C2 anchoring C2.
    pub const PYRIMIDINE_CHI_C2: f64 = deg(-120.0);
    pub const ANG_C1P_N1_C2: f64 = deg(120.0);

    // Cytosine-specific.
    pub const C2_O2_C: f64 = 1.240;
    pub const C4_N4_C: f64 = 1.337;
    pub const ANG_N3_C2_O2_C: f64 = deg(121.0);
    pub const ANG_N3_C4_N4_C: f64 = deg(118.0);
    // Uracil-specific.
    pub const C2_O2_U: f64 = 1.220;
    pub const C4_O4_U: f64 = 1.215;
    pub const ANG_N3_C2_O2_U: f64 = deg(122.0);
    pub const ANG_N3_C4_O4_U: f64 = deg(119.0);

    // Hydrogen bond lengths.
    pub const C_H_AROM: f64 = 1.080;
    pub const N_H_AROM: f64 = 1.010;
    pub const C_H_ALIPH: f64 = 1.090;
    pub const O_H: f64 = 0.957;
    pub const ANG_C_O_H: f64 = deg(108.0);
    pub const ANG_C_N_H: f64 = deg(120.0); // sp² amine / amide
}

/// Lookup the CHARMM27 r₀ for the bond `(a-b)` of nucleotide `nt`
/// directly from the loaded force field.  Used to give the base
/// placements per-nucleotide equilibrium lengths (the underlying
/// CHARMM r₀ varies — e.g. CN5-NN3A (A C2-N1) is 1.312 Å while
/// CN1-NN2G (G C6-N1) is 1.396 Å on the same `C6-N1` bond label).
/// Falls back to the supplied `fallback` if the FF doesn't have
/// the pair (shouldn't happen for any bond we actually place).
fn base_r0(ff: &ForceField, nt: chem::Nucleotide, a: &str, b: &str, fallback: f64) -> f64 {
    let (Some(ta), Some(tb)) = (classify_rna(nt, a), classify_rna(nt, b)) else {
        return fallback;
    };
    ff.bond(ta, tb).map(|p| p.r0).unwrap_or(fallback)
}

/// Place the fourth tetrahedral substituent at an sp³ centre that
/// already has three placed neighbours.  Returns the H position at
/// `bond_length` from `c`, lying on the ray opposite the sum of the
/// three placed-neighbour direction vectors — i.e. the unique
/// position that keeps all four pairwise angles near 109.47°.
fn place_sp3_one_h(c: Vec3, neighbours: [Vec3; 3], bond_length: f64) -> Vec3 {
    let d0 = (neighbours[0] - c).normalize();
    let d1 = (neighbours[1] - c).normalize();
    let d2 = (neighbours[2] - c).normalize();
    let sum = d0 + d1 + d2;
    let h_dir = -sum.normalize();
    c + h_dir * bond_length
}

/// Place two sp³ hydrogens at a centre with only two placed neighbours
/// (the C5' / -CH₂- case).  Both H's at 109.47° from each placed
/// neighbour and from each other, symmetric about the plane spanned
/// by the two known bond directions.
fn place_sp3_two_h(c: Vec3, n1: Vec3, n2: Vec3, bond_length: f64) -> (Vec3, Vec3) {
    let d1 = (n1 - c).normalize();
    let d2 = (n2 - c).normalize();
    let bisector = (d1 + d2).normalize();
    let normal = d1.cross(&d2).normalize();
    // cos(54.74°) = 1/√3 ≈ 0.5774, sin(54.74°) = √(2/3) ≈ 0.8165.
    const C: f64 = 0.577_350_269_189_625_8; // 1/√3
    const S: f64 = 0.816_496_580_927_726;   // √(2/3)
    let h1_dir = -bisector * C + normal * S;
    let h2_dir = -bisector * C - normal * S;
    (c + h1_dir * bond_length, c + h2_dir * bond_length)
}

/// Add the seven sugar-phosphate backbone hydrogens to `atoms`.
///
/// `c5` / `c4` / `c3` / `c2` / `c1` are the placed sugar carbons,
/// `o5` / `o4` / `o3` / `o2` the placed sugar oxygens, and `n_glyc`
/// the glycosidic nitrogen (N9 or N1).
#[allow(clippy::too_many_arguments)]
fn place_rna_backbone_hydrogens(
    atoms: &mut Vec<PlacedAtom>,
    o5: Vec3,
    c5: Vec3,
    c4: Vec3,
    o4: Vec3,
    c3: Vec3,
    o3: Vec3,
    c2: Vec3,
    o2: Vec3,
    c1: Vec3,
    n_glyc: Vec3,
) {
    let push = |atoms: &mut Vec<PlacedAtom>, name: &'static str, pos: Vec3| {
        atoms.push(PlacedAtom { name, element: Element::H, position: pos });
    };
    // C5' has two H's; both other H-bearing carbons are sp³ with 3
    // placed heavy neighbours each.
    let (h5p, h5pp) = place_sp3_two_h(c5, o5, c4, rna_ic::C_H_ALIPH);
    let h4p = place_sp3_one_h(c4, [c5, o4, c3], rna_ic::C_H_ALIPH);
    let h3p = place_sp3_one_h(c3, [c4, o3, c2], rna_ic::C_H_ALIPH);
    let h2p = place_sp3_one_h(c2, [c3, o2, c1], rna_ic::C_H_ALIPH);
    let h1p = place_sp3_one_h(c1, [c2, o4, n_glyc], rna_ic::C_H_ALIPH);
    // The 2'-hydroxyl H sits anti to C1' (gauche to C3'), a stable
    // RNA conformer.
    let ho2p = place_atom(c1, c2, o2, rna_ic::O_H, rna_ic::ANG_C_O_H, PI);

    push(atoms, "H5'", h5p);
    push(atoms, "H5''", h5pp);
    push(atoms, "H4'", h4p);
    push(atoms, "H3'", h3p);
    push(atoms, "HO2'", ho2p);
    push(atoms, "H2'", h2p);
    push(atoms, "H1'", h1p);
}

/// Place the purine ring + exocyclic substituents + ring/exocyclic
/// hydrogens for adenine or guanine.  `c2p` / `c1p` / `n9` are the
/// already-placed sugar atoms anchoring the base, `nt` selects the
/// adenine-vs-guanine exocyclic chemistry (N6 vs O6 + N2).
fn place_purine_base(
    atoms: &mut Vec<PlacedAtom>,
    nt: chem::Nucleotide,
    ff: &ForceField,
    c2p: Vec3,
    c1p: Vec3,
    n9: Vec3,
) {
    use chem::Nucleotide;
    // Per-nucleotide bond r₀ lookups (CHARMM27 r₀ for the actual
    // atom-type pair — these differ noticeably between A and G for
    // some bonds, e.g. C6-N1 is 1.342 Å on A vs 1.396 Å on G).
    let r = |a: &str, b: &str, fb: f64| base_r0(ff, nt, a, b, fb);
    // Walk the fused 5-/6-ring with all dihedrals in the ring plane.
    // C4 anchors the base orientation; C8 closes the 5-ring; C6/N1/
    // C2/N3 trace the 6-ring back to its closure at C4.
    let c4 = place_atom(c2p, c1p, n9, r("N9", "C4", rna_ic::N9_C4_PUR), rna_ic::ANG_C1P_N9_C4, rna_ic::PURINE_CHI_C4);
    let c5 = place_atom(c1p, n9, c4, r("C5", "C4", rna_ic::C5_C4_BASE), rna_ic::ANG_C5_C4_N9, PI);
    let n7 = place_atom(n9, c4, c5, r("N7", "C5", rna_ic::N7_C5_BASE), rna_ic::ANG_N7_C5_C4, 0.0);
    let c8 = place_atom(c4, c5, n7, r("C8", "N7", rna_ic::C8_N7_BASE), rna_ic::ANG_C8_N7_C5, 0.0);
    let c6 = place_atom(n7, c4, c5, r("C5", "C6", rna_ic::C5_C6_BASE), rna_ic::ANG_C4_C5_C6, PI);
    let n1 = place_atom(c4, c5, c6, r("C6", "N1", rna_ic::C6_N1_PUR), rna_ic::ANG_C5_C6_N1_PUR, 0.0);
    let c2 = place_atom(c5, c6, n1, r("N1", "C2", rna_ic::N1_C2_PUR), rna_ic::ANG_C6_N1_C2_PUR, 0.0);
    let n3 = place_atom(c6, n1, c2, r("C2", "N3", rna_ic::C2_N3_PUR), rna_ic::ANG_N1_C2_N3_PUR, 0.0);
    // Exocyclic substituent at C6 (N6 for A, O6 for G) — coplanar
    // with the 6-ring, anti to N1 across the C5-C6 bond.
    let (exo_c6_name, exo_c6_el, exo_c6_pos) = match nt {
        Nucleotide::Adenine => (
            "N6",
            Element::N,
            place_atom(n1, c5, c6, r("C6", "N6", rna_ic::C6_N6), rna_ic::ANG_C5_C6_N6, PI),
        ),
        Nucleotide::Guanine => (
            "O6",
            Element::O,
            place_atom(n1, c5, c6, r("C6", "O6", rna_ic::C6_O6), rna_ic::ANG_C5_C6_O6, PI),
        ),
        _ => unreachable!("place_purine_base called with non-purine"),
    };
    // Exocyclic N2 only on guanine, off C2.
    let g_n2 = if matches!(nt, Nucleotide::Guanine) {
        Some(place_atom(n3, n1, c2, r("C2", "N2", rna_ic::C2_N2), rna_ic::ANG_N1_C2_N2, PI))
    } else {
        None
    };

    let push_h = |atoms: &mut Vec<PlacedAtom>, name: &'static str, pos: Vec3| {
        atoms.push(PlacedAtom { name, element: Element::H, position: pos });
    };

    // H8 — sp² at C8, in plane, anti to N9 across N7-C8.
    let h8 = place_atom(n9, n7, c8, rna_ic::C_H_AROM, deg_from_120_sp2(rna_ic::ANG_N9_C8_N7), PI);
    // Adenine H2 on C2 (sp², between N1 and N3).
    let a_h2 = if matches!(nt, Nucleotide::Adenine) {
        Some(place_atom(c6, n1, c2, rna_ic::C_H_AROM, deg_from_120_sp2(rna_ic::ANG_N1_C2_N3_PUR), PI))
    } else { None };
    // Guanine N1-H (amide, sp²).
    let g_h1 = if matches!(nt, Nucleotide::Guanine) {
        Some(place_atom(c5, c6, n1, rna_ic::N_H_AROM, rna_ic::ANG_C_N_H, PI))
    } else { None };

    // Push heavy atoms in `Nucleotide::base_heavy_atoms()` canonical order.
    let push_heavy = |atoms: &mut Vec<PlacedAtom>, name: &'static str, el: Element, pos: Vec3| {
        atoms.push(PlacedAtom { name, element: el, position: pos });
    };
    match nt {
        Nucleotide::Adenine => {
            push_heavy(atoms, "C8", Element::C, c8);
            push_heavy(atoms, "N7", Element::N, n7);
            push_heavy(atoms, "C5", Element::C, c5);
            push_heavy(atoms, "C6", Element::C, c6);
            push_heavy(atoms, "N6", Element::N, exo_c6_pos);
            push_heavy(atoms, "N1", Element::N, n1);
            push_heavy(atoms, "C2", Element::C, c2);
            push_heavy(atoms, "N3", Element::N, n3);
            push_heavy(atoms, "C4", Element::C, c4);
            push_h(atoms, "H8", h8);
            // Exocyclic N6 amine — both H's coplanar with the ring,
            // separated by ~120°.
            let h61 = place_atom(c5, c6, exo_c6_pos, rna_ic::N_H_AROM, rna_ic::ANG_C_N_H, 0.0);
            let h62 = place_atom(c5, c6, exo_c6_pos, rna_ic::N_H_AROM, rna_ic::ANG_C_N_H, PI);
            push_h(atoms, "H61", h61);
            push_h(atoms, "H62", h62);
            push_h(atoms, "H2", a_h2.unwrap());
            let _ = exo_c6_name;
            let _ = exo_c6_el;
        }
        Nucleotide::Guanine => {
            push_heavy(atoms, "C8", Element::C, c8);
            push_heavy(atoms, "N7", Element::N, n7);
            push_heavy(atoms, "C5", Element::C, c5);
            push_heavy(atoms, "C6", Element::C, c6);
            push_heavy(atoms, "O6", Element::O, exo_c6_pos);
            push_heavy(atoms, "N1", Element::N, n1);
            push_heavy(atoms, "C2", Element::C, c2);
            push_heavy(atoms, "N2", Element::N, g_n2.unwrap());
            push_heavy(atoms, "N3", Element::N, n3);
            push_heavy(atoms, "C4", Element::C, c4);
            push_h(atoms, "H8", h8);
            push_h(atoms, "H1", g_h1.unwrap());
            let h21 = place_atom(n1, c2, g_n2.unwrap(), rna_ic::N_H_AROM, rna_ic::ANG_C_N_H, 0.0);
            let h22 = place_atom(n1, c2, g_n2.unwrap(), rna_ic::N_H_AROM, rna_ic::ANG_C_N_H, PI);
            push_h(atoms, "H21", h21);
            push_h(atoms, "H22", h22);
        }
        _ => unreachable!(),
    }
}

/// Place the pyrimidine 6-ring + exocyclic substituents + hydrogens
/// for cytosine or uracil.
fn place_pyrimidine_base(
    atoms: &mut Vec<PlacedAtom>,
    nt: chem::Nucleotide,
    ff: &ForceField,
    c2p: Vec3,
    c1p: Vec3,
    n1: Vec3,
) {
    use chem::Nucleotide;
    let r = |a: &str, b: &str, fb: f64| base_r0(ff, nt, a, b, fb);
    // 6-ring walk anchored on N1 (the glycosidic atom for pyrimidines).
    let c2 = place_atom(
        c2p, c1p, n1,
        r("N1", "C2", rna_ic::N1_C2_PYR), rna_ic::ANG_C1P_N1_C2, rna_ic::PYRIMIDINE_CHI_C2,
    );
    let n3 = place_atom(c1p, n1, c2, r("C2", "N3", rna_ic::C2_N3_PYR), rna_ic::ANG_N1_C2_N3_PYR, PI);
    let c4 = place_atom(n1, c2, n3, r("N3", "C4", rna_ic::N3_C4_PYR), rna_ic::ANG_C2_N3_C4_PYR, 0.0);
    let c5 = place_atom(c2, n3, c4, r("C4", "C5", rna_ic::C4_C5_PYR), rna_ic::ANG_N3_C4_C5_PYR, 0.0);
    let c6 = place_atom(n3, c4, c5, r("C5", "C6", rna_ic::C5_C6_PYR), rna_ic::ANG_C4_C5_C6_PYR, 0.0);
    // Exocyclic substituents.
    let push_h = |atoms: &mut Vec<PlacedAtom>, name: &'static str, pos: Vec3| {
        atoms.push(PlacedAtom { name, element: Element::H, position: pos });
    };
    let push_heavy = |atoms: &mut Vec<PlacedAtom>, name: &'static str, el: Element, pos: Vec3| {
        atoms.push(PlacedAtom { name, element: el, position: pos });
    };
    let o2 = match nt {
        Nucleotide::Cytosine => place_atom(n1, n3, c2, r("C2", "O2", rna_ic::C2_O2_C), rna_ic::ANG_N3_C2_O2_C, PI),
        Nucleotide::Uracil => place_atom(n1, n3, c2, r("C2", "O2", rna_ic::C2_O2_U), rna_ic::ANG_N3_C2_O2_U, PI),
        _ => unreachable!(),
    };
    // H5 bonded to C5 (sp²), in plane, anti to N3 across C4-C5.
    let h5 = place_atom(n3, c4, c5, rna_ic::C_H_AROM, deg_from_120_sp2(rna_ic::ANG_C4_C5_C6_PYR), PI);
    // H6 bonded to C6 (sp²), in plane, anti to C4 across C5-C6.
    let h6 = place_atom(c4, c5, c6, rna_ic::C_H_AROM, deg_from_120_sp2(rna_ic::ANG_C5_C6_N1_PYR), PI);

    match nt {
        Nucleotide::Cytosine => {
            let n4 = place_atom(n3, c5, c4, r("C4", "N4", rna_ic::C4_N4_C), rna_ic::ANG_N3_C4_N4_C, PI);
            push_heavy(atoms, "N1", Element::N, n1);
            push_heavy(atoms, "C2", Element::C, c2);
            push_heavy(atoms, "O2", Element::O, o2);
            push_heavy(atoms, "N3", Element::N, n3);
            push_heavy(atoms, "C4", Element::C, c4);
            push_heavy(atoms, "N4", Element::N, n4);
            push_heavy(atoms, "C5", Element::C, c5);
            push_heavy(atoms, "C6", Element::C, c6);
            let h41 = place_atom(c5, c4, n4, rna_ic::N_H_AROM, rna_ic::ANG_C_N_H, 0.0);
            let h42 = place_atom(c5, c4, n4, rna_ic::N_H_AROM, rna_ic::ANG_C_N_H, PI);
            push_h(atoms, "H41", h41);
            push_h(atoms, "H42", h42);
            push_h(atoms, "H5", h5);
            push_h(atoms, "H6", h6);
        }
        Nucleotide::Uracil => {
            // O4 carbonyl on C4 — anti to C2 across N3-C4 in plane.
            let o4 = place_atom(c2, n3, c4, r("C4", "O4", rna_ic::C4_O4_U), rna_ic::ANG_N3_C4_O4_U, PI);
            // N3-H amide (uracil only) — anchored at N3, in ring plane,
            // outside ring (anti to C5 across the C4-N3 axis).
            let h3 = place_atom(c5, c4, n3, rna_ic::N_H_AROM, rna_ic::ANG_C_N_H, PI);
            push_heavy(atoms, "N1", Element::N, n1);
            push_heavy(atoms, "C2", Element::C, c2);
            push_heavy(atoms, "O2", Element::O, o2);
            push_heavy(atoms, "N3", Element::N, n3);
            push_heavy(atoms, "C4", Element::C, c4);
            push_heavy(atoms, "O4", Element::O, o4);
            push_heavy(atoms, "C5", Element::C, c5);
            push_heavy(atoms, "C6", Element::C, c6);
            push_h(atoms, "H3", h3);
            push_h(atoms, "H5", h5);
            push_h(atoms, "H6", h6);
        }
        _ => unreachable!(),
    }
}

/// For an sp² atom with one fixed in-ring angle θ_ring, the two
/// out-of-ring substituents sit at (360° − θ_ring) / 2.  Used to
/// derive sp² C-H and C-X exocyclic angles from the interior ring
/// angle.
fn deg_from_120_sp2(ring_angle_rad: f64) -> f64 {
    (2.0 * PI - ring_angle_rad) / 2.0
}

/// Build an extended RNA chain (sugar-phosphate backbone + ribose ring
/// + glycosidic nitrogen + base ring + all hydrogens) from a
/// nucleotide sequence. Every residue is a `Monomer::Rna`.
pub fn build_extended_rna_chain(
    sequence: &[chem::Nucleotide],
) -> Result<Structure, BuildError> {
    use chem::Nucleotide;
    use crate::structure::Monomer;
    if sequence.is_empty() {
        return Err(BuildError::Empty);
    }
    // CHARMM27 r₀ table — read once, used by the base placement
    // helpers to look up per-nucleotide bond equilibrium distances.
    let ff = standard_ff();
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

        // ---- Backbone hydrogens ----
        place_rna_backbone_hydrogens(&mut atoms, o5, c5, c4, o4, c3, o3, c2, o2, c1, n);

        // ---- Base ring atoms (heavy + H), in PDB canonical order ----
        // The per-base helper pushes its own atoms, starting with the
        // glycosidic N9/N1 — matching `Nucleotide::base_heavy_atoms()`.
        match nt {
            Nucleotide::Adenine | Nucleotide::Guanine => {
                // N9 is the first base-heavy atom in canonical order;
                // push it before calling the purine ring walker.
                push(&mut atoms, "N9", Element::N, n);
                place_purine_base(&mut atoms, nt, ff, c2, c1, n);
            }
            Nucleotide::Cytosine | Nucleotide::Uracil => {
                // For pyrimidines the helper pushes N1 itself as part
                // of the canonical-order push, so we don't push it here.
                place_pyrimidine_base(&mut atoms, nt, ff, c2, c1, n);
            }
        }

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
    fn rna_chain_places_full_atom_roster_per_residue() {
        use chem::Nucleotide;
        let s = build_extended_rna_chain(&[
            Nucleotide::Adenine,
            Nucleotide::Uracil,
            Nucleotide::Guanine,
            Nucleotide::Cytosine,
        ])
        .unwrap();
        assert_eq!(s.residues.len(), 4);
        // Per-nucleotide totals (heavy backbone 12 + H backbone 7 +
        // heavy base + H base): A 33, U 30, G 34, C 31.
        for r in &s.residues {
            let nt = r.monomer.as_nucleotide().unwrap();
            assert_eq!(
                r.atoms.len(),
                nt.all_atoms().len(),
                "{nt:?}: built atom count != canonical roster"
            );
            // Every named atom from the canonical roster is present.
            for (name, _) in nt.all_atoms() {
                assert!(
                    r.position(name).is_some(),
                    "{nt:?} missing atom {name}"
                );
            }
        }
    }

    #[test]
    fn rna_purine_ring_atoms_coplanar() {
        // The fused 5-/6-ring purine should be flat — every atom within
        // a tight tolerance of the best-fit ring plane.
        use chem::Nucleotide;
        let s = build_extended_rna_chain(&[Nucleotide::Adenine]).unwrap();
        let r = &s.residues[0];
        let ring_names = ["N9", "C8", "N7", "C5", "C4", "C6", "N1", "C2", "N3"];
        let pts: Vec<Vec3> = ring_names.iter().map(|n| r.position(n).unwrap()).collect();
        let centroid: Vec3 = pts.iter().sum::<Vec3>() / pts.len() as f64;
        // Plane normal from the first three atoms.
        let n_hat = (pts[1] - pts[0]).cross(&(pts[2] - pts[0])).normalize();
        for (name, p) in ring_names.iter().zip(pts.iter()) {
            let offset = (p - centroid).dot(&n_hat).abs();
            assert!(offset < 0.05, "{name} {offset} Å off ring plane");
        }
    }

    #[test]
    fn rna_pyrimidine_ring_atoms_coplanar() {
        use chem::Nucleotide;
        let s = build_extended_rna_chain(&[Nucleotide::Uracil]).unwrap();
        let r = &s.residues[0];
        let ring_names = ["N1", "C2", "N3", "C4", "C5", "C6"];
        let pts: Vec<Vec3> = ring_names.iter().map(|n| r.position(n).unwrap()).collect();
        let centroid: Vec3 = pts.iter().sum::<Vec3>() / pts.len() as f64;
        let n_hat = (pts[1] - pts[0]).cross(&(pts[2] - pts[0])).normalize();
        for (name, p) in ring_names.iter().zip(pts.iter()) {
            let offset = (p - centroid).dot(&n_hat).abs();
            assert!(offset < 0.05, "{name} {offset} Å off ring plane");
        }
    }

    #[test]
    fn rna_no_clashes_in_built_chain() {
        // No two atoms (other than bonded pairs / 1-3 / 1-4) should be
        // closer than 0.7 Å in a freshly built single nucleotide.
        // Threshold matches the protein extended-chain test.
        use chem::Nucleotide;
        for nt in [
            Nucleotide::Adenine,
            Nucleotide::Uracil,
            Nucleotide::Guanine,
            Nucleotide::Cytosine,
        ] {
            let s = build_extended_rna_chain(&[nt]).unwrap();
            let atoms: Vec<&PlacedAtom> = s.iter_atoms().map(|(_, a)| a).collect();
            for i in 0..atoms.len() {
                for j in (i + 1)..atoms.len() {
                    let d = (atoms[i].position - atoms[j].position).norm();
                    assert!(
                        d > 0.7,
                        "{:?}: clash {} ↔ {}: {} Å",
                        nt,
                        atoms[i].name,
                        atoms[j].name,
                        d
                    );
                }
            }
        }
    }

    #[test]
    fn rna_pyrimidine_ring_closure_bond_is_reasonable() {
        // The 6-ring closure N1-C6 emerges from the ring walk, not
        // NeRF'd directly. Should land within 0.2 Å of canonical.
        use chem::Nucleotide;
        for nt in [Nucleotide::Cytosine, Nucleotide::Uracil] {
            let s = build_extended_rna_chain(&[nt]).unwrap();
            let r = &s.residues[0];
            let n1 = r.position("N1").unwrap();
            let c6 = r.position("C6").unwrap();
            let d = measure::distance(n1, c6);
            assert!(
                (d - rna_ic::C6_N1_PYR).abs() < 0.2,
                "{nt:?} 6-ring closure: N1-C6 = {d} Å, want ≈ {}",
                rna_ic::C6_N1_PYR
            );
        }
    }

    #[test]
    fn rna_base_heavy_bond_lengths_canonical() {
        // Spot-check one named base bond per nucleotide — confirms
        // the NeRF chain places atoms at the CHARMM27 r₀ now that
        // the builder looks up per-nucleotide r₀ from the FF.
        use chem::Nucleotide;
        // CHARMM27 r₀ values directly out of par_all27_na.prm:
        //   CN2-NN1   (A C6-N6 amine):    1.366
        //   CN1-ON1   (G C6=O6 carbonyl): 1.234
        //   CN1-ON1C  (C C2=O2):          1.245
        //   CN1-ON1   (U C4=O4):          1.234
        let cases: &[(Nucleotide, &str, &str, f64)] = &[
            (Nucleotide::Adenine, "C6", "N6", 1.366),
            (Nucleotide::Guanine, "C6", "O6", 1.234),
            (Nucleotide::Cytosine, "C2", "O2", 1.245),
            (Nucleotide::Uracil, "C4", "O4", 1.234),
        ];
        for &(nt, a, b, want) in cases {
            let s = build_extended_rna_chain(&[nt]).unwrap();
            let r = &s.residues[0];
            let pa = r.position(a).unwrap();
            let pb = r.position(b).unwrap();
            let d = measure::distance(pa, pb);
            assert!(
                (d - want).abs() < 5e-3,
                "{nt:?} {a}-{b} = {d} Å, want {want}"
            );
        }
    }

    #[test]
    fn rna_purine_ring_closure_bonds_are_reasonable() {
        // The 5-ring closure C8-N9 and 6-ring closure N3-C4 are
        // implicit (not NeRF'd) — they emerge from the geometry of
        // ring walking.  Tolerance: within 0.2 Å of the canonical
        // bond length.
        use chem::Nucleotide;
        let s = build_extended_rna_chain(&[Nucleotide::Adenine]).unwrap();
        let r = &s.residues[0];
        let n9 = r.position("N9").unwrap();
        let c8 = r.position("C8").unwrap();
        let n3 = r.position("N3").unwrap();
        let c4 = r.position("C4").unwrap();
        assert!(
            (measure::distance(n9, c8) - rna_ic::N9_C8_BASE).abs() < 0.2,
            "5-ring closure off: N9-C8 = {} Å",
            measure::distance(n9, c8)
        );
        assert!(
            (measure::distance(n3, c4) - rna_ic::N3_C4_PUR).abs() < 0.2,
            "6-ring closure off: N3-C4 = {} Å",
            measure::distance(n3, c4)
        );
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
