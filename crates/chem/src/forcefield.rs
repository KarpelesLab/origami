//! CHARMM36m force-field parameter loader.
//!
//! At first use, parses the vendored `par_all36m_prot.prm` (Huang &
//! MacKerell 2016) into typed lookup tables. Parameters for atom types
//! we don't model are silently skipped.
//!
//! Units (CHARMM convention, kept verbatim):
//! - Bond force constants: kcal/mol/Å²
//! - Equilibrium bond length: Å
//! - Angle force constants: kcal/mol/rad²
//! - Equilibrium angle: degrees
//! - Dihedral / improper force constants: kcal/mol
//! - Phase / equilibrium dihedral: degrees
//! - Lennard-Jones ε: kcal/mol (CHARMM stores it as a negative number)
//! - Lennard-Jones Rmin/2: Å (note: this is half of the LJ minimum-energy
//!   separation; σ = Rmin / 2^(1/6))
//!
//! The energy crate is responsible for unit conversion to kJ/mol and radians.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::amino_acid::AminoAcid;
use crate::atom_type::AtomType;

#[derive(Debug, Clone, Copy)]
pub struct BondParams {
    pub k: f64,    // kcal/mol/Å²
    pub r0: f64,   // Å
}

#[derive(Debug, Clone, Copy)]
pub struct AngleParams {
    pub k: f64,        // kcal/mol/rad²
    pub theta0_deg: f64,
}

/// One periodic term in a multi-term dihedral expansion.
/// V = k × (1 + cos(n × χ − δ))
#[derive(Debug, Clone, Copy)]
pub struct DihedralTerm {
    pub k: f64,         // kcal/mol
    pub n: u32,         // multiplicity
    pub delta_deg: f64, // phase shift in degrees
}

#[derive(Debug, Clone, Copy)]
pub struct ImproperParams {
    pub k: f64,         // kcal/mol
    pub psi0_deg: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct NonbondedParams {
    pub epsilon: f64,   // kcal/mol (positive — CHARMM stores -eps; we negate)
    pub rmin_half: f64, // Å
    /// 1-4 special parameters when present in the file. CHARMM lets you
    /// override LJ for 1-4 (third-bonded) pairs.
    pub epsilon_14: Option<f64>,
    pub rmin_half_14: Option<f64>,
}

#[derive(Debug, Default)]
pub struct ForceField {
    bonds: HashMap<(AtomType, AtomType), BondParams>,
    angles: HashMap<(AtomType, AtomType, AtomType), AngleParams>,
    /// Specific dihedrals (all four atom types resolved).
    dihedrals: HashMap<(AtomType, AtomType, AtomType, AtomType), Vec<DihedralTerm>>,
    /// Wildcard dihedrals (X B C X form): keyed on the central pair.
    wildcard_dihedrals: HashMap<(AtomType, AtomType), Vec<DihedralTerm>>,
    /// Specific impropers (all four atom types resolved).
    impropers: HashMap<(AtomType, AtomType, AtomType, AtomType), ImproperParams>,
    /// Wildcard impropers: central atom + two off-centre wildcards (CC X X CT2 etc.).
    /// Keyed on (central, sole_specified_off_centre) — the order in the file is
    /// `central X X specific_off`. We store as (central, specific) → params.
    wildcard_impropers: HashMap<(AtomType, AtomType), ImproperParams>,
    nonbonded: HashMap<AtomType, NonbondedParams>,
    /// Per-(residue, atom-name) partial charge from the .rtf topology file.
    /// Atom names are stored in PDB v3.3 form (matching what our chain
    /// builder produces).
    partial_charges: HashMap<(AminoAcid, String), f64>,
}

impl ForceField {
    pub fn bond(&self, a: AtomType, b: AtomType) -> Option<&BondParams> {
        let key = canonical_pair(a, b);
        self.bonds.get(&key)
    }

    pub fn angle(&self, a: AtomType, b: AtomType, c: AtomType) -> Option<&AngleParams> {
        let key = canonical_triple(a, b, c);
        self.angles.get(&key)
    }

    pub fn dihedral(
        &self,
        a: AtomType,
        b: AtomType,
        c: AtomType,
        d: AtomType,
    ) -> Option<&[DihedralTerm]> {
        let key = canonical_quad(a, b, c, d);
        if let Some(terms) = self.dihedrals.get(&key) {
            return Some(terms.as_slice());
        }
        // Fall back to wildcard X-b-c-X.
        let central = canonical_pair(b, c);
        self.wildcard_dihedrals.get(&central).map(|t| t.as_slice())
    }

    pub fn improper(
        &self,
        a: AtomType,
        b: AtomType,
        c: AtomType,
        d: AtomType,
    ) -> Option<&ImproperParams> {
        // Try a few orderings; impropers in CHARMM are written central-first.
        let centrals = [b, a, c, d]; // try each as central
        let off_atoms = |central: AtomType| -> [AtomType; 3] {
            let mut others = [a, b, c, d]
                .iter()
                .copied()
                .filter(|t| *t != central)
                .collect::<Vec<_>>();
            // Pad to 3 just in case (won't happen normally).
            while others.len() < 3 {
                others.push(central);
            }
            [others[0], others[1], others[2]]
        };
        for &central in &centrals {
            let mut off = off_atoms(central);
            off.sort();
            let key = (off[0], central, off[1], off[2]);
            if let Some(p) = self.impropers.get(&key) {
                return Some(p);
            }
            // Wildcard: central + one specific off-atom (CC X X CT2).
            for &spec in &off {
                if let Some(p) = self.wildcard_impropers.get(&(central, spec)) {
                    return Some(p);
                }
            }
        }
        None
    }

    pub fn nonbonded(&self, t: AtomType) -> Option<&NonbondedParams> {
        self.nonbonded.get(&t)
    }

    pub fn partial_charge(&self, aa: AminoAcid, atom_name: &str) -> Option<f64> {
        self.partial_charges
            .get(&(aa, atom_name.to_owned()))
            .copied()
    }
}

fn canonical_pair(a: AtomType, b: AtomType) -> (AtomType, AtomType) {
    if a <= b { (a, b) } else { (b, a) }
}

fn canonical_triple(a: AtomType, b: AtomType, c: AtomType) -> (AtomType, AtomType, AtomType) {
    if a <= c { (a, b, c) } else { (c, b, a) }
}

fn canonical_quad(
    a: AtomType,
    b: AtomType,
    c: AtomType,
    d: AtomType,
) -> (AtomType, AtomType, AtomType, AtomType) {
    if (b, a) <= (c, d) {
        (a, b, c, d)
    } else {
        (d, c, b, a)
    }
}

/// Get the bundled CHARMM36m force field, parsed once on first call.
pub fn standard() -> &'static ForceField {
    static FF: OnceLock<ForceField> = OnceLock::new();
    FF.get_or_init(|| {
        let par = include_str!("../../../data/charmm36/par_all36m_prot.prm");
        let rtf = include_str!("../../../data/charmm36/top_all36_prot.rtf");
        let mut ff = parse(par);
        parse_rtf_charges(rtf, &mut ff);
        ff
    })
}

/// Parse a CHARMM .prm file. Wildcard atom-type tokens (`X`) are recognised.
/// Lines whose atom-type tokens don't match anything in our [`AtomType`]
/// enum are silently skipped — the file contains parameters for many types
/// we don't model.
pub fn parse(text: &str) -> ForceField {
    let mut ff = ForceField::default();
    let mut section = Section::None;

    for raw in text.lines() {
        // Drop comments.
        let line = match raw.find('!') {
            Some(idx) => &raw[..idx],
            None => raw,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Section headers.
        if let Some(s) = match_section_header(trimmed) {
            section = s;
            continue;
        }

        // Skip lines that look like preamble or directives we don't parse
        // (e.g., the `cutnb 14.0 ctofnb...` line continuing NONBONDED).
        if section == Section::None {
            continue;
        }

        match section {
            Section::Bonds => parse_bond_line(trimmed, &mut ff),
            Section::Angles => parse_angle_line(trimmed, &mut ff),
            Section::Dihedrals => parse_dihedral_line(trimmed, &mut ff),
            Section::Impropers => parse_improper_line(trimmed, &mut ff),
            Section::Nonbonded => parse_nonbonded_line(trimmed, &mut ff),
            // Sections we ignore.
            Section::Cmap | Section::Hbond | Section::None => {}
            Section::End => break,
        }
    }

    ff
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Section {
    None,
    Bonds,
    Angles,
    Dihedrals,
    Impropers,
    Cmap,
    Nonbonded,
    Hbond,
    End,
}

fn match_section_header(line: &str) -> Option<Section> {
    // The header may be just the keyword or the keyword followed by config
    // flags (e.g. "NONBONDED nbxmod  5 atom cdiel ...").
    let first = line.split_ascii_whitespace().next()?;
    match first {
        "BONDS" => Some(Section::Bonds),
        "ANGLES" => Some(Section::Angles),
        "DIHEDRALS" => Some(Section::Dihedrals),
        "IMPROPER" | "IMPROPERS" => Some(Section::Impropers),
        "CMAP" => Some(Section::Cmap),
        "NONBONDED" => Some(Section::Nonbonded),
        "HBOND" => Some(Section::Hbond),
        "END" => Some(Section::End),
        _ => None,
    }
}

fn parse_atom(token: &str) -> AtomTypeOrWildcard {
    if token == "X" || token == "x" {
        AtomTypeOrWildcard::Wildcard
    } else {
        match AtomType::from_charmm_name(token) {
            Some(t) => AtomTypeOrWildcard::Specific(t),
            None => AtomTypeOrWildcard::Unknown,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum AtomTypeOrWildcard {
    Specific(AtomType),
    Wildcard,
    Unknown, // Atom type not in our enum — caller should skip the line.
}

fn parse_bond_line(line: &str, ff: &mut ForceField) {
    let tokens: Vec<&str> = line.split_ascii_whitespace().collect();
    if tokens.len() < 4 {
        return;
    }
    let (AtomTypeOrWildcard::Specific(a), AtomTypeOrWildcard::Specific(b)) =
        (parse_atom(tokens[0]), parse_atom(tokens[1])) else { return };
    let Ok(k) = tokens[2].parse::<f64>() else { return };
    let Ok(r0) = tokens[3].parse::<f64>() else { return };
    let key = canonical_pair(a, b);
    ff.bonds.entry(key).or_insert(BondParams { k, r0 });
}

fn parse_angle_line(line: &str, ff: &mut ForceField) {
    let tokens: Vec<&str> = line.split_ascii_whitespace().collect();
    if tokens.len() < 5 {
        return;
    }
    let (
        AtomTypeOrWildcard::Specific(a),
        AtomTypeOrWildcard::Specific(b),
        AtomTypeOrWildcard::Specific(c),
    ) = (
        parse_atom(tokens[0]),
        parse_atom(tokens[1]),
        parse_atom(tokens[2]),
    ) else {
        return;
    };
    let Ok(k) = tokens[3].parse::<f64>() else { return };
    let Ok(theta0_deg) = tokens[4].parse::<f64>() else { return };
    let key = canonical_triple(a, b, c);
    ff.angles.entry(key).or_insert(AngleParams { k, theta0_deg });
}

fn parse_dihedral_line(line: &str, ff: &mut ForceField) {
    let tokens: Vec<&str> = line.split_ascii_whitespace().collect();
    if tokens.len() < 7 {
        return;
    }
    let a = parse_atom(tokens[0]);
    let b = parse_atom(tokens[1]);
    let c = parse_atom(tokens[2]);
    let d = parse_atom(tokens[3]);
    let Ok(k) = tokens[4].parse::<f64>() else { return };
    let Ok(n) = tokens[5].parse::<u32>() else { return };
    let Ok(delta_deg) = tokens[6].parse::<f64>() else { return };
    let term = DihedralTerm { k, n, delta_deg };
    match (a, b, c, d) {
        (
            AtomTypeOrWildcard::Specific(a),
            AtomTypeOrWildcard::Specific(b),
            AtomTypeOrWildcard::Specific(c),
            AtomTypeOrWildcard::Specific(d),
        ) => {
            let key = canonical_quad(a, b, c, d);
            ff.dihedrals.entry(key).or_default().push(term);
        }
        (
            AtomTypeOrWildcard::Wildcard,
            AtomTypeOrWildcard::Specific(b),
            AtomTypeOrWildcard::Specific(c),
            AtomTypeOrWildcard::Wildcard,
        ) => {
            let key = canonical_pair(b, c);
            ff.wildcard_dihedrals.entry(key).or_default().push(term);
        }
        _ => {} // Other wildcard patterns aren't used in CHARMM36m for proteins.
    }
}

fn parse_improper_line(line: &str, ff: &mut ForceField) {
    let tokens: Vec<&str> = line.split_ascii_whitespace().collect();
    if tokens.len() < 7 {
        return;
    }
    let a = parse_atom(tokens[0]);
    let b = parse_atom(tokens[1]);
    let c = parse_atom(tokens[2]);
    let d = parse_atom(tokens[3]);
    let Ok(k) = tokens[4].parse::<f64>() else { return };
    // tokens[5] is a placeholder (always 0)
    let Ok(psi0_deg) = tokens[6].parse::<f64>() else { return };
    let params = ImproperParams { k, psi0_deg };
    match (a, b, c, d) {
        (
            AtomTypeOrWildcard::Specific(central),
            AtomTypeOrWildcard::Wildcard,
            AtomTypeOrWildcard::Wildcard,
            AtomTypeOrWildcard::Specific(spec),
        ) => {
            // CHARMM convention: first atom is the central sp² atom.
            ff.wildcard_impropers
                .entry((central, spec))
                .or_insert(params);
        }
        (
            AtomTypeOrWildcard::Specific(a),
            AtomTypeOrWildcard::Specific(b),
            AtomTypeOrWildcard::Specific(c),
            AtomTypeOrWildcard::Specific(d),
        ) => {
            // Central atom is first in CHARMM impropers; canonicalise off-atoms.
            let mut off = [b, c, d];
            off.sort();
            let key = (off[0], a, off[1], off[2]);
            ff.impropers.entry(key).or_insert(params);
        }
        _ => {}
    }
}

fn parse_nonbonded_line(line: &str, ff: &mut ForceField) {
    // Skip the NONBONDED config-continuation line ("cutnb ... wmin 1.5").
    if line.contains("cutnb") || line.contains("ctofnb") {
        return;
    }
    let tokens: Vec<&str> = line.split_ascii_whitespace().collect();
    if tokens.len() < 4 {
        return;
    }
    let AtomTypeOrWildcard::Specific(t) = parse_atom(tokens[0]) else {
        return;
    };
    // tokens[1] is "ignored" (always 0 in CHARMM)
    let Ok(eps_signed) = tokens[2].parse::<f64>() else { return };
    let Ok(rmin_half) = tokens[3].parse::<f64>() else { return };
    let mut params = NonbondedParams {
        epsilon: -eps_signed, // CHARMM stores -ε, we want positive ε
        rmin_half,
        epsilon_14: None,
        rmin_half_14: None,
    };
    if tokens.len() >= 7 {
        // 1-4 LJ parameters present.
        if let (Ok(eps14), Ok(rmin14)) =
            (tokens[5].parse::<f64>(), tokens[6].parse::<f64>())
        {
            params.epsilon_14 = Some(-eps14);
            params.rmin_half_14 = Some(rmin14);
        }
    }
    ff.nonbonded.entry(t).or_insert(params);
}

/// Parse the RESI blocks of a CHARMM topology .rtf file and populate the
/// per-atom partial charges. Only the standard 20 residues are loaded;
/// HSE, HSP, and patch entries are ignored. Atom names from CHARMM are
/// translated to PDB v3.3 conventions (HN→H, methylene H pairs renumbered,
/// Ile CD→CD1 etc.) so callers can look up by the same names our chain
/// builder uses.
fn parse_rtf_charges(text: &str, ff: &mut ForceField) {
    let mut current: Option<AminoAcid> = None;
    for raw in text.lines() {
        let line = match raw.find('!') {
            Some(idx) => &raw[..idx],
            None => raw,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut tokens = trimmed.split_ascii_whitespace();
        let head = match tokens.next() {
            Some(h) => h,
            None => continue,
        };
        if head == "RESI" {
            let name = tokens.next().unwrap_or("");
            current = match name {
                "ALA" => Some(AminoAcid::Ala),
                "ARG" => Some(AminoAcid::Arg),
                "ASN" => Some(AminoAcid::Asn),
                "ASP" => Some(AminoAcid::Asp),
                "CYS" => Some(AminoAcid::Cys),
                "GLN" => Some(AminoAcid::Gln),
                "GLU" => Some(AminoAcid::Glu),
                "GLY" => Some(AminoAcid::Gly),
                "HSD" => Some(AminoAcid::His), // default neutral His tautomer
                "ILE" => Some(AminoAcid::Ile),
                "LEU" => Some(AminoAcid::Leu),
                "LYS" => Some(AminoAcid::Lys),
                "MET" => Some(AminoAcid::Met),
                "PHE" => Some(AminoAcid::Phe),
                "PRO" => Some(AminoAcid::Pro),
                "SER" => Some(AminoAcid::Ser),
                "THR" => Some(AminoAcid::Thr),
                "TRP" => Some(AminoAcid::Trp),
                "TYR" => Some(AminoAcid::Tyr),
                "VAL" => Some(AminoAcid::Val),
                _ => None, // HSE, HSP, ALAD, CYM, patches → skip
            };
            continue;
        }
        if head == "PRES" {
            // Patch residue — skip until the next RESI.
            current = None;
            continue;
        }
        if head != "ATOM" {
            continue;
        }
        let Some(aa) = current else { continue };
        let charmm_name = tokens.next().unwrap_or("");
        let _atom_type = tokens.next().unwrap_or("");
        let charge_str = tokens.next().unwrap_or("");
        let Ok(charge) = charge_str.parse::<f64>() else { continue };
        let pdb_name = charmm_to_pdb_atom_name(aa, charmm_name);
        ff.partial_charges.insert((aa, pdb_name.to_owned()), charge);
    }
}

/// Translate a CHARMM atom name to the PDB v3.3 form our chain builder uses.
///
/// Most names are identical. Three classes of mismatch:
/// 1. `HN` (CHARMM amide hydrogen) → `H` (PDB).
/// 2. CH₂ groups: CHARMM names the two hydrogens with `1`/`2` suffixes;
///    PDB v3.3 uses `2`/`3`. Per-residue, since the relevant CH₂ atoms vary.
/// 3. Isoleucine: CHARMM names the lone δ-carbon `CD` with hydrogens `HD1/2/3`
///    and the γ-CH₂ hydrogens `HG11/12`; PDB v3.3 uses `CD1`, `HD11/12/13`,
///    `HG12/13`.
fn charmm_to_pdb_atom_name(aa: AminoAcid, charmm: &str) -> &'static str {
    use AminoAcid::*;

    // Universal: amide H.
    if charmm == "HN" {
        return "H";
    }

    match (aa, charmm) {
        // Glycine α-hydrogens
        (Gly, "HA1") => "HA2",
        (Gly, "HA2") => "HA3",

        // Isoleucine: CD → CD1; HD1/HD2/HD3 → HD11/HD12/HD13;
        // HG11/HG12 → HG12/HG13.
        (Ile, "CD") => "CD1",
        (Ile, "HD1") => "HD11",
        (Ile, "HD2") => "HD12",
        (Ile, "HD3") => "HD13",
        (Ile, "HG11") => "HG12",
        (Ile, "HG12") => "HG13",

        // Serine hydroxyl, Cysteine thiol
        (Ser, "HG1") => "HG",
        (Cys, "HG1") => "HG",

        // CB methylene shift (residues whose Cβ has 2 hydrogens):
        (Leu | Met | Pro | Ser | Cys | Asn | Gln | Asp | Glu
            | Lys | Arg | His | Phe | Tyr | Trp, "HB1") => "HB2",
        (Leu | Met | Pro | Ser | Cys | Asn | Gln | Asp | Glu
            | Lys | Arg | His | Phe | Tyr | Trp, "HB2") => "HB3",

        // CG methylene shift (residues whose Cγ has 2 hydrogens):
        (Met | Pro | Gln | Glu | Lys | Arg, "HG1") => "HG2",
        (Met | Pro | Gln | Glu | Lys | Arg, "HG2") => "HG3",

        // CD methylene shift:
        (Pro | Lys | Arg, "HD1") => "HD2",
        (Pro | Lys | Arg, "HD2") => "HD3",

        // CE methylene shift (Lys CE):
        (Lys, "HE1") => "HE2",
        (Lys, "HE2") => "HE3",

        // Everything else: the names already agree, but we need a static
        // reference. Map back through a static catalogue of PDB names. We
        // achieve this by listing the unmodified atom names in a static
        // table — but for simplicity, we just leak via a lookup: the most
        // common names are present in our topology data, so we can borrow
        // those.
        _ => return_static_name(aa, charmm),
    }
}

/// Return a `&'static str` for an atom name that doesn't need translation —
/// looks it up in the residue's topology so we get a `'static` reference
/// matching what the chain builder uses. Backbone atoms are returned via a
/// hardcoded match; side-chain atoms via the topology table.
fn return_static_name(aa: AminoAcid, charmm: &str) -> &'static str {
    // Backbone names are universal.
    match charmm {
        "N" => return "N",
        "CA" => return "CA",
        "C" => return "C",
        "O" => return "O",
        "HA" => return "HA",
        _ => {}
    }
    // Side-chain: search the residue's topology for a matching name.
    for sc in aa.topology().sidechain {
        if sc.name == charmm {
            return sc.name;
        }
    }
    // Not found — return a sentinel (the caller will silently skip storing
    // a charge for an atom we don't model).
    ""
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_loads() {
        let ff = standard();
        // We expect a non-trivial parameter table.
        assert!(!ff.bonds.is_empty());
        assert!(!ff.angles.is_empty());
        assert!(!ff.dihedrals.is_empty());
        assert!(!ff.nonbonded.is_empty());
    }

    #[test]
    fn known_bond_params() {
        // CT1-CT3 (sp3 C with 1 H bonded to a methyl): 222.5 kcal/mol/Å², r0 = 1.538 Å.
        let ff = standard();
        let p = ff.bond(AtomType::CT1, AtomType::CT3).expect("CT1-CT3 bond");
        assert!((p.k - 222.5).abs() < 1.0);
        assert!((p.r0 - 1.538).abs() < 0.05);
    }

    #[test]
    fn known_angle_params() {
        // N-CT1-C: backbone CA angle, has standard CHARMM value.
        let ff = standard();
        let p = ff.angle(AtomType::NH1, AtomType::CT1, AtomType::C).expect("NH1-CT1-C");
        assert!(p.k > 0.0);
        assert!((p.theta0_deg - 110.0).abs() < 15.0);
    }

    #[test]
    fn known_nonbonded_params() {
        let ff = standard();
        let c = ff.nonbonded(AtomType::C).expect("C nonbonded");
        // CHARMM36m C atom: -ε = -0.11, Rmin/2 = 2.0 Å.
        assert!((c.epsilon - 0.11).abs() < 0.005);
        assert!((c.rmin_half - 2.0).abs() < 0.05);
        // H polar: very small ε.
        let h = ff.nonbonded(AtomType::H).expect("H nonbonded");
        assert!(h.epsilon < 0.1);
    }

    #[test]
    fn dihedral_with_wildcard_fallback() {
        // Most CT3 / CT2 dihedrals are wildcard X-CT3-CT2-X form.
        let ff = standard();
        // Should resolve via wildcard.
        let _ = ff.dihedral(AtomType::HA3, AtomType::CT3, AtomType::CT2, AtomType::HA2)
            .expect("HA3-CT3-CT2-HA2 via wildcard");
    }

    #[test]
    fn partial_charges_loaded() {
        let ff = standard();
        // Backbone N and CA charges are well known.
        let n_charge = ff.partial_charge(AminoAcid::Ala, "N").expect("Ala N charge");
        assert!((n_charge - (-0.47)).abs() < 0.01);
        let ca_charge = ff.partial_charge(AminoAcid::Ala, "CA").expect("Ala CA charge");
        assert!((ca_charge - 0.07).abs() < 0.01);
        // The amide H in PDB-named form.
        let h_charge = ff.partial_charge(AminoAcid::Ala, "H").expect("Ala H charge");
        assert!((h_charge - 0.31).abs() < 0.01);
    }

    #[test]
    fn methylene_charge_translation() {
        let ff = standard();
        // Leu HB2 / HB3 (PDB v3.3) come from CHARMM HB1 / HB2.
        // Both should carry the same +0.09 alkane H charge.
        assert!(ff.partial_charge(AminoAcid::Leu, "HB2").is_some());
        assert!(ff.partial_charge(AminoAcid::Leu, "HB3").is_some());
        let hb2 = ff.partial_charge(AminoAcid::Leu, "HB2").unwrap();
        let hb3 = ff.partial_charge(AminoAcid::Leu, "HB3").unwrap();
        assert!((hb2 - 0.09).abs() < 0.01);
        assert!((hb3 - 0.09).abs() < 0.01);
    }

    #[test]
    fn isoleucine_cd1_translation() {
        let ff = standard();
        // CHARMM "CD" → our "CD1"; CHARMM "HD1/HD2/HD3" → our "HD11/HD12/HD13".
        assert!(ff.partial_charge(AminoAcid::Ile, "CD1").is_some());
        assert!(ff.partial_charge(AminoAcid::Ile, "HD11").is_some());
    }

    #[test]
    fn histidine_uses_hsd_charges() {
        let ff = standard();
        // HSD has HD1 with +0.32 (the proton on ND1).
        let hd1 = ff.partial_charge(AminoAcid::His, "HD1").expect("His HD1 charge");
        assert!((hd1 - 0.32).abs() < 0.02);
        // ND1 in HSD has -0.36.
        let nd1 = ff.partial_charge(AminoAcid::His, "ND1").expect("His ND1 charge");
        assert!((nd1 - (-0.36)).abs() < 0.02);
    }

    #[test]
    fn improper_for_peptide_bond() {
        let ff = standard();
        // Peptide bond improper around the carbonyl C: CHARMM defines this as
        // various impropers; we check that lookup returns something for the
        // backbone C centre with NH1 / CT1 / O around it.
        let imp = ff.improper(AtomType::CT1, AtomType::C, AtomType::O, AtomType::NH1);
        assert!(imp.is_some(), "expected an improper for the peptide bond");
    }
}
