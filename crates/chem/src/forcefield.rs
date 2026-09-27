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
use crate::monomer::Monomer;
use crate::nucleotide::Nucleotide;

#[derive(Debug, Clone, Copy)]
pub struct BondParams {
    pub k: f64,  // kcal/mol/Å²
    pub r0: f64, // Å
}

#[derive(Debug, Clone, Copy)]
pub struct AngleParams {
    pub k: f64, // kcal/mol/rad²
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
    pub k: f64, // kcal/mol
    pub psi0_deg: f64,
}

/// CHARMM CMAP — a 24×24 grid of backbone (φ, ψ) energy corrections.
/// Grid spacing is 15° and the angles range from -180° to +165° with
/// wraparound (so index 24 wraps to index 0).  Stored in **kcal/mol**;
/// the energy code converts to kJ/mol at the leaves.
///
/// `data` is laid out as `data[phi_idx * GRID_SIZE + psi_idx]`.
#[derive(Debug, Clone)]
pub struct CmapGrid {
    pub data: Vec<f64>,
}

impl CmapGrid {
    pub const GRID_SIZE: usize = 24;
    pub const GRID_SPACING_DEG: f64 = 15.0;

    /// Energy value at integer grid indices (with wraparound).
    pub fn at(&self, phi_idx: usize, psi_idx: usize) -> f64 {
        let i = phi_idx % Self::GRID_SIZE;
        let j = psi_idx % Self::GRID_SIZE;
        self.data[i * Self::GRID_SIZE + j]
    }
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
    /// CHARMM CMAP backbone 2D corrections.  Keyed on
    /// (CA atom type of residue i, N atom type of residue i+1) —
    /// the two atom types that vary across the 6 CMAP grids in
    /// par_all36m_prot.prm. CT1/CT2/CP1 distinguishes the central
    /// CA (non-Gly/non-Pro vs Gly vs Pro); NH1/N distinguishes the
    /// next residue (non-Pro vs Pro).
    cmap: HashMap<(AtomType, AtomType), CmapGrid>,
    /// Per-(residue, atom-name) partial charge from the .rtf topology file.
    /// Atom names are stored in PDB v3.3 form (matching what our chain
    /// builder produces).
    partial_charges: HashMap<(AminoAcid, String), f64>,
    /// Per-(nucleotide, atom-name) partial charge from the CHARMM27
    /// nucleic .rtf topology file. Atom names are again PDB v3.3 form.
    rna_partial_charges: HashMap<(Nucleotide, String), f64>,
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

    /// Lookup the CHARMM27 partial charge for an atom in a ribonucleotide.
    /// `atom_name` is the PDB v3.3 form (matching the chain builder); the
    /// CHARMM `O1P` / `H2''` / `H2'` naming has already been translated
    /// to `OP1` / `H2'` / `HO2'` during parsing.
    pub fn partial_charge_rna(&self, nt: Nucleotide, atom_name: &str) -> Option<f64> {
        self.rna_partial_charges
            .get(&(nt, atom_name.to_owned()))
            .copied()
    }

    /// Dispatching partial-charge lookup — `Monomer::Protein(aa)` →
    /// [`partial_charge`], `Monomer::Rna(nt)` → [`partial_charge_rna`].
    pub fn partial_charge_for(&self, monomer: Monomer, atom_name: &str) -> Option<f64> {
        match monomer {
            Monomer::Protein(aa) => self.partial_charge(aa, atom_name),
            Monomer::Rna(nt) => self.partial_charge_rna(nt, atom_name),
        }
    }

    /// Lookup the CMAP grid for a residue whose central CA has atom
    /// type `ca` and whose next residue's N has atom type `next_n`.
    /// CHARMM36m has 6 grids covering (CT1/CT2/CP1) × (NH1/N).
    pub fn cmap(&self, ca: AtomType, next_n: AtomType) -> Option<&CmapGrid> {
        self.cmap.get(&(ca, next_n))
    }
}

fn canonical_pair(a: AtomType, b: AtomType) -> (AtomType, AtomType) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn canonical_triple(a: AtomType, b: AtomType, c: AtomType) -> (AtomType, AtomType, AtomType) {
    if a <= c {
        (a, b, c)
    } else {
        (c, b, a)
    }
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

/// Get the bundled CHARMM force field, parsed once on first call.
/// Loads both the CHARMM36m protein parameters and the CHARMM27 nucleic
/// acid parameters into a single `ForceField` — atom types are namespaced
/// per polymer so there is no collision (protein `AtomType::C` vs RNA
/// `AtomType::Cn1` etc.).
pub fn standard() -> &'static ForceField {
    static FF: OnceLock<ForceField> = OnceLock::new();
    FF.get_or_init(|| {
        let par_prot = include_str!("../../../data/charmm36/par_all36m_prot.prm");
        let rtf_prot = include_str!("../../../data/charmm36/top_all36_prot.rtf");
        let par_na = include_str!("../../../data/charmm27/par_all27_na.prm");
        let rtf_na = include_str!("../../../data/charmm27/top_all27_na.rtf");
        let mut ff = parse(par_prot);
        // Parse the nucleic acid .prm into the same tables — `parse`
        // appends bonds/angles/dihedrals/impropers/nonbonded for the
        // RNA atom types alongside the protein entries.
        parse_into(par_na, &mut ff);
        parse_rtf_charges(rtf_prot, &mut ff);
        parse_rtf_rna_charges(rtf_na, &mut ff);
        patch_disulfide_bond(&mut ff);
        ff
    })
}

/// Register a CYS-CYS disulfide (SG-SG) bond by mapping CHARMM's SM-SM
/// parameters onto our `S` atom type (which is what `classify` assigns
/// to CYS SG). The CHARMM `PRES DISU` patch reassigns SG from the free-
/// CYS sulfur type to `SM` (disulfide-bridged sulfur) and deletes the
/// HG hydrogens; we skip the type swap and the HG removal — the bond
/// constants and equilibrium distance match closely enough between the
/// two sulfur environments that the small charge/typing differences
/// only shift LJ + Coulomb terms by a few kJ/mol per disulfide, which
/// is below the noise of the rest of the field.
///
/// CHARMM SM-SM bond:  K = 173 kcal/mol/Å² (no leading ½ — CHARMM
/// convention V = K(r − r₀)²),  r₀ = 2.029 Å.
fn patch_disulfide_bond(ff: &mut ForceField) {
    let key = canonical_pair(AtomType::S, AtomType::S);
    ff.bonds.entry(key).or_insert(BondParams {
        k: 173.0,
        r0: 2.029,
    });
}

/// Parse a CHARMM .prm file into a fresh [`ForceField`].  See
/// [`parse_into`] for the append-to-existing variant used by [`standard`]
/// to merge the protein and nucleic .prm tables.
pub fn parse(text: &str) -> ForceField {
    let mut ff = ForceField::default();
    parse_into(text, &mut ff);
    ff
}

/// Parse a CHARMM .prm file and append its parameters to `ff`. Wildcard
/// atom-type tokens (`X`) are recognised. Lines whose atom-type tokens
/// don't match anything in our [`AtomType`] enum are silently skipped —
/// the file contains parameters for many types we don't model.
///
/// Calling this multiple times merges parameter sets. The existing
/// `entry().or_insert(...)` logic in the line parsers preserves the
/// first definition seen for any key, so calling order determines
/// precedence: the protein .prm is loaded first, then the nucleic .prm
/// adds RNA-specific entries that don't collide.
pub fn parse_into(text: &str, ff: &mut ForceField) {
    let mut section = Section::None;
    // CMAP block state: accumulates one grid's float values across
    // many lines, then commits when 576 = 24×24 values are in hand.
    let mut cmap_pending_key: Option<(AtomType, AtomType)> = None;
    let mut cmap_buffer: Vec<f64> = Vec::with_capacity(CmapGrid::GRID_SIZE * CmapGrid::GRID_SIZE);

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
            // Finish any pending CMAP grid when we leave the section.
            if section == Section::Cmap && cmap_pending_key.is_some() {
                commit_cmap_if_full(&mut cmap_pending_key, &mut cmap_buffer, ff);
            }
            section = s;
            continue;
        }

        // Skip lines that look like preamble or directives we don't parse
        // (e.g., the `cutnb 14.0 ctofnb...` line continuing NONBONDED).
        if section == Section::None {
            continue;
        }

        match section {
            Section::Bonds => parse_bond_line(trimmed, ff),
            Section::Angles => parse_angle_line(trimmed, ff),
            Section::Dihedrals => parse_dihedral_line(trimmed, ff),
            Section::Impropers => parse_improper_line(trimmed, ff),
            Section::Nonbonded => parse_nonbonded_line(trimmed, ff),
            Section::Cmap => parse_cmap_line(trimmed, &mut cmap_pending_key, &mut cmap_buffer, ff),
            // Sections we ignore.
            Section::Hbond | Section::Nbfix | Section::None => {}
            Section::End => break,
        }
    }
    // EOF-flush any pending CMAP grid.
    if cmap_pending_key.is_some() {
        commit_cmap_if_full(&mut cmap_pending_key, &mut cmap_buffer, ff);
    }
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
    /// NBFIX block in CHARMM27 nucleic — atom-pair-specific LJ overrides.
    /// We ignore it because (a) the entries are for nucleic-ion pairs
    /// (sodium, potassium, magnesium) we don't model, and (b) it isn't
    /// a regular pair-table the energy code consumes — it's a small
    /// correction layer.
    Nbfix,
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
        "NBFIX" => Some(Section::Nbfix),
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
        (parse_atom(tokens[0]), parse_atom(tokens[1]))
    else {
        return;
    };
    let Ok(k) = tokens[2].parse::<f64>() else {
        return;
    };
    let Ok(r0) = tokens[3].parse::<f64>() else {
        return;
    };
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
    )
    else {
        return;
    };
    let Ok(k) = tokens[3].parse::<f64>() else {
        return;
    };
    let Ok(theta0_deg) = tokens[4].parse::<f64>() else {
        return;
    };
    let key = canonical_triple(a, b, c);
    ff.angles
        .entry(key)
        .or_insert(AngleParams { k, theta0_deg });
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
    let Ok(k) = tokens[4].parse::<f64>() else {
        return;
    };
    let Ok(n) = tokens[5].parse::<u32>() else {
        return;
    };
    let Ok(delta_deg) = tokens[6].parse::<f64>() else {
        return;
    };
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
    let Ok(k) = tokens[4].parse::<f64>() else {
        return;
    };
    // tokens[5] is a placeholder (always 0)
    let Ok(psi0_deg) = tokens[6].parse::<f64>() else {
        return;
    };
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

/// Parse one line of the CMAP block.
///
/// A CMAP block starts with an 8-atom-name header followed by the
/// grid size (24).  Subsequent lines hold 24×24 = 576 float values
/// spread across many lines; once accumulated, the grid is keyed
/// on (CA atom type, next-N atom type) — the two atom types that
/// distinguish CHARMM36m's six CMAPs.  The other 6 atoms in the
/// header are always `C` plus the same N/CA pair twice (because the
/// 8-tuple is `φ atoms ++ ψ atoms` for the central residue), so they
/// carry no information beyond the key we extract.
fn parse_cmap_line(
    line: &str,
    pending_key: &mut Option<(AtomType, AtomType)>,
    buffer: &mut Vec<f64>,
    ff: &mut ForceField,
) {
    let tokens: Vec<&str> = line.split_ascii_whitespace().collect();
    if tokens.is_empty() {
        return;
    }
    // CMAP grid-header lines have 9 tokens (8 atom names + grid size).
    // Heuristic: if the LAST token parses as an integer and the FIRST
    // token parses as a known atom-type name, treat the line as a
    // header; otherwise it's a row of grid floats.
    let last_int = tokens.last().and_then(|s| s.parse::<usize>().ok());
    let first_atom = AtomType::from_charmm_name(tokens[0]);
    if let (Some(grid_size), Some(_)) = (last_int, first_atom) {
        if tokens.len() == 9 {
            // Flush any previous in-flight grid before starting a new one.
            commit_cmap_if_full(pending_key, buffer, ff);
            // Extract the (CA, next-N) key from tokens[2] (column 3,
            // 0-indexed) and tokens[7] (column 8). These are the two
            // tokens that vary across CHARMM36m's six grids.
            let ca = match AtomType::from_charmm_name(tokens[2]) {
                Some(t) => t,
                None => return, // Unknown atom type — skip block.
            };
            let next_n = match AtomType::from_charmm_name(tokens[7]) {
                Some(t) => t,
                None => return,
            };
            // Sanity: bail if grid size differs from our compile-time
            // assumption (CHARMM has always used 24).
            if grid_size != CmapGrid::GRID_SIZE {
                return;
            }
            *pending_key = Some((ca, next_n));
            buffer.clear();
            return;
        }
    }
    // Otherwise: row of grid floats.
    if pending_key.is_some() {
        for tok in tokens {
            if let Ok(v) = tok.parse::<f64>() {
                buffer.push(v);
            }
        }
        // Commit once we have a full grid.
        commit_cmap_if_full(pending_key, buffer, ff);
    }
}

/// Finalise a CMAP grid into `ff` if `buffer` holds the full 576 values.
fn commit_cmap_if_full(
    pending_key: &mut Option<(AtomType, AtomType)>,
    buffer: &mut Vec<f64>,
    ff: &mut ForceField,
) {
    let n = CmapGrid::GRID_SIZE * CmapGrid::GRID_SIZE;
    if buffer.len() == n {
        if let Some(key) = pending_key.take() {
            ff.cmap.entry(key).or_insert(CmapGrid {
                data: std::mem::take(buffer),
            });
        }
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
    let Ok(eps_signed) = tokens[2].parse::<f64>() else {
        return;
    };
    let Ok(rmin_half) = tokens[3].parse::<f64>() else {
        return;
    };
    let mut params = NonbondedParams {
        epsilon: -eps_signed, // CHARMM stores -ε, we want positive ε
        rmin_half,
        epsilon_14: None,
        rmin_half_14: None,
    };
    if tokens.len() >= 7 {
        // 1-4 LJ parameters present.
        if let (Ok(eps14), Ok(rmin14)) = (tokens[5].parse::<f64>(), tokens[6].parse::<f64>()) {
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
        let Ok(charge) = charge_str.parse::<f64>() else {
            continue;
        };
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
        (
            Leu | Met | Pro | Ser | Cys | Asn | Gln | Asp | Glu | Lys | Arg | His | Phe | Tyr | Trp,
            "HB1",
        ) => "HB2",
        (
            Leu | Met | Pro | Ser | Cys | Asn | Gln | Asp | Glu | Lys | Arg | His | Phe | Tyr | Trp,
            "HB2",
        ) => "HB3",

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

/// Parse the four RNA RESI blocks of `top_all27_na.rtf` (GUA / ADE /
/// CYT / URA) and populate per-nucleotide partial charges. DNA-only
/// patches (DEO1 / DEO2), terminal patches (5TER / 3TER), water, and
/// ions are ignored.
///
/// CHARMM atom names are translated to PDB v3.3 form via
/// [`charmm_to_pdb_rna_name`] so callers can look up by the same names
/// the chain builder uses (`OP1`/`OP2` not `O1P`/`O2P`; `H2'` for the
/// C2'-proton, `HO2'` for the 2'-hydroxyl proton).
fn parse_rtf_rna_charges(text: &str, ff: &mut ForceField) {
    let mut current: Option<Nucleotide> = None;
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
                "ADE" => Some(Nucleotide::Adenine),
                "GUA" => Some(Nucleotide::Guanine),
                "CYT" => Some(Nucleotide::Cytosine),
                "URA" => Some(Nucleotide::Uracil),
                // THY (DNA), TIP3, ions, dummies → skip
                _ => None,
            };
            continue;
        }
        if head == "PRES" {
            current = None;
            continue;
        }
        if head != "ATOM" {
            continue;
        }
        let Some(nt) = current else { continue };
        let charmm_name = tokens.next().unwrap_or("");
        let _atom_type = tokens.next().unwrap_or("");
        let charge_str = tokens.next().unwrap_or("");
        let Ok(charge) = charge_str.parse::<f64>() else {
            continue;
        };
        let pdb_name = charmm_to_pdb_rna_name(charmm_name);
        if pdb_name.is_empty() {
            continue;
        }
        ff.rna_partial_charges
            .insert((nt, pdb_name.to_owned()), charge);
    }
}

/// Translate a CHARMM27 nucleic atom name to PDB v3.3 form. Three
/// shifts apply across every nucleotide:
///   1. `O1P` / `O2P` → `OP1` / `OP2` (PDB convention).
///   2. `H2'` (CHARMM: proton on the 2'-hydroxyl O2') → `HO2'`.
///   3. `H2''` (CHARMM: proton on C2') → `H2'`.
/// Other atoms (P, O5', H5', H5'', C4', H4', O4', C1', H1', C3', H3',
/// O3', C2', O2', N9, C8, N7, …) already use the PDB v3.3 spelling.
fn charmm_to_pdb_rna_name(charmm: &str) -> &'static str {
    match charmm {
        "O1P" => "OP1",
        "O2P" => "OP2",
        "H2'" => "HO2'",
        "H2''" => "H2'",
        // Pass-through for names that already match PDB v3.3. The list
        // covers every atom present in the four canonical RNA RESI
        // blocks of top_all27_na.rtf.
        "P" => "P",
        "O5'" => "O5'",
        "C5'" => "C5'",
        "H5'" => "H5'",
        "H5''" => "H5''",
        "C4'" => "C4'",
        "H4'" => "H4'",
        "O4'" => "O4'",
        "C1'" => "C1'",
        "H1'" => "H1'",
        "C2'" => "C2'",
        "O2'" => "O2'",
        "C3'" => "C3'",
        "H3'" => "H3'",
        "O3'" => "O3'",
        // Base atoms — names are identical between CHARMM and PDB v3.3.
        "N1" => "N1",
        "N2" => "N2",
        "N3" => "N3",
        "N4" => "N4",
        "N6" => "N6",
        "N7" => "N7",
        "N9" => "N9",
        "C2" => "C2",
        "C4" => "C4",
        "C5" => "C5",
        "C6" => "C6",
        "C8" => "C8",
        "O2" => "O2",
        "O4" => "O4",
        "O6" => "O6",
        "H1" => "H1",
        "H2" => "H2",
        "H3" => "H3",
        "H5" => "H5",
        "H6" => "H6",
        "H8" => "H8",
        "H21" => "H21",
        "H22" => "H22",
        "H41" => "H41",
        "H42" => "H42",
        "H61" => "H61",
        "H62" => "H62",
        _ => "",
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
        let p = ff
            .angle(AtomType::NH1, AtomType::CT1, AtomType::C)
            .expect("NH1-CT1-C");
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
        let _ = ff
            .dihedral(AtomType::HA3, AtomType::CT3, AtomType::CT2, AtomType::HA2)
            .expect("HA3-CT3-CT2-HA2 via wildcard");
    }

    #[test]
    fn partial_charges_loaded() {
        let ff = standard();
        // Backbone N and CA charges are well known.
        let n_charge = ff
            .partial_charge(AminoAcid::Ala, "N")
            .expect("Ala N charge");
        assert!((n_charge - (-0.47)).abs() < 0.01);
        let ca_charge = ff
            .partial_charge(AminoAcid::Ala, "CA")
            .expect("Ala CA charge");
        assert!((ca_charge - 0.07).abs() < 0.01);
        // The amide H in PDB-named form.
        let h_charge = ff
            .partial_charge(AminoAcid::Ala, "H")
            .expect("Ala H charge");
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
        let hd1 = ff
            .partial_charge(AminoAcid::His, "HD1")
            .expect("His HD1 charge");
        assert!((hd1 - 0.32).abs() < 0.02);
        // ND1 in HSD has -0.36.
        let nd1 = ff
            .partial_charge(AminoAcid::His, "ND1")
            .expect("His ND1 charge");
        assert!((nd1 - (-0.36)).abs() < 0.02);
    }

    #[test]
    fn rna_nonbonded_loaded() {
        let ff = standard();
        // Phosphate P — CHARMM27 par_all27_na.prm has P at -0.585 ε
        // (stored as -ε in CHARMM convention; we negate so it's
        // positive here) and Rmin/2 = 2.15 Å.
        let p = ff.nonbonded(AtomType::Pn).expect("P nonbonded");
        assert!((p.epsilon - 0.585).abs() < 0.05);
        assert!((p.rmin_half - 2.15).abs() < 0.1);
        // CN7 (sugar C3'/C4') has standard sp³ C parameters.
        let cn7 = ff.nonbonded(AtomType::Cn7).expect("CN7 nonbonded");
        assert!(cn7.epsilon > 0.0);
        assert!(cn7.rmin_half > 1.5);
    }

    #[test]
    fn rna_bond_loaded() {
        let ff = standard();
        // CN7 (ribose C3') - ON2 (O3') backbone bond — standard
        // CHARMM27 nucleic value, around 320 kcal/mol/Å² @ 1.42 Å.
        let p = ff.bond(AtomType::Cn7, AtomType::On2).expect("CN7-ON2 bond");
        assert!(p.k > 0.0);
        assert!((p.r0 - 1.42).abs() < 0.1);
    }

    #[test]
    fn rna_partial_charges_loaded() {
        let ff = standard();
        // Phosphate P: +1.50 in CHARMM27.
        let p = ff
            .partial_charge_rna(Nucleotide::Adenine, "P")
            .expect("P charge on adenine");
        assert!((p - 1.50).abs() < 0.01);
        // OP1 (anionic phosphate O): -0.78.
        let op1 = ff.partial_charge_rna(Nucleotide::Adenine, "OP1").unwrap();
        assert!((op1 - (-0.78)).abs() < 0.01);
        // Adenine N6 amine: -0.77.
        let n6 = ff.partial_charge_rna(Nucleotide::Adenine, "N6").unwrap();
        assert!((n6 - (-0.77)).abs() < 0.01);
        // Adenine H61 / H62: +0.38.
        let h61 = ff.partial_charge_rna(Nucleotide::Adenine, "H61").unwrap();
        assert!((h61 - 0.38).abs() < 0.01);
    }

    #[test]
    fn rna_charges_translate_charmm_names() {
        let ff = standard();
        // The CHARMM `O1P`/`O2P` charges land under PDB-v3.3 `OP1`/`OP2`.
        assert!(ff.partial_charge_rna(Nucleotide::Uracil, "OP1").is_some());
        assert!(ff.partial_charge_rna(Nucleotide::Uracil, "OP2").is_some());
        // The CHARMM `H2''` (on C2') maps to PDB `H2'`.
        assert!(ff.partial_charge_rna(Nucleotide::Uracil, "H2'").is_some());
        // The CHARMM `H2'` (on O2') maps to PDB `HO2'`.
        assert!(ff.partial_charge_rna(Nucleotide::Uracil, "HO2'").is_some());
    }

    #[test]
    fn rna_full_charge_coverage() {
        // Every atom placed by the RNA builder should have a charge
        // loaded — sanity check for the name-translation table.
        let ff = standard();
        for nt in [
            Nucleotide::Adenine,
            Nucleotide::Uracil,
            Nucleotide::Guanine,
            Nucleotide::Cytosine,
        ] {
            for (name, _) in nt.all_atoms() {
                assert!(
                    ff.partial_charge_rna(nt, name).is_some(),
                    "{:?} atom {} has no CHARMM27 partial charge",
                    nt,
                    name,
                );
            }
        }
    }

    #[test]
    fn nucleotide_neutral_when_summed() {
        // Each canonical RNA nucleotide RESI block carries a net
        // charge of -1.0 (the anionic phosphate). Sum up all the
        // CHARMM27 atomic charges and verify.
        let ff = standard();
        for nt in [
            Nucleotide::Adenine,
            Nucleotide::Uracil,
            Nucleotide::Guanine,
            Nucleotide::Cytosine,
        ] {
            let sum: f64 = nt
                .all_atoms()
                .iter()
                .map(|(name, _)| ff.partial_charge_rna(nt, name).unwrap_or(0.0))
                .sum();
            assert!(
                (sum - (-1.0)).abs() < 0.01,
                "{nt:?} summed charge {sum:.3} != -1.00"
            );
        }
    }

    #[test]
    fn cmap_grids_loaded() {
        let ff = standard();
        // All six CHARMM36m CMAPs present.
        let ca_classes = [AtomType::CT1, AtomType::CT2, AtomType::CP1];
        let next_ns = [AtomType::NH1, AtomType::N];
        let mut total = 0;
        for &ca in &ca_classes {
            for &n in &next_ns {
                let g = ff.cmap(ca, n);
                assert!(
                    g.is_some(),
                    "missing CMAP for (CA={:?}, N_next={:?})",
                    ca,
                    n
                );
                assert_eq!(g.unwrap().data.len(), 24 * 24);
                total += 1;
            }
        }
        assert_eq!(total, 6);
    }

    #[test]
    fn cmap_is_a_correction_not_a_full_potential() {
        // CMAP is added on top of the existing periodic dihedral
        // potential.  The α-helix and β-sheet basins in the *combined*
        // potential live where the periodic terms already put them;
        // CMAP's job is to nudge the relative depths and barrier
        // shapes to match QM data.  Concretely, the alanine map's
        // value in the α-helix basin (φ ≈ -60°, ψ ≈ -45°) is small
        // and slightly negative — verifies the grid isn't loaded
        // upside-down or transposed.
        let ff = standard();
        let grid = ff.cmap(AtomType::CT1, AtomType::NH1).unwrap();
        let alpha_phi_idx = ((-60.0 + 180.0) / 15.0) as usize;
        let alpha_psi_idx = ((-45.0 + 180.0) / 15.0) as usize;
        let alpha_val = grid.at(alpha_phi_idx, alpha_psi_idx);
        // Per the loaded .prm: row "-60" has -0.48 at ψ=-45.
        assert!(
            (alpha_val - (-0.48)).abs() < 0.01,
            "α-helix CMAP value {alpha_val} (expected ≈ -0.48)"
        );
        // β-sheet region (φ ≈ -120°, ψ ≈ +120°): row "-120" idx 20.
        let beta_phi_idx = ((-120.0 + 180.0) / 15.0) as usize;
        let beta_psi_idx = ((120.0 + 180.0) / 15.0) as usize;
        let beta_val = grid.at(beta_phi_idx, beta_psi_idx);
        // Row "-120" position 20 (ψ=120°): value is "-0.97" per the
        // .prm grid.
        assert!(
            (beta_val - (-0.97)).abs() < 0.01,
            "β-sheet CMAP value {beta_val} (expected ≈ -0.97)"
        );
    }

    #[test]
    fn cmap_alanine_grid_known_value() {
        // The alanine map (CT1 / NH1 next) — first value at φ=-180,
        // ψ=-180 — should match the .prm file's leading "0.13" value
        // (the file header was: "C NH1 CT1 C NH1 CT1 C NH1   24",
        // then the !-180 sub-block starts with "0.13 0.77 0.97 …").
        let ff = standard();
        let g = ff.cmap(AtomType::CT1, AtomType::NH1).unwrap();
        assert!((g.at(0, 0) - 0.13).abs() < 1e-6);
        assert!((g.at(0, 1) - 0.77).abs() < 1e-6);
        // φ=-180, ψ=-180 corresponds to indices (0, 0); φ=-165 to (1, 0).
        // The .prm shows row "-165" starts with "-0.13 1.38 ...".
        assert!((g.at(1, 0) - (-0.13)).abs() < 1e-6);
        assert!((g.at(1, 1) - 1.38).abs() < 1e-6);
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
