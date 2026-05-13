//! PDB ATOM-record reader.
//!
//! Reads the subset of the PDB format we need: ATOM records, single chain,
//! all-atom proteins built from the 20 standard residues. Intentionally
//! lenient about whitespace and column drift (real PDB files vary), but
//! strict about content — unrecognised residues or atoms are reported as
//! errors.
//!
//! Atom-name normalisation: PDB v2 (and many older files) used names like
//! "1HD1" with the leading character being the suffix digit. PDB v3.3 (and
//! our writer) uses "HD11". We accept both forms.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read};

use chem::{AminoAcid, Element};
use thiserror::Error;

use geom::structure::{PlacedAtom, PlacedResidue, Structure};
use geom::Vec3;

#[derive(Debug, Error)]
pub enum PdbReadError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("line {0}: malformed ATOM record: {1}")]
    Malformed(usize, String),
    #[error("line {0}: residue name {1:?} is not one of the 20 standard amino acids")]
    UnknownResidue(usize, String),
    #[error("line {0}: atom {1:?} is not part of residue {2:?}")]
    UnknownAtom(usize, String, AminoAcid),
    #[error("PDB contains no ATOM records")]
    Empty,
}

/// Read every MODEL in a multi-frame PDB. A file with no `MODEL` records
/// is treated as a single-model trajectory and returns one structure.
pub fn read_pdb_trajectory<R: Read>(reader: R) -> Result<Vec<Structure>, PdbReadError> {
    let buf = BufReader::new(reader);
    let mut frames: Vec<Structure> = Vec::new();
    let mut current: Vec<ParsedAtom> = Vec::new();

    for (lineno_zero, line) in buf.lines().enumerate() {
        let lineno = lineno_zero + 1;
        let line = line?;

        if line.starts_with("ENDMDL") {
            if !current.is_empty() {
                frames.push(assemble_structure(std::mem::take(&mut current)));
            }
            continue;
        }
        if line.starts_with("END") {
            if !current.is_empty() {
                frames.push(assemble_structure(std::mem::take(&mut current)));
            }
            break;
        }
        if line.starts_with("MODEL") {
            // Starting a new MODEL — if we accumulated atoms outside a
            // MODEL block, flush them first.
            if !current.is_empty() {
                frames.push(assemble_structure(std::mem::take(&mut current)));
            }
            continue;
        }
        if line.starts_with("TER") {
            continue;
        }
        if !line.starts_with("ATOM") {
            continue;
        }

        let rec = parse_atom_record(&line, lineno)?;
        if rec.alt_loc != ' ' && rec.alt_loc != 'A' {
            continue;
        }
        let aa = AminoAcid::from_three_letter(&rec.res_name)
            .ok_or_else(|| PdbReadError::UnknownResidue(lineno, rec.res_name.clone()))?;
        let norm_owned = normalise_atom_name(&rec.atom_name);
        let canonical_name = match canonical_atom_name(aa, &norm_owned) {
            Some(n) => n,
            None => {
                if is_terminal_patch_atom(&norm_owned) {
                    continue;
                }
                return Err(PdbReadError::UnknownAtom(lineno, rec.atom_name, aa));
            }
        };
        let element = if let Some(e) = parse_element(&rec.element) {
            e
        } else {
            element_from_atom_name(canonical_name).ok_or_else(|| {
                PdbReadError::Malformed(
                    lineno,
                    format!("can't determine element for atom {canonical_name:?}"),
                )
            })?
        };
        current.push(ParsedAtom {
            res_key: (rec.chain, rec.res_seq),
            aa,
            atom: PlacedAtom {
                name: canonical_name,
                element,
                position: Vec3::new(rec.x, rec.y, rec.z),
            },
        });
    }
    if !current.is_empty() {
        frames.push(assemble_structure(current));
    }
    if frames.is_empty() {
        return Err(PdbReadError::Empty);
    }
    Ok(frames)
}

struct ParsedAtom {
    res_key: (char, i32),
    aa: AminoAcid,
    atom: PlacedAtom,
}

fn assemble_structure(parsed: Vec<ParsedAtom>) -> Structure {
    let mut residues: Vec<PlacedResidue> = Vec::new();
    let mut current_key: Option<(char, i32)> = None;
    let mut seen: BTreeMap<(usize, &'static str), ()> = BTreeMap::new();
    for p in parsed {
        if Some(p.res_key) != current_key {
            residues.push(PlacedResidue {
                aa: p.aa,
                atoms: Vec::new(),
                chain: p.res_key.0,
            });
            current_key = Some(p.res_key);
        }
        let idx = residues.len() - 1;
        if seen.insert((idx, p.atom.name), ()).is_none() {
            residues.last_mut().unwrap().atoms.push(p.atom);
        }
    }
    Structure { residues }
}

pub fn read_pdb<R: Read>(reader: R) -> Result<Structure, PdbReadError> {
    let buf = BufReader::new(reader);

    // (chain, residue_seq) → (AminoAcid, Vec<(serial, name, element, position)>)
    // We buffer all parsed atoms then assemble residues in input order.
    #[derive(Debug)]
    struct ResidueBuf {
        aa: AminoAcid,
        atoms: Vec<PlacedAtom>,
        atom_names_seen: BTreeMap<&'static str, ()>,
        chain: char,
    }
    let mut residues: Vec<ResidueBuf> = Vec::new();
    // Track current key for grouping consecutive ATOMs of the same residue.
    let mut current: Option<(char, i32)> = None;
    let mut found_any = false;

    for (lineno_zero, line) in buf.lines().enumerate() {
        let lineno = lineno_zero + 1;
        let line = line?;

        // Treat END, ENDMDL, and TER (after we've started reading) as stop
        // markers. NMR ensembles have ENDMDL between models — we only read
        // the first model.
        if line.starts_with("END") {
            // Catches both "END" and "ENDMDL".
            if !residues.is_empty() {
                break;
            }
            continue;
        }
        if line.starts_with("MODEL") && !residues.is_empty() {
            break;
        }
        // TER separates chains within a PDB (chain A's TER comes before
        // chain B's ATOMs). Skip it but keep reading — multi-chain
        // proteins like insulin (2HIU) need to capture chain B's atoms
        // after chain A's TER.
        if line.starts_with("TER") {
            continue;
        }
        if !line.starts_with("ATOM") {
            continue;
        }

        let rec = parse_atom_record(&line, lineno)?;

        // Only take the primary alternate location.
        if rec.alt_loc != ' ' && rec.alt_loc != 'A' {
            continue;
        }

        let aa = AminoAcid::from_three_letter(&rec.res_name)
            .ok_or_else(|| PdbReadError::UnknownResidue(lineno, rec.res_name.clone()))?;

        // Normalise the atom name to wwPDB v3.3.
        let norm_owned = normalise_atom_name(&rec.atom_name);
        let canonical_name = match canonical_atom_name(aa, &norm_owned) {
            Some(n) => n,
            None => {
                // Terminal patches (NH3+ extra H's, C-terminal OXT) are not
                // in our M2/M3 atom set. Silently skip those rather than
                // bailing — they'd be re-added by an explicit terminus
                // patcher (out of M3 scope).
                if is_terminal_patch_atom(&norm_owned) {
                    continue;
                }
                return Err(PdbReadError::UnknownAtom(lineno, rec.atom_name, aa));
            }
        };

        // Determine element.
        let element = if let Some(e) = parse_element(&rec.element) {
            e
        } else {
            element_from_atom_name(canonical_name)
                .ok_or_else(|| PdbReadError::Malformed(lineno, format!("can't determine element for atom {canonical_name:?}")))?
        };

        let key = (rec.chain, rec.res_seq);
        if Some(key) != current {
            residues.push(ResidueBuf {
                aa,
                atoms: Vec::new(),
                atom_names_seen: BTreeMap::new(),
                chain: rec.chain,
            });
            current = Some(key);
        }
        let last = residues.last_mut().unwrap();
        if last.atom_names_seen.insert(canonical_name, ()).is_none() {
            last.atoms.push(PlacedAtom {
                name: canonical_name,
                element,
                position: Vec3::new(rec.x, rec.y, rec.z),
            });
        }
        found_any = true;
    }

    if !found_any {
        return Err(PdbReadError::Empty);
    }

    let placed: Vec<PlacedResidue> = residues
        .into_iter()
        .map(|r| PlacedResidue { aa: r.aa, atoms: r.atoms, chain: r.chain })
        .collect();

    Ok(Structure { residues: placed })
}

#[derive(Debug)]
struct AtomRecord {
    chain: char,
    res_seq: i32,
    res_name: String,
    atom_name: String,
    alt_loc: char,
    element: String,
    x: f64,
    y: f64,
    z: f64,
}

fn parse_atom_record(line: &str, lineno: usize) -> Result<AtomRecord, PdbReadError> {
    // PDB columns are 1-indexed; Rust strings are 0-indexed. We slice carefully.
    // We pad short lines with spaces so column extraction doesn't panic.
    let mut padded: String = line.to_owned();
    while padded.len() < 80 {
        padded.push(' ');
    }
    let s = padded.as_str();

    let atom_name = s[12..16].trim().to_owned();
    let alt_loc = s.as_bytes()[16] as char;
    let res_name = s[17..20].trim().to_owned();
    let chain = s.as_bytes()[21] as char;
    let res_seq: i32 = s[22..26]
        .trim()
        .parse()
        .map_err(|_| PdbReadError::Malformed(lineno, "residue seq number".into()))?;
    let x: f64 = s[30..38]
        .trim()
        .parse()
        .map_err(|_| PdbReadError::Malformed(lineno, "x coordinate".into()))?;
    let y: f64 = s[38..46]
        .trim()
        .parse()
        .map_err(|_| PdbReadError::Malformed(lineno, "y coordinate".into()))?;
    let z: f64 = s[46..54]
        .trim()
        .parse()
        .map_err(|_| PdbReadError::Malformed(lineno, "z coordinate".into()))?;
    let element = s[76..78].trim().to_owned();

    Ok(AtomRecord {
        chain,
        res_seq,
        res_name,
        atom_name,
        alt_loc,
        element,
        x,
        y,
        z,
    })
}

/// Atoms that appear in real PDB files because the residue is at a chain
/// terminus, but which our chain builder doesn't yet model. We silently
/// skip them when reading.
fn is_terminal_patch_atom(name: &str) -> bool {
    matches!(name, "H1" | "H2" | "H3" | "OXT" | "HXT")
}

/// Convert a PDB atom name to wwPDB v3.3 form (digit-suffixed).
/// e.g. "1HD1" -> "HD11". If the name is already v3.3 (or has no leading
/// digit), it's returned unchanged.
fn normalise_atom_name(name: &str) -> String {
    let bytes = name.as_bytes();
    if bytes.len() >= 4 && bytes[0].is_ascii_digit() {
        // "1HD1" -> "HD11"
        let mut out = String::with_capacity(name.len());
        out.push_str(&name[1..]);
        out.push(name.chars().next().unwrap());
        out
    } else {
        name.to_owned()
    }
}

/// Look up the canonical wwPDB v3.3 atom name (returning a `&'static str` so
/// it matches our chem topology data). Returns `None` if the atom isn't in
/// the residue's known atom list.
fn canonical_atom_name(aa: AminoAcid, name: &str) -> Option<&'static str> {
    // Backbone atoms.
    match name {
        "N" | "CA" | "C" | "O" | "H" | "HA" | "HA2" | "HA3"
        | "OXT" | "H1" | "H2" | "H3" => {
            // Map common PDB synonyms; we ignore terminal-residue extras for now.
            match name {
                "N" => return Some("N"),
                "CA" => return Some("CA"),
                "C" => return Some("C"),
                "O" => return Some("O"),
                "H" => return Some("H"),
                "HA" => return Some("HA"),
                "HA2" => return Some("HA2"),
                "HA3" => return Some("HA3"),
                _ => return None, // OXT, H1/H2/H3 — terminal patches, not modelled in M2/M3
            }
        }
        _ => {}
    }
    // Side-chain atoms — match against the topology table.
    for sc in aa.topology().sidechain {
        if sc.name == name {
            return Some(sc.name);
        }
    }
    None
}

fn parse_element(s: &str) -> Option<Element> {
    match s.trim().to_ascii_uppercase().as_str() {
        "H" => Some(Element::H),
        "C" => Some(Element::C),
        "N" => Some(Element::N),
        "O" => Some(Element::O),
        "S" => Some(Element::S),
        _ => None,
    }
}

/// Fallback element derivation from the atom name.
fn element_from_atom_name(name: &str) -> Option<Element> {
    match name.chars().next()? {
        'H' => Some(Element::H),
        'C' => Some(Element::C),
        'N' => Some(Element::N),
        'O' => Some(Element::O),
        'S' => Some(Element::S),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::chem::AminoAcid;
    use geom::build_extended_chain;

    #[test]
    fn round_trip_alanine_chain() {
        // Build an Ala-Ala-Ala chain, write to PDB, read back, compare atom counts.
        let original = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala, AminoAcid::Ala]).unwrap();
        let mut buf = Vec::new();
        crate::write_pdb(&mut buf, &original, "round-trip").unwrap();
        let parsed = read_pdb(buf.as_slice()).expect("read back");
        assert_eq!(parsed.residues.len(), original.residues.len());
        for (a, b) in parsed.residues.iter().zip(original.residues.iter()) {
            assert_eq!(a.aa, b.aa);
            assert_eq!(a.atoms.len(), b.atoms.len());
            for (pa, pb) in a.atoms.iter().zip(b.atoms.iter()) {
                assert_eq!(pa.name, pb.name);
                assert_eq!(pa.element, pb.element);
                assert!((pa.position - pb.position).norm() < 1e-3);
            }
        }
    }

    #[test]
    fn round_trip_all_residues() {
        let seq: Vec<AminoAcid> = AminoAcid::ALL.to_vec();
        let original = build_extended_chain(&seq).unwrap();
        let mut buf = Vec::new();
        crate::write_pdb(&mut buf, &original, "all-20").unwrap();
        let parsed = read_pdb(buf.as_slice()).expect("read back");
        assert_eq!(parsed.residues.len(), 20);
        let total_orig: usize = original.residues.iter().map(|r| r.atoms.len()).sum();
        let total_parsed: usize = parsed.residues.iter().map(|r| r.atoms.len()).sum();
        assert_eq!(total_orig, total_parsed);
    }

    #[test]
    fn normalise_legacy_atom_name() {
        assert_eq!(normalise_atom_name("1HD1"), "HD11");
        assert_eq!(normalise_atom_name("HD11"), "HD11");
        assert_eq!(normalise_atom_name("CA"), "CA");
        assert_eq!(normalise_atom_name("HA"), "HA");
    }

    #[test]
    fn rejects_unknown_residue() {
        let pdb = "ATOM      1  N   XXX A   1       0.000   0.000   0.000  1.00  0.00           N\n";
        let err = read_pdb(pdb.as_bytes()).unwrap_err();
        assert!(matches!(err, PdbReadError::UnknownResidue(_, _)));
    }

    #[test]
    fn rejects_empty() {
        let pdb = "REMARK only\n";
        let err = read_pdb(pdb.as_bytes()).unwrap_err();
        assert!(matches!(err, PdbReadError::Empty));
    }

    #[test]
    fn reads_real_trp_cage_pdb() {
        // 1L2Y MODEL 1 — Trp-cage NMR structure, 20 residues:
        // NLYIQWLKDGGPSSGRPPPS
        let pdb = include_str!("../tests/fixtures/1L2Y_model1.pdb");
        let s = read_pdb(pdb.as_bytes()).expect("parse 1L2Y");
        assert_eq!(s.residues.len(), 20);
        let expected_seq = "NLYIQWLKDGGPSSGRPPPS";
        let actual_seq: String = s.residues.iter().map(|r| r.aa.one_letter()).collect();
        assert_eq!(actual_seq, expected_seq);
    }

    #[test]
    fn round_trip_trajectory() {
        let s1 = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Gly]).unwrap();
        let mut s2 = s1.clone();
        // Perturb the second frame so the two structures differ.
        s2.residues[0].atoms[0].position += Vec3::new(0.5, 0.0, 0.0);
        let frames = [s1.clone(), s2.clone(), s1.clone()];
        let mut buf = Vec::new();
        crate::write_pdb_trajectory(&mut buf, "round-trip traj", frames.iter()).unwrap();
        let parsed = read_pdb_trajectory(buf.as_slice()).expect("parse traj");
        assert_eq!(parsed.len(), 3);
        for (a, b) in parsed.iter().zip(frames.iter()) {
            assert_eq!(a.residues.len(), b.residues.len());
            for (ar, br) in a.residues.iter().zip(b.residues.iter()) {
                assert_eq!(ar.atoms.len(), br.atoms.len());
                for (aa, bb) in ar.atoms.iter().zip(br.atoms.iter()) {
                    assert!((aa.position - bb.position).norm() < 1e-3);
                }
            }
        }
        // The second frame's first atom should be offset by ~0.5 Å along x.
        let dx = parsed[1].residues[0].atoms[0].position.x
            - parsed[0].residues[0].atoms[0].position.x;
        assert!((dx - 0.5).abs() < 1e-3);
    }

    #[test]
    fn handles_alt_loc() {
        // Two records for the same N atom with different alt_loc; only A should be kept.
        let pdb = concat!(
            "ATOM      1  N  AALA A   1       0.000   0.000   0.000  0.50  0.00           N\n",
            "ATOM      2  N  BALA A   1       1.000   0.000   0.000  0.50  0.00           N\n",
            "ATOM      3  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n",
            "TER\n"
        );
        let s = read_pdb(pdb.as_bytes()).unwrap();
        let r = &s.residues[0];
        assert_eq!(r.atoms.len(), 2);
        assert_eq!(r.atoms[0].name, "N");
        assert!((r.atoms[0].position - Vec3::new(0.0, 0.0, 0.0)).norm() < 1e-6);
    }
}
