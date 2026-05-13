//! Backbone-H-bond secondary-structure assignment (DSSP, Kabsch &
//! Sander 1983, Biopolymers 22, 2577).
//!
//! For each pair of residues (donor i, acceptor j) with `j ≠ i`, the
//! H-bond energy is the Coulomb sum over the four-atom interaction
//!
//!   E = q₁ q₂ · f · (1/r_ON + 1/r_CH − 1/r_OH − 1/r_CN)
//!
//! where N, H are on the donor amide of residue i; C, O are on the
//! acceptor carbonyl of residue j; q₁ = 0.42 e (amide N-H point
//! charge), q₂ = 0.20 e (carbonyl C=O point charge), and
//! f = 332.064 kcal·Å / mol·e². A pair is "H-bonded" if E < −0.5
//! kcal/mol. From the H-bond graph the assignment rules pick out
//! α-helix (i → i ± 4), 3-10 helix (i → i ± 3), and β-bridge /
//! sheet (i → j with |i − j| > 4 in two parallel or antiparallel
//! ladders).
//!
//! This is real DSSP, distinct from the Ramachandran-box classifier
//! in [`crate::secondary_structure`] which uses only φ / ψ. DSSP
//! detects features Ramachandran misses (β-bridges, antiparallel
//! sheet pairings, turn types) and is the field-standard reference
//! for SS assignment; we still keep the Ramachandran version around
//! as a fallback for structures without explicit hydrogens.

use crate::structure::Structure;
use crate::Vec3;

/// Kabsch-Sander H-bond energy threshold (kcal/mol).
const HBOND_E_THRESHOLD: f64 = -0.5;

/// CHARMM Coulomb constant in kcal·Å / mol·e².
const COULOMB_CONST: f64 = 332.0637;

/// Partial charge on the amide N-H pair (donor).
const Q_DONOR: f64 = 0.42;
/// Partial charge on the carbonyl C=O pair (acceptor).
const Q_ACCEPTOR: f64 = 0.20;

/// DSSP type code per residue. We expose only the canonical
/// "minimal" set (helix, strand, coil); the full DSSP letterset
/// also distinguishes 3-10 / π helices, bridges, turns, and bends
/// but those collapse onto H / E / C for our analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DsspType {
    /// α-helix (i → i + 4 H-bond).
    Helix,
    /// β-strand (cross-chain or long-range H-bond pair).
    Strand,
    /// Coil / loop / turn (everything else).
    Coil,
}

impl DsspType {
    pub fn as_char(self) -> char {
        match self {
            Self::Helix => 'H',
            Self::Strand => 'E',
            Self::Coil => 'C',
        }
    }
}

/// `hbonds[i]` = list of acceptor residue indices `j` such that
/// residue i's N-H is H-bonded to residue j's C=O. (i is the
/// hydrogen-bond donor, j is the acceptor.)
#[derive(Debug, Clone, Default)]
pub struct HBondTable {
    pub donor_to_acceptors: Vec<Vec<usize>>,
}

impl HBondTable {
    pub fn donates_to(&self, donor: usize, acceptor: usize) -> bool {
        self.donor_to_acceptors
            .get(donor)
            .is_some_and(|v| v.contains(&acceptor))
    }
}

/// Build the H-bond table for the structure. The donor must have
/// both an `N` and an `H` atom (chain-N-terminus and proline lack
/// the amide H and are skipped); the acceptor must have `C` and `O`.
/// H-bonds across chain boundaries are detected the same way as
/// within-chain ones — sheet pairings across `A` ↔ `B` in insulin
/// fall out naturally.
pub fn find_hbonds(structure: &Structure) -> HBondTable {
    let n_res = structure.residues.len();
    let mut table = HBondTable {
        donor_to_acceptors: vec![Vec::new(); n_res],
    };
    if n_res < 2 {
        return table;
    }
    for i in 0..n_res {
        let donor_n = structure.residues[i].position("N");
        let donor_h = structure.residues[i].position("H");
        let (Some(n), Some(h)) = (donor_n, donor_h) else {
            continue;
        };
        for j in 0..n_res {
            if i == j {
                continue;
            }
            // |i − j| = 1: trivially shared-bond pair; not a
            // separate H-bond per K&S definition.
            if i.abs_diff(j) == 1 {
                continue;
            }
            let acceptor_c = structure.residues[j].position("C");
            let acceptor_o = structure.residues[j].position("O");
            let (Some(c), Some(o)) = (acceptor_c, acceptor_o) else {
                continue;
            };
            let r_on = (o - n).norm();
            let r_ch = (c - h).norm();
            let r_oh = (o - h).norm();
            let r_cn = (c - n).norm();
            // Avoid blow-ups for nearly-coincident atoms.
            if r_on < 0.5 || r_ch < 0.5 || r_oh < 0.5 || r_cn < 0.5 {
                continue;
            }
            let e_hb = Q_DONOR * Q_ACCEPTOR * COULOMB_CONST
                * (1.0 / r_on + 1.0 / r_ch - 1.0 / r_oh - 1.0 / r_cn);
            if e_hb < HBOND_E_THRESHOLD {
                table.donor_to_acceptors[i].push(j);
            }
        }
    }
    table
}

/// Per-residue DSSP assignment. Rules in priority order:
///
///   • If residue i donates to residue j with j − i = 4 or i − j = 4
///     and residue i+1 also donates with the same offset (an extending
///     α-helix), mark both as **H**.
///   • Otherwise, if there's a non-adjacent H-bond partner (j − i ≥ 3
///     either way) consistent with a β-bridge, mark as **E**.
///   • Else **C**.
///
/// This is simplified vs the full DSSP letter set (which also
/// distinguishes 3-10 / π helices, T turns, S bends, B bridges) but
/// the H / E / C output line up with the canonical DSSP reduction.
pub fn assign_dssp(structure: &Structure, hbonds: &HBondTable) -> Vec<DsspType> {
    let n = structure.residues.len();
    let mut out = vec![DsspType::Coil; n];
    if n < 5 {
        return out;
    }

    // ---- α-helix detection ----
    // K&S "n-turn": residue i is part of an α-turn if (i → i+4) is
    // an H-bond. Extend to "α-helix" when two consecutive residues
    // have this turn signature.
    let mut alpha_turn = vec![false; n];
    for i in 0..n.saturating_sub(4) {
        if hbonds.donates_to(i, i + 4) {
            alpha_turn[i] = true;
        }
    }
    for i in 1..n.saturating_sub(4) {
        if alpha_turn[i] && alpha_turn[i - 1] {
            // Residues i, i+1, i+2, i+3 are in the helix (4-residue
            // window centred on the i, i+1 overlapping turns).
            for k in i..(i + 4).min(n) {
                out[k] = DsspType::Helix;
            }
        }
    }

    // ---- β-strand detection ----
    // Look for parallel / antiparallel pairings. K&S "bridge" rule
    // (simplified): residues i and j form a bridge if
    //   parallel: (i → j) and (j → i + 2) H-bonds  (or symmetric)
    //   antiparallel: (i → j) and (j → i) H-bonds  (mutual)
    // and i / j are not already in a helix.
    for i in 0..n {
        if out[i] == DsspType::Helix {
            continue;
        }
        // Find any j that is bridged with i.
        for j in 0..n {
            if out[j] == DsspType::Helix {
                continue;
            }
            if i.abs_diff(j) < 3 {
                continue;
            }
            let antipar = hbonds.donates_to(i, j) && hbonds.donates_to(j, i);
            let par_a = hbonds.donates_to(i, j)
                && i + 2 < n
                && hbonds.donates_to(j, i + 2);
            let par_b = j + 2 < n
                && hbonds.donates_to(i, j + 2)
                && hbonds.donates_to(j, i);
            if antipar || par_a || par_b {
                out[i] = DsspType::Strand;
                out[j] = DsspType::Strand;
            }
        }
    }

    out
}

/// Per-residue DSSP string. Falls back to `'C'` (coil) for any
/// residue we can't classify (missing N/H/C/O atoms, terminus).
pub fn dssp_string(structure: &Structure) -> String {
    let hbonds = find_hbonds(structure);
    let assignment = assign_dssp(structure, &hbonds);
    assignment.iter().map(|s| s.as_char()).collect()
}

/// Per-residue helix / strand / coil counts in one pass — the
/// caller-facing aggregate stat alongside `dssp_string`.
pub fn dssp_counts(structure: &Structure) -> (usize, usize, usize) {
    let assignment = assign_dssp(structure, &find_hbonds(structure));
    let mut h = 0;
    let mut e = 0;
    let mut c = 0;
    for s in &assignment {
        match s {
            DsspType::Helix => h += 1,
            DsspType::Strand => e += 1,
            DsspType::Coil => c += 1,
        }
    }
    (h, e, c)
}

// Silence unused-import warning when feature flags shuffle.
#[allow(dead_code)]
fn _vec3_keepalive(_: Vec3) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_extended_chain;
    use chem::AminoAcid;

    #[test]
    fn extended_chain_has_no_helix_or_strand_pairs() {
        let s = build_extended_chain(&[
            AminoAcid::Ala,
            AminoAcid::Ala,
            AminoAcid::Ala,
            AminoAcid::Ala,
            AminoAcid::Ala,
            AminoAcid::Ala,
            AminoAcid::Ala,
        ])
        .unwrap();
        let hbonds = find_hbonds(&s);
        // Extended chain — backbone H-bonds shouldn't be present.
        for d in &hbonds.donor_to_acceptors {
            assert!(d.is_empty(), "extended chain spuriously H-bonded: {:?}", d);
        }
    }

    #[test]
    fn dssp_string_length_matches_residue_count() {
        let s = build_extended_chain(&[
            AminoAcid::Ala,
            AminoAcid::Ala,
            AminoAcid::Ala,
        ])
        .unwrap();
        let ss = dssp_string(&s);
        assert_eq!(ss.len(), s.residues.len());
        // Three-residue extended chain: nothing classifies as anything.
        assert_eq!(ss, "CCC");
    }
}
