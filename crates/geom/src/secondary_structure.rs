//! Per-residue secondary-structure assignment via Ramachandran-region
//! lookup (a.k.a. "DSSP-lite"). For each residue:
//!
//!   • φ = dihedral(C(i-1), N(i), CA(i), C(i))
//!   • ψ = dihedral(N(i), CA(i), C(i), N(i+1))
//!
//! and a φ / ψ pair is classified as:
//!
//!   H — α-helix       (φ ∈ [-90°, -30°], ψ ∈ [-77°,  -7°])
//!   E — β-strand      (φ ∈ [-180°, -45°], ψ ∈ [ 90°, 180°] ∪ [-180°, -160°])
//!   C — coil / other  (everything else, including missing termini)
//!
//! The Ramachandran-region cutoffs are the conventional Lovell-style
//! "allowed" boxes; they catch most regular secondary structure but
//! they miss H-bond-based features that real DSSP would resolve (3-10
//! helices, sheet pairing, distinguishing turn vs coil). That's the
//! "lite" — good enough to characterise an MD trajectory's α / β /
//! coil composition over time without implementing the full Kabsch-
//! Sander H-bond algorithm.

use crate::measure::dihedral;
use crate::structure::Structure;

/// Three-letter abbreviations consistent with DSSP / STRIDE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SsType {
    Helix,
    Strand,
    Coil,
}

impl SsType {
    pub fn as_char(self) -> char {
        match self {
            Self::Helix => 'H',
            Self::Strand => 'E',
            Self::Coil => 'C',
        }
    }
}

/// φ in radians for residue `i`, requires the previous residue's C
/// atom plus this residue's N / CA / C. Returns `None` at the chain
/// start or when any required atom is missing.
pub fn phi(structure: &Structure, i: usize) -> Option<f64> {
    if i == 0 {
        return None;
    }
    // Don't span chain breaks — φ is only meaningful within one chain.
    if structure.residues[i - 1].chain != structure.residues[i].chain {
        return None;
    }
    let prev_c = structure.residues[i - 1].position("C")?;
    let n = structure.residues[i].position("N")?;
    let ca = structure.residues[i].position("CA")?;
    let c = structure.residues[i].position("C")?;
    Some(dihedral(prev_c, n, ca, c))
}

/// ψ in radians for residue `i`, requires N / CA / C of this residue
/// plus the next residue's N. Returns `None` at the chain end.
pub fn psi(structure: &Structure, i: usize) -> Option<f64> {
    if i + 1 >= structure.residues.len() {
        return None;
    }
    if structure.residues[i].chain != structure.residues[i + 1].chain {
        return None;
    }
    let n = structure.residues[i].position("N")?;
    let ca = structure.residues[i].position("CA")?;
    let c = structure.residues[i].position("C")?;
    let next_n = structure.residues[i + 1].position("N")?;
    Some(dihedral(n, ca, c, next_n))
}

/// Classify a (φ, ψ) pair into helix / strand / coil. Both angles are
/// in radians. The boxes are the conventional Ramachandran "allowed"
/// regions, in degrees:
///
///   H: φ ∈ [-90, -30], ψ ∈ [-77, -7]
///   E: φ ∈ [-180, -45], (ψ ∈ [90, 180] ∪ ψ ∈ [-180, -160])
pub fn classify(phi_rad: f64, psi_rad: f64) -> SsType {
    let p = phi_rad.to_degrees();
    let q = psi_rad.to_degrees();
    if (-90.0..=-30.0).contains(&p) && (-77.0..=-7.0).contains(&q) {
        return SsType::Helix;
    }
    if (-180.0..=-45.0).contains(&p) {
        if (90.0..=180.0).contains(&q) || (-180.0..=-160.0).contains(&q) {
            return SsType::Strand;
        }
    }
    SsType::Coil
}

/// Per-residue secondary-structure string for the whole structure.
/// Residues whose φ or ψ can't be computed (chain termini, missing
/// atoms) come out as 'C' so the length always matches the residue
/// count.
pub fn secondary_structure_string(structure: &Structure) -> String {
    let mut out = String::with_capacity(structure.residues.len());
    for i in 0..structure.residues.len() {
        let ss = match (phi(structure, i), psi(structure, i)) {
            (Some(p), Some(q)) => classify(p, q),
            _ => SsType::Coil,
        };
        out.push(ss.as_char());
    }
    out
}

/// Per-frame helix / strand / coil counts in a single pass. Useful
/// for aggregate trajectory statistics (e.g. average %-helix).
pub fn ss_counts(structure: &Structure) -> (usize, usize, usize) {
    let mut h = 0;
    let mut e = 0;
    let mut c = 0;
    for i in 0..structure.residues.len() {
        let ss = match (phi(structure, i), psi(structure, i)) {
            (Some(p), Some(q)) => classify(p, q),
            _ => SsType::Coil,
        };
        match ss {
            SsType::Helix => h += 1,
            SsType::Strand => e += 1,
            SsType::Coil => c += 1,
        }
    }
    (h, e, c)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deg(d: f64) -> f64 { d.to_radians() }

    #[test]
    fn alpha_helix_angles_classify_as_helix() {
        // Canonical α-helix Ramachandran point: (-60°, -45°).
        assert_eq!(classify(deg(-60.0), deg(-45.0)), SsType::Helix);
    }

    #[test]
    fn beta_strand_angles_classify_as_strand() {
        // Canonical antiparallel β-strand: (-120°, +120°).
        assert_eq!(classify(deg(-120.0), deg(120.0)), SsType::Strand);
    }

    #[test]
    fn extended_chain_angles_classify_as_strand_or_coil() {
        // Extended chain is built at φ = -135°, ψ = +135° — that's in
        // the β region (E).
        assert_eq!(classify(deg(-135.0), deg(135.0)), SsType::Strand);
    }

    #[test]
    fn loop_angles_classify_as_coil() {
        // Mid-Ramachandran "left-handed" region (+60, +60) — neither
        // H nor E.
        assert_eq!(classify(deg(60.0), deg(60.0)), SsType::Coil);
    }

}
