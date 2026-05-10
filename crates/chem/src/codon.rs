use std::fmt;
use std::str::FromStr;
use thiserror::Error;

use crate::amino_acid::AminoAcid;

/// Single RNA base. `T` in DNA input is normalized to `U`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Base {
    A,
    C,
    G,
    U,
}

impl Base {
    pub const fn from_byte(b: u8) -> Option<Self> {
        Some(match b {
            b'A' | b'a' => Base::A,
            b'C' | b'c' => Base::C,
            b'G' | b'g' => Base::G,
            b'U' | b'u' | b'T' | b't' => Base::U,
            _ => return None,
        })
    }

    pub const fn as_byte(self) -> u8 {
        match self {
            Base::A => b'A',
            Base::C => b'C',
            Base::G => b'G',
            Base::U => b'U',
        }
    }
}

impl fmt::Display for Base {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Base::A => "A",
            Base::C => "C",
            Base::G => "G",
            Base::U => "U",
        })
    }
}

/// A 5'→3' RNA triplet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Codon(pub [Base; 3]);

impl Codon {
    pub fn new(a: Base, b: Base, c: Base) -> Self {
        Codon([a, b, c])
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ParseCodonError> {
        if bytes.len() != 3 {
            return Err(ParseCodonError::WrongLength(bytes.len()));
        }
        let a = Base::from_byte(bytes[0]).ok_or(ParseCodonError::InvalidBase(bytes[0] as char))?;
        let b = Base::from_byte(bytes[1]).ok_or(ParseCodonError::InvalidBase(bytes[1] as char))?;
        let c = Base::from_byte(bytes[2]).ok_or(ParseCodonError::InvalidBase(bytes[2] as char))?;
        Ok(Codon([a, b, c]))
    }

    pub fn translate(self) -> Translation {
        // Standard genetic code (NCBI table 1). Tabulated literally for
        // legibility; the compiler folds this to a jump table.
        use Base::*;
        let Codon([a, b, c]) = self;
        match (a, b, c) {
            // Phenylalanine
            (U, U, U) | (U, U, C) => Translation::Amino(AminoAcid::Phe),
            // Leucine
            (U, U, A) | (U, U, G)
            | (C, U, U) | (C, U, C) | (C, U, A) | (C, U, G) => Translation::Amino(AminoAcid::Leu),
            // Isoleucine
            (A, U, U) | (A, U, C) | (A, U, A) => Translation::Amino(AminoAcid::Ile),
            // Methionine (and start codon)
            (A, U, G) => Translation::Amino(AminoAcid::Met),
            // Valine
            (G, U, _) => Translation::Amino(AminoAcid::Val),

            // Serine — note the split: UCN and AGU/AGC
            (U, C, _) | (A, G, U) | (A, G, C) => Translation::Amino(AminoAcid::Ser),
            // Proline
            (C, C, _) => Translation::Amino(AminoAcid::Pro),
            // Threonine
            (A, C, _) => Translation::Amino(AminoAcid::Thr),
            // Alanine
            (G, C, _) => Translation::Amino(AminoAcid::Ala),

            // Tyrosine
            (U, A, U) | (U, A, C) => Translation::Amino(AminoAcid::Tyr),
            // Stop (UAA, UAG, UGA)
            (U, A, A) | (U, A, G) | (U, G, A) => Translation::Stop,
            // Histidine
            (C, A, U) | (C, A, C) => Translation::Amino(AminoAcid::His),
            // Glutamine
            (C, A, A) | (C, A, G) => Translation::Amino(AminoAcid::Gln),
            // Asparagine
            (A, A, U) | (A, A, C) => Translation::Amino(AminoAcid::Asn),
            // Lysine
            (A, A, A) | (A, A, G) => Translation::Amino(AminoAcid::Lys),
            // Aspartate
            (G, A, U) | (G, A, C) => Translation::Amino(AminoAcid::Asp),
            // Glutamate
            (G, A, A) | (G, A, G) => Translation::Amino(AminoAcid::Glu),

            // Cysteine
            (U, G, U) | (U, G, C) => Translation::Amino(AminoAcid::Cys),
            // Tryptophan
            (U, G, G) => Translation::Amino(AminoAcid::Trp),
            // Arginine — CGN and AGA/AGG
            (C, G, _) | (A, G, A) | (A, G, G) => Translation::Amino(AminoAcid::Arg),
            // Glycine
            (G, G, _) => Translation::Amino(AminoAcid::Gly),
        }
    }

    /// Iterate over all 64 codons in canonical order (AAA, AAC, …, UUU).
    pub fn all() -> impl Iterator<Item = Codon> {
        const BASES: [Base; 4] = [Base::A, Base::C, Base::G, Base::U];
        BASES
            .into_iter()
            .flat_map(move |a| BASES.into_iter().flat_map(move |b| BASES.into_iter().map(move |c| Codon([a, b, c]))))
    }
}

impl fmt::Display for Codon {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}{}", self.0[0], self.0[1], self.0[2])
    }
}

#[derive(Debug, Error)]
pub enum ParseCodonError {
    #[error("codon must be exactly 3 bases, got {0}")]
    WrongLength(usize),
    #[error("invalid base character {0:?}")]
    InvalidBase(char),
}

impl FromStr for Codon {
    type Err = ParseCodonError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Codon::from_bytes(s.as_bytes())
    }
}

/// Outcome of translating a codon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Translation {
    Amino(AminoAcid),
    Stop,
}

impl Translation {
    pub fn amino(self) -> Option<AminoAcid> {
        match self {
            Translation::Amino(a) => Some(a),
            Translation::Stop => None,
        }
    }

    pub fn is_stop(self) -> bool {
        matches!(self, Translation::Stop)
    }
}
