use std::fmt;
use std::str::FromStr;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AminoAcid {
    Ala,
    Arg,
    Asn,
    Asp,
    Cys,
    Gln,
    Glu,
    Gly,
    His,
    Ile,
    Leu,
    Lys,
    Met,
    Phe,
    Pro,
    Ser,
    Thr,
    Trp,
    Tyr,
    Val,
}

impl AminoAcid {
    pub const ALL: [AminoAcid; 20] = [
        AminoAcid::Ala,
        AminoAcid::Arg,
        AminoAcid::Asn,
        AminoAcid::Asp,
        AminoAcid::Cys,
        AminoAcid::Gln,
        AminoAcid::Glu,
        AminoAcid::Gly,
        AminoAcid::His,
        AminoAcid::Ile,
        AminoAcid::Leu,
        AminoAcid::Lys,
        AminoAcid::Met,
        AminoAcid::Phe,
        AminoAcid::Pro,
        AminoAcid::Ser,
        AminoAcid::Thr,
        AminoAcid::Trp,
        AminoAcid::Tyr,
        AminoAcid::Val,
    ];

    pub const fn one_letter(self) -> char {
        match self {
            AminoAcid::Ala => 'A',
            AminoAcid::Arg => 'R',
            AminoAcid::Asn => 'N',
            AminoAcid::Asp => 'D',
            AminoAcid::Cys => 'C',
            AminoAcid::Gln => 'Q',
            AminoAcid::Glu => 'E',
            AminoAcid::Gly => 'G',
            AminoAcid::His => 'H',
            AminoAcid::Ile => 'I',
            AminoAcid::Leu => 'L',
            AminoAcid::Lys => 'K',
            AminoAcid::Met => 'M',
            AminoAcid::Phe => 'F',
            AminoAcid::Pro => 'P',
            AminoAcid::Ser => 'S',
            AminoAcid::Thr => 'T',
            AminoAcid::Trp => 'W',
            AminoAcid::Tyr => 'Y',
            AminoAcid::Val => 'V',
        }
    }

    pub const fn three_letter(self) -> &'static str {
        match self {
            AminoAcid::Ala => "Ala",
            AminoAcid::Arg => "Arg",
            AminoAcid::Asn => "Asn",
            AminoAcid::Asp => "Asp",
            AminoAcid::Cys => "Cys",
            AminoAcid::Gln => "Gln",
            AminoAcid::Glu => "Glu",
            AminoAcid::Gly => "Gly",
            AminoAcid::His => "His",
            AminoAcid::Ile => "Ile",
            AminoAcid::Leu => "Leu",
            AminoAcid::Lys => "Lys",
            AminoAcid::Met => "Met",
            AminoAcid::Phe => "Phe",
            AminoAcid::Pro => "Pro",
            AminoAcid::Ser => "Ser",
            AminoAcid::Thr => "Thr",
            AminoAcid::Trp => "Trp",
            AminoAcid::Tyr => "Tyr",
            AminoAcid::Val => "Val",
        }
    }

    pub const fn full_name(self) -> &'static str {
        match self {
            AminoAcid::Ala => "Alanine",
            AminoAcid::Arg => "Arginine",
            AminoAcid::Asn => "Asparagine",
            AminoAcid::Asp => "Aspartate",
            AminoAcid::Cys => "Cysteine",
            AminoAcid::Gln => "Glutamine",
            AminoAcid::Glu => "Glutamate",
            AminoAcid::Gly => "Glycine",
            AminoAcid::His => "Histidine",
            AminoAcid::Ile => "Isoleucine",
            AminoAcid::Leu => "Leucine",
            AminoAcid::Lys => "Lysine",
            AminoAcid::Met => "Methionine",
            AminoAcid::Phe => "Phenylalanine",
            AminoAcid::Pro => "Proline",
            AminoAcid::Ser => "Serine",
            AminoAcid::Thr => "Threonine",
            AminoAcid::Trp => "Tryptophan",
            AminoAcid::Tyr => "Tyrosine",
            AminoAcid::Val => "Valine",
        }
    }

    pub const fn from_one_letter(c: char) -> Option<Self> {
        Some(match c {
            'A' | 'a' => AminoAcid::Ala,
            'R' | 'r' => AminoAcid::Arg,
            'N' | 'n' => AminoAcid::Asn,
            'D' | 'd' => AminoAcid::Asp,
            'C' | 'c' => AminoAcid::Cys,
            'Q' | 'q' => AminoAcid::Gln,
            'E' | 'e' => AminoAcid::Glu,
            'G' | 'g' => AminoAcid::Gly,
            'H' | 'h' => AminoAcid::His,
            'I' | 'i' => AminoAcid::Ile,
            'L' | 'l' => AminoAcid::Leu,
            'K' | 'k' => AminoAcid::Lys,
            'M' | 'm' => AminoAcid::Met,
            'F' | 'f' => AminoAcid::Phe,
            'P' | 'p' => AminoAcid::Pro,
            'S' | 's' => AminoAcid::Ser,
            'T' | 't' => AminoAcid::Thr,
            'W' | 'w' => AminoAcid::Trp,
            'Y' | 'y' => AminoAcid::Tyr,
            'V' | 'v' => AminoAcid::Val,
            _ => return None,
        })
    }

    pub fn from_three_letter(s: &str) -> Option<Self> {
        if s.len() != 3 {
            return None;
        }
        Self::ALL
            .into_iter()
            .find(|aa| s.eq_ignore_ascii_case(aa.three_letter()))
    }
}

impl fmt::Display for AminoAcid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.three_letter())
    }
}

#[derive(Debug, Error)]
pub enum ParseAminoAcidError {
    #[error("not a recognized amino-acid code: {0:?}")]
    Unrecognized(String),
}

impl FromStr for AminoAcid {
    type Err = ParseAminoAcidError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed = s.trim();
        if trimmed.len() == 1 {
            if let Some(aa) = Self::from_one_letter(trimmed.chars().next().unwrap()) {
                return Ok(aa);
            }
        }
        if let Some(aa) = Self::from_three_letter(trimmed) {
            return Ok(aa);
        }
        Err(ParseAminoAcidError::Unrecognized(trimmed.to_owned()))
    }
}
