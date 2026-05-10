use chem::{AminoAcid, Codon, Translation};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranslationOutcome {
    pub protein: Vec<AminoAcid>,
    /// True if translation ended at a canonical stop codon. False if it ran off
    /// the end of the input without one.
    pub terminated: bool,
}

#[derive(Debug, Error)]
pub enum TranslationError {
    #[error("invalid base {0:?} at position {1}")]
    InvalidBase(char, usize),
    #[error("empty sequence")]
    Empty,
}

/// Translate a sequence reading in-frame from position 0.
/// Stops at the first stop codon or at end-of-input. Trailing nucleotides
/// (1 or 2) past the last full codon are silently ignored.
pub fn translate_codons(seq: &[u8]) -> Result<TranslationOutcome, TranslationError> {
    if seq.is_empty() {
        return Err(TranslationError::Empty);
    }
    let mut protein = Vec::with_capacity(seq.len() / 3);
    let mut terminated = false;
    let mut i = 0;
    while i + 3 <= seq.len() {
        let codon = Codon::from_bytes(&seq[i..i + 3]).map_err(|_| {
            // Find the first non-ACGU character in this triplet.
            for offset in 0..3 {
                let b = seq[i + offset];
                if !matches!(b, b'A' | b'C' | b'G' | b'U') {
                    return TranslationError::InvalidBase(b as char, i + offset);
                }
            }
            unreachable!("Codon::from_bytes failed but all bytes are ACGU")
        })?;
        match codon.translate() {
            Translation::Amino(aa) => protein.push(aa),
            Translation::Stop => {
                terminated = true;
                break;
            }
        }
        i += 3;
    }
    Ok(TranslationOutcome { protein, terminated })
}

/// Convenience: render a list of amino acids as a one-letter string.
pub fn one_letter_string(protein: &[AminoAcid]) -> String {
    protein.iter().map(|a| a.one_letter()).collect()
}

/// Convenience: render a list of amino acids as a hyphen-separated three-letter string.
pub fn three_letter_string(protein: &[AminoAcid]) -> String {
    let mut s = String::with_capacity(protein.len() * 4);
    for (idx, aa) in protein.iter().enumerate() {
        if idx > 0 {
            s.push('-');
        }
        s.push_str(aa.three_letter());
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_translation_with_stop() {
        // AUG GCA UGG UAA → Met-Ala-Trp + stop
        let out = translate_codons(b"AUGGCAUGGUAA").unwrap();
        assert!(out.terminated);
        assert_eq!(
            out.protein,
            vec![AminoAcid::Met, AminoAcid::Ala, AminoAcid::Trp]
        );
        assert_eq!(one_letter_string(&out.protein), "MAW");
    }

    #[test]
    fn unterminated_translation() {
        // AUG GCA UGG → Met-Ala-Trp, no stop
        let out = translate_codons(b"AUGGCAUGG").unwrap();
        assert!(!out.terminated);
        assert_eq!(out.protein.len(), 3);
    }

    #[test]
    fn trailing_partial_codon_ignored() {
        let out = translate_codons(b"AUGGCAUGGU").unwrap(); // 10 bases, last 1 ignored
        assert_eq!(out.protein.len(), 3);
    }

    #[test]
    fn three_letter_rendering() {
        let out = translate_codons(b"AUGGCA").unwrap();
        assert_eq!(three_letter_string(&out.protein), "Met-Ala");
    }
}
