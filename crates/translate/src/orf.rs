use chem::{AminoAcid, Codon, Translation};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Frame {
    Plus0,
    Plus1,
    Plus2,
}

impl Frame {
    pub const ALL: [Frame; 3] = [Frame::Plus0, Frame::Plus1, Frame::Plus2];
    pub const fn offset(self) -> usize {
        match self {
            Frame::Plus0 => 0,
            Frame::Plus1 => 1,
            Frame::Plus2 => 2,
        }
    }
    pub const fn label(self) -> &'static str {
        match self {
            Frame::Plus0 => "+0",
            Frame::Plus1 => "+1",
            Frame::Plus2 => "+2",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Orf {
    pub frame: Frame,
    /// Index of the start (AUG) codon's first base in the input sequence.
    pub start: usize,
    /// Index just past the stop codon's last base. If unterminated, equals `seq.len()`.
    pub end: usize,
    pub protein: Vec<AminoAcid>,
    pub terminated: bool,
}

impl Orf {
    pub fn len(&self) -> usize {
        self.protein.len()
    }
    pub fn is_empty(&self) -> bool {
        self.protein.is_empty()
    }
}

/// Find ORFs in the three forward reading frames.
///
/// An ORF starts at AUG (Met) and runs to the next in-frame stop codon. If the
/// frame runs off the end of the sequence without a stop, the ORF is still
/// emitted with `terminated = false`. Only ORFs whose protein length is at
/// least `min_aa` are returned.
pub fn find_orfs(seq: &[u8], min_aa: usize) -> Vec<Orf> {
    let mut out = Vec::new();
    for frame in Frame::ALL {
        let offset = frame.offset();
        let mut current: Option<(usize, Vec<AminoAcid>)> = None;
        let mut i = offset;
        while i + 3 <= seq.len() {
            let triplet = &seq[i..i + 3];
            // Skip any non-ACGU triplet by treating as a frame break.
            let codon = match Codon::from_bytes(triplet) {
                Ok(c) => c,
                Err(_) => {
                    if let Some((start, protein)) = current.take() {
                        if protein.len() >= min_aa {
                            out.push(Orf { frame, start, end: i, protein, terminated: false });
                        }
                    }
                    i += 3;
                    continue;
                }
            };
            match codon.translate() {
                Translation::Amino(aa) => {
                    if current.is_none() && aa == AminoAcid::Met {
                        current = Some((i, vec![aa]));
                    } else if let Some((_start, protein)) = current.as_mut() {
                        protein.push(aa);
                    }
                }
                Translation::Stop => {
                    if let Some((start, protein)) = current.take() {
                        if protein.len() >= min_aa {
                            out.push(Orf {
                                frame,
                                start,
                                end: i + 3,
                                protein,
                                terminated: true,
                            });
                        }
                    }
                }
            }
            i += 3;
        }
        if let Some((start, protein)) = current.take() {
            if protein.len() >= min_aa {
                out.push(Orf { frame, start, end: seq.len(), protein, terminated: false });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_orf_in_frame_zero() {
        // AUG GCA UGG UAA = MAW + stop, in frame 0.
        // (Note: this sequence also has an AUG at pos 5 in frame +2,
        // producing a 2-aa ORF "MV" — verified separately below.)
        let orfs = find_orfs(b"AUGGCAUGGUAA", 1);
        let f0 = orfs.iter().find(|o| o.frame == Frame::Plus0).unwrap();
        assert_eq!(f0.start, 0);
        assert_eq!(f0.end, 12);
        assert!(f0.terminated);
        assert_eq!(f0.len(), 3);
        assert_eq!(f0.protein, vec![AminoAcid::Met, AminoAcid::Ala, AminoAcid::Trp]);
    }

    #[test]
    fn finds_overlapping_orfs_in_different_frames() {
        // AUGGCAUGGUAA contains an in-frame Met at +0 (→MAW) and another
        // out-of-frame Met at position 5 in +2 (→MV, unterminated).
        let orfs = find_orfs(b"AUGGCAUGGUAA", 1);
        assert_eq!(orfs.len(), 2);
        assert!(orfs.iter().any(|o| o.frame == Frame::Plus0 && o.protein.len() == 3));
        assert!(orfs.iter().any(|o| o.frame == Frame::Plus2 && o.protein.len() == 2));
    }

    #[test]
    fn finds_orf_with_leading_utr() {
        // CC-AUG-GCA-UAA: Met-Ala then stop, in frame +2
        let orfs = find_orfs(b"CCAUGGCAUAA", 1);
        assert!(orfs.iter().any(|o| o.frame == Frame::Plus2 && o.protein.len() == 2));
    }

    #[test]
    fn min_length_filters_short_orfs() {
        // Both ORFs (length 3 and 2) filtered out by min_aa = 5
        let orfs = find_orfs(b"AUGGCAUGGUAA", 5);
        assert!(orfs.is_empty());
    }

    #[test]
    fn unterminated_orf_returned_when_no_stop() {
        // AUG GCA UGG (no stop): one frame-0 ORF "MAW",
        // plus a frame-+2 ORF "M" starting at pos 5 (also unterminated).
        let orfs = find_orfs(b"AUGGCAUGG", 1);
        let f0 = orfs.iter().find(|o| o.frame == Frame::Plus0).unwrap();
        assert!(!f0.terminated);
        assert_eq!(f0.len(), 3);
    }
}
