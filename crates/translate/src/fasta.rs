use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    pub id: String,
    pub description: String,
    /// Sequence with whitespace stripped, uppercased, T→U normalized.
    pub sequence: Vec<u8>,
}

#[derive(Debug, Error)]
pub enum FastaError {
    #[error("FASTA input is empty")]
    Empty,
    #[error("invalid character {0:?} at line {1}")]
    InvalidBase(char, usize),
    #[error("sequence data appears before first header")]
    SequenceBeforeHeader,
}

pub fn parse_fasta(input: &str) -> Result<Vec<Record>, FastaError> {
    let mut records = Vec::new();
    let mut current: Option<Record> = None;

    for (lineno_zero, raw_line) in input.lines().enumerate() {
        let lineno = lineno_zero + 1;
        let line = raw_line.trim_end();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix('>') {
            if let Some(rec) = current.take() {
                records.push(rec);
            }
            let (id, description) = split_header(rest);
            current = Some(Record {
                id,
                description,
                sequence: Vec::new(),
            });
        } else {
            let rec = current
                .as_mut()
                .ok_or(FastaError::SequenceBeforeHeader)?;
            for ch in line.chars() {
                if ch.is_ascii_whitespace() {
                    continue;
                }
                let normalized = match ch.to_ascii_uppercase() {
                    'A' => b'A',
                    'C' => b'C',
                    'G' => b'G',
                    'U' | 'T' => b'U',
                    'N' => b'N',
                    _ => return Err(FastaError::InvalidBase(ch, lineno)),
                };
                rec.sequence.push(normalized);
            }
        }
    }
    if let Some(rec) = current.take() {
        records.push(rec);
    }
    if records.is_empty() {
        return Err(FastaError::Empty);
    }
    Ok(records)
}

fn split_header(rest: &str) -> (String, String) {
    let trimmed = rest.trim_start();
    match trimmed.split_once(char::is_whitespace) {
        Some((id, desc)) => (id.to_owned(), desc.trim().to_owned()),
        None => (trimmed.to_owned(), String::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_record() {
        let input = ">test\nAUGGCAUGGUAA\n";
        let records = parse_fasta(input).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].id, "test");
        assert_eq!(records[0].sequence, b"AUGGCAUGGUAA");
    }

    #[test]
    fn parses_multi_line_sequence() {
        let input = ">x\nAUGG\nCAUG\nGUAA";
        let records = parse_fasta(input).unwrap();
        assert_eq!(records[0].sequence, b"AUGGCAUGGUAA");
    }

    #[test]
    fn parses_multiple_records() {
        let input = ">a desc1\nAUG\n>b desc2\nUAA\n";
        let records = parse_fasta(input).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].id, "a");
        assert_eq!(records[0].description, "desc1");
        assert_eq!(records[1].id, "b");
        assert_eq!(records[1].description, "desc2");
    }

    #[test]
    fn dna_t_normalized_to_u() {
        let input = ">x\nATGGCATGGTAA\n";
        let records = parse_fasta(input).unwrap();
        assert_eq!(records[0].sequence, b"AUGGCAUGGUAA");
    }

    #[test]
    fn lowercase_handled() {
        let records = parse_fasta(">x\naug\n").unwrap();
        assert_eq!(records[0].sequence, b"AUG");
    }

    #[test]
    fn rejects_bad_characters() {
        let err = parse_fasta(">x\nAUGZ\n").unwrap_err();
        assert!(matches!(err, FastaError::InvalidBase('Z', _)));
    }

    #[test]
    fn rejects_empty_input() {
        assert!(matches!(parse_fasta(""), Err(FastaError::Empty)));
    }

    #[test]
    fn rejects_sequence_before_header() {
        assert!(matches!(
            parse_fasta("AUG\n>x\n"),
            Err(FastaError::SequenceBeforeHeader)
        ));
    }
}
