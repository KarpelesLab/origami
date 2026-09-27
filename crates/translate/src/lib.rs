pub mod fasta;
pub mod orf;
pub mod translate;

pub use fasta::{FastaError, Record, parse_fasta};
pub use orf::{Frame, Orf, find_orfs};
pub use translate::{TranslationError, TranslationOutcome, translate_codons};
