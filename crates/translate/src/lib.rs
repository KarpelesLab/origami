pub mod fasta;
pub mod orf;
pub mod translate;

pub use fasta::{parse_fasta, FastaError, Record};
pub use orf::{find_orfs, Frame, Orf};
pub use translate::{translate_codons, TranslationError, TranslationOutcome};
