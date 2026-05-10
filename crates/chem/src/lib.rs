pub mod amino_acid;
pub mod codon;
pub mod properties;

pub use amino_acid::{AminoAcid, ParseAminoAcidError};
pub use codon::{Base, Codon, ParseCodonError, Translation};
pub use properties::AminoAcidProperties;
