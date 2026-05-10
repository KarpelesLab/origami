use translate::{parse_fasta, translate_codons};
use translate::translate::one_letter_string;

/// Human preproinsulin (UniProt P01308). The reference protein sequence below
/// is the canonical 110-residue translation product including the signal
/// peptide. The mRNA below is NCBI RefSeq NM_000207 CDS.
#[test]
fn human_preproinsulin_translates_correctly() {
    let fasta = include_str!("../../../examples/insulin.fasta");
    let records = parse_fasta(fasta).expect("parse insulin.fasta");
    assert_eq!(records.len(), 1);

    let outcome = translate_codons(&records[0].sequence).expect("translate");
    assert!(outcome.terminated, "should hit a stop codon");

    let actual = one_letter_string(&outcome.protein);
    let expected = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN";
    assert_eq!(actual, expected, "preproinsulin sequence mismatch");
    assert_eq!(actual.len(), 110);
}
