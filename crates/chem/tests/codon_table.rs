use chem::{AminoAcid, Codon, Translation};
use std::collections::HashSet;
use std::str::FromStr;

#[test]
fn all_64_codons_are_classified() {
    let mut count = 0;
    for codon in Codon::all() {
        let _ = codon.translate();
        count += 1;
    }
    assert_eq!(count, 64);
}

#[test]
fn all_twenty_amino_acids_are_reachable() {
    let mut seen: HashSet<AminoAcid> = HashSet::new();
    for codon in Codon::all() {
        if let Translation::Amino(aa) = codon.translate() {
            seen.insert(aa);
        }
    }
    assert_eq!(seen.len(), 20, "missing AAs: {:?}",
        AminoAcid::ALL.iter().filter(|a| !seen.contains(a)).collect::<Vec<_>>());
}

#[test]
fn three_stop_codons() {
    let stops: Vec<Codon> = Codon::all()
        .filter(|c| c.translate().is_stop())
        .collect();
    assert_eq!(stops.len(), 3);
    let stop_strs: HashSet<String> = stops.iter().map(|c| c.to_string()).collect();
    assert!(stop_strs.contains("UAA"));
    assert!(stop_strs.contains("UAG"));
    assert!(stop_strs.contains("UGA"));
}

#[test]
fn aug_is_methionine_start() {
    let aug = Codon::from_str("AUG").unwrap();
    assert_eq!(aug.translate(), Translation::Amino(AminoAcid::Met));
}

#[test]
fn dna_t_is_normalized_to_u() {
    // ATG (DNA) and AUG (RNA) should both translate to Met.
    let dna = Codon::from_str("ATG").unwrap();
    let rna = Codon::from_str("AUG").unwrap();
    assert_eq!(dna, rna);
    assert_eq!(dna.translate(), Translation::Amino(AminoAcid::Met));
}

#[test]
fn lowercase_input_works() {
    let c = Codon::from_str("aug").unwrap();
    assert_eq!(c.translate(), Translation::Amino(AminoAcid::Met));
}

#[test]
fn invalid_codons_rejected() {
    assert!(Codon::from_str("AU").is_err()); // too short
    assert!(Codon::from_str("AUGC").is_err()); // too long
    assert!(Codon::from_str("AXG").is_err()); // bad base
}

#[test]
fn known_codon_assignments() {
    // Spot-check against NCBI table 1.
    let cases = [
        ("UUU", AminoAcid::Phe),
        ("UUC", AminoAcid::Phe),
        ("UUA", AminoAcid::Leu),
        ("CUG", AminoAcid::Leu),
        ("AUC", AminoAcid::Ile),
        ("AUG", AminoAcid::Met),
        ("GUC", AminoAcid::Val),
        ("UCG", AminoAcid::Ser),
        ("AGU", AminoAcid::Ser), // serine alternative
        ("CCG", AminoAcid::Pro),
        ("ACU", AminoAcid::Thr),
        ("GCC", AminoAcid::Ala),
        ("UAU", AminoAcid::Tyr),
        ("CAC", AminoAcid::His),
        ("CAA", AminoAcid::Gln),
        ("AAU", AminoAcid::Asn),
        ("AAA", AminoAcid::Lys),
        ("GAU", AminoAcid::Asp),
        ("GAG", AminoAcid::Glu),
        ("UGU", AminoAcid::Cys),
        ("UGG", AminoAcid::Trp),
        ("CGU", AminoAcid::Arg),
        ("AGA", AminoAcid::Arg), // arginine alternative
        ("GGG", AminoAcid::Gly),
    ];
    for (s, expected) in cases {
        let c = Codon::from_str(s).unwrap();
        assert_eq!(
            c.translate(),
            Translation::Amino(expected),
            "codon {s} should be {expected:?}"
        );
    }
}

#[test]
fn one_letter_round_trip() {
    for aa in AminoAcid::ALL {
        let c = aa.one_letter();
        let parsed = AminoAcid::from_one_letter(c).unwrap();
        assert_eq!(parsed, aa);
    }
}

#[test]
fn three_letter_round_trip() {
    for aa in AminoAcid::ALL {
        let s = aa.three_letter();
        let parsed = AminoAcid::from_three_letter(s).unwrap();
        assert_eq!(parsed, aa);
        // Case-insensitive
        let parsed_upper = AminoAcid::from_three_letter(&s.to_uppercase()).unwrap();
        assert_eq!(parsed_upper, aa);
    }
}

#[test]
fn properties_are_consistent() {
    use approx::assert_relative_eq;

    let leu = AminoAcid::Leu.properties();
    assert_relative_eq!(leu.hydropathy, 3.8, epsilon = 1e-6);
    assert_eq!(leu.net_charge, 0.0);
    assert!(leu.sidechain_pka.is_none());

    let lys = AminoAcid::Lys.properties();
    assert_eq!(lys.net_charge, 1.0);
    assert!(lys.sidechain_pka.is_some());

    let gly = AminoAcid::Gly.properties();
    assert_relative_eq!(gly.residue_mass_da, 57.0519, epsilon = 1e-3);
}
