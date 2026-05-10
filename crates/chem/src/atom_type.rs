//! CHARMM36-granularity atom types for proteins.
//!
//! These follow the standard CHARMM36 protein parameter file naming
//! (par_all36m_prot.prm / top_all36_prot.rtf). Going granular keeps us
//! faithful to the published parameters: each (atom-type-pair) bond
//! constant, etc., comes straight from CHARMM36 without averaging across
//! types. We only include the types we actually use (the standard 20
//! amino acids in their physiological-pH protonation states; no terminal
//! patches, no disulfides, no protonated Asp/Glu, no charged His).
//!
//! Histidine: we model only the HSD (HD1) tautomer — proton on the δ
//! nitrogen, neutral overall.

use crate::amino_acid::AminoAcid;
use crate::element::Element;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AtomType {
    // ---- Carbons ----
    /// Peptide-bond carbonyl C (backbone C; also Arg Cζ guanidinium centre).
    C,
    /// Aromatic carbon (Phe / Tyr ring; Trp 6-ring CZ3, CH2; Trp CD1 of pyrrole).
    CA,
    /// Aromatic carbon next to the bridgehead in indole — Trp CE3, CZ2.
    CAI,
    /// Side-chain carbonyl/carboxylate C (Asn Cγ, Gln Cδ, Asp Cγ, Glu Cδ).
    CC,
    /// sp³ carbon with one hydrogen (backbone Cα non-Gly; CB of Val/Ile/Thr; CG of Leu).
    CT1,
    /// sp³ carbon with two hydrogens (CH₂; e.g. Gly Cα, Leu CB, Met CB/CG).
    CT2,
    /// sp³ CH₂ adjacent to an sp² polar group (CB of Asn/Asp/Gln/Glu/His; Gln CG; Glu CG).
    CT2A,
    /// sp³ carbon with three hydrogens (methyl).
    CT3,
    /// Proline Cα.
    CP1,
    /// Proline CB and CG.
    CP2,
    /// Proline CD.
    CP3,
    /// Histidine ring CG and CD2.
    CPH1,
    /// Histidine ring CE1.
    CPH2,
    /// Tryptophan bridgehead aromatic carbons (CD2, CE2 — fused 5/6 ring).
    CPT,
    /// Tryptophan pyrrole-ring carbons (CG, CD1).
    CY,

    // ---- Nitrogens ----
    /// Proline backbone N (sp², no H).
    N,
    /// Backbone peptide N with H (sp²).
    NH1,
    /// Side-chain amide N with two H (Asn ND2, Gln NE2).
    NH2,
    /// Ammonium N, sp³ +1 (Lys Nζ).
    NH3,
    /// Guanidinium N (Arg NE, NH1, NH2; charge delocalised).
    NC2,
    /// Histidine ring nitrogen — neutral, protonated (HSD tautomer ND1).
    NR1,
    /// Histidine ring nitrogen — neutral, unprotonated (HSD tautomer NE2).
    NR2,
    /// Tryptophan pyrrole N (NE1).
    NY,

    // ---- Oxygens ----
    /// Carbonyl O (backbone, Asn OD1, Gln OE1).
    O,
    /// Carboxylate O (Asp OD1/2, Glu OE1/2; -COO⁻).
    OC,
    /// Hydroxyl O (Ser, Thr, Tyr).
    OH1,

    // ---- Sulfur ----
    /// Sulfur (Met SD thioether, Cys SG thiol — CHARMM uses one type).
    S,

    // ---- Hydrogens ----
    /// Polar H bonded to NH1, NH2, OH1, NY.
    H,
    /// Generic aliphatic H (used for proline side-chain hydrogens).
    HA,
    /// Aliphatic H on CT1 (single H on a CH).
    HA1,
    /// Aliphatic H on CT2 / CT2A (one of two on a CH₂).
    HA2,
    /// Aliphatic H on CT3 (one of three on a methyl).
    HA3,
    /// Backbone Hα for non-Gly, non-Pro residues (the lone H on backbone Cα CT1).
    HB1,
    /// Backbone Hα for Gly (one of two H on backbone Cα CT2).
    HB2,
    /// Charged-amine H bonded to NH3 or NC2 (Lys ammonium, Arg guanidinium).
    HC,
    /// Aromatic ring H bonded to CA.
    HP,
    /// Histidine HE1 (the H on CPH2 in the HSD tautomer; bonded to CE1).
    HR1,
    /// Histidine ring H on CPH1 in the neutral tautomer (HD2).
    HR3,
    /// Cysteine thiol H.
    HS,
}

impl AtomType {
    pub const fn element(self) -> Element {
        use AtomType::*;
        match self {
            C | CA | CAI | CC | CT1 | CT2 | CT2A | CT3 | CP1 | CP2 | CP3
            | CPH1 | CPH2 | CPT | CY => Element::C,
            N | NH1 | NH2 | NH3 | NC2 | NR1 | NR2 | NY => Element::N,
            O | OC | OH1 => Element::O,
            S => Element::S,
            H | HA | HA1 | HA2 | HA3 | HB1 | HB2 | HC | HP | HR1 | HR3 | HS => Element::H,
        }
    }

    /// Reverse of [`charmm_name`]. Returns `None` for atom types we don't
    /// model (e.g. CHARMM's NP for N-terminal proline, OS for ester O,
    /// CS / SS for thiolate, SM for disulfide).
    pub fn from_charmm_name(name: &str) -> Option<Self> {
        use AtomType::*;
        Some(match name {
            "C" => C, "CA" => CA, "CAI" => CAI, "CC" => CC,
            "CT1" => CT1, "CT2" => CT2, "CT2A" => CT2A, "CT3" => CT3,
            "CP1" => CP1, "CP2" => CP2, "CP3" => CP3,
            "CPH1" => CPH1, "CPH2" => CPH2, "CPT" => CPT, "CY" => CY,
            "N" => N, "NH1" => NH1, "NH2" => NH2, "NH3" => NH3,
            "NC2" => NC2, "NR1" => NR1, "NR2" => NR2, "NY" => NY,
            "O" => O, "OC" => OC, "OH1" => OH1,
            "S" => S,
            "H" => H, "HA" => HA, "HA1" => HA1, "HA2" => HA2, "HA3" => HA3,
            "HB1" => HB1, "HB2" => HB2, "HC" => HC, "HP" => HP,
            "HR1" => HR1, "HR3" => HR3, "HS" => HS,
            _ => return None,
        })
    }

    /// CHARMM force-field name (matches par_all36m_prot.prm exactly so we
    /// can index into vendored parameter tables without translation).
    pub const fn charmm_name(self) -> &'static str {
        use AtomType::*;
        match self {
            C => "C", CA => "CA", CAI => "CAI", CC => "CC",
            CT1 => "CT1", CT2 => "CT2", CT2A => "CT2A", CT3 => "CT3",
            CP1 => "CP1", CP2 => "CP2", CP3 => "CP3",
            CPH1 => "CPH1", CPH2 => "CPH2", CPT => "CPT", CY => "CY",
            N => "N", NH1 => "NH1", NH2 => "NH2", NH3 => "NH3",
            NC2 => "NC2", NR1 => "NR1", NR2 => "NR2", NY => "NY",
            O => "O", OC => "OC", OH1 => "OH1",
            S => "S",
            H => "H", HA => "HA", HA1 => "HA1", HA2 => "HA2", HA3 => "HA3",
            HB1 => "HB1", HB2 => "HB2", HC => "HC", HP => "HP",
            HR1 => "HR1", HR3 => "HR3", HS => "HS",
        }
    }
}

/// Classify an atom by its (residue, atom-name). Returns `None` for atoms
/// that aren't part of the residue's modelled atom set.
pub fn classify(aa: AminoAcid, atom_name: &str) -> Option<AtomType> {
    use AminoAcid::*;
    use AtomType::*;

    // Backbone first — uniform across residues except Pro and Gly.
    match (aa, atom_name) {
        // Proline backbone: no H, special types.
        (Pro, "N") => return Some(N),
        (Pro, "CA") => return Some(CP1),
        (Pro, "C") => return Some(C),
        (Pro, "O") => return Some(O),
        (Pro, "HA") => return Some(HB1),
        // Glycine backbone: CT2 + two HB2 hydrogens.
        (Gly, "N") => return Some(NH1),
        (Gly, "CA") => return Some(CT2),
        (Gly, "C") => return Some(C),
        (Gly, "O") => return Some(O),
        (Gly, "H") => return Some(H),
        (Gly, "HA2" | "HA3") => return Some(HB2),
        // All other residues share standard backbone.
        (_, "N") => return Some(NH1),
        (_, "CA") => return Some(CT1),
        (_, "C") => return Some(C),
        (_, "O") => return Some(O),
        (_, "H") => return Some(H),
        (_, "HA") => return Some(HB1),
        _ => {}
    }

    // Side chains.
    let t = match (aa, atom_name) {
        (Gly, _) => return None, // Gly has no side chain.

        // Alanine — CB methyl
        (Ala, "CB") => CT3,
        (Ala, "HB1" | "HB2" | "HB3") => HA3,

        // Valine — CB(CH) → 2 methyls
        (Val, "CB") => CT1,
        (Val, "HB") => HA1,
        (Val, "CG1" | "CG2") => CT3,
        (Val, "HG11" | "HG12" | "HG13" | "HG21" | "HG22" | "HG23") => HA3,

        // Leucine — CB(CH₂) → CG(CH) → 2 methyls
        (Leu, "CB") => CT2,
        (Leu, "HB2" | "HB3") => HA2,
        (Leu, "CG") => CT1,
        (Leu, "HG") => HA1,
        (Leu, "CD1" | "CD2") => CT3,
        (Leu, "HD11" | "HD12" | "HD13" | "HD21" | "HD22" | "HD23") => HA3,

        // Isoleucine — CB(CH) branches to CG2 methyl + CG1(CH₂)
        (Ile, "CB") => CT1,
        (Ile, "HB") => HA1,
        (Ile, "CG2") => CT3,
        (Ile, "HG21" | "HG22" | "HG23") => HA3,
        (Ile, "CG1") => CT2,
        (Ile, "HG12" | "HG13") => HA2,
        (Ile, "CD1") => CT3,
        (Ile, "HD11" | "HD12" | "HD13") => HA3,

        // Methionine — CB-CG-SD-CE
        (Met, "CB" | "CG") => CT2,
        (Met, "HB2" | "HB3" | "HG2" | "HG3") => HA2,
        (Met, "SD") => S,
        (Met, "CE") => CT3,
        (Met, "HE1" | "HE2" | "HE3") => HA3,

        // Proline side-chain (ring continues from N-CA)
        (Pro, "CB" | "CG") => CP2,
        (Pro, "CD") => CP3,
        (Pro, "HB2" | "HB3" | "HG2" | "HG3" | "HD2" | "HD3") => HA2,

        // Serine
        (Ser, "CB") => CT2,
        (Ser, "HB2" | "HB3") => HA2,
        (Ser, "OG") => OH1,
        (Ser, "HG") => H,

        // Threonine
        (Thr, "CB") => CT1,
        (Thr, "HB") => HA1,
        (Thr, "OG1") => OH1,
        (Thr, "HG1") => H,
        (Thr, "CG2") => CT3,
        (Thr, "HG21" | "HG22" | "HG23") => HA3,

        // Cysteine
        (Cys, "CB") => CT2,
        (Cys, "HB2" | "HB3") => HA2,
        (Cys, "SG") => S,
        (Cys, "HG") => HS,

        // Asparagine — CB(CH₂) → CG(CC=O) - ND2(H,H)
        (Asn, "CB") => CT2,
        (Asn, "HB2" | "HB3") => HA2,
        (Asn, "CG") => CC,
        (Asn, "OD1") => O,
        (Asn, "ND2") => NH2,
        (Asn, "HD21" | "HD22") => H,

        // Glutamine — CB(CT2) → CG(CT2) → CD(CC=O) - NE2(H,H)
        (Gln, "CB") => CT2,
        (Gln, "HB2" | "HB3") => HA2,
        (Gln, "CG") => CT2,
        (Gln, "HG2" | "HG3") => HA2,
        (Gln, "CD") => CC,
        (Gln, "OE1") => O,
        (Gln, "NE2") => NH2,
        (Gln, "HE21" | "HE22") => H,

        // Aspartate
        (Asp, "CB") => CT2A,
        (Asp, "HB2" | "HB3") => HA2,
        (Asp, "CG") => CC,
        (Asp, "OD1" | "OD2") => OC,

        // Glutamate — CB(CT2A) → CG(CT2) → CD(CC=O⁻)
        (Glu, "CB") => CT2A,
        (Glu, "HB2" | "HB3") => HA2,
        (Glu, "CG") => CT2,
        (Glu, "HG2" | "HG3") => HA2,
        (Glu, "CD") => CC,
        (Glu, "OE1" | "OE2") => OC,

        // Lysine
        (Lys, "CB" | "CG" | "CD" | "CE") => CT2,
        (Lys, "HB2" | "HB3" | "HG2" | "HG3"
            | "HD2" | "HD3" | "HE2" | "HE3") => HA2,
        (Lys, "NZ") => NH3,
        (Lys, "HZ1" | "HZ2" | "HZ3") => HC,

        // Arginine
        (Arg, "CB" | "CG" | "CD") => CT2,
        (Arg, "HB2" | "HB3" | "HG2" | "HG3" | "HD2" | "HD3") => HA2,
        (Arg, "NE") => NC2,
        (Arg, "HE") => HC,
        (Arg, "CZ") => C, // guanidinium centre — CHARMM uses C for sp² trigonal
        (Arg, "NH1" | "NH2") => NC2,
        (Arg, "HH11" | "HH12" | "HH21" | "HH22") => HC,

        // Histidine — HSD tautomer (HD1 on ND1, NE2 unprotonated)
        (His, "CB") => CT2,
        (His, "HB2" | "HB3") => HA2,
        (His, "CG") => CPH1,
        (His, "ND1") => NR1,
        (His, "HD1") => H,
        (His, "CE1") => CPH2,
        (His, "HE1") => HR1,
        (His, "NE2") => NR2,
        (His, "CD2") => CPH1,
        (His, "HD2") => HR3,

        // Phenylalanine
        (Phe, "CB") => CT2,
        (Phe, "HB2" | "HB3") => HA2,
        (Phe, "CG" | "CD1" | "CD2" | "CE1" | "CE2" | "CZ") => CA,
        (Phe, "HD1" | "HD2" | "HE1" | "HE2" | "HZ") => HP,

        // Tyrosine
        (Tyr, "CB") => CT2,
        (Tyr, "HB2" | "HB3") => HA2,
        (Tyr, "CG" | "CD1" | "CD2" | "CE1" | "CE2" | "CZ") => CA,
        (Tyr, "HD1" | "HD2" | "HE1" | "HE2") => HP,
        (Tyr, "OH") => OH1,
        (Tyr, "HH") => H,

        // Tryptophan
        (Trp, "CB") => CT2,
        (Trp, "HB2" | "HB3") => HA2,
        (Trp, "CG") => CY,
        (Trp, "CD1") => CA, // pyrrole 5-ring CD1 is "CA" type in CHARMM, not CY
        (Trp, "HD1") => HP,
        (Trp, "NE1") => NY,
        (Trp, "HE1") => H,
        (Trp, "CE2") => CPT,
        (Trp, "CD2") => CPT,
        (Trp, "CE3" | "CZ2") => CAI, // adjacent to bridgehead
        (Trp, "CZ3" | "CH2") => CA,
        (Trp, "HE3" | "HZ2" | "HZ3" | "HH2") => HP,

        _ => return None,
    };
    Some(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backbone_classification() {
        assert_eq!(classify(AminoAcid::Ala, "N"), Some(AtomType::NH1));
        assert_eq!(classify(AminoAcid::Pro, "N"), Some(AtomType::N));
        assert_eq!(classify(AminoAcid::Ala, "CA"), Some(AtomType::CT1));
        assert_eq!(classify(AminoAcid::Gly, "CA"), Some(AtomType::CT2));
        assert_eq!(classify(AminoAcid::Pro, "CA"), Some(AtomType::CP1));
        assert_eq!(classify(AminoAcid::Ala, "C"), Some(AtomType::C));
        assert_eq!(classify(AminoAcid::Ala, "O"), Some(AtomType::O));
        assert_eq!(classify(AminoAcid::Ala, "H"), Some(AtomType::H));
        assert_eq!(classify(AminoAcid::Ala, "HA"), Some(AtomType::HB1));
        assert_eq!(classify(AminoAcid::Gly, "HA2"), Some(AtomType::HB2));
        assert_eq!(classify(AminoAcid::Pro, "HA"), Some(AtomType::HB1));
    }

    #[test]
    fn aliphatic_distinguishes_h_count() {
        // CB methyl (CT3) of Ala — its H's are HA3 type.
        assert_eq!(classify(AminoAcid::Ala, "CB"), Some(AtomType::CT3));
        assert_eq!(classify(AminoAcid::Ala, "HB1"), Some(AtomType::HA3));
        // CB single-H of Val.
        assert_eq!(classify(AminoAcid::Val, "CB"), Some(AtomType::CT1));
        assert_eq!(classify(AminoAcid::Val, "HB"), Some(AtomType::HA1));
        // CB CH2 of Leu.
        assert_eq!(classify(AminoAcid::Leu, "CB"), Some(AtomType::CT2));
        assert_eq!(classify(AminoAcid::Leu, "HB2"), Some(AtomType::HA2));
    }

    #[test]
    fn polar_centers_distinguish_ct2a() {
        // CHARMM36 .rtf assignments:
        // Asp CB = CT2A; Asn CB = CT2; Glu CB = CT2A; Gln CB = CT2; His CB = CT2.
        assert_eq!(classify(AminoAcid::Asp, "CB"), Some(AtomType::CT2A));
        assert_eq!(classify(AminoAcid::Glu, "CB"), Some(AtomType::CT2A));
        assert_eq!(classify(AminoAcid::Asn, "CB"), Some(AtomType::CT2));
        assert_eq!(classify(AminoAcid::Gln, "CB"), Some(AtomType::CT2));
        assert_eq!(classify(AminoAcid::His, "CB"), Some(AtomType::CT2));
    }

    #[test]
    fn aromatic_rings() {
        // Phe ring all CA, all H HP.
        for ring in ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"] {
            assert_eq!(classify(AminoAcid::Phe, ring), Some(AtomType::CA));
        }
        for h in ["HD1", "HD2", "HE1", "HE2", "HZ"] {
            assert_eq!(classify(AminoAcid::Phe, h), Some(AtomType::HP));
        }
        // Trp pyrrole side gets CY/CA/NY/CPT, 6-ring CE3/CZ2 are CAI, CZ3/CH2 are CA.
        assert_eq!(classify(AminoAcid::Trp, "CG"), Some(AtomType::CY));
        assert_eq!(classify(AminoAcid::Trp, "CD1"), Some(AtomType::CA));
        assert_eq!(classify(AminoAcid::Trp, "NE1"), Some(AtomType::NY));
        assert_eq!(classify(AminoAcid::Trp, "CE2"), Some(AtomType::CPT));
        assert_eq!(classify(AminoAcid::Trp, "CD2"), Some(AtomType::CPT));
        assert_eq!(classify(AminoAcid::Trp, "CE3"), Some(AtomType::CAI));
        assert_eq!(classify(AminoAcid::Trp, "CZ2"), Some(AtomType::CAI));
        assert_eq!(classify(AminoAcid::Trp, "CZ3"), Some(AtomType::CA));
        assert_eq!(classify(AminoAcid::Trp, "CH2"), Some(AtomType::CA));
    }

    #[test]
    fn histidine_hsd_tautomer() {
        assert_eq!(classify(AminoAcid::His, "ND1"), Some(AtomType::NR1));
        assert_eq!(classify(AminoAcid::His, "HD1"), Some(AtomType::H));
        assert_eq!(classify(AminoAcid::His, "NE2"), Some(AtomType::NR2));
        assert_eq!(classify(AminoAcid::His, "CE1"), Some(AtomType::CPH2));
        assert_eq!(classify(AminoAcid::His, "HE1"), Some(AtomType::HR1));
        assert_eq!(classify(AminoAcid::His, "CG"), Some(AtomType::CPH1));
    }

    #[test]
    fn charged_groups() {
        assert_eq!(classify(AminoAcid::Lys, "NZ"), Some(AtomType::NH3));
        assert_eq!(classify(AminoAcid::Lys, "HZ1"), Some(AtomType::HC));
        assert_eq!(classify(AminoAcid::Arg, "CZ"), Some(AtomType::C));
        assert_eq!(classify(AminoAcid::Arg, "NE"), Some(AtomType::NC2));
        assert_eq!(classify(AminoAcid::Arg, "HH11"), Some(AtomType::HC));
        assert_eq!(classify(AminoAcid::Asp, "OD1"), Some(AtomType::OC));
    }

    #[test]
    fn proline_special() {
        assert_eq!(classify(AminoAcid::Pro, "CD"), Some(AtomType::CP3));
        assert_eq!(classify(AminoAcid::Pro, "CB"), Some(AtomType::CP2));
        assert_eq!(classify(AminoAcid::Pro, "CG"), Some(AtomType::CP2));
        // Proline side-chain Hs use HA2 in CHARMM36 (matches the .prm bond table).
        assert_eq!(classify(AminoAcid::Pro, "HD2"), Some(AtomType::HA2));
        assert_eq!(classify(AminoAcid::Pro, "HB2"), Some(AtomType::HA2));
    }

    #[test]
    fn unknown_returns_none() {
        assert_eq!(classify(AminoAcid::Ala, "XX"), None);
        assert_eq!(classify(AminoAcid::Gly, "CB"), None);
    }

    #[test]
    fn element_consistency() {
        for aa in AminoAcid::ALL {
            for sc in aa.topology().sidechain {
                let t = classify(aa, sc.name).unwrap_or_else(|| {
                    panic!("missing classification for {:?} {}", aa, sc.name)
                });
                assert_eq!(t.element(), sc.element,
                    "{:?} {}: classified as {:?} (element {:?}) but topology element is {:?}",
                    aa, sc.name, t, t.element(), sc.element);
            }
        }
    }

    #[test]
    fn charmm_names_are_unique() {
        // Quick sanity: every AtomType maps to a distinct CHARMM name.
        let all = [
            AtomType::C, AtomType::CA, AtomType::CAI, AtomType::CC,
            AtomType::CT1, AtomType::CT2, AtomType::CT2A, AtomType::CT3,
            AtomType::CP1, AtomType::CP2, AtomType::CP3,
            AtomType::CPH1, AtomType::CPH2, AtomType::CPT, AtomType::CY,
            AtomType::N, AtomType::NH1, AtomType::NH2, AtomType::NH3,
            AtomType::NC2, AtomType::NR1, AtomType::NR2, AtomType::NY,
            AtomType::O, AtomType::OC, AtomType::OH1,
            AtomType::S,
            AtomType::H, AtomType::HA, AtomType::HA1, AtomType::HA2, AtomType::HA3,
            AtomType::HB1, AtomType::HB2, AtomType::HC, AtomType::HP,
            AtomType::HR1, AtomType::HR3, AtomType::HS,
        ];
        let mut names: Vec<&str> = all.iter().map(|t| t.charmm_name()).collect();
        names.sort();
        let mut deduped = names.clone();
        deduped.dedup();
        assert_eq!(names, deduped);
    }
}
