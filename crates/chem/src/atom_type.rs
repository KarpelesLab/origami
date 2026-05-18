//! CHARMM-granularity atom types for proteins and RNA.
//!
//! Protein types follow the standard CHARMM36 protein parameter file naming
//! (par_all36m_prot.prm / top_all36_prot.rtf). RNA types follow the CHARMM27
//! all-hydrogen nucleic acid file (par_all27_na.prm / top_all27_na.rtf,
//! Foloppe & MacKerell 2000).  Going granular keeps us faithful to the
//! published parameters: each (atom-type-pair) bond constant, etc., comes
//! straight from CHARMM without averaging across types. We only include the
//! types our 20 amino acids (physiological-pH protonation states; no
//! terminal patches, no protonated Asp/Glu, no charged His) and 4 RNA
//! ribonucleotides actually use.
//!
//! Histidine: we model only the HSD (HD1) tautomer — proton on the δ
//! nitrogen, neutral overall.

use crate::amino_acid::AminoAcid;
use crate::element::Element;
use crate::nucleotide::Nucleotide;

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

    // ---- Nucleic-acid hydrogens ----
    /// Exocyclic amine H (cytosine N4, adenine N6, guanine N2).
    Hn1,
    /// Aromatic ring-N H — H on cytosine/uracil N3 / guanine N1.
    Hn2,
    /// Aromatic ring-C H — H on C2/C8 of purines, H5/H6 of pyrimidines.
    Hn3,
    /// Phosphate hydroxyl H (terminal — not used in canonical RNA backbone
    /// here but kept for parser coverage).
    Hn4,
    /// Ribose 2'-hydroxyl H (HO2').
    Hn5,
    /// Sugar CH H — H1', H2', H3', H4', H5', H5''.
    Hn7,
    /// Sugar CH₂ H on C5' (HN8 in CHARMM27).
    Hn8,
    /// CH₃ proton (methylated base methyl H — kept for parser coverage).
    Hn9,

    // ============================================================
    // CHARMM27 nucleic-acid atom types (par_all27_na.prm).
    // Naming follows the .prm exactly so we can index the table
    // without translation.  Comments give the chemical context
    // taken straight from the CHARMM27 MASS section.
    // ============================================================

    // ---- Nucleic-acid carbons ----
    /// Adenine/guanine C6 — sp² aromatic, exocyclic substituent attaches.
    Cn1,
    /// Thymine C2 carbonyl.
    Cn1t,
    /// Cytosine C2 carbonyl / guanine C2.
    Cn2,
    /// Cytosine C5 / uracil C5 (sp², bears H5).
    Cn3,
    /// Adenine C2 / adenine C8 / guanine C8 — sp² aromatic ring C bearing H.
    Cn4,
    /// Adenine C4/C5 bridgehead — purine fused-ring shared edge.
    Cn5,
    /// Guanine C4/C5 bridgehead carbons.
    Cn5g,
    /// Ribose C3'/C4' (sp³ CH).
    Cn7,
    /// Ribose C1'/C2' (sp³ CH bonded to a heteroatom).
    Cn7b,
    /// Deoxyribose C2' / ribose C5' (sp³ CH₂).
    Cn8,
    /// Ribose C5' specifically — sp³ CH₂ bonded to O5'.
    Cn8b,
    /// Methylene methyl carbon (e.g. methylated bases — kept for completeness).
    Cn9,

    // ---- Nucleic-acid nitrogens ----
    /// Nucleic-acid amine (NH₂ exocyclic on cytosine N4, adenine N6,
    /// guanine N2).
    Nn1,
    /// Aromatic N bonded to sugar (N9 of purines, N1 of pyrimidines).
    Nn2,
    /// Aromatic N (cytosine/uracil N3) — bonded to H in U/T, lone-pair in C.
    Nn2b,
    /// Guanine N1 — bonded to H.
    Nn2g,
    /// Uracil/thymine N3 — bonded to H.
    Nn2u,
    /// Aromatic N with lone pair (no H) — pyrimidine ring acceptor.
    Nn3,
    /// Adenine N1/N3/N7 — protonated lone-pair-accepting ring N.
    Nn3a,
    /// Guanine N3.
    Nn3g,
    /// Purine N7 — H-bond acceptor.
    Nn4,

    // ---- Nucleic-acid oxygens ----
    /// Carbonyl O on uracil/thymine.
    On1,
    /// Cytosine carbonyl O.
    On1c,
    /// Backbone O5'/O3' — bonded to phosphate.
    On2,
    /// Anionic phosphate O (OP1, OP2).
    On3,
    /// Phosphate-O5' methyl-style (not used by canonical RNA backbone but
    /// kept for completeness).
    On4,
    /// Ribose 2'-hydroxyl O.
    On5,
    /// Furanose ring O (O4').
    On6,
    /// Furanose ring O bonded to a heteroatom-bearing C1' — RNA O4'.
    On6b,

    // ---- Phosphorus ----
    /// Phosphate P.
    Pn,
}

impl AtomType {
    pub const fn element(self) -> Element {
        use AtomType::*;
        match self {
            // Protein carbons
            C | CA | CAI | CC | CT1 | CT2 | CT2A | CT3 | CP1 | CP2 | CP3
            | CPH1 | CPH2 | CPT | CY => Element::C,
            // Protein nitrogens
            N | NH1 | NH2 | NH3 | NC2 | NR1 | NR2 | NY => Element::N,
            // Protein oxygens
            O | OC | OH1 => Element::O,
            // Protein sulfur
            S => Element::S,
            // Protein hydrogens
            H | HA | HA1 | HA2 | HA3 | HB1 | HB2 | HC | HP | HR1 | HR3 | HS => Element::H,

            // RNA carbons
            Cn1 | Cn1t | Cn2 | Cn3 | Cn4 | Cn5 | Cn5g | Cn7 | Cn7b | Cn8 | Cn8b | Cn9 => Element::C,
            // RNA nitrogens
            Nn1 | Nn2 | Nn2b | Nn2g | Nn2u | Nn3 | Nn3a | Nn3g | Nn4 => Element::N,
            // RNA oxygens
            On1 | On1c | On2 | On3 | On4 | On5 | On6 | On6b => Element::O,
            // RNA phosphorus
            Pn => Element::P,
            // RNA hydrogens
            Hn1 | Hn2 | Hn3 | Hn4 | Hn5 | Hn7 | Hn8 | Hn9 => Element::H,
        }
    }

    /// Reverse of [`charmm_name`]. Returns `None` for atom types we don't
    /// model (e.g. CHARMM's NP for N-terminal proline, OS for ester O,
    /// CS / SS for thiolate, SM for disulfide).
    pub fn from_charmm_name(name: &str) -> Option<Self> {
        use AtomType::*;
        Some(match name {
            // Protein types
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
            // RNA carbons
            "CN1" => Cn1, "CN1T" => Cn1t, "CN2" => Cn2, "CN3" => Cn3, "CN4" => Cn4,
            "CN5" => Cn5, "CN5G" => Cn5g, "CN7" => Cn7, "CN7B" => Cn7b,
            "CN8" => Cn8, "CN8B" => Cn8b, "CN9" => Cn9,
            // RNA nitrogens
            "NN1" => Nn1, "NN2" => Nn2, "NN2B" => Nn2b, "NN2G" => Nn2g, "NN2U" => Nn2u,
            "NN3" => Nn3, "NN3A" => Nn3a, "NN3G" => Nn3g, "NN4" => Nn4,
            // RNA oxygens
            "ON1" => On1, "ON1C" => On1c, "ON2" => On2, "ON3" => On3, "ON4" => On4,
            "ON5" => On5, "ON6" => On6, "ON6B" => On6b,
            // RNA phosphorus
            "P" => Pn,
            // RNA hydrogens
            "HN1" => Hn1, "HN2" => Hn2, "HN3" => Hn3, "HN4" => Hn4,
            "HN5" => Hn5, "HN7" => Hn7, "HN8" => Hn8, "HN9" => Hn9,
            _ => return None,
        })
    }

    /// CHARMM force-field name (matches par_all36m_prot.prm exactly so we
    /// can index into vendored parameter tables without translation).
    pub const fn charmm_name(self) -> &'static str {
        use AtomType::*;
        match self {
            // Protein
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
            // RNA
            Cn1 => "CN1", Cn1t => "CN1T", Cn2 => "CN2", Cn3 => "CN3",
            Cn4 => "CN4", Cn5 => "CN5", Cn5g => "CN5G",
            Cn7 => "CN7", Cn7b => "CN7B", Cn8 => "CN8", Cn8b => "CN8B", Cn9 => "CN9",
            Nn1 => "NN1", Nn2 => "NN2", Nn2b => "NN2B", Nn2g => "NN2G", Nn2u => "NN2U",
            Nn3 => "NN3", Nn3a => "NN3A", Nn3g => "NN3G", Nn4 => "NN4",
            On1 => "ON1", On1c => "ON1C", On2 => "ON2", On3 => "ON3", On4 => "ON4",
            On5 => "ON5", On6 => "ON6", On6b => "ON6B",
            Pn => "P",
            Hn1 => "HN1", Hn2 => "HN2", Hn3 => "HN3", Hn4 => "HN4",
            Hn5 => "HN5", Hn7 => "HN7", Hn8 => "HN8", Hn9 => "HN9",
        }
    }
}

/// Classify an RNA atom by its (nucleotide, atom-name) into the CHARMM27
/// nucleic-acid atom type.  Atom names are in PDB v3.3 form (matching the
/// chain builder); the CHARMM-vs-PDB name differences (`O1P`/`OP1`,
/// `H2'`/`HO2'`, `H2''`/`H2'`) are translated inside this function.
pub fn classify_rna(nt: Nucleotide, atom_name: &str) -> Option<AtomType> {
    use AtomType::*;
    use Nucleotide::*;

    // Backbone first — sugar + phosphate + ribose hydrogens are
    // identical across all four canonical nucleotides.
    match atom_name {
        "P" => return Some(Pn),
        "OP1" | "OP2" => return Some(On3),
        "O5'" => return Some(On2),
        "C5'" => return Some(Cn8b),
        "H5'" | "H5''" => return Some(Hn8),
        "C4'" => return Some(Cn7),
        "H4'" => return Some(Hn7),
        "O4'" => return Some(On6b),
        "C3'" => return Some(Cn7),
        "H3'" => return Some(Hn7),
        "O3'" => return Some(On2),
        "C2'" => return Some(Cn7b),
        "H2'" => return Some(Hn7),   // PDB v3.3 H2' = CHARMM H2''
        "O2'" => return Some(On5),
        "HO2'" => return Some(Hn5),  // PDB v3.3 HO2' = CHARMM H2'
        "C1'" => return Some(Cn7b),
        "H1'" => return Some(Hn7),
        _ => {}
    }

    // Base atoms — per-nucleotide.
    let t = match (nt, atom_name) {
        // ---- Adenine ----
        (Adenine, "N9") => Nn2,
        (Adenine, "C8") => Cn4,
        (Adenine, "H8") => Hn3,
        (Adenine, "N7") => Nn4,
        (Adenine, "C5") => Cn5,
        (Adenine, "C6") => Cn2,
        (Adenine, "N6") => Nn1,
        (Adenine, "H61" | "H62") => Hn1,
        (Adenine, "N1") => Nn3a,
        (Adenine, "C2") => Cn4,
        (Adenine, "H2") => Hn3,
        (Adenine, "N3") => Nn3a,
        (Adenine, "C4") => Cn5,

        // ---- Guanine ----
        (Guanine, "N9") => Nn2b,
        (Guanine, "C8") => Cn4,
        (Guanine, "H8") => Hn3,
        (Guanine, "N7") => Nn4,
        (Guanine, "C5") => Cn5g,
        (Guanine, "C6") => Cn1,
        (Guanine, "O6") => On1,
        (Guanine, "N1") => Nn2g,
        (Guanine, "H1") => Hn2,
        (Guanine, "C2") => Cn2,
        (Guanine, "N2") => Nn1,
        (Guanine, "H21" | "H22") => Hn1,
        (Guanine, "N3") => Nn3g,
        (Guanine, "C4") => Cn5,

        // ---- Cytosine ----
        (Cytosine, "N1") => Nn2,
        (Cytosine, "C2") => Cn1,
        (Cytosine, "O2") => On1c,
        (Cytosine, "N3") => Nn3,
        (Cytosine, "C4") => Cn2,
        (Cytosine, "N4") => Nn1,
        (Cytosine, "H41" | "H42") => Hn1,
        (Cytosine, "C5") => Cn3,
        (Cytosine, "H5") => Hn3,
        (Cytosine, "C6") => Cn3,
        (Cytosine, "H6") => Hn3,

        // ---- Uracil ----
        (Uracil, "N1") => Nn2b,
        (Uracil, "C2") => Cn1t,
        (Uracil, "O2") => On1,
        (Uracil, "N3") => Nn2u,
        (Uracil, "H3") => Hn2,
        (Uracil, "C4") => Cn1,
        (Uracil, "O4") => On1,
        (Uracil, "C5") => Cn3,
        (Uracil, "H5") => Hn3,
        (Uracil, "C6") => Cn3,
        (Uracil, "H6") => Hn3,

        _ => return None,
    };
    Some(t)
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
    fn rna_backbone_classification() {
        for nt in [Nucleotide::Adenine, Nucleotide::Uracil,
                   Nucleotide::Guanine, Nucleotide::Cytosine] {
            assert_eq!(classify_rna(nt, "P"), Some(AtomType::Pn));
            assert_eq!(classify_rna(nt, "OP1"), Some(AtomType::On3));
            assert_eq!(classify_rna(nt, "OP2"), Some(AtomType::On3));
            assert_eq!(classify_rna(nt, "O5'"), Some(AtomType::On2));
            assert_eq!(classify_rna(nt, "C5'"), Some(AtomType::Cn8b));
            assert_eq!(classify_rna(nt, "H5'"), Some(AtomType::Hn8));
            assert_eq!(classify_rna(nt, "H5''"), Some(AtomType::Hn8));
            assert_eq!(classify_rna(nt, "C4'"), Some(AtomType::Cn7));
            assert_eq!(classify_rna(nt, "O4'"), Some(AtomType::On6b));
            assert_eq!(classify_rna(nt, "C1'"), Some(AtomType::Cn7b));
            // PDB v3.3 vs CHARMM naming for the 2'-position.
            assert_eq!(classify_rna(nt, "H2'"), Some(AtomType::Hn7));   // on C2'
            assert_eq!(classify_rna(nt, "HO2'"), Some(AtomType::Hn5));  // on O2'
            assert_eq!(classify_rna(nt, "O2'"), Some(AtomType::On5));
        }
    }

    #[test]
    fn rna_glycosidic_n_distinguishes_purines_from_pyrimidines() {
        // N9 (glycosidic on purines) gets NN2 (A) / NN2B (G).
        assert_eq!(classify_rna(Nucleotide::Adenine, "N9"), Some(AtomType::Nn2));
        assert_eq!(classify_rna(Nucleotide::Guanine, "N9"), Some(AtomType::Nn2b));
        // N1 (glycosidic on pyrimidines) gets NN2 (C) / NN2B (U).
        assert_eq!(classify_rna(Nucleotide::Cytosine, "N1"), Some(AtomType::Nn2));
        assert_eq!(classify_rna(Nucleotide::Uracil, "N1"), Some(AtomType::Nn2b));
    }

    #[test]
    fn rna_carbonyl_oxygens_distinct_types() {
        // Cytosine O2 is "ON1C" (cytosine-specific) — different from
        // the uracil/guanine carbonyl O which is "ON1".
        assert_eq!(classify_rna(Nucleotide::Cytosine, "O2"), Some(AtomType::On1c));
        assert_eq!(classify_rna(Nucleotide::Uracil, "O2"), Some(AtomType::On1));
        assert_eq!(classify_rna(Nucleotide::Uracil, "O4"), Some(AtomType::On1));
        assert_eq!(classify_rna(Nucleotide::Guanine, "O6"), Some(AtomType::On1));
    }

    #[test]
    fn rna_full_roster_classified() {
        // Every atom in `Nucleotide::all_atoms()` should classify.
        for nt in [Nucleotide::Adenine, Nucleotide::Uracil,
                   Nucleotide::Guanine, Nucleotide::Cytosine] {
            for (name, _) in nt.all_atoms() {
                assert!(
                    classify_rna(nt, name).is_some(),
                    "{:?} atom {} did not classify",
                    nt, name,
                );
            }
        }
    }

    #[test]
    fn rna_element_consistency() {
        for nt in [Nucleotide::Adenine, Nucleotide::Uracil,
                   Nucleotide::Guanine, Nucleotide::Cytosine] {
            for (name, expected_el) in nt.all_atoms() {
                let t = classify_rna(nt, name).unwrap();
                assert_eq!(t.element(), expected_el,
                    "{:?} {}: classified {:?} (element {:?}) vs roster element {:?}",
                    nt, name, t, t.element(), expected_el);
            }
        }
    }

    #[test]
    fn charmm_names_are_unique() {
        // Quick sanity: every AtomType maps to a distinct CHARMM name.
        let all = [
            // Protein
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
            // RNA
            AtomType::Cn1, AtomType::Cn1t, AtomType::Cn2, AtomType::Cn3,
            AtomType::Cn4, AtomType::Cn5, AtomType::Cn5g,
            AtomType::Cn7, AtomType::Cn7b, AtomType::Cn8, AtomType::Cn8b, AtomType::Cn9,
            AtomType::Nn1, AtomType::Nn2, AtomType::Nn2b, AtomType::Nn2g, AtomType::Nn2u,
            AtomType::Nn3, AtomType::Nn3a, AtomType::Nn3g, AtomType::Nn4,
            AtomType::On1, AtomType::On1c, AtomType::On2, AtomType::On3, AtomType::On4,
            AtomType::On5, AtomType::On6, AtomType::On6b,
            AtomType::Pn,
            AtomType::Hn1, AtomType::Hn2, AtomType::Hn3, AtomType::Hn4,
            AtomType::Hn5, AtomType::Hn7, AtomType::Hn8, AtomType::Hn9,
        ];
        let mut names: Vec<&str> = all.iter().map(|t| t.charmm_name()).collect();
        names.sort();
        let mut deduped = names.clone();
        deduped.dedup();
        assert_eq!(names, deduped);
    }
}
