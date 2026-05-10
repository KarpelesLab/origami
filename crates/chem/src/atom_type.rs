//! CHARMM-style atom types.
//!
//! Each atom in a built structure is classified into one of these types,
//! which then key into the force-field parameter tables (LJ σ/ε, partial
//! charges, bond/angle/dihedral constants).
//!
//! The naming follows CHARMM36 conventions where reasonable; we collapse
//! some near-duplicate types (e.g., several CHARMM aromatic-C types) where
//! the chemistry doesn't warrant the distinction in our simplified scheme.

use crate::amino_acid::AminoAcid;
use crate::element::Element;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AtomType {
    // ---- Carbons ----
    /// sp³ aliphatic carbon (Cα, Cβ, Cγ in saturated residues)
    Caliph,
    /// sp² aromatic ring carbon (Phe / Tyr / Trp / His ring members)
    Carom,
    /// sp² carbonyl/carboxyl/amide carbon (backbone C, Asn Cγ, Gln Cδ, Asp Cγ, Glu Cδ)
    Ccarb,
    /// Guanidinium central carbon (Arg Cζ)
    Cguan,

    // ---- Nitrogens ----
    /// Amide nitrogen (backbone N of non-Pro residues; Asn ND2; Gln NE2; Trp NE1)
    Namide,
    /// Aromatic ring nitrogen (His ND1/NE2)
    Naromatic,
    /// sp³ ammonium nitrogen (Lys Nζ, +1)
    Nammon,
    /// Guanidinium nitrogen (Arg NE/NH1/NH2; +1 charge delocalised across the three N's)
    Nguan,
    /// Proline backbone nitrogen — sp² (peptide) but no H attached
    Nproline,

    // ---- Oxygens ----
    /// Carbonyl/amide oxygen (backbone O; Asn OD1; Gln OE1)
    Ocarb,
    /// Carboxylate oxygen, -COO⁻ (Asp OD1/OD2; Glu OE1/OE2)
    Ocarboxyl,
    /// Hydroxyl oxygen, -OH (Ser OG; Thr OG1; Tyr OH)
    Ohydroxyl,

    // ---- Sulfurs ----
    /// Thioether sulfur (Met SD)
    Sthio,
    /// Thiol sulfur (Cys SG)
    Sthiol,

    // ---- Hydrogens ----
    /// Bonded to an aliphatic sp³ carbon
    Haliph,
    /// Bonded to an aromatic ring carbon
    Harom,
    /// Amide H (backbone H; Asn HD2x; Gln HE2x; Trp HE1)
    Hamide,
    /// Bonded to ammonium N (Lys HZx)
    Hammon,
    /// Bonded to guanidinium N (Arg HE; HH1x; HH2x)
    Hguan,
    /// Hydroxyl H (Ser HG; Thr HG1; Tyr HH)
    Hhydroxyl,
    /// Thiol H (Cys HG)
    Hthiol,
}

impl AtomType {
    pub const fn element(self) -> Element {
        use AtomType::*;
        match self {
            Caliph | Carom | Ccarb | Cguan => Element::C,
            Namide | Naromatic | Nammon | Nguan | Nproline => Element::N,
            Ocarb | Ocarboxyl | Ohydroxyl => Element::O,
            Sthio | Sthiol => Element::S,
            Haliph | Harom | Hamide | Hammon | Hguan | Hhydroxyl | Hthiol => Element::H,
        }
    }
}

/// Classify an atom by its (residue, atom-name). Returns `None` if the name
/// is not recognised for that residue.
pub fn classify(aa: AminoAcid, atom_name: &str) -> Option<AtomType> {
    use AminoAcid::*;
    use AtomType::*;

    // Backbone atoms — uniform across residues except Pro and Gly.
    match atom_name {
        "N" => return Some(if aa == Pro { Nproline } else { Namide }),
        "CA" => return Some(Caliph),
        "C" => return Some(Ccarb),
        "O" => return Some(Ocarb),
        "H" => return Some(Hamide),
        "HA" | "HA2" | "HA3" => return Some(Haliph),
        _ => {}
    }

    // Side-chain atoms.
    let t = match (aa, atom_name) {
        // Glycine has no side chain.
        (Gly, _) => return None,

        // Alanine
        (Ala, "CB") => Caliph,
        (Ala, "HB1" | "HB2" | "HB3") => Haliph,

        // Valine
        (Val, "CB" | "CG1" | "CG2") => Caliph,
        (Val, "HB" | "HG11" | "HG12" | "HG13" | "HG21" | "HG22" | "HG23") => Haliph,

        // Leucine
        (Leu, "CB" | "CG" | "CD1" | "CD2") => Caliph,
        (Leu, "HB2" | "HB3" | "HG"
            | "HD11" | "HD12" | "HD13"
            | "HD21" | "HD22" | "HD23") => Haliph,

        // Isoleucine
        (Ile, "CB" | "CG1" | "CG2" | "CD1") => Caliph,
        (Ile, "HB" | "HG12" | "HG13"
            | "HG21" | "HG22" | "HG23"
            | "HD11" | "HD12" | "HD13") => Haliph,

        // Methionine — Met SD is a thioether
        (Met, "CB" | "CG" | "CE") => Caliph,
        (Met, "SD") => Sthio,
        (Met, "HB2" | "HB3" | "HG2" | "HG3"
            | "HE1" | "HE2" | "HE3") => Haliph,

        // Proline
        (Pro, "CB" | "CG" | "CD") => Caliph,
        (Pro, "HB2" | "HB3" | "HG2" | "HG3" | "HD2" | "HD3") => Haliph,

        // Serine
        (Ser, "CB") => Caliph,
        (Ser, "OG") => Ohydroxyl,
        (Ser, "HG") => Hhydroxyl,
        (Ser, "HB2" | "HB3") => Haliph,

        // Threonine
        (Thr, "CB" | "CG2") => Caliph,
        (Thr, "OG1") => Ohydroxyl,
        (Thr, "HG1") => Hhydroxyl,
        (Thr, "HB" | "HG21" | "HG22" | "HG23") => Haliph,

        // Cysteine
        (Cys, "CB") => Caliph,
        (Cys, "SG") => Sthiol,
        (Cys, "HG") => Hthiol,
        (Cys, "HB2" | "HB3") => Haliph,

        // Asparagine
        (Asn, "CB") => Caliph,
        (Asn, "CG") => Ccarb,
        (Asn, "OD1") => Ocarb,
        (Asn, "ND2") => Namide,
        (Asn, "HB2" | "HB3") => Haliph,
        (Asn, "HD21" | "HD22") => Hamide,

        // Glutamine
        (Gln, "CB" | "CG") => Caliph,
        (Gln, "CD") => Ccarb,
        (Gln, "OE1") => Ocarb,
        (Gln, "NE2") => Namide,
        (Gln, "HB2" | "HB3" | "HG2" | "HG3") => Haliph,
        (Gln, "HE21" | "HE22") => Hamide,

        // Aspartate
        (Asp, "CB") => Caliph,
        (Asp, "CG") => Ccarb,
        (Asp, "OD1" | "OD2") => Ocarboxyl,
        (Asp, "HB2" | "HB3") => Haliph,

        // Glutamate
        (Glu, "CB" | "CG") => Caliph,
        (Glu, "CD") => Ccarb,
        (Glu, "OE1" | "OE2") => Ocarboxyl,
        (Glu, "HB2" | "HB3" | "HG2" | "HG3") => Haliph,

        // Lysine
        (Lys, "CB" | "CG" | "CD" | "CE") => Caliph,
        (Lys, "NZ") => Nammon,
        (Lys, "HB2" | "HB3" | "HG2" | "HG3"
            | "HD2" | "HD3" | "HE2" | "HE3") => Haliph,
        (Lys, "HZ1" | "HZ2" | "HZ3") => Hammon,

        // Arginine
        (Arg, "CB" | "CG" | "CD") => Caliph,
        (Arg, "NE") => Nguan,
        (Arg, "CZ") => Cguan,
        (Arg, "NH1" | "NH2") => Nguan,
        (Arg, "HB2" | "HB3" | "HG2" | "HG3" | "HD2" | "HD3") => Haliph,
        (Arg, "HE" | "HH11" | "HH12" | "HH21" | "HH22") => Hguan,

        // Histidine
        (His, "CB") => Caliph,
        (His, "CG" | "CD2" | "CE1") => Carom,
        (His, "ND1" | "NE2") => Naromatic,
        (His, "HB2" | "HB3") => Haliph,
        (His, "HD1") => Hamide, // imidazole NH (in HD1 tautomer)
        (His, "HD2" | "HE1") => Harom,

        // Phenylalanine
        (Phe, "CB") => Caliph,
        (Phe, "CG" | "CD1" | "CD2" | "CE1" | "CE2" | "CZ") => Carom,
        (Phe, "HB2" | "HB3") => Haliph,
        (Phe, "HD1" | "HD2" | "HE1" | "HE2" | "HZ") => Harom,

        // Tyrosine
        (Tyr, "CB") => Caliph,
        (Tyr, "CG" | "CD1" | "CD2" | "CE1" | "CE2" | "CZ") => Carom,
        (Tyr, "OH") => Ohydroxyl,
        (Tyr, "HH") => Hhydroxyl,
        (Tyr, "HB2" | "HB3") => Haliph,
        (Tyr, "HD1" | "HD2" | "HE1" | "HE2") => Harom,

        // Tryptophan
        (Trp, "CB") => Caliph,
        (Trp, "CG" | "CD1" | "CD2" | "CE2" | "CE3"
            | "CZ2" | "CZ3" | "CH2") => Carom,
        (Trp, "NE1") => Namide, // indole N-H is amide-like
        (Trp, "HB2" | "HB3") => Haliph,
        (Trp, "HE1") => Hamide,
        (Trp, "HD1" | "HE3" | "HZ2" | "HZ3" | "HH2") => Harom,

        _ => return None,
    };
    Some(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backbone_atoms_classify() {
        assert_eq!(classify(AminoAcid::Ala, "N"), Some(AtomType::Namide));
        assert_eq!(classify(AminoAcid::Pro, "N"), Some(AtomType::Nproline));
        assert_eq!(classify(AminoAcid::Ala, "CA"), Some(AtomType::Caliph));
        assert_eq!(classify(AminoAcid::Ala, "C"), Some(AtomType::Ccarb));
        assert_eq!(classify(AminoAcid::Ala, "O"), Some(AtomType::Ocarb));
        assert_eq!(classify(AminoAcid::Ala, "H"), Some(AtomType::Hamide));
        assert_eq!(classify(AminoAcid::Gly, "HA2"), Some(AtomType::Haliph));
        assert_eq!(classify(AminoAcid::Gly, "HA3"), Some(AtomType::Haliph));
    }

    #[test]
    fn aromatic_ring_atoms_classify() {
        assert_eq!(classify(AminoAcid::Phe, "CG"), Some(AtomType::Carom));
        assert_eq!(classify(AminoAcid::Phe, "HZ"), Some(AtomType::Harom));
        assert_eq!(classify(AminoAcid::Trp, "NE1"), Some(AtomType::Namide));
        assert_eq!(classify(AminoAcid::Trp, "HE1"), Some(AtomType::Hamide));
        assert_eq!(classify(AminoAcid::Trp, "CG"), Some(AtomType::Carom));
        assert_eq!(classify(AminoAcid::His, "ND1"), Some(AtomType::Naromatic));
    }

    #[test]
    fn charged_atoms_classify() {
        assert_eq!(classify(AminoAcid::Lys, "NZ"), Some(AtomType::Nammon));
        assert_eq!(classify(AminoAcid::Lys, "HZ1"), Some(AtomType::Hammon));
        assert_eq!(classify(AminoAcid::Arg, "CZ"), Some(AtomType::Cguan));
        assert_eq!(classify(AminoAcid::Arg, "NE"), Some(AtomType::Nguan));
        assert_eq!(classify(AminoAcid::Arg, "HH11"), Some(AtomType::Hguan));
        assert_eq!(classify(AminoAcid::Asp, "OD1"), Some(AtomType::Ocarboxyl));
        assert_eq!(classify(AminoAcid::Glu, "OE2"), Some(AtomType::Ocarboxyl));
    }

    #[test]
    fn polar_atoms_classify() {
        assert_eq!(classify(AminoAcid::Ser, "OG"), Some(AtomType::Ohydroxyl));
        assert_eq!(classify(AminoAcid::Ser, "HG"), Some(AtomType::Hhydroxyl));
        assert_eq!(classify(AminoAcid::Thr, "OG1"), Some(AtomType::Ohydroxyl));
        assert_eq!(classify(AminoAcid::Tyr, "OH"), Some(AtomType::Ohydroxyl));
        assert_eq!(classify(AminoAcid::Cys, "SG"), Some(AtomType::Sthiol));
        assert_eq!(classify(AminoAcid::Cys, "HG"), Some(AtomType::Hthiol));
        assert_eq!(classify(AminoAcid::Met, "SD"), Some(AtomType::Sthio));
    }

    #[test]
    fn unknown_atom_returns_none() {
        assert_eq!(classify(AminoAcid::Ala, "XX"), None);
        assert_eq!(classify(AminoAcid::Gly, "CB"), None); // Gly has no CB
    }

    #[test]
    fn element_consistency() {
        // Every atom type's element matches its name prefix convention.
        assert_eq!(AtomType::Caliph.element(), Element::C);
        assert_eq!(AtomType::Carom.element(), Element::C);
        assert_eq!(AtomType::Namide.element(), Element::N);
        assert_eq!(AtomType::Ohydroxyl.element(), Element::O);
        assert_eq!(AtomType::Sthio.element(), Element::S);
        assert_eq!(AtomType::Haliph.element(), Element::H);
    }
}
