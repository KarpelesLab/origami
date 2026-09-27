//! Codon-usage tables for codon-paced translation timing.
//!
//! Values are the **relative adaptiveness** w_i for each codon in
//! the highly-expressed E. coli K-12 gene set (Sharp & Li 1987,
//! *Nucleic Acids Res.* **15**, 1281).  Within each amino acid's
//! synonymous-codon family the most-used codon has `w_i = 1.0` and
//! the rest are scaled by their relative frequency.
//!
//! ## Why a slowdown floor of 0.2?
//!
//! Ribosome-profiling studies in *E. coli* show rare codons translate
//! roughly **3–5× slower** than common ones — not the 30× a naive
//! `1/w_i` would suggest for w ≈ 0.03 codons like AGG (Arg).  The slow
//! step is tRNA delivery, which depends on tRNA pool sizes and is
//! bounded below by the cell's slowest tRNA.  We model this by
//! clamping `w` at 0.2 before inverting:  `factor = 1.0 / max(w, 0.2)`.
//! Caps the per-codon slowdown at 5×, matching the empirical ceiling.

use crate::codon::{Base, Codon};

/// Multiplier applied to the per-residue base interval in
/// codon-paced cotranslation.  `1.0` = a maximally-common codon,
/// `5.0` = a rare codon capped at the empirical 5× slowdown.
pub fn ecoli_k12_rarity_factor(codon: Codon) -> f64 {
    let w = ecoli_k12_w(codon);
    1.0 / w.max(0.2)
}

/// Raw relative adaptiveness w_i (CAI's per-codon weight) for the
/// E. coli K-12 highly-expressed gene set.  Stop codons return 1.0
/// (they aren't translated to a residue; the caller never asks).
pub fn ecoli_k12_w(codon: Codon) -> f64 {
    use Base::*;
    let Codon([a, b, c]) = codon;
    match (a, b, c) {
        // Phe (F)
        (U, U, U) => 0.49,
        (U, U, C) => 1.00,
        // Leu (L) — CUG dominates dramatically in E. coli
        (U, U, A) => 0.13,
        (U, U, G) => 0.13,
        (C, U, U) => 0.10,
        (C, U, C) => 0.10,
        (C, U, A) => 0.04,
        (C, U, G) => 1.00,
        // Ile (I)
        (A, U, U) => 0.49,
        (A, U, C) => 1.00,
        (A, U, A) => 0.07, // rare
        // Met (M) — single codon
        (A, U, G) => 1.00,
        // Val (V)
        (G, U, U) => 0.66,
        (G, U, C) => 0.30,
        (G, U, A) => 0.30,
        (G, U, G) => 1.00,
        // Ser (S)
        (U, C, U) => 0.43,
        (U, C, C) => 0.55,
        (U, C, A) => 0.20,
        (U, C, G) => 0.55,
        (A, G, U) => 0.43,
        (A, G, C) => 1.00,
        // Pro (P)
        (C, C, U) => 0.18,
        (C, C, C) => 0.05, // rare
        (C, C, A) => 0.18,
        (C, C, G) => 1.00,
        // Thr (T)
        (A, C, U) => 0.40,
        (A, C, C) => 1.00,
        (A, C, A) => 0.15,
        (A, C, G) => 0.40,
        // Ala (A) — flatter distribution, GCU slightly favoured
        (G, C, U) => 1.00,
        (G, C, C) => 0.32,
        (G, C, A) => 0.42,
        (G, C, G) => 0.46,
        // Tyr (Y)
        (U, A, U) => 0.53,
        (U, A, C) => 1.00,
        // Stop — w=1.0 placeholder; cotranslation halts on stop.
        (U, A, A) | (U, A, G) | (U, G, A) => 1.00,
        // His (H)
        (C, A, U) => 0.65,
        (C, A, C) => 1.00,
        // Gln (Q)
        (C, A, A) => 0.55,
        (C, A, G) => 1.00,
        // Asn (N)
        (A, A, U) => 0.51,
        (A, A, C) => 1.00,
        // Lys (K)
        (A, A, A) => 1.00,
        (A, A, G) => 0.21,
        // Asp (D)
        (G, A, U) => 0.73,
        (G, A, C) => 1.00,
        // Glu (E)
        (G, A, A) => 1.00,
        (G, A, G) => 0.39,
        // Cys (C)
        (U, G, U) => 0.46,
        (U, G, C) => 1.00,
        // Trp (W) — single codon
        (U, G, G) => 1.00,
        // Arg (R) — AGA / AGG / CGA / CGG are infamously rare in E. coli
        (C, G, U) => 1.00,
        (C, G, C) => 0.92,
        (C, G, A) => 0.10, // rare
        (C, G, G) => 0.10, // rare
        (A, G, A) => 0.03, // very rare
        (A, G, G) => 0.03, // very rare
        // Gly (G)
        (G, G, U) => 0.71,
        (G, G, C) => 1.00,
        (G, G, A) => 0.09, // rare
        (G, G, G) => 0.20,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rarity_factor_is_one_for_dominant_codons() {
        // For each AA's most-used codon, w = 1.0 → factor = 1.0.
        for &c in &[
            // most-frequent codons across the 18 multi-codon AAs
            "UUC", "CUG", "AUC", "GUG", "AGC", "CCG", "ACC", "GCU", "UAC", "CAC", "CAG", "AAC",
            "AAA", "GAC", "GAA", "UGC", "CGU", "GGC", // single-codon AAs (Met and Trp)
            "AUG", "UGG",
        ] {
            let codon = parse(c);
            let f = ecoli_k12_rarity_factor(codon);
            assert!(
                (f - 1.0).abs() < 1e-9,
                "{c}: rarity factor {f} != 1.0 (w = {})",
                ecoli_k12_w(codon),
            );
        }
    }

    #[test]
    fn rare_codons_capped_at_five() {
        // The pathological E. coli rare codons (AGG, AGA at w=0.03;
        // CCC at 0.05; CGA / CGG at 0.10) all share the 5× ceiling.
        for &c in &["AGG", "AGA", "CCC", "CGA", "CGG", "AUA", "CUA", "GGA"] {
            let f = ecoli_k12_rarity_factor(parse(c));
            assert!((f - 5.0).abs() < 1e-9, "{c}: factor {f} != 5.0");
        }
    }

    #[test]
    fn every_sense_codon_has_a_finite_rarity() {
        for codon in Codon::all() {
            let f = ecoli_k12_rarity_factor(codon);
            assert!(f.is_finite() && f >= 1.0 && f <= 5.0, "{codon:?}: {f}");
        }
    }

    fn parse(s: &str) -> Codon {
        Codon::from_bytes(s.as_bytes()).unwrap()
    }
}
