//! `add_rna_hydrogens` round-trips correctly: stripping H atoms and
//! re-adding them via the same canonical helpers used by the chain
//! builder reproduces the original geometry to machine precision.

use chem::{Element, Nucleotide};
use geom::{add_rna_hydrogens, build_a_form_rna_chain, build_extended_rna_chain};

fn strip_hydrogens(s: &mut geom::Structure) {
    for r in &mut s.residues {
        r.atoms.retain(|a| a.element != Element::H);
    }
}

#[test]
fn strip_and_readd_extended_rna_recovers_original() {
    let seq = [Nucleotide::Uracil, Nucleotide::Cytosine,
               Nucleotide::Adenine, Nucleotide::Guanine];
    let reference = build_extended_rna_chain(&seq).unwrap();

    // Strip and re-add.
    let mut probe = reference.clone();
    strip_hydrogens(&mut probe);
    let summary = add_rna_hydrogens(&mut probe);
    eprintln!("UCAG extended: {summary:?}");

    assert_eq!(summary.residues_touched, 4);
    assert!(summary.h_added > 0);
    assert_eq!(summary.h_already_present, 0);
    assert_eq!(summary.h_skipped_missing_anchors, 0);
    assert_eq!(probe.atom_count(), reference.atom_count(),
        "atom count mismatch after re-hydrogenation");

    // Every H in `reference` should be in `probe` at the same position.
    for (r_ref, r_probe) in reference.residues.iter().zip(probe.residues.iter()) {
        for a in &r_ref.atoms {
            if a.element != Element::H { continue; }
            let p = r_probe.position(a.name)
                .unwrap_or_else(|| panic!("H atom {} missing after re-add", a.name));
            let d = (p - a.position).norm();
            assert!(d < 1e-9,
                "H atom {} moved by {d} Å after strip+readd", a.name);
        }
    }
}

#[test]
fn strip_and_readd_a_form_rna_recovers_original() {
    let seq = [Nucleotide::Adenine, Nucleotide::Uracil, Nucleotide::Guanine,
               Nucleotide::Cytosine];
    let reference = build_a_form_rna_chain(&seq).unwrap();

    let mut probe = reference.clone();
    strip_hydrogens(&mut probe);
    let summary = add_rna_hydrogens(&mut probe);
    eprintln!("AUGC A-form: {summary:?}");
    assert_eq!(probe.atom_count(), reference.atom_count());
    for (r_ref, r_probe) in reference.residues.iter().zip(probe.residues.iter()) {
        for a in &r_ref.atoms {
            if a.element != Element::H { continue; }
            let p = r_probe.position(a.name).unwrap();
            assert!((p - a.position).norm() < 1e-9);
        }
    }
}

#[test]
fn readd_on_already_hydrogenated_is_a_noop() {
    let seq = [Nucleotide::Uracil, Nucleotide::Adenine];
    let reference = build_extended_rna_chain(&seq).unwrap();
    let mut probe = reference.clone();
    let summary = add_rna_hydrogens(&mut probe);
    assert_eq!(summary.h_added, 0);
    assert!(summary.h_already_present > 0);
    assert_eq!(probe.atom_count(), reference.atom_count());
}
