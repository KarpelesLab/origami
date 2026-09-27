//! Diagnostic: dump the inter-residue geometry of a built extended RNA
//! chain — C1'-C1' distance, base-centroid distance, etc.

use chem::Nucleotide;
use geom::{build_extended_rna_chain, Vec3};

#[test]
fn dump_inter_residue_distances() {
    let nts: Vec<Nucleotide> = "GGCACUUCGGUGCC"
        .chars()
        .map(|c| Nucleotide::from_one_letter(c).unwrap())
        .collect();
    let s = build_extended_rna_chain(&nts).unwrap();

    eprintln!("\nInter-residue geometry of extended GGCACUUCGGUGCC:");
    eprintln!(
        "{:>4} {:>4} {:>9} {:>14} {:>14}",
        "ri", "ri+1", "C1'-C1'", "base-base (Å)", "P-P (Å)"
    );
    for i in 0..s.residues.len() - 1 {
        let c1_i = s.residues[i].position("C1'").unwrap();
        let c1_n = s.residues[i + 1].position("C1'").unwrap();
        let p_i = s.residues[i].position("P");
        let p_n = s.residues[i + 1].position("P").unwrap();
        let base_i = base_centroid(&s.residues[i]);
        let base_n = base_centroid(&s.residues[i + 1]);
        eprintln!(
            "{:>4} {:>4} {:>9.3} {:>14.3} {:>14}",
            i,
            i + 1,
            (c1_i - c1_n).norm(),
            (base_i - base_n).norm(),
            p_i.map(|p| format!("{:.3}", (p - p_n).norm()))
                .unwrap_or_else(|| "-".into()),
        );
    }
}

fn base_centroid(res: &geom::structure::PlacedResidue) -> Vec3 {
    let nt = res.monomer.as_nucleotide().unwrap();
    let names: &[&str] = match nt {
        Nucleotide::Adenine | Nucleotide::Guanine => {
            &["N9", "C8", "N7", "C5", "C4", "N1", "C2", "N3"]
        }
        Nucleotide::Cytosine | Nucleotide::Uracil => &["N1", "C2", "N3", "C4", "C5", "C6"],
    };
    let pts: Vec<Vec3> = names.iter().filter_map(|n| res.position(n)).collect();
    pts.iter().sum::<Vec3>() / pts.len() as f64
}
