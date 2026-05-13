//! Read a PDB and print which Cys-Cys pairs were auto-detected as
//! disulfide bridges by `build_topology_graph`. Quick sanity check
//! for the M14 disulfide work — given a structure with declared
//! SSBOND records, confirm our geometric detection finds the same
//! pairs.

use chem::AminoAcid;
use geom::build_topology_graph;
use io::read_pdb;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| {
        eprintln!("usage: check_disulfides <pdb>");
        std::process::exit(2)
    });
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let s = read_pdb(bytes.as_slice()).unwrap_or_else(|e| panic!("parse {path}: {e}"));
    let g = build_topology_graph(&s);

    let mut sg_per_res: Vec<(usize, Option<usize>)> = Vec::new();
    let mut atom_idx = 0;
    for (ri, r) in s.residues.iter().enumerate() {
        let mut sg = None;
        for a in &r.atoms {
            if a.name == "SG" {
                sg = Some(atom_idx);
            }
            atom_idx += 1;
        }
        if r.aa == AminoAcid::Cys {
            sg_per_res.push((ri, sg));
        }
    }
    println!("{}: {} cysteines", path, sg_per_res.len());
    let mut detected = 0;
    for (i, (ri, sgi)) in sg_per_res.iter().enumerate() {
        for (rj, sgj) in sg_per_res.iter().skip(i + 1) {
            if let (Some(a), Some(b)) = (sgi, sgj) {
                if g.is_bonded(*a, *b) {
                    println!("  detected disulfide: Cys{} -- Cys{}", ri + 1, rj + 1);
                    detected += 1;
                }
            }
        }
    }
    println!("total detected: {}", detected);
}
