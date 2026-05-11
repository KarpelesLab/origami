//! Print Cα RMSD of each model in a multi-MODEL trajectory PDB against
//! a reference PDB. Usage:
//!     rmsd_trace <reference.pdb> <trajectory.pdb>
//!
//! Reports the RMSD per frame as a tab-separated table (frame_index,
//! rmsd_a). Useful for checking whether a folding trajectory is moving
//! toward or away from a known native fold.

use geom::rmsd_ca;
use io::{read_pdb, read_pdb_trajectory};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: rmsd_trace <reference.pdb> <trajectory.pdb>");
        std::process::exit(2);
    }
    let ref_pdb = std::fs::read_to_string(&args[1])
        .unwrap_or_else(|e| panic!("read {}: {}", args[1], e));
    let reference = read_pdb(ref_pdb.as_bytes())
        .unwrap_or_else(|e| panic!("parse {}: {}", args[1], e));
    let traj_pdb = std::fs::read_to_string(&args[2])
        .unwrap_or_else(|e| panic!("read {}: {}", args[2], e));
    let frames = read_pdb_trajectory(traj_pdb.as_bytes())
        .unwrap_or_else(|e| panic!("parse {}: {}", args[2], e));

    println!("# frame\trmsd_ca_A");
    let mut min_rmsd = f64::INFINITY;
    let mut min_idx = 0usize;
    for (i, frame) in frames.iter().enumerate() {
        let r = rmsd_ca(&reference, frame).unwrap_or(f64::NAN);
        println!("{}\t{:.3}", i, r);
        if r < min_rmsd {
            min_rmsd = r;
            min_idx = i;
        }
    }
    eprintln!(
        "min RMSD over {} frames: {:.3} Å at frame {}",
        frames.len(),
        min_rmsd,
        min_idx
    );
}
