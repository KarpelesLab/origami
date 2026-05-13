//! Crambin (PDB 1CRN, 46 residues) is a small plant protein stabilised
//! by three disulfide bridges: Cys3-Cys40, Cys4-Cys32, Cys16-Cys26.
//! Two checks:
//!
//!   1. `build_topology_graph` auto-detects all three S-S bonds from
//!      the native PDB's SG-SG distances (no SSBOND-record parsing
//!      needed — the geometry is the source of truth).
//!
//!   2. After a short Langevin run from the native conformation the
//!      chain stays compact and finite (no divergence, sensible
//!      Cα RMSD against the start). The disulfides are part of the
//!      bonded topology so their harmonic restraint is what holds the
//!      tertiary structure together; if the SG-SG bond force were
//!      broken (wrong params, or the bond wasn't added at all) the
//!      chain would unfold in 1-2 ps.

use chem::standard_ff;
use chem::AminoAcid;
use dynamics::{run_langevin, LangevinOptions};
use geom::{build_topology_graph, rmsd_ca};
use io::read_pdb;

fn read_fixture(path: &str) -> geom::Structure {
    let pdb = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    read_pdb(pdb.as_bytes()).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

fn cys_sg_indices(s: &geom::Structure) -> Vec<(usize, usize)> {
    // Return list of (residue_idx_1based, global_atom_idx) for each
    // cysteine's SG atom in residue order.
    let mut out = Vec::new();
    let mut atom_idx = 0usize;
    for (ri, r) in s.residues.iter().enumerate() {
        let mut sg = None;
        for a in &r.atoms {
            if a.name == "SG" {
                sg = Some(atom_idx);
            }
            atom_idx += 1;
        }
        if r.aa == AminoAcid::Cys {
            if let Some(idx) = sg {
                out.push((ri + 1, idx));
            }
        }
    }
    out
}

#[test]
fn crambin_disulfides_auto_detected() {
    let s = read_fixture("../io/tests/fixtures/1CRN_crambin.pdb");
    let g = build_topology_graph(&s);
    let cyses = cys_sg_indices(&s);
    assert_eq!(cyses.len(), 6, "crambin has 6 cysteines");

    let mut detected: Vec<(usize, usize)> = Vec::new();
    for (i, (ri, sgi)) in cyses.iter().enumerate() {
        for (rj, sgj) in cyses.iter().skip(i + 1) {
            if g.is_bonded(*sgi, *sgj) {
                detected.push((*ri, *rj));
            }
        }
    }
    detected.sort();
    let expected: Vec<(usize, usize)> = vec![(3, 40), (4, 32), (16, 26)];
    assert_eq!(
        detected, expected,
        "expected crambin's three disulfides {expected:?}, detected {detected:?}"
    );
}

#[test]
fn crambin_stays_native_like_during_short_md() {
    let initial = read_fixture("../io/tests/fixtures/1CRN_crambin.pdb");
    let mut s = initial.clone();
    let g = build_topology_graph(&s);
    let ff = standard_ff();
    let opts = LangevinOptions {
        dt_fs: 1.0,
        temperature_k: 310.0,
        friction_ps_inv: 2.0,
        steps: 1000, // 1 ps — short, just enough to exercise the disulfide forces
        save_every: 0,
        seed: 0,
        randomise_initial_velocities: true,
        include_sasa: false,
    };
    let summary = run_langevin(&mut s, &g, ff, opts, |_| {});
    assert!(!summary.diverged, "trajectory diverged");
    let rmsd = rmsd_ca(&initial, &s).expect("rmsd");
    eprintln!("Crambin native MD 1 ps: Cα RMSD = {:.3} Å", rmsd);
    // The structure has heavy-atom-only coordinates (the 1CRN deposit
    // omits hydrogens), so our chain builder placed hydrogens via NeRF;
    // the resulting bonded-term strain is higher than for a fully-
    // resolved native, giving more drift than the Trp-cage analogue.
    // Bound is 4 Å — generous, but anything > 5 Å on a stable
    // 46-residue protein would mean we'd broken the disulfide topology.
    assert!(
        rmsd < 4.0,
        "Crambin Cα RMSD {rmsd} exceeds 4 Å — disulfide bridges may not be holding"
    );
}
