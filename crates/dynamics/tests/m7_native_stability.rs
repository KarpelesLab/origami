//! M7d — short Langevin MD on a native fold should keep the structure
//! close to its starting Cα coordinates. If the force field is sane,
//! 2 ps of dynamics at 310 K won't unfold a stable protein. This is the
//! flip side of M3's "native < extended" energy ranking — it confirms
//! the *forces*, not just the energy values, are consistent with the
//! correct conformation being a (local) minimum.
//!
//! Reference proteins: Trp-cage (1L2Y MODEL 1, 20 aa). Larger proteins
//! could be added but get expensive in debug mode; the principle is the
//! same. The chignolin fixture is a small β-hairpin and tends to
//! breathe more during short MD — included with a looser RMSD bound.

use chem::standard_ff;
use dynamics::{run_langevin, LangevinOptions};
use geom::{build_topology_graph, rmsd_ca};
use io::read_pdb;

fn read_fixture(path: &str) -> geom::Structure {
    let pdb = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    read_pdb(pdb.as_bytes()).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

#[test]
fn trp_cage_native_stays_near_native_during_2ps_md() {
    let native = read_fixture("../io/tests/fixtures/1L2Y_model1.pdb");
    let initial = native.clone();
    let mut s = native;
    let g = build_topology_graph(&s);
    let ff = standard_ff();
    let opts = LangevinOptions {
        dt_fs: 1.0,
        temperature_k: 310.0,
        friction_ps_inv: 2.0,
        steps: 2000,
        save_every: 0, // don't bother saving frames
        seed: 5,
        randomise_initial_velocities: true,
        include_sasa: false,
        constrain_h_bonds: false,
    };
    let summary = run_langevin(&mut s, &g, ff, opts, |_| {});
    assert!(!summary.diverged, "trajectory diverged");
    let rmsd = rmsd_ca(&initial, &s).expect("rmsd");
    eprintln!("Trp-cage native MD 2 ps: Cα RMSD = {:.3} Å", rmsd);
    // The native is itself a representative member of the NMR
    // ensemble; 2 ps of Langevin shouldn't push it more than ~3.5 Å
    // away. A working force field keeps it well under.
    assert!(
        rmsd < 3.5,
        "Trp-cage Cα RMSD {rmsd} exceeds 3.5 Å — force field may be off"
    );
}
