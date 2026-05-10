//! M4 acceptance test: minimising the extended Trp-cage chain takes the
//! +47506 kJ/mol structure built in M2 down to a stable, sub-1000 kJ/mol
//! conformation in fewer than 500 L-BFGS steps. Bond term must drop from
//! +23494 kJ/mol to under 200, confirming the atoms relax to CHARMM
//! equilibrium values.

use chem::{standard_ff, AminoAcid};
use dynamics::{minimize, Algorithm, MinimizeOptions};
use energy::bonded::bonded_energy;
use geom::{build_extended_chain, build_topology_graph};

#[test]
fn extended_trp_cage_minimises_to_low_energy() {
    let seq: Vec<AminoAcid> = "NLYIQWLKDGGPSSGRPPPS"
        .chars()
        .filter_map(AminoAcid::from_one_letter)
        .collect();
    assert_eq!(seq.len(), 20);

    let mut s = build_extended_chain(&seq).expect("build extended chain");
    let g = build_topology_graph(&s);
    let ff = standard_ff();

    let opts = MinimizeOptions {
        algorithm: Algorithm::Lbfgs,
        max_steps: 500,
        gradient_tol: 5.0,
        max_step_a: 0.1,
        ..Default::default()
    };
    let result = minimize(&mut s, &g, ff, opts);

    eprintln!(
        "Trp-cage extended → minimised: {:.1} → {:.1} kJ/mol in {} steps",
        result.initial_energy, result.final_energy, result.steps,
    );
    assert!(
        result.final_energy < 1000.0,
        "minimised total {} > 1000 kJ/mol", result.final_energy,
    );
    assert!(
        result.steps < 500,
        "took too many steps ({})", result.steps,
    );

    // Re-evaluate bonded breakdown on the final structure.
    let bonded = bonded_energy(&s, &g, ff);
    assert!(
        bonded.bond_kj_mol < 200.0,
        "bond term {} > 200 kJ/mol after minimisation", bonded.bond_kj_mol,
    );
}
