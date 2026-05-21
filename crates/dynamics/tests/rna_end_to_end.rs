//! End-to-end RNA dynamics: build an extended ribonucleotide chain,
//! compute its energy, run a short L-BFGS minimisation, then a short
//! BAOAB Langevin trajectory at body temperature.  Asserts the whole
//! pipeline (Monomer dispatch → CHARMM27 typing → CHARMM27 parameters
//! → bonded + LJ + Coulomb + GB → integrator) is wired correctly.

use chem::{standard_ff, Nucleotide};
use dynamics::energy_eval::total_energy;
use dynamics::{minimize, run_langevin, Algorithm, LangevinOptions, MinimizeOptions};
use energy::bonded::bonded_energy;
use energy::gb::gb_energy;
use energy::nonbonded::{nonbonded_energy, DEFAULT_CUTOFF_A};
use geom::{build_extended_rna_chain, build_topology_graph};

/// Build a 4-residue UCAG, compute its energy, and assert every term
/// is finite. Catches NaN / infinity / unclassified-atom panics in the
/// new RNA force-field dispatch path.
#[test]
fn rna_chain_energy_is_finite() {
    let s = build_extended_rna_chain(&[
        Nucleotide::Uracil,
        Nucleotide::Cytosine,
        Nucleotide::Adenine,
        Nucleotide::Guanine,
    ])
    .unwrap();
    let g = build_topology_graph(&s);
    let ff = standard_ff();

    let bonded = bonded_energy(&s, &g, ff);
    let nb = nonbonded_energy(&s, &g, ff, DEFAULT_CUTOFF_A);
    let gb = gb_energy(&s, ff);

    assert!(bonded.bond_kj_mol.is_finite(), "bond: {}", bonded.bond_kj_mol);
    assert!(bonded.angle_kj_mol.is_finite(), "angle: {}", bonded.angle_kj_mol);
    assert!(bonded.dihedral_kj_mol.is_finite(), "dihedral: {}", bonded.dihedral_kj_mol);
    assert!(bonded.improper_kj_mol.is_finite(), "improper: {}", bonded.improper_kj_mol);
    assert!(nb.lj_kj_mol.is_finite(), "LJ: {}", nb.lj_kj_mol);
    assert!(nb.coulomb_kj_mol.is_finite(), "Coul: {}", nb.coulomb_kj_mol);
    assert!(gb.gb_kj_mol.is_finite(), "GB: {}", gb.gb_kj_mol);

    let total = total_energy(&s, &g, ff);
    assert!(total.is_finite(), "total energy not finite: {total}");

    eprintln!(
        "UCAG extended-chain energy: total {:.1} | bond {:.1} angle {:.1} \
         dih {:.1} imp {:.1} LJ {:.1} Coul {:.1} GB {:.1}",
        total,
        bonded.bond_kj_mol,
        bonded.angle_kj_mol,
        bonded.dihedral_kj_mol,
        bonded.improper_kj_mol,
        nb.lj_kj_mol,
        nb.coulomb_kj_mol,
        gb.gb_kj_mol,
    );
}

/// L-BFGS minimisation on the extended UCAG: assert energy drops and
/// every atom remains finite.
#[test]
fn rna_chain_minimises() {
    let mut s = build_extended_rna_chain(&[
        Nucleotide::Adenine,
        Nucleotide::Uracil,
        Nucleotide::Guanine,
        Nucleotide::Cytosine,
    ])
    .unwrap();
    let g = build_topology_graph(&s);
    let ff = standard_ff();

    let before = total_energy(&s, &g, ff);
    assert!(before.is_finite());

    let result = minimize(
        &mut s,
        &g,
        ff,
        MinimizeOptions {
            algorithm: Algorithm::Lbfgs,
            max_steps: 100,
            ..Default::default()
        },
    );
    let after = result.final_energy;

    assert!(after.is_finite(), "post-minimise energy NaN/inf: {after}");
    assert!(after < before, "minimisation didn't drop energy: {before} → {after}");
    for residue in &s.residues {
        for atom in &residue.atoms {
            assert!(atom.position.x.is_finite());
            assert!(atom.position.y.is_finite());
            assert!(atom.position.z.is_finite());
        }
    }
    eprintln!(
        "AUGC minimise: {:.1} → {:.1} kJ/mol over {} L-BFGS steps",
        before, after, result.steps
    );
}

/// Run a short BAOAB Langevin trajectory on a minimised UCAG dimer at
/// body temperature.  The acceptance is conservative — we're not
/// asserting thermodynamics, just that the integrator doesn't blow up
/// once it's fed RNA forces through the new dispatch path.
#[test]
fn rna_chain_langevin_runs_without_explosion() {
    let mut s = build_extended_rna_chain(&[
        Nucleotide::Uracil,
        Nucleotide::Adenine,
    ])
    .unwrap();
    let g = build_topology_graph(&s);
    let ff = standard_ff();

    // Brief minimisation so the integrator doesn't start with a
    // jagged extended chain (forces are huge on the first step
    // otherwise, since the extended NeRF geometry isn't at the
    // CHARMM27 r0 for every bond/angle).
    minimize(
        &mut s,
        &g,
        ff,
        MinimizeOptions {
            algorithm: Algorithm::Lbfgs,
            max_steps: 200,
            ..MinimizeOptions::default()
        },
    );

    let opts = LangevinOptions {
        dt_fs: 1.0,
        temperature_k: 310.0,
        friction_ps_inv: 2.0,
        steps: 200,
        save_every: 20,
        seed: 1729,
        randomise_initial_velocities: true,
        include_sasa: false,
        include_cmap: false,
        constrain_h_bonds: false,
    };

    let mut last_t = 0.0;
    let summary = run_langevin(&mut s, &g, ff, opts, |frame| {
        last_t = frame.instantaneous_temperature_k;
        assert!(
            frame.kinetic_energy_kj_mol.is_finite(),
            "step {}: kinetic energy NaN/inf",
            frame.step,
        );
    });

    assert!(!summary.diverged, "Langevin trajectory diverged");
    assert!(summary.steps_run >= 100);
    assert!(last_t.is_finite() && last_t > 0.0 && last_t < 10_000.0,
        "final instantaneous T off-scale: {last_t} K");

    for residue in &s.residues {
        for atom in &residue.atoms {
            assert!(atom.position.x.is_finite());
            assert!(atom.position.y.is_finite());
            assert!(atom.position.z.is_finite());
        }
    }
    eprintln!(
        "UA dimer: {} Langevin steps complete, last T_inst = {:.0} K",
        summary.steps_run, last_t
    );
}
