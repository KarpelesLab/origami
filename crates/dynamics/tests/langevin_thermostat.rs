//! M5 acceptance: BAOAB Langevin holds an Ala₃ chain near 310 K and
//! preserves equipartition.

use chem::{standard_ff, AminoAcid};
use dynamics::{run_langevin, LangevinOptions, BOLTZMANN_KJ_PER_MOL_K};
use geom::{build_extended_chain, build_topology_graph};

#[test]
fn ala3_holds_target_temperature_after_burn_in() {
    let mut s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala, AminoAcid::Ala]).unwrap();
    let graph = build_topology_graph(&s);
    let ff = standard_ff();

    let target = 310.0;
    let opts = LangevinOptions {
        dt_fs: 1.0,
        temperature_k: target,
        friction_ps_inv: 2.0, // a bit stiffer than the default to settle faster
        steps: 2000,
        save_every: 50,
        seed: 7,
        randomise_initial_velocities: true,
        include_sasa: false,
        constrain_h_bonds: false,
    };

    // Collect (step, T_inst, KE) at each checkpoint so we can post-filter
    // the burn-in.
    let mut samples: Vec<(usize, f64, f64)> = Vec::new();
    let summary = run_langevin(&mut s, &graph, ff, opts, |frame| {
        samples.push((
            frame.step,
            frame.instantaneous_temperature_k,
            frame.kinetic_energy_kj_mol,
        ));
    });

    assert!(!summary.diverged, "Langevin trajectory diverged");
    assert!(summary.steps_run > 0);

    // Drop the burn-in: first 100 steps' worth of samples.
    let burn_in = 100usize;
    let post: Vec<&(usize, f64, f64)> =
        samples.iter().filter(|(step, _, _)| *step > burn_in).collect();
    assert!(post.len() > 10, "not enough post-burn-in samples: {}", post.len());

    let t_mean: f64 = post.iter().map(|(_, t, _)| *t).sum::<f64>() / post.len() as f64;
    assert!(
        (t_mean - target).abs() < 60.0,
        "mean temperature {t_mean} K too far from {target} K"
    );

    // Equipartition: (1/2) k_B T per DoF means mean KE = (3N/2) k_B T.
    let dof = (3 * s.atom_count()) as f64;
    let target_ke = 0.5 * dof * BOLTZMANN_KJ_PER_MOL_K * target;
    let mean_ke: f64 = post.iter().map(|(_, _, k)| *k).sum::<f64>() / post.len() as f64;
    let ratio = mean_ke / target_ke;
    assert!(
        (ratio - 1.0).abs() < 0.20,
        "equipartition ratio {ratio} (expected ~1.0)"
    );

    // No NaN, no exploded positions.
    for r in &s.residues {
        for a in &r.atoms {
            assert!(a.position.x.is_finite());
            assert!(a.position.y.is_finite());
            assert!(a.position.z.is_finite());
            assert!(a.position.norm() < 1000.0, "atom drifted too far: {:?}", a.position);
        }
    }
}

#[test]
fn ala3_trajectory_round_trips_through_pdb() {
    use io::{read_pdb_trajectory, write_pdb_trajectory};

    let mut s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala, AminoAcid::Ala]).unwrap();
    let initial_atoms = s.atom_count();
    let graph = build_topology_graph(&s);
    let ff = standard_ff();
    let opts = LangevinOptions {
        steps: 200,
        save_every: 25,
        seed: 1,
        ..Default::default()
    };
    let mut frames: Vec<geom::Structure> = Vec::new();
    run_langevin(&mut s, &graph, ff, opts, |frame| {
        frames.push(frame.structure.clone());
    });
    assert!(frames.len() >= 8, "expected ≥8 saved frames, got {}", frames.len());

    let mut buf = Vec::new();
    write_pdb_trajectory(&mut buf, "round-trip test", frames.iter()).unwrap();
    let parsed = read_pdb_trajectory(buf.as_slice()).unwrap();
    assert_eq!(parsed.len(), frames.len(), "frame count round-trip mismatch");
    for p in &parsed {
        assert_eq!(p.atom_count(), initial_atoms);
    }
}
