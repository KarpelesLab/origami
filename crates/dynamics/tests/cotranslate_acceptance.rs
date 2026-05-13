//! M6 acceptance: cotranslate grows a chain residue-by-residue, stays
//! finite, and produces a multi-frame trajectory.

use chem::{standard_ff, AminoAcid};
use dynamics::{
    run_cotranslate, CylindricalTunnel, ExternalPotential, LangevinOptions, UniformRibosome,
};
use geom::Vec3;

#[test]
fn cotranslate_ala5_grows_to_five_residues() {
    let r = UniformRibosome::new(
        vec![
            AminoAcid::Ala,
            AminoAcid::Gly,
            AminoAcid::Ala,
            AminoAcid::Gly,
            AminoAcid::Ala,
        ],
        100.0, // 100 fs per residue
    );
    let opts = LangevinOptions {
        dt_fs: 1.0,
        temperature_k: 310.0,
        friction_ps_inv: 2.0,
        steps: 0,
        save_every: 25,
        seed: 7,
        randomise_initial_velocities: true,
        include_sasa: false,
        constrain_h_bonds: false,
    };
    let ff = standard_ff();

    let mut residue_history: Vec<usize> = Vec::new();
    let final_s = run_cotranslate(&r, ff, opts, 200, None, |frame| {
        residue_history.push(frame.residue_count);
        for resi in &frame.structure.residues {
            for atom in &resi.atoms {
                assert!(atom.position.x.is_finite());
                assert!(atom.position.y.is_finite());
                assert!(atom.position.z.is_finite());
            }
        }
    });

    assert_eq!(final_s.residues.len(), 5);
    // Atom count for Ala-Gly-Ala-Gly-Ala = 10 + 7 + 10 + 7 + 10 = 44.
    assert_eq!(final_s.atom_count(), 44);

    // Residue count should never decrease and should reach 5.
    let max_seen = residue_history.iter().copied().max().unwrap_or(0);
    assert_eq!(max_seen, 5);
    let mut prev = 0;
    for &c in &residue_history {
        assert!(c >= prev, "residue count went backwards: {} -> {}", prev, c);
        prev = c;
    }
    assert!(residue_history.len() > 5, "expected several frames, got {}", residue_history.len());
}

#[test]
fn cotranslate_with_tunnel_keeps_chain_inside_radius() {
    // A tight tunnel forces the chain to stay near the axis. After 5
    // residues, no atom should be more than `radius + slop` from the
    // tunnel axis (within the tunnel's axial extent).
    let r = UniformRibosome::new(vec![AminoAcid::Ala; 4], 80.0);
    let tunnel = CylindricalTunnel {
        axis_origin: Vec3::new(-2.0, 0.0, 0.0),
        axis_direction: Vec3::new(1.0, 0.0, 0.0),
        radius_a: 6.0,
        length_a: 60.0,
        k_confine: 200.0,
    };
    let external: &dyn ExternalPotential = &tunnel;
    let opts = LangevinOptions {
        dt_fs: 1.0,
        temperature_k: 310.0,
        friction_ps_inv: 2.0,
        steps: 0,
        save_every: 25,
        seed: 11,
        randomise_initial_velocities: true,
        include_sasa: false,
        constrain_h_bonds: false,
    };
    let ff = standard_ff();

    let mut max_radial = 0.0_f64;
    let final_s = run_cotranslate(&r, ff, opts, 200, Some(external), |frame| {
        for resi in &frame.structure.residues {
            for atom in &resi.atoms {
                let v = atom.position - tunnel.axis_origin;
                let along = v.dot(&tunnel.axis_direction);
                if (0.0..=tunnel.length_a).contains(&along) {
                    let perp = v - tunnel.axis_direction * along;
                    let d = perp.norm();
                    if d > max_radial {
                        max_radial = d;
                    }
                }
            }
        }
    });
    assert_eq!(final_s.residues.len(), 4);
    // Soft confinement: atoms can wander a bit beyond the radius
    // (overshoot ~ √(k_BT / k_confine)) before being pushed back, but
    // shouldn't fly off.
    assert!(
        max_radial < tunnel.radius_a + 4.0,
        "max radial distance {} > {} + slop", max_radial, tunnel.radius_a
    );
}
