//! Hydrophobic-term forces (PSA.2).
//!
//! Computes `F_k = − ∂E_SASA / ∂r_k` for every atom by central-differencing
//! the per-atom PowerSasa area function. Now that PSA.1 is accurate to
//! <1 % vs Shrake-Rupley with 4096 dots, the gradient is smooth enough
//! for minimisation and Langevin dynamics.
//!
//! Optimisation: only the perturbed atom k and its caps-overlap neighbours
//! have changing per-atom areas. We recompute SASA for those atoms only
//! at each perturbation, not the whole structure. Cost per gradient ~
//! O(N · ⟨neighbours⟩²) which is a few tenths of a second on Trp-cage —
//! fine for minimisation, marginal for production dynamics. Analytical
//! Klenin §3 derivatives (PSA.2-followup) would drop this to O(N).

use chem::{Element, ForceField};
use geom::{Structure, Vec3};

use crate::powersasa::arrangement::{build_caps, find_boundary};
use crate::powersasa::area::accessible_area;
use crate::powersasa::{surface_tension_kcal, vdw_radius, PROBE_RADIUS_A};
use crate::units::kcal_to_kj;

/// Central-difference step in Å. 1e-4 keeps both terms within float
/// resolution and is comfortably inside the boundary-topology stable
/// regime for atom-overlap geometry.
const SASA_FORCE_STEP_A: f64 = 1.0e-4;

/// Add `F_k = −∂E_SASA/∂r_k` to `forces` for every atom.
///
/// `forces` length must equal `structure.atom_count()`.
pub fn add_sasa_forces(structure: &Structure, _ff: &ForceField, forces: &mut [Vec3]) {
    let n = structure.atom_count();
    assert_eq!(forces.len(), n);
    let mut positions: Vec<Vec3> = Vec::with_capacity(n);
    let mut radii: Vec<f64> = Vec::with_capacity(n);
    let mut elements: Vec<Element> = Vec::with_capacity(n);
    for residue in &structure.residues {
        for atom in &residue.atoms {
            positions.push(atom.position);
            radii.push(vdw_radius(atom.element) + PROBE_RADIUS_A);
            elements.push(atom.element);
        }
    }

    // Build neighbour adjacency from the initial positions — for the
    // small perturbation step we use here, the cap-overlap topology is
    // stable, so the static neighbour list captures every atom whose
    // per-atom area changes when k moves.
    let mut neighbour_idx: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        let pi = positions[i];
        let ri = radii[i];
        for j in (i + 1)..n {
            let d = (positions[j] - pi).norm();
            if d <= ri + radii[j] {
                neighbour_idx[i].push(j);
                neighbour_idx[j].push(i);
            }
        }
    }

    // Per-element surface tensions in kJ/mol/Å² (γ).
    let gamma: Vec<f64> = elements
        .iter()
        .map(|&e| kcal_to_kj(surface_tension_kcal(e)))
        .collect();

    let mut affected: Vec<usize> = Vec::with_capacity(64);
    let mut neighbour_buf: Vec<(usize, Vec3, f64)> = Vec::with_capacity(64);
    for k in 0..n {
        affected.clear();
        affected.extend_from_slice(&neighbour_idx[k]);
        affected.push(k);
        // Drop atoms with γ = 0 — moving k doesn't affect a polar atom's
        // contribution to E_SASA = Σ γ A even if A changes.
        affected.retain(|&i| gamma[i] != 0.0);
        if affected.is_empty() {
            continue;
        }

        for axis in 0..3 {
            let original = positions[k][axis];
            positions[k][axis] = original + SASA_FORCE_STEP_A;
            let e_plus: f64 = affected
                .iter()
                .map(|&i| {
                    gamma[i]
                        * compute_atom_area(i, &positions, &radii, &neighbour_idx[i], &mut neighbour_buf)
                })
                .sum();
            positions[k][axis] = original - SASA_FORCE_STEP_A;
            let e_minus: f64 = affected
                .iter()
                .map(|&i| {
                    gamma[i]
                        * compute_atom_area(i, &positions, &radii, &neighbour_idx[i], &mut neighbour_buf)
                })
                .sum();
            positions[k][axis] = original;
            // F_kx = −∂E/∂r_kx.
            forces[k][axis] -= (e_plus - e_minus) / (2.0 * SASA_FORCE_STEP_A);
        }
    }
}

/// PowerSasa area for a single atom, given the precomputed neighbour
/// indices for that atom (so we skip the O(N) per-call neighbour scan).
/// `neighbour_buf` is reused across calls to avoid allocation.
fn compute_atom_area(
    atom_idx: usize,
    positions: &[Vec3],
    radii: &[f64],
    neighbour_indices: &[usize],
    neighbour_buf: &mut Vec<(usize, Vec3, f64)>,
) -> f64 {
    let pi = positions[atom_idx];
    let ri = radii[atom_idx];
    neighbour_buf.clear();
    for &j in neighbour_indices {
        let pj = positions[j];
        let d = (pj - pi).norm();
        // The static index list is correct as long as the perturbation
        // step is small enough that no new neighbour pairs come into
        // range. Re-test the distance defensively so a glancing pair
        // doesn't sneak through with a negative cone-half-angle.
        if d <= ri + radii[j] {
            neighbour_buf.push((j, pj, radii[j]));
        }
    }
    let (caps, _owners) = match build_caps(pi, ri, neighbour_buf) {
        Some(c) => c,
        None => return 0.0,
    };
    let boundary = find_boundary(&caps);
    accessible_area(ri, &caps, &boundary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chem::{standard_ff, AminoAcid};
    use geom::{build_extended_chain, structure::PlacedAtom, structure::PlacedResidue, Structure};

    #[test]
    fn isolated_atom_has_zero_sasa_force() {
        // A lone carbon has constant area 4πR², so any perturbation
        // leaves E_SASA unchanged → F = 0.
        let s = Structure {
            residues: vec![PlacedResidue {
                aa: AminoAcid::Ala,
                atoms: vec![PlacedAtom {
                    name: "CB",
                    element: Element::C,
                    position: Vec3::zeros(),
                }],
            }],
        };
        let ff = standard_ff();
        let mut forces = vec![Vec3::zeros(); 1];
        add_sasa_forces(&s, ff, &mut forces);
        assert!(forces[0].norm() < 1e-8, "expected ~0 force, got {:?}", forces[0]);
    }

    #[test]
    fn ala3_sasa_forces_sum_to_zero() {
        // Newton's third law: internal forces balance.
        let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala, AminoAcid::Ala]).unwrap();
        let ff = standard_ff();
        let n = s.atom_count();
        let mut forces = vec![Vec3::zeros(); n];
        add_sasa_forces(&s, ff, &mut forces);
        let net = forces.iter().fold(Vec3::zeros(), |acc, f| acc + *f);
        assert!(
            net.norm() < 1e-3,
            "net SASA force not balanced: {:?} (|net| = {})",
            net,
            net.norm()
        );
    }

    #[test]
    fn sasa_forces_finite_difference_matches_total_e_sasa() {
        // The forces should equal the central-difference gradient of the
        // total E_SASA energy. Verify on a small built chain.
        let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Val]).unwrap();
        let ff = standard_ff();
        let n = s.atom_count();
        let mut forces = vec![Vec3::zeros(); n];
        add_sasa_forces(&s, ff, &mut forces);

        // Compute total E_SASA at a few perturbed positions and compare
        // the central difference to the reported force.
        let eps = 1e-4;
        for k in 0..n.min(4) {
            for axis in 0..3 {
                let mut s_plus = s.clone();
                let mut s_minus = s.clone();
                bump(&mut s_plus, k, axis, eps);
                bump(&mut s_minus, k, axis, -eps);
                let e_plus = crate::powersasa::powersasa_energy(&s_plus, ff).sasa_kj_mol;
                let e_minus = crate::powersasa::powersasa_energy(&s_minus, ff).sasa_kj_mol;
                let numerical_force = -(e_plus - e_minus) / (2.0 * eps);
                let reported = forces[k][axis];
                assert!(
                    (reported - numerical_force).abs() < 0.5,
                    "atom {} axis {}: analytical {} vs numerical {} (diff {})",
                    k,
                    axis,
                    reported,
                    numerical_force,
                    (reported - numerical_force).abs(),
                );
            }
        }
    }

    fn bump(s: &mut Structure, atom_idx: usize, axis: usize, eps: f64) {
        let mut count = 0usize;
        for residue in &mut s.residues {
            for atom in &mut residue.atoms {
                if count == atom_idx {
                    atom.position[axis] += eps;
                    return;
                }
                count += 1;
            }
        }
    }
}
