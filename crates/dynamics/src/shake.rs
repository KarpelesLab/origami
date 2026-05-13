//! SHAKE bond-length constraint solver (Ryckaert, Ciccotti, Berendsen
//! 1977, J. Comput. Phys. 23, 327). For each constrained bond `(i, j)`
//! with target length `d`, the unconstrained position update is
//! corrected by a Lagrange multiplier along the **reference** vector
//! `r_ij^old` (the constraint-satisfying separation before the step):
//!
//!   Δr_i = −(1 / m_i) · λ · r_ij^old
//!   Δr_j = +(1 / m_j) · λ · r_ij^old
//!
//! Substituting into the constraint `|r_ij^new + (Δr_i − Δr_j)|² = d²`
//! and linearising (dropping the O(λ²) term gives one Newton step per
//! constraint, which we then iterate to convergence):
//!
//!   λ = (|r_ij^new|² − d²) / [ 2 · (1/m_i + 1/m_j) · (r_ij^new · r_ij^old) ]
//!
//! The iteration converges in 2-4 passes for typical X-H bonds at
//! dt = 2 fs and tolerance 1e-6 Å². Convergence failure (within
//! `max_iters`) is reported back so the integrator can flag a step.
//!
//! ## What we constrain
//!
//! Only X-H bonds. Heavy-atom bonds vibrate at frequencies low enough
//! to be resolved by 2 fs; the hydrogen stretches (~3000 cm⁻¹ = 11 fs
//! period) are the only ones that force dt ≤ 1 fs. Freezing them lets
//! the integrator double its timestep with no loss of structural
//! sampling fidelity, since H positions in heavy-atom-bonded units
//! follow the heavy atom rigidly anyway.

use chem::{AtomType, Element, ForceField};
use geom::{Structure, TopologyGraph, Vec3};

/// One constrained bond between two atom indices, with the squared
/// target separation precomputed for the inner-loop comparison.
#[derive(Debug, Clone, Copy)]
pub struct Constraint {
    pub i: usize,
    pub j: usize,
    pub d: f64,
    pub d_sq: f64,
}

/// Build the list of X-H bond constraints from the topology graph. The
/// target length is the force field's equilibrium bond length `r0`
/// (looked up by the same atom types the bonded-energy code uses); if
/// the FF doesn't have an entry for a particular bond type the
/// constraint is silently skipped (very rare for X-H pairs in CHARMM36).
pub fn build_h_bond_constraints(
    structure: &Structure,
    graph: &TopologyGraph,
    ff: &ForceField,
    atom_types: &[AtomType],
) -> Vec<Constraint> {
    let atoms_flat: Vec<Element> = structure
        .residues
        .iter()
        .flat_map(|r| r.atoms.iter().map(|a| a.element))
        .collect();
    let mut out = Vec::new();
    for b in &graph.bonds {
        let is_xh = atoms_flat[b.a] == Element::H || atoms_flat[b.b] == Element::H;
        if !is_xh {
            continue;
        }
        let Some(p) = ff.bond(atom_types[b.a], atom_types[b.b]) else {
            continue;
        };
        let d = p.r0;
        out.push(Constraint {
            i: b.a,
            j: b.b,
            d,
            d_sq: d * d,
        });
    }
    out
}

/// In-place SHAKE iteration. `positions` are the post-unconstrained-
/// step coordinates (after the A half-step). `reference_positions` are
/// the pre-step coordinates, used for the Lagrange direction.
/// `inv_masses[i]` = 1 / m_i in Da⁻¹.
///
/// Returns `Some(iters)` if every constraint converged to within
/// `tol_sq` (squared length tolerance in Å²) inside `max_iters`
/// passes, otherwise `None`.
pub fn shake_iterate(
    positions: &mut [Vec3],
    reference_positions: &[Vec3],
    inv_masses: &[f64],
    constraints: &[Constraint],
    tol_sq: f64,
    max_iters: usize,
) -> Option<usize> {
    for iter in 1..=max_iters {
        let mut max_err: f64 = 0.0;
        for c in constraints {
            let r_new = positions[c.i] - positions[c.j];
            let len2_new = r_new.norm_squared();
            let err = len2_new - c.d_sq;
            if err.abs() > max_err {
                max_err = err.abs();
            }
            if err.abs() < tol_sq {
                continue;
            }
            let r_old = reference_positions[c.i] - reference_positions[c.j];
            let dot = r_new.dot(&r_old);
            let denom = 2.0 * (inv_masses[c.i] + inv_masses[c.j]) * dot;
            if denom.abs() < 1e-18 {
                // Reference vector is nearly perpendicular to current
                // vector — SHAKE's linearisation breaks down. Skip and
                // hope the next iteration sees better geometry.
                continue;
            }
            let lambda = err / denom;
            let delta = r_old * lambda;
            positions[c.i] -= delta * inv_masses[c.i];
            positions[c.j] += delta * inv_masses[c.j];
        }
        if max_err < tol_sq {
            return Some(iter);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_bond_settles_to_target_length() {
        // Two atoms 1.5 Å apart; constrain to 1.0 Å.
        let mut positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.5, 0.0, 0.0),
        ];
        let reference = positions.clone();
        let inv_masses = vec![1.0, 1.0];
        let constraints = vec![Constraint {
            i: 0,
            j: 1,
            d: 1.0,
            d_sq: 1.0,
        }];
        let iters = shake_iterate(&mut positions, &reference, &inv_masses, &constraints, 1e-8, 20)
            .expect("SHAKE converged");
        let r = (positions[0] - positions[1]).norm();
        assert!(
            (r - 1.0).abs() < 1e-3,
            "expected ~1.0 Å, got {r} (iters={iters})"
        );
    }

    #[test]
    fn already_satisfied_returns_immediately() {
        let mut positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        ];
        let reference = positions.clone();
        let inv_masses = vec![1.0, 1.0];
        let constraints = vec![Constraint {
            i: 0,
            j: 1,
            d: 1.0,
            d_sq: 1.0,
        }];
        let iters = shake_iterate(&mut positions, &reference, &inv_masses, &constraints, 1e-8, 5)
            .unwrap();
        assert_eq!(iters, 1);
    }

    #[test]
    fn unequal_masses_split_correction_proportionally() {
        // Atom 0 has mass 1, atom 1 has mass 100 (much heavier). After
        // SHAKE almost all the correction lands on atom 0.
        let mut positions = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.5, 0.0, 0.0),
        ];
        let reference = positions.clone();
        let inv_masses = vec![1.0, 0.01];
        let constraints = vec![Constraint {
            i: 0,
            j: 1,
            d: 1.0,
            d_sq: 1.0,
        }];
        let _ = shake_iterate(&mut positions, &reference, &inv_masses, &constraints, 1e-8, 20)
            .expect("SHAKE converged");
        let r = (positions[0] - positions[1]).norm();
        assert!((r - 1.0).abs() < 1e-3);
        // Atom 1 (heavy) should have moved much less than atom 0.
        let moved_0 = (positions[0] - reference[0]).norm();
        let moved_1 = (positions[1] - reference[1]).norm();
        assert!(moved_0 > 10.0 * moved_1, "moved_0={moved_0}, moved_1={moved_1}");
    }
}
