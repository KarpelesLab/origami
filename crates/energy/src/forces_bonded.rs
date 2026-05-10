//! Analytical gradients of the bonded energy terms.
//!
//! Each function adds its contribution to a `forces: &mut [Vec3]` array
//! (one entry per atom). Forces are negative gradients: F = −∂V/∂r.
//! Returned in **kJ/mol/Å** so that the optimizer can step in Å directly.
//!
//! All four routines share the same broad structure: read the relevant
//! geometry, look up parameters, compute the scalar derivative `dV/dq`
//! where q ∈ {r, θ, φ, ω}, then distribute it across atoms via the chain
//! rule for q.

use chem::{classify, AtomType, ForceField};
use geom::{Structure, TopologyGraph, Vec3};

use crate::units::{deg_to_rad, kcal_to_kj};

/// `kJ/mol/Å` per `kcal/mol/Å` — same factor as energy conversion.
fn kcal_per_a_to_kj_per_a(k: f64) -> f64 {
    kcal_to_kj(k)
}

pub fn build_atom_types(structure: &Structure) -> Vec<AtomType> {
    let mut out = Vec::with_capacity(structure.atom_count());
    for residue in &structure.residues {
        for atom in &residue.atoms {
            out.push(
                classify(residue.aa, atom.name)
                    .unwrap_or_else(|| panic!("unclassified atom {:?} {}", residue.aa, atom.name)),
            );
        }
    }
    out
}

pub fn flatten_positions(structure: &Structure) -> Vec<Vec3> {
    let mut out = Vec::with_capacity(structure.atom_count());
    for residue in &structure.residues {
        for atom in &residue.atoms {
            out.push(atom.position);
        }
    }
    out
}

/// Add bond forces to `forces`. Bond term V = K (r − r₀)² ⇒
/// F_i = 2K (r − r₀) (r_j − r_i)/r ; F_j = −F_i.
pub fn add_bond_forces(
    positions: &[Vec3],
    graph: &TopologyGraph,
    ff: &ForceField,
    atom_types: &[AtomType],
    forces: &mut [Vec3],
) {
    for b in &graph.bonds {
        let Some(p) = ff.bond(atom_types[b.a], atom_types[b.b]) else { continue };
        let d = positions[b.b] - positions[b.a];
        let r = d.norm();
        if r < 1e-9 {
            continue; // pathological coincident atoms — skip
        }
        let dr = r - p.r0;
        // Magnitude of force in kcal/mol/Å, along d̂ from a → b for atom a
        // (positive when stretched: pulls a toward b).
        let mag_kcal = 2.0 * p.k * dr;
        let mag = kcal_per_a_to_kj_per_a(mag_kcal);
        let f = d * (mag / r);
        forces[b.a] += f;
        forces[b.b] -= f;
    }
}

/// Add angle forces. V = K(θ − θ₀)² with central atom b. The force
/// distribution uses the standard cross-product formulation:
/// F_a = (2K(θ−θ₀)/sin θ) × (v̂ − cos θ × û)/|u|
/// F_c = (2K(θ−θ₀)/sin θ) × (û − cos θ × v̂)/|v|
/// F_b = −(F_a + F_c).
pub fn add_angle_forces(
    positions: &[Vec3],
    graph: &TopologyGraph,
    ff: &ForceField,
    atom_types: &[AtomType],
    forces: &mut [Vec3],
) {
    for ang in &graph.angles {
        let Some(p) = ff.angle(atom_types[ang.a], atom_types[ang.b], atom_types[ang.c]) else {
            continue;
        };
        let u = positions[ang.a] - positions[ang.b];
        let v = positions[ang.c] - positions[ang.b];
        let u_norm = u.norm();
        let v_norm = v.norm();
        if u_norm < 1e-9 || v_norm < 1e-9 {
            continue;
        }
        let u_hat = u / u_norm;
        let v_hat = v / v_norm;
        let cos_theta = u_hat.dot(&v_hat).clamp(-1.0, 1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        if sin_theta < 1e-9 {
            continue; // near-linear angle — derivative undefined
        }
        let theta = cos_theta.acos();
        let theta0 = deg_to_rad(p.theta0_deg);
        let dvdtheta_kcal = 2.0 * p.k * (theta - theta0);
        let coeff = kcal_per_a_to_kj_per_a(dvdtheta_kcal) / sin_theta;
        // F = -dV/dr = -dV/dθ × dθ/dr.
        // dθ/dr_a points along (cos θ × û - v̂) / (|u| sin θ).
        // So F_a = -coeff × (cos θ × û - v̂) / |u| = coeff × (v̂ - cos θ × û)/|u|.
        let f_a = (v_hat - u_hat * cos_theta) * (coeff / u_norm);
        let f_c = (u_hat - v_hat * cos_theta) * (coeff / v_norm);
        let f_b = -(f_a + f_c);
        forces[ang.a] += f_a;
        forces[ang.b] += f_b;
        forces[ang.c] += f_c;
    }
}

/// Compute the gradient of a torsion-angle φ (defined by atoms a-b-c-d)
/// with respect to each of the four atom positions. Returns
/// `(∂φ/∂r_a, ∂φ/∂r_b, ∂φ/∂r_c, ∂φ/∂r_d)`.
///
/// Standard formulation (Bekker 1996; Smith 1993). Independent of whether
/// the surrounding term is a periodic dihedral or a harmonic improper.
fn dihedral_gradient(
    pa: Vec3,
    pb: Vec3,
    pc: Vec3,
    pd: Vec3,
) -> Option<(Vec3, Vec3, Vec3, Vec3, f64)> {
    let b1 = pb - pa;
    let b2 = pc - pb;
    let b3 = pd - pc;
    let b2_norm = b2.norm();
    if b2_norm < 1e-9 {
        return None;
    }
    let s = b1.cross(&b2);
    let t = b2.cross(&b3);
    let s_norm_sq = s.norm_squared();
    let t_norm_sq = t.norm_squared();
    if s_norm_sq < 1e-18 || t_norm_sq < 1e-18 {
        return None;
    }
    // Sign convention: φ = atan2((s × b̂2)·t, s·t) — opposite-signed from
    // GROMACS's φ. So our gradient formulas are negated relative to GROMACS:
    //   ∂φ/∂r_a = +|b2|/|s|² × s  (cross-checked against finite differences
    //   in dihedral_gradient_isolated_check).
    let dphi_da = s * (b2_norm / s_norm_sq);
    let dphi_dd = -t * (b2_norm / t_norm_sq);
    let inv_b2_norm_sq = 1.0 / (b2_norm * b2_norm);
    let f1 = b1.dot(&b2) * inv_b2_norm_sq;
    let f2 = b3.dot(&b2) * inv_b2_norm_sq;
    // Inner atoms via the chain-rule contribution of moving b (which moves
    // both b1 and b2). After negation from GROMACS's convention:
    let dphi_db = -(f1 + 1.0) * dphi_da + f2 * dphi_dd;
    let dphi_dc = -(dphi_da + dphi_db + dphi_dd);

    // Compute φ for use by the caller (avoids recomputing).
    let phi = {
        let m1 = s.cross(&(b2 / b2_norm));
        let x = s.dot(&t);
        let y = m1.dot(&t);
        y.atan2(x)
    };

    Some((dphi_da, dphi_db, dphi_dc, dphi_dd, phi))
}

/// Add proper-dihedral forces. V = Σ kₙ [1 + cos(nφ − δ)]
/// ⇒ dV/dφ = −Σ kₙ × n × sin(nφ − δ)
/// ⇒ F_X = +Σ kₙ × n × sin(nφ − δ) × ∂φ/∂r_X.
pub fn add_dihedral_forces(
    positions: &[Vec3],
    graph: &TopologyGraph,
    ff: &ForceField,
    atom_types: &[AtomType],
    forces: &mut [Vec3],
) {
    for d in &graph.dihedrals {
        let Some(terms) = ff.dihedral(
            atom_types[d.a], atom_types[d.b], atom_types[d.c], atom_types[d.d],
        ) else { continue };
        let Some((da, db, dc, dd, phi)) = dihedral_gradient(
            positions[d.a], positions[d.b], positions[d.c], positions[d.d],
        ) else { continue };
        // Sum dV/dφ across all periodic terms (in kcal/mol/rad).
        let mut dvdphi_kcal = 0.0;
        for term in terms {
            let arg = term.n as f64 * phi - deg_to_rad(term.delta_deg);
            dvdphi_kcal += -term.k * (term.n as f64) * arg.sin();
        }
        let dvdphi = kcal_to_kj(dvdphi_kcal);
        // F_X = -dV/dr_X = -dV/dφ × dφ/dr_X. With our sign convention for
        // dφ/dr_X (cross-checked in dihedral_gradient_isolated_check), the
        // sign comes out as forces[X] -= dvdphi × dφ/dr_X.
        // Verified against the dihedral-only energy in dihedral_force_finite_difference.
        forces[d.a] -= da * dvdphi;
        forces[d.b] -= db * dvdphi;
        forces[d.c] -= dc * dvdphi;
        forces[d.d] -= dd * dvdphi;
    }
}

/// Add improper forces. V = K (ω − ω₀)² ⇒ dV/dω = 2K (ω − ω₀)
/// ⇒ F_X = −2K (ω − ω₀) × ∂ω/∂r_X.
/// We wrap (ω − ω₀) into [−π, π] before applying.
pub fn add_improper_forces(
    positions: &[Vec3],
    graph: &TopologyGraph,
    ff: &ForceField,
    atom_types: &[AtomType],
    forces: &mut [Vec3],
) {
    for imp in &graph.impropers {
        let Some(p) = ff.improper(
            atom_types[imp.a], atom_types[imp.b], atom_types[imp.c], atom_types[imp.d],
        ) else { continue };
        let Some((da, db, dc, dd, omega)) = dihedral_gradient(
            positions[imp.a], positions[imp.b], positions[imp.c], positions[imp.d],
        ) else { continue };
        let omega0 = deg_to_rad(p.psi0_deg);
        let mut domega = omega - omega0;
        while domega > std::f64::consts::PI {
            domega -= 2.0 * std::f64::consts::PI;
        }
        while domega < -std::f64::consts::PI {
            domega += 2.0 * std::f64::consts::PI;
        }
        let dvdomega_kcal = 2.0 * p.k * domega;
        let dvdomega = kcal_to_kj(dvdomega_kcal);
        forces[imp.a] -= da * dvdomega;
        forces[imp.b] -= db * dvdomega;
        forces[imp.c] -= dc * dvdomega;
        forces[imp.d] -= dd * dvdomega;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bonded::{
        angle_energy, bond_energy, dihedral_energy, improper_energy, BondedBreakdown,
    };
    use chem::{standard_ff, AminoAcid};
    use geom::{build_extended_chain, build_topology_graph};

    /// Compare an analytical force component to a central-difference
    /// numerical gradient of the same energy expression.
    #[allow(clippy::too_many_arguments)]
    fn finite_diff_check<F>(
        s: &geom::Structure,
        analytical: &[Vec3],
        atom_idx: usize,
        axis: usize,
        ff: &chem::ForceField,
        graph: &TopologyGraph,
        atom_types: &[AtomType],
        eps: f64,
        energy_fn: F,
        tol: f64,
        label: &str,
    ) where
        F: Fn(
            &geom::Structure,
            &TopologyGraph,
            &chem::ForceField,
            &[AtomType],
            &mut BondedBreakdown,
        ) -> f64,
    {
        let mut s_plus = s.clone();
        let mut s_minus = s.clone();
        bump(&mut s_plus, atom_idx, axis, eps);
        bump(&mut s_minus, atom_idx, axis, -eps);
        let mut br = BondedBreakdown::default();
        let e_plus = energy_fn(&s_plus, graph, ff, atom_types, &mut br);
        let mut br = BondedBreakdown::default();
        let e_minus = energy_fn(&s_minus, graph, ff, atom_types, &mut br);
        let numeric = -(e_plus - e_minus) / (2.0 * eps); // F = -dE/dx
        let an = analytical[atom_idx][axis];
        assert!(
            (an - numeric).abs() < tol,
            "{} atom {} axis {}: analytical={:.4}, numeric={:.4}",
            label, atom_idx, axis, an, numeric
        );
    }

    fn bump(s: &mut geom::Structure, atom_idx: usize, axis: usize, eps: f64) {
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

    #[test]
    fn bond_force_finite_difference() {
        let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala]).unwrap();
        let g = build_topology_graph(&s);
        let ff = standard_ff();
        let atom_types = build_atom_types(&s);
        let positions = flatten_positions(&s);
        let n = positions.len();
        let mut forces = vec![Vec3::zeros(); n];
        add_bond_forces(&positions, &g, ff, &atom_types, &mut forces);
        for i in 0..n.min(8) {
            for axis in 0..3 {
                finite_diff_check(&s, &forces, i, axis, ff, &g, &atom_types, 1e-5, bond_energy, 1e-2, "bond");
            }
        }
    }

    #[test]
    fn angle_force_finite_difference() {
        let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala]).unwrap();
        let g = build_topology_graph(&s);
        let ff = standard_ff();
        let atom_types = build_atom_types(&s);
        let positions = flatten_positions(&s);
        let n = positions.len();
        let mut forces = vec![Vec3::zeros(); n];
        add_angle_forces(&positions, &g, ff, &atom_types, &mut forces);
        for i in 0..n.min(8) {
            for axis in 0..3 {
                finite_diff_check(&s, &forces, i, axis, ff, &g, &atom_types, 1e-5, angle_energy, 1e-1, "angle");
            }
        }
    }

    #[test]
    fn dihedral_gradient_isolated_check() {
        // 4 atoms, compute φ analytically and numerically for each axis of each atom,
        // verify dihedral_gradient matches.
        let pa = Vec3::new(0.5, -0.7, 0.0);
        let pb = Vec3::new(1.0, 0.0, 0.3);
        let pc = Vec3::new(2.1, 0.4, -0.2);
        let pd = Vec3::new(2.7, 1.2, 0.5);
        let (da, db, dc, dd, _phi) = dihedral_gradient(pa, pb, pc, pd).unwrap();
        let analytical = [da, db, dc, dd];

        let eps = 1e-5;
        for atom_idx in 0..4 {
            for axis in 0..3 {
                let mut p = [pa, pb, pc, pd];
                p[atom_idx][axis] += eps;
                let (_, _, _, _, phi_plus) =
                    dihedral_gradient(p[0], p[1], p[2], p[3]).unwrap();
                let mut p = [pa, pb, pc, pd];
                p[atom_idx][axis] -= eps;
                let (_, _, _, _, phi_minus) =
                    dihedral_gradient(p[0], p[1], p[2], p[3]).unwrap();
                let mut numeric = (phi_plus - phi_minus) / (2.0 * eps);
                // Wrap differences > π to handle the atan2 branch cut.
                if numeric > std::f64::consts::PI {
                    numeric -= 2.0 * std::f64::consts::PI;
                }
                if numeric < -std::f64::consts::PI {
                    numeric += 2.0 * std::f64::consts::PI;
                }
                let an = analytical[atom_idx][axis];
                assert!(
                    (an - numeric).abs() < 1e-3,
                    "dihedral gradient mismatch atom {} axis {}: analytical={:.4}, numeric={:.4}",
                    atom_idx, axis, an, numeric
                );
            }
        }
    }

    #[test]
    fn dihedral_force_finite_difference() {
        let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala, AminoAcid::Ala]).unwrap();
        let g = build_topology_graph(&s);
        let ff = standard_ff();
        let atom_types = build_atom_types(&s);
        let positions = flatten_positions(&s);
        let n = positions.len();
        let mut forces = vec![Vec3::zeros(); n];
        add_dihedral_forces(&positions, &g, ff, &atom_types, &mut forces);
        for i in 0..n.min(8) {
            for axis in 0..3 {
                finite_diff_check(&s, &forces, i, axis, ff, &g, &atom_types, 1e-5, dihedral_energy, 1e-1, "dihedral");
            }
        }
    }

    #[test]
    fn improper_force_finite_difference() {
        let s = build_extended_chain(&[AminoAcid::Asn, AminoAcid::Ala]).unwrap();
        let g = build_topology_graph(&s);
        let ff = standard_ff();
        let atom_types = build_atom_types(&s);
        let positions = flatten_positions(&s);
        let n = positions.len();
        let mut forces = vec![Vec3::zeros(); n];
        add_improper_forces(&positions, &g, ff, &atom_types, &mut forces);
        for i in 0..n.min(8) {
            for axis in 0..3 {
                finite_diff_check(&s, &forces, i, axis, ff, &g, &atom_types, 1e-5, improper_energy, 1e-1, "improper");
            }
        }
    }

    #[test]
    fn bond_force_zero_at_equilibrium() {
        // Two atoms exactly at CHARMM r₀ for some bond pair: force should be zero.
        // Use the N-CA bond (NH1-CT1) with r₀ = 1.430 Å.
        use geom::structure::{PlacedAtom, PlacedResidue};
        use chem::Element;
        let r0 = 1.430;
        let s = geom::Structure {
            residues: vec![PlacedResidue {
                aa: AminoAcid::Ala,
                atoms: vec![
                    PlacedAtom { name: "N", element: Element::N, position: Vec3::zeros() },
                    PlacedAtom { name: "CA", element: Element::C, position: Vec3::new(r0, 0.0, 0.0) },
                ],
            }],
        };
        let g = build_topology_graph(&s);
        let ff = standard_ff();
        let atom_types = build_atom_types(&s);
        let positions = flatten_positions(&s);
        let mut forces = vec![Vec3::zeros(); positions.len()];
        add_bond_forces(&positions, &g, ff, &atom_types, &mut forces);
        // Force magnitudes should be near zero (within 1 kJ/mol/Å is plenty).
        for f in &forces {
            assert!(f.norm() < 1.0, "force {} should be ~0 at equilibrium", f);
        }
    }

    #[test]
    fn newtons_third_law_for_bonds() {
        // The total force across all atoms in a single Ala₂ chain due to
        // bonded terms should sum to (approximately) zero — bonded forces
        // are internal and cancel.
        let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala]).unwrap();
        let g = build_topology_graph(&s);
        let ff = standard_ff();
        let atom_types = build_atom_types(&s);
        let positions = flatten_positions(&s);
        let n = positions.len();
        let mut forces = vec![Vec3::zeros(); n];
        add_bond_forces(&positions, &g, ff, &atom_types, &mut forces);
        add_angle_forces(&positions, &g, ff, &atom_types, &mut forces);
        add_dihedral_forces(&positions, &g, ff, &atom_types, &mut forces);
        add_improper_forces(&positions, &g, ff, &atom_types, &mut forces);
        let total: Vec3 = forces.iter().fold(Vec3::zeros(), |acc, f| acc + f);
        assert!(
            total.norm() < 1e-6,
            "net bonded force should be zero, got {} kJ/mol/Å", total.norm()
        );
    }
}
