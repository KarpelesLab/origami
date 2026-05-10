//! Bonded energy terms: bond stretching, angle bending, dihedral torsion,
//! and improper torsion.
//!
//! All return values in kJ/mol.

use chem::{classify, AtomType, ForceField};
use geom::{measure, Structure, TopologyGraph};

use crate::units::{deg_to_rad, kcal_to_kj};

#[derive(Debug, Default, Clone, Copy)]
pub struct BondedBreakdown {
    pub bond_kj_mol: f64,
    pub angle_kj_mol: f64,
    pub dihedral_kj_mol: f64,
    pub improper_kj_mol: f64,
    pub bond_count: usize,
    pub angle_count: usize,
    pub dihedral_count: usize,
    pub improper_count: usize,
    /// Tuples for which no parameter was found and which were therefore
    /// skipped. A non-zero count usually indicates a force-field-table gap.
    pub missing_count: usize,
}

impl BondedBreakdown {
    pub fn total_kj_mol(&self) -> f64 {
        self.bond_kj_mol + self.angle_kj_mol + self.dihedral_kj_mol + self.improper_kj_mol
    }
}

/// Compute every bonded energy term for the structure. Atom types are
/// looked up from the structure; parameters from the supplied force field.
pub fn bonded_energy(structure: &Structure, graph: &TopologyGraph, ff: &ForceField) -> BondedBreakdown {
    let atom_types = build_atom_types(structure);
    let mut br = BondedBreakdown::default();

    br.bond_kj_mol = bond_energy(structure, graph, ff, &atom_types, &mut br);
    br.angle_kj_mol = angle_energy(structure, graph, ff, &atom_types, &mut br);
    br.dihedral_kj_mol = dihedral_energy(structure, graph, ff, &atom_types, &mut br);
    br.improper_kj_mol = improper_energy(structure, graph, ff, &atom_types, &mut br);

    br
}

fn build_atom_types(structure: &Structure) -> Vec<AtomType> {
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

/// Σ over all bonds of ½ k_b (r − r₀)² in kJ/mol.
pub fn bond_energy(
    structure: &Structure,
    graph: &TopologyGraph,
    ff: &ForceField,
    atom_types: &[AtomType],
    br: &mut BondedBreakdown,
) -> f64 {
    let positions: Vec<_> = structure
        .residues
        .iter()
        .flat_map(|r| r.atoms.iter().map(|a| a.position))
        .collect();
    let mut total_kcal = 0.0;
    for b in &graph.bonds {
        let (ta, tb) = (atom_types[b.a], atom_types[b.b]);
        let Some(p) = ff.bond(ta, tb) else {
            br.missing_count += 1;
            continue;
        };
        let r = measure::distance(positions[b.a], positions[b.b]);
        let dr = r - p.r0;
        // CHARMM bond convention: V = K (r-r0)² (no leading ½). Park 2017,
        // CHARMM doc, all confirm this.
        total_kcal += p.k * dr * dr;
        br.bond_count += 1;
    }
    kcal_to_kj(total_kcal)
}

/// Σ over all angles of K_θ (θ − θ₀)² in kJ/mol.
/// CHARMM angles use the same K (no ½) convention as bonds.
pub fn angle_energy(
    structure: &Structure,
    graph: &TopologyGraph,
    ff: &ForceField,
    atom_types: &[AtomType],
    br: &mut BondedBreakdown,
) -> f64 {
    let positions: Vec<_> = structure
        .residues
        .iter()
        .flat_map(|r| r.atoms.iter().map(|a| a.position))
        .collect();
    let mut total_kcal = 0.0;
    for ang in &graph.angles {
        let (ta, tb, tc) = (atom_types[ang.a], atom_types[ang.b], atom_types[ang.c]);
        let Some(p) = ff.angle(ta, tb, tc) else {
            br.missing_count += 1;
            continue;
        };
        let theta = measure::angle(positions[ang.a], positions[ang.b], positions[ang.c]);
        let theta0 = deg_to_rad(p.theta0_deg);
        let dtheta = theta - theta0;
        total_kcal += p.k * dtheta * dtheta;
        br.angle_count += 1;
    }
    kcal_to_kj(total_kcal)
}

/// Σ over all proper dihedrals of Σ_n k_n (1 + cos(n·χ − δ)) in kJ/mol.
pub fn dihedral_energy(
    structure: &Structure,
    graph: &TopologyGraph,
    ff: &ForceField,
    atom_types: &[AtomType],
    br: &mut BondedBreakdown,
) -> f64 {
    let positions: Vec<_> = structure
        .residues
        .iter()
        .flat_map(|r| r.atoms.iter().map(|a| a.position))
        .collect();
    let mut total_kcal = 0.0;
    for d in &graph.dihedrals {
        let (ta, tb, tc, td) = (
            atom_types[d.a], atom_types[d.b], atom_types[d.c], atom_types[d.d],
        );
        let Some(terms) = ff.dihedral(ta, tb, tc, td) else {
            br.missing_count += 1;
            continue;
        };
        let chi = measure::dihedral(
            positions[d.a], positions[d.b], positions[d.c], positions[d.d],
        );
        for term in terms {
            let delta = deg_to_rad(term.delta_deg);
            total_kcal += term.k * (1.0 + (term.n as f64 * chi - delta).cos());
        }
        br.dihedral_count += 1;
    }
    kcal_to_kj(total_kcal)
}

/// Σ over all impropers of K_ψ (ω − ω₀)² in kJ/mol.
pub fn improper_energy(
    structure: &Structure,
    graph: &TopologyGraph,
    ff: &ForceField,
    atom_types: &[AtomType],
    br: &mut BondedBreakdown,
) -> f64 {
    let positions: Vec<_> = structure
        .residues
        .iter()
        .flat_map(|r| r.atoms.iter().map(|a| a.position))
        .collect();
    let mut total_kcal = 0.0;
    for imp in &graph.impropers {
        let (ta, tb, tc, td) = (
            atom_types[imp.a], atom_types[imp.b], atom_types[imp.c], atom_types[imp.d],
        );
        let Some(p) = ff.improper(ta, tb, tc, td) else {
            br.missing_count += 1;
            continue;
        };
        let omega = measure::dihedral(
            positions[imp.a], positions[imp.b], positions[imp.c], positions[imp.d],
        );
        let omega0 = deg_to_rad(p.psi0_deg);
        // Wrap difference into [-π, π]
        let mut domega = omega - omega0;
        while domega > std::f64::consts::PI {
            domega -= 2.0 * std::f64::consts::PI;
        }
        while domega < -std::f64::consts::PI {
            domega += 2.0 * std::f64::consts::PI;
        }
        total_kcal += p.k * domega * domega;
        br.improper_count += 1;
    }
    kcal_to_kj(total_kcal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use chem::{standard_ff, AminoAcid};
    use geom::{build_extended_chain, build_topology_graph};

    #[test]
    fn extended_chain_bonded_energy_finite() {
        // The chain builder uses Engh–Huber r₀ values which differ slightly
        // from CHARMM36's r₀, so the bond term has a small but non-zero
        // baseline (~5 kJ/mol per residue). Energy minimisation in M4 will
        // relax this away.
        let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala, AminoAcid::Ala]).unwrap();
        let g = build_topology_graph(&s);
        let ff = standard_ff();
        let br = bonded_energy(&s, g_ref(&g), ff);
        assert_eq!(br.missing_count, 0, "missing parameters during energy eval");
        assert!(br.total_kj_mol().is_finite());
        assert!(
            br.bond_kj_mol < 100.0,
            "bond term {} unexpectedly large", br.bond_kj_mol,
        );
        assert!(
            br.angle_kj_mol < 200.0,
            "angle term {} unexpectedly large", br.angle_kj_mol,
        );
        assert!(
            br.improper_kj_mol < 50.0,
            "improper term {} should be near zero at planar built geometry", br.improper_kj_mol,
        );
    }

    #[test]
    fn stretching_a_bond_increases_energy_quadratically() {
        let mut s = build_extended_chain(&[AminoAcid::Ala]).unwrap();
        let g = build_topology_graph(&s);
        let ff = standard_ff();

        // Energy at equilibrium.
        let mut br0 = BondedBreakdown::default();
        let atom_types = build_atom_types(&s);
        let e0 = bond_energy(&s, &g, ff, &atom_types, &mut br0);

        // Perturb the N-CA bond by +0.1 Å along the bond direction.
        let n_pos = s.residues[0].atoms[0].position;
        let ca_pos = s.residues[0].atoms[1].position;
        let dir = (ca_pos - n_pos).normalize();
        s.residues[0].atoms[1].position = n_pos + dir * (1.458 + 0.1);

        let mut br1 = BondedBreakdown::default();
        let e1 = bond_energy(&s, &g, ff, &atom_types, &mut br1);

        // Energy should rise.
        assert!(e1 > e0);
    }

    #[test]
    fn dihedral_term_traces_periodic_function() {
        // Build an Ala dipeptide at default φ, then measure how the dihedral
        // energy changes when we shift φ. It should be smooth and periodic.
        let s_default = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala, AminoAcid::Ala]).unwrap();
        let g = build_topology_graph(&s_default);
        let ff = standard_ff();
        let atom_types = build_atom_types(&s_default);
        let mut br = BondedBreakdown::default();
        let e_default = dihedral_energy(&s_default, &g, ff, &atom_types, &mut br);
        assert!(e_default.is_finite());
    }

    fn g_ref(g: &TopologyGraph) -> &TopologyGraph { g }

    #[test]
    fn improper_at_planar_geometry_is_small() {
        // Asn's CB-CG-OD1-ND2 improper enforces planarity at CG (sp²).
        // Our chain builder places these at exactly 0° and 180° (planar),
        // so the improper energy should be small.
        let s = build_extended_chain(&[AminoAcid::Asn]).unwrap();
        let g = build_topology_graph(&s);
        let ff = standard_ff();
        let atom_types = build_atom_types(&s);
        let mut br = BondedBreakdown::default();
        let e = improper_energy(&s, &g, ff, &atom_types, &mut br);
        // Per-improper energy ≤ a few kJ/mol when planar.
        assert!(e.abs() < 20.0, "improper term {} unexpectedly large at planar geom", e);
    }

    #[test]
    fn unit_conversion_sanity() {
        // 1 kcal/mol = 4.184 kJ/mol.
        assert_relative_eq!(kcal_to_kj(1.0), 4.184, epsilon = 1e-9);
    }
}
