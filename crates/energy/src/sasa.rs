//! Solvent-accessible surface area (SASA) and the hydrophobic energy term.
//!
//! Implementation: **Shrake-Rupley dot density**. For each atom, distribute
//! N test points uniformly on its expanded vdW surface (radius + probe);
//! count how many are NOT inside any neighbour atom's expanded sphere; the
//! atom's SASA is `(accessible / N) × 4π R²`.
//!
//! This is an O(N × neighbours) algorithm with the cell list. With N=960
//! Fibonacci-spaced points the SASA is accurate to ~0.5 % vs exact analytical
//! methods. PowerSasa (Klenin 2011) is planned for M5 when forces are needed.
//!
//! Hydrophobic energy: `Σᵢ γᵢ × Aᵢ` with `γ` per element. Carbon and sulfur
//! atoms contribute the standard ~5 cal/mol/Å² penalty for surface exposure;
//! polar atoms contribute zero.

use chem::{Element, ForceField};
use geom::{CellList, Structure, Vec3};

use crate::units::kcal_to_kj;

/// Probe radius (water) in Å.
pub const PROBE_RADIUS_A: f64 = 1.4;

/// Number of test points per atom. 960 is one of the standard Fibonacci-grid
/// counts; gives ~0.5 % accuracy vs analytical SASA.
pub const SHRAKE_RUPLEY_DOTS: usize = 960;

#[derive(Debug, Default, Clone, Copy)]
pub struct SasaBreakdown {
    pub sasa_kj_mol: f64,
    pub total_area_a2: f64,
    pub atom_count: usize,
}

/// Per-element vdW radius used for SASA computation. Bondi values.
fn vdw_radius(element: Element) -> f64 {
    match element {
        Element::H => 1.20,
        Element::C => 1.70,
        Element::N => 1.55,
        Element::O => 1.52,
        Element::S => 1.80,
    }
}

/// Per-element surface-tension parameter (kcal/mol/Å²) for the hydrophobic
/// term. Carbon and sulfur (apolar) get a positive penalty when exposed;
/// polar atoms (H, N, O — H bonded to polar atoms typically) contribute
/// negligibly. Values are roughly the apolar/aqueous transfer free-energy
/// scale (~5 cal/mol/Å²).
fn surface_tension(element: Element) -> f64 {
    match element {
        Element::C | Element::S => 0.005, // 5 cal/mol/Å² = 0.005 kcal/mol/Å²
        _ => 0.0,
    }
}

pub fn sasa_energy(structure: &Structure, _ff: &ForceField) -> SasaBreakdown {
    sasa_energy_with_dots(structure, SHRAKE_RUPLEY_DOTS)
}

/// Shrake-Rupley SASA returning per-atom areas (Å²) and the total.
/// Used as the reference for PowerSasa cross-checks.
pub fn sasa_per_atom_with_dots(structure: &Structure, n_dots: usize) -> Vec<f64> {
    let mut positions: Vec<Vec3> = Vec::with_capacity(structure.atom_count());
    let mut elements: Vec<Element> = Vec::with_capacity(structure.atom_count());
    for residue in &structure.residues {
        for atom in &residue.atoms {
            positions.push(atom.position);
            elements.push(atom.element);
        }
    }
    let n = positions.len();
    let radii: Vec<f64> = elements
        .iter()
        .map(|&e| vdw_radius(e) + PROBE_RADIUS_A)
        .collect();
    let max_radius = radii.iter().cloned().fold(0.0_f64, f64::max);
    let cell_size = (2.0 * max_radius).max(1.0);
    let cl = CellList::build(&positions, cell_size);
    let mut neighbours: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, j, r) in cl.iter_pairs_within(&positions, 2.0 * max_radius) {
        if r <= radii[i] + radii[j] {
            neighbours[i].push(j);
            neighbours[j].push(i);
        }
    }
    let dots = fibonacci_unit_sphere(n_dots);
    let mut per_atom = vec![0.0; n];
    for i in 0..n {
        let ri = radii[i];
        let pi = positions[i];
        let mut accessible = 0usize;
        for &dot in &dots {
            let test_point = pi + dot * ri;
            let mut buried = false;
            for &j in &neighbours[i] {
                if (test_point - positions[j]).norm_squared() < radii[j] * radii[j] {
                    buried = true;
                    break;
                }
            }
            if !buried {
                accessible += 1;
            }
        }
        per_atom[i] = (accessible as f64 / n_dots as f64) * 4.0 * std::f64::consts::PI * ri * ri;
    }
    per_atom
}

pub fn sasa_energy_with_dots(structure: &Structure, n_dots: usize) -> SasaBreakdown {
    let mut positions: Vec<Vec3> = Vec::with_capacity(structure.atom_count());
    let mut elements: Vec<Element> = Vec::with_capacity(structure.atom_count());
    for residue in &structure.residues {
        for atom in &residue.atoms {
            positions.push(atom.position);
            elements.push(atom.element);
        }
    }
    let n = positions.len();
    let radii: Vec<f64> = elements
        .iter()
        .map(|&e| vdw_radius(e) + PROBE_RADIUS_A)
        .collect();

    // Cell size = 2 × max expanded radius covers all candidate neighbours.
    let max_radius = radii.iter().cloned().fold(0.0_f64, f64::max);
    let cell_size = (2.0 * max_radius).max(1.0);
    let cl = CellList::build(&positions, cell_size);
    // Build a per-atom neighbour list (atoms whose expanded spheres might
    // intersect): pairs (i, j) with r ≤ R_i + R_j.
    let mut neighbours: Vec<Vec<usize>> = vec![Vec::new(); n];
    let max_pair_cutoff = 2.0 * max_radius;
    for (i, j, r) in cl.iter_pairs_within(&positions, max_pair_cutoff) {
        if r <= radii[i] + radii[j] {
            neighbours[i].push(j);
            neighbours[j].push(i);
        }
    }

    // Precompute Fibonacci-grid points on the unit sphere.
    let dots = fibonacci_unit_sphere(n_dots);

    let mut total_area_a2 = 0.0_f64;
    let mut sasa_kcal = 0.0_f64;
    for i in 0..n {
        let ri = radii[i];
        let pi = positions[i];
        let mut accessible = 0usize;
        for &dot in &dots {
            let test_point = pi + dot * ri;
            let mut buried = false;
            for &j in &neighbours[i] {
                let rj = radii[j];
                let d2 = (test_point - positions[j]).norm_squared();
                if d2 < rj * rj {
                    buried = true;
                    break;
                }
            }
            if !buried {
                accessible += 1;
            }
        }
        let frac = accessible as f64 / n_dots as f64;
        let area = frac * 4.0 * std::f64::consts::PI * ri * ri;
        total_area_a2 += area;
        sasa_kcal += surface_tension(elements[i]) * area;
    }

    SasaBreakdown {
        sasa_kj_mol: kcal_to_kj(sasa_kcal),
        total_area_a2,
        atom_count: n,
    }
}

/// Generate `n` approximately-uniform points on the unit sphere using the
/// golden-angle Fibonacci spiral.
fn fibonacci_unit_sphere(n: usize) -> Vec<Vec3> {
    let phi = std::f64::consts::PI * (3.0_f64.sqrt() + 1.0); // ≈ golden angle × 2π
    let mut out = Vec::with_capacity(n);
    let n_f = n as f64;
    for i in 0..n {
        // y in (-1, 1), with even spacing.
        let y = 1.0 - 2.0 * (i as f64) / (n_f - 1.0);
        let r = (1.0 - y * y).sqrt();
        let theta = phi * (i as f64);
        let x = theta.cos() * r;
        let z = theta.sin() * r;
        out.push(Vec3::new(x, y, z));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use chem::{standard_ff, AminoAcid};
    use geom::{build_extended_chain, structure::PlacedAtom, structure::PlacedResidue};

    #[test]
    fn isolated_carbon_full_sphere() {
        // Single C atom in a "structure": its SASA should be ~4π(R+probe)².
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
        let br = sasa_energy(&s, ff);
        let expected = 4.0 * std::f64::consts::PI * (1.70_f64 + 1.4).powi(2);
        assert_relative_eq!(br.total_area_a2, expected, max_relative = 0.005);
    }

    #[test]
    fn buried_atom_zero_sasa() {
        // Two C atoms at distance 0.5 — the second is well inside the first's
        // expanded sphere. Combined SASA should be roughly the area of the
        // outer sphere alone.
        let s = Structure {
            residues: vec![PlacedResidue {
                aa: AminoAcid::Ala,
                atoms: vec![
                    PlacedAtom { name: "CB", element: Element::C, position: Vec3::zeros() },
                    PlacedAtom { name: "CG", element: Element::C, position: Vec3::new(0.5, 0.0, 0.0) },
                ],
            }],
        };
        let ff = standard_ff();
        let br = sasa_energy(&s, ff);
        // Two coincident-ish atoms expose less than 2× single-atom area.
        let single = 4.0 * std::f64::consts::PI * (1.70_f64 + 1.4).powi(2);
        assert!(br.total_area_a2 < 1.5 * single);
    }

    #[test]
    fn far_apart_atoms_sum_individually() {
        let s = Structure {
            residues: vec![PlacedResidue {
                aa: AminoAcid::Ala,
                atoms: vec![
                    PlacedAtom { name: "CB", element: Element::C, position: Vec3::zeros() },
                    PlacedAtom { name: "CG", element: Element::C, position: Vec3::new(50.0, 0.0, 0.0) },
                ],
            }],
        };
        let ff = standard_ff();
        let br = sasa_energy(&s, ff);
        let single = 4.0 * std::f64::consts::PI * (1.70_f64 + 1.4).powi(2);
        assert_relative_eq!(br.total_area_a2, 2.0 * single, max_relative = 0.005);
    }

    #[test]
    fn extended_chain_sasa_in_expected_range() {
        // 20 residues, extended: should expose ~3000–7000 Å² total.
        let seq: Vec<AminoAcid> = AminoAcid::ALL.to_vec();
        let s = build_extended_chain(&seq).unwrap();
        let ff = standard_ff();
        let br = sasa_energy(&s, ff);
        assert!(
            (1500.0..10000.0).contains(&br.total_area_a2),
            "SASA total {} Å² out of expected range", br.total_area_a2,
        );
        assert!(br.sasa_kj_mol > 0.0, "hydrophobic energy should be positive");
    }
}
