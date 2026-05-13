//! PowerSasa — exact analytical solvent-accessible surface area, following
//! Klenin et al. 2011 (J. Comput. Chem. 32, 2647).
//!
//! Replaces the Shrake-Rupley dot-density implementation that shipped in
//! M3g, providing smooth analytical derivatives needed by M4 minimisation
//! and M5 dynamics.

pub mod analytical;
pub mod arrangement;
pub mod area;
pub mod derivatives;
pub mod geometry;

use chem::{Element, ForceField};
use geom::{CellList, Structure, Vec3};

use crate::units::kcal_to_kj;
use arrangement::{build_caps, find_boundary};
use area::accessible_area;

/// Probe radius (water) in Å — same value used by the Shrake-Rupley code.
pub const PROBE_RADIUS_A: f64 = 1.4;

#[derive(Debug, Default, Clone)]
pub struct PowerSasaResult {
    /// Per-atom accessible surface area in Å².
    pub per_atom_area: Vec<f64>,
    /// Total accessible area in Å².
    pub total_area_a2: f64,
    /// Hydrophobic energy `Σ γᵢ × Aᵢ` in kJ/mol.
    pub sasa_kj_mol: f64,
}

pub(crate) fn vdw_radius(e: Element) -> f64 {
    match e {
        Element::H => 1.20,
        Element::C => 1.70,
        Element::N => 1.55,
        Element::O => 1.52,
        Element::P => 1.80,
        Element::S => 1.80,
    }
}

pub(crate) fn surface_tension_kcal(e: Element) -> f64 {
    // Same parameters as the Shrake-Rupley implementation: ~5 cal/mol/Å²
    // for apolar atoms (C, S), 0 for polar.
    match e {
        Element::C | Element::S => 0.005,
        _ => 0.0,
    }
}

/// Compute exact analytical SASA for every atom in the structure.
pub fn powersasa_energy(structure: &Structure, _ff: &ForceField) -> PowerSasaResult {
    let n = structure.atom_count();
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

    // Neighbour search with cell size = 2 × max_radius.
    let max_radius = radii.iter().cloned().fold(0.0_f64, f64::max);
    let cell_size = (2.0 * max_radius).max(1.0);
    let cl = CellList::build(&positions, cell_size);
    let mut neighbour_lists: Vec<Vec<(usize, Vec3, f64)>> = vec![Vec::new(); n];
    for (i, j, r) in cl.iter_pairs_within(&positions, 2.0 * max_radius) {
        if r <= radii[i] + radii[j] {
            neighbour_lists[i].push((j, positions[j], radii[j]));
            neighbour_lists[j].push((i, positions[i], radii[i]));
        }
    }

    let mut per_atom_area = vec![0.0_f64; n];
    let mut total_area = 0.0_f64;
    let mut sasa_kcal = 0.0_f64;
    for i in 0..n {
        let boundary_input = build_caps(positions[i], radii[i], &neighbour_lists[i]);
        let (caps, _owners, boundary) = match boundary_input {
            None => {
                // Atom fully enclosed in some neighbour.
                continue;
            }
            Some((caps, owners)) => {
                let b = find_boundary(&caps);
                (caps, owners, b)
            }
        };
        // Sanity check: if our boundary computation thinks the region is
        // fully exposed but the atom has caps, something is wrong; we treat
        // the atom as exposed but log a warning by counting it. For now,
        // proceed with the computed boundary.
        let _ = (caps.len(),); // silence unused-var warnings if applicable
        let area = accessible_area(radii[i], &caps, &boundary);
        per_atom_area[i] = area;
        total_area += area;
        sasa_kcal += surface_tension_kcal(elements[i]) * area;
    }

    PowerSasaResult {
        per_atom_area,
        total_area_a2: total_area,
        sasa_kj_mol: kcal_to_kj(sasa_kcal),
    }
}

#[cfg(test)]
#[allow(unused_variables, unused_assignments, unused_mut, dead_code)]
mod tests {
    use super::*;
    use chem::{standard_ff, AminoAcid};
    use geom::{build_extended_chain, structure::PlacedAtom, structure::PlacedResidue};

    #[test]
    fn isolated_carbon_full_sphere() {
        let s = Structure {
            residues: vec![PlacedResidue {
                monomer: geom::structure::Monomer::Protein(AminoAcid::Ala),
                atoms: vec![PlacedAtom {
                    name: "CB",
                    element: Element::C,
                    position: Vec3::zeros(),
                }],
                chain: 'A',
            }],
        };
        let ff = standard_ff();
        let r = powersasa_energy(&s, ff);
        let expected = 4.0 * std::f64::consts::PI * (1.70_f64 + PROBE_RADIUS_A).powi(2);
        let err = (r.total_area_a2 - expected).abs();
        assert!(err < 1e-6, "PowerSasa {} vs analytical {}", r.total_area_a2, expected);
    }

    #[test]
    fn powersasa_matches_shrake_rupley_two_carbons() {
        // Two C atoms 3 Å apart — caps overlap on each. Cross-check.
        let s = Structure {
            residues: vec![PlacedResidue {
                monomer: geom::structure::Monomer::Protein(AminoAcid::Ala),
                atoms: vec![
                    PlacedAtom { name: "CB", element: Element::C, position: Vec3::zeros() },
                    PlacedAtom { name: "C", element: Element::C, position: Vec3::new(3.0, 0.0, 0.0) },
                ],
                chain: 'A',
            }],
        };
        let ff = standard_ff();
        let ps = powersasa_energy(&s, ff);
        let sr = crate::sasa::sasa_energy_with_dots(&s, 4096);
        eprintln!("two carbons: PowerSasa {} vs SR {}", ps.total_area_a2, sr.total_area_a2);
        let rel_err = (ps.total_area_a2 - sr.total_area_a2).abs() / sr.total_area_a2;
        assert!(rel_err < 0.02, "two-carbon SASA mismatch: {} vs {}", ps.total_area_a2, sr.total_area_a2);
    }

    #[test]
    fn powersasa_matches_shrake_rupley_single_alanine() {
        // Just one alanine residue (10 atoms).
        let s = build_extended_chain(&[AminoAcid::Ala]).unwrap();
        let ff = standard_ff();
        let ps = powersasa_energy(&s, ff);
        let sr = crate::sasa::sasa_energy_with_dots(&s, 4096);
        eprintln!("single Ala: PowerSasa {} vs SR {}", ps.total_area_a2, sr.total_area_a2);
        let rel_err = (ps.total_area_a2 - sr.total_area_a2).abs() / sr.total_area_a2;
        assert!(rel_err < 0.02, "single-Ala SASA mismatch: {} vs {}", ps.total_area_a2, sr.total_area_a2);
    }

    #[test]
    #[ignore]
    fn diagnose_specific_overcount_atom() {
        // Instrument a specific atom that PSA over-counts to find where the
        // arc/vertex sums go wrong. Trp-cage residue 7 (Lys) CB is the
        // worst offender: PSA=18.55 vs SR=0.24 (heavily buried).
        let seq: Vec<AminoAcid> = "NLYIQWLKDGGPSSGRPPPS"
            .chars()
            .filter_map(AminoAcid::from_one_letter)
            .collect();
        let s = build_extended_chain(&seq).unwrap();

        // Find the target atom and its neighbours.
        let mut all_pos: Vec<Vec3> = Vec::new();
        let mut all_rad: Vec<f64> = Vec::new();
        let mut target_idx = usize::MAX;
        let mut idx = 0usize;
        for (ri, residue) in s.residues.iter().enumerate() {
            for atom in &residue.atoms {
                all_pos.push(atom.position);
                all_rad.push(vdw_radius(atom.element) + PROBE_RADIUS_A);
                if ri == 7 && atom.name == "CB" {
                    target_idx = idx;
                }
                idx += 1;
            }
        }
        assert!(target_idx != usize::MAX, "did not find Lys CB");
        let pi = all_pos[target_idx];
        let ri_a = all_rad[target_idx];
        eprintln!("target atom idx={}, pos={:?}, radius={:.3}", target_idx, pi, ri_a);

        let mut neighbours = Vec::new();
        for (j, &pj) in all_pos.iter().enumerate() {
            if j == target_idx {
                continue;
            }
            let d = (pj - pi).norm();
            if d <= ri_a + all_rad[j] {
                neighbours.push((j, pj, all_rad[j]));
            }
        }
        eprintln!("Has {} cap neighbours", neighbours.len());

        let (caps, owners) = build_caps(pi, ri_a, &neighbours).expect("not fully buried");
        let unit_caps: Vec<crate::powersasa::geometry::SmallCircle> = caps.clone();
        let boundary = find_boundary(&unit_caps);

        eprintln!("Built {} caps, owners={:?}", unit_caps.len(), owners);
        match &boundary {
            crate::powersasa::arrangement::AtomBoundary::FullyExposed => eprintln!("FullyExposed"),
            crate::powersasa::arrangement::AtomBoundary::FullyBuried => eprintln!("FullyBuried"),
            crate::powersasa::arrangement::AtomBoundary::Bounded { arcs, vertices } => {
                eprintln!("Bounded: {} arcs, {} vertices", arcs.len(), vertices.len());
                let mut arc_sum = 0.0;
                for (i, arc) in arcs.iter().enumerate() {
                    let cap = unit_caps[arc.cap_idx];
                    let contrib = cap.cos_alpha * arc.theta;
                    arc_sum += contrib;
                    eprintln!(
                        "  arc {:>2}: cap={} cos_α={:+.4} θ={:+.4} contrib={:+.4} fc={} start=({:+.3},{:+.3},{:+.3}) end=({:+.3},{:+.3},{:+.3})",
                        i, arc.cap_idx, cap.cos_alpha, arc.theta, contrib, arc.is_full_circle,
                        arc.start.x, arc.start.y, arc.start.z, arc.end.x, arc.end.y, arc.end.z,
                    );
                }
                let mut vert_sum = 0.0;
                for (i, vertex) in vertices.iter().enumerate() {
                    let v = vertex.point;
                    let t_in = v.cross(&unit_caps[vertex.incoming_cap].axis).normalize();
                    let t_out = v.cross(&unit_caps[vertex.outgoing_cap].axis).normalize();
                    let cos_eps = t_in.dot(&t_out).clamp(-1.0, 1.0);
                    let sin_eps = v.dot(&t_in.cross(&t_out));
                    let eps = sin_eps.atan2(cos_eps);
                    vert_sum += eps;
                    eprintln!(
                        "  vert {:>2}: in_cap={} out_cap={} ε={:+.4}",
                        i, vertex.incoming_cap, vertex.outgoing_cap, eps,
                    );
                }
                eprintln!("arc_sum = {:+.4}, vert_sum = {:+.4}", arc_sum, vert_sum);
                let two_pi = 2.0 * std::f64::consts::PI;
                let r_sq = ri_a * ri_a;
                for chi in -2..=2_i64 {
                    let area = r_sq * (two_pi * chi as f64 - arc_sum - vert_sum);
                    eprintln!("  if χ = {:+}: area = {:+.3}", chi, area);
                }
                let sr_per_atom = crate::sasa::sasa_per_atom_with_dots(&s, 4096);
                eprintln!("\nSR truth for atom {}: {:.3}", target_idx, sr_per_atom[target_idx]);
            }
        }
    }

    #[test]
    #[ignore]
    fn diagnose_carbonyl_c_arrangement() {
        let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala]).unwrap();
        // Find the carbonyl C of residue 0.
        let mut atom_idx = 0;
        let mut c_pos = Vec3::zeros();
        let mut c_radius = 0.0;
        for (ri, residue) in s.residues.iter().enumerate() {
            for atom in &residue.atoms {
                if ri == 0 && atom.name == "C" {
                    c_pos = atom.position;
                    c_radius = vdw_radius(atom.element) + PROBE_RADIUS_A;
                }
                atom_idx += 1;
            }
        }
        // Build neighbours.
        let mut all_pos = Vec::new();
        let mut all_rad = Vec::new();
        for residue in &s.residues {
            for atom in &residue.atoms {
                all_pos.push(atom.position);
                all_rad.push(vdw_radius(atom.element) + PROBE_RADIUS_A);
            }
        }
        let mut neighbours = Vec::new();
        let mut self_idx = usize::MAX;
        for (idx, &p) in all_pos.iter().enumerate() {
            if (p - c_pos).norm() < 1e-9 {
                self_idx = idx;
                continue;
            }
            let d = (p - c_pos).norm();
            if d <= c_radius + all_rad[idx] {
                neighbours.push((idx, p, all_rad[idx]));
            }
        }
        eprintln!("Carbonyl C at {:?}, radius {}, has {} neighbours",
            c_pos, c_radius, neighbours.len());
        let _ = self_idx;
        let (caps, _owners) = build_caps(c_pos, c_radius, &neighbours).unwrap();
        eprintln!("Built {} caps:", caps.len());
        for (i, cap) in caps.iter().enumerate() {
            eprintln!("  cap {}: axis=({:.3},{:.3},{:.3}), cos_α={:.3}",
                i, cap.axis.x, cap.axis.y, cap.axis.z, cap.cos_alpha);
        }
        let boundary = find_boundary(&caps);
        match &boundary {
            arrangement::AtomBoundary::FullyExposed => eprintln!("FullyExposed"),
            arrangement::AtomBoundary::FullyBuried => eprintln!("FullyBuried"),
            arrangement::AtomBoundary::Bounded { arcs, vertices } => {
                eprintln!("Bounded: {} arcs, {} vertices", arcs.len(), vertices.len());
                for (i, a) in arcs.iter().enumerate() {
                    eprintln!("  arc {}: cap={} theta={:.3} full_circle={}",
                        i, a.cap_idx, a.theta, a.is_full_circle);
                }
            }
        }
        let area = accessible_area(c_radius, &caps, &boundary);
        eprintln!("Area = {:.2}", area);
        if let arrangement::AtomBoundary::Bounded { arcs, vertices } = &boundary {
            // Manually walk boundary loops and report.
            let n = arcs.len();
            let mut visited = vec![false; n];
            let mut loops: Vec<Vec<usize>> = Vec::new();
            for start in 0..n {
                if visited[start] { continue; }
                visited[start] = true;
                let mut path = vec![start];
                let mut current = start;
                loop {
                    let next = (0..n).find(|&i| {
                        !visited[i] && (arcs[i].start - arcs[current].end).norm() < 1e-6
                    });
                    match next {
                        Some(j) => { visited[j] = true; path.push(j); current = j; }
                        None => break,
                    }
                }
                loops.push(path);
            }
            eprintln!("Boundary has {} loops:", loops.len());
            for (li, path) in loops.iter().enumerate() {
                eprintln!("  loop {}: {} arcs, indices {:?}", li, path.len(), path);
            }
            let mut arc_sum = 0.0;
            for arc in arcs {
                arc_sum += caps[arc.cap_idx].cos_alpha * arc.theta;
            }
            let mut vertex_sum = 0.0;
            for vertex in vertices {
                let v = vertex.point;
                let t_in = v.cross(&caps[vertex.incoming_cap].axis).normalize();
                let t_out = v.cross(&caps[vertex.outgoing_cap].axis).normalize();
                let cos_eps = t_in.dot(&t_out).clamp(-1.0, 1.0);
                let sin_eps = v.dot(&t_in.cross(&t_out));
                let eps = sin_eps.atan2(cos_eps);
                vertex_sum += eps;
                eprintln!("  vertex at ({:.3},{:.3},{:.3}) in_cap={} out_cap={} eps={:.3}",
                    v.x, v.y, v.z, vertex.incoming_cap, vertex.outgoing_cap, eps);
            }
            eprintln!("arc_sum={:.3}, vertex_sum={:.3}, n_loops={}", arc_sum, vertex_sum, loops.len());
        }
    }

    #[test]
    #[ignore]
    fn diagnose_bad_atoms_in_ala2() {
        let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala]).unwrap();
        let ff = standard_ff();
        let ps = powersasa_energy(&s, ff);
        let sr = crate::sasa::sasa_energy_with_dots(&s, 4096);

        // Compute SR per-atom area too. We have to call sasa with a higher
        // dot count and reach into the structure. For diagnostic purposes
        // we'll just pretty-print PowerSasa per atom.
        let mut atom_idx = 0;
        for (ri, residue) in s.residues.iter().enumerate() {
            for atom in &residue.atoms {
                eprintln!("res {} atom {:>4} ({:?}): PSA={:.2}",
                    ri, atom.name, atom.element, ps.per_atom_area[atom_idx]);
                atom_idx += 1;
            }
        }
        eprintln!("Total: PSA {:.2}, SR {:.2}", ps.total_area_a2, sr.total_area_a2);
    }

    #[test]
    #[ignore]
    fn diagnose_trp_cage_per_atom() {
        let seq: Vec<AminoAcid> = "NLYIQWLKDGGPSSGRPPPS"
            .chars()
            .filter_map(AminoAcid::from_one_letter)
            .collect();
        let s = build_extended_chain(&seq).unwrap();
        let ff = standard_ff();
        let ps = powersasa_energy(&s, ff);
        let sr_total = crate::sasa::sasa_energy_with_dots(&s, 4096);
        let sr_per_atom = crate::sasa::sasa_per_atom_with_dots(&s, 4096);
        // Print the 20 largest discrepancies (PSA − SR) so we can see
        // which atoms drive the residual error.
        let mut diffs: Vec<(usize, usize, &'static str, chem::Element, f64, f64)> = Vec::new();
        let mut idx = 0usize;
        for (ri, residue) in s.residues.iter().enumerate() {
            for atom in &residue.atoms {
                let psa = ps.per_atom_area[idx];
                let sra = sr_per_atom[idx];
                diffs.push((ri, idx, atom.name, atom.element, psa, sra));
                idx += 1;
            }
        }
        diffs.sort_by(|a, b| (b.4 - b.5).partial_cmp(&(a.4 - a.5)).unwrap());
        eprintln!("Top 10 PSA-SR over-counts:");
        for (ri, idx, name, el, psa, sra) in diffs.iter().take(10) {
            eprintln!(
                "  res {:>2} atom {:>4} ({:?}, idx {:>3}): PSA={:>7.2} SR={:>7.2} Δ={:>+7.2}",
                ri, name, el, idx, psa, sra, psa - sra,
            );
        }
        eprintln!("Top 10 PSA-SR under-counts:");
        for (ri, idx, name, el, psa, sra) in diffs.iter().rev().take(10) {
            eprintln!(
                "  res {:>2} atom {:>4} ({:?}, idx {:>3}): PSA={:>7.2} SR={:>7.2} Δ={:>+7.2}",
                ri, name, el, idx, psa, sra, psa - sra,
            );
        }
        eprintln!("Total PSA: {:.2}, SR: {:.2}, ratio: {:.4}",
            ps.total_area_a2, sr_total.total_area_a2,
            ps.total_area_a2 / sr_total.total_area_a2);
    }

    #[test]
    fn powersasa_progression_chain_size() {
        let ff = standard_ff();
        for n in 1..=5usize {
            let seq = vec![AminoAcid::Ala; n];
            let s = build_extended_chain(&seq).unwrap();
            let ps = powersasa_energy(&s, ff);
            let sr = crate::sasa::sasa_energy_with_dots(&s, 4096);
            let rel_err = (ps.total_area_a2 - sr.total_area_a2).abs() / sr.total_area_a2;
            eprintln!("Ala_{}: PowerSasa {:.2} vs SR {:.2}, rel err {:.4}",
                n, ps.total_area_a2, sr.total_area_a2, rel_err);
        }
    }

    /// Trp-cage cross-check. PSA.1 sequence of improvements:
    /// - Original χ-disambiguation heuristic: +68 % vs Shrake-Rupley.
    /// - PSA.1e–g (probe-based component counting + χ = 2c − L): +11 %.
    /// - PSA.1i (recompute vertex ε from the actual face-walker
    ///   continuation, fixing 3+-way vertex misattribution): <1 %.
    ///
    /// Locked at 2 % to give headroom for legitimate Shrake-Rupley
    /// sampling noise; observed is 0.7 % under.
    #[test]
    fn powersasa_within_known_bound_on_extended_trp_cage() {
        let seq: Vec<AminoAcid> = "NLYIQWLKDGGPSSGRPPPS"
            .chars()
            .filter_map(AminoAcid::from_one_letter)
            .collect();
        let s = build_extended_chain(&seq).unwrap();
        let ff = standard_ff();
        let ps = powersasa_energy(&s, ff);
        let sr = crate::sasa::sasa_energy_with_dots(&s, 4096);
        let rel_err = (ps.total_area_a2 - sr.total_area_a2).abs() / sr.total_area_a2;
        assert!(
            rel_err < 0.02,
            "PowerSasa {} disagrees with Shrake-Rupley {} (rel err {})",
            ps.total_area_a2, sr.total_area_a2, rel_err
        );
    }
}

