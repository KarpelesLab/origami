//! CHARMM CMAP backbone (φ, ψ) energy correction.
//!
//! For each internal protein residue `i` (i.e. residues 1..N-1 of a
//! chain, since CMAP needs C(i-1), N(i+1) atoms that don't exist
//! at chain termini), CHARMM adds a 2D correction term keyed by:
//!
//!   φ_i = dihedral C(i-1) – N(i) – CA(i) – C(i)
//!   ψ_i = dihedral N(i)   – CA(i) – C(i) – N(i+1)
//!
//! The correction value comes from a 24×24 grid of energies at
//! 15° spacing, indexed by (CA(i) atom type, N(i+1) atom type) — the
//! two types that distinguish CHARMM36m's 6 grids:
//!
//!   (CT1, NH1) — alanine-like + non-Pro next
//!   (CT1, N)   — alanine-like + Pro next
//!   (CP1, NH1) — Pro + non-Pro next
//!   (CP1, N)   — Pro + Pro next
//!   (CT2, NH1) — Gly + non-Pro next
//!   (CT2, N)   — Gly + Pro next
//!
//! Interpolation between grid points is **bilinear** for now. The
//! forces have small discontinuities at the 15° grid boundaries
//! (the gradient is piecewise constant in (φ, ψ)), which is fine for
//! minimisation and the resulting integrator noise is much smaller
//! than the noise from finite cutoffs. Bicubic interpolation with
//! continuous gradients can come later if the cutoff ringing turns
//! out to matter for a quantity we care about.

use chem::{AtomType, CmapGrid, ForceField, classify_atom};
use geom::{Structure, TopologyGraph};

use crate::units::kcal_to_kj;

/// Total CMAP correction energy across all internal residues of a
/// `Structure`, in kJ/mol.  Residues whose CA atom type isn't one of
/// the CMAP-keyed classes (CT1/CT2/CP1) are silently skipped — non-
/// protein residues, terminal residues missing C(i-1) or N(i+1), or
/// atom-classification failures.
pub fn cmap_energy(structure: &Structure, _graph: &TopologyGraph, ff: &ForceField) -> f64 {
    let mut total_kcal = 0.0;
    let (atom_index, atom_types) = build_atom_index_and_types(structure);
    for ri in 1..structure.residues.len().saturating_sub(1) {
        let Some((phi_rad, psi_rad, ca_type, next_n_type)) =
            residue_phi_psi(structure, ri, &atom_index, &atom_types)
        else {
            continue;
        };
        let Some(grid) = ff.cmap(ca_type, next_n_type) else {
            continue;
        };
        total_kcal += bilinear_lookup(grid, phi_rad, psi_rad);
    }
    kcal_to_kj(total_kcal)
}

/// Bilinear-interpolate the CMAP grid at (phi_rad, psi_rad).
/// Both angles are wrapped into [-180°, 180°) before lookup.
pub fn bilinear_lookup(grid: &CmapGrid, phi_rad: f64, psi_rad: f64) -> f64 {
    let (i, fx) = phi_to_grid(phi_rad);
    let (j, fy) = phi_to_grid(psi_rad);
    let g00 = grid.at(i, j);
    let g10 = grid.at(i + 1, j);
    let g01 = grid.at(i, j + 1);
    let g11 = grid.at(i + 1, j + 1);
    (1.0 - fx) * (1.0 - fy) * g00 + fx * (1.0 - fy) * g10 + (1.0 - fx) * fy * g01 + fx * fy * g11
}

/// Bilinear partial derivatives (∂E/∂φ, ∂E/∂ψ) at the same point.
/// The result is in **kcal/mol/rad** so it matches the energy units;
/// callers convert to kJ/mol/Å as part of the chain rule.
pub fn bilinear_derivatives(grid: &CmapGrid, phi_rad: f64, psi_rad: f64) -> (f64, f64) {
    let (i, fx) = phi_to_grid(phi_rad);
    let (j, fy) = phi_to_grid(psi_rad);
    let g00 = grid.at(i, j);
    let g10 = grid.at(i + 1, j);
    let g01 = grid.at(i, j + 1);
    let g11 = grid.at(i + 1, j + 1);
    // Per-unit-of-(fx,fy) derivative, then convert (fx, fy) units
    // (one cell = GRID_SPACING_DEG of angle) into rad.
    let de_dfx = (1.0 - fy) * (g10 - g00) + fy * (g11 - g01);
    let de_dfy = (1.0 - fx) * (g01 - g00) + fx * (g11 - g10);
    let cell_rad = CmapGrid::GRID_SPACING_DEG.to_radians();
    (de_dfx / cell_rad, de_dfy / cell_rad)
}

/// Returns (integer grid index in [0, GRID_SIZE), fractional part in [0, 1))
/// such that the angle corresponds to `index + frac` on the grid.
fn phi_to_grid(angle_rad: f64) -> (usize, f64) {
    let deg = wrap_180(angle_rad.to_degrees());
    // Angle ∈ [-180, 180); grid index 0 = -180°, spacing 15°.
    let cell = (deg + 180.0) / CmapGrid::GRID_SPACING_DEG;
    let i = (cell.floor() as isize).rem_euclid(CmapGrid::GRID_SIZE as isize) as usize;
    let fx = cell - cell.floor();
    (i, fx)
}

/// Wrap an angle in degrees into the half-open interval [-180, 180).
fn wrap_180(deg: f64) -> f64 {
    let mut x = (deg + 180.0).rem_euclid(360.0) - 180.0;
    if x == 180.0 {
        x = -180.0;
    }
    x
}

/// Build a fast (residue, atom_name) → global-atom-index map AND a
/// parallel atom-type vector keyed by global index.  Reused by the
/// force path.
pub(crate) fn build_atom_index_and_types(
    structure: &Structure,
) -> (
    Vec<std::collections::HashMap<&'static str, usize>>,
    Vec<AtomType>,
) {
    let mut atom_index: Vec<std::collections::HashMap<&'static str, usize>> =
        Vec::with_capacity(structure.residues.len());
    let mut atom_types: Vec<AtomType> = Vec::with_capacity(structure.atom_count());
    let mut global = 0usize;
    for residue in &structure.residues {
        let mut map = std::collections::HashMap::with_capacity(residue.atoms.len());
        for atom in &residue.atoms {
            map.insert(atom.name, global);
            let ty = classify_atom(residue.monomer, atom.name)
                .unwrap_or_else(|| panic!("unclassified atom {:?} {}", residue.monomer, atom.name));
            atom_types.push(ty);
            global += 1;
        }
        atom_index.push(map);
    }
    (atom_index, atom_types)
}

/// Resolve `(C(i-1), N(i), CA(i), C(i), N(i+1))` global indices and
/// compute (φ, ψ) for residue `ri`.  Returns `None` if any required
/// atom is missing (e.g. terminus, RNA residue, atypical residue).
///
/// **Sign convention**: φ and ψ are returned in the IUPAC convention
/// (the one CHARMM's CMAP grids are indexed by — `geom::measure::dihedral`
/// implements it directly).  Note that `forces_bonded::dihedral_gradient`
/// uses the *opposite* sign convention; the force path negates its
/// per-atom gradients before applying the chain rule.
pub(crate) fn residue_phi_psi(
    structure: &Structure,
    ri: usize,
    atom_index: &[std::collections::HashMap<&'static str, usize>],
    atom_types: &[AtomType],
) -> Option<(f64, f64, AtomType, AtomType)> {
    // Bail on RNA / non-protein residues; CMAP is protein-only.
    structure.residues[ri].monomer.as_amino_acid()?;
    structure
        .residues
        .get(ri.wrapping_sub(1))?
        .monomer
        .as_amino_acid()?;
    structure.residues.get(ri + 1)?.monomer.as_amino_acid()?;
    let prev_c = *atom_index[ri - 1].get("C")?;
    let cur_n = *atom_index[ri].get("N")?;
    let cur_ca = *atom_index[ri].get("CA")?;
    let cur_c = *atom_index[ri].get("C")?;
    let next_n = *atom_index[ri + 1].get("N")?;
    let positions = collect_positions(structure);
    let phi = geom::measure::dihedral(
        positions[prev_c],
        positions[cur_n],
        positions[cur_ca],
        positions[cur_c],
    );
    let psi = geom::measure::dihedral(
        positions[cur_n],
        positions[cur_ca],
        positions[cur_c],
        positions[next_n],
    );
    Some((phi, psi, atom_types[cur_ca], atom_types[next_n]))
}

fn collect_positions(structure: &Structure) -> Vec<geom::Vec3> {
    structure
        .residues
        .iter()
        .flat_map(|r| r.atoms.iter().map(|a| a.position))
        .collect()
}

/// Accumulate CMAP analytical forces into `forces` (kJ/mol/Å).
///
/// For each internal residue `i`, looks up the CMAP grid by
/// `(CA(i) type, N(i+1) type)`, computes (φ, ψ) and (∂E/∂φ, ∂E/∂ψ)
/// from bilinear interpolation, then chain-rules through the standard
/// dihedral atomic-gradient formula to distribute the force across
/// the 5 atoms involved: `C(i-1), N(i), CA(i), C(i), N(i+1)`.
///
/// The forces are zero in the interior of a grid cell with respect
/// to the third derivative — gradient is piecewise-constant in (φ, ψ)
/// because bilinear interpolation makes the gradient locally constant
/// in `(fx, fy)`. Within a cell the *atomic* force still varies with
/// position because the dihedral geometric Jacobian `dφ/dr_X` does.
pub fn add_cmap_forces(
    structure: &Structure,
    _graph: &TopologyGraph,
    ff: &ForceField,
    forces: &mut [geom::Vec3],
) {
    let (atom_index, atom_types) = build_atom_index_and_types(structure);
    let positions = collect_positions(structure);
    for ri in 1..structure.residues.len().saturating_sub(1) {
        // Need the same five global atom indices we computed for the
        // energy path — re-derive them so we have explicit handles.
        if structure.residues[ri].monomer.as_amino_acid().is_none()
            || structure.residues[ri - 1].monomer.as_amino_acid().is_none()
            || structure.residues[ri + 1].monomer.as_amino_acid().is_none()
        {
            continue;
        }
        let Some(&prev_c) = atom_index[ri - 1].get("C") else {
            continue;
        };
        let Some(&cur_n) = atom_index[ri].get("N") else {
            continue;
        };
        let Some(&cur_ca) = atom_index[ri].get("CA") else {
            continue;
        };
        let Some(&cur_c) = atom_index[ri].get("C") else {
            continue;
        };
        let Some(&next_n) = atom_index[ri + 1].get("N") else {
            continue;
        };

        let grid = match ff.cmap(atom_types[cur_ca], atom_types[next_n]) {
            Some(g) => g,
            None => continue,
        };
        // Dihedral geometric Jacobians.  `dihedral_gradient` returns
        // dφ_fb/dr_X where φ_fb is its internal sign convention —
        // opposite-signed from IUPAC, so dφ_IUPAC/dr_X = -dφ_fb/dr_X.
        // We negate before using.
        let (dphi_a, dphi_b, dphi_c, dphi_d, _) = match crate::forces_bonded::dihedral_gradient(
            positions[prev_c],
            positions[cur_n],
            positions[cur_ca],
            positions[cur_c],
        ) {
            Some(t) => t,
            None => continue,
        };
        let (dpsi_a, dpsi_b, dpsi_c, dpsi_d, _) = match crate::forces_bonded::dihedral_gradient(
            positions[cur_n],
            positions[cur_ca],
            positions[cur_c],
            positions[next_n],
        ) {
            Some(t) => t,
            None => continue,
        };
        // IUPAC-convention atomic gradients.
        let dphi_a = -dphi_a;
        let dphi_b = -dphi_b;
        let dphi_c = -dphi_c;
        let dphi_d = -dphi_d;
        let dpsi_a = -dpsi_a;
        let dpsi_b = -dpsi_b;
        let dpsi_c = -dpsi_c;
        let dpsi_d = -dpsi_d;

        // IUPAC-convention (φ, ψ) — the same convention the CMAP grids
        // are indexed by.
        let phi = geom::measure::dihedral(
            positions[prev_c],
            positions[cur_n],
            positions[cur_ca],
            positions[cur_c],
        );
        let psi = geom::measure::dihedral(
            positions[cur_n],
            positions[cur_ca],
            positions[cur_c],
            positions[next_n],
        );
        let (de_dphi_kcal, de_dpsi_kcal) = bilinear_derivatives(grid, phi, psi);
        let de_dphi = crate::units::kcal_to_kj(de_dphi_kcal);
        let de_dpsi = crate::units::kcal_to_kj(de_dpsi_kcal);

        // Standard chain-rule pattern: F_X = -dE/dφ · dφ/dr_X.
        // Atoms appearing in both dihedrals accumulate both contributions.
        forces[prev_c] -= dphi_a * de_dphi;
        forces[cur_n] -= dphi_b * de_dphi + dpsi_a * de_dpsi;
        forces[cur_ca] -= dphi_c * de_dphi + dpsi_b * de_dpsi;
        forces[cur_c] -= dphi_d * de_dphi + dpsi_c * de_dpsi;
        forces[next_n] -= dpsi_d * de_dpsi;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chem::{AminoAcid, standard_ff};
    use geom::build_extended_chain;

    #[test]
    fn cmap_energy_finite_on_built_chain() {
        // A small Ala-Ala-Ala chain has one internal residue (ri=1).
        // The extended-chain (φ ≈ -120°, ψ ≈ +140°) lands in the
        // β-sheet basin of the alanine CMAP — a small (sub-1 kcal/mol)
        // positive value per residue.
        let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala, AminoAcid::Ala]).unwrap();
        let g = geom::build_topology_graph(&s);
        let ff = standard_ff();
        let e = cmap_energy(&s, &g, ff);
        assert!(e.is_finite(), "cmap energy NaN/inf: {e}");
        // Sanity: |E_total| < 50 kJ/mol — far from a clash region.
        assert!(e.abs() < 50.0, "cmap energy oddly large: {e}");
    }

    #[test]
    fn wrap_180_works() {
        assert!((wrap_180(0.0) - 0.0).abs() < 1e-12);
        assert!((wrap_180(180.0) - (-180.0)).abs() < 1e-12);
        assert!((wrap_180(-180.0) - (-180.0)).abs() < 1e-12);
        assert!((wrap_180(190.0) - (-170.0)).abs() < 1e-12);
        assert!((wrap_180(-190.0) - 170.0).abs() < 1e-12);
    }

    #[test]
    fn cmap_forces_match_finite_difference() {
        // Build a small Ala-Ala-Ala chain so there's exactly one
        // internal residue (ri=1) contributing CMAP.  The extended
        // chain has φ = -120° and ψ = +140°; -120° lands exactly on
        // a CMAP grid point, where bilinear interpolation has a
        // discontinuous gradient.  Perturb every atom by a tiny
        // random amount so (φ, ψ) land mid-cell and FD ≈ analytical
        // to within float noise.
        //
        // Tolerance: 1e-2 kJ/mol/Å absolute on every backbone-atom
        // axis.
        use geom::Vec3;
        let mut s =
            build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala, AminoAcid::Ala]).unwrap();
        // Deterministic per-axis nudge so the chain isn't sitting on
        // a CMAP grid boundary.  Magnitude ~0.05 Å is enough to walk
        // φ and ψ ~1° off the nearest grid point without breaking
        // bond lengths.
        let mut k = 0u64;
        for residue in &mut s.residues {
            for atom in &mut residue.atoms {
                k = k.wrapping_add(0x9E3779B97F4A7C15);
                atom.position.x += ((k as f64) * 1e-19).sin() * 0.05;
                atom.position.y += ((k.wrapping_mul(3) as f64) * 1e-19).sin() * 0.05;
                atom.position.z += ((k.wrapping_mul(7) as f64) * 1e-19).sin() * 0.05;
            }
        }
        let g = geom::build_topology_graph(&s);
        let ff = standard_ff();

        let n = s.atom_count();
        let mut analytical = vec![Vec3::zeros(); n];
        add_cmap_forces(&s, &g, ff, &mut analytical);

        // Locate the 5 atoms involved in the lone CMAP term:
        // C(0), N(1), CA(1), C(1), N(2).
        let bump =
            |s: &geom::Structure, atom_idx: usize, axis: usize, eps: f64| -> geom::Structure {
                let mut s2 = s.clone();
                let mut count = 0usize;
                'outer: for residue in &mut s2.residues {
                    for atom in &mut residue.atoms {
                        if count == atom_idx {
                            atom.position[axis] += eps;
                            break 'outer;
                        }
                        count += 1;
                    }
                }
                s2
            };

        let mut global = 0usize;
        let mut atom_indices = Vec::new();
        for (ri, r) in s.residues.iter().enumerate() {
            for atom in &r.atoms {
                if (ri == 0 && atom.name == "C")
                    || (ri == 1 && matches!(atom.name, "N" | "CA" | "C"))
                    || (ri == 2 && atom.name == "N")
                {
                    atom_indices.push(global);
                }
                global += 1;
            }
        }
        assert_eq!(atom_indices.len(), 5);

        let eps = 1e-5;
        for &i in &atom_indices {
            for axis in 0..3 {
                let s_plus = bump(&s, i, axis, eps);
                let s_minus = bump(&s, i, axis, -eps);
                let e_plus = cmap_energy(&s_plus, &g, ff);
                let e_minus = cmap_energy(&s_minus, &g, ff);
                let numeric = -(e_plus - e_minus) / (2.0 * eps);
                let an = analytical[i][axis];
                let err = (an - numeric).abs();
                assert!(
                    err < 5e-4,
                    "atom {i} axis {axis}: analytical={an:.6e}, numeric={numeric:.6e} (err {err:.3e})"
                );
            }
        }
    }

    #[test]
    fn diagnose_phi_sign_convention() {
        // Compare the φ angle as computed by `geom::measure::dihedral`
        // and by `forces_bonded::dihedral_gradient` on the extended
        // Ala3 chain.  If they differ in sign, the energy and force
        // code must use the same one.
        let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala, AminoAcid::Ala]).unwrap();
        let positions: Vec<geom::Vec3> = s
            .residues
            .iter()
            .flat_map(|r| r.atoms.iter().map(|a| a.position))
            .collect();
        let mut atom_idx = std::collections::HashMap::new();
        let mut g_idx = 0;
        for (ri, r) in s.residues.iter().enumerate() {
            for atom in &r.atoms {
                atom_idx.insert((ri, atom.name.to_string()), g_idx);
                g_idx += 1;
            }
        }
        let c0 = atom_idx[&(0, "C".to_string())];
        let n1 = atom_idx[&(1, "N".to_string())];
        let ca1 = atom_idx[&(1, "CA".to_string())];
        let c1 = atom_idx[&(1, "C".to_string())];
        let phi_geom =
            geom::measure::dihedral(positions[c0], positions[n1], positions[ca1], positions[c1]);
        let (_, _, _, _, phi_fb) = crate::forces_bonded::dihedral_gradient(
            positions[c0],
            positions[n1],
            positions[ca1],
            positions[c1],
        )
        .unwrap();
        eprintln!(
            "phi geom: {:.4} rad ({:.2}°)",
            phi_geom,
            phi_geom.to_degrees()
        );
        eprintln!("phi fb:   {:.4} rad ({:.2}°)", phi_fb, phi_fb.to_degrees());
    }

    #[test]
    fn grid_lookup_at_exact_grid_points() {
        // At an exact grid point (no fractional part), bilinear lookup
        // should return the grid value verbatim.
        let ff = standard_ff();
        let grid = ff.cmap(AtomType::CT1, AtomType::NH1).unwrap();
        let phi = -180.0_f64.to_radians();
        let psi = -180.0_f64.to_radians();
        let e = bilinear_lookup(grid, phi, psi);
        assert!(
            (e - grid.at(0, 0)).abs() < 1e-9,
            "expected {}, got {}",
            grid.at(0, 0),
            e
        );
    }
}
