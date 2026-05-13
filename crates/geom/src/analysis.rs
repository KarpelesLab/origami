//! Trajectory analysis helpers: scalar per-frame metrics (radius of
//! gyration, end-to-end distance) and trajectory-averaged metrics
//! (residue-residue contact frequency map).
//!
//! All metrics operate on Cα atoms only, matching the convention used
//! elsewhere in `geom` (e.g. [`crate::rmsd_ca`]). Cα is the canonical
//! per-residue representative because backbone N/C are constrained by
//! peptide-bond geometry and side chains have variable size; Cα tracks
//! tertiary motion cleanly.

use crate::structure::Structure;
use crate::Vec3;

/// Radius of gyration computed from Cα positions:
///   Rg² = (1/N) Σ |r_i − r_centroid|²
///
/// Returns `None` if the structure has no residues or any residue is
/// missing its Cα atom.
pub fn radius_of_gyration_ca(structure: &Structure) -> Option<f64> {
    let positions = ca_positions(structure)?;
    Some(radius_of_gyration_points(&positions))
}

/// Radius of gyration over an arbitrary point set. Treats every point as
/// having unit mass (mass-weighted Rg would need atomic masses; for Cα
/// representation that detail is conventionally dropped).
pub fn radius_of_gyration_points(points: &[Vec3]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    let n = points.len() as f64;
    let centroid: Vec3 = points.iter().copied().sum::<Vec3>() / n;
    let sq_sum: f64 = points
        .iter()
        .map(|p| (*p - centroid).norm_squared())
        .sum();
    (sq_sum / n).sqrt()
}

/// Distance between the first and last residue's Cα atoms. A scalar
/// proxy for chain extension; small Rg paired with small end-to-end is
/// "compact globule", small Rg with large end-to-end is "ring-like".
///
/// Returns `None` if the structure has fewer than two residues or
/// either terminus is missing its Cα.
pub fn end_to_end_ca(structure: &Structure) -> Option<f64> {
    if structure.residues.len() < 2 {
        return None;
    }
    let first = structure.residues.first()?.position("CA")?;
    let last = structure.residues.last()?.position("CA")?;
    Some((first - last).norm())
}

/// Residue-residue contact frequency matrix over a trajectory: entry
/// `(i, j)` is the fraction of frames in which residue *i* and residue
/// *j* have any heavy-atom pair within `cutoff_a`.
///
/// "Heavy atom" = anything not hydrogen — the cutoff is meant to be
/// applied to side-chain / backbone contacts the way contact maps are
/// conventionally read. Diagonal entries are always 1.0 (a residue
/// always contacts itself). Sequence-adjacent pairs (|i − j| ≤ 1) are
/// also 1.0 by chain connectivity; callers usually mask those when
/// rendering the map.
///
/// Returns `None` if `frames` is empty or frames have inconsistent
/// residue counts. Heavy-atom selection happens via element checks on
/// the atom's `element` field; missing-Cα structures pass through —
/// we don't require Cα for contact detection.
pub fn contact_map_ca(frames: &[Structure], cutoff_a: f64) -> Option<Vec<Vec<f64>>> {
    if frames.is_empty() {
        return None;
    }
    let n_res = frames[0].residues.len();
    if frames.iter().any(|f| f.residues.len() != n_res) {
        return None;
    }
    let cutoff_sq = cutoff_a * cutoff_a;
    let mut counts: Vec<Vec<u32>> = vec![vec![0u32; n_res]; n_res];
    let n_frames = frames.len() as f64;

    for frame in frames {
        for i in 0..n_res {
            for j in (i + 1)..n_res {
                if any_heavy_contact(&frame.residues[i], &frame.residues[j], cutoff_sq) {
                    counts[i][j] += 1;
                    counts[j][i] += 1;
                }
            }
        }
    }

    let mut freq: Vec<Vec<f64>> = vec![vec![0.0; n_res]; n_res];
    for i in 0..n_res {
        for j in 0..n_res {
            freq[i][j] = if i == j {
                1.0
            } else {
                counts[i][j] as f64 / n_frames
            };
        }
    }
    Some(freq)
}

fn any_heavy_contact(
    a: &crate::structure::PlacedResidue,
    b: &crate::structure::PlacedResidue,
    cutoff_sq: f64,
) -> bool {
    for at_a in &a.atoms {
        if at_a.element == chem::Element::H {
            continue;
        }
        for at_b in &b.atoms {
            if at_b.element == chem::Element::H {
                continue;
            }
            if (at_a.position - at_b.position).norm_squared() <= cutoff_sq {
                return true;
            }
        }
    }
    false
}

fn ca_positions(structure: &Structure) -> Option<Vec<Vec3>> {
    let mut out = Vec::with_capacity(structure.residues.len());
    for r in &structure.residues {
        out.push(r.position("CA")?);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rg_zero_for_single_point() {
        let p = vec![Vec3::new(1.0, 2.0, 3.0)];
        assert!(radius_of_gyration_points(&p).abs() < 1e-12);
    }

    #[test]
    fn rg_for_symmetric_pair() {
        // Two points 2 Å apart → Rg = 1 Å.
        let p = vec![Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)];
        let rg = radius_of_gyration_points(&p);
        assert!((rg - 1.0).abs() < 1e-12, "expected 1.0, got {rg}");
    }

    #[test]
    fn rg_invariant_under_translation() {
        let p = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 0.0, 2.0),
        ];
        let p_shifted: Vec<Vec3> = p.iter().map(|v| v + Vec3::new(10.0, -5.0, 7.5)).collect();
        let a = radius_of_gyration_points(&p);
        let b = radius_of_gyration_points(&p_shifted);
        assert!((a - b).abs() < 1e-12);
    }
}
