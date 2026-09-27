//! CPU-side construction of the per-i_tile interaction list for
//! [`crate::TileNonbondedPipeline`].
//!
//! Given Morton-sorted atom positions (so atoms in the same tile are
//! spatially close) and a cutoff, returns a CSR layout:
//!
//!   tile_count[i_tile]  : how many j_tiles atom `i_tile` interacts with
//!   tile_start[i_tile]  : offset into tile_indices for atom `i_tile`'s list
//!   tile_indices[]      : flat j_tile indices
//!
//! Algorithm:
//!   1. For each i_tile, compute its axis-aligned bbox over the 64
//!      atoms in it (last tile may have < 64 atoms).
//!   2. For each pair (i_tile, j_tile), compute the minimum
//!      separation between their bboxes.  If it's less than
//!      `cutoff + skin`, the pair has at least one (i, j) within
//!      cutoff and the kernel needs to visit it.
//!
//! O(N_TILES²).  At 5840 atoms / 64-atom tiles = 91 tiles, 91² = 8281
//! pair tests — sub-millisecond per Verlet rebuild.

use crate::spatial_sort::TILE_SIZE;

/// CSR per-i_tile list of interacting j_tiles.
#[derive(Debug, Clone)]
pub struct TileInteractionList {
    pub tile_count: Vec<u32>,
    pub tile_start: Vec<u32>,
    pub tile_indices: Vec<u32>,
    /// Number of i_tiles (= ceil(n_atoms / TILE_SIZE)).
    pub n_tiles: usize,
}

/// Compute the tile-interaction list from Morton-sorted positions.
///
/// `positions_gpu_order[i]` is the position of GPU atom i (post-Morton
/// permutation).  `cutoff_a` should be the full pair cutoff PLUS the
/// Verlet skin so the list stays valid for the same set of steps the
/// Verlet list does.
pub fn build_tile_interaction_list(
    positions_gpu_order: &[[f32; 3]],
    cutoff_a: f32,
) -> TileInteractionList {
    let n_atoms = positions_gpu_order.len();
    let n_tiles = n_atoms.div_ceil(TILE_SIZE);
    let cutoff_sq = cutoff_a * cutoff_a;

    // Per-tile bbox: [min_x, min_y, min_z, max_x, max_y, max_z].
    let mut bbox: Vec<[f32; 6]> = vec![
        [
            f32::INFINITY,
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY
        ];
        n_tiles
    ];
    for atom_idx in 0..n_atoms {
        let tile = atom_idx / TILE_SIZE;
        let p = positions_gpu_order[atom_idx];
        let b = &mut bbox[tile];
        for k in 0..3 {
            if p[k] < b[k] {
                b[k] = p[k];
            }
            if p[k] > b[k + 3] {
                b[k + 3] = p[k];
            }
        }
    }

    // O(N_TILES²) bbox-bbox distance test.  For each i_tile, every
    // j_tile (including j_tile == i_tile for the diagonal) is a
    // candidate.  We DON'T impose i_tile <= j_tile because the
    // kernel processes ALL j_tiles per i_tile to accumulate the full
    // force on each i (asymmetric — i's force from j is computed in
    // i_tile's workgroup, j's force from i in j_tile's workgroup).
    let mut tile_count = vec![0u32; n_tiles];
    let mut per_tile_j_lists: Vec<Vec<u32>> = vec![Vec::new(); n_tiles];
    for i_tile in 0..n_tiles {
        let bi = bbox[i_tile];
        for j_tile in 0..n_tiles {
            let bj = bbox[j_tile];
            // Min separation per axis: max(0, max_i - min_j) if i is
            // to the right of j; max(0, max_j - min_i) if j is to the
            // right; 0 if they overlap.  Equivalently:
            //   d_k = max(0, bi.min[k] - bj.max[k], bj.min[k] - bi.max[k])
            let mut d_sq = 0.0_f32;
            for k in 0..3 {
                let lo = bi[k] - bj[k + 3]; // i_min - j_max
                let hi = bj[k] - bi[k + 3]; // j_min - i_max
                let d = lo.max(hi).max(0.0);
                d_sq += d * d;
            }
            if d_sq <= cutoff_sq {
                per_tile_j_lists[i_tile].push(j_tile as u32);
            }
        }
        tile_count[i_tile] = per_tile_j_lists[i_tile].len() as u32;
    }

    // Flatten into CSR.
    let mut tile_start = vec![0u32; n_tiles];
    let mut total = 0u32;
    for t in 0..n_tiles {
        tile_start[t] = total;
        total += tile_count[t];
    }
    let mut tile_indices = Vec::with_capacity(total as usize);
    for list in &per_tile_j_lists {
        tile_indices.extend_from_slice(list);
    }

    TileInteractionList {
        tile_count,
        tile_start,
        tile_indices,
        n_tiles,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_only_when_atoms_isolated() {
        // 64 atoms in one cluster + 64 in a far cluster → each tile
        // only interacts with itself.
        let mut positions = Vec::with_capacity(128);
        for _ in 0..64 {
            positions.push([0.0, 0.0, 0.0]);
        }
        for _ in 0..64 {
            positions.push([1000.0, 0.0, 0.0]); // far away
        }
        let list = build_tile_interaction_list(&positions, 10.0);
        assert_eq!(list.n_tiles, 2);
        // Each tile interacts only with itself.
        assert_eq!(list.tile_count[0], 1);
        assert_eq!(list.tile_count[1], 1);
        assert_eq!(list.tile_indices[list.tile_start[0] as usize], 0);
        assert_eq!(list.tile_indices[list.tile_start[1] as usize], 1);
    }

    #[test]
    fn neighbouring_tiles_interact() {
        // Two clusters 5 Å apart, both inside 10 Å cutoff → tiles see
        // each other.
        let mut positions = Vec::with_capacity(128);
        for _ in 0..64 {
            positions.push([0.0, 0.0, 0.0]);
        }
        for _ in 0..64 {
            positions.push([5.0, 0.0, 0.0]);
        }
        let list = build_tile_interaction_list(&positions, 10.0);
        // Both tiles should see each other AND themselves.
        assert_eq!(list.tile_count[0], 2);
        assert_eq!(list.tile_count[1], 2);
    }

    #[test]
    fn small_system_fits_in_one_tile() {
        // 30 atoms → 1 tile, only diagonal interaction.
        let positions: Vec<[f32; 3]> = (0..30).map(|i| [i as f32 * 0.5, 0.0, 0.0]).collect();
        let list = build_tile_interaction_list(&positions, 10.0);
        assert_eq!(list.n_tiles, 1);
        assert_eq!(list.tile_count[0], 1);
    }
}
