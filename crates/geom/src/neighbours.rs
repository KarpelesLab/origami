//! Cell-list spatial hash for fast neighbour search.
//!
//! Bucketing atoms into a 3D grid of cells lets us enumerate candidate
//! pairs within a cutoff in O(N × ⟨neighbour count⟩) instead of O(N²).
//! Picking cell size = cutoff guarantees that any atom within the cutoff
//! of `i` lives in `i`'s own cell or one of the 26 neighbouring cells
//! (3³ − 1).

use crate::Vec3;

#[derive(Debug, Clone)]
pub struct CellList {
    /// Side length of each cubic cell in Å. Should be ≥ the largest cutoff.
    cell_size: f64,
    /// Lower-left corner of cell (0, 0, 0). Retained for future use (e.g.
    /// re-binning a single atom without rebuilding).
    #[allow(dead_code)]
    origin: Vec3,
    /// Number of cells along x, y, z.
    dims: [usize; 3],
    /// Flat row-major storage: cells[index_for(ix, iy, iz)] = list of atom indices.
    cells: Vec<Vec<usize>>,
    /// Per-atom: which cell does it belong to (3D index, flattened).
    atom_to_cell: Vec<usize>,
}

impl CellList {
    pub fn build(positions: &[Vec3], cell_size: f64) -> Self {
        let mut min = Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut max = Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for p in positions {
            for i in 0..3 {
                if p[i] < min[i] {
                    min[i] = p[i];
                }
                if p[i] > max[i] {
                    max[i] = p[i];
                }
            }
        }
        // Pad slightly to avoid boundary surprises with floating-point.
        let pad = cell_size * 0.05;
        let origin = Vec3::new(min.x - pad, min.y - pad, min.z - pad);
        let extent = max - min + Vec3::new(2.0 * pad, 2.0 * pad, 2.0 * pad);
        let dims = [
            ((extent.x / cell_size).ceil() as usize).max(1),
            ((extent.y / cell_size).ceil() as usize).max(1),
            ((extent.z / cell_size).ceil() as usize).max(1),
        ];
        let total_cells = dims[0] * dims[1] * dims[2];
        let mut cells: Vec<Vec<usize>> = vec![Vec::new(); total_cells];
        let mut atom_to_cell = vec![0usize; positions.len()];
        for (i, &p) in positions.iter().enumerate() {
            let ix = ((p.x - origin.x) / cell_size) as usize;
            let iy = ((p.y - origin.y) / cell_size) as usize;
            let iz = ((p.z - origin.z) / cell_size) as usize;
            let idx = (iz * dims[1] + iy) * dims[0] + ix;
            cells[idx].push(i);
            atom_to_cell[i] = idx;
        }
        CellList { cell_size, origin, dims, cells, atom_to_cell }
    }

    pub fn cell_size(&self) -> f64 {
        self.cell_size
    }

    /// Iterate all unordered pairs (i, j) with i < j whose distance is at
    /// most `cutoff`. This is the correct interface for non-bonded energy
    /// summation.
    pub fn iter_pairs_within<'a>(
        &'a self,
        positions: &'a [Vec3],
        cutoff: f64,
    ) -> impl Iterator<Item = (usize, usize, f64)> + 'a {
        assert!(cutoff <= self.cell_size + 1e-9,
            "cutoff {} exceeds cell size {}", cutoff, self.cell_size);
        let cutoff_sq = cutoff * cutoff;
        let dims = self.dims;
        // For each cell, we visit its 13 "forward" neighbours plus itself —
        // covers each pair exactly once when combined with the i < j filter.
        // For simplicity, iterate every cell and every neighbouring cell
        // (including the same cell), but use i < j to avoid double-counting.
        (0..self.cells.len()).flat_map(move |cell_idx| {
            let ix = cell_idx % dims[0];
            let iy = (cell_idx / dims[0]) % dims[1];
            let iz = cell_idx / (dims[0] * dims[1]);
            let mut neigh_pairs: Vec<(usize, usize)> = Vec::new();
            for dz in -1i32..=1 {
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let nx = ix as i32 + dx;
                        let ny = iy as i32 + dy;
                        let nz = iz as i32 + dz;
                        if nx < 0 || ny < 0 || nz < 0 {
                            continue;
                        }
                        let (nx, ny, nz) = (nx as usize, ny as usize, nz as usize);
                        if nx >= dims[0] || ny >= dims[1] || nz >= dims[2] {
                            continue;
                        }
                        let neigh_idx = (nz * dims[1] + ny) * dims[0] + nx;
                        if neigh_idx < cell_idx {
                            continue; // already covered by the symmetric pair
                        }
                        neigh_pairs.push((cell_idx, neigh_idx));
                    }
                }
            }
            neigh_pairs.into_iter().flat_map(move |(c1, c2)| {
                let same = c1 == c2;
                self.cells[c1].iter().enumerate().flat_map(move |(idx_in_c1, &i)| {
                    let starts_at = if same { idx_in_c1 + 1 } else { 0 };
                    self.cells[c2][starts_at..].iter().filter_map(move |&j| {
                        if i == j {
                            return None;
                        }
                        let (a, b) = if i < j { (i, j) } else { (j, i) };
                        let r2 = (positions[a] - positions[b]).norm_squared();
                        if r2 <= cutoff_sq {
                            Some((a, b, r2.sqrt()))
                        } else {
                            None
                        }
                    })
                })
            })
        })
    }

    pub fn atom_cell(&self, atom_idx: usize) -> usize {
        self.atom_to_cell[atom_idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn pairs_within_cutoff_match_brute_force() {
        // Random-ish positions, brute force the distances and compare.
        let positions: Vec<Vec3> = (0..50)
            .map(|i| {
                let f = i as f64;
                Vec3::new(
                    (f * 0.7).cos() * 8.0,
                    (f * 1.3).sin() * 8.0,
                    (f * 0.4).cos() * 8.0,
                )
            })
            .collect();
        let cutoff = 5.0;
        let cl = CellList::build(&positions, 5.0);
        let cell_pairs: HashSet<(usize, usize)> = cl
            .iter_pairs_within(&positions, cutoff)
            .map(|(i, j, _)| (i, j))
            .collect();
        let mut brute_pairs: HashSet<(usize, usize)> = HashSet::new();
        for i in 0..positions.len() {
            for j in (i + 1)..positions.len() {
                if (positions[i] - positions[j]).norm() <= cutoff {
                    brute_pairs.insert((i, j));
                }
            }
        }
        assert_eq!(cell_pairs, brute_pairs);
    }

    #[test]
    fn empty_input_produces_no_pairs() {
        let cl = CellList::build(&[Vec3::zeros()], 5.0);
        let n = cl.iter_pairs_within(&[Vec3::zeros()], 5.0).count();
        assert_eq!(n, 0);
    }

    #[test]
    fn distant_pair_filtered_out() {
        let positions = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(20.0, 0.0, 0.0)];
        let cl = CellList::build(&positions, 5.0);
        let n = cl.iter_pairs_within(&positions, 5.0).count();
        assert_eq!(n, 0);
    }
}
