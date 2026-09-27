//! Spatial sort of atoms by 3D Morton (Z-order) curve.
//!
//! Reorders atom indices so that spatially-close atoms have close
//! indices.  After this, consecutive workgroups in a GPU kernel
//! process atoms that are physically near each other — and their
//! Verlet neighbour sets overlap heavily, which is the precondition
//! for cache-locality wins and (eventually) workgroup shared-memory
//! cooperative loads.
//!
//! Returns the permutation as two `Vec<u32>`:
//!   - `gpu_to_cpu[gpu_idx] = cpu_idx`
//!   - `cpu_to_gpu[cpu_idx] = gpu_idx`
//!
//! Use either depending on direction of translation.  Both are
//! inverses of each other.
//!
//! The sort runs once per FullGpuIntegrator construction — it's not
//! re-applied during a trajectory.  Atoms drift, but for the
//! position ranges typical of all-atom MD (sub-Å per fs, the system
//! diameter changes very little over thousands of steps) the
//! initial Morton order stays a useful spatial coherence proxy for
//! the lifetime of a typical run.  Users who want to re-tune the
//! ordering should construct a new FullGpuIntegrator from the
//! current configuration.

/// Workgroup size of the GPU kernels and the spatial-tile size of
/// the tiled nonbonded kernel.  Must match the `@workgroup_size(64)`
/// declarations in every WGSL kernel — changing it requires updating
/// both this constant and the shader source.
pub const TILE_SIZE: usize = 64;

/// Spread the low 10 bits of `x` across the low 30 bits of the
/// result with 2 bits of zeros between each input bit.  Standard
/// Morton-code primitive.
fn part_1by2_u32(x: u32) -> u32 {
    let mut x = x & 0x3FF; // 10 bits
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x << 8)) & 0x0300F00F;
    x = (x | (x << 4)) & 0x030C30C3;
    x = (x | (x << 2)) & 0x09249249;
    x
}

/// 30-bit interleaved Morton code from 10-bit per-axis integer
/// coordinates.  Output fits in a u32 (top 2 bits zero).
pub fn morton_code(x: u32, y: u32, z: u32) -> u32 {
    part_1by2_u32(x) | (part_1by2_u32(y) << 1) | (part_1by2_u32(z) << 2)
}

/// Build the GPU↔CPU permutation tables from per-atom positions.
///
/// 10 bits per axis = 1024 cells along each axis = plenty of
/// resolution for any system that fits in a few hundred Å in each
/// dimension.  Positions outside the [0, 1024) cell index range get
/// clamped — fine for sane MD initial conditions.
///
/// Returns `(gpu_to_cpu, cpu_to_gpu)`.
pub fn morton_permutation(positions_cpu_order: &[[f32; 3]]) -> (Vec<u32>, Vec<u32>) {
    let n = positions_cpu_order.len();
    // Bounding box.
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for p in positions_cpu_order {
        for k in 0..3 {
            if p[k] < mn[k] {
                mn[k] = p[k];
            }
            if p[k] > mx[k] {
                mx[k] = p[k];
            }
        }
    }
    // Avoid divide-by-zero on degenerate axes.
    let extent = [
        (mx[0] - mn[0]).max(1e-6),
        (mx[1] - mn[1]).max(1e-6),
        (mx[2] - mn[2]).max(1e-6),
    ];
    let scale = [1023.0 / extent[0], 1023.0 / extent[1], 1023.0 / extent[2]];
    // Pair each CPU index with its Morton code.
    let mut keyed: Vec<(u32, u32)> = (0..n as u32)
        .map(|cpu_idx| {
            let p = positions_cpu_order[cpu_idx as usize];
            let ix = (((p[0] - mn[0]) * scale[0]).clamp(0.0, 1023.0)) as u32;
            let iy = (((p[1] - mn[1]) * scale[1]).clamp(0.0, 1023.0)) as u32;
            let iz = (((p[2] - mn[2]) * scale[2]).clamp(0.0, 1023.0)) as u32;
            (morton_code(ix, iy, iz), cpu_idx)
        })
        .collect();
    // Stable sort by Morton code.  Ties (atoms in the same Morton
    // cell) keep CPU-index order, which is fine.
    keyed.sort_by_key(|&(m, _)| m);
    // Build forward + inverse maps.
    let gpu_to_cpu: Vec<u32> = keyed.iter().map(|&(_, cpu_idx)| cpu_idx).collect();
    let mut cpu_to_gpu = vec![0u32; n];
    for (gpu_idx, &cpu_idx) in gpu_to_cpu.iter().enumerate() {
        cpu_to_gpu[cpu_idx as usize] = gpu_idx as u32;
    }
    (gpu_to_cpu, cpu_to_gpu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutation_is_a_bijection() {
        let positions: Vec<[f32; 3]> = (0..100)
            .map(|i| {
                let f = i as f32;
                [f.cos() * 5.0, f.sin() * 5.0, (f * 0.3).cos() * 2.0]
            })
            .collect();
        let (gpu_to_cpu, cpu_to_gpu) = morton_permutation(&positions);
        assert_eq!(gpu_to_cpu.len(), 100);
        assert_eq!(cpu_to_gpu.len(), 100);
        for cpu_idx in 0..100u32 {
            let gpu_idx = cpu_to_gpu[cpu_idx as usize];
            assert_eq!(
                gpu_to_cpu[gpu_idx as usize], cpu_idx,
                "permutation not a bijection at cpu_idx={cpu_idx}"
            );
        }
    }

    #[test]
    fn close_atoms_get_close_indices_on_average() {
        // Two spatial clusters: 50 atoms near (0,0,0), 50 near (100,100,100).
        let mut positions = Vec::new();
        for i in 0..50 {
            let f = i as f32 * 0.1;
            positions.push([f, f, f]);
        }
        for i in 0..50 {
            let f = 100.0 + i as f32 * 0.1;
            positions.push([f, f, f]);
        }
        let (gpu_to_cpu, _) = morton_permutation(&positions);
        // After sorting, the first 50 GPU indices should all map to
        // CPU indices in the first cluster (0..50).
        for gpu_idx in 0..50 {
            let cpu_idx = gpu_to_cpu[gpu_idx];
            assert!(
                cpu_idx < 50,
                "gpu_idx={gpu_idx} should map to first cluster, got cpu_idx={cpu_idx}"
            );
        }
        for gpu_idx in 50..100 {
            let cpu_idx = gpu_to_cpu[gpu_idx];
            assert!(
                cpu_idx >= 50,
                "gpu_idx={gpu_idx} should map to second cluster, got cpu_idx={cpu_idx}"
            );
        }
    }
}
