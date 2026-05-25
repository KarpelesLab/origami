// Tile-based nonbonded force kernel — OpenMM-style architecture.
//
// One workgroup per `i_tile` of 64 spatially-close atoms (atoms are
// Morton-sorted on the CPU before upload).  The workgroup walks a
// precomputed list of `j_tiles` that contain any atom within
// cutoff + skin of any atom in this i_tile.  For each j_tile,
// the 64 threads cooperatively load the 64 j-atoms' data into
// workgroup shared memory; each thread then computes interactions
// for its assigned i against all 64 j's in shared memory.
//
// Cooperative load + 64-way reuse means each j-atom's
// (position, charge, lj_data) is fetched from global memory exactly
// once per j_tile visit (instead of once per neighbouring i's Verlet
// inner loop).  At ribosome scale where memory bandwidth dominates,
// the win is large; the trade-off is more total pair-distance tests
// (since the cutoff is enforced inside the kernel instead of by a
// pre-built neighbour list).
//
// Forces accumulate in thread registers across all j_tiles for the
// thread's i, then a single write to forces[i] at the end — no
// atomics needed because each thread owns its own i and no other
// workgroup writes to forces[i].
//
// Same physics, sign conventions, and 1-4 specials handling as
// `nonbonded_verlet.wgsl`.

const TILE_SIZE: u32 = 64u;

struct Params {
    n_atoms: u32,
    cutoff_sq: f32,
    inv_rc3: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> atom_lj_data: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> charges: array<f32>;
@group(0) @binding(4) var<storage, read> exclusions: array<u32>;
@group(0) @binding(5) var<storage, read> one_four_mask: array<u32>;
@group(0) @binding(6) var<storage, read> tile_count: array<u32>;     // per-i_tile: how many j_tiles to walk
@group(0) @binding(7) var<storage, read> tile_start: array<u32>;     // per-i_tile: offset into tile_indices
@group(0) @binding(8) var<storage, read> tile_indices: array<u32>;   // flat j_tile indices
@group(0) @binding(9) var<storage, read_write> forces: array<vec4<f32>>;

const COULOMB_K_KJ: f32 = 1389.35455;

fn is_excluded(i: u32, j: u32) -> bool {
    let bit_idx = i * params.n_atoms + j;
    let word = exclusions[bit_idx / 32u];
    let mask = 1u << (bit_idx % 32u);
    return (word & mask) != 0u;
}

fn is_one_four(i: u32, j: u32) -> bool {
    let bit_idx = i * params.n_atoms + j;
    let word = one_four_mask[bit_idx / 32u];
    let mask = 1u << (bit_idx % 32u);
    return (word & mask) != 0u;
}

// ---- Workgroup-shared j-atom data ----
var<workgroup> shared_j_pos: array<vec4<f32>, 64>;
var<workgroup> shared_j_lj: array<vec4<f32>, 64>;
var<workgroup> shared_j_charge: array<f32, 64>;

@compute @workgroup_size(64)
fn tile_force(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_index) local_id: u32,
) {
    let i_tile = wg_id.x;
    let i = i_tile * TILE_SIZE + local_id;
    let i_in_range = i < params.n_atoms;

    // Load this thread's i-atom data (only if in range — but read
    // unconditionally to keep all threads on the same control path
    // for the workgroupBarriers later).  Out-of-range threads write
    // garbage to thread-private registers; they get filtered before
    // the final forces[i] write.
    let i_safe = select(0u, i, i_in_range);
    let pi = positions[i_safe].xyz;
    let qi = charges[i_safe];
    let lj_i = atom_lj_data[i_safe];

    var acc = vec3<f32>(0.0, 0.0, 0.0);
    let n_j_tiles = tile_count[i_tile];
    let j_tile_base = tile_start[i_tile];

    for (var t: u32 = 0u; t < n_j_tiles; t = t + 1u) {
        let j_tile = tile_indices[j_tile_base + t];

        // Cooperative load: thread `local_id` loads atom
        // (j_tile * TILE_SIZE + local_id) into the workgroup-shared
        // tile cache.
        let j_load = j_tile * TILE_SIZE + local_id;
        let j_load_in_range = j_load < params.n_atoms;
        if (j_load_in_range) {
            shared_j_pos[local_id] = positions[j_load];
            shared_j_lj[local_id] = atom_lj_data[j_load];
            shared_j_charge[local_id] = charges[j_load];
        }
        // All threads must finish the cooperative load before any
        // thread starts reading shared memory.
        workgroupBarrier();

        if (i_in_range) {
            // Walk the 64 j's in this tile from shared memory.
            for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
                let j = j_tile * TILE_SIZE + k;
                if (j >= params.n_atoms) { continue; }
                if (i == j) { continue; }
                if (is_excluded(i, j)) { continue; }

                let dx = shared_j_pos[k].xyz - pi;
                let r2 = dot(dx, dx);
                if (r2 > params.cutoff_sq || r2 < 1e-6) { continue; }

                let r = sqrt(r2);
                let inv_r2 = 1.0 / r2;
                let lj_j = shared_j_lj[k];
                let one_four = is_one_four(i, j);
                let eps_i = select(lj_i.x, lj_i.z, one_four);
                let rmin_half_i = select(lj_i.y, lj_i.w, one_four);
                let eps_j = select(lj_j.x, lj_j.z, one_four);
                let rmin_half_j = select(lj_j.y, lj_j.w, one_four);
                let eps = sqrt(eps_i * eps_j);
                let rmin = rmin_half_i + rmin_half_j;
                let ratio = rmin / r;
                let r2_ratio = ratio * ratio;
                let r6_ratio = r2_ratio * r2_ratio * r2_ratio;
                let r12_ratio = r6_ratio * r6_ratio;
                let lj_coeff = 12.0 * eps * inv_r2 * (r6_ratio - r12_ratio);
                let qq = qi * shared_j_charge[k];
                let coul_coeff = -COULOMB_K_KJ * qq * (inv_r2 / r - params.inv_rc3);
                acc = acc + dx * (lj_coeff + coul_coeff);
            }
        }
        // Ensure all threads finish reading shared memory before the
        // next iteration overwrites it.
        workgroupBarrier();
    }

    if (i_in_range) {
        // Accumulate into the persistent forces buffer — bonded,
        // GB, etc. write to it before/after.
        forces[i] = forces[i] + vec4<f32>(acc, 0.0);
    }
}
