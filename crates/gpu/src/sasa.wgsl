// Per-atom solvent-accessible surface area via dot-density
// (Shrake-Rupley).  One thread per atom; the thread iterates over
// `N_DOTS` Fibonacci-spiral test points distributed on atom i's
// expanded vdW sphere, counts how many are not buried by any
// neighbouring sphere, and writes the per-atom area
//   A_i = (accessible / N_DOTS) · 4π R_i².
//
// `radii[i]` is atom i's expanded radius (vdW + probe, typically
// 1.4 Å water probe).  The neighbour list is the per-atom CSR
// produced by the CPU-side `ensure_sasa_neighbours` cell-list
// scan — every j whose expanded-vdW sphere overlaps i's, plus the
// Verlet skin so the list survives small atom drift.
//
// Why not analytical:  the dot-density area is approximate (~1 %
// vs N=4096), but the algorithm is embarrassingly parallel and
// every thread does the same amount of work (vs the topology-based
// analytical SASA where per-atom work varies wildly).  Forces from
// this kernel would have discontinuities at dot accessibility flips,
// so it's energy-only for now; smooth (sigmoidal) coverage forces
// are the next step.

const N_DOTS: u32 = 256u;

struct Params {
    n_atoms: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> radii: array<f32>;            // expanded (vdW + probe)
@group(0) @binding(3) var<storage, read> dots: array<vec4<f32>>;       // N_DOTS unit-sphere points
@group(0) @binding(4) var<storage, read> nbr_count: array<u32>;
@group(0) @binding(5) var<storage, read> nbr_start: array<u32>;
@group(0) @binding(6) var<storage, read> nbr_indices: array<u32>;
@group(0) @binding(7) var<storage, read_write> per_atom_area: array<f32>;

@compute @workgroup_size(64)
fn sasa_dot_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_atoms) {
        return;
    }
    let pi = positions[i].xyz;
    let ri = radii[i];
    let count = nbr_count[i];
    let start = nbr_start[i];

    var accessible: u32 = 0u;
    for (var k: u32 = 0u; k < N_DOTS; k = k + 1u) {
        let dot_pos = pi + dots[k].xyz * ri;
        var buried = false;
        for (var t: u32 = 0u; t < count; t = t + 1u) {
            let j = nbr_indices[start + t];
            let d = dot_pos - positions[j].xyz;
            let d_sq = dot(d, d);
            let rj = radii[j];
            if (d_sq < rj * rj) {
                buried = true;
                break;
            }
        }
        if (!buried) {
            accessible = accessible + 1u;
        }
    }

    let four_pi = 12.566370614;
    let r2 = ri * ri;
    per_atom_area[i] = (f32(accessible) / f32(N_DOTS)) * four_pi * r2;
}
