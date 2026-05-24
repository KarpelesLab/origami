// SHAKE iterative bond-constraint solver — GPU implementation.
//
// Constrains every X-H bond to its target distance via the standard
// SHAKE update (Ryckaert et al. 1977):
//   r_new = pos[i] - pos[j]
//   err   = |r_new|² - d_target²
//   λ     = err / (2 (1/m_i + 1/m_j) (r_new · r_old))
//   pos[i] -= λ r_old / m_i
//   pos[j] += λ r_old / m_j
// iterated until |err| < tol² for all constraints.
//
// Parallelisation: one thread per heavy atom X.  Each X owns ALL the
// X-H bonds it participates in (typically 1-3 for proteins).  Since
// each H is in exactly one X-H bond, and the H's are touched only by
// their parent X's thread, there are NO cross-thread races — the
// per-X iteration is pure Gauss-Seidel inside the thread, with no
// atomics, no syncs, no convergence problems beyond what CPU SHAKE
// already has.
//
// Heavy atoms with no bonded H (e.g. backbone carbonyl O) get an
// h_count of 0 and the kernel short-circuits.
//
// CSR layout, per atom i (0..n_atoms):
//   h_count[i]                                 : u32   (0..MAX_H_PER_X)
//   per_atom_h_atoms[i*MAX_H_PER_X + k]        : u32   (H atom index)
//   per_atom_h_d_sq[i*MAX_H_PER_X + k]         : f32   (d_target² in Å²)
//   inv_mass[i]                                : f32   (1/Da)
//
// `ref_positions` holds positions captured BEFORE the velocity step
// that broke the constraint; the SHAKE projection moves the current
// `positions` back onto the constraint surface using `ref_positions`
// as the linearisation reference.

const MAX_H_PER_X: u32 = 4u;

struct ShakeParams {
    n_atoms: u32,
    max_iters: u32,
    tol_sq: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: ShakeParams;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> ref_positions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> inv_mass: array<f32>;
@group(0) @binding(4) var<storage, read> h_count: array<u32>;
@group(0) @binding(5) var<storage, read> per_atom_h_atoms: array<u32>;
@group(0) @binding(6) var<storage, read> per_atom_h_d_sq: array<f32>;

@compute @workgroup_size(64)
fn shake_per_x(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_atoms) {
        return;
    }
    let count = h_count[i];
    if (count == 0u) {
        return;
    }
    let inv_m_x = inv_mass[i];
    let p_x_ref = ref_positions[i].xyz;

    // Load all of X's H neighbours into thread-private registers.
    // Fixed cap of MAX_H_PER_X = 4 (covers CH3, NH3+, etc.).
    var h_idx: array<u32, 4>;
    var h_pos: array<vec3<f32>, 4>;
    var h_ref: array<vec3<f32>, 4>;
    var h_d_sq: array<f32, 4>;
    var h_inv_m: array<f32, 4>;
    var p_x = positions[i].xyz;
    let base = i * MAX_H_PER_X;
    // Manually unroll the load loop into 4 explicit blocks — WGSL
    // can't index variable arrays of structs by a non-const index in
    // all stages.
    for (var k: u32 = 0u; k < count; k = k + 1u) {
        let h = per_atom_h_atoms[base + k];
        h_idx[k] = h;
        h_pos[k] = positions[h].xyz;
        h_ref[k] = ref_positions[h].xyz;
        h_d_sq[k] = per_atom_h_d_sq[base + k];
        h_inv_m[k] = inv_mass[h];
    }

    for (var iter: u32 = 0u; iter < params.max_iters; iter = iter + 1u) {
        var max_err: f32 = 0.0;
        for (var k: u32 = 0u; k < count; k = k + 1u) {
            let r_new = p_x - h_pos[k];
            let r_old = p_x_ref - h_ref[k];
            let len2 = dot(r_new, r_new);
            let err = len2 - h_d_sq[k];
            let abs_err = abs(err);
            if (abs_err > max_err) {
                max_err = abs_err;
            }
            if (abs_err < params.tol_sq) {
                continue;
            }
            let dot_nr = dot(r_new, r_old);
            let denom = 2.0 * (inv_m_x + h_inv_m[k]) * dot_nr;
            if (abs(denom) < 1e-18) {
                // Reference vector nearly perpendicular to current —
                // linearisation breaks down.  Skip; hope next iteration
                // sees better geometry.
                continue;
            }
            let lambda = err / denom;
            let delta = r_old * lambda;
            p_x = p_x - delta * inv_m_x;
            h_pos[k] = h_pos[k] + delta * h_inv_m[k];
        }
        if (max_err < params.tol_sq) {
            break;
        }
    }

    // Write back the (possibly-corrected) positions.
    positions[i] = vec4<f32>(p_x, 0.0);
    for (var k: u32 = 0u; k < count; k = k + 1u) {
        positions[h_idx[k]] = vec4<f32>(h_pos[k], 0.0);
    }
}
