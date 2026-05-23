// Verlet-list nonbonded kernel: LJ + reaction-field Coulomb, but
// each thread loops only over its precomputed neighbour list
// instead of all N atoms.  Same physics, same sign conventions,
// same 1-4 specials path as nonbonded.wgsl — the only difference
// is the inner-loop neighbour discovery.
//
// Neighbour list is built on the CPU once per skin-rebuild
// (typically every ~20 steps with a 2 Å skin) and uploaded as
// flat arrays.  Within each call the GPU just reads it.

struct Params {
    n_atoms: u32,
    cutoff_sq: f32,
    inv_rc3: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> type_index: array<u32>;
@group(0) @binding(3) var<storage, read> lj_params: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> lj_params_14: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> charges: array<f32>;
@group(0) @binding(6) var<storage, read> exclusions: array<u32>;
@group(0) @binding(7) var<storage, read> one_four_mask: array<u32>;
@group(0) @binding(8) var<storage, read> nbr_count: array<u32>;        // per-atom neighbour count
@group(0) @binding(9) var<storage, read> nbr_start: array<u32>;        // per-atom offset into nbr_indices
@group(0) @binding(10) var<storage, read> nbr_indices: array<u32>;     // flat neighbour-j array
@group(0) @binding(11) var<storage, read_write> forces: array<vec4<f32>>;

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

@compute @workgroup_size(64)
fn nonbonded_verlet(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_atoms) {
        return;
    }
    let pi = positions[i].xyz;
    let ti = type_index[i];
    let qi = charges[i];
    let pi_eps = lj_params[ti].x;
    let pi_rmin_half = lj_params[ti].y;
    let pi_eps_14 = lj_params_14[ti].x;
    let pi_rmin_half_14 = lj_params_14[ti].y;

    var acc = vec3<f32>(0.0, 0.0, 0.0);
    let count = nbr_count[i];
    let start = nbr_start[i];
    for (var k: u32 = 0u; k < count; k = k + 1u) {
        let j = nbr_indices[start + k];
        if (is_excluded(i, j)) {
            continue;
        }
        let dx = positions[j].xyz - pi;
        let r2 = dot(dx, dx);
        if (r2 > params.cutoff_sq || r2 < 1e-6) {
            continue;
        }
        let r = sqrt(r2);
        let inv_r2 = 1.0 / r2;
        let tj = type_index[j];
        let one_four = is_one_four(i, j);
        let eps_i = select(pi_eps, pi_eps_14, one_four);
        let rmin_half_i = select(pi_rmin_half, pi_rmin_half_14, one_four);
        let eps_j = select(lj_params[tj].x, lj_params_14[tj].x, one_four);
        let rmin_half_j = select(lj_params[tj].y, lj_params_14[tj].y, one_four);
        let eps = sqrt(eps_i * eps_j);
        let rmin = rmin_half_i + rmin_half_j;
        let ratio = rmin / r;
        let r2_ratio = ratio * ratio;
        let r6_ratio = r2_ratio * r2_ratio * r2_ratio;
        let r12_ratio = r6_ratio * r6_ratio;
        let lj_coeff = 12.0 * eps * inv_r2 * (r6_ratio - r12_ratio);
        let qq = qi * charges[j];
        let coul_coeff = -COULOMB_K_KJ * qq * (inv_r2 / r - params.inv_rc3);
        acc = acc + dx * (lj_coeff + coul_coeff);
    }
    forces[i] = vec4<f32>(acc, 0.0);
}
