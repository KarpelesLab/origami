// Generalized-Born OBC II Born radii kernel — Verlet-list variant.
// One thread per atom i.  Each thread walks i's precomputed neighbour
// list (built on the CPU at the 20-Å Born cutoff + skin), integrates
// the HCT pairwise descreening, then applies the OBC II tanh
// transform to convert ψ → R_eff.
//
// Output: per-atom effective Born radius (Å).

struct Params {
    n_atoms: u32,
    cutoff_sq: f32,     // squared Born cutoff in Å²
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> rho_tilde: array<f32>;   // ρ - OBC_OFFSET
@group(0) @binding(3) var<storage, read> rho: array<f32>;         // intrinsic vdW radius
@group(0) @binding(4) var<storage, read> scale: array<f32>;       // HCT scale per atom
@group(0) @binding(5) var<storage, read_write> r_eff: array<f32>; // output Born radii
@group(0) @binding(6) var<storage, read> nbr_count: array<u32>;   // per-atom neighbour count
@group(0) @binding(7) var<storage, read> nbr_start: array<u32>;   // per-atom offset into nbr_indices
@group(0) @binding(8) var<storage, read> nbr_indices: array<u32>; // flat neighbour-j array

const OBC_ALPHA: f32 = 1.0;
const OBC_BETA: f32 = 0.8;
const OBC_GAMMA: f32 = 4.85;

/// HCT pairwise descreening (matches CPU `gb::pairwise_descreening`).
fn pairwise_descreening(r: f32, rho_i_tilde: f32, s_rho_j_tilde: f32) -> f32 {
    if (r + s_rho_j_tilde <= rho_i_tilde) {
        return 0.0;
    }
    var l: f32;
    if (r - s_rho_j_tilde < rho_i_tilde) {
        l = rho_i_tilde;
    } else {
        l = r - s_rho_j_tilde;
    }
    let u = r + s_rho_j_tilde;
    if (u <= 0.0 || l <= 0.0) {
        return 0.0;
    }
    let inv_l = 1.0 / l;
    let inv_u = 1.0 / u;
    let term1 = 0.5 * (inv_l - inv_u);
    let term2 = (r / 4.0) * (inv_u * inv_u - inv_l * inv_l);
    let term3 = (1.0 / (2.0 * r)) * log(l / u);
    let term4 = (s_rho_j_tilde * s_rho_j_tilde - r * r) / (4.0 * r)
              * (inv_u * inv_u - inv_l * inv_l);
    return term1 + term2 + term3 + term4;
}

@compute @workgroup_size(64)
fn born_radii(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_atoms) {
        return;
    }
    let pi = positions[i].xyz;
    let rho_i = rho[i];
    let rho_i_tilde = rho_tilde[i];

    var integral: f32 = 0.0;
    let count = nbr_count[i];
    let start = nbr_start[i];
    for (var k: u32 = 0u; k < count; k = k + 1u) {
        let j = nbr_indices[start + k];
        let dx = positions[j].xyz - pi;
        let r2 = dot(dx, dx);
        if (r2 > params.cutoff_sq || r2 < 1e-18) {
            continue;
        }
        let r = sqrt(r2);
        let s_rho_j_tilde = scale[j] * rho_tilde[j];
        integral = integral + pairwise_descreening(r, rho_i_tilde, s_rho_j_tilde);
    }

    // OBC II tanh transform: 1/R_eff = 1/ρ̃ - tanh(αψ - βψ² + γψ³) / ρ.
    let psi = integral * rho_i_tilde;
    let tanh_arg = OBC_ALPHA * psi - OBC_BETA * psi * psi + OBC_GAMMA * psi * psi * psi;
    let inv = 1.0 / rho_i_tilde - tanh(tanh_arg) / rho_i;
    let floor_radius = max(rho_i_tilde, 0.5);
    var r_eff_val = floor_radius;
    if (inv > 0.0) {
        let r_candidate = 1.0 / inv;
        if (r_candidate > floor_radius) {
            r_eff_val = r_candidate;
        }
    }
    r_eff[i] = r_eff_val;
}
