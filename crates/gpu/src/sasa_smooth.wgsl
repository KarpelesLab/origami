// Smooth-coverage SASA: differentiable variant of `sasa.wgsl`.
//
// Replaces the binary buried/accessible verdict with a sigmoidal
// weight in [0, 1] that smoothly transitions across the neighbour-
// sphere boundary.  Per-dot accessibility is the product of (1 - b_j)
// across all neighbours j.  Per-atom area is the sum over dots.
// Forces follow by chain rule — see `sasa_smooth_force` below.
//
// Two entry points:
//   - `sasa_smooth_area`: per-atom accessible area (kJ/mol/Å² × area
//     gets you energy in `add_sasa_forces_*`-compatible units).
//   - `sasa_smooth_force`: per-atom force F_x = −γ_i · ∂A_i/∂r_x
//     summed over every i where x participates (i == x for the
//     diagonal, plus every i that has x as a SASA neighbour).
//
// The smoothing width σ controls the boundary thickness.  σ → 0
// recovers the binary case (with discontinuous forces); larger σ
// smooths out the boundary but biases per-atom areas slightly low
// near complete burial.  σ = 0.3 Å is a reasonable middle ground.

const N_DOTS: u32 = 256u;

struct Params {
    n_atoms: u32,
    sigma: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> radii: array<f32>;
@group(0) @binding(3) var<storage, read> gammas: array<f32>;                 // kJ/mol/Å² per atom
@group(0) @binding(4) var<storage, read> dots: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> nbr_count: array<u32>;
@group(0) @binding(6) var<storage, read> nbr_start: array<u32>;
@group(0) @binding(7) var<storage, read> nbr_indices: array<u32>;
@group(0) @binding(8) var<storage, read_write> per_atom_area: array<f32>;
@group(0) @binding(9) var<storage, read_write> forces: array<vec4<f32>>;
// Pre-computed per-dot W_k values, laid out flat as
// `w_cache[i * N_DOTS + k]`.  Populated by `sasa_smooth_precompute_w`
// once per force eval; the force kernel reads from it instead of
// re-deriving W_k for every (x, i, k, j) tuple — a ~50× speedup at
// the scales we care about.
@group(0) @binding(10) var<storage, read_write> w_cache: array<f32>;

// Sigmoidal "buried-ness".  d = distance from dot to neighbour j's
// centre, r_j = neighbour's expanded radius.  When d == r_j the dot
// sits on j's boundary and b = 0.5.  Smaller d → more buried (b → 1).
fn buried(d: f32, r_j: f32, sigma: f32) -> f32 {
    let t = (r_j - d) / sigma;
    return 1.0 / (1.0 + exp(-t));
}

// d(buried)/d(d) = -b(1-b)/σ.  Derivative of `buried` wrt distance
// d, used in the force chain rule.
fn buried_deriv_wrt_d(b: f32, sigma: f32) -> f32 {
    return -b * (1.0 - b) / sigma;
}

// Pre-compute per-dot W_k for every atom × dot.  One thread per
// (atom_i × dot_k) pair — workgroup_size = 64, total threads
// = N_DOTS × n_atoms.  The force kernel then reads W from the
// cache.
@compute @workgroup_size(64)
fn sasa_smooth_precompute_w(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    let total = params.n_atoms * N_DOTS;
    if (flat >= total) {
        return;
    }
    let i = flat / N_DOTS;
    let k = flat % N_DOTS;
    let pi = positions[i].xyz;
    let ri = radii[i];
    let sigma = params.sigma;
    let count = nbr_count[i];
    let start = nbr_start[i];
    let dot_pos = pi + dots[k].xyz * ri;
    var w: f32 = 1.0;
    for (var t: u32 = 0u; t < count; t = t + 1u) {
        let j = nbr_indices[start + t];
        let dv = dot_pos - positions[j].xyz;
        let d = length(dv);
        let b = buried(d, radii[j], sigma);
        w = w * (1.0 - b);
    }
    w_cache[flat] = w;
}

@compute @workgroup_size(64)
fn sasa_smooth_area(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_atoms) {
        return;
    }
    let pi = positions[i].xyz;
    let ri = radii[i];
    let sigma = params.sigma;
    let count = nbr_count[i];
    let start = nbr_start[i];

    var accessible_sum: f32 = 0.0;
    for (var k: u32 = 0u; k < N_DOTS; k = k + 1u) {
        let dot_pos = pi + dots[k].xyz * ri;
        var w: f32 = 1.0;
        for (var t: u32 = 0u; t < count; t = t + 1u) {
            let j = nbr_indices[start + t];
            let dv = dot_pos - positions[j].xyz;
            let d = length(dv);
            let b = buried(d, radii[j], sigma);
            w = w * (1.0 - b);
        }
        accessible_sum = accessible_sum + w;
    }
    let four_pi = 12.566370614;
    per_atom_area[i] = (accessible_sum / f32(N_DOTS)) * four_pi * ri * ri;
}

// SASA force kernel.
//
// For atom x, the total force is
//   F_x = -∂E/∂r_x  where E = Σ_i γ_i A_i, summed over the atoms
//   `i` whose SASA areas depend on r_x (i.e. i == x, or x is in
//   i's SASA neighbour list).
//
// We dispatch one thread per atom x.  Each thread walks its
// REVERSE neighbour list — the atoms i for which x is a SASA
// neighbour — plus i = x for the diagonal term.  Since SASA
// neighbour lists are symmetric (built via `d <= r_i + r_j`), the
// forward and reverse lists are identical, and the same
// `nbr_count[x]` / `nbr_start[x]` / `nbr_indices[]` buffers serve
// both roles.
//
// Per "involved atom i":
//   - The thread walks atom i's N_DOTS test points.
//   - For each dot k on atom i, it recomputes W_k = Π (1 − b_j),
//     and the contribution to ∂W_k/∂r_x is:
//       * if x == i (diagonal): -W_k · Σ_j (b_j' / (1 − b_j)) · û_jk
//         where û_jk = (dot_k − p_j) / d_j
//       * if x ≠ i (off-diagonal, x is one of i's j's):
//         +W_k · (b_x' / (1 − b_x)) · û_xk
//         (only one j matches — the cross-term where j == x)
@compute @workgroup_size(64)
fn sasa_smooth_force(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    if (x >= params.n_atoms) {
        return;
    }
    let px = positions[x].xyz;
    let sigma = params.sigma;

    var f_acc = vec3<f32>(0.0, 0.0, 0.0);

    // ---- Diagonal: i == x.  ∂A_x/∂r_x. ----
    //
    // F_x_diag = +γ_x · (4π R_x²/N) · Σ_k W_k · Σ_j (b_j'/(1-b_j)) · û_jk
    //
    // W_k is read from the precomputed `w_cache` (filled by
    // `sasa_smooth_precompute_w` before this kernel runs); only the
    // j-loop for the gradient sum needs to evaluate b_j(d_jk) per
    // dot — no nested W recomputation.
    let rx = radii[x];
    let gamma_x = gammas[x];
    if (gamma_x != 0.0) {
        let four_pi_rx_sq = 12.566370614 * rx * rx;
        let prefactor = gamma_x * four_pi_rx_sq / f32(N_DOTS);
        let count = nbr_count[x];
        let start = nbr_start[x];
        for (var k: u32 = 0u; k < N_DOTS; k = k + 1u) {
            let dot_pos = px + dots[k].xyz * rx;
            let w = w_cache[x * N_DOTS + k];
            var sum_term = vec3<f32>(0.0, 0.0, 0.0);
            for (var t: u32 = 0u; t < count; t = t + 1u) {
                let j = nbr_indices[start + t];
                let dv = dot_pos - positions[j].xyz;
                let d = length(dv);
                if (d < 1e-6) { continue; }
                let b = buried(d, radii[j], sigma);
                let one_minus_b = 1.0 - b;
                if (one_minus_b < 1e-12) { continue; }
                let b_prime = buried_deriv_wrt_d(b, sigma);
                let u_hat = dv / d;
                sum_term = sum_term + u_hat * (b_prime / one_minus_b);
            }
            f_acc = f_acc + sum_term * (prefactor * w);
        }
    }

    // ---- Off-diagonal: i ≠ x, x is a SASA neighbour of i.  ----
    // F = -γ_i · (4πR_i²/N) · W_k · (b_x'/(1-b_x)) · û  where
    // û = (dot_k − p_x) / d_xk.  W_k for atom i's dot k is
    // pre-cached in `w_cache[i * N_DOTS + k]`.
    let count = nbr_count[x];
    let start = nbr_start[x];
    for (var t: u32 = 0u; t < count; t = t + 1u) {
        let i = nbr_indices[start + t];
        let gamma_i = gammas[i];
        if (gamma_i == 0.0) { continue; }
        let pi = positions[i].xyz;
        let ri = radii[i];
        let four_pi_ri_sq = 12.566370614 * ri * ri;
        let prefactor = gamma_i * four_pi_ri_sq / f32(N_DOTS);
        for (var k: u32 = 0u; k < N_DOTS; k = k + 1u) {
            let dot_pos = pi + dots[k].xyz * ri;
            let w = w_cache[i * N_DOTS + k];
            // Contribution from j == x (the only relevant j for x).
            let dv = dot_pos - px;
            let d = length(dv);
            if (d < 1e-6) { continue; }
            let b = buried(d, rx, sigma);
            let one_minus_b = 1.0 - b;
            if (one_minus_b < 1e-12) { continue; }
            let b_prime = buried_deriv_wrt_d(b, sigma);
            let u_hat = dv / d;
            f_acc = f_acc - u_hat * (prefactor * w * b_prime / one_minus_b);
        }
    }

    forces[x] = forces[x] + vec4<f32>(f_acc, 0.0);
}
