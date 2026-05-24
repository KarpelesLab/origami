// BAOAB Langevin integrator — GPU implementation.
//
// Splits one BAOAB step into two dispatches so the caller can drop a
// force-recompute between them:
//
//   `baoab_first_half`:  B (using forces_total) → A → O → A
//   <caller recomputes forces at the new positions>
//   `baoab_second_half`: B (using new forces_total)
//
// This matches the CPU integrator's structure
// (`crates/dynamics/src/langevin.rs`) one-for-one.  The per-atom
// RNG state buffer holds the xoshiro128++ state used by the O step;
// each kernel invocation advances each atom's stream by three draws
// (one per axis).
//
// Sign / unit conventions match `dynamics::langevin` exactly:
//   forces  kJ/mol/Å, masses Da, positions Å, velocities Å/fs,
//   ACCEL_FACTOR = 1e-4 bridges kJ/mol/Å / Da → Å/fs².

struct Params {
    n_atoms: u32,
    half_dt: f32,
    alpha: f32,            // exp(-γ dt)
    /// σ² = (1 − α²) k_B T · accel_factor   (per-mass scaling done inside)
    o_sigma_sq_base: f32,
    accel_factor: f32,     // 1e-4
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> masses: array<f32>;
@group(0) @binding(4) var<storage, read> forces_total: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> rng_state: array<vec4<u32>>;

// ---- xoshiro128++ ----
//
// 4×u32 state (vs 4×u64 for the CPU's xoshiro256++ — WGSL only has
// u32 native).  Passes BigCrush; produces a different sequence than
// the CPU side, so GPU trajectories diverge chaotically from
// CPU-only runs after a few hundred steps even when the integrator
// math is identical.  Statistically equivalent: same temperature,
// same equipartition.

fn rotl(x: u32, k: u32) -> u32 {
    return (x << k) | (x >> (32u - k));
}

fn next_u32(state: ptr<function, vec4<u32>>) -> u32 {
    let s = *state;
    let result = rotl(s.x + s.w, 7u) + s.x;
    let t = s.y << 9u;
    var s2 = s.z ^ s.x;
    var s3 = s.w ^ s.y;
    var s1 = s.y ^ s2;
    var s0 = s.x ^ s3;
    s2 = s2 ^ t;
    s3 = rotl(s3, 11u);
    *state = vec4<u32>(s0, s1, s2, s3);
    return result;
}

fn next_f32(state: ptr<function, vec4<u32>>) -> f32 {
    // Top 24 bits → f32 in [0, 1) with ~24 bits of mantissa.
    let u = next_u32(state) >> 8u;
    return f32(u) * (1.0 / 16777216.0);
}

// Box-Muller — one Gaussian per call.  No caching across calls (each
// thread has its own state and consumes pairs in sequence).
fn gaussian(state: ptr<function, vec4<u32>>) -> f32 {
    var u1 = next_f32(state);
    if (u1 <= 0.0) {
        u1 = 1e-7;   // never zero — sub-2^-24 probability
    }
    let u2 = next_f32(state);
    let r = sqrt(-2.0 * log(u1));
    let theta = 6.283185307 * u2;   // 2π
    return r * cos(theta);
}

@compute @workgroup_size(64)
fn baoab_first_half(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_atoms) {
        return;
    }
    let mass = masses[i];
    let inv_m_accel = params.accel_factor / mass;
    let half_dt = params.half_dt;
    var pos = positions[i].xyz;
    var vel = velocities[i].xyz;
    let f = forces_total[i].xyz;

    // B: v += a · dt/2
    vel = vel + f * (inv_m_accel * half_dt);

    // A: r += v · dt/2
    pos = pos + vel * half_dt;

    // O: v = α v + σ ξ.  σ² scales as 1/m.
    var state = rng_state[i];
    let sigma = sqrt(params.o_sigma_sq_base / mass);
    let xi_x = gaussian(&state);
    let xi_y = gaussian(&state);
    let xi_z = gaussian(&state);
    vel = vec3<f32>(
        params.alpha * vel.x + sigma * xi_x,
        params.alpha * vel.y + sigma * xi_y,
        params.alpha * vel.z + sigma * xi_z,
    );

    // A: r += v · dt/2
    pos = pos + vel * half_dt;

    positions[i] = vec4<f32>(pos, 0.0);
    velocities[i] = vec4<f32>(vel, 0.0);
    rng_state[i] = state;
}

@compute @workgroup_size(64)
fn baoab_second_half(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_atoms) {
        return;
    }
    let mass = masses[i];
    let inv_m_accel = params.accel_factor / mass;
    let half_dt = params.half_dt;
    let f = forces_total[i].xyz;
    let vel = velocities[i].xyz + f * (inv_m_accel * half_dt);
    velocities[i] = vec4<f32>(vel, 0.0);
}
