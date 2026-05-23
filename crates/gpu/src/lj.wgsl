// Lennard-Jones pair force kernel.
//
// One thread per atom i. Each thread loops over all other atoms j,
// reads parameters from a table indexed by atom-type, and accumulates
// the LJ force on atom i.
//
// Excluded 1-2 / 1-3 / 1-4 pairs are skipped via a flat bitmap
// `exclusions` indexed as `i * N + j` (one bit per pair); 1 = skip.
//
// Units: kJ/mol/Å for force, Å for positions, kJ/mol for ε (note: the
// caller pre-multiplies by 4.184 from CHARMM kcal/mol).  Rmin (not σ)
// to match CHARMM's `V = ε [(Rmin/r)^12 − 2(Rmin/r)^6]` form.

struct Params {
    n_atoms: u32,
    cutoff_sq: f32,  // squared cutoff in Å²
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;     // .xyz position, .w unused
@group(0) @binding(2) var<storage, read> type_index: array<u32>;          // per-atom type index
@group(0) @binding(3) var<storage, read> lj_params: array<vec2<f32>>;     // .x = epsilon (kJ/mol), .y = rmin
@group(0) @binding(4) var<storage, read> exclusions: array<u32>;          // flat bitmap, n*n bits
@group(0) @binding(5) var<storage, read_write> forces: array<vec4<f32>>;  // output force (.xyz, .w unused)

fn is_excluded(i: u32, j: u32) -> bool {
    let bit_idx = i * params.n_atoms + j;
    let word = exclusions[bit_idx / 32u];
    let mask = 1u << (bit_idx % 32u);
    return (word & mask) != 0u;
}

@compute @workgroup_size(64)
fn lj_force(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_atoms) {
        return;
    }
    let pi = positions[i].xyz;
    let ti = type_index[i];
    let pi_eps = lj_params[ti].x;
    let pi_rmin_half = lj_params[ti].y;
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    for (var j: u32 = 0u; j < params.n_atoms; j = j + 1u) {
        if (j == i) {
            continue;
        }
        if (is_excluded(i, j)) {
            continue;
        }
        let dx = positions[j].xyz - pi;
        let r2 = dot(dx, dx);
        if (r2 > params.cutoff_sq || r2 < 1e-6) {
            continue;
        }
        let tj = type_index[j];
        let pj_eps = lj_params[tj].x;
        let pj_rmin_half = lj_params[tj].y;
        // Lorentz-Berthelot: ε = sqrt(εi εj), Rmin = Rmin/2_i + Rmin/2_j.
        let eps = sqrt(pi_eps * pj_eps);
        let rmin = pi_rmin_half + pj_rmin_half;
        let r = sqrt(r2);
        let ratio = rmin / r;
        let r2_ratio = ratio * ratio;
        let r6_ratio = r2_ratio * r2_ratio * r2_ratio;
        let r12_ratio = r6_ratio * r6_ratio;
        // V_LJ = ε [(Rmin/r)^12 − 2 (Rmin/r)^6]
        // F_i = (12 ε / r²) × [(Rmin/r)^6 − (Rmin/r)^12] × (r_j − r_i)
        // (matches forces_nonbonded::add_nonbonded_forces sign exactly)
        let coeff = 12.0 * eps / r2 * (r6_ratio - r12_ratio);
        acc = acc + dx * coeff;
    }
    forces[i] = vec4<f32>(acc, 0.0);
}
