// Generalized-Born OBC II pair-force kernel.
// One thread per atom i.  Reads effective Born radii produced by the
// `gb_born.wgsl` kernel.  Output: per-atom GB pair force (kJ/mol/Å).
//
// Sign convention matches `energy::forces_gb::add_gb_forces_soa`
// exactly:  F_i = Σ_j prefactor_kj · q_i q_j / f_GB² · df_GB/dr · r̂_ij
// where r̂_ij = (r_j - r_i) / r and the prefactor *for forces* is the
// positive version (the energy's −½ becomes +1 after differentiation
// because every pair contributes twice to the sum).

struct Params {
    n_atoms: u32,
    cutoff_sq: f32,
    /// Pre-multiplied (1/εsolute − 1/εwater) × 332.0637 × 4.184
    /// — positive value, in kJ·Å/mol/e².
    prefactor_kj: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> charges: array<f32>;
@group(0) @binding(3) var<storage, read> r_eff: array<f32>;
@group(0) @binding(4) var<storage, read_write> forces: array<vec4<f32>>;

@compute @workgroup_size(64)
fn gb_force(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_atoms) {
        return;
    }
    let qi = charges[i];
    let pi = positions[i].xyz;
    let ri = r_eff[i];
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    if (qi == 0.0) {
        forces[i] = vec4<f32>(acc, 0.0);
        return;
    }
    for (var j: u32 = 0u; j < params.n_atoms; j = j + 1u) {
        if (j == i) {
            continue;
        }
        let qj = charges[j];
        if (qj == 0.0) {
            continue;
        }
        let dx = positions[j].xyz - pi;
        let r2 = dot(dx, dx);
        if (r2 > params.cutoff_sq || r2 < 1e-18) {
            continue;
        }
        let r = sqrt(r2);
        let rij_prod = ri * r_eff[j];
        let exp_val = exp(-r2 / (4.0 * rij_prod));
        let f_gb_sq = r2 + rij_prod * exp_val;
        let f_gb = sqrt(f_gb_sq);
        let d_fgb_sq_dr = 2.0 * r - 0.5 * r * exp_val;
        let d_fgb_dr = d_fgb_sq_dr / (2.0 * f_gb);
        let coeff = params.prefactor_kj * (qi * qj) / f_gb_sq * d_fgb_dr / r;
        acc = acc + dx * coeff;
    }
    forces[i] = vec4<f32>(acc, 0.0);
}
