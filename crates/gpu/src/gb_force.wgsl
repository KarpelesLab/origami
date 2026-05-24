// Generalized-Born OBC II pair-force kernel — Verlet-list variant.
// One thread per atom i.  Walks i's precomputed neighbour list (built
// at the 20-Å Born cutoff + skin, *not* the 10-Å pair cutoff — see
// below).  Reads effective Born radii produced by `gb_born.wgsl`.
//
// Output: per-atom GB pair force (kJ/mol/Å).
//
// Sign convention matches `energy::forces_gb::add_gb_forces_soa`
// exactly:  F_i = Σ_j prefactor_kj · q_i q_j / f_GB² · df_GB/dr · r̂_ij
// where r̂_ij = (r_j - r_i) / r and the prefactor *for forces* is the
// positive version (the energy's −½ becomes +1 after differentiation
// because every pair contributes twice to the sum).
//
// Why one neighbour list for two cutoffs: the GB Born-radius pass
// needs a 20 Å list; the GB pair force only needs 10 Å.  Maintaining
// two CSR lists doubles the CPU drift-check + cell-list cost.
// Instead the kernel walks the wider 20 Å list and filters internally
// with `r2 > pair_cutoff_sq`.  At a 10 Å pair cutoff the 20 Å list
// is ~8× larger, but most extra entries reject after a single
// distance compute — the inner-loop cost penalty is ~30 % at the
// scales where the GPU is competitive in the first place.

struct Params {
    n_atoms: u32,
    cutoff_sq: f32,        // squared pair cutoff (10 Å)² in Å²
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
@group(0) @binding(5) var<storage, read> nbr_count: array<u32>;   // per-atom neighbour count
@group(0) @binding(6) var<storage, read> nbr_start: array<u32>;   // per-atom offset into nbr_indices
@group(0) @binding(7) var<storage, read> nbr_indices: array<u32>; // flat neighbour-j array

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
    let count = nbr_count[i];
    let start = nbr_start[i];
    for (var k: u32 = 0u; k < count; k = k + 1u) {
        let j = nbr_indices[start + k];
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
