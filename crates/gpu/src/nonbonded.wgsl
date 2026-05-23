// Nonbonded pair force kernel — LJ (12-6) + reaction-field Coulomb
// in a single pass.  One thread per atom i, loops over all j.
//
// Excluded 1-2 / 1-3 / 1-4 pairs are skipped via a flat bitmap
// `exclusions` indexed as `i * N + j` (one bit per pair); 1 = skip.
//
// Units: kJ/mol/Å for force, Å for positions, kJ/mol for ε (caller
// pre-multiplies by 4.184 from CHARMM kcal/mol), e for charges.
//
// Reaction-field Coulomb is the Tironi ε_RF→∞ form (matches
// `energy::nonbonded::CoulombRf`):
//   F_RF(r) = k_e q_i q_j · (1/r³ − 1/Rc³) · (r_j − r_i)
// where k_e = 332.0637 kcal·Å/mol/e² × 4.184 = 1389.354 kJ·Å/mol/e².
// Both V_RF and F_RF go smoothly to zero at r = Rc.

struct Params {
    n_atoms: u32,
    cutoff_sq: f32,  // squared cutoff in Å²
    inv_rc3: f32,    // 1/Rc³ — pre-computed reaction-field constant
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;     // .xyz position, .w unused
@group(0) @binding(2) var<storage, read> type_index: array<u32>;          // per-atom type index
@group(0) @binding(3) var<storage, read> lj_params: array<vec2<f32>>;     // .x = epsilon (kJ/mol), .y = rmin/2 (Å)
@group(0) @binding(4) var<storage, read> charges: array<f32>;             // per-atom partial charge (e)
@group(0) @binding(5) var<storage, read> exclusions: array<u32>;          // flat bitmap, n*n bits
@group(0) @binding(6) var<storage, read_write> forces: array<vec4<f32>>;  // output (.xyz, .w unused)

const COULOMB_K_KJ: f32 = 1389.35455;  // 332.0637 × 4.184

fn is_excluded(i: u32, j: u32) -> bool {
    let bit_idx = i * params.n_atoms + j;
    let word = exclusions[bit_idx / 32u];
    let mask = 1u << (bit_idx % 32u);
    return (word & mask) != 0u;
}

@compute @workgroup_size(64)
fn nonbonded_force(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_atoms) {
        return;
    }
    let pi = positions[i].xyz;
    let ti = type_index[i];
    let qi = charges[i];
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
        let r = sqrt(r2);
        let inv_r2 = 1.0 / r2;

        // ---- LJ ----
        let tj = type_index[j];
        let pj_eps = lj_params[tj].x;
        let pj_rmin_half = lj_params[tj].y;
        let eps = sqrt(pi_eps * pj_eps);
        let rmin = pi_rmin_half + pj_rmin_half;
        let ratio = rmin / r;
        let r2_ratio = ratio * ratio;
        let r6_ratio = r2_ratio * r2_ratio * r2_ratio;
        let r12_ratio = r6_ratio * r6_ratio;
        // F_LJ = (12 ε / r²) × [(Rmin/r)⁶ − (Rmin/r)¹²] × (r_j − r_i)
        let lj_coeff = 12.0 * eps * inv_r2 * (r6_ratio - r12_ratio);

        // ---- Reaction-field Coulomb ----
        // F_RF = k_e q_i q_j · (1/r³ − 1/Rc³) · (r_j − r_i)
        let qq = qi * charges[j];
        let coul_coeff = -COULOMB_K_KJ * qq * (inv_r2 / r - params.inv_rc3);

        acc = acc + dx * (lj_coeff + coul_coeff);
    }
    forces[i] = vec4<f32>(acc, 0.0);
}
