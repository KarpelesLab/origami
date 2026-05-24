// GPU bonded force kernels: bond, angle, dihedral, improper.
//
// Architecture: one thread per atom.  Each thread walks its own CSR
// participation lists (built on the CPU at accelerator construction)
// and accumulates force contributions directly into its slot of the
// shared `forces` buffer — no atomics needed because each thread
// writes only to forces[i] where i is the thread index.
//
// Each kernel ACCUMULATES (read-modify-write on forces[i]).  The
// caller is expected to zero the forces buffer once at the start of
// the per-step force evaluation, then chain bond → angle → dihedral
// → improper → nonbonded → GB.  Pass-to-pass dependencies are
// automatic across separate `begin_compute_pass` boundaries.
//
// Unit / sign conventions match `crates/energy/src/forces_bonded.rs`
// exactly: kJ/mol/Å throughout (the CPU multiplies kcal→kJ; here we
// pre-bake the conversion into the uploaded `k` parameters).

struct GlobalParams {
    n_atoms: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// ---- Common bindings (group 0) ----
@group(0) @binding(0) var<uniform> globals: GlobalParams;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> forces: array<vec4<f32>>;

// ---- Bond term ----
struct BondTerm {
    a: u32,
    b: u32,
    k_kj: f32,   // kJ/mol/Å²  (CPU pre-multiplies by 4.184)
    r0_a: f32,   // Å
}
@group(0) @binding(3) var<storage, read> bond_terms: array<BondTerm>;
@group(0) @binding(4) var<storage, read> atom_bond_count: array<u32>;
@group(0) @binding(5) var<storage, read> atom_bond_start: array<u32>;
@group(0) @binding(6) var<storage, read> atom_bond_index: array<u32>;

@compute @workgroup_size(64)
fn bond_force(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= globals.n_atoms) {
        return;
    }
    let pi = positions[i].xyz;
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    let count = atom_bond_count[i];
    let start = atom_bond_start[i];
    for (var k: u32 = 0u; k < count; k = k + 1u) {
        let term = bond_terms[atom_bond_index[start + k]];
        // Partner is whichever of (a, b) isn't `i`.
        let partner = select(term.b, term.a, i == term.b);
        let d = positions[partner].xyz - pi;
        let r2 = dot(d, d);
        if (r2 < 1e-18) { continue; }
        let r = sqrt(r2);
        let dr = r - term.r0_a;
        let mag = 2.0 * term.k_kj * dr;
        acc = acc + d * (mag / r);
    }
    forces[i] = forces[i] + vec4<f32>(acc, 0.0);
}

// ---- Angle term ----
struct AngleTerm {
    a: u32,
    b: u32,   // central
    c: u32,
    _pad: u32,
    k_kj: f32,        // kJ/mol/rad²
    theta0_rad: f32,
    _pad2: f32,
    _pad3: f32,
}
@group(0) @binding(7) var<storage, read> angle_terms: array<AngleTerm>;
@group(0) @binding(8) var<storage, read> atom_angle_count: array<u32>;
@group(0) @binding(9) var<storage, read> atom_angle_start: array<u32>;
@group(0) @binding(10) var<storage, read> atom_angle_index: array<u32>;

@compute @workgroup_size(64)
fn angle_force(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= globals.n_atoms) {
        return;
    }
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    let count = atom_angle_count[i];
    let start = atom_angle_start[i];
    for (var k: u32 = 0u; k < count; k = k + 1u) {
        let term = angle_terms[atom_angle_index[start + k]];
        let pa = positions[term.a].xyz;
        let pb = positions[term.b].xyz;
        let pc = positions[term.c].xyz;
        let u = pa - pb;
        let v = pc - pb;
        let u_norm_sq = dot(u, u);
        let v_norm_sq = dot(v, v);
        if (u_norm_sq < 1e-18 || v_norm_sq < 1e-18) { continue; }
        let u_norm = sqrt(u_norm_sq);
        let v_norm = sqrt(v_norm_sq);
        let u_hat = u / u_norm;
        let v_hat = v / v_norm;
        let cos_theta = clamp(dot(u_hat, v_hat), -1.0, 1.0);
        let sin_sq = 1.0 - cos_theta * cos_theta;
        if (sin_sq < 1e-18) { continue; }
        let sin_theta = sqrt(sin_sq);
        let theta = acos(cos_theta);
        let dvdtheta = 2.0 * term.k_kj * (theta - term.theta0_rad);
        let coeff = dvdtheta / sin_theta;
        let f_a = (v_hat - u_hat * cos_theta) * (coeff / u_norm);
        let f_c = (u_hat - v_hat * cos_theta) * (coeff / v_norm);
        let f_b = -(f_a + f_c);
        // Pick the contribution that goes to this thread's atom.
        if (i == term.a) {
            acc = acc + f_a;
        } else if (i == term.b) {
            acc = acc + f_b;
        } else {
            acc = acc + f_c;
        }
    }
    forces[i] = forces[i] + vec4<f32>(acc, 0.0);
}

// ---- Dihedral term ----
//
// Multi-term periodic dihedral: V = Σ kₙ × (1 + cos(n·φ − δ)).
// CPU representation: each dihedral has up to N_DIHEDRAL_TERMS periodic
// terms.  Stored inline (`term0..term3`) — actual count is in `n_terms`.
// In practice >95 % of dihedrals have 1-2 terms; up to 4 keeps the
// struct fixed-size at 64 bytes (vec4 alignment friendly).

const MAX_PERIODIC_TERMS: u32 = 4u;

struct PeriodicTerm {
    k_kj: f32,       // kJ/mol
    n: f32,          // multiplicity (stored as f32 to keep alignment)
    delta_rad: f32,
    _pad: f32,
}

struct DihedralTerm {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
    n_terms: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    term0: PeriodicTerm,
    term1: PeriodicTerm,
    term2: PeriodicTerm,
    term3: PeriodicTerm,
}
@group(0) @binding(11) var<storage, read> dihedral_terms: array<DihedralTerm>;
@group(0) @binding(12) var<storage, read> atom_dihedral_count: array<u32>;
@group(0) @binding(13) var<storage, read> atom_dihedral_start: array<u32>;
@group(0) @binding(14) var<storage, read> atom_dihedral_index: array<u32>;

// ---- Improper term ---- (separate buffer because params differ)
struct ImproperTerm {
    a: u32,        // central
    b: u32,
    c: u32,
    d: u32,
    k_kj: f32,     // kJ/mol/rad²
    omega0_rad: f32,
    _pad0: f32,
    _pad1: f32,
}
@group(0) @binding(15) var<storage, read> improper_terms: array<ImproperTerm>;
@group(0) @binding(16) var<storage, read> atom_improper_count: array<u32>;
@group(0) @binding(17) var<storage, read> atom_improper_start: array<u32>;
@group(0) @binding(18) var<storage, read> atom_improper_index: array<u32>;

/// Returns `(dphi_da, dphi_db, dphi_dc, dphi_dd, phi)`.
/// Mirrors `crates/energy/src/forces_bonded.rs::dihedral_gradient` with
/// identical sign conventions.  Returns `phi = 1e9` (sentinel) if the
/// geometry is degenerate; the caller then skips the term.
struct DihedralGradient {
    dphi_da: vec3<f32>,
    dphi_db: vec3<f32>,
    dphi_dc: vec3<f32>,
    dphi_dd: vec3<f32>,
    phi: f32,
    valid: u32,
}

fn dihedral_gradient(pa: vec3<f32>, pb: vec3<f32>, pc: vec3<f32>, pd: vec3<f32>) -> DihedralGradient {
    var g: DihedralGradient;
    let b1 = pb - pa;
    let b2 = pc - pb;
    let b3 = pd - pc;
    let b2_norm_sq = dot(b2, b2);
    if (b2_norm_sq < 1e-18) {
        g.valid = 0u;
        return g;
    }
    let b2_norm = sqrt(b2_norm_sq);
    let s = cross(b1, b2);
    let t = cross(b2, b3);
    let s_norm_sq = dot(s, s);
    let t_norm_sq = dot(t, t);
    if (s_norm_sq < 1e-18 || t_norm_sq < 1e-18) {
        g.valid = 0u;
        return g;
    }
    let dphi_da = s * (b2_norm / s_norm_sq);
    let dphi_dd = -t * (b2_norm / t_norm_sq);
    let inv_b2_sq = 1.0 / b2_norm_sq;
    let f1 = dot(b1, b2) * inv_b2_sq;
    let f2 = dot(b3, b2) * inv_b2_sq;
    let dphi_db = -(f1 + 1.0) * dphi_da + f2 * dphi_dd;
    let dphi_dc = -(dphi_da + dphi_db + dphi_dd);
    let m1 = cross(s, b2 / b2_norm);
    let x = dot(s, t);
    let y = dot(m1, t);
    let phi = atan2(y, x);
    g.dphi_da = dphi_da;
    g.dphi_db = dphi_db;
    g.dphi_dc = dphi_dc;
    g.dphi_dd = dphi_dd;
    g.phi = phi;
    g.valid = 1u;
    return g;
}

@compute @workgroup_size(64)
fn dihedral_force(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= globals.n_atoms) {
        return;
    }
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    let count = atom_dihedral_count[i];
    let start = atom_dihedral_start[i];
    for (var k: u32 = 0u; k < count; k = k + 1u) {
        let term = dihedral_terms[atom_dihedral_index[start + k]];
        let pa = positions[term.a].xyz;
        let pb = positions[term.b].xyz;
        let pc = positions[term.c].xyz;
        let pd = positions[term.d].xyz;
        let g = dihedral_gradient(pa, pb, pc, pd);
        if (g.valid == 0u) { continue; }
        // Sum dV/dφ across all periodic terms.
        var dvdphi: f32 = 0.0;
        if (term.n_terms > 0u) {
            let arg = term.term0.n * g.phi - term.term0.delta_rad;
            dvdphi = dvdphi - term.term0.k_kj * term.term0.n * sin(arg);
        }
        if (term.n_terms > 1u) {
            let arg = term.term1.n * g.phi - term.term1.delta_rad;
            dvdphi = dvdphi - term.term1.k_kj * term.term1.n * sin(arg);
        }
        if (term.n_terms > 2u) {
            let arg = term.term2.n * g.phi - term.term2.delta_rad;
            dvdphi = dvdphi - term.term2.k_kj * term.term2.n * sin(arg);
        }
        if (term.n_terms > 3u) {
            let arg = term.term3.n * g.phi - term.term3.delta_rad;
            dvdphi = dvdphi - term.term3.k_kj * term.term3.n * sin(arg);
        }
        // F_X = -dV/dφ × dφ/dr_X.  Same sign convention as CPU.
        var grad: vec3<f32>;
        if (i == term.a) { grad = g.dphi_da; }
        else if (i == term.b) { grad = g.dphi_db; }
        else if (i == term.c) { grad = g.dphi_dc; }
        else { grad = g.dphi_dd; }
        acc = acc - grad * dvdphi;
    }
    forces[i] = forces[i] + vec4<f32>(acc, 0.0);
}

@compute @workgroup_size(64)
fn improper_force(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= globals.n_atoms) {
        return;
    }
    var acc = vec3<f32>(0.0, 0.0, 0.0);
    let count = atom_improper_count[i];
    let start = atom_improper_start[i];
    let pi_two = 6.283185307;
    let pi_one = 3.141592653;
    let inv_pi_two = 1.0 / pi_two;
    for (var k: u32 = 0u; k < count; k = k + 1u) {
        let term = improper_terms[atom_improper_index[start + k]];
        let pa = positions[term.a].xyz;
        let pb = positions[term.b].xyz;
        let pc = positions[term.c].xyz;
        let pd = positions[term.d].xyz;
        let g = dihedral_gradient(pa, pb, pc, pd);
        if (g.valid == 0u) { continue; }
        var domega = g.phi - term.omega0_rad;
        // Wrap into (-π, π] — math wrap, safe on NaN (returns NaN
        // through but the downstream `2 * k * domega` becomes NaN and
        // forces[i] becomes NaN, surfacing the error rather than
        // hanging the GPU).
        domega = domega - pi_two * floor((domega + pi_one) * inv_pi_two);
        let dvdomega = 2.0 * term.k_kj * domega;
        var grad: vec3<f32>;
        if (i == term.a) { grad = g.dphi_da; }
        else if (i == term.b) { grad = g.dphi_db; }
        else if (i == term.c) { grad = g.dphi_dc; }
        else { grad = g.dphi_dd; }
        acc = acc - grad * dvdomega;
    }
    forces[i] = forces[i] + vec4<f32>(acc, 0.0);
}

@compute @workgroup_size(64)
fn zero_forces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= globals.n_atoms) {
        return;
    }
    forces[i] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
}
