//! Langevin molecular dynamics with the BAOAB integrator.
//!
//! BAOAB is Leimkuhler & Matthews 2013's operator splitting:
//!
//! ```text
//!   B: v ← v + (F(r)/m) (dt/2)
//!   A: r ← r + v (dt/2)
//!   O: v ← α v + sqrt((1 - α²) k_B T / m) · ξ        (per atom, ξ ~ N(0,I))
//!   A: r ← r + v (dt/2)
//!   B: v ← v + (F(r)/m) (dt/2)
//! ```
//!
//! with `α = exp(−γ dt)`. Forces are evaluated once per step (the trailing
//! B step's `F(r')` is reused as the leading B step's `F(r)` next iteration).
//! The integrator is time-reversible, second-order accurate in position,
//! and configurationally exact (its invariant distribution matches the
//! Boltzmann distribution to higher order than velocity-Verlet variants).
//!
//! ## Unit system
//!
//! - Positions: Å
//! - Velocities: Å/fs
//! - Masses: Da (atomic mass units)
//! - Forces: kJ/mol/Å
//! - Time: fs
//! - Energy: kJ/mol
//! - Temperature: K
//!
//! Acceleration `a [Å/fs²] = (F[kJ/mol/Å] / m[Da]) * ACCEL_FACTOR`
//! where `ACCEL_FACTOR = 1e-4` is the dimensional bridge derived from
//! `kJ/mol/Å / Da → m/s² → Å/fs²`. Equivalently, the kinetic-energy
//! conversion `Σ ½ m v² [Da·Å²/fs²] = KE[kJ/mol] · ACCEL_FACTOR` falls
//! out of the same constants. The bookkeeping is collected in
//! [`AccelConstants`] so the integrator inner loop stays clean.

use chem::ForceField;
use energy::{total_force_with_scratch, ForceScratch};
use energy::DEFAULT_CUTOFF_A;
use geom::{Structure, TopologyGraph, Vec3};

use crate::rng::Xoshiro256pp;

/// R = N_A · k_B in kJ/mol/K (i.e. the gas constant). At the per-molecule
/// scale this is the right Boltzmann constant for energies expressed in
/// kJ/mol.
pub const BOLTZMANN_KJ_PER_MOL_K: f64 = 8.314_462_618e-3;

/// Force-to-acceleration unit bridge. See module docs.
pub const ACCEL_FACTOR: f64 = 1.0e-4;

/// Knobs for [`run_langevin`].
#[derive(Debug, Clone, Copy)]
pub struct LangevinOptions {
    /// Integration timestep in femtoseconds. Default 1 fs.
    pub dt_fs: f64,
    /// Target heat-bath temperature in kelvin. Default 310 K (body temp).
    pub temperature_k: f64,
    /// Friction coefficient γ in ps⁻¹. Default 1.0 ps⁻¹.
    pub friction_ps_inv: f64,
    /// Number of integrator steps to run.
    pub steps: usize,
    /// Save a frame every `save_every` steps. 0 disables the callback.
    pub save_every: usize,
    /// RNG seed (deterministic; same seed reproduces the trajectory).
    pub seed: u64,
    /// If `true`, draw initial velocities from Maxwell-Boltzmann at
    /// `temperature_k` with the centre-of-mass velocity zeroed. If
    /// `false`, start from rest.
    pub randomise_initial_velocities: bool,
    /// Include SASA forces (PSA.2). Slow; off by default since the
    /// numerical SASA gradient costs ~100 ms per call on Trp-cage.
    pub include_sasa: bool,
    /// Apply SHAKE iterative bond-length constraints to every X-H
    /// bond after each position half-step. Freezes the high-frequency
    /// hydrogen stretches so the integrator can step at dt = 2 fs
    /// without the bond-vibration aliasing that forces dt ≤ 1 fs in
    /// unconstrained MD. Off by default.
    pub constrain_h_bonds: bool,
}

impl Default for LangevinOptions {
    fn default() -> Self {
        Self {
            dt_fs: 1.0,
            temperature_k: 310.0,
            friction_ps_inv: 1.0,
            steps: 1000,
            save_every: 10,
            seed: 0,
            randomise_initial_velocities: true,
            include_sasa: false,
            constrain_h_bonds: false,
        }
    }
}

/// Snapshot delivered to the frame callback.
pub struct LangevinFrame<'a> {
    pub step: usize,
    pub time_fs: f64,
    pub instantaneous_temperature_k: f64,
    pub kinetic_energy_kj_mol: f64,
    pub structure: &'a Structure,
}

/// Per-run statistics returned by [`run_langevin`].
#[derive(Debug, Clone)]
pub struct LangevinSummary {
    pub steps_run: usize,
    pub temperature_mean_k: f64,
    pub temperature_stddev_k: f64,
    pub equipartition_ratio: f64,
    pub final_kinetic_energy_kj_mol: f64,
    pub atoms_count: usize,
    pub diverged: bool,
    /// Count of integrator half-steps where the SHAKE iteration did
    /// not converge within `SHAKE_MAX_ITERS`. Zero in a healthy run;
    /// non-zero with `constrain_h_bonds = true` indicates the timestep
    /// is too large for some constraint to settle.
    pub shake_failures: usize,
}

/// One BAOAB Langevin trajectory.
///
/// `structure` is mutated in place to hold the final configuration. The
/// callback (if any) receives a fresh snapshot each `save_every` steps,
/// including step 0 (initial state) and the final step.
pub fn run_langevin<F>(
    structure: &mut Structure,
    graph: &TopologyGraph,
    ff: &ForceField,
    opts: LangevinOptions,
    mut callback: F,
) -> LangevinSummary
where
    F: FnMut(LangevinFrame<'_>),
{
    let n = structure.atom_count();
    let masses = collect_masses(structure);
    let mut velocities = vec![Vec3::zeros(); n];
    let mut rng = Xoshiro256pp::from_seed(opts.seed);

    if opts.randomise_initial_velocities {
        initialise_maxwell_boltzmann(&mut velocities, &masses, opts.temperature_k, &mut rng);
    }

    let alpha = (-opts.friction_ps_inv * opts.dt_fs * 1.0e-3).exp(); // γ ps⁻¹ → fs⁻¹
    let kbt = BOLTZMANN_KJ_PER_MOL_K * opts.temperature_k;
    let half_dt = 0.5 * opts.dt_fs;
    let dof = (3 * n) as f64;

    // Allocate one scratch + force buffer for the whole run — the SoA
    // nonbonded kernel reads/writes through this scratch, so we pay the
    // O(n²) exclusion bitmap build exactly once.
    let mut scratch = ForceScratch::new(structure, graph, ff);
    let mut forces: Vec<Vec3> = Vec::with_capacity(n);
    total_force_with_scratch(
        structure,
        graph,
        ff,
        DEFAULT_CUTOFF_A,
        opts.include_sasa,
        &mut scratch,
        &mut forces,
    );

    // Optional SHAKE constraints (X-H bonds). Build once at run start;
    // the constraint list is static for the trajectory. The reference
    // / current position buffers are kept around for the loop and
    // rewritten in place each step.
    let inv_masses: Vec<f64> = masses.iter().map(|m| 1.0 / m).collect();
    let constraints: Vec<crate::shake::Constraint> = if opts.constrain_h_bonds {
        // The same per-atom type list `energy::forces_bonded` uses.
        let atom_types = energy::forces_bonded::build_atom_types(structure);
        crate::shake::build_h_bond_constraints(structure, graph, ff, &atom_types)
    } else {
        Vec::new()
    };
    let use_shake = !constraints.is_empty();
    // Each bond constraint removes one degree of freedom. The
    // Maxwell-Boltzmann variance the O step injects then gets partly
    // projected away by SHAKE; inflating σ² by the DOF ratio keeps the
    // post-projection KE matched to (3N/2) k_B T_target so the
    // thermostat actually reaches the requested temperature. Without
    // this scaling, a Trp-cage run at 310 K equilibrates to ~245 K
    // because the H-atom velocity components get clipped.
    let total_dof = (3 * n) as f64;
    let dof_correction = if use_shake {
        let removed = constraints.len() as f64;
        total_dof / (total_dof - removed).max(1.0)
    } else {
        1.0
    };
    let mut pos_buf: Vec<Vec3> = vec![Vec3::zeros(); n];
    let mut ref_buf: Vec<Vec3> = vec![Vec3::zeros(); n];
    let mut shake_failures = 0usize;
    const SHAKE_TOL_SQ: f64 = 1e-6;
    const SHAKE_MAX_ITERS: usize = 64;

    // Running stats (Welford).
    let mut sum_t = 0.0;
    let mut sum_t2 = 0.0;
    let mut samples = 0usize;
    let mut diverged = false;

    // Emit the initial frame (step 0).
    let ke0 = kinetic_energy_kj_mol(&velocities, &masses);
    let t0 = if dof > 0.0 {
        2.0 * ke0 / (dof * BOLTZMANN_KJ_PER_MOL_K)
    } else {
        0.0
    };
    if opts.save_every > 0 {
        callback(LangevinFrame {
            step: 0,
            time_fs: 0.0,
            instantaneous_temperature_k: t0,
            kinetic_energy_kj_mol: ke0,
            structure,
        });
    }

    for step in 1..=opts.steps {
        // B (first half): v += a · dt/2
        for i in 0..n {
            let inv_m_accel = ACCEL_FACTOR / masses[i];
            velocities[i] += forces[i] * (inv_m_accel * half_dt);
        }
        // A (first half): r += v · dt/2
        if use_shake {
            flatten_positions_into(structure, &mut ref_buf);
        }
        apply_velocity_step(structure, &velocities, half_dt);
        if use_shake {
            flatten_positions_into(structure, &mut pos_buf);
            if crate::shake::shake_iterate(
                &mut pos_buf,
                &ref_buf,
                &inv_masses,
                &constraints,
                SHAKE_TOL_SQ,
                SHAKE_MAX_ITERS,
            )
            .is_none()
            {
                shake_failures += 1;
            }
            // Project velocities onto the constrained displacement —
            // SHAKE moved atoms by (pos_buf − unconstrained_pos), and
            // without folding that back into v we silently inject
            // kinetic energy. The implied velocity that *actually*
            // advanced atoms from ref → pos_buf is
            //     v_consistent = (pos_buf − ref_buf) / half_dt.
            for i in 0..n {
                velocities[i] = (pos_buf[i] - ref_buf[i]) / half_dt;
            }
            unflatten_positions_from(structure, &pos_buf);
        }

        // O: v ← α v + σ ξ, σ² = (1 − α²) k_B T / m · ACCEL_FACTOR
        let one_minus_alpha2 = 1.0 - alpha * alpha;
        for i in 0..n {
            let sigma =
                (one_minus_alpha2 * kbt * dof_correction * ACCEL_FACTOR / masses[i]).sqrt();
            let xi_x = rng.gaussian();
            let xi_y = rng.gaussian();
            let xi_z = rng.gaussian();
            velocities[i].x = alpha * velocities[i].x + sigma * xi_x;
            velocities[i].y = alpha * velocities[i].y + sigma * xi_y;
            velocities[i].z = alpha * velocities[i].z + sigma * xi_z;
        }

        // A (second half).
        if use_shake {
            flatten_positions_into(structure, &mut ref_buf);
        }
        apply_velocity_step(structure, &velocities, half_dt);
        if use_shake {
            flatten_positions_into(structure, &mut pos_buf);
            if crate::shake::shake_iterate(
                &mut pos_buf,
                &ref_buf,
                &inv_masses,
                &constraints,
                SHAKE_TOL_SQ,
                SHAKE_MAX_ITERS,
            )
            .is_none()
            {
                shake_failures += 1;
            }
            for i in 0..n {
                velocities[i] = (pos_buf[i] - ref_buf[i]) / half_dt;
            }
            unflatten_positions_from(structure, &pos_buf);
        }

        // Recompute forces at the new positions.
        total_force_with_scratch(
            structure,
            graph,
            ff,
            DEFAULT_CUTOFF_A,
            opts.include_sasa,
            &mut scratch,
            &mut forces,
        );

        // B (second half): v += a · dt/2
        for i in 0..n {
            let inv_m_accel = ACCEL_FACTOR / masses[i];
            velocities[i] += forces[i] * (inv_m_accel * half_dt);
        }

        // Check for blow-up.
        if !finite_velocities(&velocities) {
            diverged = true;
            break;
        }

        // Accumulate temperature stats and emit frame if requested.
        let ke = kinetic_energy_kj_mol(&velocities, &masses);
        let t_inst = if dof > 0.0 {
            2.0 * ke / (dof * BOLTZMANN_KJ_PER_MOL_K)
        } else {
            0.0
        };
        sum_t += t_inst;
        sum_t2 += t_inst * t_inst;
        samples += 1;

        if opts.save_every > 0 && step % opts.save_every == 0 {
            let time_fs = step as f64 * opts.dt_fs;
            callback(LangevinFrame {
                step,
                time_fs,
                instantaneous_temperature_k: t_inst,
                kinetic_energy_kj_mol: ke,
                structure,
            });
        }
    }

    let mean = if samples > 0 { sum_t / samples as f64 } else { 0.0 };
    let var = if samples > 0 {
        (sum_t2 / samples as f64 - mean * mean).max(0.0)
    } else {
        0.0
    };
    let ke_final = kinetic_energy_kj_mol(&velocities, &masses);
    let equipartition_ratio = if dof > 0.0 && opts.temperature_k > 0.0 {
        ke_final / (0.5 * dof * BOLTZMANN_KJ_PER_MOL_K * opts.temperature_k)
    } else {
        0.0
    };

    LangevinSummary {
        steps_run: samples,
        temperature_mean_k: mean,
        temperature_stddev_k: var.sqrt(),
        equipartition_ratio,
        final_kinetic_energy_kj_mol: ke_final,
        atoms_count: n,
        diverged,
        shake_failures,
    }
}

fn flatten_positions_into(structure: &Structure, out: &mut Vec<Vec3>) {
    out.clear();
    for r in &structure.residues {
        for a in &r.atoms {
            out.push(a.position);
        }
    }
}

fn unflatten_positions_from(structure: &mut Structure, src: &[Vec3]) {
    let mut idx = 0usize;
    for r in &mut structure.residues {
        for a in &mut r.atoms {
            a.position = src[idx];
            idx += 1;
        }
    }
}

/// Compute the instantaneous temperature from the velocity buffer.
pub fn instant_temperature_k(velocities: &[Vec3], masses: &[f64]) -> f64 {
    let dof = (3 * velocities.len()) as f64;
    if dof == 0.0 {
        return 0.0;
    }
    let ke = kinetic_energy_kj_mol(velocities, masses);
    2.0 * ke / (dof * BOLTZMANN_KJ_PER_MOL_K)
}

// ---------- internals ----------
// Some of these are also used by the cotranslate driver in
// `cotranslate.rs`, which runs its own BAOAB loop so the velocity buffer
// and RNG persist across residue-emission slices.

/// Crate-public masses collector (Da, one per atom in chain order).
pub(crate) fn collect_masses_pub(structure: &Structure) -> Vec<f64> {
    collect_masses(structure)
}

/// Crate-public position update step.
pub(crate) fn apply_velocity_step_pub(structure: &mut Structure, velocities: &[Vec3], dt: f64) {
    apply_velocity_step(structure, velocities, dt);
}

/// Re-seed velocities for atoms newly appended to a structure since the
/// last call. Existing atoms keep their current velocity; new atoms get
/// Maxwell-Boltzmann samples at `temperature_k`. Used by the cotranslate
/// driver each time the ribosome emits a residue.
pub fn initialise_velocities_for_new_atoms(
    structure: &Structure,
    velocities: &mut Vec<Vec3>,
    temperature_k: f64,
    rng: &mut Xoshiro256pp,
) {
    let n = structure.atom_count();
    let old_n = velocities.len();
    if n <= old_n {
        return;
    }
    // Compute σ_v per new atom based on its mass.
    let kbt = BOLTZMANN_KJ_PER_MOL_K * temperature_k;
    let mut counted = 0usize;
    for residue in &structure.residues {
        for atom in &residue.atoms {
            if counted >= old_n {
                let m = atom.element.mass_da();
                let sigma = (kbt * ACCEL_FACTOR / m).sqrt();
                let v = Vec3::new(
                    sigma * rng.gaussian(),
                    sigma * rng.gaussian(),
                    sigma * rng.gaussian(),
                );
                velocities.push(v);
            }
            counted += 1;
        }
    }
    debug_assert_eq!(velocities.len(), n);
}

fn collect_masses(structure: &Structure) -> Vec<f64> {
    let mut out = Vec::with_capacity(structure.atom_count());
    for residue in &structure.residues {
        for atom in &residue.atoms {
            out.push(atom.element.mass_da());
        }
    }
    out
}

fn apply_velocity_step(structure: &mut Structure, velocities: &[Vec3], dt: f64) {
    let mut idx = 0usize;
    for residue in &mut structure.residues {
        for atom in &mut residue.atoms {
            atom.position += velocities[idx] * dt;
            idx += 1;
        }
    }
}

pub(crate) fn kinetic_energy_kj_mol(velocities: &[Vec3], masses: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (v, m) in velocities.iter().zip(masses.iter()) {
        sum += m * v.norm_squared();
    }
    0.5 * sum / ACCEL_FACTOR
}

fn finite_velocities(velocities: &[Vec3]) -> bool {
    velocities
        .iter()
        .all(|v| v.x.is_finite() && v.y.is_finite() && v.z.is_finite())
}

/// Draw initial velocities from the Maxwell-Boltzmann distribution at
/// `temperature_k`. Per-atom σ² = k_B T / m · ACCEL_FACTOR for each
/// component. The centre-of-mass velocity is then subtracted so the
/// system has zero net momentum.
fn initialise_maxwell_boltzmann(
    velocities: &mut [Vec3],
    masses: &[f64],
    temperature_k: f64,
    rng: &mut Xoshiro256pp,
) {
    let kbt = BOLTZMANN_KJ_PER_MOL_K * temperature_k;
    for (v, m) in velocities.iter_mut().zip(masses.iter()) {
        let sigma = (kbt * ACCEL_FACTOR / m).sqrt();
        v.x = sigma * rng.gaussian();
        v.y = sigma * rng.gaussian();
        v.z = sigma * rng.gaussian();
    }
    // Subtract COM velocity.
    let mut total_mass = 0.0;
    let mut p = Vec3::zeros();
    for (v, m) in velocities.iter().zip(masses.iter()) {
        p += *v * *m;
        total_mass += m;
    }
    if total_mass > 0.0 {
        let v_com = p / total_mass;
        for v in velocities.iter_mut() {
            *v -= v_com;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chem::{standard_ff, AminoAcid};
    use geom::{build_extended_chain, build_topology_graph};

    #[test]
    fn rest_velocities_give_zero_temperature() {
        let v = vec![Vec3::zeros(); 5];
        let m = vec![12.0; 5];
        assert_eq!(instant_temperature_k(&v, &m), 0.0);
    }

    #[test]
    fn maxwell_boltzmann_recovers_target_temperature() {
        let n = 200usize;
        let masses: Vec<f64> = vec![12.011; n];
        let mut v = vec![Vec3::zeros(); n];
        let mut rng = Xoshiro256pp::from_seed(11);
        initialise_maxwell_boltzmann(&mut v, &masses, 310.0, &mut rng);
        let t = instant_temperature_k(&v, &masses);
        // 3N − 3 DoF after COM removal. Relative std-dev of the
        // instantaneous temperature ~ sqrt(2/DoF) ≈ 5.8% for N=200, so
        // a single sample can wander ±50 K from the target.
        assert!(
            (t - 310.0).abs() < 60.0,
            "post-init T = {t}, expected near 310"
        );
        // COM velocity should be ~zero.
        let mut total_mass = 0.0;
        let mut p = Vec3::zeros();
        for (vv, m) in v.iter().zip(masses.iter()) {
            p += *vv * *m;
            total_mass += m;
        }
        let v_com = p / total_mass;
        assert!(v_com.norm() < 1e-10, "COM not zeroed: {v_com:?}");
    }

    #[test]
    fn baoab_runs_short_trajectory_without_blowup() {
        // Two-residue chain; just check the integrator doesn't NaN.
        let mut s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala]).unwrap();
        let g = build_topology_graph(&s);
        let ff = standard_ff();
        let opts = LangevinOptions {
            steps: 50,
            save_every: 10,
            ..Default::default()
        };
        let summary = run_langevin(&mut s, &g, ff, opts, |_frame| {});
        assert!(!summary.diverged, "integrator diverged");
        assert!(summary.temperature_mean_k > 0.0);
        for r in &s.residues {
            for a in &r.atoms {
                assert!(a.position.x.is_finite());
                assert!(a.position.y.is_finite());
                assert!(a.position.z.is_finite());
            }
        }
    }
}
