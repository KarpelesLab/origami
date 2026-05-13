//! Replica Exchange Molecular Dynamics (parallel tempering).
//!
//! Run N copies of the same simulation at different temperatures
//! T₁ < T₂ < … < T_N. Each replica runs a BAOAB Langevin trajectory
//! at its own T; periodically the algorithm attempts to swap the
//! configurations of adjacent (in temperature) replicas, accepting
//! each swap via the Metropolis criterion
//!
//!   P_accept = min(1, exp[(1/kT_a − 1/kT_b) · (E_a − E_b)])
//!
//! where E_a, E_b are the current potential energies of replicas a, b.
//! Swaps connect the low-T trajectory (the production replica, where
//! we save frames) to higher-T conformational excursions that cross
//! free-energy barriers, then bring back the equilibrated low-T
//! state. Net effect: dramatically better conformational sampling
//! per wall-second than a single low-T trajectory.
//!
//! ## What we do here (M9 / first cut)
//!
//! - Replicas run sequentially in a single thread. The integrator
//!   inside each replica is fully accelerated (SoA + rayon-parallel
//!   nonbonded), so per-replica throughput already uses the cores.
//!   Running replicas in parallel via rayon::scope is a follow-up.
//!
//! - Swap attempts alternate odd / even adjacent pairs each round
//!   so a single configuration can traverse the full temperature
//!   ladder in roughly N/2 round-trips.
//!
//! - The callback receives every saved frame from every replica
//!   tagged with the replica index. Production analyses typically
//!   filter to replica 0 (the lowest T).

use std::ops::Range;

use chem::ForceField;
use energy::{bonded::bonded_energy, gb_energy, nonbonded_energy, DEFAULT_CUTOFF_A};
use energy::{total_force_with_scratch, ForceScratch};
use geom::{Structure, TopologyGraph, Vec3};
use rayon::prelude::*;

use crate::langevin::{
    initialise_velocities_for_new_atoms, kinetic_energy_kj_mol, ACCEL_FACTOR,
    BOLTZMANN_KJ_PER_MOL_K,
};
use crate::rng::Xoshiro256pp;
use crate::shake;

/// REMD knobs.
#[derive(Debug, Clone)]
pub struct RemdOptions {
    /// Temperatures (K) for each replica, in any order. The driver
    /// sorts them ascending and labels replica 0 as the production
    /// (lowest-T) trajectory.
    pub temperatures_k: Vec<f64>,
    /// Integration timestep (fs).
    pub dt_fs: f64,
    /// Friction γ (ps⁻¹) — applied identically to every replica.
    pub friction_ps_inv: f64,
    /// Total simulated time per replica (fs).
    pub total_time_fs: f64,
    /// How often to attempt a swap (fs).
    pub swap_interval_fs: f64,
    /// Save every N integrator steps. Applied to every replica; the
    /// callback receives the replica index so the caller can filter.
    pub save_every: usize,
    /// Base PRNG seed. Each replica gets `seed ^ (replica_idx * 0xC0FFEE)`
    /// so they sample independent thermal noise.
    pub seed: u64,
    /// Pass through SASA forces (hydrophobic collapse driver).
    pub include_sasa: bool,
    /// SHAKE the X-H bonds (enables dt = 2 fs).
    pub constrain_h_bonds: bool,
}

impl Default for RemdOptions {
    fn default() -> Self {
        Self {
            temperatures_k: vec![300.0, 350.0, 400.0, 450.0],
            dt_fs: 1.0,
            friction_ps_inv: 2.0,
            total_time_fs: 2000.0,
            swap_interval_fs: 200.0,
            save_every: 100,
            seed: 0,
            include_sasa: false,
            constrain_h_bonds: false,
        }
    }
}

/// Per-replica result summary.
#[derive(Debug, Clone)]
pub struct RemdReplicaSummary {
    pub temperature_k: f64,
    pub final_kinetic_energy_kj_mol: f64,
    pub final_potential_energy_kj_mol: f64,
    pub diverged: bool,
    pub shake_failures: usize,
}

/// REMD-level summary.
#[derive(Debug, Clone)]
pub struct RemdSummary {
    pub n_replicas: usize,
    pub atoms_count: usize,
    /// Number of swap attempts between each adjacent pair (i, i+1).
    pub swap_attempts: Vec<usize>,
    /// Number of accepted swaps between each adjacent pair.
    pub swap_accepts: Vec<usize>,
    pub per_replica: Vec<RemdReplicaSummary>,
}

impl RemdSummary {
    pub fn acceptance_ratios(&self) -> Vec<f64> {
        self.swap_attempts
            .iter()
            .zip(&self.swap_accepts)
            .map(|(&att, &acc)| if att > 0 { acc as f64 / att as f64 } else { 0.0 })
            .collect()
    }
}

/// Frame snapshot passed to the REMD callback.
pub struct RemdFrame<'a> {
    pub replica_idx: usize,
    pub temperature_k: f64,
    pub step: usize,
    pub time_fs: f64,
    pub instantaneous_temperature_k: f64,
    pub structure: &'a Structure,
}

/// Run N-replica parallel-tempering REMD. `structure` is the shared
/// initial configuration; every replica starts from a clone of it
/// with independently-sampled Maxwell-Boltzmann velocities.
pub fn run_remd<F>(
    structure: &Structure,
    graph: &TopologyGraph,
    ff: &ForceField,
    opts: RemdOptions,
    mut callback: F,
) -> RemdSummary
where
    F: FnMut(RemdFrame<'_>),
{
    // Sort temperatures so replica 0 is the lowest-T production
    // trajectory and adjacent pairs are well-defined.
    let mut temps = opts.temperatures_k.clone();
    temps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n_rep = temps.len().max(1);
    let n = structure.atom_count();
    let half_dt = 0.5 * opts.dt_fs;
    let steps_per_swap = (opts.swap_interval_fs / opts.dt_fs).round() as usize;
    let total_steps = (opts.total_time_fs / opts.dt_fs).round() as usize;
    let n_swap_rounds = total_steps.div_ceil(steps_per_swap.max(1));

    // Per-replica integrator state. Each replica owns everything it
    // needs to run a BAOAB step independently, so `replicas
    // .par_iter_mut()` is safe — no shared mutable state across
    // replicas during the parallel-advance phase.
    struct Replica {
        structure: Structure,
        velocities: Vec<Vec3>,
        forces: Vec<Vec3>,
        scratch: ForceScratch,
        masses: Vec<f64>,
        inv_masses: Vec<f64>,
        rng: Xoshiro256pp,
        temperature_k: f64,
        diverged: bool,
        shake_failures: usize,
        time_fs: f64,
        step: usize,
        // Per-replica position scratch (was shared/outer in the
        // sequential version; needs to be per-replica for rayon
        // to give each thread its own copy).
        pos_buf: Vec<Vec3>,
        ref_buf: Vec<Vec3>,
        // Frames produced during a parallel advance, drained
        // serially into the callback after the rayon scope closes.
        pending_frames: Vec<(usize /* step */, f64 /* time_fs */, f64 /* T_inst */, Structure)>,
    }

    let constraints = if opts.constrain_h_bonds {
        let atom_types = energy::forces_bonded::build_atom_types(structure);
        shake::build_h_bond_constraints(structure, graph, ff, &atom_types)
    } else {
        Vec::new()
    };
    let use_shake = !constraints.is_empty();
    let total_dof = (3 * n) as f64;
    let dof_correction = if use_shake {
        total_dof / (total_dof - constraints.len() as f64).max(1.0)
    } else {
        1.0
    };

    // Build all replicas.
    let mut replicas: Vec<Replica> = Vec::with_capacity(n_rep);
    for (i, &t) in temps.iter().enumerate() {
        let mut rep_structure = structure.clone();
        let masses = crate::langevin::collect_masses_pub(&rep_structure);
        let inv_masses: Vec<f64> = masses.iter().map(|m| 1.0 / m).collect();
        let mut rng = Xoshiro256pp::from_seed(opts.seed.wrapping_add((i as u64).wrapping_mul(0xC0FFEE)));
        let mut velocities = vec![Vec3::zeros(); n];
        initialise_velocities_for_new_atoms(&rep_structure, &mut velocities, t, &mut rng);
        let mut scratch = ForceScratch::new(&rep_structure, graph, ff);
        let mut forces: Vec<Vec3> = Vec::with_capacity(n);
        total_force_with_scratch(
            &mut rep_structure,
            graph,
            ff,
            DEFAULT_CUTOFF_A,
            opts.include_sasa,
            &mut scratch,
            &mut forces,
        );
        replicas.push(Replica {
            structure: rep_structure,
            velocities,
            forces,
            scratch,
            masses,
            inv_masses,
            rng,
            temperature_k: t,
            diverged: false,
            shake_failures: 0,
            time_fs: 0.0,
            step: 0,
            pos_buf: vec![Vec3::zeros(); n],
            ref_buf: vec![Vec3::zeros(); n],
            pending_frames: Vec::new(),
        });
    }
    const SHAKE_TOL_SQ: f64 = 1e-6;
    const SHAKE_MAX_ITERS: usize = 64;

    // Swap statistics: indexed by adjacent-pair (i, i+1).
    let n_pairs = n_rep.saturating_sub(1);
    let mut swap_attempts = vec![0usize; n_pairs];
    let mut swap_accepts = vec![0usize; n_pairs];

    // A shared swap-RNG so reproducibility doesn't depend on which
    // replica draws first.
    let mut swap_rng = Xoshiro256pp::from_seed(opts.seed.wrapping_add(0xDEADBEEF));

    // Emit the initial state of every replica at step 0 if save_every > 0.
    if opts.save_every > 0 {
        for (i, rep) in replicas.iter().enumerate() {
            let ke = kinetic_energy_kj_mol(&rep.velocities, &rep.masses);
            let dof = (3 * n) as f64;
            let t_inst = if dof > 0.0 {
                2.0 * ke / (dof * BOLTZMANN_KJ_PER_MOL_K)
            } else {
                0.0
            };
            callback(RemdFrame {
                replica_idx: i,
                temperature_k: rep.temperature_k,
                step: 0,
                time_fs: 0.0,
                instantaneous_temperature_k: t_inst,
                structure: &rep.structure,
            });
        }
    }

    // Round = one slab of steps + one swap attempt.
    for round in 0..n_swap_rounds {
        let step_range = Range {
            start: round * steps_per_swap,
            end: (round * steps_per_swap + steps_per_swap).min(total_steps),
        };
        if step_range.is_empty() {
            break;
        }

        // Parallel advance — every replica runs `step_range.len()`
        // integrator steps with no cross-replica state shared, so
        // rayon can hand each one to a different worker. The inner
        // SoA nonbonded kernel also uses rayon for its own pair
        // loop; under nested parallelism rayon's work-stealing
        // scheduler is well-behaved (the inner par_iter just queues
        // more work for whichever worker is free).
        replicas.par_iter_mut().for_each(|rep| {
            if rep.diverged {
                return;
            }
            rep.pending_frames.clear();
            let alpha = (-opts.friction_ps_inv * opts.dt_fs * 1.0e-3).exp();
            let kbt = BOLTZMANN_KJ_PER_MOL_K * rep.temperature_k;
            for _local_step in step_range.clone() {
                rep.step += 1;
                let abs_step = rep.step;

                // B (first half)
                for i in 0..n {
                    let inv_m_accel = ACCEL_FACTOR / rep.masses[i];
                    rep.velocities[i] += rep.forces[i] * (inv_m_accel * half_dt);
                }
                // A (first half)
                if use_shake {
                    flatten_positions(&rep.structure, &mut rep.ref_buf);
                }
                apply_velocity_step(&mut rep.structure, &rep.velocities, half_dt);
                if use_shake {
                    flatten_positions(&rep.structure, &mut rep.pos_buf);
                    if shake::shake_iterate(
                        &mut rep.pos_buf,
                        &rep.ref_buf,
                        &rep.inv_masses,
                        &constraints,
                        SHAKE_TOL_SQ,
                        SHAKE_MAX_ITERS,
                    )
                    .is_none()
                    {
                        rep.shake_failures += 1;
                    }
                    for i in 0..n {
                        rep.velocities[i] = (rep.pos_buf[i] - rep.ref_buf[i]) / half_dt;
                    }
                    unflatten_positions(&mut rep.structure, &rep.pos_buf);
                }
                // O
                let one_minus_alpha2 = 1.0 - alpha * alpha;
                for i in 0..n {
                    let sigma = (one_minus_alpha2
                        * kbt
                        * dof_correction
                        * ACCEL_FACTOR
                        / rep.masses[i])
                        .sqrt();
                    rep.velocities[i].x =
                        alpha * rep.velocities[i].x + sigma * rep.rng.gaussian();
                    rep.velocities[i].y =
                        alpha * rep.velocities[i].y + sigma * rep.rng.gaussian();
                    rep.velocities[i].z =
                        alpha * rep.velocities[i].z + sigma * rep.rng.gaussian();
                }
                // A (second half)
                if use_shake {
                    flatten_positions(&rep.structure, &mut rep.ref_buf);
                }
                apply_velocity_step(&mut rep.structure, &rep.velocities, half_dt);
                if use_shake {
                    flatten_positions(&rep.structure, &mut rep.pos_buf);
                    if shake::shake_iterate(
                        &mut rep.pos_buf,
                        &rep.ref_buf,
                        &rep.inv_masses,
                        &constraints,
                        SHAKE_TOL_SQ,
                        SHAKE_MAX_ITERS,
                    )
                    .is_none()
                    {
                        rep.shake_failures += 1;
                    }
                    for i in 0..n {
                        rep.velocities[i] = (rep.pos_buf[i] - rep.ref_buf[i]) / half_dt;
                    }
                    unflatten_positions(&mut rep.structure, &rep.pos_buf);
                }
                // Recompute force at new positions.
                total_force_with_scratch(
                    &mut rep.structure,
                    graph,
                    ff,
                    DEFAULT_CUTOFF_A,
                    opts.include_sasa,
                    &mut rep.scratch,
                    &mut rep.forces,
                );
                // B (second half)
                for i in 0..n {
                    let inv_m_accel = ACCEL_FACTOR / rep.masses[i];
                    rep.velocities[i] += rep.forces[i] * (inv_m_accel * half_dt);
                }
                // Divergence check.
                if !rep
                    .velocities
                    .iter()
                    .all(|v| v.x.is_finite() && v.y.is_finite() && v.z.is_finite())
                {
                    rep.diverged = true;
                    break;
                }
                rep.time_fs += opts.dt_fs;

                // Frame emit — buffered, dispatched serially below.
                if opts.save_every > 0 && abs_step % opts.save_every == 0 {
                    let ke = kinetic_energy_kj_mol(&rep.velocities, &rep.masses);
                    let dof = (3 * n) as f64;
                    let t_inst = if dof > 0.0 {
                        2.0 * ke / (dof * BOLTZMANN_KJ_PER_MOL_K)
                    } else {
                        0.0
                    };
                    rep.pending_frames
                        .push((abs_step, rep.time_fs, t_inst, rep.structure.clone()));
                }
            }
        });

        // Drain buffered frames into the caller's callback in replica
        // order — the callback is `FnMut` so we can't call it from
        // inside the parallel section.
        for (rep_idx, rep) in replicas.iter_mut().enumerate() {
            let frames = std::mem::take(&mut rep.pending_frames);
            for (step, time_fs, t_inst, structure) in frames {
                callback(RemdFrame {
                    replica_idx: rep_idx,
                    temperature_k: rep.temperature_k,
                    step,
                    time_fs,
                    instantaneous_temperature_k: t_inst,
                    structure: &structure,
                });
            }
        }

        // ---- Swap attempt ----
        // Alternate odd/even pairs each round to give every replica
        // a chance to move through the ladder.
        let parity = round % 2;
        for pair_idx in (parity..n_pairs).step_by(2) {
            let (a, b) = (pair_idx, pair_idx + 1);
            if replicas[a].diverged || replicas[b].diverged {
                continue;
            }
            swap_attempts[pair_idx] += 1;
            let e_a = potential_energy(&replicas[a].structure, graph, ff, opts.include_sasa);
            let e_b = potential_energy(&replicas[b].structure, graph, ff, opts.include_sasa);
            let beta_a = 1.0 / (BOLTZMANN_KJ_PER_MOL_K * replicas[a].temperature_k);
            let beta_b = 1.0 / (BOLTZMANN_KJ_PER_MOL_K * replicas[b].temperature_k);
            let delta = (beta_a - beta_b) * (e_a - e_b);
            let p_accept = if delta >= 0.0 { 1.0 } else { delta.exp() };
            let u = swap_rng.next_f64();
            if u < p_accept {
                swap_accepts[pair_idx] += 1;
                // Swap configurations (positions + force buffer +
                // scratch positions). Velocities also swap, then are
                // rescaled to the destination temperature so the
                // kinetic energy distribution stays consistent.
                let scale_a = (replicas[a].temperature_k / replicas[b].temperature_k).sqrt();
                let scale_b = (replicas[b].temperature_k / replicas[a].temperature_k).sqrt();
                // We need to swap structure, velocities, forces, scratch
                // between replicas a and b. The borrow checker prefers
                // `split_at_mut` here.
                let (left, right) = replicas.split_at_mut(b);
                let ra = &mut left[a];
                let rb = &mut right[0];
                std::mem::swap(&mut ra.structure, &mut rb.structure);
                std::mem::swap(&mut ra.velocities, &mut rb.velocities);
                std::mem::swap(&mut ra.forces, &mut rb.forces);
                std::mem::swap(&mut ra.scratch, &mut rb.scratch);
                // After swap: ra has what was b's config (at T_b);
                // rescale to ra's T_a.  rb conversely.
                for v in ra.velocities.iter_mut() {
                    *v *= scale_a;
                }
                for v in rb.velocities.iter_mut() {
                    *v *= scale_b;
                }
                // Force buffer is stale after position swap. The next
                // B step's velocity update will use the swapped
                // forces (which correspond to the new positions — they
                // were last evaluated at those coordinates), so we're
                // consistent. No re-evaluation needed because the
                // forces array travelled with the structure.
            }
        }
    }

    // Final summaries.
    let mut per_replica = Vec::with_capacity(n_rep);
    for rep in &replicas {
        let ke = kinetic_energy_kj_mol(&rep.velocities, &rep.masses);
        let pe = potential_energy(&rep.structure, graph, ff, opts.include_sasa);
        per_replica.push(RemdReplicaSummary {
            temperature_k: rep.temperature_k,
            final_kinetic_energy_kj_mol: ke,
            final_potential_energy_kj_mol: pe,
            diverged: rep.diverged,
            shake_failures: rep.shake_failures,
        });
    }

    RemdSummary {
        n_replicas: n_rep,
        atoms_count: n,
        swap_attempts,
        swap_accepts,
        per_replica,
    }
}

fn flatten_positions(structure: &Structure, out: &mut Vec<Vec3>) {
    out.clear();
    for r in &structure.residues {
        for a in &r.atoms {
            out.push(a.position);
        }
    }
}

fn unflatten_positions(structure: &mut Structure, src: &[Vec3]) {
    let mut idx = 0usize;
    for r in &mut structure.residues {
        for a in &mut r.atoms {
            a.position = src[idx];
            idx += 1;
        }
    }
}

fn apply_velocity_step(structure: &mut Structure, velocities: &[Vec3], dt: f64) {
    let mut idx = 0usize;
    for r in &mut structure.residues {
        for a in &mut r.atoms {
            a.position += velocities[idx] * dt;
            idx += 1;
        }
    }
}

fn potential_energy(structure: &Structure, graph: &TopologyGraph, ff: &ForceField, include_sasa: bool) -> f64 {
    let bonded = bonded_energy(structure, graph, ff);
    let nb = nonbonded_energy(structure, graph, ff, DEFAULT_CUTOFF_A);
    let gb = gb_energy(structure, ff);
    let mut total = bonded.total_kj_mol() + nb.lj_kj_mol + nb.coulomb_kj_mol + gb.gb_kj_mol;
    if include_sasa {
        let sasa = energy::sasa_energy(structure, ff);
        total += sasa.sasa_kj_mol;
    }
    total
}
