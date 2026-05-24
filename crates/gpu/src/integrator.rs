//! Full integrator-on-GPU: BAOAB + bonded + nonbonded + GB, all
//! running on-device with the integrator state (positions, velocities,
//! forces, RNG) persistent between steps.
//!
//! Replaces the per-step CPU↔GPU sync that dominates the per-step
//! cost at moderate N today.  The host calls [`IntegratorPipeline::step_n`]
//! with a step count; the GPU executes the full BAOAB sequence
//! (zero-forces → bonded → pair → BAOAB) that many times without
//! ever returning to the CPU.  Positions/velocities are read back
//! only when the caller asks (e.g. on save-frame boundaries).
//!
//! Composes [`BondedPipeline`], [`VerletNonbondedPipeline`],
//! [`GbPipeline`], and [`BaoabPipeline`] sharing one positions buffer
//! and one forces buffer.  Neighbour lists for the LJ + Coulomb (10 Å)
//! and the GB (20 Å) cutoffs are managed externally — the host
//! refreshes them whenever the Verlet drift check on the CPU side
//! says it's time, then re-uploads via the existing
//! `update_neighbours` methods.

use crate::baoab::{make_rng_state, BaoabPipeline};
use crate::bonded::{BondedPipeline, BondedSetup};
use crate::context::GpuContext;
use crate::gb::{GbPipeline, GbSetup};
use crate::nonbonded_verlet::{VerletNonbondedPipeline, VerletNonbondedSetup};

pub struct IntegratorPipeline {
    n_atoms: usize,
    bonded: BondedPipeline,
    nonbonded: VerletNonbondedPipeline,
    gb: GbPipeline,
    baoab: BaoabPipeline,
    ctx: &'static GpuContext,
}

impl IntegratorPipeline {
    /// Build the full integrator pipeline.  All four sub-pipelines
    /// share one positions buffer (owned by BAOAB, the integrator
    /// state) and one forces buffer (owned by BAOAB, the
    /// accumulation target).  After construction the host should:
    ///
    ///   1. `upload_initial_state(positions, velocities)` to seed the
    ///      integrator.
    ///   2. `update_nb_neighbours` and `update_gb_neighbours` to seed
    ///      the neighbour lists.
    ///   3. Set Langevin parameters via `set_step_params(dt, gamma, kbt)`.
    ///   4. Call `step_n(n)` to run.
    ///   5. `download_positions()` / `download_velocities()` to inspect.
    ///
    /// Total constructor work is dominated by shader compilation
    /// (~50 ms on Apple Silicon) — pay it once per simulation.
    pub fn new(
        ctx: &'static GpuContext,
        n_atoms: usize,
        masses_da: &[f32],
        rng_seed: u64,
        bonded_setup: BondedSetup,
        nb_setup: VerletNonbondedSetup,
        gb_setup: GbSetup,
    ) -> Self {
        let device = &ctx.device;
        let positions_size = (n_atoms * std::mem::size_of::<[f32; 4]>()) as u64;

        // The shared buffers must support the union of every usage
        // any binding requires.  Positions: read in every kernel,
        // write in BAOAB.  Forces: written in every force kernel,
        // read in BAOAB; readable from CPU for diagnostics.
        let positions_buf = std::sync::Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("integ_positions"),
            size: positions_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let velocities_buf = std::sync::Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("integ_velocities"),
            size: positions_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let forces_buf = std::sync::Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("integ_forces"),
            size: positions_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        // Pass Arc clones — each sub-pipeline keeps a live reference.
        // BondedPipeline takes &Buffer; deref the Arc to get one.
        let bonded = BondedPipeline::new(
            ctx,
            n_atoms,
            &positions_buf,
            &forces_buf,
            bonded_setup,
        );
        // (BondedPipeline retains the &Buffer internally via its bind
        // group; the Arc here keeps the underlying buffer alive even
        // after this function returns because BAOAB / nb / gb also hold
        // Arc clones.)
        let nonbonded = VerletNonbondedPipeline::new_with_external_buffers(
            ctx, n_atoms, nb_setup, positions_buf.clone(), Some(forces_buf.clone()),
        );
        let gb = GbPipeline::new_with_external_buffers(
            ctx, n_atoms, gb_setup, positions_buf.clone(), Some(forces_buf.clone()),
        );

        let rng_state = make_rng_state(rng_seed, n_atoms);
        let baoab = BaoabPipeline::new_with_external_buffers(
            ctx, n_atoms, masses_da, &rng_state,
            Some(positions_buf), Some(velocities_buf), Some(forces_buf),
        );

        Self { n_atoms, bonded, nonbonded, gb, baoab, ctx }
    }

    pub fn upload_positions(&mut self, positions: &[[f32; 3]]) {
        self.baoab.upload_positions(positions);
    }

    pub fn upload_velocities(&mut self, velocities: &[[f32; 3]]) {
        self.baoab.upload_velocities(velocities);
    }

    pub fn update_nb_neighbours(&mut self, counts: &[u32], starts: &[u32], indices: &[u32]) {
        self.nonbonded.update_neighbours(counts, starts, indices);
    }

    pub fn update_gb_neighbours(&mut self, counts: &[u32], starts: &[u32], indices: &[u32]) {
        self.gb.update_neighbours(counts, starts, indices);
    }

    pub fn set_step_params(&self, dt_fs: f32, gamma_ps_inv: f32, kbt_kj_mol: f32) {
        self.baoab.set_step_params(dt_fs, gamma_ps_inv, kbt_kj_mol);
    }

    /// Run `n_steps` BAOAB iterations entirely on the GPU.  No CPU↔GPU
    /// sync per step.
    ///
    /// **Folded BAOAB**.  A naive per-step recipe would be
    ///
    ///   1. force eval at r_n   ← used by leading B
    ///   2. B → A → O → A   (positions now at r_{n+1})
    ///   3. force eval at r_{n+1}   ← used by trailing B
    ///   4. B
    ///
    /// — i.e. two force evals per step.  But the trailing B at step n
    /// uses the same F(r_{n+1}) as the leading B at step n+1, so the
    /// two force evals can be folded into one:
    ///
    ///   [initial: force eval at r_0, once]
    ///   for step in 1..=n:
    ///     B → A → O → A   (positions move to r_{n+1})
    ///     force eval at r_{n+1}   ← serves trailing B *and* next step's leading B
    ///     B
    ///
    /// One force eval per step.  Halves the per-step kernel work — the
    /// 4 bonded passes + nonbonded + GB dominate per-step cost at
    /// every scale where the GPU is competitive in the first place.
    ///
    /// The "initial force eval" is done unconditionally at the start
    /// of every `step_n` call; it's one extra eval per batch (~4 % of
    /// a 25-step batch's work) and removes the need to track whether
    /// the persistent forces buffer is still in sync with the current
    /// positions across multiple step_n invocations.
    pub fn step_n(&self, n_steps: usize) {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        // Initial force evaluation — populates the persistent forces
        // buffer with F(r_0).  The first iteration's leading B will
        // read from it (folded into the trailing B of the previous
        // iteration … or, on this first iteration, just this initial
        // eval).
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("integ_initial_force_eval_encoder"),
            });
            self.record_force_eval(&mut encoder);
            queue.submit(Some(encoder.finish()));
        }

        // Chunk the loop: each integrator step is now 9 dispatches
        // (B-A-O-A + 7-pass force eval + B = 1 + 7 + 1).  On Apple
        // Silicon Metal, command buffers with hundreds of dispatches
        // stall, so we cap at CHUNK steps per encoder.  CHUNK = 16
        // keeps each submit at ~144 dispatches — well within the
        // working range and bigger than the pre-fold limit of 8.
        const CHUNK: usize = 16;
        let mut remaining = n_steps;
        while remaining > 0 {
            let this_chunk = remaining.min(CHUNK);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("integ_step_chunk_encoder"),
            });
            for _ in 0..this_chunk {
                // BAOAB first half: B-A-O-A using forces at r_n.
                self.baoab.record_first_half(&mut encoder);
                // Force eval at the new positions r_{n+1}.  Serves
                // both this step's trailing B and the next step's
                // leading B.
                self.record_force_eval(&mut encoder);
                // BAOAB second half: B using forces at r_{n+1}.
                self.baoab.record_second_half(&mut encoder);
            }
            queue.submit(Some(encoder.finish()));
            remaining -= this_chunk;
        }
        let _ = device.poll(wgpu::Maintain::Wait);
    }

    /// One full force evaluation: zero the buffer, then accumulate
    /// all-bonded (fused bond+angle+dihedral+improper) + nonbonded
    /// + GB.  Records 4 compute passes (down from 7 pre-fusion —
    /// see `PERF.gpu.14`).
    fn record_force_eval(&self, encoder: &mut wgpu::CommandEncoder) {
        self.bonded.record_zero(encoder);
        self.bonded.record_all_bonded(encoder);
        self.nonbonded.record_compute(encoder);
        self.gb.record_compute(encoder);
    }

    pub fn download_positions(&self) -> Vec<[f32; 3]> {
        self.baoab.download_positions()
    }

    pub fn download_velocities(&self) -> Vec<[f32; 3]> {
        self.baoab.download_velocities()
    }

    pub fn n_atoms(&self) -> usize { self.n_atoms }
}
