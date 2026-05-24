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

    /// Run `n_steps` integrator iterations entirely on the GPU.
    /// No CPU↔GPU sync per step.  The complete per-step recipe:
    ///
    ///   1. zero forces buffer
    ///   2. bonded kernels (bond → angle → dihedral → improper) — accumulate
    ///   3. pair kernels (nonbonded + GB) — accumulate
    ///   4. BAOAB first half (uses forces at r_n, advances to r_{n+1})
    ///   5. zero forces buffer
    ///   6. bonded kernels again at r_{n+1}
    ///   7. pair kernels again at r_{n+1}
    ///   8. BAOAB second half (B step using forces at r_{n+1})
    ///
    /// Steps 5-8 produce the F(r_{n+1}) that the next iteration
    /// reuses as its leading-B force.  The first iteration computes
    /// F(r_0) at step 1 too.
    ///
    /// All N steps are recorded into one command encoder and submitted
    /// once.  This is the kernel-fusion win the user predicted: no
    /// per-step sync, just GPU pipeline throughput.
    pub fn step_n(&self, n_steps: usize) {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        // Each integrator step records 17 compute dispatches (zero
        // + 4 bonded + nb + gb, then first-half BAOAB, then the same
        // pre-BAOAB force chain, then second-half BAOAB).  On Apple
        // Silicon Metal, command buffers with hundreds of dispatches
        // cause stalls (likely an internal GPU command-buffer size or
        // pipeline-state limit — empirically OK up to ~200 dispatches
        // per submit).  Chunk the loop and submit every CHUNK steps;
        // we don't `poll(Wait)` between submits so the GPU keeps
        // executing in parallel with the CPU recording the next
        // chunk.
        const CHUNK: usize = 8;
        let mut remaining = n_steps;
        while remaining > 0 {
            let this_chunk = remaining.min(CHUNK);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("integ_step_chunk_encoder"),
            });
            for _ in 0..this_chunk {
                self.bonded.record_zero(&mut encoder);
                self.bonded.record_bond(&mut encoder);
                self.bonded.record_angle(&mut encoder);
                self.bonded.record_dihedral(&mut encoder);
                self.bonded.record_improper(&mut encoder);
                self.nonbonded.record_compute(&mut encoder);
                self.gb.record_compute(&mut encoder);
                self.baoab.record_first_half(&mut encoder);
                self.bonded.record_zero(&mut encoder);
                self.bonded.record_bond(&mut encoder);
                self.bonded.record_angle(&mut encoder);
                self.bonded.record_dihedral(&mut encoder);
                self.bonded.record_improper(&mut encoder);
                self.nonbonded.record_compute(&mut encoder);
                self.gb.record_compute(&mut encoder);
                self.baoab.record_second_half(&mut encoder);
            }
            queue.submit(Some(encoder.finish()));
            remaining -= this_chunk;
        }
        let _ = device.poll(wgpu::Maintain::Wait);
    }

    pub fn download_positions(&self) -> Vec<[f32; 3]> {
        self.baoab.download_positions()
    }

    pub fn download_velocities(&self) -> Vec<[f32; 3]> {
        self.baoab.download_velocities()
    }

    pub fn n_atoms(&self) -> usize { self.n_atoms }
}
