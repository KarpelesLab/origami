//! GPU BAOAB Langevin step pipeline.
//!
//! [`BaoabPipeline`] holds the wgpu objects for one BAOAB step on the
//! GPU.  Two compute passes:
//!
//!   `first_half` — B (using forces_total) → A → O → A → writes
//!                  updated positions + velocities + RNG state.
//!   `second_half` — B (using forces_total) → writes final velocities.
//!
//! Per integrator step the caller is expected to:
//!   1. Dispatch `first_half` with forces evaluated at `r_n`.
//!   2. Recompute forces at the new `r_{n+1}` (via existing
//!      `VerletNonbondedPipeline` / `GbPipeline` / bonded kernels).
//!   3. Dispatch `second_half` with the new forces.
//!
//! That matches the CPU `dynamics::langevin::run_langevin` loop
//! one-for-one — folded form where the trailing B at step n is
//! combined with the leading B at step n+1.
//!
//! This module ships only the integrator kernels.  To actually replace
//! the CPU integrator end-to-end, the caller also needs:
//!   - bonded forces summed into `forces_total` on the GPU
//!   - per-step neighbour-list maintenance staying on the GPU
//! Neither is part of this commit — they're the next chunks of the
//! integrator-on-GPU arc.

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::context::GpuContext;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n_atoms: u32,
    half_dt: f32,
    alpha: f32,
    o_sigma_sq_base: f32,
    accel_factor: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Per-atom RNG seed expansion (SplitMix64-style on u32).  Mirrors the
/// CPU `Xoshiro256pp::from_seed`'s SplitMix initialisation, just
/// truncated to u32.  Identical bit-for-bit init isn't required since
/// the GPU uses xoshiro128++ (different generator).
pub fn make_rng_state(seed: u64, n_atoms: usize) -> Vec<[u32; 4]> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n_atoms);
    for _ in 0..n_atoms {
        // Per-atom: derive four u32 words via SplitMix64.
        let mut words = [0u32; 4];
        for w in &mut words {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            *w = (z & 0xFFFF_FFFF) as u32;
            if *w == 0 {
                *w = 1; // xoshiro requires non-zero state
            }
        }
        out.push(words);
    }
    out
}

pub struct BaoabPipeline {
    n_atoms: usize,
    first_half_pipeline: wgpu::ComputePipeline,
    second_half_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    params_buf: wgpu::Buffer,
    positions_buf: Arc<wgpu::Buffer>,
    velocities_buf: Arc<wgpu::Buffer>,
    masses_buf: wgpu::Buffer,
    forces_buf: Arc<wgpu::Buffer>,
    rng_state_buf: wgpu::Buffer,
    pos_readback_buf: wgpu::Buffer,
    vel_readback_buf: wgpu::Buffer,
    pos_padded: Vec<[f32; 4]>,
    vel_padded: Vec<[f32; 4]>,
    forces_padded: Vec<[f32; 4]>,
    ctx: &'static GpuContext,
}

impl BaoabPipeline {
    pub fn new(
        ctx: &'static GpuContext,
        n_atoms: usize,
        masses_da: &[f32],
        initial_rng_state: &[[u32; 4]],
    ) -> Self {
        Self::new_with_external_buffers(ctx, n_atoms, masses_da, initial_rng_state, None, None, None)
    }

    /// External-buffer constructor.  Pass `Some(buffer)` to bind any of
    /// (positions, velocities, forces) to a caller-owned buffer that's
    /// shared with other pipelines (bonded, nonbonded, GB) for the
    /// integrator-on-GPU path.  Pass `None` to have BaoabPipeline
    /// create its own buffer.  Whichever ones BaoabPipeline creates,
    /// it owns for its lifetime; the externally-supplied ones must
    /// outlive the BaoabPipeline.
    pub fn new_with_external_buffers(
        ctx: &'static GpuContext,
        n_atoms: usize,
        masses_da: &[f32],
        initial_rng_state: &[[u32; 4]],
        positions_buf: Option<Arc<wgpu::Buffer>>,
        velocities_buf: Option<Arc<wgpu::Buffer>>,
        forces_buf: Option<Arc<wgpu::Buffer>>,
    ) -> Self {
        assert_eq!(masses_da.len(), n_atoms);
        assert_eq!(initial_rng_state.len(), n_atoms);
        let device = &ctx.device;

        let n_padded = n_atoms * std::mem::size_of::<[f32; 4]>();
        let positions_buf = positions_buf.unwrap_or_else(|| {
            Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("baoab_positions"),
                size: n_padded as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }))
        });
        let velocities_buf = velocities_buf.unwrap_or_else(|| {
            Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("baoab_velocities"),
                size: n_padded as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }))
        });
        let forces_buf = forces_buf.unwrap_or_else(|| {
            Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("baoab_forces"),
                size: n_padded as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        });
        let masses_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("baoab_masses"),
            contents: bytemuck::cast_slice(masses_da),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let rng_state_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("baoab_rng_state"),
            contents: bytemuck::cast_slice(initial_rng_state),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        // Params buffer — content is overwritten each `dispatch_first_half`.
        let init_params = Params {
            n_atoms: n_atoms as u32,
            half_dt: 0.0,
            alpha: 0.0,
            o_sigma_sq_base: 0.0,
            accel_factor: 1.0e-4,
            _pad0: 0, _pad1: 0, _pad2: 0,
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("baoab_params"),
            contents: bytemuck::bytes_of(&init_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let pos_readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("baoab_pos_readback"),
            size: n_padded as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let vel_readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("baoab_vel_readback"),
            size: n_padded as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("baoab.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("baoab.wgsl").into()),
        });
        // Explicit shared bind group layout so both pipelines accept
        // the same bind group.  With implicit layouts (layout: None)
        // wgpu treats each pipeline's layout as distinct even when
        // they're structurally identical — and complains when you try
        // to set the same bind group on both.
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("baoab_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Bindings 1..5: storage buffers (positions, velocities
                // = read+write; masses, forces = read; rng_state = read+write).
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("baoab_pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let first_half_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("baoab_first_half_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("baoab_first_half"),
            compilation_options: Default::default(),
            cache: None,
        });
        let second_half_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("baoab_second_half_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("baoab_second_half"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("baoab_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: positions_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: velocities_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: masses_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: forces_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: rng_state_buf.as_entire_binding() },
            ],
        });

        Self {
            n_atoms,
            first_half_pipeline,
            second_half_pipeline,
            bind_group_layout,
            bind_group,
            params_buf,
            positions_buf,
            velocities_buf,
            masses_buf,
            forces_buf,
            rng_state_buf,
            pos_readback_buf,
            vel_readback_buf,
            pos_padded: vec![[0.0; 4]; n_atoms],
            vel_padded: vec![[0.0; 4]; n_atoms],
            forces_padded: vec![[0.0; 4]; n_atoms],
            ctx,
        }
    }

    pub fn upload_positions(&mut self, positions: &[[f32; 3]]) {
        assert_eq!(positions.len(), self.n_atoms);
        for (i, p) in positions.iter().enumerate() {
            self.pos_padded[i] = [p[0], p[1], p[2], 0.0];
        }
        self.ctx
            .queue
            .write_buffer(&self.positions_buf, 0, bytemuck::cast_slice(&self.pos_padded));
    }

    pub fn upload_velocities(&mut self, velocities: &[[f32; 3]]) {
        assert_eq!(velocities.len(), self.n_atoms);
        for (i, v) in velocities.iter().enumerate() {
            self.vel_padded[i] = [v[0], v[1], v[2], 0.0];
        }
        self.ctx
            .queue
            .write_buffer(&self.velocities_buf, 0, bytemuck::cast_slice(&self.vel_padded));
    }

    pub fn upload_forces(&mut self, forces: &[[f32; 3]]) {
        assert_eq!(forces.len(), self.n_atoms);
        for (i, f) in forces.iter().enumerate() {
            self.forces_padded[i] = [f[0], f[1], f[2], 0.0];
        }
        self.ctx
            .queue
            .write_buffer(&self.forces_buf, 0, bytemuck::cast_slice(&self.forces_padded));
    }

    fn write_params(&self, dt_fs: f32, gamma_ps_inv: f32, kbt_kj_mol: f32) {
        let alpha = (-gamma_ps_inv * dt_fs * 1.0e-3).exp();
        let one_minus_alpha2 = 1.0 - alpha * alpha;
        let accel_factor = 1.0e-4_f32;
        // σ² = (1 − α²) · k_B T · accel_factor  (then divide by m inside the kernel)
        let o_sigma_sq_base = one_minus_alpha2 * kbt_kj_mol * accel_factor;
        let params = Params {
            n_atoms: self.n_atoms as u32,
            half_dt: 0.5 * dt_fs,
            alpha,
            o_sigma_sq_base,
            accel_factor,
            _pad0: 0, _pad1: 0, _pad2: 0,
        };
        self.ctx
            .queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
    }

    /// Record the first-half dispatch (B → A → O → A) into the caller's
    /// encoder.  The caller must have already uploaded the current
    /// forces (`upload_forces`) and parameters (via
    /// [`set_step_params`](Self::set_step_params)).
    pub fn record_first_half(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("baoab_first_half_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.first_half_pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.n_atoms.div_ceil(64) as u32, 1, 1);
    }

    /// Record the second-half dispatch (B only) into the caller's
    /// encoder.  Forces buffer should contain forces re-evaluated at
    /// the post-first-half positions.
    pub fn record_second_half(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("baoab_second_half_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.second_half_pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.n_atoms.div_ceil(64) as u32, 1, 1);
    }

    pub fn set_step_params(&self, dt_fs: f32, gamma_ps_inv: f32, kbt_kj_mol: f32) {
        self.write_params(dt_fs, gamma_ps_inv, kbt_kj_mol);
    }

    pub fn positions_buffer(&self) -> &wgpu::Buffer { &self.positions_buf }
    pub fn velocities_buffer(&self) -> &wgpu::Buffer { &self.velocities_buf }
    pub fn forces_buffer(&self) -> &wgpu::Buffer { &self.forces_buf }

    /// Download current positions back to the CPU.  Used at save-frame
    /// boundaries.
    pub fn download_positions(&self) -> Vec<[f32; 3]> {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let size = (self.n_atoms * std::mem::size_of::<[f32; 4]>()) as u64;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("baoab_pos_dl_encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.positions_buf, 0, &self.pos_readback_buf, 0, size);
        queue.submit(Some(encoder.finish()));
        let slice = self.pos_readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let padded: &[[f32; 4]] = bytemuck::cast_slice(&data);
        let out: Vec<[f32; 3]> = padded.iter().map(|v| [v[0], v[1], v[2]]).collect();
        drop(data);
        self.pos_readback_buf.unmap();
        out
    }

    /// Download current velocities.
    pub fn download_velocities(&self) -> Vec<[f32; 3]> {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let size = (self.n_atoms * std::mem::size_of::<[f32; 4]>()) as u64;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("baoab_vel_dl_encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.velocities_buf, 0, &self.vel_readback_buf, 0, size);
        queue.submit(Some(encoder.finish()));
        let slice = self.vel_readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let padded: &[[f32; 4]] = bytemuck::cast_slice(&data);
        let out: Vec<[f32; 3]> = padded.iter().map(|v| [v[0], v[1], v[2]]).collect();
        drop(data);
        self.vel_readback_buf.unmap();
        out
    }

    pub fn n_atoms(&self) -> usize { self.n_atoms }

    #[allow(dead_code)]
    fn _ensure_layout_unused(&self) -> &wgpu::BindGroupLayout { &self.bind_group_layout }
}
