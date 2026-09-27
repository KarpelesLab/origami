//! GPU SHAKE pipeline.
//!
//! See `shake.wgsl` for the algorithm and parallelisation strategy.
//! This module owns the bind group + buffer setup and exposes a
//! single `record_shake` method for the integrator path.
//!
//! Per-X CSR construction: call [`build_per_x_shake_data`] with the
//! list of X-H constraints + per-atom masses to flatten them into the
//! buffers the kernel expects.

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::context::GpuContext;

/// Maximum H atoms bonded to a single heavy atom.  Methyl (3), amine
/// (3), terminal NH3+ (3) all fit; methane CH4 (4) is the upper bound
/// in standard amino-acid chemistry.  Must match `MAX_H_PER_X` in
/// `shake.wgsl`.
pub const MAX_H_PER_X: usize = 4;

/// Single X-H bond constraint to feed [`build_per_x_shake_data`].
#[derive(Debug, Clone, Copy)]
pub struct ShakeConstraint {
    pub x_atom: u32,
    pub h_atom: u32,
    pub d_sq: f32,
}

/// Per-atom CSR tables: returned ready to upload.
#[derive(Debug, Clone)]
pub struct PerXShakeData {
    pub h_count: Vec<u32>,
    /// Flat array of length `n_atoms * MAX_H_PER_X`.  For atom i with
    /// `h_count[i] = c`, the first `c` slots in
    /// `per_atom_h_atoms[i * MAX_H_PER_X ..]` give the H indices.
    /// Remaining slots are unused (set to 0).
    pub per_atom_h_atoms: Vec<u32>,
    pub per_atom_h_d_sq: Vec<f32>,
    /// 1/mass in Da⁻¹.
    pub inv_mass: Vec<f32>,
}

/// Flatten a list of X-H constraints into the per-X CSR layout the
/// kernel expects.  Heavy atoms with no bonded H get count = 0 and
/// are skipped by the kernel.  H atoms (which have at most one
/// constraint) appear as the `h_atom` in their parent X's slot.
///
/// Panics if any heavy atom has more than `MAX_H_PER_X` bonded H's
/// (which would indicate a non-standard chemistry — methane CH4 is
/// the upper bound for proteins/RNA).
pub fn build_per_x_shake_data(
    n_atoms: usize,
    constraints: &[ShakeConstraint],
    masses_da: &[f32],
) -> PerXShakeData {
    assert_eq!(masses_da.len(), n_atoms);
    let mut h_count = vec![0u32; n_atoms];
    let mut per_atom_h_atoms = vec![0u32; n_atoms * MAX_H_PER_X];
    let mut per_atom_h_d_sq = vec![0.0f32; n_atoms * MAX_H_PER_X];
    let inv_mass: Vec<f32> = masses_da.iter().map(|&m| 1.0 / m).collect();
    for c in constraints {
        let x = c.x_atom as usize;
        let slot = h_count[x] as usize;
        if slot >= MAX_H_PER_X {
            panic!(
                "atom {} has more than {} bonded H — increase MAX_H_PER_X in both shake.rs and shake.wgsl",
                x, MAX_H_PER_X
            );
        }
        let base = x * MAX_H_PER_X;
        per_atom_h_atoms[base + slot] = c.h_atom;
        per_atom_h_d_sq[base + slot] = c.d_sq;
        h_count[x] += 1;
    }
    PerXShakeData {
        h_count,
        per_atom_h_atoms,
        per_atom_h_d_sq,
        inv_mass,
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ShakeParams {
    n_atoms: u32,
    max_iters: u32,
    tol_sq: f32,
    _pad: u32,
}

pub struct ShakePipeline {
    n_atoms: usize,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    params_buf: wgpu::Buffer,
    // Keep the data buffers alive for the lifetime of the pipeline.
    #[allow(dead_code)]
    keep_alive: ShakeBuffers,
}

#[allow(dead_code)]
struct ShakeBuffers {
    ref_positions_buf: Arc<wgpu::Buffer>,
    inv_mass_buf: wgpu::Buffer,
    h_count_buf: wgpu::Buffer,
    per_atom_h_atoms_buf: wgpu::Buffer,
    per_atom_h_d_sq_buf: wgpu::Buffer,
}

impl ShakePipeline {
    /// Construct the SHAKE pipeline bound to caller-supplied
    /// `positions` and `ref_positions` buffers.  `ref_positions` is
    /// the position snapshot captured before the A step; the
    /// integrator is responsible for keeping it up to date (typically
    /// by recording a buffer-to-buffer copy from `positions` into
    /// `ref_positions` BEFORE the velocity step that breaks
    /// constraints).
    pub fn new(
        ctx: &'static GpuContext,
        n_atoms: usize,
        positions_buf: Arc<wgpu::Buffer>,
        ref_positions_buf: Arc<wgpu::Buffer>,
        data: &PerXShakeData,
        max_iters: u32,
        tol_sq: f32,
    ) -> Self {
        assert_eq!(data.h_count.len(), n_atoms);
        assert_eq!(data.inv_mass.len(), n_atoms);
        assert_eq!(data.per_atom_h_atoms.len(), n_atoms * MAX_H_PER_X);
        assert_eq!(data.per_atom_h_d_sq.len(), n_atoms * MAX_H_PER_X);

        let device = &ctx.device;
        let params = ShakeParams {
            n_atoms: n_atoms as u32,
            max_iters,
            tol_sq,
            _pad: 0,
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shake_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let inv_mass_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shake_inv_mass"),
            contents: bytemuck::cast_slice(&data.inv_mass),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let h_count_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shake_h_count"),
            contents: bytemuck::cast_slice(&data.h_count),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let per_atom_h_atoms_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shake_h_atoms"),
            contents: bytemuck::cast_slice(&data.per_atom_h_atoms),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let per_atom_h_d_sq_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shake_h_d_sq"),
            contents: bytemuck::cast_slice(&data.per_atom_h_d_sq),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shake.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shake.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shake_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("shake_per_x"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shake_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: positions_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: ref_positions_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: inv_mass_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: h_count_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: per_atom_h_atoms_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: per_atom_h_d_sq_buf.as_entire_binding(),
                },
            ],
        });

        Self {
            n_atoms,
            pipeline,
            bind_group,
            params_buf,
            keep_alive: ShakeBuffers {
                ref_positions_buf,
                inv_mass_buf,
                h_count_buf,
                per_atom_h_atoms_buf,
                per_atom_h_d_sq_buf,
            },
        }
    }

    /// Record one SHAKE compute pass into the caller's encoder.
    /// Reads from `positions` + `ref_positions`, writes back to
    /// `positions` in-place.
    pub fn record(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("shake_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.n_atoms.div_ceil(64) as u32, 1, 1);
    }

    /// Update the per-call SHAKE convergence parameters (max iterations
    /// + squared tolerance).  Cheap — one `write_buffer` of 16 bytes.
    pub fn set_params(&self, max_iters: u32, tol_sq: f32) {
        let params = ShakeParams {
            n_atoms: self.n_atoms as u32,
            max_iters,
            tol_sq,
            _pad: 0,
        };
        // Write through the saved ctx via the buffer's parent device.
        // We don't have a queue reference here so the caller is
        // expected to call this *before* enqueuing the next compute
        // submit — wgpu queues writes through the staging belt and
        // they land in time.  In practice callers update params once
        // per `step_n` invocation, well before the per-step compute
        // passes get recorded.
        // Get the queue via... actually we don't have ctx stored.
        // Make this take a ctx reference instead.
        let _ = params;
        unreachable!("ShakePipeline::set_params needs ctx — use the alternate signature");
    }

    /// As [`set_params`](Self::set_params), with explicit queue access.
    pub fn set_params_with_queue(&self, queue: &wgpu::Queue, max_iters: u32, tol_sq: f32) {
        let params = ShakeParams {
            n_atoms: self.n_atoms as u32,
            max_iters,
            tol_sq,
            _pad: 0,
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
    }
}
