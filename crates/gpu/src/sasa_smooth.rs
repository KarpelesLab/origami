//! Smooth-coverage SASA pipeline — differentiable variant of
//! [`crate::SasaPipeline`].  Per-dot accessibility is `Π (1 - b_j)`
//! where `b_j` is a sigmoidal "buried-ness" weight; forces follow
//! by chain rule.  See `sasa_smooth.wgsl` for details.

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::context::GpuContext;
use crate::sasa::fibonacci_unit_sphere_f32;

pub const SASA_SMOOTH_N_DOTS: usize = 256;

/// Default smoothing width σ in Å.  σ → 0 recovers the binary
/// dot-density verdict; larger σ gives smoother but biased areas.
/// 0.3 Å gives a transition thickness of ~1 Å around each
/// neighbour's sphere boundary — a good middle ground.
pub const SASA_SMOOTH_DEFAULT_SIGMA_A: f32 = 0.3;

pub struct SasaSmoothSetup<'a> {
    pub radii: &'a [f32],
    /// Per-atom γ in kJ/mol/Å² (already unit-converted).  See
    /// `energy::powersasa::default_sasa_gammas`.
    pub gammas: &'a [f32],
    pub sigma_a: f32,
    pub initial_indices_capacity: usize,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n_atoms: u32,
    sigma: f32,
    _pad0: u32,
    _pad1: u32,
}

pub struct SasaSmoothPipeline {
    n_atoms: usize,
    area_pipeline: wgpu::ComputePipeline,
    force_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    positions_buf: Arc<wgpu::Buffer>,
    radii_buf: wgpu::Buffer,
    gammas_buf: wgpu::Buffer,
    dots_buf: wgpu::Buffer,
    nbr_count_buf: wgpu::Buffer,
    nbr_start_buf: wgpu::Buffer,
    nbr_indices_buf: wgpu::Buffer,
    nbr_indices_capacity: usize,
    per_atom_area_buf: wgpu::Buffer,
    forces_buf: Arc<wgpu::Buffer>,
    area_readback_buf: wgpu::Buffer,
    forces_readback_buf: wgpu::Buffer,
    area_size: u64,
    forces_size: u64,
    bind_group: wgpu::BindGroup,
    pos_padded: Vec<[f32; 4]>,
    ctx: &'static GpuContext,
}

impl SasaSmoothPipeline {
    pub fn new(ctx: &'static GpuContext, n_atoms: usize, setup: SasaSmoothSetup) -> Self {
        let positions_size = (n_atoms * std::mem::size_of::<[f32; 4]>()) as u64;
        let positions_buf = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sasa_smooth_positions"),
            size: positions_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let forces_buf = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sasa_smooth_forces"),
            size: positions_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        Self::new_with_external_buffers(ctx, n_atoms, setup, positions_buf, Some(forces_buf))
    }

    pub fn new_with_external_buffers(
        ctx: &'static GpuContext,
        n_atoms: usize,
        setup: SasaSmoothSetup,
        positions_buf: Arc<wgpu::Buffer>,
        forces_buf: Option<Arc<wgpu::Buffer>>,
    ) -> Self {
        assert_eq!(setup.radii.len(), n_atoms);
        assert_eq!(setup.gammas.len(), n_atoms);

        let device = &ctx.device;
        let positions_size = (n_atoms * std::mem::size_of::<[f32; 4]>()) as u64;
        let params = Params {
            n_atoms: n_atoms as u32,
            sigma: setup.sigma_a,
            _pad0: 0, _pad1: 0,
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sasa_smooth_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let radii_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sasa_smooth_radii"),
            contents: bytemuck::cast_slice(setup.radii),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let gammas_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sasa_smooth_gammas"),
            contents: bytemuck::cast_slice(setup.gammas),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let dots = fibonacci_unit_sphere_f32();
        let dots_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sasa_smooth_dots"),
            contents: bytemuck::cast_slice(&dots),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let nbr_count_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sasa_smooth_nbr_count"),
            size: (n_atoms * std::mem::size_of::<u32>()).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let nbr_start_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sasa_smooth_nbr_start"),
            size: (n_atoms * std::mem::size_of::<u32>()).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cap = setup.initial_indices_capacity.max(64);
        let nbr_indices_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sasa_smooth_nbr_indices"),
            size: (cap * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let area_size = (n_atoms * std::mem::size_of::<f32>()) as u64;
        let per_atom_area_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sasa_smooth_per_atom_area"),
            size: area_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let forces_size = positions_size;
        let forces_buf = forces_buf.unwrap_or_else(|| {
            Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sasa_smooth_forces"),
                size: forces_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        });
        let area_readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sasa_smooth_area_readback"),
            size: area_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let forces_readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sasa_smooth_forces_readback"),
            size: forces_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sasa_smooth.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("sasa_smooth.wgsl").into()),
        });
        // Explicit shared bind-group layout — the area entry point
        // doesn't reference `forces`, so its implicit layout would
        // exclude binding 9 and refuse our 10-entry bind group.
        let storage_ro = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let storage_rw = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let uniform = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let mk = |binding: u32, ty: wgpu::BindingType| -> wgpu::BindGroupLayoutEntry {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty,
                count: None,
            }
        };
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sasa_smooth_bgl"),
            entries: &[
                mk(0, uniform),
                mk(1, storage_ro),
                mk(2, storage_ro),
                mk(3, storage_ro),
                mk(4, storage_ro),
                mk(5, storage_ro),
                mk(6, storage_ro),
                mk(7, storage_ro),
                mk(8, storage_rw),
                mk(9, storage_rw),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sasa_smooth_pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let area_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sasa_smooth_area_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("sasa_smooth_area"),
            compilation_options: Default::default(),
            cache: None,
        });
        let force_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sasa_smooth_force_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("sasa_smooth_force"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = create_bind_group(
            device,
            &bind_group_layout,
            &params_buf,
            &positions_buf,
            &radii_buf,
            &gammas_buf,
            &dots_buf,
            &nbr_count_buf,
            &nbr_start_buf,
            &nbr_indices_buf,
            &per_atom_area_buf,
            &forces_buf,
        );

        Self {
            n_atoms,
            area_pipeline,
            force_pipeline,
            bind_group_layout,
            params_buf,
            positions_buf,
            radii_buf,
            gammas_buf,
            dots_buf,
            nbr_count_buf,
            nbr_start_buf,
            nbr_indices_buf,
            nbr_indices_capacity: cap,
            per_atom_area_buf,
            forces_buf,
            area_readback_buf,
            forces_readback_buf,
            area_size,
            forces_size,
            bind_group,
            pos_padded: vec![[0.0; 4]; n_atoms],
            ctx,
        }
    }

    pub fn update_positions(&mut self, positions: &[[f32; 3]]) {
        assert_eq!(positions.len(), self.n_atoms);
        for (i, p) in positions.iter().enumerate() {
            self.pos_padded[i] = [p[0], p[1], p[2], 0.0];
        }
        self.ctx
            .queue
            .write_buffer(&self.positions_buf, 0, bytemuck::cast_slice(&self.pos_padded));
    }

    pub fn update_neighbours(&mut self, counts: &[u32], starts: &[u32], indices: &[u32]) {
        assert_eq!(counts.len(), self.n_atoms);
        assert_eq!(starts.len(), self.n_atoms);
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        if indices.len() > self.nbr_indices_capacity {
            let new_cap = indices.len().next_power_of_two().max(64);
            self.nbr_indices_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sasa_smooth_nbr_indices"),
                size: (new_cap * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.nbr_indices_capacity = new_cap;
            self.bind_group = create_bind_group(
                device,
                &self.bind_group_layout,
                &self.params_buf,
                &self.positions_buf,
                &self.radii_buf,
                &self.gammas_buf,
                &self.dots_buf,
                &self.nbr_count_buf,
                &self.nbr_start_buf,
                &self.nbr_indices_buf,
                &self.per_atom_area_buf,
                &self.forces_buf,
            );
        }
        queue.write_buffer(&self.nbr_count_buf, 0, bytemuck::cast_slice(counts));
        queue.write_buffer(&self.nbr_start_buf, 0, bytemuck::cast_slice(starts));
        if !indices.is_empty() {
            queue.write_buffer(&self.nbr_indices_buf, 0, bytemuck::cast_slice(indices));
        }
    }

    /// Compute the smooth per-atom area and read it back.
    pub fn compute_area(&self) -> Vec<f32> {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sasa_smooth_area_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sasa_smooth_area_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.area_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.n_atoms.div_ceil(64) as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.per_atom_area_buf, 0, &self.area_readback_buf, 0, self.area_size,
        );
        queue.submit(Some(encoder.finish()));
        let slice = self.area_readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
        drop(data);
        self.area_readback_buf.unmap();
        out
    }

    /// Clear the forces buffer to zero.
    pub fn clear_forces(&self) {
        let zeroes = vec![0u8; self.forces_size as usize];
        self.ctx
            .queue
            .write_buffer(&self.forces_buf, 0, &zeroes);
    }

    /// Compute SASA forces (accumulates into the persistent forces
    /// buffer).  Call `clear_forces` first if you want a fresh
    /// total; otherwise this accumulates on top of whatever's
    /// already there — the integrator path relies on that
    /// accumulate behaviour to compose SASA with bonded / GB / etc.
    pub fn compute_forces(&self) -> Vec<[f32; 3]> {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sasa_smooth_force_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sasa_smooth_force_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.force_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.n_atoms.div_ceil(64) as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.forces_buf, 0, &self.forces_readback_buf, 0, self.forces_size);
        queue.submit(Some(encoder.finish()));
        let slice = self.forces_readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let padded: &[[f32; 4]] = bytemuck::cast_slice(&data);
        let out: Vec<[f32; 3]> = padded.iter().map(|v| [v[0], v[1], v[2]]).collect();
        drop(data);
        self.forces_readback_buf.unmap();
        out
    }
}

#[allow(clippy::too_many_arguments)]
fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: &wgpu::Buffer,
    positions: &wgpu::Buffer,
    radii: &wgpu::Buffer,
    gammas: &wgpu::Buffer,
    dots: &wgpu::Buffer,
    nbr_count: &wgpu::Buffer,
    nbr_start: &wgpu::Buffer,
    nbr_indices: &wgpu::Buffer,
    per_atom_area: &wgpu::Buffer,
    forces: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sasa_smooth_bind_group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: positions.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: radii.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: gammas.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: dots.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: nbr_count.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: nbr_start.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: nbr_indices.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: per_atom_area.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: forces.as_entire_binding() },
        ],
    })
}
