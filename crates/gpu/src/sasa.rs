//! GPU SASA pipeline — dot-density (Shrake-Rupley) per-atom area.
//!
//! `SasaPipeline` wraps the WGSL kernel + bind group + buffers.
//! Constructed once for a given set of atoms + their expanded radii
//! and the Fibonacci dot pattern.  Per call:
//!
//!   1. `update_positions(...)` — upload current positions.
//!   2. `update_neighbours(...)` — upload SASA Verlet CSR (built on
//!      CPU from a cell list at the sum-of-expanded-radii cutoff
//!      plus a Verlet skin).
//!   3. `compute_area() -> Vec<f32>` — dispatch + read back the
//!      per-atom area in Å².

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::context::GpuContext;

pub const SASA_N_DOTS: usize = 256;

/// Generate `N_DOTS` Fibonacci-spiral unit vectors on the sphere.
/// Identical pattern to `energy::sasa::fibonacci_unit_sphere` so the
/// GPU and CPU implementations are using the same test points.
pub fn fibonacci_unit_sphere_f32() -> Vec<[f32; 4]> {
    let n = SASA_N_DOTS;
    let phi = std::f64::consts::PI * (3.0_f64.sqrt() + 1.0);
    let mut out = Vec::with_capacity(n);
    let n_f = n as f64;
    for i in 0..n {
        let y = 1.0 - 2.0 * (i as f64) / (n_f - 1.0);
        let r = (1.0 - y * y).sqrt();
        let theta = phi * (i as f64);
        let x = theta.cos() * r;
        let z = theta.sin() * r;
        out.push([x as f32, y as f32, z as f32, 0.0]);
    }
    out
}

pub struct SasaSetup<'a> {
    /// Expanded radius (vdW + probe) per atom, in Å.
    pub radii: &'a [f32],
    /// Initial CSR capacity in entries.  Typical SASA neighbour
    /// counts are 10-30 per atom; `n_atoms * 60` is a safe initial
    /// estimate.
    pub initial_indices_capacity: usize,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n_atoms: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub struct SasaPipeline {
    n_atoms: usize,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    positions_buf: Arc<wgpu::Buffer>,
    radii_buf: wgpu::Buffer,
    dots_buf: wgpu::Buffer,
    nbr_count_buf: wgpu::Buffer,
    nbr_start_buf: wgpu::Buffer,
    nbr_indices_buf: wgpu::Buffer,
    nbr_indices_capacity: usize,
    per_atom_area_buf: wgpu::Buffer,
    readback_buf: wgpu::Buffer,
    area_size: u64,
    bind_group: wgpu::BindGroup,
    pos_padded: Vec<[f32; 4]>,
    ctx: &'static GpuContext,
}

impl SasaPipeline {
    pub fn new(ctx: &'static GpuContext, n_atoms: usize, setup: SasaSetup) -> Self {
        let positions_size = (n_atoms * std::mem::size_of::<[f32; 4]>()) as u64;
        let positions_buf = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sasa_positions"),
            size: positions_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        Self::new_with_external_positions(ctx, n_atoms, setup, positions_buf)
    }

    pub fn new_with_external_positions(
        ctx: &'static GpuContext,
        n_atoms: usize,
        setup: SasaSetup,
        positions_buf: Arc<wgpu::Buffer>,
    ) -> Self {
        assert_eq!(setup.radii.len(), n_atoms);

        let device = &ctx.device;
        let params = Params {
            n_atoms: n_atoms as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sasa_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let radii_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sasa_radii"),
            contents: bytemuck::cast_slice(setup.radii),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let dots = fibonacci_unit_sphere_f32();
        let dots_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sasa_dots"),
            contents: bytemuck::cast_slice(&dots),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let nbr_count_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sasa_nbr_count"),
            size: (n_atoms * std::mem::size_of::<u32>()).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let nbr_start_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sasa_nbr_start"),
            size: (n_atoms * std::mem::size_of::<u32>()).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cap = setup.initial_indices_capacity.max(64);
        let nbr_indices_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sasa_nbr_indices"),
            size: (cap * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let area_size = (n_atoms * std::mem::size_of::<f32>()) as u64;
        let per_atom_area_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sasa_per_atom_area"),
            size: area_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sasa_readback"),
            size: area_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sasa.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("sasa.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sasa_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("sasa_dot_density"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = create_bind_group(
            device,
            &bind_group_layout,
            &params_buf,
            &positions_buf,
            &radii_buf,
            &dots_buf,
            &nbr_count_buf,
            &nbr_start_buf,
            &nbr_indices_buf,
            &per_atom_area_buf,
        );

        Self {
            n_atoms,
            pipeline,
            bind_group_layout,
            params_buf,
            positions_buf,
            radii_buf,
            dots_buf,
            nbr_count_buf,
            nbr_start_buf,
            nbr_indices_buf,
            nbr_indices_capacity: cap,
            per_atom_area_buf,
            readback_buf,
            area_size,
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
        self.ctx.queue.write_buffer(
            &self.positions_buf,
            0,
            bytemuck::cast_slice(&self.pos_padded),
        );
    }

    pub fn update_neighbours(&mut self, counts: &[u32], starts: &[u32], indices: &[u32]) {
        assert_eq!(counts.len(), self.n_atoms);
        assert_eq!(starts.len(), self.n_atoms);
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        if indices.len() > self.nbr_indices_capacity {
            let new_cap = indices.len().next_power_of_two().max(64);
            self.nbr_indices_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sasa_nbr_indices"),
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
                &self.dots_buf,
                &self.nbr_count_buf,
                &self.nbr_start_buf,
                &self.nbr_indices_buf,
                &self.per_atom_area_buf,
            );
        }
        queue.write_buffer(&self.nbr_count_buf, 0, bytemuck::cast_slice(counts));
        queue.write_buffer(&self.nbr_start_buf, 0, bytemuck::cast_slice(starts));
        if !indices.is_empty() {
            queue.write_buffer(&self.nbr_indices_buf, 0, bytemuck::cast_slice(indices));
        }
    }

    /// Dispatch + read back per-atom accessible area in Å².
    pub fn compute_area(&self) -> Vec<f32> {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sasa_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sasa_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.n_atoms.div_ceil(64) as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.per_atom_area_buf,
            0,
            &self.readback_buf,
            0,
            self.area_size,
        );
        queue.submit(Some(encoder.finish()));
        let slice = self.readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("map_async sender dropped")
            .expect("buffer map");
        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
        drop(data);
        self.readback_buf.unmap();
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
    dots: &wgpu::Buffer,
    nbr_count: &wgpu::Buffer,
    nbr_start: &wgpu::Buffer,
    nbr_indices: &wgpu::Buffer,
    per_atom_area: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sasa_bind_group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: positions.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: radii.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: dots.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: nbr_count.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: nbr_start.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: nbr_indices.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: per_atom_area.as_entire_binding(),
            },
        ],
    })
}
