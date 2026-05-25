//! Stateful tile-based nonbonded pipeline.
//!
//! Architectural variant of [`crate::VerletNonbondedPipeline`] that
//! uses workgroup shared memory + spatial tiling to cut global-memory
//! reads of j-atom data by ~64×.  See `tile_nonbonded.wgsl` for the
//! kernel details.  Setup signature is the same shape as
//! `VerletNonbondedSetup` except the neighbour list is replaced with
//! a per-i_tile interaction list.

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::context::GpuContext;
use crate::spatial_sort::TILE_SIZE;

pub struct TileNonbondedSetup<'a> {
    /// Per-atom packed LJ params: `[eps, rmin_half, eps_14, rmin_half_14]`.
    /// Same layout as the Verlet pipeline's setup.
    pub atom_lj_data: &'a [[f32; 4]],
    pub charges: &'a [f32],
    pub exclusions: &'a [u32],
    pub one_four_mask: &'a [u32],
    pub cutoff_a: f32,
    /// Initial capacity for the flat tile_indices buffer.  At
    /// n_tiles × ~10 average interacting j_tiles per i_tile, a
    /// generous `n_tiles * 64` is safely above worst-case.
    pub initial_tile_indices_capacity: usize,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n_atoms: u32,
    cutoff_sq: f32,
    inv_rc3: f32,
    _pad: u32,
}

pub struct TileNonbondedPipeline {
    n_atoms: usize,
    n_tiles: usize,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    positions_buf: Arc<wgpu::Buffer>,
    atom_lj_buf: wgpu::Buffer,
    charges_buf: wgpu::Buffer,
    exclusions_buf: wgpu::Buffer,
    one_four_buf: wgpu::Buffer,
    tile_count_buf: wgpu::Buffer,
    tile_start_buf: wgpu::Buffer,
    tile_indices_buf: wgpu::Buffer,
    tile_indices_capacity: usize,
    forces_buf: Arc<wgpu::Buffer>,
    readback_buf: wgpu::Buffer,
    forces_size: u64,
    bind_group: wgpu::BindGroup,
    pos_padded: Vec<[f32; 4]>,
    ctx: &'static GpuContext,
}

impl TileNonbondedPipeline {
    pub fn new(
        ctx: &'static GpuContext,
        n_atoms: usize,
        setup: TileNonbondedSetup,
    ) -> Self {
        let positions_size = (n_atoms * std::mem::size_of::<[f32; 4]>()) as u64;
        let positions_buf = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tnb_positions"),
            size: positions_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let forces_buf = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tnb_forces"),
            size: positions_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        Self::new_with_external_buffers(ctx, n_atoms, setup, positions_buf, Some(forces_buf))
    }

    /// Like [`new`](Self::new) but binds to caller-supplied position
    /// and (optionally) forces buffers — for the integrator path
    /// where all force kernels accumulate into one shared buffer.
    pub fn new_with_external_buffers(
        ctx: &'static GpuContext,
        n_atoms: usize,
        setup: TileNonbondedSetup,
        positions_buf: Arc<wgpu::Buffer>,
        forces_buf: Option<Arc<wgpu::Buffer>>,
    ) -> Self {
        assert_eq!(setup.atom_lj_data.len(), n_atoms);
        assert_eq!(setup.charges.len(), n_atoms);
        let n_excl_words = (n_atoms * n_atoms).div_ceil(32);
        assert_eq!(setup.exclusions.len(), n_excl_words);
        assert_eq!(setup.one_four_mask.len(), n_excl_words);
        let n_tiles = n_atoms.div_ceil(TILE_SIZE);

        let device = &ctx.device;
        let positions_size = (n_atoms * std::mem::size_of::<[f32; 4]>()) as u64;

        let params = Params {
            n_atoms: n_atoms as u32,
            cutoff_sq: setup.cutoff_a * setup.cutoff_a,
            inv_rc3: 1.0 / (setup.cutoff_a * setup.cutoff_a * setup.cutoff_a),
            _pad: 0,
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tnb_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let atom_lj_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tnb_atom_lj"),
            contents: bytemuck::cast_slice(setup.atom_lj_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let charges_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tnb_charges"),
            contents: bytemuck::cast_slice(setup.charges),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let exclusions_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tnb_exclusions"),
            contents: bytemuck::cast_slice(setup.exclusions),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let one_four_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tnb_one_four"),
            contents: bytemuck::cast_slice(setup.one_four_mask),
            usage: wgpu::BufferUsages::STORAGE,
        });
        // Per-i_tile CSR — zero-initialised, real values uploaded
        // via `update_tile_list`.
        let tile_count_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tnb_tile_count"),
            size: (n_tiles * std::mem::size_of::<u32>()).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tile_start_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tnb_tile_start"),
            size: (n_tiles * std::mem::size_of::<u32>()).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cap = setup.initial_tile_indices_capacity.max(64);
        let tile_indices_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tnb_tile_indices"),
            size: (cap * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let forces_size = positions_size;
        let forces_buf = forces_buf.unwrap_or_else(|| {
            Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tnb_forces"),
                size: forces_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        });
        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tnb_readback"),
            size: forces_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tile_nonbonded.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("tile_nonbonded.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tnb_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("tile_force"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = create_bind_group(
            device,
            &bind_group_layout,
            &params_buf,
            &positions_buf,
            &atom_lj_buf,
            &charges_buf,
            &exclusions_buf,
            &one_four_buf,
            &tile_count_buf,
            &tile_start_buf,
            &tile_indices_buf,
            &forces_buf,
        );

        Self {
            n_atoms,
            n_tiles,
            pipeline,
            bind_group_layout,
            params_buf,
            positions_buf,
            atom_lj_buf,
            charges_buf,
            exclusions_buf,
            one_four_buf,
            tile_count_buf,
            tile_start_buf,
            tile_indices_buf,
            tile_indices_capacity: cap,
            forces_buf,
            readback_buf,
            forces_size,
            bind_group,
            pos_padded: vec![[0.0; 4]; n_atoms],
            ctx,
        }
    }

    /// Upload a fresh per-i_tile interaction list.  Must be called
    /// before `record_compute` after every Verlet rebuild.
    pub fn update_tile_list(
        &mut self,
        tile_count: &[u32],
        tile_start: &[u32],
        tile_indices: &[u32],
    ) {
        assert_eq!(tile_count.len(), self.n_tiles);
        assert_eq!(tile_start.len(), self.n_tiles);
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        if tile_indices.len() > self.tile_indices_capacity {
            let new_cap = tile_indices.len().next_power_of_two().max(64);
            self.tile_indices_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tnb_tile_indices"),
                size: (new_cap * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.tile_indices_capacity = new_cap;
            self.bind_group = create_bind_group(
                device,
                &self.bind_group_layout,
                &self.params_buf,
                &self.positions_buf,
                &self.atom_lj_buf,
                &self.charges_buf,
                &self.exclusions_buf,
                &self.one_four_buf,
                &self.tile_count_buf,
                &self.tile_start_buf,
                &self.tile_indices_buf,
                &self.forces_buf,
            );
        }
        queue.write_buffer(&self.tile_count_buf, 0, bytemuck::cast_slice(tile_count));
        queue.write_buffer(&self.tile_start_buf, 0, bytemuck::cast_slice(tile_start));
        if !tile_indices.is_empty() {
            queue.write_buffer(&self.tile_indices_buf, 0, bytemuck::cast_slice(tile_indices));
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

    pub fn clear_forces(&self) {
        let zeroes = vec![0u8; self.forces_size as usize];
        self.ctx
            .queue
            .write_buffer(&self.forces_buf, 0, &zeroes);
    }

    pub fn record_compute(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tnb_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        // One workgroup per i_tile.
        pass.dispatch_workgroups(self.n_tiles as u32, 1, 1);
    }

    pub fn record_readback_copy(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(&self.forces_buf, 0, &self.readback_buf, 0, self.forces_size);
    }

    /// Standalone one-shot: clear forces, dispatch, copy back, read.
    pub fn compute(&self) -> Vec<[f32; 3]> {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        self.clear_forces();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tnb_encoder"),
        });
        self.record_compute(&mut encoder);
        self.record_readback_copy(&mut encoder);
        queue.submit(Some(encoder.finish()));
        let slice = self.readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let _ = device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("map_async sender dropped").expect("buffer map");
        let data = slice.get_mapped_range();
        let padded: &[[f32; 4]] = bytemuck::cast_slice(&data);
        let out: Vec<[f32; 3]> = padded.iter().map(|v| [v[0], v[1], v[2]]).collect();
        drop(data);
        self.readback_buf.unmap();
        out
    }

    pub fn n_tiles(&self) -> usize { self.n_tiles }
}

#[allow(clippy::too_many_arguments)]
fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: &wgpu::Buffer,
    positions: &wgpu::Buffer,
    atom_lj: &wgpu::Buffer,
    charges: &wgpu::Buffer,
    exclusions: &wgpu::Buffer,
    one_four: &wgpu::Buffer,
    tile_count: &wgpu::Buffer,
    tile_start: &wgpu::Buffer,
    tile_indices: &wgpu::Buffer,
    forces: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tnb_bind_group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: positions.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: atom_lj.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: charges.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: exclusions.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: one_four.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: tile_count.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: tile_start.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: tile_indices.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: forces.as_entire_binding() },
        ],
    })
}
