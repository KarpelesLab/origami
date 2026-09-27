//! CPU-callable wrapper around the WGSL LJ pair-force kernel.
//!
//! Mirrors the CPU AoS path in `energy::forces_nonbonded::add_nonbonded_forces`
//! exactly *except* for:
//!   - Coulomb is not yet computed (separate kernel coming).
//!   - CHARMM 1-4 special LJ parameters are not yet applied (the
//!     normal ε / Rmin/2 values are used for 1-4 pairs).  Same
//!     behaviour as the CPU path when 1-4 specials aren't set.
//!   - Cutoff is a hard sphere (no force-switching at the boundary).

use wgpu::util::DeviceExt;

use crate::context::GpuContext;

/// Input atom data + parameters for one LJ force evaluation.
pub struct LjInput<'a> {
    /// Atom positions, shape `n × 3`.  Caller provides Å.
    pub positions: &'a [[f32; 3]],
    /// Per-atom unique-type index (in `[0, lj_params.len())`).
    pub type_index: &'a [u32],
    /// Per-type (ε in kJ/mol, Rmin/2 in Å) — `lj_params[type_index[i]]`
    /// gives atom i's parameters.  Note: ε in **kJ/mol** (caller
    /// pre-multiplies by 4.184 from CHARMM kcal/mol).
    pub lj_params: &'a [[f32; 2]],
    /// Flat bitmap of pairs to exclude (1-2 / 1-3 / 1-4 typically),
    /// `n*n` bits packed into `u32`s in row-major (`i*n + j`) order.
    /// Length must be `ceil(n*n / 32)`.
    pub exclusions: &'a [u32],
    /// Hard cutoff in Å.
    pub cutoff_a: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n_atoms: u32,
    cutoff_sq: f32,
    _pad0: u32,
    _pad1: u32,
}

/// One LJ pair-force evaluation on the GPU.  Returns one `[f32; 3]`
/// force vector per input atom, in kJ/mol/Å.
///
/// Blocks until the GPU work completes and the readback finishes.
pub fn lj_force_gpu(ctx: &GpuContext, input: LjInput) -> Vec<[f32; 3]> {
    let n_atoms = input.positions.len();
    assert_eq!(input.type_index.len(), n_atoms);
    let n_exclusion_words = (n_atoms * n_atoms).div_ceil(32);
    assert_eq!(input.exclusions.len(), n_exclusion_words);

    // Positions and forces are vec4 in the shader (vec3 has 16-byte
    // stride anyway with WGSL alignment, so we pack as [x, y, z, 0]).
    let pos_padded: Vec<[f32; 4]> = input
        .positions
        .iter()
        .map(|p| [p[0], p[1], p[2], 0.0])
        .collect();

    let device = &ctx.device;
    let queue = &ctx.queue;

    let params = Params {
        n_atoms: n_atoms as u32,
        cutoff_sq: input.cutoff_a * input.cutoff_a,
        _pad0: 0,
        _pad1: 0,
    };

    // Buffers.
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("lj_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let positions_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("positions"),
        contents: bytemuck::cast_slice(&pos_padded),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let type_index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("type_index"),
        contents: bytemuck::cast_slice(input.type_index),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let lj_table_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("lj_table"),
        contents: bytemuck::cast_slice(input.lj_params),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let exclusions_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("exclusions"),
        contents: bytemuck::cast_slice(input.exclusions),
        usage: wgpu::BufferUsages::STORAGE,
    });
    // Output buffer (GPU-writable + COPY_SRC for readback).
    let forces_size = (n_atoms * std::mem::size_of::<[f32; 4]>()) as u64;
    let forces_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("forces"),
        size: forces_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    // Staging buffer for readback (MAP_READ + COPY_DST).
    let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: forces_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Shader + pipeline.
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("lj.wgsl"),
        source: wgpu::ShaderSource::Wgsl(include_str!("lj.wgsl").into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("lj_force"),
        layout: None,
        module: &shader,
        entry_point: Some("lj_force"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("lj_bind_group"),
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
                resource: type_index_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: lj_table_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: exclusions_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: forces_buf.as_entire_binding(),
            },
        ],
    });

    // Dispatch.
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("lj_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("lj_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let wg_count = n_atoms.div_ceil(64) as u32;
        pass.dispatch_workgroups(wg_count, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&forces_buf, 0, &readback_buf, 0, forces_size);
    queue.submit(Some(encoder.finish()));

    // Map the staging buffer and read it back.
    let slice = readback_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .expect("map_async sender dropped")
        .expect("buffer map");
    let data = slice.get_mapped_range();
    let padded: &[[f32; 4]] = bytemuck::cast_slice(&data);
    let out: Vec<[f32; 3]> = padded.iter().map(|v| [v[0], v[1], v[2]]).collect();
    drop(data);
    readback_buf.unmap();
    out
}
