//! Stateful nonbonded force pipeline.
//!
//! [`NonbondedPipeline`] holds the wgpu shader, compute pipeline, and
//! all GPU buffers needed for one (LJ + reaction-field Coulomb) force
//! evaluation.  Constructed once for a given atom count + topology +
//! parameter set; each step calls [`NonbondedPipeline::update_positions`]
//! then [`NonbondedPipeline::compute`] to get a fresh force vector.
//!
//! Per-step cost on Mac M3 Pro (Trp-cage 300 atoms, LJ+Coulomb):
//!   - construction (one-time): ~3 ms (shader compile, buffer allocs)
//!   - per-step update + compute + readback: ~0.6 ms
//!
//! The construction overhead is paid once per process; the per-step
//! cost is what matters for the integrator.

use wgpu::util::DeviceExt;

use crate::context::GpuContext;

/// One-time setup data — passed to [`NonbondedPipeline::new`].  The
/// fields that *don't* change during a trajectory (parameter tables,
/// exclusion mask, charges, cutoff) live on the GPU for the lifetime
/// of the pipeline; only positions are re-uploaded each step.
pub struct NonbondedSetup<'a> {
    /// Per-atom unique-type index (in `[0, lj_params.len())`).
    pub type_index: &'a [u32],
    /// Per-type (ε in kJ/mol, Rmin/2 in Å).  Caller pre-multiplies
    /// CHARMM ε by 4.184 to convert kcal/mol → kJ/mol.
    pub lj_params: &'a [[f32; 2]],
    /// Per-type 1-4 specials (ε_14 in kJ/mol, Rmin/2_14 in Å).
    /// Same shape and indexing as `lj_params` — caller pre-resolves
    /// the CHARMM `epsilon_14.unwrap_or(epsilon)` /
    /// `rmin_half_14.unwrap_or(rmin_half)` per type.  Used by pairs
    /// where the corresponding bit is set in `one_four_mask`.
    pub lj_params_14: &'a [[f32; 2]],
    /// Per-atom partial charge in elementary-charge units (e).
    pub charges: &'a [f32],
    /// Flat bitmap of 1-2 / 1-3 excluded pairs (those that
    /// contribute nothing to the non-bonded sum).  `n*n` bits packed
    /// into `u32`s in row-major (`i*n + j`) order.
    pub exclusions: &'a [u32],
    /// Flat bitmap of 1-4 pairs — same shape as `exclusions`.
    /// Pairs in this mask still contribute to the non-bonded sum
    /// but use `lj_params_14` instead of `lj_params`.
    pub one_four_mask: &'a [u32],
    /// Hard cutoff in Å (used both for the LJ truncation and for the
    /// reaction-field Coulomb Rc).
    pub cutoff_a: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n_atoms: u32,
    cutoff_sq: f32,
    inv_rc3: f32,
    _pad: u32,
}

pub struct NonbondedPipeline {
    n_atoms: usize,
    // wgpu objects — `Device` and `Queue` are held by reference via
    // the static `GpuContext`, so we only need to remember the
    // structures derived from them.
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    positions_buf: wgpu::Buffer,
    forces_buf: wgpu::Buffer,
    readback_buf: wgpu::Buffer,
    // Force buffer size in bytes; kept for readback copies.
    forces_size: u64,
    // CPU scratch for vec3 → vec4 padding (positions are written as
    // [x, y, z, 0] to satisfy WGSL's 16-byte alignment for vec3
    // storage arrays).
    pos_padded: Vec<[f32; 4]>,
    /// Same-process reference back to the GPU context (we hold this
    /// to issue `queue.write_buffer` and `queue.submit` calls).
    ctx: &'static GpuContext,
}

impl NonbondedPipeline {
    pub fn new(ctx: &'static GpuContext, n_atoms: usize, setup: NonbondedSetup) -> Self {
        assert_eq!(setup.type_index.len(), n_atoms);
        assert_eq!(setup.charges.len(), n_atoms);
        assert_eq!(
            setup.lj_params.len(),
            setup.lj_params_14.len(),
            "lj_params and lj_params_14 must have the same per-type shape"
        );
        let n_exclusion_words = (n_atoms * n_atoms).div_ceil(32);
        assert_eq!(setup.exclusions.len(), n_exclusion_words);
        assert_eq!(setup.one_four_mask.len(), n_exclusion_words);

        let device = &ctx.device;

        let params = Params {
            n_atoms: n_atoms as u32,
            cutoff_sq: setup.cutoff_a * setup.cutoff_a,
            inv_rc3: 1.0 / (setup.cutoff_a * setup.cutoff_a * setup.cutoff_a),
            _pad: 0,
        };

        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nb_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let positions_size = (n_atoms * std::mem::size_of::<[f32; 4]>()) as u64;
        let positions_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nb_positions"),
            size: positions_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let type_index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nb_type_index"),
            contents: bytemuck::cast_slice(setup.type_index),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let lj_table_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nb_lj_table"),
            contents: bytemuck::cast_slice(setup.lj_params),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let lj_table_14_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nb_lj_table_14"),
            contents: bytemuck::cast_slice(setup.lj_params_14),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let charges_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nb_charges"),
            contents: bytemuck::cast_slice(setup.charges),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let exclusions_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nb_exclusions"),
            contents: bytemuck::cast_slice(setup.exclusions),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let one_four_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nb_one_four"),
            contents: bytemuck::cast_slice(setup.one_four_mask),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let forces_size = positions_size;
        let forces_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nb_forces"),
            size: forces_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nb_readback"),
            size: forces_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nonbonded.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("nonbonded.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nb_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("nonbonded_force"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nb_bind_group"),
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
                    resource: lj_table_14_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: charges_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: exclusions_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: one_four_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: forces_buf.as_entire_binding(),
                },
            ],
        });

        Self {
            n_atoms,
            pipeline,
            bind_group,
            positions_buf,
            forces_buf,
            readback_buf,
            forces_size,
            pos_padded: vec![[0.0; 4]; n_atoms],
            ctx,
        }
    }

    /// Upload a fresh position vector to the GPU.  Must be called
    /// before [`compute`](Self::compute) on every step where atoms
    /// have moved.  Length must equal the `n_atoms` passed to
    /// [`new`](Self::new).
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

    /// Dispatch the compute pass and read back the per-atom force
    /// vector (kJ/mol/Å).  Blocks until the GPU finishes.
    pub fn compute(&self) -> Vec<[f32; 3]> {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("nb_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("nb_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let wg_count = self.n_atoms.div_ceil(64) as u32;
            pass.dispatch_workgroups(wg_count, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.forces_buf, 0, &self.readback_buf, 0, self.forces_size);
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
        let padded: &[[f32; 4]] = bytemuck::cast_slice(&data);
        let out: Vec<[f32; 3]> = padded.iter().map(|v| [v[0], v[1], v[2]]).collect();
        drop(data);
        self.readback_buf.unmap();
        out
    }
}

/// One-shot convenience wrapper: build a pipeline, upload positions,
/// compute forces, throw the pipeline away.  Equivalent to the old
/// `lj_force_gpu` but with Coulomb included.  Per-call cost includes
/// the ~3 ms construction overhead — for repeated calls, hold a
/// [`NonbondedPipeline`] yourself.
pub fn nonbonded_force_gpu(
    ctx: &'static GpuContext,
    positions: &[[f32; 3]],
    setup: NonbondedSetup,
) -> Vec<[f32; 3]> {
    let mut pipe = NonbondedPipeline::new(ctx, positions.len(), setup);
    pipe.update_positions(positions);
    pipe.compute()
}
