//! Stateful Generalized-Born OBC II pipeline — runs both the
//! Born-radii kernel and the GB pair-force kernel.  Constructed
//! once for a given atom count + radii + scales + charges; each
//! step calls [`GbPipeline::update_positions`] then
//! [`GbPipeline::compute_forces`].

use wgpu::util::DeviceExt;

use crate::context::GpuContext;

pub struct GbSetup<'a> {
    /// Per-atom intrinsic vdW radius (Å).  Match
    /// `energy::gb::intrinsic_radius` exactly.
    pub rho: &'a [f32],
    /// Per-atom reduced radius (ρ − OBC_OFFSET, OBC_OFFSET = 0.09 Å).
    pub rho_tilde: &'a [f32],
    /// Per-atom HCT scale factor (`energy::gb::hct_scale`).
    pub scale: &'a [f32],
    /// Per-atom partial charge (e).
    pub charges: &'a [f32],
    /// Cutoff in Å.  CPU default is `BORN_RADIUS_CUTOFF_A = 20.0` for
    /// the Born-radius integral and `GB_DEFAULT_CUTOFF_A = 10.0` for
    /// the pair-force sum; this struct uses the *same* cutoff for
    /// both stages.  Pass 20.0 to match the CPU's Born-radius scan.
    pub cutoff_a: f32,
    /// Pair-force cutoff (typically 10 Å).
    pub pair_cutoff_a: f32,
    /// Initial capacity of the per-atom neighbour-index buffer in
    /// *entries*.  Grows automatically on a too-small upload.  At a
    /// 20 Å Born cutoff a dense all-atom system has ~3000-6000
    /// neighbours per atom, so `n_atoms * 5000` is a safe default.
    pub initial_indices_capacity: usize,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BornParams {
    n_atoms: u32,
    cutoff_sq: f32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ForceParams {
    n_atoms: u32,
    cutoff_sq: f32,
    prefactor_kj: f32,
    _pad: u32,
}

pub struct GbPipeline {
    n_atoms: usize,
    // Born-radius stage.
    born_pipeline: wgpu::ComputePipeline,
    born_bind_group_layout: wgpu::BindGroupLayout,
    born_bind_group: wgpu::BindGroup,
    born_params_buf: wgpu::Buffer,
    rho_tilde_buf: wgpu::Buffer,
    rho_buf: wgpu::Buffer,
    scale_buf: wgpu::Buffer,
    // Force stage.
    force_pipeline: wgpu::ComputePipeline,
    force_bind_group_layout: wgpu::BindGroupLayout,
    force_bind_group: wgpu::BindGroup,
    force_params_buf: wgpu::Buffer,
    charges_buf: wgpu::Buffer,
    // Shared buffers.
    positions_buf: wgpu::Buffer,
    // Holds the per-atom effective Born radii produced by `gb_born.wgsl`
    // and consumed by `gb_force.wgsl`.  Bound into both bind groups; we
    // don't read it on the CPU but we have to keep it alive while the
    // pipeline exists or wgpu will drop the GPU resource.
    #[allow(dead_code)]
    r_eff_buf: wgpu::Buffer,
    forces_buf: wgpu::Buffer,
    readback_buf: wgpu::Buffer,
    forces_size: u64,
    pos_padded: Vec<[f32; 4]>,
    // Neighbour-list (CSR) buffers — shared between both compute
    // passes.  The pair-force kernel filters internally by the
    // smaller 10 Å pair cutoff; the Born-radius kernel walks the
    // full 20 Å list.
    nbr_count_buf: wgpu::Buffer,
    nbr_start_buf: wgpu::Buffer,
    nbr_indices_buf: wgpu::Buffer,
    nbr_indices_capacity: usize,
    ctx: &'static GpuContext,
}

const EPSILON_WATER: f32 = 78.5;
const EPSILON_SOLUTE: f32 = 1.0;
const KCAL_TO_KJ: f32 = 4.184;
const COULOMB_KCAL_PER_E2: f32 = 332.0637;

impl GbPipeline {
    pub fn new(ctx: &'static GpuContext, n_atoms: usize, setup: GbSetup) -> Self {
        assert_eq!(setup.rho.len(), n_atoms);
        assert_eq!(setup.rho_tilde.len(), n_atoms);
        assert_eq!(setup.scale.len(), n_atoms);
        assert_eq!(setup.charges.len(), n_atoms);

        let device = &ctx.device;

        let positions_size = (n_atoms * std::mem::size_of::<[f32; 4]>()) as u64;
        let positions_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gb_positions"),
            size: positions_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let rho_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gb_rho"),
            contents: bytemuck::cast_slice(setup.rho),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let rho_tilde_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gb_rho_tilde"),
            contents: bytemuck::cast_slice(setup.rho_tilde),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let scale_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gb_scale"),
            contents: bytemuck::cast_slice(setup.scale),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let charges_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gb_charges"),
            contents: bytemuck::cast_slice(setup.charges),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let r_eff_size = (n_atoms * std::mem::size_of::<f32>()) as u64;
        let r_eff_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gb_r_eff"),
            size: r_eff_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let forces_size = positions_size;
        let forces_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gb_forces"),
            size: forces_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gb_readback"),
            size: forces_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---- Neighbour-list buffers (shared by both passes) ----
        let nbr_count_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gb_nbr_count"),
            size: (n_atoms * std::mem::size_of::<u32>()).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let nbr_start_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gb_nbr_start"),
            size: (n_atoms * std::mem::size_of::<u32>()).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cap = setup.initial_indices_capacity.max(64);
        let nbr_indices_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gb_nbr_indices"),
            size: (cap * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ---- Born-radius stage ----
        let born_params = BornParams {
            n_atoms: n_atoms as u32,
            cutoff_sq: setup.cutoff_a * setup.cutoff_a,
            _pad0: 0,
            _pad1: 0,
        };
        let born_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gb_born_params"),
            contents: bytemuck::bytes_of(&born_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let born_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gb_born.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gb_born.wgsl").into()),
        });
        let born_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gb_born_pipeline"),
            layout: None,
            module: &born_shader,
            entry_point: Some("born_radii"),
            compilation_options: Default::default(),
            cache: None,
        });
        let born_bind_group_layout = born_pipeline.get_bind_group_layout(0);
        let born_bind_group = create_born_bind_group(
            device,
            &born_bind_group_layout,
            &born_params_buf,
            &positions_buf,
            &rho_tilde_buf,
            &rho_buf,
            &scale_buf,
            &r_eff_buf,
            &nbr_count_buf,
            &nbr_start_buf,
            &nbr_indices_buf,
        );

        // ---- GB pair-force stage ----
        let prefactor_kj =
            (1.0 / EPSILON_SOLUTE - 1.0 / EPSILON_WATER) * COULOMB_KCAL_PER_E2 * KCAL_TO_KJ;
        let force_params = ForceParams {
            n_atoms: n_atoms as u32,
            cutoff_sq: setup.pair_cutoff_a * setup.pair_cutoff_a,
            prefactor_kj,
            _pad: 0,
        };
        let force_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gb_force_params"),
            contents: bytemuck::bytes_of(&force_params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let force_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gb_force.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gb_force.wgsl").into()),
        });
        let force_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gb_force_pipeline"),
            layout: None,
            module: &force_shader,
            entry_point: Some("gb_force"),
            compilation_options: Default::default(),
            cache: None,
        });
        let force_bind_group_layout = force_pipeline.get_bind_group_layout(0);
        let force_bind_group = create_force_bind_group(
            device,
            &force_bind_group_layout,
            &force_params_buf,
            &positions_buf,
            &charges_buf,
            &r_eff_buf,
            &forces_buf,
            &nbr_count_buf,
            &nbr_start_buf,
            &nbr_indices_buf,
        );

        Self {
            n_atoms,
            born_pipeline,
            born_bind_group_layout,
            born_bind_group,
            born_params_buf,
            rho_tilde_buf,
            rho_buf,
            scale_buf,
            force_pipeline,
            force_bind_group_layout,
            force_bind_group,
            force_params_buf,
            charges_buf,
            positions_buf,
            r_eff_buf,
            forces_buf,
            readback_buf,
            forces_size,
            pos_padded: vec![[0.0; 4]; n_atoms],
            nbr_count_buf,
            nbr_start_buf,
            nbr_indices_buf,
            nbr_indices_capacity: cap,
            ctx,
        }
    }

    /// Upload a fresh neighbour list — same CSR layout as the
    /// nonbonded Verlet pipeline: `counts[i]`, `starts[i]`, and a
    /// flat `indices` array with both directions of every pair.
    /// Excluded pairs (1-2 / 1-3 etc.) do not exist for GB, so no
    /// exclusion bitmap is needed.
    ///
    /// Auto-grows the indices buffer + rebuilds bind groups if the
    /// upload exceeds the current capacity.
    pub fn update_neighbours(&mut self, counts: &[u32], starts: &[u32], indices: &[u32]) {
        assert_eq!(counts.len(), self.n_atoms);
        assert_eq!(starts.len(), self.n_atoms);
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        if indices.len() > self.nbr_indices_capacity {
            let new_cap = indices.len().next_power_of_two().max(64);
            self.nbr_indices_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gb_nbr_indices"),
                size: (new_cap * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.nbr_indices_capacity = new_cap;
            self.born_bind_group = create_born_bind_group(
                device,
                &self.born_bind_group_layout,
                &self.born_params_buf,
                &self.positions_buf,
                &self.rho_tilde_buf,
                &self.rho_buf,
                &self.scale_buf,
                &self.r_eff_buf,
                &self.nbr_count_buf,
                &self.nbr_start_buf,
                &self.nbr_indices_buf,
            );
            self.force_bind_group = create_force_bind_group(
                device,
                &self.force_bind_group_layout,
                &self.force_params_buf,
                &self.positions_buf,
                &self.charges_buf,
                &self.r_eff_buf,
                &self.forces_buf,
                &self.nbr_count_buf,
                &self.nbr_start_buf,
                &self.nbr_indices_buf,
            );
        }
        queue.write_buffer(&self.nbr_count_buf, 0, bytemuck::cast_slice(counts));
        queue.write_buffer(&self.nbr_start_buf, 0, bytemuck::cast_slice(starts));
        if !indices.is_empty() {
            queue.write_buffer(&self.nbr_indices_buf, 0, bytemuck::cast_slice(indices));
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

    /// Dispatch the Born-radii pass *and* the pair-force pass.
    /// Returns one `[f32; 3]` force vector per atom.
    pub fn compute_forces(&self) -> Vec<[f32; 3]> {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gb_encoder"),
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

    /// Record both compute passes (Born radii → pair force) into a
    /// caller-owned encoder.  Used by `dynamics::GpuAccelerator` to
    /// fuse with the nonbonded kernel into a single submit.
    pub fn record_compute(&self, encoder: &mut wgpu::CommandEncoder) {
        let wg_count = self.n_atoms.div_ceil(64) as u32;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gb_born_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.born_pipeline);
            pass.set_bind_group(0, &self.born_bind_group, &[]);
            pass.dispatch_workgroups(wg_count, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gb_force_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.force_pipeline);
            pass.set_bind_group(0, &self.force_bind_group, &[]);
            pass.dispatch_workgroups(wg_count, 1, 1);
        }
    }

    /// Append a copy from the device-local forces buffer to the
    /// CPU-mappable readback buffer.
    pub fn record_readback_copy(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(&self.forces_buf, 0, &self.readback_buf, 0, self.forces_size);
    }

    /// See [`VerletNonbondedPipeline::begin_readback`].
    pub fn begin_readback(&self) -> std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>> {
        let slice = self.readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        rx
    }

    /// See [`VerletNonbondedPipeline::take_readback`].
    pub fn take_readback(&self) -> Vec<[f32; 3]> {
        let slice = self.readback_buf.slice(..);
        let data = slice.get_mapped_range();
        let padded: &[[f32; 4]] = bytemuck::cast_slice(&data);
        let out: Vec<[f32; 3]> = padded.iter().map(|v| [v[0], v[1], v[2]]).collect();
        drop(data);
        self.readback_buf.unmap();
        out
    }
}

#[allow(clippy::too_many_arguments)]
fn create_born_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: &wgpu::Buffer,
    positions: &wgpu::Buffer,
    rho_tilde: &wgpu::Buffer,
    rho: &wgpu::Buffer,
    scale: &wgpu::Buffer,
    r_eff: &wgpu::Buffer,
    nbr_count: &wgpu::Buffer,
    nbr_start: &wgpu::Buffer,
    nbr_indices: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gb_born_bind"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: positions.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: rho_tilde.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: rho.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: scale.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: r_eff.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: nbr_count.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: nbr_start.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: nbr_indices.as_entire_binding() },
        ],
    })
}

#[allow(clippy::too_many_arguments)]
fn create_force_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: &wgpu::Buffer,
    positions: &wgpu::Buffer,
    charges: &wgpu::Buffer,
    r_eff: &wgpu::Buffer,
    forces: &wgpu::Buffer,
    nbr_count: &wgpu::Buffer,
    nbr_start: &wgpu::Buffer,
    nbr_indices: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gb_force_bind"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: positions.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: charges.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: r_eff.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: forces.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: nbr_count.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: nbr_start.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: nbr_indices.as_entire_binding() },
        ],
    })
}
