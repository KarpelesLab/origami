//! GPU nonbonded force pipeline that walks a precomputed Verlet
//! neighbour list instead of doing the O(N²) all-pairs scan.
//!
//! Why a CPU-built list instead of building the list on-GPU:
//!
//! - The Verlet list is rebuilt only every ~30-100 MD steps (whenever
//!   an atom drifts more than `VERLET_SKIN / 2`).  Building it via a
//!   cell list on the CPU is already O(N) and amortises across many
//!   force evaluations.  Adding a cell-list construction kernel to
//!   the GPU adds significant complexity (atomic counters, variable
//!   per-cell capacity, two-pass prefix-sum) for negligible win.
//! - The pair-force inner loop, which *does* run every step, becomes
//!   O(⟨neighbours⟩) per atom instead of O(N) — the actual hot path.
//!
//! API mirrors [`NonbondedPipeline`]:
//!   1. construct once with [`new`](Self::new)
//!   2. per skin-rebuild, call [`update_neighbours`](Self::update_neighbours)
//!   3. per force eval, call [`update_positions`](Self::update_positions)
//!      and [`compute`](Self::compute)
//!
//! Same kJ/mol/Å sign convention, same 1-4 LJ specials handling, same
//! reaction-field Coulomb sign as `NonbondedPipeline`.

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::context::GpuContext;

/// One-time setup data for the Verlet pipeline.  Identical fields to
/// [`crate::nonbonded::NonbondedSetup`] except that the candidate-pair
/// list is no longer enumerated inside the kernel — instead, the caller
/// uploads a per-atom neighbour list via [`VerletNonbondedPipeline::update_neighbours`].
pub struct VerletNonbondedSetup<'a> {
    /// Per-atom LJ parameters, pre-resolved from the
    /// (type → params) indirection: each entry is
    /// `[eps, rmin_half, eps_14, rmin_half_14]`.  Doing the
    /// type-lookup up front eliminates one indirect global load
    /// per inner-loop iteration on the GPU.
    pub atom_lj_data: &'a [[f32; 4]],
    pub charges: &'a [f32],
    pub exclusions: &'a [u32],
    pub one_four_mask: &'a [u32],
    pub cutoff_a: f32,
    /// Initial capacity of the neighbour-index buffer in *entries*.
    /// The buffer grows automatically if a later
    /// [`update_neighbours`](VerletNonbondedPipeline::update_neighbours)
    /// supplies more indices than capacity, but giving a realistic
    /// initial estimate avoids the first reallocation.  A typical
    /// dense all-atom system has ~150-300 neighbours per atom at a
    /// 10-Å cutoff; `n_atoms * 200` is a safe over-estimate.
    pub initial_indices_capacity: usize,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n_atoms: u32,
    cutoff_sq: f32,
    inv_rc3: f32,
    _pad: u32,
}

pub struct VerletNonbondedPipeline {
    n_atoms: usize,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    positions_buf: Arc<wgpu::Buffer>,
    atom_lj_buf: wgpu::Buffer,
    charges_buf: wgpu::Buffer,
    exclusions_buf: wgpu::Buffer,
    one_four_buf: wgpu::Buffer,
    nbr_count_buf: wgpu::Buffer,
    nbr_start_buf: wgpu::Buffer,
    nbr_indices_buf: wgpu::Buffer,
    nbr_indices_capacity: usize,
    forces_buf: Arc<wgpu::Buffer>,
    readback_buf: wgpu::Buffer,
    forces_size: u64,
    bind_group: wgpu::BindGroup,
    pos_padded: Vec<[f32; 4]>,
    ctx: &'static GpuContext,
}

impl VerletNonbondedPipeline {
    pub fn new(ctx: &'static GpuContext, n_atoms: usize, setup: VerletNonbondedSetup) -> Self {
        let positions_size = (n_atoms * std::mem::size_of::<[f32; 4]>()) as u64;
        let positions_buf = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nbv_positions"),
            size: positions_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let forces_buf = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nbv_forces"),
            size: positions_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        Self::new_with_external_buffers(ctx, n_atoms, setup, positions_buf, Some(forces_buf))
    }

    /// Like [`new`](Self::new) but binds to caller-supplied positions
    /// and (optionally) forces buffers — they're wrapped in `Arc` so
    /// the integrator-on-GPU path can share them across pipelines.
    pub fn new_with_external_buffers(
        ctx: &'static GpuContext,
        n_atoms: usize,
        setup: VerletNonbondedSetup,
        positions_buf: Arc<wgpu::Buffer>,
        forces_buf: Option<Arc<wgpu::Buffer>>,
    ) -> Self {
        assert_eq!(setup.atom_lj_data.len(), n_atoms);
        assert_eq!(setup.charges.len(), n_atoms);
        let n_excl_words = (n_atoms * n_atoms).div_ceil(32);
        assert_eq!(setup.exclusions.len(), n_excl_words);
        assert_eq!(setup.one_four_mask.len(), n_excl_words);

        let device = &ctx.device;

        let params = Params {
            n_atoms: n_atoms as u32,
            cutoff_sq: setup.cutoff_a * setup.cutoff_a,
            inv_rc3: 1.0 / (setup.cutoff_a * setup.cutoff_a * setup.cutoff_a),
            _pad: 0,
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nbv_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let positions_size = (n_atoms * std::mem::size_of::<[f32; 4]>()) as u64;
        let atom_lj_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nbv_atom_lj"),
            contents: bytemuck::cast_slice(setup.atom_lj_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let charges_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nbv_charges"),
            contents: bytemuck::cast_slice(setup.charges),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let exclusions_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nbv_exclusions"),
            contents: bytemuck::cast_slice(setup.exclusions),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let one_four_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nbv_one_four"),
            contents: bytemuck::cast_slice(setup.one_four_mask),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Neighbour-list buffers — initialised empty (zero-length counts /
        // starts) until the first `update_neighbours` call.
        let nbr_count_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nbv_nbr_count"),
            size: (n_atoms * std::mem::size_of::<u32>()).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let nbr_start_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nbv_nbr_start"),
            size: (n_atoms * std::mem::size_of::<u32>()).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cap = setup.initial_indices_capacity.max(64);
        let nbr_indices_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nbv_nbr_indices"),
            size: (cap * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let forces_size = positions_size;
        let forces_buf = forces_buf.unwrap_or_else(|| {
            Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("nbv_forces"),
                size: forces_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        });
        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nbv_readback"),
            size: forces_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nonbonded_verlet.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("nonbonded_verlet.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nbv_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("nonbonded_verlet"),
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
            &nbr_count_buf,
            &nbr_start_buf,
            &nbr_indices_buf,
            &forces_buf,
        );

        Self {
            n_atoms,
            pipeline,
            bind_group_layout,
            params_buf,
            positions_buf,
            atom_lj_buf,
            charges_buf,
            exclusions_buf,
            one_four_buf,
            nbr_count_buf,
            nbr_start_buf,
            nbr_indices_buf,
            nbr_indices_capacity: cap,
            forces_buf,
            readback_buf,
            forces_size,
            bind_group,
            pos_padded: vec![[0.0; 4]; n_atoms],
            ctx,
        }
    }

    /// Upload a fresh neighbour list.  Layout follows the standard CSR
    /// convention:
    ///
    /// - `counts[i]` is the number of neighbours that atom `i` has.
    /// - `starts[i]` is the offset into `indices` where atom `i`'s
    ///   neighbour-list begins.
    /// - `indices` is the flat concatenation of all per-atom lists.
    ///
    /// `counts.len()` and `starts.len()` must both equal the `n_atoms`
    /// passed to [`new`](Self::new).  The list must already include
    /// *both* directions of every pair — i.e. if `j` is in `i`'s list,
    /// `i` should also be in `j`'s list — because each thread sums
    /// forces only over its own neighbours.
    ///
    /// Excluded pairs (1-2 / 1-3) can stay in the list and will be
    /// skipped by the kernel via the exclusion bitmap.  Keeping them
    /// in lets the same list serve all force-eval calls between
    /// rebuilds without per-call filtering.
    pub fn update_neighbours(&mut self, counts: &[u32], starts: &[u32], indices: &[u32]) {
        assert_eq!(counts.len(), self.n_atoms);
        assert_eq!(starts.len(), self.n_atoms);
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        // Grow the indices buffer if needed.
        if indices.len() > self.nbr_indices_capacity {
            let new_cap = indices.len().next_power_of_two().max(64);
            self.nbr_indices_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("nbv_nbr_indices"),
                size: (new_cap * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.nbr_indices_capacity = new_cap;
            // Bind group must be rebuilt because we replaced a buffer.
            self.bind_group = create_bind_group(
                device,
                &self.bind_group_layout,
                &self.params_buf,
                &self.positions_buf,
                &self.atom_lj_buf,
                &self.charges_buf,
                &self.exclusions_buf,
                &self.one_four_buf,
                &self.nbr_count_buf,
                &self.nbr_start_buf,
                &self.nbr_indices_buf,
                &self.forces_buf,
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

    pub fn compute(&self) -> Vec<[f32; 3]> {
        let device = &self.ctx.device;
        let queue = &self.ctx.queue;
        // The kernel accumulates into `forces` (so it can compose with
        // bonded kernels into the same buffer in the integrator path).
        // Standalone callers expect each `compute()` to return that
        // step's forces, not a running sum, so we zero the buffer
        // first.
        let zeroes = vec![0u8; self.forces_size as usize];
        queue.write_buffer(&self.forces_buf, 0, &zeroes);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("nbv_encoder"),
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

    /// Zero the pipeline's persistent forces buffer.  Call before
    /// `record_compute` in the fused integrator path so accumulation
    /// starts from zero each step.  Cheap (one `write_buffer`, ~70 KB
    /// at ~5000 atoms).
    pub fn clear_forces(&self) {
        let zeroes = vec![0u8; self.forces_size as usize];
        self.ctx
            .queue
            .write_buffer(&self.forces_buf, 0, &zeroes);
    }

    /// Record the compute pass into a caller-owned encoder.  Used by
    /// the kernel-fused integrator path in `dynamics::GpuAccelerator`
    /// to batch multiple kernels into a single encoder submit.  Does
    /// NOT enqueue a readback — pair with
    /// [`record_readback_copy`](Self::record_readback_copy).
    pub fn record_compute(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("nbv_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        let wg_count = self.n_atoms.div_ceil(64) as u32;
        pass.dispatch_workgroups(wg_count, 1, 1);
    }

    /// Append a copy from the device-local forces buffer to the
    /// CPU-mappable readback buffer.  Required between
    /// [`record_compute`](Self::record_compute) and any subsequent
    /// readback.
    pub fn record_readback_copy(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(&self.forces_buf, 0, &self.readback_buf, 0, self.forces_size);
    }

    /// Start an asynchronous CPU map of the readback buffer.  The
    /// returned receiver lets the caller wait for the map to complete
    /// (typically via `device.poll(Maintain::Wait)` followed by
    /// `recv()`).  Pair with [`take_readback`](Self::take_readback)
    /// after the map completes.
    pub fn begin_readback(&self) -> std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>> {
        let slice = self.readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        rx
    }

    /// Consume the mapped readback buffer into a `Vec<[f32; 3]>`.
    /// Must be called after the receiver returned by
    /// [`begin_readback`](Self::begin_readback) has resolved.
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
fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: &wgpu::Buffer,
    positions: &wgpu::Buffer,
    atom_lj: &wgpu::Buffer,
    charges: &wgpu::Buffer,
    exclusions: &wgpu::Buffer,
    one_four: &wgpu::Buffer,
    nbr_count: &wgpu::Buffer,
    nbr_start: &wgpu::Buffer,
    nbr_indices: &wgpu::Buffer,
    forces: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("nbv_bind_group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: positions.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: atom_lj.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: charges.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: exclusions.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: one_four.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: nbr_count.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: nbr_start.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: nbr_indices.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: forces.as_entire_binding() },
        ],
    })
}

/// CPU-side helper: convert a symmetric pair list (each pair stored
/// once as `(min, max)`) into the CSR neighbour-list layout that
/// [`VerletNonbondedPipeline::update_neighbours`] expects.  Returns
/// `(counts, starts, indices)`.
///
/// Use this when the calling code keeps the pair list as a flat
/// `Vec<(u32, u32)>` (as `ForceScratch` does).  Cost is O(N + P).
pub fn pair_list_to_csr(n_atoms: usize, pairs: &[(u32, u32)]) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut counts = vec![0u32; n_atoms];
    for &(i, j) in pairs {
        counts[i as usize] += 1;
        counts[j as usize] += 1;
    }
    let mut starts = vec![0u32; n_atoms];
    let mut total = 0u32;
    for i in 0..n_atoms {
        starts[i] = total;
        total += counts[i];
    }
    let mut indices = vec![0u32; total as usize];
    // Reuse `counts` as a write cursor — restore at the end.
    let mut cursor = vec![0u32; n_atoms];
    for &(i, j) in pairs {
        let ii = i as usize;
        let jj = j as usize;
        indices[(starts[ii] + cursor[ii]) as usize] = j;
        cursor[ii] += 1;
        indices[(starts[jj] + cursor[jj]) as usize] = i;
        cursor[jj] += 1;
    }
    // Sorting each atom's neighbour slice was tried and reverted —
    // it improved cache locality in the kernel but the sort itself
    // (O(N · avg_count · log avg_count) at every Verlet rebuild)
    // cost more than the cache wins saved.  At 5840 atoms / 200
    // nbrs the sort alone added ~40 ms/rebuild.  See bench note in
    // commit PERF.gpu.18 if this gets revisited.
    (counts, starts, indices)
}
