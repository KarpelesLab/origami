//! GPU bonded force pipeline.
//!
//! Compiles the four bonded-term kernels (bond, angle, dihedral,
//! improper) and a `zero_forces` kernel that the caller invokes at
//! the start of each force evaluation.  All five kernels share one
//! bind group — same forces buffer, same per-atom inverse-topology
//! buffers.
//!
//! Caller responsibility: at construction time, hand over fully-
//! resolved bonded terms (force-field params already looked up) and
//! the per-atom CSR participation lists.  See `BondedSetup`.
//!
//! Per-step usage:
//!
//!   1. `record_zero`        — wipe the shared forces buffer
//!   2. `record_bond`        — add bond contributions
//!   3. `record_angle`       — add angle contributions
//!   4. `record_dihedral`    — add dihedral contributions
//!   5. `record_improper`    — add improper contributions
//!
//! All into the same encoder.  Pair forces (nonbonded + GB) then
//! accumulate into the same buffer.

use wgpu::util::DeviceExt;

use crate::context::GpuContext;

// ---- Term structs sent verbatim to the GPU ----

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BondTerm {
    pub a: u32,
    pub b: u32,
    pub k_kj: f32,
    pub r0_a: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AngleTerm {
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub _pad: u32,
    pub k_kj: f32,
    pub theta0_rad: f32,
    pub _pad2: f32,
    pub _pad3: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PeriodicTerm {
    pub k_kj: f32,
    pub n: f32,
    pub delta_rad: f32,
    pub _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DihedralTerm {
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub d: u32,
    pub n_terms: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub term0: PeriodicTerm,
    pub term1: PeriodicTerm,
    pub term2: PeriodicTerm,
    pub term3: PeriodicTerm,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ImproperTerm {
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub d: u32,
    pub k_kj: f32,
    pub omega0_rad: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

pub struct BondedSetup<'a> {
    pub bond_terms: &'a [BondTerm],
    pub atom_bond_count: &'a [u32],
    pub atom_bond_start: &'a [u32],
    pub atom_bond_index: &'a [u32],
    pub angle_terms: &'a [AngleTerm],
    pub atom_angle_count: &'a [u32],
    pub atom_angle_start: &'a [u32],
    pub atom_angle_index: &'a [u32],
    pub dihedral_terms: &'a [DihedralTerm],
    pub atom_dihedral_count: &'a [u32],
    pub atom_dihedral_start: &'a [u32],
    pub atom_dihedral_index: &'a [u32],
    pub improper_terms: &'a [ImproperTerm],
    pub atom_improper_count: &'a [u32],
    pub atom_improper_start: &'a [u32],
    pub atom_improper_index: &'a [u32],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GlobalParams {
    n_atoms: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub struct BondedPipeline {
    n_atoms: usize,
    zero_pipeline: wgpu::ComputePipeline,
    bond_pipeline: wgpu::ComputePipeline,
    angle_pipeline: wgpu::ComputePipeline,
    dihedral_pipeline: wgpu::ComputePipeline,
    improper_pipeline: wgpu::ComputePipeline,
    /// Fused bond+angle+dihedral+improper kernel — one dispatch
    /// replaces all four when the caller doesn't need to split.
    all_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    /// We keep references to the shader buffers in the struct so they
    /// outlive the bind group.  Most are read-only and never touched
    /// after construction; the forces buffer is shared with other
    /// pipelines (caller-supplied at construction).
    #[allow(dead_code)]
    keep_alive: KeepAlive,
}

// Bundles all the GPU buffers we need to keep alive for the lifetime
// of the bind group.  Marked dead-code to silence warnings — they exist
// for resource lifetime only.
#[allow(dead_code)]
struct KeepAlive {
    globals_buf: wgpu::Buffer,
    bond_terms_buf: wgpu::Buffer,
    atom_bond_count_buf: wgpu::Buffer,
    atom_bond_start_buf: wgpu::Buffer,
    atom_bond_index_buf: wgpu::Buffer,
    angle_terms_buf: wgpu::Buffer,
    atom_angle_count_buf: wgpu::Buffer,
    atom_angle_start_buf: wgpu::Buffer,
    atom_angle_index_buf: wgpu::Buffer,
    dihedral_terms_buf: wgpu::Buffer,
    atom_dihedral_count_buf: wgpu::Buffer,
    atom_dihedral_start_buf: wgpu::Buffer,
    atom_dihedral_index_buf: wgpu::Buffer,
    improper_terms_buf: wgpu::Buffer,
    atom_improper_count_buf: wgpu::Buffer,
    atom_improper_start_buf: wgpu::Buffer,
    atom_improper_index_buf: wgpu::Buffer,
}

impl BondedPipeline {
    /// Build the bonded pipeline.  `positions_buf` and `forces_buf` are
    /// shared with the rest of the integrator (e.g. nonbonded /
    /// BAOAB).  Read-only for positions; read-write for forces.
    pub fn new(
        ctx: &'static GpuContext,
        n_atoms: usize,
        positions_buf: &wgpu::Buffer,
        forces_buf: &wgpu::Buffer,
        setup: BondedSetup,
    ) -> Self {
        let device = &ctx.device;
        let globals = GlobalParams {
            n_atoms: n_atoms as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let globals_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bonded_globals"),
            contents: bytemuck::bytes_of(&globals),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let make_storage = |label: &'static str, bytes: &[u8]| -> wgpu::Buffer {
            // Even empty arrays need a minimum 4-byte buffer; pad up.
            if bytes.is_empty() {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: 4,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            } else {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                })
            }
        };

        let bond_terms_buf = make_storage("bond_terms", bytemuck::cast_slice(setup.bond_terms));
        let atom_bond_count_buf = make_storage(
            "atom_bond_count",
            bytemuck::cast_slice(setup.atom_bond_count),
        );
        let atom_bond_start_buf = make_storage(
            "atom_bond_start",
            bytemuck::cast_slice(setup.atom_bond_start),
        );
        let atom_bond_index_buf = make_storage(
            "atom_bond_index",
            bytemuck::cast_slice(setup.atom_bond_index),
        );

        let angle_terms_buf = make_storage("angle_terms", bytemuck::cast_slice(setup.angle_terms));
        let atom_angle_count_buf = make_storage(
            "atom_angle_count",
            bytemuck::cast_slice(setup.atom_angle_count),
        );
        let atom_angle_start_buf = make_storage(
            "atom_angle_start",
            bytemuck::cast_slice(setup.atom_angle_start),
        );
        let atom_angle_index_buf = make_storage(
            "atom_angle_index",
            bytemuck::cast_slice(setup.atom_angle_index),
        );

        let dihedral_terms_buf =
            make_storage("dihedral_terms", bytemuck::cast_slice(setup.dihedral_terms));
        let atom_dihedral_count_buf = make_storage(
            "atom_dihedral_count",
            bytemuck::cast_slice(setup.atom_dihedral_count),
        );
        let atom_dihedral_start_buf = make_storage(
            "atom_dihedral_start",
            bytemuck::cast_slice(setup.atom_dihedral_start),
        );
        let atom_dihedral_index_buf = make_storage(
            "atom_dihedral_index",
            bytemuck::cast_slice(setup.atom_dihedral_index),
        );

        let improper_terms_buf =
            make_storage("improper_terms", bytemuck::cast_slice(setup.improper_terms));
        let atom_improper_count_buf = make_storage(
            "atom_improper_count",
            bytemuck::cast_slice(setup.atom_improper_count),
        );
        let atom_improper_start_buf = make_storage(
            "atom_improper_start",
            bytemuck::cast_slice(setup.atom_improper_start),
        );
        let atom_improper_index_buf = make_storage(
            "atom_improper_index",
            bytemuck::cast_slice(setup.atom_improper_index),
        );

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bonded.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("bonded.wgsl").into()),
        });
        // Each WGSL entry point only references its own subset of the
        // 19 bindings, so the implicit pipeline-layouts each see a
        // different (and shrunken) bind-group layout — wgpu then
        // refuses to create a 19-entry bind group against any of
        // them.  Build an explicit shared layout listing all 19
        // bindings, pass it to every pipeline.
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
        let mk_entry = |binding: u32, ty: wgpu::BindingType| -> wgpu::BindGroupLayoutEntry {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty,
                count: None,
            }
        };
        let entries = [
            mk_entry(0, uniform),
            mk_entry(1, storage_ro), // positions
            mk_entry(2, storage_rw), // forces
            mk_entry(3, storage_ro), // bond_terms
            mk_entry(4, storage_ro), // atom_bond_count
            mk_entry(5, storage_ro),
            mk_entry(6, storage_ro),
            mk_entry(7, storage_ro), // angle_terms
            mk_entry(8, storage_ro),
            mk_entry(9, storage_ro),
            mk_entry(10, storage_ro),
            mk_entry(11, storage_ro), // dihedral_terms
            mk_entry(12, storage_ro),
            mk_entry(13, storage_ro),
            mk_entry(14, storage_ro),
            mk_entry(15, storage_ro), // improper_terms
            mk_entry(16, storage_ro),
            mk_entry(17, storage_ro),
            mk_entry(18, storage_ro),
        ];
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bonded_bgl"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bonded_pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let make_pipeline = |label: &'static str, entry: &str| -> wgpu::ComputePipeline {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let zero_pipeline = make_pipeline("bonded_zero_pipeline", "zero_forces");
        let bond_pipeline = make_pipeline("bonded_bond_pipeline", "bond_force");
        let angle_pipeline = make_pipeline("bonded_angle_pipeline", "angle_force");
        let dihedral_pipeline = make_pipeline("bonded_dihedral_pipeline", "dihedral_force");
        let improper_pipeline = make_pipeline("bonded_improper_pipeline", "improper_force");
        let all_pipeline = make_pipeline("bonded_all_pipeline", "all_bonded_force");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bonded_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: globals_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: positions_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: forces_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bond_terms_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: atom_bond_count_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: atom_bond_start_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: atom_bond_index_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: angle_terms_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: atom_angle_count_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: atom_angle_start_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: atom_angle_index_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: dihedral_terms_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: atom_dihedral_count_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: atom_dihedral_start_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: atom_dihedral_index_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: improper_terms_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: atom_improper_count_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: atom_improper_start_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: atom_improper_index_buf.as_entire_binding(),
                },
            ],
        });

        Self {
            n_atoms,
            zero_pipeline,
            bond_pipeline,
            angle_pipeline,
            dihedral_pipeline,
            improper_pipeline,
            all_pipeline,
            bind_group,
            keep_alive: KeepAlive {
                globals_buf,
                bond_terms_buf,
                atom_bond_count_buf,
                atom_bond_start_buf,
                atom_bond_index_buf,
                angle_terms_buf,
                atom_angle_count_buf,
                atom_angle_start_buf,
                atom_angle_index_buf,
                dihedral_terms_buf,
                atom_dihedral_count_buf,
                atom_dihedral_start_buf,
                atom_dihedral_index_buf,
                improper_terms_buf,
                atom_improper_count_buf,
                atom_improper_start_buf,
                atom_improper_index_buf,
            },
        }
    }

    fn record_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        label: &str,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(self.n_atoms.div_ceil(64) as u32, 1, 1);
    }

    pub fn record_zero(&self, encoder: &mut wgpu::CommandEncoder) {
        self.record_pass(encoder, &self.zero_pipeline, "bonded_zero_pass");
    }

    pub fn record_bond(&self, encoder: &mut wgpu::CommandEncoder) {
        self.record_pass(encoder, &self.bond_pipeline, "bonded_bond_pass");
    }

    pub fn record_angle(&self, encoder: &mut wgpu::CommandEncoder) {
        self.record_pass(encoder, &self.angle_pipeline, "bonded_angle_pass");
    }

    pub fn record_dihedral(&self, encoder: &mut wgpu::CommandEncoder) {
        self.record_pass(encoder, &self.dihedral_pipeline, "bonded_dihedral_pass");
    }

    pub fn record_improper(&self, encoder: &mut wgpu::CommandEncoder) {
        self.record_pass(encoder, &self.improper_pipeline, "bonded_improper_pass");
    }

    /// Record the fused bond+angle+dihedral+improper pass.  One
    /// dispatch instead of four; same numerical result.  Use this in
    /// performance-sensitive paths (the integrator).  The split
    /// `record_bond` / `record_angle` / ... methods remain for
    /// validation tests that need to compare per-term forces against
    /// CPU references.
    pub fn record_all_bonded(&self, encoder: &mut wgpu::CommandEncoder) {
        self.record_pass(encoder, &self.all_pipeline, "bonded_all_pass");
    }

    /// Record all bonded passes split term-by-term (zero → bond →
    /// angle → dihedral → improper).  Used by the validation test
    /// path that asserts each term matches CPU individually.  For
    /// the integrator hot path, prefer
    /// [`record_zero`](Self::record_zero) + [`record_all_bonded`](Self::record_all_bonded).
    pub fn record_all(&self, encoder: &mut wgpu::CommandEncoder) {
        self.record_zero(encoder);
        self.record_bond(encoder);
        self.record_angle(encoder);
        self.record_dihedral(encoder);
        self.record_improper(encoder);
    }
}
