//! Validate the GPU bonded kernels against the production CPU
//! `energy::forces_bonded` path.  Uses a built Ala-Lys-Glu tripeptide —
//! all four term types (bond / angle / dihedral / improper) are
//! present in non-trivial quantities.
//!
//! Each kernel is tested in isolation: run only the zero pass + that
//! one term kernel on GPU, run only that one term's CPU function,
//! compare per-atom forces.  If all four pass individually, the
//! `record_all` chain (zero → bond → angle → dihedral → improper)
//! produces the sum.

use chem::{classify_atom, standard_ff, AminoAcid, AtomType, ForceField};
use energy::units::{deg_to_rad, kcal_to_kj};
use geom::{build_extended_chain, build_topology_graph, Structure, TopologyGraph, Vec3};
use gpu::{
    AngleTerm, BondTerm, BondedPipeline, BondedSetup, DihedralTerm, GpuContext, ImproperTerm,
    PeriodicTerm,
};

// ---- Per-atom inverse-topology construction (mirrors what the
// integrator will do in `dynamics::gpu_accel`).

fn build_atom_types(s: &Structure) -> Vec<AtomType> {
    let mut out = Vec::with_capacity(s.atom_count());
    for r in &s.residues {
        for a in &r.atoms {
            out.push(classify_atom(r.monomer, a.name).unwrap());
        }
    }
    out
}

fn build_bonds(graph: &TopologyGraph, ff: &ForceField, atom_types: &[AtomType], n: usize)
    -> (Vec<BondTerm>, Vec<u32>, Vec<u32>, Vec<u32>)
{
    let mut terms: Vec<BondTerm> = Vec::new();
    let mut per_atom: Vec<Vec<u32>> = vec![Vec::new(); n];
    for b in &graph.bonds {
        let Some(p) = ff.bond(atom_types[b.a], atom_types[b.b]) else { continue };
        let idx = terms.len() as u32;
        terms.push(BondTerm {
            a: b.a as u32, b: b.b as u32,
            k_kj: kcal_to_kj(p.k) as f32,
            r0_a: p.r0 as f32,
        });
        per_atom[b.a].push(idx);
        per_atom[b.b].push(idx);
    }
    flatten_csr(per_atom, n)
        .map_terms(terms)
}

fn build_angles(graph: &TopologyGraph, ff: &ForceField, atom_types: &[AtomType], n: usize)
    -> (Vec<AngleTerm>, Vec<u32>, Vec<u32>, Vec<u32>)
{
    let mut terms: Vec<AngleTerm> = Vec::new();
    let mut per_atom: Vec<Vec<u32>> = vec![Vec::new(); n];
    for ang in &graph.angles {
        let Some(p) = ff.angle(atom_types[ang.a], atom_types[ang.b], atom_types[ang.c])
        else { continue };
        let idx = terms.len() as u32;
        terms.push(AngleTerm {
            a: ang.a as u32, b: ang.b as u32, c: ang.c as u32, _pad: 0,
            k_kj: kcal_to_kj(p.k) as f32,
            theta0_rad: deg_to_rad(p.theta0_deg) as f32,
            _pad2: 0.0, _pad3: 0.0,
        });
        per_atom[ang.a].push(idx);
        per_atom[ang.b].push(idx);
        per_atom[ang.c].push(idx);
    }
    flatten_csr(per_atom, n).map_terms(terms)
}

fn build_dihedrals(graph: &TopologyGraph, ff: &ForceField, atom_types: &[AtomType], n: usize)
    -> (Vec<DihedralTerm>, Vec<u32>, Vec<u32>, Vec<u32>)
{
    let mut terms: Vec<DihedralTerm> = Vec::new();
    let mut per_atom: Vec<Vec<u32>> = vec![Vec::new(); n];
    for d in &graph.dihedrals {
        let Some(pterms) = ff.dihedral(
            atom_types[d.a], atom_types[d.b], atom_types[d.c], atom_types[d.d],
        ) else { continue };
        let mut packed = DihedralTerm {
            a: d.a as u32, b: d.b as u32, c: d.c as u32, d: d.d as u32,
            n_terms: pterms.len().min(4) as u32,
            _pad0: 0, _pad1: 0, _pad2: 0,
            term0: PeriodicTerm { k_kj: 0.0, n: 0.0, delta_rad: 0.0, _pad: 0.0 },
            term1: PeriodicTerm { k_kj: 0.0, n: 0.0, delta_rad: 0.0, _pad: 0.0 },
            term2: PeriodicTerm { k_kj: 0.0, n: 0.0, delta_rad: 0.0, _pad: 0.0 },
            term3: PeriodicTerm { k_kj: 0.0, n: 0.0, delta_rad: 0.0, _pad: 0.0 },
        };
        for (i, t) in pterms.iter().take(4).enumerate() {
            let pt = PeriodicTerm {
                k_kj: kcal_to_kj(t.k) as f32,
                n: t.n as f32,
                delta_rad: deg_to_rad(t.delta_deg) as f32,
                _pad: 0.0,
            };
            match i {
                0 => packed.term0 = pt,
                1 => packed.term1 = pt,
                2 => packed.term2 = pt,
                _ => packed.term3 = pt,
            }
        }
        let idx = terms.len() as u32;
        terms.push(packed);
        per_atom[d.a].push(idx);
        per_atom[d.b].push(idx);
        per_atom[d.c].push(idx);
        per_atom[d.d].push(idx);
    }
    flatten_csr(per_atom, n).map_terms(terms)
}

fn build_impropers(graph: &TopologyGraph, ff: &ForceField, atom_types: &[AtomType], n: usize)
    -> (Vec<ImproperTerm>, Vec<u32>, Vec<u32>, Vec<u32>)
{
    let mut terms: Vec<ImproperTerm> = Vec::new();
    let mut per_atom: Vec<Vec<u32>> = vec![Vec::new(); n];
    for imp in &graph.impropers {
        let Some(p) = ff.improper(
            atom_types[imp.a], atom_types[imp.b], atom_types[imp.c], atom_types[imp.d],
        ) else { continue };
        let idx = terms.len() as u32;
        terms.push(ImproperTerm {
            a: imp.a as u32, b: imp.b as u32, c: imp.c as u32, d: imp.d as u32,
            k_kj: kcal_to_kj(p.k) as f32,
            omega0_rad: deg_to_rad(p.psi0_deg) as f32,
            _pad0: 0.0, _pad1: 0.0,
        });
        per_atom[imp.a].push(idx);
        per_atom[imp.b].push(idx);
        per_atom[imp.c].push(idx);
        per_atom[imp.d].push(idx);
    }
    flatten_csr(per_atom, n).map_terms(terms)
}

struct CsrParts { counts: Vec<u32>, starts: Vec<u32>, indices: Vec<u32> }
impl CsrParts {
    fn map_terms<T>(self, terms: Vec<T>) -> (Vec<T>, Vec<u32>, Vec<u32>, Vec<u32>) {
        (terms, self.counts, self.starts, self.indices)
    }
}

fn flatten_csr(per_atom: Vec<Vec<u32>>, n: usize) -> CsrParts {
    let counts: Vec<u32> = per_atom.iter().map(|v| v.len() as u32).collect();
    let mut starts = vec![0u32; n];
    let mut total = 0u32;
    for i in 0..n {
        starts[i] = total;
        total += counts[i];
    }
    let mut indices = vec![0u32; total as usize];
    for (i, list) in per_atom.iter().enumerate() {
        let base = starts[i] as usize;
        for (k, &v) in list.iter().enumerate() {
            indices[base + k] = v;
        }
    }
    CsrParts { counts, starts, indices }
}

fn setup_and_run(s: &Structure, g: &TopologyGraph, ff: &ForceField, ctx: &'static GpuContext)
    -> (BondedPipeline, wgpu_lite::PositionsForces)
{
    let n = s.atom_count();
    let atom_types = build_atom_types(s);
    let positions_f32: Vec<[f32; 4]> = s.residues.iter()
        .flat_map(|r| r.atoms.iter().map(|a| [a.position.x as f32, a.position.y as f32, a.position.z as f32, 0.0]))
        .collect();
    let device = &ctx.device;
    let queue = &ctx.queue;
    let positions_size = (n * std::mem::size_of::<[f32; 4]>()) as u64;
    let positions_buf = device.create_buffer(&gpu::wgpu::BufferDescriptor {
        label: Some("test_positions"),
        size: positions_size,
        usage: gpu::wgpu::BufferUsages::STORAGE | gpu::wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&positions_buf, 0, bytemuck::cast_slice(&positions_f32));
    let forces_buf = device.create_buffer(&gpu::wgpu::BufferDescriptor {
        label: Some("test_forces"),
        size: positions_size,
        usage: gpu::wgpu::BufferUsages::STORAGE | gpu::wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buf = device.create_buffer(&gpu::wgpu::BufferDescriptor {
        label: Some("test_readback"),
        size: positions_size,
        usage: gpu::wgpu::BufferUsages::MAP_READ | gpu::wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let (bond_terms, ab_count, ab_start, ab_index) = build_bonds(g, ff, &atom_types, n);
    let (angle_terms, aa_count, aa_start, aa_index) = build_angles(g, ff, &atom_types, n);
    let (dihedral_terms, ad_count, ad_start, ad_index) = build_dihedrals(g, ff, &atom_types, n);
    let (improper_terms, ai_count, ai_start, ai_index) = build_impropers(g, ff, &atom_types, n);

    let pipe = BondedPipeline::new(
        ctx, n, &positions_buf, &forces_buf,
        BondedSetup {
            bond_terms: &bond_terms,
            atom_bond_count: &ab_count,
            atom_bond_start: &ab_start,
            atom_bond_index: &ab_index,
            angle_terms: &angle_terms,
            atom_angle_count: &aa_count,
            atom_angle_start: &aa_start,
            atom_angle_index: &aa_index,
            dihedral_terms: &dihedral_terms,
            atom_dihedral_count: &ad_count,
            atom_dihedral_start: &ad_start,
            atom_dihedral_index: &ad_index,
            improper_terms: &improper_terms,
            atom_improper_count: &ai_count,
            atom_improper_start: &ai_start,
            atom_improper_index: &ai_index,
        },
    );
    (pipe, wgpu_lite::PositionsForces {
        positions_buf, forces_buf, readback_buf, size: positions_size, n,
    })
}

mod wgpu_lite {
    pub struct PositionsForces {
        pub positions_buf: gpu::wgpu::Buffer,
        pub forces_buf: gpu::wgpu::Buffer,
        pub readback_buf: gpu::wgpu::Buffer,
        pub size: u64,
        pub n: usize,
    }
    impl PositionsForces {
        pub fn read_forces(&self, ctx: &gpu::GpuContext) -> Vec<[f32; 3]> {
            let device = &ctx.device;
            let queue = &ctx.queue;
            let mut encoder = device.create_command_encoder(&gpu::wgpu::CommandEncoderDescriptor {
                label: Some("readback_encoder"),
            });
            encoder.copy_buffer_to_buffer(&self.forces_buf, 0, &self.readback_buf, 0, self.size);
            queue.submit(Some(encoder.finish()));
            let slice = self.readback_buf.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(gpu::wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            let _ = device.poll(gpu::wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();
            let data = slice.get_mapped_range();
            let padded: &[[f32; 4]] = bytemuck::cast_slice(&data);
            let out: Vec<[f32; 3]> = padded.iter().map(|v| [v[0], v[1], v[2]]).collect();
            drop(data);
            self.readback_buf.unmap();
            out
        }
    }
}

fn dispatch_zero_plus_one<F: Fn(&BondedPipeline, &mut gpu::wgpu::CommandEncoder)>(
    ctx: &gpu::GpuContext,
    pipe: &BondedPipeline,
    record_one: F,
) {
    let device = &ctx.device;
    let queue = &ctx.queue;
    let mut encoder = device.create_command_encoder(&gpu::wgpu::CommandEncoderDescriptor {
        label: Some("bonded_test_encoder"),
    });
    pipe.record_zero(&mut encoder);
    record_one(pipe, &mut encoder);
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(gpu::wgpu::Maintain::Wait);
}

fn compare_against_cpu(label: &str, gpu_f: &[[f32; 3]], cpu_f: &[Vec3]) -> f64 {
    assert_eq!(gpu_f.len(), cpu_f.len());
    let mut max_err = 0.0_f64;
    let mut argmax = String::new();
    for (i, (gf, cf)) in gpu_f.iter().zip(cpu_f.iter()).enumerate() {
        for axis in 0..3 {
            let gv = gf[axis] as f64;
            let cv = cf[axis];
            let err = (gv - cv).abs();
            if err > max_err {
                max_err = err;
                argmax = format!("atom {i} axis {axis}: cpu={cv:.6} gpu={gv:.6} err={err:.6}");
            }
        }
    }
    eprintln!("{label}: max GPU-vs-CPU discrepancy {:.3e} kJ/mol/Å — {argmax}", max_err);
    max_err
}

#[test]
fn gpu_bonded_kernels_match_cpu_on_ala_lys_glu() {
    let ctx = match GpuContext::get() {
        Ok(c) => c,
        Err(e) => { eprintln!("GPU unavailable: {e}"); return; }
    };
    let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Lys, AminoAcid::Glu]).unwrap();
    let g = build_topology_graph(&s);
    let ff = standard_ff();
    let n = s.atom_count();
    let atom_types = build_atom_types(&s);
    let positions: Vec<Vec3> = s.residues.iter()
        .flat_map(|r| r.atoms.iter().map(|a| a.position)).collect();

    let (pipe, bufs) = setup_and_run(&s, &g, ff, ctx);

    // ---- Bond ----
    dispatch_zero_plus_one(ctx, &pipe, |p, e| p.record_bond(e));
    let gpu_f = bufs.read_forces(ctx);
    let mut cpu_f = vec![Vec3::zeros(); n];
    energy::forces_bonded::add_bond_forces(&positions, &g, ff, &atom_types, &mut cpu_f);
    let err = compare_against_cpu("BOND", &gpu_f, &cpu_f);
    assert!(err < 1e-2, "bond force mismatch");

    // ---- Angle ----
    dispatch_zero_plus_one(ctx, &pipe, |p, e| p.record_angle(e));
    let gpu_f = bufs.read_forces(ctx);
    let mut cpu_f = vec![Vec3::zeros(); n];
    energy::forces_bonded::add_angle_forces(&positions, &g, ff, &atom_types, &mut cpu_f);
    let err = compare_against_cpu("ANGLE", &gpu_f, &cpu_f);
    assert!(err < 1e-2, "angle force mismatch");

    // ---- Dihedral ----
    dispatch_zero_plus_one(ctx, &pipe, |p, e| p.record_dihedral(e));
    let gpu_f = bufs.read_forces(ctx);
    let mut cpu_f = vec![Vec3::zeros(); n];
    energy::forces_bonded::add_dihedral_forces(&positions, &g, ff, &atom_types, &mut cpu_f);
    let err = compare_against_cpu("DIHEDRAL", &gpu_f, &cpu_f);
    assert!(err < 1.0, "dihedral force mismatch — tolerance loose because f32 trig accumulates");

    // ---- Improper ----
    dispatch_zero_plus_one(ctx, &pipe, |p, e| p.record_improper(e));
    let gpu_f = bufs.read_forces(ctx);
    let mut cpu_f = vec![Vec3::zeros(); n];
    energy::forces_bonded::add_improper_forces(&positions, &g, ff, &atom_types, &mut cpu_f);
    let err = compare_against_cpu("IMPROPER", &gpu_f, &cpu_f);
    assert!(err < 1.0, "improper force mismatch");

    // ---- All combined (split kernels) ----
    let device = &ctx.device;
    let queue = &ctx.queue;
    let mut encoder = device.create_command_encoder(&gpu::wgpu::CommandEncoderDescriptor {
        label: Some("bonded_all_encoder"),
    });
    pipe.record_all(&mut encoder);
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(gpu::wgpu::Maintain::Wait);
    let gpu_total = bufs.read_forces(ctx);
    let mut cpu_total = vec![Vec3::zeros(); n];
    energy::forces_bonded::add_bond_forces(&positions, &g, ff, &atom_types, &mut cpu_total);
    energy::forces_bonded::add_angle_forces(&positions, &g, ff, &atom_types, &mut cpu_total);
    energy::forces_bonded::add_dihedral_forces(&positions, &g, ff, &atom_types, &mut cpu_total);
    energy::forces_bonded::add_improper_forces(&positions, &g, ff, &atom_types, &mut cpu_total);
    let err = compare_against_cpu("ALL_BONDED (split)", &gpu_total, &cpu_total);
    assert!(err < 1.0, "combined bonded forces mismatch");

    // ---- Fused all_bonded_force kernel ----
    // One dispatch replaces four; should give bit-identical results
    // (same math, same accumulation order per atom — only the kernel
    // launch boundary changes).
    let mut encoder = device.create_command_encoder(&gpu::wgpu::CommandEncoderDescriptor {
        label: Some("bonded_fused_encoder"),
    });
    pipe.record_zero(&mut encoder);
    pipe.record_all_bonded(&mut encoder);
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(gpu::wgpu::Maintain::Wait);
    let gpu_fused = bufs.read_forces(ctx);
    let err = compare_against_cpu("ALL_BONDED (fused)", &gpu_fused, &cpu_total);
    assert!(err < 1.0, "fused bonded forces disagree with CPU");
    // And vs the split kernels — should be identical.
    let mut max_split_fused = 0.0_f32;
    for (a, b) in gpu_total.iter().zip(gpu_fused.iter()) {
        for axis in 0..3 {
            let d = (a[axis] - b[axis]).abs();
            if d > max_split_fused { max_split_fused = d; }
        }
    }
    eprintln!("max split-vs-fused disagreement: {:.3e}", max_split_fused);
    assert!(max_split_fused < 1e-3,
        "fused bonded force should match split-kernel result bit-for-bit");
}
