//! Validate the GPU SHAKE kernel against the production CPU
//! `dynamics::shake::shake_iterate`.
//!
//! Builds Ala-Lys-Glu, extracts the X-H bond list as constraints,
//! perturbs every H position by a small offset along its bond, then
//! runs SHAKE on both CPU and GPU starting from identical inputs.
//! Asserts the final positions agree to f32 precision.

use std::sync::Arc;

use chem::{classify_atom, standard_ff, AminoAcid, AtomType, Element};
use dynamics::shake::{build_h_bond_constraints, shake_iterate, Constraint as CpuConstraint};
use geom::{build_extended_chain, build_topology_graph, Vec3};
use gpu::{build_per_x_shake_data, GpuContext, ShakeConstraint, ShakePipeline};

fn atom_types_for(s: &geom::Structure) -> Vec<AtomType> {
    let mut out = Vec::with_capacity(s.atom_count());
    for r in &s.residues {
        for a in &r.atoms {
            out.push(classify_atom(r.monomer, a.name).unwrap());
        }
    }
    out
}

/// Promote CPU constraints to (X, H) form: X is whichever endpoint
/// is the heavy atom, H is the hydrogen.  The GPU per-X CSR layout
/// requires this orientation.
fn cpu_to_gpu_constraints(cpu: &[CpuConstraint], s: &geom::Structure) -> Vec<ShakeConstraint> {
    let atoms_flat: Vec<Element> = s
        .residues
        .iter()
        .flat_map(|r| r.atoms.iter().map(|a| a.element))
        .collect();
    cpu.iter()
        .map(|c| {
            let (x, h) = if atoms_flat[c.i] == Element::H {
                (c.j as u32, c.i as u32)
            } else {
                (c.i as u32, c.j as u32)
            };
            ShakeConstraint {
                x_atom: x,
                h_atom: h,
                d_sq: c.d_sq as f32,
            }
        })
        .collect()
}

#[test]
fn gpu_shake_matches_cpu_on_ala_lys_glu() {
    let ctx = match GpuContext::get() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("GPU unavailable: {e}");
            return;
        }
    };
    let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Lys, AminoAcid::Glu]).unwrap();
    let g = build_topology_graph(&s);
    let ff = standard_ff();
    let n = s.atom_count();
    let atom_types = atom_types_for(&s);

    // Constraints — same builder both paths use.
    let cpu_constraints = build_h_bond_constraints(&s, &g, ff, &atom_types);
    eprintln!(
        "Ala-Lys-Glu: {} atoms, {} X-H constraints",
        n,
        cpu_constraints.len()
    );

    let masses_f32: Vec<f32> = s
        .residues
        .iter()
        .flat_map(|r| r.atoms.iter().map(|a| a.element.mass_da() as f32))
        .collect();
    let masses_f64: Vec<f64> = masses_f32.iter().map(|&m| m as f64).collect();
    let inv_masses_f64: Vec<f64> = masses_f64.iter().map(|m| 1.0 / m).collect();

    // Reference positions: the unperturbed crystal-builder geometry.
    let ref_positions_f64: Vec<Vec3> = s
        .residues
        .iter()
        .flat_map(|r| r.atoms.iter().map(|a| a.position))
        .collect();

    // Perturbed positions: shove every H atom 0.1 Å along its X-H
    // bond vector.  This breaks constraints in a way SHAKE has to
    // fix.  Heavy atoms stay put.
    let mut perturbed_f64 = ref_positions_f64.clone();
    let atoms_flat: Vec<Element> = s
        .residues
        .iter()
        .flat_map(|r| r.atoms.iter().map(|a| a.element))
        .collect();
    for c in &cpu_constraints {
        let (x, h) = if atoms_flat[c.i] == Element::H {
            (c.j, c.i)
        } else {
            (c.i, c.j)
        };
        let bond = perturbed_f64[h] - perturbed_f64[x];
        let bond_len = bond.norm();
        if bond_len > 1e-9 {
            let unit = bond / bond_len;
            // Shove H 0.1 Å farther from X.
            perturbed_f64[h] += unit * 0.1;
        }
    }

    // ---- CPU SHAKE ----
    let mut cpu_pos = perturbed_f64.clone();
    let cpu_iters = shake_iterate(
        &mut cpu_pos,
        &ref_positions_f64,
        &inv_masses_f64,
        &cpu_constraints,
        1e-6_f64,
        64,
    )
    .expect("CPU SHAKE converged");
    eprintln!("CPU SHAKE converged in {} iterations", cpu_iters);

    // ---- GPU SHAKE ----
    let gpu_constraints = cpu_to_gpu_constraints(&cpu_constraints, &s);
    let data = build_per_x_shake_data(n, &gpu_constraints, &masses_f32);

    let device = &ctx.device;
    let queue = &ctx.queue;
    let positions_size = (n * std::mem::size_of::<[f32; 4]>()) as u64;
    let positions_buf = Arc::new(device.create_buffer(&gpu::wgpu::BufferDescriptor {
        label: Some("test_shake_positions"),
        size: positions_size,
        usage: gpu::wgpu::BufferUsages::STORAGE
            | gpu::wgpu::BufferUsages::COPY_DST
            | gpu::wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    }));
    let ref_positions_buf = Arc::new(device.create_buffer(&gpu::wgpu::BufferDescriptor {
        label: Some("test_shake_ref_positions"),
        size: positions_size,
        usage: gpu::wgpu::BufferUsages::STORAGE | gpu::wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    let readback_buf = device.create_buffer(&gpu::wgpu::BufferDescriptor {
        label: Some("test_shake_readback"),
        size: positions_size,
        usage: gpu::wgpu::BufferUsages::MAP_READ | gpu::wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let perturbed_f32: Vec<[f32; 4]> = perturbed_f64
        .iter()
        .map(|p| [p.x as f32, p.y as f32, p.z as f32, 0.0])
        .collect();
    let ref_f32: Vec<[f32; 4]> = ref_positions_f64
        .iter()
        .map(|p| [p.x as f32, p.y as f32, p.z as f32, 0.0])
        .collect();
    queue.write_buffer(&positions_buf, 0, bytemuck::cast_slice(&perturbed_f32));
    queue.write_buffer(&ref_positions_buf, 0, bytemuck::cast_slice(&ref_f32));

    let pipe = ShakePipeline::new(
        ctx,
        n,
        positions_buf.clone(),
        ref_positions_buf.clone(),
        &data,
        64,
        1e-6_f32,
    );

    let mut encoder = device.create_command_encoder(&gpu::wgpu::CommandEncoderDescriptor {
        label: Some("test_shake_encoder"),
    });
    pipe.record(&mut encoder);
    encoder.copy_buffer_to_buffer(&positions_buf, 0, &readback_buf, 0, positions_size);
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(gpu::wgpu::Maintain::Wait);

    let slice = readback_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(gpu::wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(gpu::wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let data_ptr = slice.get_mapped_range();
    let padded: &[[f32; 4]] = bytemuck::cast_slice(&data_ptr);
    let gpu_pos: Vec<Vec3> = padded
        .iter()
        .map(|p| Vec3::new(p[0] as f64, p[1] as f64, p[2] as f64))
        .collect();
    drop(data_ptr);
    readback_buf.unmap();

    // Compare GPU vs CPU final positions.
    let mut max_err = 0.0_f64;
    let mut argmax = String::new();
    for i in 0..n {
        for axis in 0..3 {
            let d = (cpu_pos[i][axis] - gpu_pos[i][axis]).abs();
            if d > max_err {
                max_err = d;
                argmax = format!(
                    "atom {i} axis {axis}: cpu={:.6} gpu={:.6} err={:.6}",
                    cpu_pos[i][axis], gpu_pos[i][axis], d
                );
            }
        }
    }
    eprintln!(
        "max GPU-vs-CPU SHAKE position discrepancy: {:.3e} Å — {argmax}",
        max_err
    );
    // f32 round-trip + GPU running iter to convergence in parallel vs
    // CPU serial Gauss-Seidel: expect ~1e-3 Å agreement.  The H-X bond
    // length tolerance (sqrt(1e-6) ≈ 1e-3 Å) bounds the result.
    assert!(
        max_err < 5e-3,
        "GPU SHAKE diverged from CPU result past 5e-3 Å — {argmax}"
    );

    // Independent check: every constraint should be satisfied to
    // within tol on the GPU output.
    let mut max_constraint_err = 0.0_f64;
    for c in &cpu_constraints {
        let r2 = (gpu_pos[c.i] - gpu_pos[c.j]).norm_squared();
        let err = (r2 - c.d_sq).abs();
        if err > max_constraint_err {
            max_constraint_err = err;
        }
    }
    eprintln!("max GPU constraint error: {:.3e} Å²", max_constraint_err);
    assert!(
        max_constraint_err < 1e-3,
        "GPU SHAKE left a constraint unsatisfied: {max_constraint_err}"
    );
}
