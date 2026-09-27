//! Direct comparison: tile-based nonbonded kernel vs the existing
//! Verlet kernel.  Both should produce identical forces (modulo
//! f32 reorder noise) since they implement the same physics.
//!
//! Builds Ala-Lys-Glu, applies Morton sort, builds both:
//!   - Verlet pair list (existing infrastructure)
//!   - Tile interaction list (new)
//! Runs both kernels on the same input, compares per-atom forces.

use chem::{classify_atom, standard_ff, AminoAcid, AtomType};
use energy::DEFAULT_CUTOFF_A;
use geom::{build_extended_chain, build_topology_graph, Vec3};
use gpu::{
    build_tile_interaction_list, morton_permutation, pair_list_to_csr, GpuContext,
    TileNonbondedPipeline, TileNonbondedSetup, VerletNonbondedPipeline, VerletNonbondedSetup,
};

const KCAL_TO_KJ: f32 = 4.184;

/// Helper: run both kernels against the same system and assert agreement.
/// Returns (max_err, n_atoms, n_tiles, n_tile_interactions).
fn run_pair_test(seq: &[AminoAcid], ctx: &'static GpuContext) -> (f64, usize, usize, usize) {
    let s = build_extended_chain(seq).unwrap();
    let g = build_topology_graph(&s);
    let ff = standard_ff();
    let n = s.atom_count();
    let atom_types: Vec<AtomType> = s
        .residues
        .iter()
        .flat_map(|r| {
            r.atoms
                .iter()
                .map(|a| classify_atom(r.monomer, a.name).unwrap())
        })
        .collect();
    let positions_cpu: Vec<[f32; 3]> = s
        .residues
        .iter()
        .flat_map(|r| {
            r.atoms.iter().map(|a| {
                [
                    a.position.x as f32,
                    a.position.y as f32,
                    a.position.z as f32,
                ]
            })
        })
        .collect();
    let charges_cpu: Vec<f32> = s
        .residues
        .iter()
        .flat_map(|r| {
            r.atoms
                .iter()
                .map(|a| ff.partial_charge_for(r.monomer, a.name).unwrap_or(0.0) as f32)
        })
        .collect();
    let lj_cpu: Vec<[f32; 4]> = atom_types
        .iter()
        .map(|t| {
            let p = ff.nonbonded(*t).unwrap();
            let eps_14 = p.epsilon_14.unwrap_or(p.epsilon);
            let rmin_half_14 = p.rmin_half_14.unwrap_or(p.rmin_half);
            [
                (p.epsilon as f32) * KCAL_TO_KJ,
                p.rmin_half as f32,
                (eps_14 as f32) * KCAL_TO_KJ,
                rmin_half_14 as f32,
            ]
        })
        .collect();
    let (gpu_to_cpu, cpu_to_gpu) = morton_permutation(&positions_cpu);
    let positions: Vec<[f32; 3]> = (0..n)
        .map(|gi| positions_cpu[gpu_to_cpu[gi] as usize])
        .collect();
    let charges: Vec<f32> = (0..n)
        .map(|gi| charges_cpu[gpu_to_cpu[gi] as usize])
        .collect();
    let atom_lj_data: Vec<[f32; 4]> = (0..n).map(|gi| lj_cpu[gpu_to_cpu[gi] as usize]).collect();

    let n_words = (n * n).div_ceil(32);
    let mut exclusions = vec![0u32; n_words];
    let mut one_four = vec![0u32; n_words];
    let set_bit = |buf: &mut [u32], a: usize, b: usize| {
        let bit = a * n + b;
        buf[bit / 32] |= 1u32 << (bit % 32);
        let bit = b * n + a;
        buf[bit / 32] |= 1u32 << (bit % 32);
    };
    for b in &g.bonds {
        set_bit(
            &mut exclusions,
            cpu_to_gpu[b.a] as usize,
            cpu_to_gpu[b.b] as usize,
        );
    }
    for a in &g.angles {
        set_bit(
            &mut exclusions,
            cpu_to_gpu[a.a] as usize,
            cpu_to_gpu[a.c] as usize,
        );
    }
    for d in &g.dihedrals {
        set_bit(
            &mut one_four,
            cpu_to_gpu[d.a] as usize,
            cpu_to_gpu[d.d] as usize,
        );
    }

    let cutoff = DEFAULT_CUTOFF_A as f32;
    let cutoff_sq = (cutoff as f64) * (cutoff as f64);
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = positions[i][0] as f64 - positions[j][0] as f64;
            let dy = positions[i][1] as f64 - positions[j][1] as f64;
            let dz = positions[i][2] as f64 - positions[j][2] as f64;
            if dx * dx + dy * dy + dz * dz <= cutoff_sq {
                pairs.push((i as u32, j as u32));
            }
        }
    }
    let (counts, starts, indices) = pair_list_to_csr(n, &pairs);

    let mut verlet = VerletNonbondedPipeline::new(
        ctx,
        n,
        VerletNonbondedSetup {
            atom_lj_data: &atom_lj_data,
            charges: &charges,
            exclusions: &exclusions,
            one_four_mask: &one_four,
            cutoff_a: cutoff,
            initial_indices_capacity: indices.len().max(64),
        },
    );
    verlet.update_neighbours(&counts, &starts, &indices);
    verlet.update_positions(&positions);
    let verlet_forces = verlet.compute();

    let tile_list = build_tile_interaction_list(&positions, cutoff);
    let mut tile = TileNonbondedPipeline::new(
        ctx,
        n,
        TileNonbondedSetup {
            atom_lj_data: &atom_lj_data,
            charges: &charges,
            exclusions: &exclusions,
            one_four_mask: &one_four,
            cutoff_a: cutoff,
            initial_tile_indices_capacity: tile_list.tile_indices.len().max(64),
        },
    );
    tile.update_tile_list(
        &tile_list.tile_count,
        &tile_list.tile_start,
        &tile_list.tile_indices,
    );
    tile.update_positions(&positions);
    let tile_forces = tile.compute();

    let mut max_err = 0.0_f64;
    for i in 0..n {
        for axis in 0..3 {
            let vv = verlet_forces[i][axis] as f64;
            let tv = tile_forces[i][axis] as f64;
            let err = (vv - tv).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }
    (max_err, n, tile_list.n_tiles, tile_list.tile_indices.len())
}

#[test]
fn tile_kernel_matches_verlet_kernel_on_ala_lys_glu() {
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
    let atom_types: Vec<AtomType> = s
        .residues
        .iter()
        .flat_map(|r| {
            r.atoms
                .iter()
                .map(|a| classify_atom(r.monomer, a.name).unwrap())
        })
        .collect();

    // CPU-ordered per-atom data.
    let positions_cpu: Vec<[f32; 3]> = s
        .residues
        .iter()
        .flat_map(|r| {
            r.atoms.iter().map(|a| {
                [
                    a.position.x as f32,
                    a.position.y as f32,
                    a.position.z as f32,
                ]
            })
        })
        .collect();
    let charges_cpu: Vec<f32> = s
        .residues
        .iter()
        .flat_map(|r| {
            r.atoms
                .iter()
                .map(|a| ff.partial_charge_for(r.monomer, a.name).unwrap_or(0.0) as f32)
        })
        .collect();
    let lj_cpu: Vec<[f32; 4]> = atom_types
        .iter()
        .map(|t| {
            let p = ff.nonbonded(*t).unwrap();
            let eps_14 = p.epsilon_14.unwrap_or(p.epsilon);
            let rmin_half_14 = p.rmin_half_14.unwrap_or(p.rmin_half);
            [
                (p.epsilon as f32) * KCAL_TO_KJ,
                p.rmin_half as f32,
                (eps_14 as f32) * KCAL_TO_KJ,
                rmin_half_14 as f32,
            ]
        })
        .collect();

    // Morton sort.
    let (gpu_to_cpu, cpu_to_gpu) = morton_permutation(&positions_cpu);
    let permute_vec3 = |src: &[[f32; 3]]| -> Vec<[f32; 3]> {
        (0..n).map(|g| src[gpu_to_cpu[g] as usize]).collect()
    };
    let permute_vec4 = |src: &[[f32; 4]]| -> Vec<[f32; 4]> {
        (0..n).map(|g| src[gpu_to_cpu[g] as usize]).collect()
    };
    let permute_f32 =
        |src: &[f32]| -> Vec<f32> { (0..n).map(|g| src[gpu_to_cpu[g] as usize]).collect() };
    let positions = permute_vec3(&positions_cpu);
    let charges = permute_f32(&charges_cpu);
    let atom_lj_data = permute_vec4(&lj_cpu);

    // GPU-indexed exclusion + 1-4 bitmaps (sparse build).
    let n_words = (n * n).div_ceil(32);
    let mut exclusions = vec![0u32; n_words];
    let mut one_four = vec![0u32; n_words];
    let set_bit = |buf: &mut [u32], a: usize, b: usize| {
        let bit = a * n + b;
        buf[bit / 32] |= 1u32 << (bit % 32);
        let bit = b * n + a;
        buf[bit / 32] |= 1u32 << (bit % 32);
    };
    for b in &g.bonds {
        set_bit(
            &mut exclusions,
            cpu_to_gpu[b.a] as usize,
            cpu_to_gpu[b.b] as usize,
        );
    }
    for a in &g.angles {
        set_bit(
            &mut exclusions,
            cpu_to_gpu[a.a] as usize,
            cpu_to_gpu[a.c] as usize,
        );
    }
    for d in &g.dihedrals {
        set_bit(
            &mut one_four,
            cpu_to_gpu[d.a] as usize,
            cpu_to_gpu[d.d] as usize,
        );
    }

    let cutoff = DEFAULT_CUTOFF_A as f32;
    let cutoff_sq = (cutoff as f64) * (cutoff as f64);

    // ---- Verlet baseline ----
    // Build brute-force pair list at the same cutoff.  Positions are
    // already in GPU order so the resulting pair list is GPU-indexed.
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = positions[i][0] as f64 - positions[j][0] as f64;
            let dy = positions[i][1] as f64 - positions[j][1] as f64;
            let dz = positions[i][2] as f64 - positions[j][2] as f64;
            if dx * dx + dy * dy + dz * dz <= cutoff_sq {
                pairs.push((i as u32, j as u32));
            }
        }
    }
    let (counts, starts, indices) = pair_list_to_csr(n, &pairs);
    let mut verlet = VerletNonbondedPipeline::new(
        ctx,
        n,
        VerletNonbondedSetup {
            atom_lj_data: &atom_lj_data,
            charges: &charges,
            exclusions: &exclusions,
            one_four_mask: &one_four,
            cutoff_a: cutoff,
            initial_indices_capacity: indices.len().max(64),
        },
    );
    verlet.update_neighbours(&counts, &starts, &indices);
    verlet.update_positions(&positions);
    let verlet_forces = verlet.compute();

    // ---- Tile kernel ----
    let tile_list = build_tile_interaction_list(&positions, cutoff);
    eprintln!(
        "tile list: {} tiles, {} total tile interactions",
        tile_list.n_tiles,
        tile_list.tile_indices.len()
    );
    let mut tile = TileNonbondedPipeline::new(
        ctx,
        n,
        TileNonbondedSetup {
            atom_lj_data: &atom_lj_data,
            charges: &charges,
            exclusions: &exclusions,
            one_four_mask: &one_four,
            cutoff_a: cutoff,
            initial_tile_indices_capacity: tile_list.tile_indices.len().max(64),
        },
    );
    tile.update_tile_list(
        &tile_list.tile_count,
        &tile_list.tile_start,
        &tile_list.tile_indices,
    );
    tile.update_positions(&positions);
    let tile_forces = tile.compute();

    // Compare per-atom forces.
    assert_eq!(verlet_forces.len(), tile_forces.len());
    let mut max_err = 0.0_f64;
    let mut label = String::new();
    for i in 0..n {
        for axis in 0..3 {
            let vv = verlet_forces[i][axis] as f64;
            let tv = tile_forces[i][axis] as f64;
            let err = (vv - tv).abs();
            if err > max_err {
                max_err = err;
                label = format!("atom {i} axis {axis}: verlet={vv:.6} tile={tv:.6} err={err:.6}");
            }
        }
    }
    eprintln!(
        "max verlet-vs-tile force discrepancy: {:.3e} kJ/mol/Å — {label}",
        max_err
    );
    // Same physics, same params, same f32 ops → should agree to f32
    // noise floor.  Tolerance is loose to absorb floating-point
    // reordering noise (the two kernels accumulate in different
    // orders).
    let _ = label;
    assert!(
        max_err < 1e-2,
        "tile and verlet kernels disagree past tolerance"
    );

    // Also: every non-zero force from one kernel should be matched
    // by a non-zero force from the other.
    let verlet_norm: f64 = verlet_forces
        .iter()
        .map(|f| (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]) as f64)
        .sum::<f64>()
        .sqrt();
    let tile_norm: f64 = tile_forces
        .iter()
        .map(|f| (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]) as f64)
        .sum::<f64>()
        .sqrt();
    eprintln!(
        "force magnitude — verlet: {:.3} tile: {:.3}",
        verlet_norm, tile_norm
    );
    assert!(verlet_norm > 0.0 && tile_norm > 0.0);
    assert!(
        (verlet_norm - tile_norm).abs() / verlet_norm < 0.001,
        "force magnitude difference too large"
    );
}

#[test]
fn tile_kernel_matches_verlet_kernel_at_multiple_tile_scales() {
    let ctx = match GpuContext::get() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("GPU unavailable: {e}");
            return;
        }
    };
    // Build a sequence long enough to span multiple 64-atom tiles.
    // 20 residues of AGLEK ≈ 290 atoms → ~5 tiles.  40 ≈ 580 atoms
    // → ~9 tiles.
    for n_reps in [20usize, 40] {
        let block = [
            AminoAcid::Ala,
            AminoAcid::Gly,
            AminoAcid::Leu,
            AminoAcid::Glu,
            AminoAcid::Lys,
        ];
        let seq: Vec<AminoAcid> = block.iter().cloned().cycle().take(n_reps * 5).collect();
        let (err, n, n_tiles, n_inter) = run_pair_test(&seq, ctx);
        eprintln!(
            "n_reps={n_reps}: {n} atoms, {n_tiles} tiles, {n_inter} tile interactions, max err {:.3e}",
            err
        );
        // f32 reorder noise — same physics, different accumulation
        // order across kernels.
        assert!(
            err < 1.0_f64,
            "tile vs verlet disagreement {err:.3e} at n_reps={n_reps}"
        );
    }
}
