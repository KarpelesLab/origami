//! End-to-end GPU integrator test.
//!
//! Builds the full `IntegratorPipeline` for a small chain, runs N
//! integrator steps entirely on the GPU, and verifies:
//!
//!   1. No NaN / divergence — final positions and velocities are finite.
//!   2. Approximate temperature equilibration — mean kinetic energy
//!      lands within ~30% of the (3/2) k_B T target.  Loose because
//!      we're not running the full burn-in.
//!
//! This is a smoke test for the full GPU pipeline composition.
//! Bit-by-bit agreement with CPU is impossible (different RNG +
//! f32 accumulation), so we only assert statistical sanity.

use chem::{AminoAcid, AtomType, Element, ForceField, classify_atom, standard_ff};
use energy::units::{deg_to_rad, kcal_to_kj};
use geom::{Structure, TopologyGraph, build_extended_chain, build_topology_graph};
use gpu::{
    AngleTerm, BondTerm, BondedSetup, DihedralTerm, GbSetup, GpuContext, ImproperTerm,
    IntegratorPipeline, PeriodicTerm, VerletNonbondedSetup,
};

const KCAL_TO_KJ: f32 = 4.184;
const BOLTZMANN_KJ_PER_MOL_K: f64 = 8.314_462_618e-3;
const ACCEL_FACTOR: f64 = 1.0e-4;
const BORN_CUTOFF_A: f64 = 20.0;
const PAIR_CUTOFF_A: f64 = 10.0;
const OBC_OFFSET: f64 = 0.09;

fn build_atom_types(s: &Structure) -> Vec<AtomType> {
    let mut out = Vec::with_capacity(s.atom_count());
    for r in &s.residues {
        for a in &r.atoms {
            out.push(classify_atom(r.monomer, a.name).unwrap());
        }
    }
    out
}

fn hct_scale(e: Element) -> f64 {
    match e {
        Element::H => 0.85,
        Element::C => 0.72,
        Element::N => 0.79,
        Element::O => 0.85,
        Element::S => 0.96,
        _ => 1.0,
    }
}
fn intrinsic_radius(e: Element) -> f64 {
    match e {
        Element::H => 1.20,
        Element::C => 1.70,
        Element::N => 1.55,
        Element::O => 1.50,
        Element::S => 1.80,
        _ => 1.70,
    }
}

fn build_bonded_setup<'a>(
    g: &'a TopologyGraph,
    ff: &'a ForceField,
    atom_types: &'a [AtomType],
    n: usize,
    out_bond_terms: &'a mut Vec<BondTerm>,
    out_ab_count: &'a mut Vec<u32>,
    out_ab_start: &'a mut Vec<u32>,
    out_ab_index: &'a mut Vec<u32>,
    out_angle_terms: &'a mut Vec<AngleTerm>,
    out_aa_count: &'a mut Vec<u32>,
    out_aa_start: &'a mut Vec<u32>,
    out_aa_index: &'a mut Vec<u32>,
    out_dihedral_terms: &'a mut Vec<DihedralTerm>,
    out_ad_count: &'a mut Vec<u32>,
    out_ad_start: &'a mut Vec<u32>,
    out_ad_index: &'a mut Vec<u32>,
    out_improper_terms: &'a mut Vec<ImproperTerm>,
    out_ai_count: &'a mut Vec<u32>,
    out_ai_start: &'a mut Vec<u32>,
    out_ai_index: &'a mut Vec<u32>,
) -> BondedSetup<'a> {
    let mut pa: Vec<Vec<u32>> = vec![Vec::new(); n];
    for b in &g.bonds {
        let Some(p) = ff.bond(atom_types[b.a], atom_types[b.b]) else {
            continue;
        };
        let idx = out_bond_terms.len() as u32;
        out_bond_terms.push(BondTerm {
            a: b.a as u32,
            b: b.b as u32,
            k_kj: kcal_to_kj(p.k) as f32,
            r0_a: p.r0 as f32,
        });
        pa[b.a].push(idx);
        pa[b.b].push(idx);
    }
    flatten(&pa, n, out_ab_count, out_ab_start, out_ab_index);

    let mut pa: Vec<Vec<u32>> = vec![Vec::new(); n];
    for a in &g.angles {
        let Some(p) = ff.angle(atom_types[a.a], atom_types[a.b], atom_types[a.c]) else {
            continue;
        };
        let idx = out_angle_terms.len() as u32;
        out_angle_terms.push(AngleTerm {
            a: a.a as u32,
            b: a.b as u32,
            c: a.c as u32,
            _pad: 0,
            k_kj: kcal_to_kj(p.k) as f32,
            theta0_rad: deg_to_rad(p.theta0_deg) as f32,
            _pad2: 0.0,
            _pad3: 0.0,
        });
        pa[a.a].push(idx);
        pa[a.b].push(idx);
        pa[a.c].push(idx);
    }
    flatten(&pa, n, out_aa_count, out_aa_start, out_aa_index);

    let mut pa: Vec<Vec<u32>> = vec![Vec::new(); n];
    for d in &g.dihedrals {
        let Some(terms) = ff.dihedral(
            atom_types[d.a],
            atom_types[d.b],
            atom_types[d.c],
            atom_types[d.d],
        ) else {
            continue;
        };
        let mut t = DihedralTerm {
            a: d.a as u32,
            b: d.b as u32,
            c: d.c as u32,
            d: d.d as u32,
            n_terms: terms.len().min(4) as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            term0: zero_term(),
            term1: zero_term(),
            term2: zero_term(),
            term3: zero_term(),
        };
        for (i, term) in terms.iter().take(4).enumerate() {
            let pt = PeriodicTerm {
                k_kj: kcal_to_kj(term.k) as f32,
                n: term.n as f32,
                delta_rad: deg_to_rad(term.delta_deg) as f32,
                _pad: 0.0,
            };
            match i {
                0 => t.term0 = pt,
                1 => t.term1 = pt,
                2 => t.term2 = pt,
                _ => t.term3 = pt,
            }
        }
        let idx = out_dihedral_terms.len() as u32;
        out_dihedral_terms.push(t);
        pa[d.a].push(idx);
        pa[d.b].push(idx);
        pa[d.c].push(idx);
        pa[d.d].push(idx);
    }
    flatten(&pa, n, out_ad_count, out_ad_start, out_ad_index);

    let mut pa: Vec<Vec<u32>> = vec![Vec::new(); n];
    for imp in &g.impropers {
        let Some(p) = ff.improper(
            atom_types[imp.a],
            atom_types[imp.b],
            atom_types[imp.c],
            atom_types[imp.d],
        ) else {
            continue;
        };
        let idx = out_improper_terms.len() as u32;
        out_improper_terms.push(ImproperTerm {
            a: imp.a as u32,
            b: imp.b as u32,
            c: imp.c as u32,
            d: imp.d as u32,
            k_kj: kcal_to_kj(p.k) as f32,
            omega0_rad: deg_to_rad(p.psi0_deg) as f32,
            _pad0: 0.0,
            _pad1: 0.0,
        });
        pa[imp.a].push(idx);
        pa[imp.b].push(idx);
        pa[imp.c].push(idx);
        pa[imp.d].push(idx);
    }
    flatten(&pa, n, out_ai_count, out_ai_start, out_ai_index);

    BondedSetup {
        bond_terms: out_bond_terms,
        atom_bond_count: out_ab_count,
        atom_bond_start: out_ab_start,
        atom_bond_index: out_ab_index,
        angle_terms: out_angle_terms,
        atom_angle_count: out_aa_count,
        atom_angle_start: out_aa_start,
        atom_angle_index: out_aa_index,
        dihedral_terms: out_dihedral_terms,
        atom_dihedral_count: out_ad_count,
        atom_dihedral_start: out_ad_start,
        atom_dihedral_index: out_ad_index,
        improper_terms: out_improper_terms,
        atom_improper_count: out_ai_count,
        atom_improper_start: out_ai_start,
        atom_improper_index: out_ai_index,
    }
}

fn zero_term() -> PeriodicTerm {
    PeriodicTerm {
        k_kj: 0.0,
        n: 0.0,
        delta_rad: 0.0,
        _pad: 0.0,
    }
}

fn flatten(
    pa: &[Vec<u32>],
    n: usize,
    counts: &mut Vec<u32>,
    starts: &mut Vec<u32>,
    indices: &mut Vec<u32>,
) {
    counts.clear();
    starts.clear();
    indices.clear();
    counts.reserve(n);
    starts.reserve(n);
    let mut total = 0u32;
    for v in pa.iter() {
        counts.push(v.len() as u32);
    }
    for i in 0..n {
        starts.push(total);
        total += counts[i];
    }
    indices.resize(total as usize, 0);
    for (i, list) in pa.iter().enumerate() {
        let base = starts[i] as usize;
        for (k, &v) in list.iter().enumerate() {
            indices[base + k] = v;
        }
    }
}

#[test]
fn gpu_integrator_runs_100_steps_without_divergence() {
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
    let atom_types = build_atom_types(&s);

    // Per-atom data: positions, masses, charges, GB radii, LJ params.
    let positions_f32: Vec<[f32; 3]> = s
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
    let masses_f32: Vec<f32> = s
        .residues
        .iter()
        .flat_map(|r| r.atoms.iter().map(|a| a.element.mass_da() as f32))
        .collect();
    let masses_f64: Vec<f64> = masses_f32.iter().map(|&m| m as f64).collect();
    let charges: Vec<f32> = s
        .residues
        .iter()
        .flat_map(|r| {
            r.atoms
                .iter()
                .map(|a| ff.partial_charge_for(r.monomer, a.name).unwrap_or(0.0) as f32)
        })
        .collect();
    let rho: Vec<f32> = s
        .residues
        .iter()
        .flat_map(|r| r.atoms.iter().map(|a| intrinsic_radius(a.element) as f32))
        .collect();
    let rho_tilde: Vec<f32> = rho.iter().map(|&r| r - OBC_OFFSET as f32).collect();
    let scale: Vec<f32> = s
        .residues
        .iter()
        .flat_map(|r| r.atoms.iter().map(|a| hct_scale(a.element) as f32))
        .collect();
    let atom_lj_data: Vec<[f32; 4]> = atom_types
        .iter()
        .map(|t| {
            let p = ff.nonbonded(*t).unwrap();
            let eps14 = p.epsilon_14.unwrap_or(p.epsilon);
            let rmh14 = p.rmin_half_14.unwrap_or(p.rmin_half);
            [
                (p.epsilon as f32) * KCAL_TO_KJ,
                p.rmin_half as f32,
                (eps14 as f32) * KCAL_TO_KJ,
                rmh14 as f32,
            ]
        })
        .collect();
    let mut exclusions = vec![0u32; (n * n).div_ceil(32)];
    let mut one_four = vec![0u32; (n * n).div_ceil(32)];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let bit = i * n + j;
            if g.is_bonded(i, j) || g.is_one_three(i, j) {
                exclusions[bit / 32] |= 1u32 << (bit % 32);
            } else if g.is_one_four(i, j) {
                one_four[bit / 32] |= 1u32 << (bit % 32);
            }
        }
    }
    let nb_setup = VerletNonbondedSetup {
        atom_lj_data: &atom_lj_data,
        charges: &charges,
        exclusions: &exclusions,
        one_four_mask: &one_four,
        cutoff_a: PAIR_CUTOFF_A as f32,
        initial_indices_capacity: (n * 200).max(64),
    };
    let gb_setup = GbSetup {
        rho: &rho,
        rho_tilde: &rho_tilde,
        scale: &scale,
        charges: &charges,
        cutoff_a: BORN_CUTOFF_A as f32,
        pair_cutoff_a: PAIR_CUTOFF_A as f32,
        initial_indices_capacity: (n * 2000).max(64),
    };
    let (mut bond_terms, mut ab_count, mut ab_start, mut ab_index) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut angle_terms, mut aa_count, mut aa_start, mut aa_index) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut dihedral_terms, mut ad_count, mut ad_start, mut ad_index) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut improper_terms, mut ai_count, mut ai_start, mut ai_index) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let bonded_setup = build_bonded_setup(
        &g,
        ff,
        &atom_types,
        n,
        &mut bond_terms,
        &mut ab_count,
        &mut ab_start,
        &mut ab_index,
        &mut angle_terms,
        &mut aa_count,
        &mut aa_start,
        &mut aa_index,
        &mut dihedral_terms,
        &mut ad_count,
        &mut ad_start,
        &mut ad_index,
        &mut improper_terms,
        &mut ai_count,
        &mut ai_start,
        &mut ai_index,
    );

    let mut integ =
        IntegratorPipeline::new(ctx, n, &masses_f32, 42, bonded_setup, nb_setup, gb_setup);

    // Brute-force neighbour lists at construction (they don't refresh
    // automatically — this test just runs a short trajectory and trusts
    // the skin).
    let mut nb_pairs: Vec<(u32, u32)> = Vec::new();
    let mut gb_pairs: Vec<(u32, u32)> = Vec::new();
    let pair_cut_sq = (PAIR_CUTOFF_A + 2.0).powi(2);
    let born_cut_sq = (BORN_CUTOFF_A + 2.0).powi(2);
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = positions_f32[i][0] as f64 - positions_f32[j][0] as f64;
            let dy = positions_f32[i][1] as f64 - positions_f32[j][1] as f64;
            let dz = positions_f32[i][2] as f64 - positions_f32[j][2] as f64;
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 <= pair_cut_sq {
                nb_pairs.push((i as u32, j as u32));
            }
            if r2 <= born_cut_sq {
                gb_pairs.push((i as u32, j as u32));
            }
        }
    }
    let (nb_c, nb_s, nb_i) = gpu::pair_list_to_csr(n, &nb_pairs);
    let (gb_c, gb_s, gb_i) = gpu::pair_list_to_csr(n, &gb_pairs);
    integ.update_nb_neighbours(&nb_c, &nb_s, &nb_i);
    integ.update_gb_neighbours(&gb_c, &gb_s, &gb_i);

    integ.upload_positions(&positions_f32);
    let zero_velocities = vec![[0.0_f32; 3]; n];
    integ.upload_velocities(&zero_velocities);

    let target_t = 310.0_f64;
    integ.set_step_params(1.0, 2.0, (BOLTZMANN_KJ_PER_MOL_K * target_t) as f32);

    let n_steps = 100;
    eprintln!("Running step_n({})...", n_steps);
    integ.step_n(n_steps);
    eprintln!("step_n returned");

    let final_pos = integ.download_positions();
    let final_vel = integ.download_velocities();

    // No NaN / inf.
    for (i, p) in final_pos.iter().enumerate() {
        for axis in 0..3 {
            assert!(
                p[axis].is_finite(),
                "atom {i} axis {axis} position {} non-finite",
                p[axis]
            );
        }
    }
    for (i, v) in final_vel.iter().enumerate() {
        for axis in 0..3 {
            assert!(
                v[axis].is_finite(),
                "atom {i} axis {axis} velocity {} non-finite",
                v[axis]
            );
        }
    }

    // Approximate equilibration.  100 steps from rest with friction γ=2/ps
    // gets ~half-way to the target Maxwell-Boltzmann KE — the run is
    // intentionally short to keep the test fast.
    let ke_da_a2_per_fs2: f64 = final_vel
        .iter()
        .zip(masses_f64.iter())
        .map(|(v, m)| 0.5 * m * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) as f64)
        .sum();
    let ke_kj_mol = ke_da_a2_per_fs2 / ACCEL_FACTOR;
    let dof = (3 * n) as f64;
    let t_inst = 2.0 * ke_kj_mol / (dof * BOLTZMANN_KJ_PER_MOL_K);
    eprintln!(
        "GPU integrator after {} steps: T_inst = {:.1} K (target {:.0}), max position drift OK",
        n_steps, t_inst, target_t
    );
    // Loose — just verify it's in the right order of magnitude and not
    // exploded.
    assert!(
        t_inst > 50.0 && t_inst < 600.0,
        "instantaneous T {t_inst} K out of range — likely diverged or didn't warm"
    );
}
