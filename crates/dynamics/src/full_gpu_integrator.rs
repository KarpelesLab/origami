//! GPU-resident BAOAB integrator.  Wraps `gpu::IntegratorPipeline`
//! and handles the bits the pipeline doesn't:
//!
//!   - Building the bonded-term tables + per-atom CSR inverse topology
//!     from the CPU `TopologyGraph` + `ForceField` at construction.
//!   - Building the LJ + Coulomb and GB neighbour lists at construction
//!     and refreshing them between integrator batches (uses the same
//!     CPU drift detector as `energy::forces_nonbonded::ensure_verlet_list`).
//!   - Translating between the CPU `Structure` (positions in Å as f64
//!     `Vec3`) and the GPU's f32 `[f32; 3]` per-atom arrays.
//!
//! Usage from the integrator:
//!
//! ```ignore
//! let mut full = FullGpuIntegrator::new(structure, graph, ff, dt, gamma, temp, seed);
//! full.upload_initial_state(&structure, &velocities);
//! for _ in 0..n_batches {
//!     full.step_batch(save_every);                     // GPU only — no per-step sync
//!     full.download_positions_into(&mut structure);    // CPU sync at frame boundary
//!     callback(structure);
//! }
//! ```
//!
//! `step_batch` does one CPU-side neighbour-list check + (optional)
//! rebuild before calling `IntegratorPipeline::step_n`.  The Verlet
//! skin (2 Å) protects the list inside the batch — as long as no atom
//! drifts more than `skin/2` in `n_steps` steps, the cached list stays
//! correct.  At the trajectory scales the existing CPU integrator runs,
//! that's typically true for batches of 20-50 steps at dt = 1 fs.

use chem::{classify_atom, AtomType, Element, ForceField};
use energy::scratch::ForceScratch;
use energy::forces_nonbonded::ensure_verlet_list;
use energy::gb::{
    ensure_gb_verlet_list, hct_scale_pub, intrinsic_radius_pub, BORN_RADIUS_CUTOFF_A_PUB,
    OBC_OFFSET_PUB,
};
use energy::forces_gb::GB_DEFAULT_CUTOFF_A_PUB;
use energy::units::{deg_to_rad, kcal_to_kj};
use energy::DEFAULT_CUTOFF_A;
use geom::{Structure, TopologyGraph, Vec3};
use gpu::{
    pair_list_to_csr, AngleTerm, BondTerm, BondedSetup, DihedralTerm, GbSetup, GpuContext,
    ImproperTerm, IntegratorPipeline, PerXShakeData, PeriodicTerm, VerletNonbondedSetup,
};

const KCAL_TO_KJ: f32 = 4.184;

pub struct FullGpuIntegrator {
    n_atoms: usize,
    integ: IntegratorPipeline,
    /// CPU-side scratch used for the Verlet-list drift check and
    /// rebuild.  Forces fields are not touched; only the position +
    /// neighbour-list bookkeeping.
    scratch: ForceScratch,
    /// Cached CSR buffers — re-uploaded only when the CPU rebuilt
    /// the corresponding list.
    nb_counts: Vec<u32>,
    nb_starts: Vec<u32>,
    nb_indices: Vec<u32>,
    gb_counts: Vec<u32>,
    gb_starts: Vec<u32>,
    gb_indices: Vec<u32>,
    /// Per-atom positions scratch for upload/download.
    pos_buf: Vec<[f32; 3]>,
    vel_buf: Vec<[f32; 3]>,
}

impl FullGpuIntegrator {
    pub fn new(
        structure: &Structure,
        graph: &TopologyGraph,
        ff: &ForceField,
        dt_fs: f64,
        gamma_ps_inv: f64,
        temperature_k: f64,
        rng_seed: u64,
    ) -> Result<Self, gpu::context::GpuInitError> {
        let ctx = GpuContext::get()?;
        let n = structure.atom_count();
        let atom_types = build_atom_types(structure);
        let mut masses_f32: Vec<f32> = Vec::with_capacity(n);
        let mut charges: Vec<f32> = Vec::with_capacity(n);
        let mut rho: Vec<f32> = Vec::with_capacity(n);
        let mut rho_tilde: Vec<f32> = Vec::with_capacity(n);
        let mut scale: Vec<f32> = Vec::with_capacity(n);
        for r in &structure.residues {
            for a in &r.atoms {
                masses_f32.push(a.element.mass_da() as f32);
                charges.push(ff.partial_charge_for(r.monomer, a.name).unwrap_or(0.0) as f32);
                let r0 = intrinsic_radius_pub(a.element);
                rho.push(r0 as f32);
                rho_tilde.push((r0 - OBC_OFFSET_PUB) as f32);
                scale.push(hct_scale_pub(a.element) as f32);
            }
        }
        let atom_lj_data: Vec<[f32; 4]> = atom_types
            .iter()
            .map(|t| {
                let p = ff.nonbonded(*t).unwrap_or_else(|| {
                    panic!("no nonbonded params for {:?}", t)
                });
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
        // Build the exclusion + 1-4 bitmaps by walking the graph's
        // sparse bond/angle/dihedral lists instead of doing the naive
        // O(N²) (i, j) → graph-lookup scan.  At 5840 atoms the naive
        // path takes ~1.7 s of CPU (34 M pair tests × ~50 ns per
        // is_one_four lookup which itself walks O(degree²) bonded_to
        // neighbours).  Sparse traversal is O(N · avg_degree) and
        // takes ~1 ms.  Identical bit pattern at the end: each pair
        // that's 1-2, 1-3, or 1-4 gets its corresponding flag set.
        let n_words = (n * n).div_ceil(32);
        let mut exclusions = vec![0u32; n_words];
        let mut one_four = vec![0u32; n_words];
        let set_bit = |buf: &mut [u32], a: usize, b: usize| {
            let bit = a * n + b;
            buf[bit / 32] |= 1u32 << (bit % 32);
            let bit = b * n + a;
            buf[bit / 32] |= 1u32 << (bit % 32);
        };
        for b in &graph.bonds {
            set_bit(&mut exclusions, b.a, b.b);
        }
        for a in &graph.angles {
            // The angle (a, b, c) implies the 1-3 pair (a, c).  b is
            // the central atom; we already handled the 1-2 a-b and
            // b-c via the bond list above.
            set_bit(&mut exclusions, a.a, a.c);
        }
        for d in &graph.dihedrals {
            // Dihedral (a, b, c, d) gives the 1-4 pair (a, d).
            // If that pair is *also* 1-2 or 1-3 (e.g. a 4-membered
            // ring), the exclusion bit is already set above and the
            // 1-4 bit here is harmless — the kernel's exclusion check
            // takes precedence.
            set_bit(&mut one_four, d.a, d.d);
        }

        // Bonded term tables + per-atom CSR.
        let (bond_terms, ab_count, ab_start, ab_index) = build_bonds(graph, ff, &atom_types, n);
        let (angle_terms, aa_count, aa_start, aa_index) = build_angles(graph, ff, &atom_types, n);
        let (dihedral_terms, ad_count, ad_start, ad_index) =
            build_dihedrals(graph, ff, &atom_types, n);
        let (improper_terms, ai_count, ai_start, ai_index) =
            build_impropers(graph, ff, &atom_types, n);

        let nb_setup = VerletNonbondedSetup {
            atom_lj_data: &atom_lj_data,
            charges: &charges,
            exclusions: &exclusions,
            one_four_mask: &one_four,
            cutoff_a: DEFAULT_CUTOFF_A as f32,
            initial_indices_capacity: (n * 200).max(64),
        };
        let gb_setup = GbSetup {
            rho: &rho,
            rho_tilde: &rho_tilde,
            scale: &scale,
            charges: &charges,
            cutoff_a: BORN_RADIUS_CUTOFF_A_PUB as f32,
            pair_cutoff_a: GB_DEFAULT_CUTOFF_A_PUB as f32,
            initial_indices_capacity: (n * 2000).max(64),
        };
        let bonded_setup = BondedSetup {
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
        };

        let integ = IntegratorPipeline::new(
            ctx, n, &masses_f32, rng_seed, bonded_setup, nb_setup, gb_setup,
        );

        // BAOAB Langevin params.
        const BOLTZMANN_KJ_PER_MOL_K: f64 = 8.314_462_618e-3;
        let kbt = (BOLTZMANN_KJ_PER_MOL_K * temperature_k) as f32;
        integ.set_step_params(dt_fs as f32, gamma_ps_inv as f32, kbt);

        let scratch = ForceScratch::new(structure, graph, ff);
        Ok(Self {
            n_atoms: n,
            integ,
            scratch,
            nb_counts: Vec::new(),
            nb_starts: Vec::new(),
            nb_indices: Vec::new(),
            gb_counts: Vec::new(),
            gb_starts: Vec::new(),
            gb_indices: Vec::new(),
            pos_buf: vec![[0.0; 3]; n],
            vel_buf: vec![[0.0; 3]; n],
        })
    }

    pub fn upload_initial_state(&mut self, structure: &Structure, velocities: &[Vec3]) {
        let mut idx = 0;
        for r in &structure.residues {
            for a in &r.atoms {
                self.pos_buf[idx] = [
                    a.position.x as f32,
                    a.position.y as f32,
                    a.position.z as f32,
                ];
                idx += 1;
            }
        }
        for (i, v) in velocities.iter().enumerate() {
            self.vel_buf[i] = [v.x as f32, v.y as f32, v.z as f32];
        }
        self.integ.upload_positions(&self.pos_buf);
        self.integ.upload_velocities(&self.vel_buf);
        self.refresh_neighbour_lists();
    }

    /// Run `n_steps` integrator iterations on the GPU.  Refreshes the
    /// LJ + Coulomb (10 Å) and GB (20 Å) neighbour lists first if any
    /// atom has drifted past `VERLET_SKIN / 2` since the last refresh.
    ///
    /// The drift check pulls positions from the GPU.  At the bench
    /// scales where this method is worthwhile (≥1000 atoms), the
    /// download is 12 KB to 70 KB and costs ~0.1 ms — negligible
    /// compared to the saved per-step CPU sync that the existing
    /// `GpuAccelerator` path pays every step.
    pub fn step_batch(&mut self, n_steps: usize) {
        self.refresh_neighbour_lists();
        self.integ.step_n(n_steps);
    }

    /// Enable SHAKE on the underlying integrator pipeline.  After
    /// this, [`step_batch_shake`] is callable.
    pub fn enable_shake(&mut self, shake_data: &PerXShakeData, max_iters: u32, tol_sq: f32) {
        self.integ.enable_shake(shake_data, max_iters, tol_sq);
    }

    /// SHAKE-mode batched step.  Same drift-check + neighbour refresh
    /// machinery as [`step_batch`]; calls `step_n_shake` instead of
    /// `step_n`.
    pub fn step_batch_shake(&mut self, n_steps: usize) {
        self.refresh_neighbour_lists();
        self.integ.step_n_shake(n_steps);
    }

    fn refresh_neighbour_lists(&mut self) {
        let positions = self.integ.download_positions();
        // Sync into scratch positions for the drift detector.
        for i in 0..self.n_atoms {
            self.scratch.xs[i] = positions[i][0] as f64;
            self.scratch.ys[i] = positions[i][1] as f64;
            self.scratch.zs[i] = positions[i][2] as f64;
        }
        let nb_rebuilt = ensure_verlet_list(&mut self.scratch, DEFAULT_CUTOFF_A);
        let gb_rebuilt = ensure_gb_verlet_list(&mut self.scratch, BORN_RADIUS_CUTOFF_A_PUB);
        if nb_rebuilt || self.nb_counts.is_empty() {
            let (c, s, i) = pair_list_to_csr(self.n_atoms, &self.scratch.verlet_pairs);
            self.nb_counts = c;
            self.nb_starts = s;
            self.nb_indices = i;
            self.integ.update_nb_neighbours(&self.nb_counts, &self.nb_starts, &self.nb_indices);
        }
        if gb_rebuilt || self.gb_counts.is_empty() {
            let (c, s, i) = pair_list_to_csr(self.n_atoms, &self.scratch.gb_verlet_pairs);
            self.gb_counts = c;
            self.gb_starts = s;
            self.gb_indices = i;
            self.integ.update_gb_neighbours(&self.gb_counts, &self.gb_starts, &self.gb_indices);
        }
    }

    /// Pull current positions from the GPU and write them back into
    /// `structure`.
    pub fn download_positions_into(&self, structure: &mut Structure) {
        let positions = self.integ.download_positions();
        let mut idx = 0;
        for r in &mut structure.residues {
            for a in &mut r.atoms {
                a.position.x = positions[idx][0] as f64;
                a.position.y = positions[idx][1] as f64;
                a.position.z = positions[idx][2] as f64;
                idx += 1;
            }
        }
    }

    pub fn download_velocities(&self) -> Vec<Vec3> {
        let v = self.integ.download_velocities();
        v.into_iter()
            .map(|vv| Vec3::new(vv[0] as f64, vv[1] as f64, vv[2] as f64))
            .collect()
    }

    pub fn n_atoms(&self) -> usize { self.n_atoms }
}

// ---- Builder helpers (move into a shared util if other code needs them too) ----

fn build_atom_types(s: &Structure) -> Vec<AtomType> {
    let mut out = Vec::with_capacity(s.atom_count());
    for r in &s.residues {
        for a in &r.atoms {
            out.push(
                classify_atom(r.monomer, a.name).unwrap_or_else(|| {
                    panic!("unclassified atom {:?} {}", r.monomer, a.name)
                }),
            );
        }
    }
    out
}

fn _silence_element(_: Element) {}

fn build_bonds(g: &TopologyGraph, ff: &ForceField, atom_types: &[AtomType], n: usize)
    -> (Vec<BondTerm>, Vec<u32>, Vec<u32>, Vec<u32>)
{
    let mut terms: Vec<BondTerm> = Vec::new();
    let mut per_atom: Vec<Vec<u32>> = vec![Vec::new(); n];
    for b in &g.bonds {
        let Some(p) = ff.bond(atom_types[b.a], atom_types[b.b]) else { continue };
        let idx = terms.len() as u32;
        terms.push(BondTerm {
            a: b.a as u32, b: b.b as u32,
            k_kj: kcal_to_kj(p.k) as f32, r0_a: p.r0 as f32,
        });
        per_atom[b.a].push(idx);
        per_atom[b.b].push(idx);
    }
    let (c, s, i) = flatten_csr(per_atom, n);
    (terms, c, s, i)
}

fn build_angles(g: &TopologyGraph, ff: &ForceField, atom_types: &[AtomType], n: usize)
    -> (Vec<AngleTerm>, Vec<u32>, Vec<u32>, Vec<u32>)
{
    let mut terms: Vec<AngleTerm> = Vec::new();
    let mut per_atom: Vec<Vec<u32>> = vec![Vec::new(); n];
    for a in &g.angles {
        let Some(p) = ff.angle(atom_types[a.a], atom_types[a.b], atom_types[a.c]) else { continue };
        let idx = terms.len() as u32;
        terms.push(AngleTerm {
            a: a.a as u32, b: a.b as u32, c: a.c as u32, _pad: 0,
            k_kj: kcal_to_kj(p.k) as f32, theta0_rad: deg_to_rad(p.theta0_deg) as f32,
            _pad2: 0.0, _pad3: 0.0,
        });
        per_atom[a.a].push(idx);
        per_atom[a.b].push(idx);
        per_atom[a.c].push(idx);
    }
    let (c, s, i) = flatten_csr(per_atom, n);
    (terms, c, s, i)
}

fn build_dihedrals(g: &TopologyGraph, ff: &ForceField, atom_types: &[AtomType], n: usize)
    -> (Vec<DihedralTerm>, Vec<u32>, Vec<u32>, Vec<u32>)
{
    let mut terms: Vec<DihedralTerm> = Vec::new();
    let mut per_atom: Vec<Vec<u32>> = vec![Vec::new(); n];
    for d in &g.dihedrals {
        let Some(pterms) = ff.dihedral(
            atom_types[d.a], atom_types[d.b], atom_types[d.c], atom_types[d.d],
        ) else { continue };
        let mut packed = DihedralTerm {
            a: d.a as u32, b: d.b as u32, c: d.c as u32, d: d.d as u32,
            n_terms: pterms.len().min(4) as u32,
            _pad0: 0, _pad1: 0, _pad2: 0,
            term0: zero_term(), term1: zero_term(), term2: zero_term(), term3: zero_term(),
        };
        for (i, t) in pterms.iter().take(4).enumerate() {
            let pt = PeriodicTerm {
                k_kj: kcal_to_kj(t.k) as f32,
                n: t.n as f32,
                delta_rad: deg_to_rad(t.delta_deg) as f32, _pad: 0.0,
            };
            match i { 0 => packed.term0 = pt, 1 => packed.term1 = pt, 2 => packed.term2 = pt, _ => packed.term3 = pt }
        }
        let idx = terms.len() as u32;
        terms.push(packed);
        per_atom[d.a].push(idx);
        per_atom[d.b].push(idx);
        per_atom[d.c].push(idx);
        per_atom[d.d].push(idx);
    }
    let (c, s, i) = flatten_csr(per_atom, n);
    (terms, c, s, i)
}

fn build_impropers(g: &TopologyGraph, ff: &ForceField, atom_types: &[AtomType], n: usize)
    -> (Vec<ImproperTerm>, Vec<u32>, Vec<u32>, Vec<u32>)
{
    let mut terms: Vec<ImproperTerm> = Vec::new();
    let mut per_atom: Vec<Vec<u32>> = vec![Vec::new(); n];
    for imp in &g.impropers {
        let Some(p) = ff.improper(
            atom_types[imp.a], atom_types[imp.b], atom_types[imp.c], atom_types[imp.d],
        ) else { continue };
        let idx = terms.len() as u32;
        terms.push(ImproperTerm {
            a: imp.a as u32, b: imp.b as u32, c: imp.c as u32, d: imp.d as u32,
            k_kj: kcal_to_kj(p.k) as f32, omega0_rad: deg_to_rad(p.psi0_deg) as f32,
            _pad0: 0.0, _pad1: 0.0,
        });
        per_atom[imp.a].push(idx);
        per_atom[imp.b].push(idx);
        per_atom[imp.c].push(idx);
        per_atom[imp.d].push(idx);
    }
    let (c, s, i) = flatten_csr(per_atom, n);
    (terms, c, s, i)
}

fn zero_term() -> PeriodicTerm {
    PeriodicTerm { k_kj: 0.0, n: 0.0, delta_rad: 0.0, _pad: 0.0 }
}

fn flatten_csr(per_atom: Vec<Vec<u32>>, n: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
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
    (counts, starts, indices)
}
