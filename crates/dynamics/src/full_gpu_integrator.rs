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

use chem::{AtomType, Element, ForceField, classify_atom};
use energy::DEFAULT_CUTOFF_A;
use energy::forces_gb::GB_DEFAULT_CUTOFF_A_PUB;
use energy::forces_nonbonded::ensure_verlet_list;
use energy::gb::{
    BORN_RADIUS_CUTOFF_A_PUB, OBC_OFFSET_PUB, ensure_gb_verlet_list, hct_scale_pub,
    intrinsic_radius_pub,
};
use energy::scratch::ForceScratch;
use energy::units::{deg_to_rad, kcal_to_kj};
use geom::{Structure, TopologyGraph, Vec3};
use gpu::{
    AngleTerm, BondTerm, BondedSetup, DihedralTerm, GbSetup, GpuContext, ImproperTerm,
    IntegratorPipeline, PerXShakeData, PeriodicTerm, SASA_SMOOTH_DEFAULT_SIGMA_A, SasaSmoothSetup,
    TileNonbondedSetup, VerletNonbondedSetup, build_tile_interaction_list, morton_permutation,
    pair_list_to_csr,
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
    // ---- Spatial reindexing ----
    //
    // The GPU sees atoms in Morton (Z-order) order: atoms that are
    // spatially close get adjacent GPU indices, so consecutive
    // workgroups process atoms with heavily-overlapping Verlet
    // neighbour sets.  The CPU's `structure` keeps original order;
    // we translate at the upload / download boundaries.
    //   - `cpu_to_gpu[cpu_idx] = gpu_idx`
    //   - `gpu_to_cpu[gpu_idx] = cpu_idx`
    cpu_to_gpu: Vec<u32>,
    #[allow(dead_code)]
    gpu_to_cpu: Vec<u32>,
    /// When `true`, the integrator dispatches the tile-based
    /// nonbonded kernel (workgroup shared memory j-data cache) and
    /// the per-batch refresh builds the tile interaction list.
    /// When `false` (default), the legacy Verlet-list kernel is
    /// used.  Enable via [`enable_tile_nb_mode`].
    tile_nb_mode: bool,
    /// Cached tile-list bookkeeping, only populated when
    /// `tile_nb_mode` is true.  Re-uploaded only when the underlying
    /// Verlet list rebuilds.
    tile_count: Vec<u32>,
    tile_start: Vec<u32>,
    tile_indices: Vec<u32>,
    /// Per-atom params cached for tile-list construction (the tile
    /// list builder needs Morton-sorted positions, which we get
    /// from the GPU each refresh, plus the LJ + Coulomb cutoff).
    nb_cutoff_a: f32,
    /// SASA mode: when true, `refresh_neighbour_lists` also
    /// rebuilds the SASA CSR and uploads it.  Enabled via
    /// `enable_sasa_mode`.
    sasa_mode: bool,
    /// Per-atom expanded SASA radii in CPU index order (vdW + probe).
    sasa_radii_cpu_order: Vec<f64>,
    /// Cached SASA CSR — refreshed when the underlying SASA Verlet
    /// list rebuilds.
    sasa_counts: Vec<u32>,
    sasa_starts: Vec<u32>,
    sasa_indices: Vec<u32>,
    /// Drift-detection reference for SASA neighbour-list refresh,
    /// in CPU index order.
    sasa_ref_x: Vec<f64>,
    sasa_ref_y: Vec<f64>,
    sasa_ref_z: Vec<f64>,
    sasa_valid: bool,
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
        let atom_types = build_atom_types(structure); // CPU-indexed
        // First pass: gather per-atom data + initial positions in CPU
        // order — the same order `structure.residues` walks.
        let mut masses_cpu: Vec<f32> = Vec::with_capacity(n);
        let mut charges_cpu: Vec<f32> = Vec::with_capacity(n);
        let mut rho_cpu: Vec<f32> = Vec::with_capacity(n);
        let mut rho_tilde_cpu: Vec<f32> = Vec::with_capacity(n);
        let mut scale_cpu: Vec<f32> = Vec::with_capacity(n);
        let mut positions_cpu: Vec<[f32; 3]> = Vec::with_capacity(n);
        for r in &structure.residues {
            for a in &r.atoms {
                masses_cpu.push(a.element.mass_da() as f32);
                charges_cpu.push(ff.partial_charge_for(r.monomer, a.name).unwrap_or(0.0) as f32);
                let r0 = intrinsic_radius_pub(a.element);
                rho_cpu.push(r0 as f32);
                rho_tilde_cpu.push((r0 - OBC_OFFSET_PUB) as f32);
                scale_cpu.push(hct_scale_pub(a.element) as f32);
                positions_cpu.push([
                    a.position.x as f32,
                    a.position.y as f32,
                    a.position.z as f32,
                ]);
            }
        }
        let atom_lj_data_cpu: Vec<[f32; 4]> = atom_types
            .iter()
            .map(|t| {
                let p = ff
                    .nonbonded(*t)
                    .unwrap_or_else(|| panic!("no nonbonded params for {:?}", t));
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

        // ---- Morton spatial sort ----
        //
        // Reorder the per-atom GPU buffers so spatially-close atoms
        // get adjacent indices.  After this every per-atom buffer
        // sent to the GPU (positions, velocities, masses, charges,
        // LJ params, GB params, RNG seeds, bonded CSR atom indices,
        // exclusion bitmap) is in GPU order.  The CPU `structure`
        // stays in original order — we translate at the upload /
        // download boundaries.
        let (gpu_to_cpu, cpu_to_gpu) = morton_permutation(&positions_cpu);
        let permute =
            |src: &[f32]| -> Vec<f32> { (0..n).map(|g| src[gpu_to_cpu[g] as usize]).collect() };
        let permute_vec4 = |src: &[[f32; 4]]| -> Vec<[f32; 4]> {
            (0..n).map(|g| src[gpu_to_cpu[g] as usize]).collect()
        };
        let masses_f32 = permute(&masses_cpu);
        let charges = permute(&charges_cpu);
        let rho = permute(&rho_cpu);
        let rho_tilde = permute(&rho_tilde_cpu);
        let scale = permute(&scale_cpu);
        let atom_lj_data = permute_vec4(&atom_lj_data_cpu);

        // Build the exclusion + 1-4 bitmaps in *GPU* index space —
        // walk the graph's sparse lists in CPU index space, translate
        // each pair, set bits in the GPU-indexed bitmap.  Same
        // sparse-vs-O(N²) win as before; just an extra
        // `cpu_to_gpu[...]` per pair to translate.
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
            set_bit(
                &mut exclusions,
                cpu_to_gpu[b.a] as usize,
                cpu_to_gpu[b.b] as usize,
            );
        }
        for a in &graph.angles {
            set_bit(
                &mut exclusions,
                cpu_to_gpu[a.a] as usize,
                cpu_to_gpu[a.c] as usize,
            );
        }
        for d in &graph.dihedrals {
            set_bit(
                &mut one_four,
                cpu_to_gpu[d.a] as usize,
                cpu_to_gpu[d.d] as usize,
            );
        }

        // Bonded term tables + per-atom CSR — built in GPU index
        // space so the kernel's `forces[i]` accumulator lands in the
        // right GPU slot.  The builders look up FF params by atom
        // *type* (which is index-position independent), then store
        // GPU-translated atom indices in each term struct.
        let (bond_terms, ab_count, ab_start, ab_index) =
            build_bonds(graph, ff, &atom_types, &cpu_to_gpu, n);
        let (angle_terms, aa_count, aa_start, aa_index) =
            build_angles(graph, ff, &atom_types, &cpu_to_gpu, n);
        let (dihedral_terms, ad_count, ad_start, ad_index) =
            build_dihedrals(graph, ff, &atom_types, &cpu_to_gpu, n);
        let (improper_terms, ai_count, ai_start, ai_index) =
            build_impropers(graph, ff, &atom_types, &cpu_to_gpu, n);

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
            ctx,
            n,
            &masses_f32,
            rng_seed,
            bonded_setup,
            nb_setup,
            gb_setup,
        );

        // BAOAB Langevin params.
        const BOLTZMANN_KJ_PER_MOL_K: f64 = 8.314_462_618e-3;
        let kbt = (BOLTZMANN_KJ_PER_MOL_K * temperature_k) as f32;
        integ.set_step_params(dt_fs as f32, gamma_ps_inv as f32, kbt);

        let scratch = ForceScratch::new(structure, graph, ff);
        // Save the per-atom data needed if the caller later enables
        // tile mode — these are CPU-ordered.
        let _ = (&atom_lj_data, &charges, &exclusions, &one_four);
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
            cpu_to_gpu,
            gpu_to_cpu,
            tile_nb_mode: false,
            tile_count: Vec::new(),
            tile_start: Vec::new(),
            tile_indices: Vec::new(),
            nb_cutoff_a: DEFAULT_CUTOFF_A as f32,
            sasa_mode: false,
            sasa_radii_cpu_order: Vec::new(),
            sasa_counts: Vec::new(),
            sasa_starts: Vec::new(),
            sasa_indices: Vec::new(),
            sasa_ref_x: Vec::new(),
            sasa_ref_y: Vec::new(),
            sasa_ref_z: Vec::new(),
            sasa_valid: false,
        })
    }

    /// Enable smooth-coverage SASA forces.  The integrator will
    /// dispatch the GPU SASA force kernel after the GB pass in
    /// every force-eval, and `refresh_neighbour_lists` will rebuild
    /// the SASA Verlet list whenever atoms drift past the skin.
    ///
    /// `gammas_cpu_order` is in kJ/mol/Å² in CPU index order (use
    /// `energy::powersasa::default_sasa_gammas(structure)`).
    /// Internally translated to GPU/Morton order.
    pub fn enable_sasa_mode(&mut self, structure: &Structure, gammas_cpu_order: &[f64]) {
        let n = self.n_atoms;
        assert_eq!(gammas_cpu_order.len(), n);
        // Per-atom radii (vdW + probe) in CPU order.
        const PROBE_RADIUS_A: f64 = 1.4;
        let mut radii_cpu: Vec<f64> = Vec::with_capacity(n);
        for r in &structure.residues {
            for a in &r.atoms {
                let vdw = energy::powersasa::vdw_radius(a.element);
                radii_cpu.push(vdw + PROBE_RADIUS_A);
            }
        }
        // Translate to GPU order for the kernel buffers.
        let radii_gpu: Vec<f32> = (0..n)
            .map(|g| radii_cpu[self.gpu_to_cpu[g] as usize] as f32)
            .collect();
        let gammas_gpu: Vec<f32> = (0..n)
            .map(|g| gammas_cpu_order[self.gpu_to_cpu[g] as usize] as f32)
            .collect();
        self.integ.enable_sasa(SasaSmoothSetup {
            radii: &radii_gpu,
            gammas: &gammas_gpu,
            sigma_a: SASA_SMOOTH_DEFAULT_SIGMA_A,
            initial_indices_capacity: (n * 60).max(64),
        });
        self.sasa_mode = true;
        self.sasa_radii_cpu_order = radii_cpu;
        self.sasa_ref_x = vec![0.0; n];
        self.sasa_ref_y = vec![0.0; n];
        self.sasa_ref_z = vec![0.0; n];
        self.sasa_valid = false;
    }

    /// Enable the tile-based nonbonded kernel.  Must be called
    /// before the first `step_batch` / `step_batch_shake`.
    ///
    /// The integrator forwards the LJ + Coulomb parameters it
    /// already built at construction (in Morton/GPU order) into a
    /// fresh `TileNonbondedPipeline`, then routes subsequent force
    /// evaluations through it instead of the Verlet kernel.
    ///
    /// Tile interaction lists are rebuilt at every Verlet refresh
    /// from the current GPU positions.
    pub fn enable_tile_nb_mode(
        &mut self,
        structure: &Structure,
        graph: &geom::TopologyGraph,
        ff: &ForceField,
    ) {
        // Reconstruct the per-atom buffers in GPU/Morton order.  We
        // already paid this cost at construction; redoing it here
        // keeps the integrator's external API additive instead of
        // requiring `new_with_tile_nb`.
        let n = self.n_atoms;
        let atom_types = build_atom_types(structure);
        let mut charges_cpu: Vec<f32> = Vec::with_capacity(n);
        for r in &structure.residues {
            for a in &r.atoms {
                charges_cpu.push(ff.partial_charge_for(r.monomer, a.name).unwrap_or(0.0) as f32);
            }
        }
        let atom_lj_data_cpu: Vec<[f32; 4]> = atom_types
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
        let charges: Vec<f32> = (0..n)
            .map(|g| charges_cpu[self.gpu_to_cpu[g] as usize])
            .collect();
        let atom_lj_data: Vec<[f32; 4]> = (0..n)
            .map(|g| atom_lj_data_cpu[self.gpu_to_cpu[g] as usize])
            .collect();
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
            set_bit(
                &mut exclusions,
                self.cpu_to_gpu[b.a] as usize,
                self.cpu_to_gpu[b.b] as usize,
            );
        }
        for a in &graph.angles {
            set_bit(
                &mut exclusions,
                self.cpu_to_gpu[a.a] as usize,
                self.cpu_to_gpu[a.c] as usize,
            );
        }
        for d in &graph.dihedrals {
            set_bit(
                &mut one_four,
                self.cpu_to_gpu[d.a] as usize,
                self.cpu_to_gpu[d.d] as usize,
            );
        }
        self.integ.enable_tile_nb(TileNonbondedSetup {
            atom_lj_data: &atom_lj_data,
            charges: &charges,
            exclusions: &exclusions,
            one_four_mask: &one_four,
            cutoff_a: self.nb_cutoff_a,
            initial_tile_indices_capacity: (n * 16).max(64),
        });
        self.tile_nb_mode = true;
        // Force a rebuild on the next refresh.
        self.tile_count.clear();
        self.tile_start.clear();
        self.tile_indices.clear();
    }

    pub fn upload_initial_state(&mut self, structure: &Structure, velocities: &[Vec3]) {
        // CPU `structure` is in CPU index order; GPU expects Morton
        // (GPU) order.  Translate via `cpu_to_gpu`.
        let mut cpu_idx = 0;
        for r in &structure.residues {
            for a in &r.atoms {
                let g = self.cpu_to_gpu[cpu_idx] as usize;
                self.pos_buf[g] = [
                    a.position.x as f32,
                    a.position.y as f32,
                    a.position.z as f32,
                ];
                cpu_idx += 1;
            }
        }
        for (cpu_idx, v) in velocities.iter().enumerate() {
            let g = self.cpu_to_gpu[cpu_idx] as usize;
            self.vel_buf[g] = [v.x as f32, v.y as f32, v.z as f32];
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
    /// this, [`step_batch_shake`](Self::step_batch_shake) is callable.
    ///
    /// **Important**: the caller passes `shake_data` constructed from
    /// CPU-indexed atom indices (typical usage:
    /// `build_h_bond_constraints` + `build_per_x_shake_data` from
    /// `dynamics::shake`).  This method translates the per-atom
    /// tables into GPU (Morton) index space before forwarding.
    pub fn enable_shake(&mut self, shake_data: &PerXShakeData, max_iters: u32, tol_sq: f32) {
        let translated = translate_shake_data_to_gpu(shake_data, &self.cpu_to_gpu);
        self.integ.enable_shake(&translated, max_iters, tol_sq);
    }

    /// SHAKE-mode batched step.  Same drift-check + neighbour refresh
    /// machinery as [`step_batch`](Self::step_batch); calls `step_n_shake` instead of
    /// `step_n`.
    pub fn step_batch_shake(&mut self, n_steps: usize) {
        self.refresh_neighbour_lists();
        self.integ.step_n_shake(n_steps);
    }

    fn refresh_neighbour_lists(&mut self) {
        let positions_gpu_order = self.integ.download_positions();
        // Translate GPU-ordered positions back into CPU-ordered
        // scratch.xs/ys/zs — the Verlet drift detector + cell list
        // both work in CPU index space (`scratch` was built with
        // ForceScratch::new(structure, ...) which uses CPU order).
        for cpu_idx in 0..self.n_atoms {
            let g = self.cpu_to_gpu[cpu_idx] as usize;
            self.scratch.xs[cpu_idx] = positions_gpu_order[g][0] as f64;
            self.scratch.ys[cpu_idx] = positions_gpu_order[g][1] as f64;
            self.scratch.zs[cpu_idx] = positions_gpu_order[g][2] as f64;
        }
        let nb_rebuilt = ensure_verlet_list(&mut self.scratch, DEFAULT_CUTOFF_A);
        let gb_rebuilt = ensure_gb_verlet_list(&mut self.scratch, BORN_RADIUS_CUTOFF_A_PUB);
        if nb_rebuilt || self.nb_counts.is_empty() {
            if self.tile_nb_mode {
                // Tile mode: build the tile interaction list directly
                // from GPU-ordered positions.  The Verlet-list rebuild
                // above is unused in this branch, but the drift
                // detector still ran (which is what triggered us).
                let skin_a = energy::scratch::VERLET_SKIN_A as f32;
                let tile_cutoff = self.nb_cutoff_a + skin_a;
                let list = build_tile_interaction_list(&positions_gpu_order, tile_cutoff);
                self.tile_count = list.tile_count;
                self.tile_start = list.tile_start;
                self.tile_indices = list.tile_indices;
                self.integ.update_tile_nb_list(
                    &self.tile_count,
                    &self.tile_start,
                    &self.tile_indices,
                );
            } else {
                // Verlet mode: translate pairs CPU→GPU then upload CSR.
                let gpu_pairs: Vec<(u32, u32)> = self
                    .scratch
                    .verlet_pairs
                    .iter()
                    .map(|&(a, b)| (self.cpu_to_gpu[a as usize], self.cpu_to_gpu[b as usize]))
                    .collect();
                let (c, s, i) = pair_list_to_csr(self.n_atoms, &gpu_pairs);
                self.nb_counts = c;
                self.nb_starts = s;
                self.nb_indices = i;
                self.integ
                    .update_nb_neighbours(&self.nb_counts, &self.nb_starts, &self.nb_indices);
            }
        }
        if gb_rebuilt || self.gb_counts.is_empty() {
            let gpu_pairs: Vec<(u32, u32)> = self
                .scratch
                .gb_verlet_pairs
                .iter()
                .map(|&(a, b)| (self.cpu_to_gpu[a as usize], self.cpu_to_gpu[b as usize]))
                .collect();
            let (c, s, i) = pair_list_to_csr(self.n_atoms, &gpu_pairs);
            self.gb_counts = c;
            self.gb_starts = s;
            self.gb_indices = i;
            self.integ
                .update_gb_neighbours(&self.gb_counts, &self.gb_starts, &self.gb_indices);
        }
        // ---- SASA neighbour list ----
        //
        // SASA cutoff is per-pair `r_i + r_j + skin` where r is the
        // expanded vdW + probe radius (~3-6 Å total).  Much tighter
        // than the LJ (10 Å) or GB (20 Å) lists.  Use a 1 Å skin —
        // larger than the LJ skin since the SASA boundary itself is
        // sub-Å sharp.
        if self.sasa_mode {
            const SASA_SKIN_A: f64 = 1.0;
            let half_skin_sq = (0.5 * SASA_SKIN_A) * (0.5 * SASA_SKIN_A);
            let sasa_rebuilt = !self.sasa_valid || {
                let mut moved = false;
                for i in 0..self.n_atoms {
                    let dx = self.scratch.xs[i] - self.sasa_ref_x[i];
                    let dy = self.scratch.ys[i] - self.sasa_ref_y[i];
                    let dz = self.scratch.zs[i] - self.sasa_ref_z[i];
                    if dx * dx + dy * dy + dz * dz > half_skin_sq {
                        moved = true;
                        break;
                    }
                }
                moved
            };
            if sasa_rebuilt {
                // Direct O(N²) pair build — at SASA cutoffs each atom
                // has only ~10-30 neighbours, and the per-rebuild
                // wall time is small relative to the per-step force
                // eval.  Could switch to a cell list if profile
                // shows this dominating.
                let n = self.n_atoms;
                let mut per_atom_nbrs: Vec<Vec<u32>> = vec![Vec::new(); n];
                for cpu_i in 0..n {
                    let ri = self.sasa_radii_cpu_order[cpu_i];
                    for cpu_j in 0..n {
                        if cpu_i == cpu_j {
                            continue;
                        }
                        let dx = self.scratch.xs[cpu_i] - self.scratch.xs[cpu_j];
                        let dy = self.scratch.ys[cpu_i] - self.scratch.ys[cpu_j];
                        let dz = self.scratch.zs[cpu_i] - self.scratch.zs[cpu_j];
                        let r_sum = ri + self.sasa_radii_cpu_order[cpu_j] + SASA_SKIN_A;
                        if dx * dx + dy * dy + dz * dz <= r_sum * r_sum {
                            // Add the GPU-translated index — the SASA
                            // kernel walks neighbour lists in GPU
                            // space, same as nb / GB.
                            per_atom_nbrs[self.cpu_to_gpu[cpu_i] as usize]
                                .push(self.cpu_to_gpu[cpu_j]);
                        }
                    }
                }
                // Flatten into CSR (keyed on GPU atom index).
                self.sasa_counts = vec![0u32; n];
                self.sasa_starts = vec![0u32; n];
                let mut total = 0u32;
                for i in 0..n {
                    self.sasa_starts[i] = total;
                    self.sasa_counts[i] = per_atom_nbrs[i].len() as u32;
                    total += self.sasa_counts[i];
                }
                self.sasa_indices = Vec::with_capacity(total as usize);
                for list in &per_atom_nbrs {
                    self.sasa_indices.extend_from_slice(list);
                }
                self.integ.update_sasa_neighbours(
                    &self.sasa_counts,
                    &self.sasa_starts,
                    &self.sasa_indices,
                );
                // Snapshot drift ref.
                self.sasa_ref_x.copy_from_slice(&self.scratch.xs);
                self.sasa_ref_y.copy_from_slice(&self.scratch.ys);
                self.sasa_ref_z.copy_from_slice(&self.scratch.zs);
                self.sasa_valid = true;
            }
        }
    }

    /// Pull current positions from the GPU and write them back into
    /// `structure`.  Translates from GPU index space (Morton order)
    /// back to CPU order via `cpu_to_gpu`.
    pub fn download_positions_into(&self, structure: &mut Structure) {
        let positions_gpu_order = self.integ.download_positions();
        let mut cpu_idx = 0;
        for r in &mut structure.residues {
            for a in &mut r.atoms {
                let g = self.cpu_to_gpu[cpu_idx] as usize;
                a.position.x = positions_gpu_order[g][0] as f64;
                a.position.y = positions_gpu_order[g][1] as f64;
                a.position.z = positions_gpu_order[g][2] as f64;
                cpu_idx += 1;
            }
        }
    }

    /// Returns velocities in CPU index order — translated from the
    /// GPU's Morton order via `cpu_to_gpu`.
    pub fn download_velocities(&self) -> Vec<Vec3> {
        let v_gpu_order = self.integ.download_velocities();
        let mut out = Vec::with_capacity(self.n_atoms);
        for cpu_idx in 0..self.n_atoms {
            let g = self.cpu_to_gpu[cpu_idx] as usize;
            let vv = v_gpu_order[g];
            out.push(Vec3::new(vv[0] as f64, vv[1] as f64, vv[2] as f64));
        }
        out
    }

    pub fn n_atoms(&self) -> usize {
        self.n_atoms
    }
}

/// Translate a `PerXShakeData` built in CPU index space into a
/// fresh `PerXShakeData` in GPU (Morton) index space, using the
/// given permutation.  The shape (`MAX_H_PER_X` × N) is preserved;
/// only the atom indices and the row indexing change.
fn translate_shake_data_to_gpu(
    cpu_data: &gpu::PerXShakeData,
    cpu_to_gpu: &[u32],
) -> gpu::PerXShakeData {
    use gpu::MAX_H_PER_X;
    let n = cpu_data.h_count.len();
    assert_eq!(n, cpu_to_gpu.len());
    let mut h_count = vec![0u32; n];
    let mut per_atom_h_atoms = vec![0u32; n * MAX_H_PER_X];
    let mut per_atom_h_d_sq = vec![0.0f32; n * MAX_H_PER_X];
    let mut inv_mass = vec![0.0f32; n];
    for cpu_idx in 0..n {
        let g = cpu_to_gpu[cpu_idx] as usize;
        // inv_mass is per-atom — permute.
        inv_mass[g] = cpu_data.inv_mass[cpu_idx];
        // h_count is per-atom — permute.
        h_count[g] = cpu_data.h_count[cpu_idx];
        // The CSR rows: each H listed in cpu_idx's slot needs to
        // become an H listed in g's slot, with the H index itself
        // translated CPU→GPU.
        let count = cpu_data.h_count[cpu_idx] as usize;
        let cpu_base = cpu_idx * MAX_H_PER_X;
        let gpu_base = g * MAX_H_PER_X;
        for k in 0..count {
            let h_cpu = cpu_data.per_atom_h_atoms[cpu_base + k] as usize;
            let h_gpu = cpu_to_gpu[h_cpu];
            per_atom_h_atoms[gpu_base + k] = h_gpu;
            per_atom_h_d_sq[gpu_base + k] = cpu_data.per_atom_h_d_sq[cpu_base + k];
        }
    }
    gpu::PerXShakeData {
        h_count,
        per_atom_h_atoms,
        per_atom_h_d_sq,
        inv_mass,
    }
}

// ---- Builder helpers (move into a shared util if other code needs them too) ----

fn build_atom_types(s: &Structure) -> Vec<AtomType> {
    let mut out = Vec::with_capacity(s.atom_count());
    for r in &s.residues {
        for a in &r.atoms {
            out.push(
                classify_atom(r.monomer, a.name)
                    .unwrap_or_else(|| panic!("unclassified atom {:?} {}", r.monomer, a.name)),
            );
        }
    }
    out
}

fn _silence_element(_: Element) {}

fn build_bonds(
    g: &TopologyGraph,
    ff: &ForceField,
    atom_types: &[AtomType],
    cpu_to_gpu: &[u32],
    n: usize,
) -> (Vec<BondTerm>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut terms: Vec<BondTerm> = Vec::new();
    let mut per_atom: Vec<Vec<u32>> = vec![Vec::new(); n];
    for b in &g.bonds {
        // FF lookup uses CPU index (atom_types[] is CPU-ordered).
        let Some(p) = ff.bond(atom_types[b.a], atom_types[b.b]) else {
            continue;
        };
        let idx = terms.len() as u32;
        // Store GPU-translated atom indices so the kernel finds them
        // in the right slot of the reordered positions buffer.
        let ga = cpu_to_gpu[b.a];
        let gb = cpu_to_gpu[b.b];
        terms.push(BondTerm {
            a: ga,
            b: gb,
            k_kj: kcal_to_kj(p.k) as f32,
            r0_a: p.r0 as f32,
        });
        per_atom[ga as usize].push(idx);
        per_atom[gb as usize].push(idx);
    }
    let (c, s, i) = flatten_csr(per_atom, n);
    (terms, c, s, i)
}

fn build_angles(
    g: &TopologyGraph,
    ff: &ForceField,
    atom_types: &[AtomType],
    cpu_to_gpu: &[u32],
    n: usize,
) -> (Vec<AngleTerm>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut terms: Vec<AngleTerm> = Vec::new();
    let mut per_atom: Vec<Vec<u32>> = vec![Vec::new(); n];
    for a in &g.angles {
        let Some(p) = ff.angle(atom_types[a.a], atom_types[a.b], atom_types[a.c]) else {
            continue;
        };
        let idx = terms.len() as u32;
        let ga = cpu_to_gpu[a.a];
        let gb = cpu_to_gpu[a.b];
        let gc = cpu_to_gpu[a.c];
        terms.push(AngleTerm {
            a: ga,
            b: gb,
            c: gc,
            _pad: 0,
            k_kj: kcal_to_kj(p.k) as f32,
            theta0_rad: deg_to_rad(p.theta0_deg) as f32,
            _pad2: 0.0,
            _pad3: 0.0,
        });
        per_atom[ga as usize].push(idx);
        per_atom[gb as usize].push(idx);
        per_atom[gc as usize].push(idx);
    }
    let (c, s, i) = flatten_csr(per_atom, n);
    (terms, c, s, i)
}

fn build_dihedrals(
    g: &TopologyGraph,
    ff: &ForceField,
    atom_types: &[AtomType],
    cpu_to_gpu: &[u32],
    n: usize,
) -> (Vec<DihedralTerm>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut terms: Vec<DihedralTerm> = Vec::new();
    let mut per_atom: Vec<Vec<u32>> = vec![Vec::new(); n];
    for d in &g.dihedrals {
        let Some(pterms) = ff.dihedral(
            atom_types[d.a],
            atom_types[d.b],
            atom_types[d.c],
            atom_types[d.d],
        ) else {
            continue;
        };
        let ga = cpu_to_gpu[d.a];
        let gb = cpu_to_gpu[d.b];
        let gc = cpu_to_gpu[d.c];
        let gd = cpu_to_gpu[d.d];
        // CHARMM27 nucleic-acid dihedrals around the phosphodiester
        // backbone (e.g. CN7-CN7-ON2-Pn at the α/ζ torsion) carry up
        // to 5 periodic terms — and CHARMM36 protein params can in
        // principle exceed 4 too.  The GPU kernel packs a fixed 4 terms
        // per `DihedralTerm`, so split a >4-term dihedral into multiple
        // GPU records on the same atom 4-tuple; the kernel sums the
        // contributions just like multiple independent dihedrals.
        for chunk in pterms.chunks(4) {
            let mut packed = DihedralTerm {
                a: ga,
                b: gb,
                c: gc,
                d: gd,
                n_terms: chunk.len() as u32,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
                term0: zero_term(),
                term1: zero_term(),
                term2: zero_term(),
                term3: zero_term(),
            };
            for (i, t) in chunk.iter().enumerate() {
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
            per_atom[ga as usize].push(idx);
            per_atom[gb as usize].push(idx);
            per_atom[gc as usize].push(idx);
            per_atom[gd as usize].push(idx);
        }
    }
    let (c, s, i) = flatten_csr(per_atom, n);
    (terms, c, s, i)
}

fn build_impropers(
    g: &TopologyGraph,
    ff: &ForceField,
    atom_types: &[AtomType],
    cpu_to_gpu: &[u32],
    n: usize,
) -> (Vec<ImproperTerm>, Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut terms: Vec<ImproperTerm> = Vec::new();
    let mut per_atom: Vec<Vec<u32>> = vec![Vec::new(); n];
    for imp in &g.impropers {
        let Some(p) = ff.improper(
            atom_types[imp.a],
            atom_types[imp.b],
            atom_types[imp.c],
            atom_types[imp.d],
        ) else {
            continue;
        };
        let idx = terms.len() as u32;
        let ga = cpu_to_gpu[imp.a];
        let gb = cpu_to_gpu[imp.b];
        let gc = cpu_to_gpu[imp.c];
        let gd = cpu_to_gpu[imp.d];
        terms.push(ImproperTerm {
            a: ga,
            b: gb,
            c: gc,
            d: gd,
            k_kj: kcal_to_kj(p.k) as f32,
            omega0_rad: deg_to_rad(p.psi0_deg) as f32,
            _pad0: 0.0,
            _pad1: 0.0,
        });
        per_atom[ga as usize].push(idx);
        per_atom[gb as usize].push(idx);
        per_atom[gc as usize].push(idx);
        per_atom[gd as usize].push(idx);
    }
    let (c, s, i) = flatten_csr(per_atom, n);
    (terms, c, s, i)
}

fn zero_term() -> PeriodicTerm {
    PeriodicTerm {
        k_kj: 0.0,
        n: 0.0,
        delta_rad: 0.0,
        _pad: 0.0,
    }
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
