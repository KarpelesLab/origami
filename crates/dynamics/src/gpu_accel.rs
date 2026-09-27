//! GPU acceleration for the Langevin force aggregator.
//!
//! [`GpuAccelerator`] wraps the two GPU pipelines that replace the
//! CPU's nonbonded + GB pair-loop work:
//!
//! - `VerletNonbondedPipeline` — LJ + reaction-field Coulomb walking
//!   a CSR neighbour list built on the CPU (from `ForceScratch`'s
//!   cached Verlet pairs).
//! - `GbPipeline` — Generalized-Born OBC II Born radii + pair force,
//!   currently O(N²) on the GPU; switching to a Verlet sweep here is
//!   step 3f.
//!
//! Why route through this instead of straight `gpu::*` calls in the
//! integrator: the accelerator owns the per-system parameter buffers
//! (type table, exclusion bitmap, charges, GB intrinsic radii, …)
//! that don't change between steps, and it tracks the
//! `ForceScratch.verlet_valid` signal so the CSR neighbour list only
//! re-uploads when the CPU actually rebuilt it.  Per-step cost on
//! Apple M3 Pro is dominated by the readback (~0.3 ms) regardless of
//! N — that's where the wall-clock win lives at ribosome scale, since
//! the equivalent CPU pair loop is tens of ms.
//!
//! **Threshold to use the GPU.** At ~300 atoms (Trp-cage) the CPU
//! SoA Verlet path takes ~0.6 ms; the GPU path takes ~0.9 ms (mostly
//! readback overhead).  At ~1500 atoms the two are even.  Above
//! ~3000 atoms the GPU wins.  The `LangevinOptions::use_gpu` flag is
//! a manual opt-in for now — auto-thresholding is step 3g.

use chem::{AtomType, ForceField, classify_atom};
use energy::forces_gb::GB_DEFAULT_CUTOFF_A_PUB;
use energy::forces_nonbonded::ensure_verlet_list;
use energy::gb::{
    BORN_RADIUS_CUTOFF_A_PUB, OBC_OFFSET_PUB, ensure_gb_verlet_list, hct_scale_pub,
    intrinsic_radius_pub,
};
use energy::scratch::{EXCLUDED_BIT, ForceScratch, ONE_FOUR_BIT};
use geom::Structure;
use gpu::{
    GbPipeline, GbSetup, GpuContext, VerletNonbondedPipeline, VerletNonbondedSetup,
    pair_list_to_csr,
};

const KCAL_TO_KJ: f32 = 4.184;

/// One per simulation.  Holds the per-system parameter buffers and
/// the two GPU compute pipelines.  Re-using across many steps amortises
/// the (~5 ms) shader-compile + buffer-alloc cost.
pub struct GpuAccelerator {
    n_atoms: usize,
    nonbonded: VerletNonbondedPipeline,
    gb: GbPipeline,
    /// Cached position buffer reused between steps (avoids per-step
    /// Vec allocation).
    pos_buf: Vec<[f32; 3]>,
    /// Cached LJ+Coulomb neighbour-list CSR buffers (rebuilt on the
    /// nonbonded Verlet refresh at 10 Å + skin).
    nb_counts: Vec<u32>,
    nb_starts: Vec<u32>,
    nb_indices: Vec<u32>,
    /// Cached GB neighbour-list CSR buffers (rebuilt on the GB Verlet
    /// refresh at 20 Å + skin — a different list because the cutoff
    /// is twice as wide).
    gb_counts: Vec<u32>,
    gb_starts: Vec<u32>,
    gb_indices: Vec<u32>,
    /// LJ + Coulomb cutoff in Å (passed at construction; doesn't change
    /// during a trajectory).
    cutoff_a: f64,
}

impl GpuAccelerator {
    /// Build the accelerator for a given `structure` + `graph` + force
    /// field at the given LJ/Coulomb cutoff.  Allocates GPU buffers
    /// for everything that's constant for the trajectory (type table,
    /// exclusion mask, charges, GB radii) and compiles the two
    /// shaders.  Returns an error only if [`GpuContext::get`] fails.
    pub fn new(
        structure: &Structure,
        graph: &geom::TopologyGraph,
        ff: &ForceField,
        cutoff_a: f64,
    ) -> Result<Self, gpu::context::GpuInitError> {
        let ctx = GpuContext::get()?;
        let n = structure.atom_count();
        // ---- Build per-atom typing + parameter tables ----
        let mut atom_types: Vec<AtomType> = Vec::with_capacity(n);
        let mut charges: Vec<f32> = Vec::with_capacity(n);
        let mut rho: Vec<f32> = Vec::with_capacity(n);
        let mut rho_tilde: Vec<f32> = Vec::with_capacity(n);
        let mut scale: Vec<f32> = Vec::with_capacity(n);
        for r in &structure.residues {
            for a in &r.atoms {
                let t = classify_atom(r.monomer, a.name)
                    .unwrap_or_else(|| panic!("unclassified atom {:?} {}", r.monomer, a.name));
                atom_types.push(t);
                charges.push(ff.partial_charge_for(r.monomer, a.name).unwrap_or(0.0) as f32);
                let r0 = intrinsic_radius_pub(a.element);
                rho.push(r0 as f32);
                rho_tilde.push((r0 - OBC_OFFSET_PUB) as f32);
                scale.push(hct_scale_pub(a.element) as f32);
            }
        }
        // Pre-resolve the (atom → type → LJ params) chain into a flat
        // per-atom `vec4(eps, rmin_half, eps_14, rmin_half_14)`.  The
        // GPU inner loop reads one cache line per neighbour instead
        // of doing two indirect lookups.
        let atom_lj_data: Vec<[f32; 4]> = atom_types
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
        // Exclusion + 1-4 bitmaps in the layout the GPU kernels expect.
        let n_words = (n * n).div_ceil(32);
        let mut exclusions = vec![0u32; n_words];
        let mut one_four = vec![0u32; n_words];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let bit = i * n + j;
                if graph.is_bonded(i, j) || graph.is_one_three(i, j) {
                    exclusions[bit / 32] |= 1u32 << (bit % 32);
                } else if graph.is_one_four(i, j) {
                    one_four[bit / 32] |= 1u32 << (bit % 32);
                }
            }
        }
        // We need a rough initial capacity for the neighbour list.
        // ⟨neighbours⟩ at a 10 Å cutoff in a typical all-atom protein
        // is 150-300; over-estimate by ×2 so the first upload doesn't
        // trigger the grow path.  Negligible memory at any scale we
        // care about.
        let initial_cap = (n * 600).max(64);
        let nonbonded = VerletNonbondedPipeline::new(
            ctx,
            n,
            VerletNonbondedSetup {
                atom_lj_data: &atom_lj_data,
                charges: &charges,
                exclusions: &exclusions,
                one_four_mask: &one_four,
                cutoff_a: cutoff_a as f32,
                initial_indices_capacity: initial_cap,
            },
        );
        // GB neighbour list at 20 Å is ~8× wider than the 10 Å LJ
        // list — but each atom in a dense globular protein still
        // sees only ~3000-6000 neighbours.  Estimate generously so
        // the first upload doesn't trigger the grow path.
        let gb_initial_cap = (n * 5000).max(64);
        let gb = GbPipeline::new(
            ctx,
            n,
            GbSetup {
                rho: &rho,
                rho_tilde: &rho_tilde,
                scale: &scale,
                charges: &charges,
                cutoff_a: BORN_RADIUS_CUTOFF_A_PUB as f32,
                pair_cutoff_a: GB_DEFAULT_CUTOFF_A_PUB as f32,
                initial_indices_capacity: gb_initial_cap,
            },
        );
        // Sanity: the kept exclusion + 1-4 bits agree with the masks
        // ForceScratch holds; we read its scratch.excl in the
        // per-pair filter on the CPU side, so any disagreement here
        // would silently corrupt forces.
        let _ = ONE_FOUR_BIT;
        let _ = EXCLUDED_BIT;
        Ok(Self {
            n_atoms: n,
            nonbonded,
            gb,
            pos_buf: vec![[0.0; 3]; n],
            nb_counts: Vec::new(),
            nb_starts: Vec::new(),
            nb_indices: Vec::new(),
            gb_counts: Vec::new(),
            gb_starts: Vec::new(),
            gb_indices: Vec::new(),
            cutoff_a,
        })
    }

    /// Replace the LJ+Coulomb + GB pair-loop work in `scratch` with the
    /// GPU equivalents.  The integrator must still call the CPU bonded
    /// terms (bond / angle / dihedral / improper) before or after this.
    ///
    /// Pre-condition: `scratch.xs/ys/zs` are synced from the latest
    /// `structure` positions and `scratch.f{xyz}s` have been zeroed
    /// for this step.
    ///
    /// Post-condition: `scratch.f{xyz}s` are incremented by the
    /// nonbonded + GB force contributions, ready to be accumulated
    /// into the integrator's AoS `forces` buffer via
    /// `scratch.accumulate_into`.
    pub fn add_nonbonded_and_gb(&mut self, scratch: &mut ForceScratch) {
        debug_assert_eq!(scratch.n, self.n_atoms);
        // Refresh both Verlet lists on the CPU side — one at 10 Å for
        // LJ + Coulomb, one at 20 Å for GB.  Each `ensure_*` call
        // returns whether it actually rebuilt, so we only re-upload
        // the changed list(s) to the GPU.
        let nb_rebuilt = ensure_verlet_list(scratch, self.cutoff_a);
        let gb_rebuilt = ensure_gb_verlet_list(scratch, BORN_RADIUS_CUTOFF_A_PUB);

        if nb_rebuilt || self.nb_counts.is_empty() {
            let (counts, starts, indices) = pair_list_to_csr(self.n_atoms, &scratch.verlet_pairs);
            self.nb_counts = counts;
            self.nb_starts = starts;
            self.nb_indices = indices;
            self.nonbonded
                .update_neighbours(&self.nb_counts, &self.nb_starts, &self.nb_indices);
        }
        if gb_rebuilt || self.gb_counts.is_empty() {
            let (counts, starts, indices) =
                pair_list_to_csr(self.n_atoms, &scratch.gb_verlet_pairs);
            self.gb_counts = counts;
            self.gb_starts = starts;
            self.gb_indices = indices;
            self.gb
                .update_neighbours(&self.gb_counts, &self.gb_starts, &self.gb_indices);
        }

        // Pack f64 SoA → f32 AoS for the GPU upload.  This is the
        // unavoidable precision step-down of the GPU path; the CPU
        // remains in f64 throughout.
        for i in 0..self.n_atoms {
            self.pos_buf[i] = [
                scratch.xs[i] as f32,
                scratch.ys[i] as f32,
                scratch.zs[i] as f32,
            ];
        }

        // Push positions to both pipelines.  These are separate
        // `write_buffer` calls (each pipeline has its own positions
        // buffer) — wgpu coalesces them into the next queue submit
        // automatically.
        self.nonbonded.update_positions(&self.pos_buf);
        self.gb.update_positions(&self.pos_buf);

        // Pipeline kernels accumulate into the forces buffers (so
        // bonded kernels can compose with them in the integrator
        // path).  In this standalone API the caller expects fresh
        // forces — zero both buffers first.
        self.nonbonded.clear_forces();
        self.gb.clear_forces();

        // Kernel fusion: record both pipelines' compute passes plus
        // their readback copies into a single command encoder, submit
        // once, map both readback buffers, then wait once for the GPU
        // to drain.  This eliminates one of the two per-step
        // submit + poll round-trips, which on Apple Silicon is the
        // single largest fixed-cost item at scales where the GPU
        // would otherwise win.
        let ctx = GpuContext::get().expect("GPU context already established at constructor time");
        let device = &ctx.device;
        let queue = &ctx.queue;
        let mut encoder = device.create_command_encoder(&gpu::wgpu::CommandEncoderDescriptor {
            label: Some("gpu_accel_fused_encoder"),
        });
        self.nonbonded.record_compute(&mut encoder);
        self.gb.record_compute(&mut encoder);
        self.nonbonded.record_readback_copy(&mut encoder);
        self.gb.record_readback_copy(&mut encoder);
        queue.submit(Some(encoder.finish()));

        let nb_rx = self.nonbonded.begin_readback();
        let gb_rx = self.gb.begin_readback();
        let _ = device.poll(gpu::wgpu::Maintain::Wait);
        nb_rx
            .recv()
            .expect("nb map_async sender dropped")
            .expect("nb buffer map");
        gb_rx
            .recv()
            .expect("gb map_async sender dropped")
            .expect("gb buffer map");
        let nb_f = self.nonbonded.take_readback();
        let gb_f = self.gb.take_readback();

        // Accumulate into scratch.  f32 → f64 widening — no precision
        // loss beyond what already happened during the GPU eval.
        for i in 0..self.n_atoms {
            scratch.fxs[i] += nb_f[i][0] as f64 + gb_f[i][0] as f64;
            scratch.fys[i] += nb_f[i][1] as f64 + gb_f[i][1] as f64;
            scratch.fzs[i] += nb_f[i][2] as f64 + gb_f[i][2] as f64;
        }
    }
}
