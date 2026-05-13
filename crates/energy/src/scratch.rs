//! `ForceScratch`: caller-supplied scratch buffer that caches every
//! piece of per-force-call work that doesn't change across the lifetime
//! of a simulation (or doesn't change between successive steps).
//!
//! Used by the SoA-aware force kernels — the integrator allocates one
//! `ForceScratch`, syncs positions from the `Structure` once per step,
//! and the kernels read from flat `[f64]` arrays instead of paying the
//! `Vec3` AoS price on every load. The exclusion masks and per-atom
//! force-field parameters are populated once at the start of the
//! simulation (or whenever atoms are added, as in cotranslate) so the
//! pair-loop hot path is pure arithmetic with no `HashMap` /
//! `Vec::contains` lookups.
//!
//! # Layout
//!
//! All flat arrays have length `n_atoms`:
//!   - `xs, ys, zs`           — positions in SoA form
//!   - `fxs, fys, fzs`        — force accumulators in SoA form
//!   - `charges`              — partial charges (e)
//!   - `rmin_half, epsilon`   — CHARMM LJ params, full strength
//!   - `rmin_half_14, epsilon_14` — CHARMM LJ params for 1-4 pairs
//!
//! Exclusion masks are flat `Vec<u8>` of length `n_atoms²`, indexed by
//! `i * n + j`. Two bits per entry:
//!   - bit 0: excluded (1-2 or 1-3 pair, skip non-bonded)
//!   - bit 1: 1-4 pair (apply scaled LJ params + scaled Coulomb)

use chem::{classify, AminoAcid, AtomType, ForceField};
use geom::{Structure, TopologyGraph};

pub const EXCLUDED_BIT: u8 = 1 << 0;
pub const ONE_FOUR_BIT: u8 = 1 << 1;

#[derive(Debug, Clone)]
pub struct ForceScratch {
    pub n: usize,
    // Positions (SoA).
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    pub zs: Vec<f64>,
    // Force accumulators (SoA).
    pub fxs: Vec<f64>,
    pub fys: Vec<f64>,
    pub fzs: Vec<f64>,
    // Per-atom force-field params (kcal/mol/Å units inside, matching the
    // existing nonbonded code).
    pub charges: Vec<f64>,
    pub rmin_half: Vec<f64>,
    pub epsilon: Vec<f64>,
    pub rmin_half_14: Vec<f64>,
    pub epsilon_14: Vec<f64>,
    pub atom_types: Vec<AtomType>,
    /// Flat exclusion mask, n² bytes, indexed `i * n + j`.
    pub excl: Vec<u8>,
    /// Per-thread force accumulators for the rayon-parallel pair loop.
    /// Each entry is `(fx, fy, fz)` of length `n`. Allocated once at
    /// `new` (one entry per rayon worker) and zeroed at the start of
    /// each parallel force call instead of being re-allocated, which
    /// was previously the dominant overhead per `add_nonbonded_forces_soa`
    /// call. Stored as flat `n_threads × n` contiguous Vecs so each
    /// worker can grab a disjoint slice via `chunks_mut`.
    pub par_fx: Vec<f64>,
    pub par_fy: Vec<f64>,
    pub par_fz: Vec<f64>,
    pub n_par_threads: usize,
}

impl ForceScratch {
    /// Build a scratch for the given structure + topology + force field.
    /// Allocates everything; this is the one expensive call.
    pub fn new(structure: &Structure, graph: &TopologyGraph, ff: &ForceField) -> Self {
        let n = structure.atom_count();
        let n_threads = rayon::current_num_threads().max(1);
        let mut s = Self {
            n,
            xs: vec![0.0; n],
            ys: vec![0.0; n],
            zs: vec![0.0; n],
            fxs: vec![0.0; n],
            fys: vec![0.0; n],
            fzs: vec![0.0; n],
            charges: vec![0.0; n],
            rmin_half: vec![0.0; n],
            epsilon: vec![0.0; n],
            rmin_half_14: vec![0.0; n],
            epsilon_14: vec![0.0; n],
            atom_types: Vec::with_capacity(n),
            excl: vec![0u8; n * n],
            par_fx: vec![0.0; n_threads * n],
            par_fy: vec![0.0; n_threads * n],
            par_fz: vec![0.0; n_threads * n],
            n_par_threads: n_threads,
        };
        s.rebuild_params(structure, ff);
        s.rebuild_exclusions(graph);
        s.sync_positions(structure);
        s
    }

    /// Re-populate per-atom params from the structure + force field.
    /// Call this once at construction and again when atoms are added/
    /// removed (cotranslate). Most simulations call it once.
    pub fn rebuild_params(&mut self, structure: &Structure, ff: &ForceField) {
        let n = structure.atom_count();
        self.atom_types.clear();
        self.atom_types.reserve(n);
        let mut idx = 0;
        for residue in &structure.residues {
            for atom in &residue.atoms {
                let ty = classify(residue.aa, atom.name).unwrap_or_else(|| {
                    panic!("unclassified atom {:?} {}", residue.aa, atom.name)
                });
                self.atom_types.push(ty);
                let q = charge_for(ff, residue.aa, atom.name);
                self.charges[idx] = q;
                if let Some(p) = ff.nonbonded(ty) {
                    self.rmin_half[idx] = p.rmin_half;
                    self.epsilon[idx] = p.epsilon;
                    self.rmin_half_14[idx] = p.rmin_half_14.unwrap_or(p.rmin_half);
                    self.epsilon_14[idx] = p.epsilon_14.unwrap_or(p.epsilon);
                } else {
                    self.rmin_half[idx] = 0.0;
                    self.epsilon[idx] = 0.0;
                    self.rmin_half_14[idx] = 0.0;
                    self.epsilon_14[idx] = 0.0;
                }
                idx += 1;
            }
        }
    }

    /// Populate the flat exclusion mask from the topology graph. O(n²)
    /// but only runs once per simulation (or when topology changes).
    pub fn rebuild_exclusions(&mut self, graph: &TopologyGraph) {
        let n = self.n;
        self.excl.fill(0);
        for i in 0..n {
            for j in (i + 1)..n {
                let mask = if graph.is_bonded(i, j) || graph.is_one_three(i, j) {
                    EXCLUDED_BIT
                } else if graph.is_one_four(i, j) {
                    ONE_FOUR_BIT
                } else {
                    0
                };
                if mask != 0 {
                    self.excl[i * n + j] = mask;
                    self.excl[j * n + i] = mask;
                }
            }
        }
    }

    /// Copy positions from the structure into the SoA arrays. Cheap;
    /// call once per force evaluation. Zeroing of the force buffers is
    /// separate (see `zero_forces`).
    pub fn sync_positions(&mut self, structure: &Structure) {
        let mut idx = 0;
        for residue in &structure.residues {
            for atom in &residue.atoms {
                self.xs[idx] = atom.position.x;
                self.ys[idx] = atom.position.y;
                self.zs[idx] = atom.position.z;
                idx += 1;
            }
        }
    }

    /// Zero the force accumulators in-place.
    pub fn zero_forces(&mut self) {
        self.fxs.fill(0.0);
        self.fys.fill(0.0);
        self.fzs.fill(0.0);
    }

    #[inline]
    pub fn is_excluded(&self, i: usize, j: usize) -> bool {
        (self.excl[i * self.n + j] & EXCLUDED_BIT) != 0
    }

    #[inline]
    pub fn is_one_four(&self, i: usize, j: usize) -> bool {
        (self.excl[i * self.n + j] & ONE_FOUR_BIT) != 0
    }

    /// Accumulate the SoA force buffer into the given Vec3 buffer (for
    /// the existing integrator path that consumes `&mut [Vec3]`).
    pub fn accumulate_into(&self, forces: &mut [geom::Vec3]) {
        for (k, f) in forces.iter_mut().enumerate() {
            f.x += self.fxs[k];
            f.y += self.fys[k];
            f.z += self.fzs[k];
        }
    }
}

/// Helper mirroring the partial-charge lookup used elsewhere.
fn charge_for(ff: &ForceField, aa: AminoAcid, atom_name: &str) -> f64 {
    ff.partial_charge(aa, atom_name).unwrap_or(0.0)
}
