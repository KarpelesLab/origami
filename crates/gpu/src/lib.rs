//! GPU-accelerated force kernels for origami.
//!
//! First slice (this commit): a single WGSL compute kernel for the
//! Lennard-Jones pair force with a bare cutoff and the standard
//! 1-2 / 1-3 / 1-4 exclusion bitmap.  No Coulomb, no GB, no SASA
//! yet; CHARMM 1-4 LJ special parameters not yet supported.
//!
//! ## Why first slice
//!
//! The full Langevin step on Trp-cage (300 atoms) takes ~0.7 ms
//! with the SoA + Verlet CPU path.  Of that, the LJ + Coulomb
//! nonbonded pair loop is ~0.5 ms.  GPU dispatch overhead is
//! ~100 µs per kernel launch (one launch per force eval), so for
//! Trp-cage the GPU win is marginal at best.
//!
//! The long-horizon target is the *ribosome* (200 000 atoms).
//! There, the CPU nonbonded loop is O(N) with the Verlet list
//! but still tens of milliseconds per step, while a GPU kernel
//! with O(N) threads can finish in <1 ms.  This crate is the
//! foundation for getting there.
//!
//! ## What's here
//!
//! - [`GpuContext`] — lazy wgpu device/queue init (Metal on Mac,
//!   Vulkan elsewhere via wgpu's backend fallback).
//! - [`lj`] — the LJ kernel + its CPU-callable wrapper.

pub mod baoab;
pub mod bonded;
pub mod context;
pub mod gb;
pub mod integrator;
pub mod lj;
pub mod nonbonded;
pub mod nonbonded_verlet;
pub mod sasa;
pub mod shake;
pub mod spatial_sort;
pub mod tile_list;
pub mod tile_nonbonded;

pub use baoab::{make_rng_state, BaoabPipeline};
pub use bonded::{
    AngleTerm, BondTerm, BondedPipeline, BondedSetup, DihedralTerm, ImproperTerm, PeriodicTerm,
};
pub use context::GpuContext;
pub use gb::{GbPipeline, GbSetup};
pub use integrator::IntegratorPipeline;
pub use nonbonded::{nonbonded_force_gpu, NonbondedPipeline, NonbondedSetup};
pub use nonbonded_verlet::{pair_list_to_csr, VerletNonbondedPipeline, VerletNonbondedSetup};
pub use sasa::{fibonacci_unit_sphere_f32, SasaPipeline, SasaSetup, SASA_N_DOTS};
pub use shake::{build_per_x_shake_data, PerXShakeData, ShakeConstraint, ShakePipeline, MAX_H_PER_X};
pub use spatial_sort::{morton_code, morton_permutation, TILE_SIZE};
pub use tile_list::{build_tile_interaction_list, TileInteractionList};
pub use tile_nonbonded::{TileNonbondedPipeline, TileNonbondedSetup};
/// Re-export the wgpu crate so downstream consumers (e.g.
/// `dynamics::GpuAccelerator`) can record into our pipelines'
/// command encoders without needing a separate `wgpu` dependency.
pub use wgpu;
