//! Energy minimization for origami.
//!
//! Two algorithms in M4: steepest descent with backtracking line search
//! (the foundation, robust on rough force-field surfaces), and L-BFGS for
//! production-quality convergence. Both share the same line search and
//! convergence criteria.

pub mod cotranslate;
pub mod energy_eval;
pub mod full_gpu_integrator;
pub mod gpu_accel;
pub mod langevin;
pub mod lbfgs;
pub mod line_search;
pub mod minimize;
pub mod remd;
pub mod rng;
pub mod shake;
pub mod steepest_descent;

pub use full_gpu_integrator::FullGpuIntegrator;
pub use gpu_accel::GpuAccelerator;

pub use cotranslate::{
    CodonPacedRibosome, CotranslateFrame, CylindricalTunnel, ExternalPotential, MrnaParseError,
    Ribosome, UniformRibosome, run_cotranslate,
};
pub use langevin::{
    BOLTZMANN_KJ_PER_MOL_K, LangevinFrame, LangevinOptions, LangevinSummary,
    initialise_velocities_for_new_atoms, instant_temperature_k, run_langevin,
};
pub use minimize::{Algorithm, MinimizationResult, MinimizeOptions, minimize};
pub use rng::Xoshiro256pp;
