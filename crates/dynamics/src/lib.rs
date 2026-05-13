//! Energy minimization for origami.
//!
//! Two algorithms in M4: steepest descent with backtracking line search
//! (the foundation, robust on rough force-field surfaces), and L-BFGS for
//! production-quality convergence. Both share the same line search and
//! convergence criteria.

pub mod cotranslate;
pub mod energy_eval;
pub mod langevin;
pub mod lbfgs;
pub mod line_search;
pub mod minimize;
pub mod remd;
pub mod rng;
pub mod shake;
pub mod steepest_descent;

pub use cotranslate::{
    run_cotranslate, CotranslateFrame, CylindricalTunnel, ExternalPotential, Ribosome,
    UniformRibosome,
};
pub use langevin::{
    initialise_velocities_for_new_atoms, instant_temperature_k, run_langevin, LangevinFrame,
    LangevinOptions, LangevinSummary, BOLTZMANN_KJ_PER_MOL_K,
};
pub use minimize::{minimize, Algorithm, MinimizationResult, MinimizeOptions};
pub use rng::Xoshiro256pp;
