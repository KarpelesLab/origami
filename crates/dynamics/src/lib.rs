//! Energy minimization for origami.
//!
//! Two algorithms in M4: steepest descent with backtracking line search
//! (the foundation, robust on rough force-field surfaces), and L-BFGS for
//! production-quality convergence. Both share the same line search and
//! convergence criteria.

pub mod energy_eval;
pub mod lbfgs;
pub mod line_search;
pub mod minimize;
pub mod steepest_descent;

pub use minimize::{minimize, Algorithm, MinimizationResult, MinimizeOptions};
