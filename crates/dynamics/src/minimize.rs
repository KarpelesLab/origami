//! Top-level driver that chooses an algorithm and runs the minimisation.

use chem::ForceField;
use geom::{Structure, TopologyGraph};

use crate::lbfgs::{lbfgs, LbfgsOptions, LbfgsResult};
use crate::steepest_descent::{steepest_descent, SdOptions, SdResult};

#[derive(Debug, Clone, Copy)]
pub enum Algorithm {
    SteepestDescent,
    Lbfgs,
}

#[derive(Debug, Clone, Copy)]
pub struct MinimizeOptions {
    pub algorithm: Algorithm,
    pub max_steps: usize,
    pub gradient_tol: f64,
    pub energy_tol: f64,
    pub max_step_a: f64,
}

impl Default for MinimizeOptions {
    fn default() -> Self {
        MinimizeOptions {
            algorithm: Algorithm::Lbfgs,
            max_steps: 500,
            gradient_tol: 1.0,
            energy_tol: 0.01,
            max_step_a: 0.1,
        }
    }
}

#[derive(Debug)]
pub struct MinimizationResult {
    pub algorithm: Algorithm,
    pub steps: usize,
    pub initial_energy: f64,
    pub final_energy: f64,
    pub max_force: f64,
    pub converged: bool,
}

impl From<SdResult> for MinimizationResult {
    fn from(r: SdResult) -> MinimizationResult {
        MinimizationResult {
            algorithm: Algorithm::SteepestDescent,
            steps: r.steps,
            initial_energy: r.initial_energy,
            final_energy: r.final_energy,
            max_force: r.max_force,
            converged: r.converged,
        }
    }
}

impl From<LbfgsResult> for MinimizationResult {
    fn from(r: LbfgsResult) -> MinimizationResult {
        MinimizationResult {
            algorithm: Algorithm::Lbfgs,
            steps: r.steps,
            initial_energy: r.initial_energy,
            final_energy: r.final_energy,
            max_force: r.max_force,
            converged: r.converged,
        }
    }
}

pub fn minimize(
    structure: &mut Structure,
    graph: &TopologyGraph,
    ff: &ForceField,
    opts: MinimizeOptions,
) -> MinimizationResult {
    match opts.algorithm {
        Algorithm::SteepestDescent => {
            let sd_opts = SdOptions {
                max_steps: opts.max_steps,
                gradient_tol: opts.gradient_tol,
                energy_tol: opts.energy_tol,
                max_step_a: opts.max_step_a,
                ..Default::default()
            };
            steepest_descent(structure, graph, ff, sd_opts).into()
        }
        Algorithm::Lbfgs => {
            let lbfgs_opts = LbfgsOptions {
                max_steps: opts.max_steps,
                gradient_tol: opts.gradient_tol,
                energy_tol: opts.energy_tol,
                max_step_a: opts.max_step_a,
                ..Default::default()
            };
            lbfgs(structure, graph, ff, lbfgs_opts).into()
        }
    }
}
