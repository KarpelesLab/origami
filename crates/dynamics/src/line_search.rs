//! Backtracking Armijo line search.
//!
//! Given a descent direction p and current state (x₀, E₀, g₀), find α
//! such that E(x₀ + α p) ≤ E₀ + c₁ α (p · g₀). Halve α until satisfied or
//! a step-size limit is hit.

use chem::ForceField;
use geom::{Structure, TopologyGraph};

use crate::energy_eval::{apply_displacement, total_energy_with_options};

#[derive(Debug, Clone, Copy)]
pub struct LineSearchOptions {
    /// Initial step length.
    pub alpha0: f64,
    /// Contraction factor on each backtrack.
    pub contraction: f64,
    /// Armijo c₁ constant.
    pub c1: f64,
    /// Hard floor on α — give up if it drops below this.
    pub min_alpha: f64,
    /// Include SASA in the energy evaluated along the search direction
    /// (PSA.2). The optimiser's gradient and the line search must agree
    /// on whether SASA is in or out, so this propagates from
    /// `MinimizeOptions::include_sasa`.
    pub include_sasa: bool,
}

impl Default for LineSearchOptions {
    fn default() -> Self {
        LineSearchOptions {
            alpha0: 1.0,
            contraction: 0.5,
            c1: 1e-4,
            min_alpha: 1e-12,
            include_sasa: false,
        }
    }
}

#[derive(Debug)]
pub struct LineSearchResult {
    pub alpha: f64,
    pub new_energy: f64,
    pub iterations: usize,
}

/// Run backtracking line search. Returns the chosen `alpha` and the new
/// energy at the accepted point. The caller is responsible for applying the
/// step (this routine restores the structure to its pre-call state on exit
/// regardless of success or failure).
pub fn backtracking(
    structure: &mut Structure,
    graph: &TopologyGraph,
    ff: &ForceField,
    direction_flat: &[f64],
    gradient_flat: &[f64],
    e0: f64,
    options: LineSearchOptions,
) -> Option<LineSearchResult> {
    let p_dot_g: f64 = direction_flat
        .iter()
        .zip(gradient_flat.iter())
        .map(|(p, g)| p * g)
        .sum();
    if p_dot_g >= 0.0 {
        // Not a descent direction.
        return None;
    }
    let mut alpha = options.alpha0;
    let mut step = vec![0.0_f64; direction_flat.len()];
    let mut iters = 0usize;
    while alpha >= options.min_alpha {
        for (s, p) in step.iter_mut().zip(direction_flat.iter()) {
            *s = alpha * p;
        }
        apply_displacement(structure, &step);
        let e_new = total_energy_with_options(structure, graph, ff, options.include_sasa);
        // Undo the step so the caller sees the structure unchanged.
        for s in step.iter_mut() {
            *s = -*s;
        }
        apply_displacement(structure, &step);
        // Restore original sign for next iteration's logic.
        for s in step.iter_mut() {
            *s = -*s;
        }
        iters += 1;

        let armijo_target = e0 + options.c1 * alpha * p_dot_g;
        if e_new <= armijo_target {
            return Some(LineSearchResult {
                alpha,
                new_energy: e_new,
                iterations: iters,
            });
        }
        alpha *= options.contraction;
    }
    None
}
