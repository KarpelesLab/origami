//! Limited-memory BFGS minimization.
//!
//! Algorithm (Nocedal & Wright §7.2):
//!   1. Maintain rolling history of `m` (sₖ, yₖ) pairs where sₖ = xₖ₊₁ − xₖ
//!      and yₖ = gₖ₊₁ − gₖ.
//!   2. Each step: search direction p = −H⁻¹ g, computed by the two-loop
//!      recursion using the history. With no history, fall back to p = −g
//!      (steepest descent).
//!   3. Backtracking Armijo line search along p.
//!
//! Same convergence criteria as SD: max gradient component < `gradient_tol`
//! AND |ΔE| < `energy_tol`.

use std::collections::VecDeque;

use chem::ForceField;
use geom::{Structure, TopologyGraph};

use crate::energy_eval::{
    apply_displacement, flatten_vec3, linf_norm, total_energy_with_options, total_force_opts,
};
use crate::line_search::{backtracking, LineSearchOptions};

#[derive(Debug, Clone, Copy)]
pub struct LbfgsOptions {
    pub max_steps: usize,
    pub gradient_tol: f64,
    pub energy_tol: f64,
    pub line_search: LineSearchOptions,
    pub max_step_a: f64,
    /// History size — number of (s, y) pairs to retain.
    pub history: usize,
    /// Include SASA in energy + forces (PSA.2). Slow but smooth.
    pub include_sasa: bool,
}

impl Default for LbfgsOptions {
    fn default() -> Self {
        LbfgsOptions {
            max_steps: 500,
            gradient_tol: 1.0,
            energy_tol: 0.01,
            line_search: LineSearchOptions::default(),
            max_step_a: 0.1,
            history: 10,
            include_sasa: false,
        }
    }
}

#[derive(Debug)]
pub struct LbfgsResult {
    pub steps: usize,
    pub final_energy: f64,
    pub initial_energy: f64,
    pub max_force: f64,
    pub converged: bool,
    pub line_search_failures: usize,
}

pub fn lbfgs(
    structure: &mut Structure,
    graph: &TopologyGraph,
    ff: &ForceField,
    options: LbfgsOptions,
) -> LbfgsResult {
    let n_dofs = structure.atom_count() * 3;

    let initial_energy = total_energy_with_options(structure, graph, ff, options.include_sasa);
    let mut e_prev = initial_energy;

    let mut history_s: VecDeque<Vec<f64>> = VecDeque::with_capacity(options.history);
    let mut history_y: VecDeque<Vec<f64>> = VecDeque::with_capacity(options.history);
    let mut history_rho: VecDeque<f64> = VecDeque::with_capacity(options.history);

    let mut g_prev: Vec<f64> = vec![0.0; n_dofs];
    let mut g_curr: Vec<f64> = vec![0.0; n_dofs];
    let mut force_buffer = vec![geom::Vec3::zeros(); structure.atom_count()];

    let mut ls_failures = 0usize;
    let mut steps = 0usize;
    let mut converged = false;
    let mut max_force = f64::INFINITY;
    let mut x_prev: Vec<f64> = vec![0.0; n_dofs];

    for step in 0..options.max_steps {
        steps = step;
        force_buffer = total_force_opts(structure, graph, ff, options.include_sasa);
        // gradient = -force.
        flatten_vec3(&force_buffer, &mut g_curr);
        for g in g_curr.iter_mut() {
            *g = -*g;
        }
        max_force = linf_norm(&g_curr);
        if max_force < options.gradient_tol {
            converged = true;
            break;
        }

        // Build search direction.
        let direction = if history_s.is_empty() {
            // First step: steepest descent.
            g_curr.iter().map(|g| -*g).collect::<Vec<f64>>()
        } else {
            two_loop_recursion(&g_curr, &history_s, &history_y, &history_rho)
        };

        // Cap initial α so atoms don't move > max_step_a.
        let max_dir = linf_norm(&direction);
        let mut ls_options = options.line_search;
        ls_options.include_sasa = options.include_sasa;
        if max_dir > 0.0 {
            ls_options.alpha0 = ls_options.alpha0.min(options.max_step_a / max_dir);
        }

        let e_now = total_energy_with_options(structure, graph, ff, options.include_sasa);
        let res = backtracking(
            structure,
            graph,
            ff,
            &direction,
            &g_curr,
            e_now,
            ls_options,
        );
        let Some(ls) = res else {
            ls_failures += 1;
            // Fallback: try a steepest-descent step before giving up entirely.
            let sd_dir: Vec<f64> = g_curr.iter().map(|g| -*g).collect();
            let max_dir2 = linf_norm(&sd_dir);
            let mut ls_options2 = options.line_search;
            ls_options2.include_sasa = options.include_sasa;
            if max_dir2 > 0.0 {
                ls_options2.alpha0 = ls_options2.alpha0.min(options.max_step_a / max_dir2);
            }
            let res2 = backtracking(structure, graph, ff, &sd_dir, &g_curr, e_now, ls_options2);
            let Some(ls2) = res2 else {
                ls_failures += 1;
                break;
            };
            let alpha = ls2.alpha;
            let step_vec: Vec<f64> = sd_dir.iter().map(|d| alpha * d).collect();
            // Capture x and g before moving.
            // (We already captured g_curr; capture position too.)
            flatten_positions(structure, &mut x_prev);
            apply_displacement(structure, &step_vec);
            // Reset history because we deviated from the L-BFGS direction.
            history_s.clear();
            history_y.clear();
            history_rho.clear();
            g_prev.copy_from_slice(&g_curr);
            e_prev = ls2.new_energy;
            continue;
        };
        let alpha = ls.alpha;
        let step_vec: Vec<f64> = direction.iter().map(|d| alpha * d).collect();

        // Save x_prev = current x; then move.
        flatten_positions(structure, &mut x_prev);
        apply_displacement(structure, &step_vec);

        // Update history with (s = x_new - x_prev, y = g_new - g_prev) using
        // the next gradient.
        let force_after = total_force_opts(structure, graph, ff, options.include_sasa);
        let mut g_after = vec![0.0_f64; n_dofs];
        flatten_vec3(&force_after, &mut g_after);
        for g in g_after.iter_mut() {
            *g = -*g;
        }

        let s_vec: Vec<f64> = step_vec.clone();
        let y_vec: Vec<f64> = g_after
            .iter()
            .zip(g_curr.iter())
            .map(|(a, b)| a - b)
            .collect();
        let s_dot_y: f64 = s_vec.iter().zip(y_vec.iter()).map(|(a, b)| a * b).sum();
        if s_dot_y > 1e-12 {
            if history_s.len() == options.history {
                history_s.pop_front();
                history_y.pop_front();
                history_rho.pop_front();
            }
            history_s.push_back(s_vec);
            history_y.push_back(y_vec);
            history_rho.push_back(1.0 / s_dot_y);
        } else {
            // Skip this update — non-positive curvature would corrupt the
            // approximate Hessian.
        }

        g_prev.copy_from_slice(&g_curr);
        let e_new = ls.new_energy;
        if (e_prev - e_new).abs() < options.energy_tol && max_force < options.gradient_tol {
            converged = true;
            steps = step + 1;
            break;
        }
        e_prev = e_new;
    }

    LbfgsResult {
        steps,
        final_energy: total_energy_with_options(structure, graph, ff, options.include_sasa),
        initial_energy,
        max_force,
        converged,
        line_search_failures: ls_failures,
    }
}

fn two_loop_recursion(
    g: &[f64],
    history_s: &VecDeque<Vec<f64>>,
    history_y: &VecDeque<Vec<f64>>,
    history_rho: &VecDeque<f64>,
) -> Vec<f64> {
    let m = history_s.len();
    let n = g.len();
    let mut q = g.to_vec();
    let mut alpha = vec![0.0_f64; m];
    // First loop: backwards through history.
    for i in (0..m).rev() {
        let s = &history_s[i];
        let y = &history_y[i];
        let rho = history_rho[i];
        let dot_sq: f64 = s.iter().zip(q.iter()).map(|(a, b)| a * b).sum();
        let a = rho * dot_sq;
        alpha[i] = a;
        for k in 0..n {
            q[k] -= a * y[k];
        }
    }
    // Initial Hessian scaling: γ = (s_last · y_last) / (y_last · y_last).
    let s_last = &history_s[m - 1];
    let y_last = &history_y[m - 1];
    let s_y: f64 = s_last.iter().zip(y_last.iter()).map(|(a, b)| a * b).sum();
    let y_y: f64 = y_last.iter().map(|v| v * v).sum();
    let gamma = if y_y > 0.0 { s_y / y_y } else { 1.0 };
    let mut r: Vec<f64> = q.iter().map(|q_i| gamma * q_i).collect();
    // Second loop: forward through history.
    for i in 0..m {
        let s = &history_s[i];
        let y = &history_y[i];
        let rho = history_rho[i];
        let yr: f64 = y.iter().zip(r.iter()).map(|(a, b)| a * b).sum();
        let beta = rho * yr;
        for k in 0..n {
            r[k] += (alpha[i] - beta) * s[k];
        }
    }
    // p = -H^{-1} g = -r.
    r.iter().map(|x| -*x).collect()
}

fn flatten_positions(structure: &Structure, out: &mut [f64]) {
    let mut idx = 0usize;
    for residue in &structure.residues {
        for atom in &residue.atoms {
            out[idx * 3] = atom.position.x;
            out[idx * 3 + 1] = atom.position.y;
            out[idx * 3 + 2] = atom.position.z;
            idx += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chem::{standard_ff, AminoAcid, Element};
    use geom::structure::{PlacedAtom, PlacedResidue};
    use geom::{build_extended_chain, build_topology_graph, Vec3};

    #[test]
    fn lbfgs_two_atoms_relax_to_equilibrium() {
        let s = Structure {
            residues: vec![PlacedResidue {
                monomer: geom::structure::Monomer::Protein(AminoAcid::Ala),
                atoms: vec![
                    PlacedAtom { name: "N", element: Element::N, position: Vec3::zeros() },
                    PlacedAtom { name: "CA", element: Element::C, position: Vec3::new(1.7, 0.0, 0.0) },
                ],
                chain: 'A',
            }],
        };
        let mut s = s;
        let g = build_topology_graph(&s);
        let ff = standard_ff();
        let opts = LbfgsOptions {
            max_steps: 100,
            gradient_tol: 0.5,
            ..Default::default()
        };
        let res = lbfgs(&mut s, &g, ff, opts);
        assert!(res.converged, "L-BFGS didn't converge: {:?}", res);
        let r = (s.residues[0].atoms[1].position - s.residues[0].atoms[0].position).norm();
        assert!((r - 1.430).abs() < 0.05);
    }

    #[test]
    fn lbfgs_ala3_chain_drops_below_sd_in_fewer_steps() {
        let mut s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala, AminoAcid::Ala]).unwrap();
        let g = build_topology_graph(&s);
        let ff = standard_ff();
        let opts = LbfgsOptions {
            max_steps: 200,
            gradient_tol: 5.0,
            ..Default::default()
        };
        let res = lbfgs(&mut s, &g, ff, opts);
        assert!(res.final_energy < res.initial_energy);
    }
}
