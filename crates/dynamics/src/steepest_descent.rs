//! Steepest-descent minimization: search along the negative gradient
//! (i.e. the force direction) with a backtracking line search.
//!
//! Convergence: max gradient component falls below `gradient_tol` AND the
//! energy change between consecutive accepted steps falls below `energy_tol`.

use chem::ForceField;
use geom::{Structure, TopologyGraph};

use crate::energy_eval::{
    apply_displacement, flatten_vec3, linf_norm, total_energy_with_options, total_force_opts,
};
use crate::line_search::{backtracking, LineSearchOptions};

#[derive(Debug, Clone, Copy)]
pub struct SdOptions {
    pub max_steps: usize,
    /// Convergence threshold on max |force component|, in kJ/mol/Å.
    pub gradient_tol: f64,
    /// Convergence threshold on |ΔE|, in kJ/mol.
    pub energy_tol: f64,
    pub line_search: LineSearchOptions,
    /// Maximum step length per atom (Å) — clamps the line-search initial α.
    pub max_step_a: f64,
    /// Include SASA in energy + forces (PSA.2). When `true`, both
    /// gradient evaluations and line-search energy comparisons include
    /// the hydrophobic term. Slow (~100 ms/grad on Trp-cage).
    pub include_sasa: bool,
}

impl Default for SdOptions {
    fn default() -> Self {
        SdOptions {
            max_steps: 5000,
            gradient_tol: 1.0,
            energy_tol: 0.01,
            line_search: LineSearchOptions::default(),
            max_step_a: 0.1,
            include_sasa: false,
        }
    }
}

#[derive(Debug)]
pub struct SdResult {
    pub steps: usize,
    pub final_energy: f64,
    pub initial_energy: f64,
    pub max_force: f64,
    pub converged: bool,
    pub line_search_failures: usize,
}

pub fn steepest_descent(
    structure: &mut Structure,
    graph: &TopologyGraph,
    ff: &ForceField,
    options: SdOptions,
) -> SdResult {
    let n_atoms = structure.atom_count();
    let n_dofs = n_atoms * 3;
    let initial_energy = total_energy_with_options(structure, graph, ff, options.include_sasa);
    let mut e_prev = initial_energy;
    let mut gradient_flat = vec![0.0_f64; n_dofs];
    let mut direction_flat = vec![0.0_f64; n_dofs];
    let mut ls_failures = 0usize;
    let mut steps = 0usize;
    let mut converged = false;
    let mut max_force = f64::INFINITY;

    for step in 0..options.max_steps {
        steps = step;
        let forces = total_force_opts(structure, graph, ff, options.include_sasa);
        // gradient = -force; descent direction = -gradient = +force.
        flatten_vec3(&forces, &mut direction_flat);
        // gradient_flat is the negative of direction_flat for line search:
        for (g, d) in gradient_flat.iter_mut().zip(direction_flat.iter()) {
            *g = -*d;
        }
        max_force = linf_norm(&gradient_flat);
        if max_force < options.gradient_tol {
            converged = true;
            break;
        }

        // Cap initial α so no atom moves more than max_step_a per step.
        let max_dir = linf_norm(&direction_flat);
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
            &direction_flat,
            &gradient_flat,
            e_now,
            ls_options,
        );
        let Some(ls) = res else {
            ls_failures += 1;
            break;
        };
        let alpha = ls.alpha;
        // Apply the accepted step.
        let step_vec: Vec<f64> = direction_flat.iter().map(|d| alpha * d).collect();
        apply_displacement(structure, &step_vec);
        let e_new = ls.new_energy;
        if (e_prev - e_new).abs() < options.energy_tol && max_force < options.gradient_tol {
            converged = true;
            steps = step + 1;
            break;
        }
        e_prev = e_new;
    }
    SdResult {
        steps,
        final_energy: total_energy_with_options(structure, graph, ff, options.include_sasa),
        initial_energy,
        max_force,
        converged,
        line_search_failures: ls_failures,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chem::{standard_ff, AminoAcid, Element};
    use geom::structure::{PlacedAtom, PlacedResidue};
    use geom::{build_extended_chain, build_topology_graph, Vec3};

    #[test]
    fn two_atoms_relax_to_equilibrium() {
        // Two atoms on a single bond, pulled apart from r₀ — should converge.
        // Use the N-CA bond (NH1-CT1, r₀ = 1.430 Å). Start at 1.7 Å.
        let s = Structure {
            residues: vec![PlacedResidue {
                aa: AminoAcid::Ala,
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
        let opts = SdOptions {
            max_steps: 200,
            gradient_tol: 0.5,
            ..Default::default()
        };
        let res = steepest_descent(&mut s, &g, ff, opts);
        assert!(res.converged, "SD didn't converge: {:?}", res);
        let n_pos = s.residues[0].atoms[0].position;
        let ca_pos = s.residues[0].atoms[1].position;
        let r = (ca_pos - n_pos).norm();
        assert!(
            (r - 1.430).abs() < 0.05,
            "final bond length {} should be near r₀ = 1.43", r
        );
    }

    #[test]
    fn ala3_chain_energy_drops() {
        let mut s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala, AminoAcid::Ala]).unwrap();
        let g = build_topology_graph(&s);
        let ff = standard_ff();
        let opts = SdOptions {
            max_steps: 500,
            gradient_tol: 5.0,
            ..Default::default()
        };
        let res = steepest_descent(&mut s, &g, ff, opts);
        assert!(
            res.final_energy < res.initial_energy,
            "energy didn't drop: {} → {}", res.initial_energy, res.final_energy
        );
    }
}
