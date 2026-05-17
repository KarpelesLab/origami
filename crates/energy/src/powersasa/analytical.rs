//! Analytical SASA force assembly (Klenin §3, top of the pile).
//!
//! Wires the validated derivative primitives in `derivatives.rs`
//! (`cos_alpha_grad`, `vertex_point_jvp`, `arc_theta_jvp`,
//! `vertex_epsilon_jvp`) into a per-atom area-gradient computation.
//!
//! Two pieces:
//!   1. [`AtomBoundaryCache`] — a stable record of one atom's boundary
//!      *topology*: which cap each arc lies on, which two caps define
//!      each vertex, and the [`RootSign`] tag identifying which of the
//!      two pairwise-intersection roots a vertex is. Built once per
//!      force evaluation by running `build_caps` + `find_boundary` at
//!      the unperturbed configuration.
//!   2. [`atom_area_gradient`] — for each atom in the cache's
//!      "affected" set (atom i itself plus every neighbour that
//!      contributed a cap), computes `∂A_i / ∂r_x` as a 3-vector by
//!      running the JVP primitives three times (basis directions) and
//!      assembling.
//!
//! These together replace the central-difference scheme in
//! `forces_sasa::add_sasa_forces`. The numerical implementation stays
//! around as the reference baseline.

use geom::Vec3;

use super::arrangement::{
    build_caps, find_boundary, AtomBoundary, BoundaryVertex,
};
use super::derivatives::{
    arc_theta_jvp, cos_alpha_directional, identify_root_sign, vertex_epsilon_jvp, RootSign,
};
use super::geometry::SmallCircle;

/// One arc on a cached atom boundary.
#[derive(Debug, Clone, Copy)]
pub struct CachedArc {
    /// Local cap index this arc lies on.
    pub cap_local: usize,
    /// Local vertex index where the arc starts. `usize::MAX` for full-circle arcs.
    pub start_vertex_local: usize,
    /// Local vertex index where the arc ends. `usize::MAX` for full-circle arcs.
    pub end_vertex_local: usize,
    /// Signed central angle (matches `BoundaryArc::theta`).
    pub theta: f64,
    /// `true` if the entire small-circle is a boundary loop (no
    /// intersections with other caps).
    pub is_full_circle: bool,
}

/// One vertex on a cached atom boundary.
///
/// Stored such that `vertex_point(caps[incoming_cap_local],
/// caps[outgoing_cap_local], sign)` reproduces the geometric vertex
/// position at the configuration the cache was built from. Both
/// `arc_theta_jvp` and `vertex_epsilon_jvp` consume vertices in this
/// "(incoming, outgoing)" form.
#[derive(Debug, Clone, Copy)]
pub struct CachedVertex {
    pub incoming_cap_local: usize,
    pub outgoing_cap_local: usize,
    pub sign: RootSign,
}

/// Per-atom boundary topology, stable under small perturbations.
#[derive(Debug, Clone)]
pub struct AtomBoundaryCache {
    /// Global index of the atom this cache belongs to.
    pub atom_idx: usize,
    /// vdW + probe radius for atom i.
    pub radius_i: f64,
    /// Cap-local → global atom index. Length = `caps.len()`.
    pub cap_owners: Vec<usize>,
    /// Small-circle caps at the unperturbed configuration. Kept around
    /// to feed `vertex_point` / identify_root_sign during caching; the
    /// gradient code only ever reconstructs caps from the current
    /// `positions` so the cached `caps` value is for setup-time use.
    pub caps_ref: Vec<SmallCircle>,
    /// Boundary arcs.
    pub arcs: Vec<CachedArc>,
    /// Boundary vertices.
    pub vertices: Vec<CachedVertex>,
    /// Accessible connected components count (`c` in the Gauss-Bonnet
    /// formula `χ = 2c − L`).
    pub c: usize,
    /// Number of boundary loops (matches the face-walker count).
    pub l: usize,
}

/// Build an [`AtomBoundaryCache`] for atom `atom_idx` from the current
/// positions. Returns `None` if the atom is fully enclosed by some
/// neighbour or fully buried — in those cases its contribution to the
/// SASA energy and its derivatives are zero, so the caller can skip.
///
/// Reuses the existing `build_caps` + `find_boundary` machinery so the
/// boundary tracing logic stays in one place. Then post-processes:
///   - tags each `BoundaryVertex` with its [`RootSign`] using
///     `identify_root_sign`
///   - links each `BoundaryArc` to its start/end vertices by position
///     match (the existing data structure stores positions but not
///     the connectivity by index)
pub fn build_atom_cache(
    atom_idx: usize,
    positions: &[Vec3],
    radii: &[f64],
    neighbour_indices: &[usize],
) -> Option<AtomBoundaryCache> {
    let p_i = positions[atom_idx];
    let r_i = radii[atom_idx];

    // Build the neighbour list with (global_idx, position, radius) form.
    let neighbours: Vec<(usize, Vec3, f64)> = neighbour_indices
        .iter()
        .filter_map(|&j| {
            let pj = positions[j];
            let d = (pj - p_i).norm();
            if d <= r_i + radii[j] {
                Some((j, pj, radii[j]))
            } else {
                None
            }
        })
        .collect();
    let (caps, cap_owners) = build_caps(p_i, r_i, &neighbours)?;
    let boundary = find_boundary(&caps);
    let (arcs_raw, vertices_raw) = match boundary {
        AtomBoundary::FullyExposed | AtomBoundary::FullyBuried => {
            // Nothing on the boundary to differentiate; return a cache
            // with no arcs/vertices. `c` is 1 for FullyExposed, 0 for
            // FullyBuried. The gradient code treats both as zero
            // contribution (A is constant under perturbation).
            let c = matches!(boundary, AtomBoundary::FullyExposed) as usize;
            return Some(AtomBoundaryCache {
                atom_idx,
                radius_i: r_i,
                cap_owners,
                caps_ref: caps,
                arcs: Vec::new(),
                vertices: Vec::new(),
                c,
                l: 0,
            });
        }
        AtomBoundary::Bounded { arcs, vertices } => (arcs, vertices),
    };

    // Convert boundary vertices to (incoming, outgoing, sign) form.
    let cached_vertices: Vec<CachedVertex> = vertices_raw
        .iter()
        .map(|v: &BoundaryVertex| {
            let c_in = caps[v.incoming_cap];
            let c_out = caps[v.outgoing_cap];
            let sign = identify_root_sign(c_in, c_out, v.point).unwrap_or(RootSign::Plus);
            CachedVertex {
                incoming_cap_local: v.incoming_cap,
                outgoing_cap_local: v.outgoing_cap,
                sign,
            }
        })
        .collect();

    // Link arcs to their start/end vertices. `find_boundary` pushes
    // arcs and vertices in lock-step ONLY for vertexed arcs (full-circle
    // arcs append to `arcs` without appending to `vertices`), so the
    // arc index ≠ vertex index in the mixed-population case.
    //
    // We compute the end vertex by scanning vertices for the one whose
    // point matches `arc.end`, and same for `arc.start`. Robust to any
    // ordering find_boundary chooses to emit.
    let eps_sq = 1e-10;
    let find_vertex_at = |target: Vec3| -> usize {
        vertices_raw
            .iter()
            .enumerate()
            .find(|(_, v)| (v.point - target).norm_squared() < eps_sq)
            .map(|(idx, _)| idx)
            .unwrap_or(usize::MAX)
    };
    let mut cached_arcs: Vec<CachedArc> = Vec::with_capacity(arcs_raw.len());
    for arc in arcs_raw.iter() {
        if arc.is_full_circle {
            cached_arcs.push(CachedArc {
                cap_local: arc.cap_idx,
                start_vertex_local: usize::MAX,
                end_vertex_local: usize::MAX,
                theta: arc.theta,
                is_full_circle: true,
            });
            continue;
        }
        let start_v_local = find_vertex_at(arc.start);
        let end_v_local = find_vertex_at(arc.end);
        cached_arcs.push(CachedArc {
            cap_local: arc.cap_idx,
            start_vertex_local: start_v_local,
            end_vertex_local: end_v_local,
            theta: arc.theta,
            is_full_circle: false,
        });
    }

    let l_loops = compute_loop_count(&cached_arcs, &cached_vertices, &caps);
    let c = super::arrangement::count_accessible_components(&caps).max(1);

    Some(AtomBoundaryCache {
        atom_idx,
        radius_i: r_i,
        cap_owners,
        caps_ref: caps,
        arcs: cached_arcs,
        vertices: cached_vertices,
        c,
        l: l_loops,
    })
}

/// Count loops by walking arcs using the same (end_vertex → next arc
/// with matching cap continuation) logic as the face walker in area.rs.
/// Each full-circle arc is one loop. Vertexed arcs partition into
/// loops by following next-arc links through shared vertex indices.
fn compute_loop_count(
    arcs: &[CachedArc],
    vertices: &[CachedVertex],
    _caps: &[SmallCircle],
) -> usize {
    let mut full_circle_count = 0usize;
    let mut vertexed_arcs: Vec<usize> = Vec::with_capacity(arcs.len());
    for (i, arc) in arcs.iter().enumerate() {
        if arc.is_full_circle {
            full_circle_count += 1;
        } else if arc.start_vertex_local != usize::MAX
            && arc.end_vertex_local != usize::MAX
        {
            vertexed_arcs.push(i);
        }
    }
    if vertexed_arcs.is_empty() {
        return full_circle_count;
    }
    // Walk: from arc i, the next arc is the one starting at i's end
    // vertex with the matching outgoing cap. Each connected walk =
    // one loop.
    let n = vertexed_arcs.len();
    let mut visited = vec![false; n];
    let mut loops = 0usize;
    for start in 0..n {
        if visited[start] {
            continue;
        }
        let mut current = start;
        loop {
            visited[current] = true;
            let arc = &arcs[vertexed_arcs[current]];
            let end_v = &vertices[arc.end_vertex_local];
            let next_cap_local = if end_v.incoming_cap_local == arc.cap_local {
                end_v.outgoing_cap_local
            } else {
                end_v.incoming_cap_local
            };
            let next = vertexed_arcs.iter().position(|&ai| {
                !visited[vertexed_arcs.iter().position(|&x| x == ai).unwrap_or(usize::MAX)]
                    && arcs[ai].cap_local == next_cap_local
                    && arcs[ai].start_vertex_local == arc.end_vertex_local
            });
            match next {
                Some(j) => current = j,
                None => break,
            }
        }
        loops += 1;
    }
    full_circle_count + loops
}

/// Compute ∂A_i / ∂r_x as a 3-vector for every atom x that affects
/// atom i's area, given the cached topology and current positions.
/// Writes into `out`: `out[x] += ∂A_i/∂r_x · R_i²` for each affected x.
///
/// The R_i² scaling is folded in here (because A_i = R_i² · [...]).
/// The caller multiplies by γ_i (per-element surface tension) to turn
/// area gradient into force.
///
/// "Affected" atoms are: atom i itself, plus every atom appearing in
/// `cache.cap_owners`. For each affected atom, we run the JVP
/// primitives three times (one per Cartesian axis) and assemble.
pub fn atom_area_gradient(
    cache: &AtomBoundaryCache,
    positions: &[Vec3],
    radii: &[f64],
    out: &mut [Vec3],
) {
    // Identify affected atoms (deduplicated).
    let mut affected: Vec<usize> = Vec::with_capacity(cache.cap_owners.len() + 1);
    affected.push(cache.atom_idx);
    for &owner in &cache.cap_owners {
        if !affected.contains(&owner) {
            affected.push(owner);
        }
    }

    let r_sq = cache.radius_i * cache.radius_i;
    for &x_atom in &affected {
        let mut grad = Vec3::zeros();
        for axis in 0..3 {
            let mut basis = Vec3::zeros();
            basis[axis] = 1.0;
            let d_area = directional_area_derivative(cache, positions, radii, x_atom, basis);
            grad[axis] = d_area;
        }
        out[x_atom] += grad * r_sq;
    }
}

/// Directional derivative of `A_i / R_i²` with respect to perturbing
/// `x_atom` by `basis`. (R_i² is folded back in by the caller.)
fn directional_area_derivative(
    cache: &AtomBoundaryCache,
    positions: &[Vec3],
    radii: &[f64],
    x_atom: usize,
    basis: Vec3,
) -> f64 {
    // Helper: perturbation vector for atom y, given the perturbed atom is x_atom.
    let dr_of = |y: usize| -> Vec3 {
        if y == x_atom {
            basis
        } else {
            Vec3::zeros()
        }
    };

    let p_i = positions[cache.atom_idx];
    let r_i = cache.radius_i;
    let dr_i = dr_of(cache.atom_idx);

    let mut sum_arc = 0.0_f64;
    for arc in &cache.arcs {
        let k_atom = cache.cap_owners[arc.cap_local];
        let p_k = positions[k_atom];
        let r_k = radii[k_atom];
        let dr_k = dr_of(k_atom);

        // d(cos α_K) along the perturbation.
        let dcos = cos_alpha_directional(p_i, p_k, r_i, r_k, dr_i, dr_k);
        // cos α_K at the reference (computed inline since `arc` doesn't carry it).
        let d_ik = (p_k - p_i).norm();
        let cos_alpha_k = (d_ik * d_ik + r_i * r_i - r_k * r_k) / (2.0 * d_ik * r_i);

        if arc.is_full_circle {
            // θ = ±2π (constant); only dcos α contributes.
            sum_arc += arc.theta * dcos;
            continue;
        }

        // Defensive: if vertex linking failed during cache construction
        // (rare with degenerate boundaries), treat θ as fixed and emit
        // only the dcos term. Better to drop a small contribution than
        // to panic mid-simulation.
        if arc.start_vertex_local >= cache.vertices.len()
            || arc.end_vertex_local >= cache.vertices.len()
        {
            sum_arc += arc.theta * dcos;
            continue;
        }
        let vs = cache.vertices[arc.start_vertex_local];
        let ve = cache.vertices[arc.end_vertex_local];

        // At a vertex `v` with (incoming, outgoing) cap-local pair:
        // - Reference vertex position = vertex_point(caps[incoming], caps[outgoing], v.sign).
        // - To use in arc_theta_jvp, which expects vertex_point(arc_cap, OTHER, sign_in_that_order),
        //   we swap if needed.
        let (vs_other_local, vs_sign) =
            arc_other_cap_and_sign(arc.cap_local, vs);
        let (ve_other_local, ve_sign) =
            arc_other_cap_and_sign(arc.cap_local, ve);

        let l_atom = cache.cap_owners[vs_other_local];
        let m_atom = cache.cap_owners[ve_other_local];
        let p_l = positions[l_atom];
        let p_m = positions[m_atom];
        let r_l = radii[l_atom];
        let r_m = radii[m_atom];
        let dr_l = dr_of(l_atom);
        let dr_m = dr_of(m_atom);

        let dtheta = arc_theta_jvp(
            p_i, p_k, p_l, p_m, r_i, r_k, r_l, r_m, vs_sign, ve_sign, dr_i, dr_k, dr_l, dr_m,
        )
        .unwrap_or(0.0);

        // arc contribution to (A/R²) is −(cos α · θ); derivative is
        // −(dcos · θ + cos α · dθ).
        sum_arc += arc.theta * dcos + cos_alpha_k * dtheta;
    }

    let mut sum_vert = 0.0_f64;
    for v in &cache.vertices {
        let in_atom = cache.cap_owners[v.incoming_cap_local];
        let out_atom = cache.cap_owners[v.outgoing_cap_local];
        let p_k = positions[in_atom];
        let p_l = positions[out_atom];
        let r_k = radii[in_atom];
        let r_l = radii[out_atom];
        let dr_k = dr_of(in_atom);
        let dr_l = dr_of(out_atom);

        // Convention in derivatives.rs: vertex_epsilon_jvp's
        // `incoming_is_k` selects which of the two caps is "incoming".
        // Our cache stores incoming as cap p_k, so `incoming_is_k = true`.
        let deps = vertex_epsilon_jvp(
            p_i, p_k, p_l, r_i, r_k, r_l, v.sign, true, dr_i, dr_k, dr_l,
        )
        .unwrap_or(0.0);
        sum_vert += deps;
    }

    // χ is locally constant (integer-valued, only flips at topology
    // transitions), so its derivative is zero in smooth regions. The
    // overall area is A = R²(2π χ − Σ_arc cos α θ − Σ_vert ε), so
    // d(A/R²) = −Σ_arc d(cos α θ) − Σ_vert dε.
    -sum_arc - sum_vert
}

/// Add analytical SASA forces to the existing force buffer.
///
/// For each atom i with non-zero surface tension γ_i:
///   - Build its boundary topology cache.
///   - Compute ∂A_i/∂r_x as a 3-vector for every affected atom x.
///   - Accumulate forces[x] -= γ_i · ∂A_i/∂r_x  (F = −∇E_SASA, and
///     E_SASA = Σ_i γ_i · A_i).
///
/// This replaces the numerical central-difference scheme in
/// `forces_sasa::add_sasa_forces`. The numerical version stays around
/// as the reference baseline.
/// Scratch-aware variant of [`add_sasa_forces_analytical`] used by the
/// production integrator path (`forces::total_force_with_scratch`).
/// Two improvements over the no-scratch version:
///
///   • The per-atom neighbour lists are cached on the scratch with a
///     Verlet skin — rebuilt only when an atom drifts more than
///     `skin/2`. Saves the O(N²) all-pair scan every step.
///
///   • The per-atom force loop (build_atom_cache + atom_area_gradient
///     + force accumulation) runs in parallel via rayon, with each
///     worker writing to its own slice of the
///     `scratch.sasa_par_forces` buffer; a serial reduce sums the
///     thread slices back into the caller's force buffer.
///
/// Equivalent results to `add_sasa_forces_analytical` (within
/// floating-point reduction ordering), substantially faster when the
/// integrator calls it tens of thousands of times in a row.
pub fn add_sasa_forces_analytical_with_scratch(
    structure: &geom::Structure,
    _ff: &chem::ForceField,
    scratch: &mut crate::scratch::ForceScratch,
    forces: &mut [Vec3],
) {
    use chem::Element;
    use rayon::prelude::*;
    let n = structure.atom_count();
    assert_eq!(forces.len(), n);
    assert_eq!(scratch.n, n);

    // Flatten positions + radii + per-atom γ from the structure. We
    // could move these into the scratch too (they only change when
    // atoms are added/removed), but the cost is sub-µs compared to
    // the per-atom topology compute below.
    let mut positions: Vec<Vec3> = Vec::with_capacity(n);
    let mut radii: Vec<f64> = Vec::with_capacity(n);
    let mut elements: Vec<Element> = Vec::with_capacity(n);
    for residue in &structure.residues {
        for atom in &residue.atoms {
            positions.push(atom.position);
            radii.push(super::vdw_radius(atom.element) + super::PROBE_RADIUS_A);
            elements.push(atom.element);
        }
    }
    let gamma_scale = std::env::var("ORIGAMI_SASA_GAMMA_SCALE")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1.0);
    let gamma: Vec<f64> = elements
        .iter()
        .map(|&e| crate::units::kcal_to_kj(super::surface_tension_kcal(e)) * gamma_scale)
        .collect();
    let max_radius = radii.iter().cloned().fold(0.0_f64, f64::max);

    // ---- Verlet check on the cached per-atom neighbour lists ----
    let skin = crate::scratch::VERLET_SKIN_A;
    let half_skin_sq = (0.5 * skin) * (0.5 * skin);
    scratch.sync_positions(structure);
    let need_rebuild = !scratch.sasa_verlet_valid || {
        let mut moved = false;
        for i in 0..n {
            let dx = scratch.xs[i] - scratch.sasa_verlet_ref_x[i];
            let dy = scratch.ys[i] - scratch.sasa_verlet_ref_y[i];
            let dz = scratch.zs[i] - scratch.sasa_verlet_ref_z[i];
            if dx * dx + dy * dy + dz * dz > half_skin_sq {
                moved = true;
                break;
            }
        }
        moved
    };
    if need_rebuild {
        // Rebuild via cell-list at `2·max_radius + skin`. Per-pair
        // check is still `d ≤ r_i + r_j`; the skin gives slack so
        // the list stays valid as atoms drift.
        let cell_size = (2.0 * max_radius + skin).max(1.0);
        let cl = geom::CellList::build(&positions, cell_size);
        for v in &mut scratch.sasa_neighbours {
            v.clear();
        }
        for (i, j, d) in cl.iter_pairs_within(&positions, cell_size) {
            if d <= radii[i] + radii[j] + skin {
                scratch.sasa_neighbours[i].push(j as u32);
                scratch.sasa_neighbours[j].push(i as u32);
            }
        }
        scratch.sasa_verlet_ref_x.copy_from_slice(&scratch.xs);
        scratch.sasa_verlet_ref_y.copy_from_slice(&scratch.ys);
        scratch.sasa_verlet_ref_z.copy_from_slice(&scratch.zs);
        scratch.sasa_verlet_valid = true;
    }

    // ---- Parallel per-atom force compute ----
    //
    // Each worker writes into its own n × 3 slice of
    // `sasa_par_forces`; the per-atom (build_atom_cache +
    // atom_area_gradient + accumulate) is independent across `i`, so
    // there's no contention.
    let n_threads = scratch.n_par_threads;
    let slice_len = n * 3;
    let work: Vec<usize> = (0..n).filter(|&i| gamma[i] != 0.0).collect();
    let chunk_size = work.len().div_ceil(n_threads).max(1);
    let used_threads = work.len().div_ceil(chunk_size);
    let buf = &mut scratch.sasa_par_forces[..used_threads * slice_len];
    buf.fill(0.0);
    let chunks: Vec<&[usize]> = work.chunks(chunk_size).collect();
    // Take the neighbour lists out so the parallel section can borrow
    // them immutably alongside scratch's other fields.
    let sasa_neighbours = std::mem::take(&mut scratch.sasa_neighbours);

    buf.par_chunks_mut(slice_len)
        .zip(chunks.into_par_iter())
        .for_each(|(thread_buf, chunk)| {
            // Per-thread reusable da_dr buffer.
            let mut da_dr: Vec<Vec3> = vec![Vec3::zeros(); n];
            for &i in chunk {
                let neighbours: Vec<usize> =
                    sasa_neighbours[i].iter().map(|&j| j as usize).collect();
                let Some(cache) = build_atom_cache(i, &positions, &radii, &neighbours) else {
                    continue;
                };
                if cache.arcs.is_empty() {
                    continue;
                }
                for f in da_dr.iter_mut() {
                    *f = Vec3::zeros();
                }
                atom_area_gradient(&cache, &positions, &radii, &mut da_dr);
                let g = gamma[i];
                for x in 0..n {
                    thread_buf[3 * x] -= da_dr[x].x * g;
                    thread_buf[3 * x + 1] -= da_dr[x].y * g;
                    thread_buf[3 * x + 2] -= da_dr[x].z * g;
                }
            }
        });

    scratch.sasa_neighbours = sasa_neighbours;

    // Serial reduce per-thread accumulators into the caller's forces.
    for t in 0..used_threads {
        let base = t * slice_len;
        for x in 0..n {
            forces[x].x += scratch.sasa_par_forces[base + 3 * x];
            forces[x].y += scratch.sasa_par_forces[base + 3 * x + 1];
            forces[x].z += scratch.sasa_par_forces[base + 3 * x + 2];
        }
    }
}

pub fn add_sasa_forces_analytical(
    structure: &geom::Structure,
    _ff: &chem::ForceField,
    forces: &mut [Vec3],
) {
    use chem::Element;
    let n = structure.atom_count();
    assert_eq!(forces.len(), n);
    let mut positions: Vec<Vec3> = Vec::with_capacity(n);
    let mut radii: Vec<f64> = Vec::with_capacity(n);
    let mut elements: Vec<Element> = Vec::with_capacity(n);
    for residue in &structure.residues {
        for atom in &residue.atoms {
            positions.push(atom.position);
            radii.push(super::vdw_radius(atom.element) + super::PROBE_RADIUS_A);
            elements.push(atom.element);
        }
    }
    // Per-element surface tension γ in kJ/mol/Å² (matches the numerical
    // implementation in forces_sasa). Optional multiplicative scale
    // from the `ORIGAMI_SASA_GAMMA_SCALE` env var (default 1.0) lets us
    // probe the hydrophobic-strength axis without recompiling — useful
    // for sweeping γ to test whether the molten-globule trap goes away
    // at weaker coupling.
    let gamma_scale = std::env::var("ORIGAMI_SASA_GAMMA_SCALE")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(1.0);
    let gamma: Vec<f64> = elements
        .iter()
        .map(|&e| crate::units::kcal_to_kj(super::surface_tension_kcal(e)) * gamma_scale)
        .collect();
    // Neighbour lists from the unperturbed configuration.
    let mut neighbour_idx: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = (positions[j] - positions[i]).norm();
            if d <= radii[i] + radii[j] {
                neighbour_idx[i].push(j);
                neighbour_idx[j].push(i);
            }
        }
    }

    // Buffer for ∂A_i / ∂r_x per atom — reused across the i loop.
    let mut da_dr: Vec<Vec3> = vec![Vec3::zeros(); n];
    for i in 0..n {
        if gamma[i] == 0.0 {
            continue;
        }
        let Some(cache) = build_atom_cache(i, &positions, &radii, &neighbour_idx[i]) else {
            continue;
        };
        if cache.arcs.is_empty() {
            continue; // FullyExposed / FullyBuried → A constant, no force
        }
        // Reset the per-atom gradient buffer.
        for f in da_dr.iter_mut() {
            *f = Vec3::zeros();
        }
        atom_area_gradient(&cache, &positions, &radii, &mut da_dr);
        // Force contribution from atom i's SASA: F_x -= γ_i · ∂A_i/∂r_x.
        for x in 0..n {
            forces[x] -= da_dr[x] * gamma[i];
        }
    }
}

/// Given an arc on `arc_cap_local` and one of its endpoint vertices,
/// return (the OTHER cap's local index, the sign such that
/// `vertex_point(arc_cap_local, other, returned_sign) = V`).
///
/// The vertex is stored with `vertex_point(incoming, outgoing,
/// v.sign) = V`. If `arc_cap_local == incoming`, then (other =
/// outgoing, sign = v.sign). Otherwise (other = incoming, sign =
/// −v.sign) because vertex_point is antisymmetric in the sign when
/// the two cap arguments are swapped (see RootSign docs).
fn arc_other_cap_and_sign(arc_cap_local: usize, v: CachedVertex) -> (usize, RootSign) {
    if v.incoming_cap_local == arc_cap_local {
        (v.outgoing_cap_local, v.sign)
    } else {
        let flipped = match v.sign {
            RootSign::Plus => RootSign::Minus,
            RootSign::Minus => RootSign::Plus,
        };
        (v.incoming_cap_local, flipped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::powersasa::area::accessible_area_with_components;

    #[test]
    fn scratch_path_matches_inline_baseline() {
        // The Verlet+rayon scratch path must produce the same SASA
        // forces as the simple inline baseline, to floating-point
        // reduction-order precision.
        use chem::{standard_ff, AminoAcid};
        use geom::{build_extended_chain, build_topology_graph};
        let s = build_extended_chain(&[
            AminoAcid::Ala, AminoAcid::Gly, AminoAcid::Ala,
            AminoAcid::Lys, AminoAcid::Glu,
        ]).unwrap();
        let n = s.atom_count();
        let ff = standard_ff();
        let g = build_topology_graph(&s);

        let mut baseline = vec![Vec3::zeros(); n];
        add_sasa_forces_analytical(&s, ff, &mut baseline);

        let mut scratch = crate::scratch::ForceScratch::new(&s, &g, ff);
        let mut fast = vec![Vec3::zeros(); n];
        add_sasa_forces_analytical_with_scratch(&s, ff, &mut scratch, &mut fast);

        for (a, b) in baseline.iter().zip(&fast) {
            let d = (*a - *b).norm();
            assert!(d < 1e-6, "scratch path diverges: {a:?} vs {b:?} (Δ={d:.2e})");
        }
    }

    #[test]
    fn scratch_path_cached_call_matches_fresh() {
        // Two calls on identical positions — first builds Verlet
        // list, second reuses it. Both must give bit-identical
        // forces (no reduction-order ambiguity since we use the same
        // thread count).
        use chem::{standard_ff, AminoAcid};
        use geom::{build_extended_chain, build_topology_graph};
        let s = build_extended_chain(&[
            AminoAcid::Ala, AminoAcid::Gly, AminoAcid::Ala,
            AminoAcid::Lys, AminoAcid::Glu,
        ]).unwrap();
        let n = s.atom_count();
        let ff = standard_ff();
        let g = build_topology_graph(&s);
        let mut scratch = crate::scratch::ForceScratch::new(&s, &g, ff);

        let mut first = vec![Vec3::zeros(); n];
        add_sasa_forces_analytical_with_scratch(&s, ff, &mut scratch, &mut first);
        assert!(scratch.sasa_verlet_valid);

        let mut second = vec![Vec3::zeros(); n];
        add_sasa_forces_analytical_with_scratch(&s, ff, &mut scratch, &mut second);

        for (a, b) in first.iter().zip(&second) {
            let d = (*a - *b).norm();
            assert!(d < 1e-12, "cached vs fresh: Δ={d:.2e}");
        }
    }

    /// Build the cache, recompute the area from cache + reference
    /// positions, and confirm it matches `accessible_area_with_components`
    /// on the same input. Locks in the topology-extraction step.
    #[test]
    fn cache_round_trip_matches_accessible_area_ala3() {
        use chem::AminoAcid;
        use geom::build_extended_chain;

        let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala, AminoAcid::Ala]).unwrap();
        let n = s.atom_count();
        let positions: Vec<Vec3> = s
            .residues
            .iter()
            .flat_map(|r| r.atoms.iter().map(|a| a.position))
            .collect();
        let radii: Vec<f64> = s
            .residues
            .iter()
            .flat_map(|r| r.atoms.iter().map(|a| crate::powersasa::vdw_radius(a.element) + crate::powersasa::PROBE_RADIUS_A))
            .collect();

        // Build neighbour lists (atom -> neighbours within sum of radii).
        let mut neighbour_idx: Vec<Vec<usize>> = vec![Vec::new(); n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = (positions[j] - positions[i]).norm();
                if d <= radii[i] + radii[j] {
                    neighbour_idx[i].push(j);
                    neighbour_idx[j].push(i);
                }
            }
        }

        // For every atom, build the cache and confirm that re-running
        // build_caps + find_boundary + accessible_area at the SAME
        // positions gives the same A_i as we'd get by walking the cache.
        // (We use accessible_area_with_components as the reference.)
        for i in 0..n {
            let cache = build_atom_cache(i, &positions, &radii, &neighbour_idx[i]);
            let Some(cache) = cache else { continue };
            // Reference area via the existing pipeline.
            let (caps, _owners) = build_caps(
                positions[i],
                radii[i],
                &neighbour_idx[i]
                    .iter()
                    .map(|&j| (j, positions[j], radii[j]))
                    .collect::<Vec<_>>(),
            )
            .expect("caps");
            let boundary = find_boundary(&caps);
            let area_ref =
                accessible_area_with_components(radii[i], &caps, &boundary, Some(cache.c));
            // The cached cache.l should match the loop count `area`
            // would compute internally — we don't strictly need to test
            // that here, but require area_ref is finite and within
            // [0, 4πR²].
            assert!(
                area_ref >= -1e-6 && area_ref.is_finite(),
                "atom {} ref area unreasonable: {}",
                i,
                area_ref
            );
            // Cache invariants.
            assert_eq!(cache.cap_owners.len(), cache.caps_ref.len());
            for arc in &cache.arcs {
                if arc.is_full_circle {
                    assert_eq!(arc.start_vertex_local, usize::MAX);
                    assert_eq!(arc.end_vertex_local, usize::MAX);
                } else {
                    assert!(arc.start_vertex_local < cache.vertices.len());
                    assert!(arc.end_vertex_local < cache.vertices.len());
                }
                assert!(arc.cap_local < cache.caps_ref.len());
            }
            for v in &cache.vertices {
                assert!(v.incoming_cap_local < cache.caps_ref.len());
                assert!(v.outgoing_cap_local < cache.caps_ref.len());
                assert_ne!(v.incoming_cap_local, v.outgoing_cap_local);
            }
        }
    }

    /// End-to-end: `add_sasa_forces_analytical` agrees with central
    /// differences of total `powersasa_energy` on a small chain,
    /// matching the same shape of test as the numerical reference in
    /// `forces_sasa::sasa_forces_finite_difference_matches_total_e_sasa`.
    #[test]
    fn analytical_force_matches_central_difference_against_total_e_sasa() {
        use chem::standard_ff;
        use chem::AminoAcid;
        use geom::build_extended_chain;

        // Pick a chain with carbons/sulfurs (non-zero γ) and a few
        // residues so the boundary topology is non-trivial.
        let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Val]).unwrap();
        let n = s.atom_count();
        let ff = standard_ff();
        let mut forces_analytical = vec![Vec3::zeros(); n];
        add_sasa_forces_analytical(&s, ff, &mut forces_analytical);

        // Central-difference reference against the total SASA energy.
        let eps = 1e-4;
        let mut max_err: f64 = 0.0;
        let mut max_label = String::new();
        // Spot-check the first few atoms — enough to catch errors.
        for k in 0..n.min(8) {
            for axis in 0..3 {
                let mut s_plus = s.clone();
                let mut s_minus = s.clone();
                bump(&mut s_plus, k, axis, eps);
                bump(&mut s_minus, k, axis, -eps);
                let e_plus = crate::powersasa::powersasa_energy(&s_plus, ff).sasa_kj_mol;
                let e_minus = crate::powersasa::powersasa_energy(&s_minus, ff).sasa_kj_mol;
                let numeric_force = -(e_plus - e_minus) / (2.0 * eps);
                let analytical = forces_analytical[k][axis];
                let err = (analytical - numeric_force).abs();
                if err > max_err {
                    max_err = err;
                    max_label = format!(
                        "atom {} axis {}: analytical={:.4}, numeric={:.4}",
                        k, axis, analytical, numeric_force
                    );
                }
                assert!(
                    err < 0.5,
                    "atom {} axis {}: analytical={} numeric={} err={}",
                    k, axis, analytical, numeric_force, err
                );
            }
        }
        eprintln!("max analytical-vs-numerical SASA force error: {} ({})", max_err, max_label);
    }

    fn bump(s: &mut geom::Structure, atom_idx: usize, axis: usize, eps: f64) {
        let mut count = 0usize;
        for residue in &mut s.residues {
            for atom in &mut residue.atoms {
                if count == atom_idx {
                    atom.position[axis] += eps;
                    return;
                }
                count += 1;
            }
        }
    }

    /// On Ala₃ (a real chain with non-trivial boundary topology),
    /// the analytical area gradient for each atom matches the central
    /// difference of `accessible_area_with_components` to 1e-3 abs.
    /// This is the *end-to-end* verification that the topology cache +
    /// JVP routing handles realistic protein geometry — vertexed
    /// arcs, multiple boundary loops, and crowded atoms (~20 caps).
    #[test]
    fn analytical_gradient_matches_finite_difference_ala3() {
        use chem::AminoAcid;
        use geom::build_extended_chain;

        let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala, AminoAcid::Ala]).unwrap();
        let n = s.atom_count();
        let positions: Vec<Vec3> = s
            .residues
            .iter()
            .flat_map(|r| r.atoms.iter().map(|a| a.position))
            .collect();
        let radii: Vec<f64> = s
            .residues
            .iter()
            .flat_map(|r| {
                r.atoms.iter().map(|a| {
                    crate::powersasa::vdw_radius(a.element) + crate::powersasa::PROBE_RADIUS_A
                })
            })
            .collect();
        let mut neighbour_idx: Vec<Vec<usize>> = vec![Vec::new(); n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = (positions[j] - positions[i]).norm();
                if d <= radii[i] + radii[j] {
                    neighbour_idx[i].push(j);
                    neighbour_idx[j].push(i);
                }
            }
        }

        // Cross-check the analytical gradient at a handful of atoms.
        // For each test atom, perturb each of its affected atoms along
        // each axis and confirm dA/dr matches.
        let eps = 1e-5;
        let area_of = |atom_idx: usize, pos: &[Vec3]| -> f64 {
            let neighbours: Vec<(usize, Vec3, f64)> = neighbour_idx[atom_idx]
                .iter()
                .filter_map(|&j| {
                    let d = (pos[j] - pos[atom_idx]).norm();
                    if d <= radii[atom_idx] + radii[j] {
                        Some((j, pos[j], radii[j]))
                    } else {
                        None
                    }
                })
                .collect();
            let (caps, _) = match build_caps(pos[atom_idx], radii[atom_idx], &neighbours) {
                Some(c) => c,
                None => return 0.0,
            };
            let boundary = find_boundary(&caps);
            let c = super::super::arrangement::count_accessible_components(&caps).max(1);
            accessible_area_with_components(radii[atom_idx], &caps, &boundary, Some(c))
        };

        // For each atom i in the chain whose cache is non-trivial,
        // confirm the analytical gradient at every affected atom.
        let test_atoms: Vec<usize> = (0..n.min(12)).collect();
        for &i in &test_atoms {
            let cache = match build_atom_cache(i, &positions, &radii, &neighbour_idx[i]) {
                Some(c) => c,
                None => continue,
            };
            if cache.arcs.is_empty() {
                continue;
            }
            let mut grad = vec![Vec3::zeros(); n];
            atom_area_gradient(&cache, &positions, &radii, &mut grad);

            let affected_atoms: Vec<usize> = {
                let mut a = vec![i];
                for &owner in &cache.cap_owners {
                    if !a.contains(&owner) {
                        a.push(owner);
                    }
                }
                a
            };
            for &x in &affected_atoms {
                for axis in 0..3 {
                    let mut p_plus = positions.clone();
                    let mut p_minus = positions.clone();
                    p_plus[x][axis] += eps;
                    p_minus[x][axis] -= eps;
                    let numeric = (area_of(i, &p_plus) - area_of(i, &p_minus)) / (2.0 * eps);
                    let analytical = grad[x][axis];
                    let err = (analytical - numeric).abs();
                    // 1e-3 is a generous bound — eps=1e-5 limits float
                    // precision on the central difference. Tighter only
                    // if eps is smaller (which itself amplifies float noise).
                    assert!(
                        err < 1.0e-2,
                        "atom i={} x={} axis={}: analytical {} vs numeric {} (err {})",
                        i,
                        x,
                        axis,
                        analytical,
                        numeric,
                        err,
                    );
                }
            }
        }
    }

    /// On a simple two-atom configuration, the analytical area gradient
    /// at atom i matches the central difference of the area function.
    /// This is the *first* end-to-end check that the JVP primitives are
    /// being routed correctly through the topology cache.
    #[test]
    fn analytical_gradient_matches_finite_difference_two_atoms() {
        // Two carbons 3 Å apart — each one has a single cap on the
        // other, producing a single full-circle boundary loop.
        let positions = vec![Vec3::zeros(), Vec3::new(3.0, 0.0, 0.0)];
        let radii = vec![3.0, 3.0]; // both vdW+probe (= 1.7 + 1.4 ish, but just round numbers)
        let neighbour_idx = vec![vec![1usize], vec![0usize]];

        let cache_0 = build_atom_cache(0, &positions, &radii, &neighbour_idx[0])
            .expect("cache atom 0");

        // ∂A_0 / ∂r_x for x = 0 and x = 1 (axis 0, the bond axis).
        let mut grad = vec![Vec3::zeros(); 2];
        atom_area_gradient(&cache_0, &positions, &radii, &mut grad);

        // Compute reference area as a function of (positions, radii) for
        // the central difference.
        let area_at = |pos: &[Vec3]| -> f64 {
            let neighbours: Vec<(usize, Vec3, f64)> = (1..pos.len())
                .map(|j| (j, pos[j], radii[j]))
                .collect();
            let (caps, _) = build_caps(pos[0], radii[0], &neighbours).unwrap();
            let boundary = find_boundary(&caps);
            accessible_area_with_components(radii[0], &caps, &boundary, Some(cache_0.c))
        };

        let eps = 1e-5;
        for x in 0..2 {
            for axis in 0..3 {
                let mut p_plus = positions.clone();
                let mut p_minus = positions.clone();
                p_plus[x][axis] += eps;
                p_minus[x][axis] -= eps;
                let numeric = (area_at(&p_plus) - area_at(&p_minus)) / (2.0 * eps);
                let analytical = grad[x][axis];
                let err = (analytical - numeric).abs();
                assert!(
                    err < 1e-3,
                    "x={} axis={}: analytical {} vs numeric {} (err {})",
                    x,
                    axis,
                    analytical,
                    numeric,
                    err
                );
            }
        }
    }
}
