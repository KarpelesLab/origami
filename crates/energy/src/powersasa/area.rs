//! Gauss-Bonnet area integration for the accessible region.
//!
//! For a region M on a sphere of radius R bounded by piecewise small-circle
//! arcs joined at vertices, A satisfies (informally)
//! A/R^2 = 2 pi chi(M) - sum_arcs cos(alpha)*theta - sum_vertices epsilon
//! where chi(M) = 2 minus the number of boundary loops for a simply-bounded
//! region on the sphere; alpha is the arc cone half-angle; theta is the
//! signed central angle (sign by traversal); epsilon is the exterior angle
//! at the vertex (positive = turn toward the accessible region).

use super::arrangement::{count_accessible_components, AtomBoundary, BoundaryArc, BoundaryVertex};
use super::geometry::SmallCircle;

use geom::Vec3;

/// Return the accessible-surface area of an atom given the precomputed
/// boundary topology. `radius` is the *physical* radius (vdW + probe) — not
/// the unit-sphere radius.
pub fn accessible_area(radius: f64, caps: &[SmallCircle], boundary: &AtomBoundary) -> f64 {
    match boundary {
        AtomBoundary::FullyExposed => 4.0 * std::f64::consts::PI * radius * radius,
        AtomBoundary::FullyBuried => 0.0,
        AtomBoundary::Bounded { arcs, vertices } => bounded_area(radius, caps, arcs, vertices),
    }
}

fn bounded_area(
    radius: f64,
    caps: &[SmallCircle],
    arcs: &[BoundaryArc],
    vertices: &[BoundaryVertex],
) -> f64 {
    // Spherical Gauss-Bonnet, generalised to multi-component / multi-loop
    // accessible regions.
    //
    // For a connected region M ⊂ S² bounded by L_M loops:
    //   χ(M) = 2 − L_M     (consequence of χ(S²) = 2 and complement = L_M
    //                       disjoint disks for typical cap arrangements)
    //
    // For disconnected M with c components, χ is additive:
    //   χ(M) = 2c − L      (L = total loops, c = total accessible
    //                       connected components)
    //
    // Plug into Gauss-Bonnet on a curvature-K = 1/R² sphere:
    //   A/R² = 2π χ(M) − Σ_arcs cos(α)·θ − Σ_vertices ε
    //        = 2π(2c − L) − S
    //
    // The original PSA.1 implementation tried to choose between two
    // candidate values of χ heuristically (the "disjoint" χ = c = L and
    // the "annular" χ = 2 − L); for crowded atoms (≥ ~10 caps) neither
    // candidate is right. Counting `c` directly via accessible-side
    // probes is robust and the formula then handles all topologies
    // uniformly.

    let r_sq = radius * radius;
    let four_pi_r2 = 4.0 * std::f64::consts::PI * r_sq;

    if arcs.is_empty() {
        return 0.0;
    }

    // Total arc-length and vertex-turning contributions across the entire
    // boundary. We don't need to walk loops — we just need the sums to plug
    // into Gauss-Bonnet alongside the global χ.
    let mut total_arc_sum = 0.0;
    for arc in arcs {
        let cap = caps[arc.cap_idx];
        total_arc_sum += cap.cos_alpha * arc.theta;
    }
    let mut total_vertex_sum = 0.0;
    for vertex in vertices {
        let v = vertex.point;
        let t_in = v.cross(&caps[vertex.incoming_cap].axis).normalize();
        let t_out = v.cross(&caps[vertex.outgoing_cap].axis).normalize();
        let cos_eps = t_in.dot(&t_out).clamp(-1.0, 1.0);
        let sin_eps = v.dot(&t_in.cross(&t_out));
        total_vertex_sum += sin_eps.atan2(cos_eps);
    }

    // L = number of closed boundary loops on the accessible region,
    // computed as the number of connected components of the graph whose
    // nodes are boundary-vertex positions and edges are arcs. The
    // alternative — walking arcs via (end-pos, outgoing-cap) matching —
    // over-counts L when 3+-way vertex concurrencies are present (the
    // recorded `outgoing_cap` on the vertex may not match the actual
    // face-traversal continuation), under-counting area to zero on a
    // number of small-window atoms. The graph approach mis-fuses loops
    // that *legitimately* share a vertex point, which slightly over-
    // estimates area on aromatic ring carbons — but the failure mode
    // is bounded by the per-atom sphere area instead of "zero out a
    // valid atom". A proper half-edge face-traversal (tracked as
    // PSA.1h-followup) would resolve both directions cleanly.
    let l = count_boundary_loops_face_walk(arcs, caps);
    if l == 0 {
        return 0.0;
    }

    // Probe-based component count. Each accessible component must
    // contribute ≥ 1 boundary loop on the sphere, so `c ≤ L` is a hard
    // topological invariant — clamp accordingly. Probes can sometimes
    // over-count when an accessible region has a thin neck the probe
    // grid doesn't bridge.
    let c_raw = count_accessible_components(caps);
    let c = c_raw.min(l).max(1);
    // χ for the global accessible region: with c connected components
    // each contributing 2 − L_F to the Euler characteristic (per
    // Gauss-Bonnet on a sphere), summing over all components:
    //   χ = Σ (2 − L_F) = 2c − L_total
    let chi = 2 * c as i64 - l as i64;
    let area =
        r_sq * (2.0 * std::f64::consts::PI * chi as f64 - total_arc_sum - total_vertex_sum);
    area.clamp(0.0, four_pi_r2)
}

/// Loop count via half-edge-style face walking. At each vertex, when
/// multiple arcs continue the boundary, we pick the one whose outgoing
/// tangent is the immediate CCW successor of the incoming arc's tangent
/// (right-turn around the vertex's outward normal), so 3+-way vertex
/// concurrencies are disambiguated by local geometry rather than by
/// arbitrary list order.
///
/// For each arc, the next arc is determined as:
///   1. Match start position to current end position (within tolerance).
///   2. Among matches, pick the one whose tangent at the vertex has the
///      smallest positive CCW angle from the incoming tangent.
///
/// Two-way vertices have exactly one match and behave identically to a
/// simple positional walker.
fn count_boundary_loops_face_walk(arcs: &[BoundaryArc], caps: &[SmallCircle]) -> usize {
    let n = arcs.len();
    let mut visited = vec![false; n];
    let mut loops = 0usize;
    let eps_sq = 1e-10;
    let two_pi = 2.0 * std::f64::consts::PI;

    for start in 0..n {
        if visited[start] {
            continue;
        }
        if arcs[start].is_full_circle {
            visited[start] = true;
            loops += 1;
            continue;
        }
        let mut current = start;
        loop {
            visited[current] = true;
            let end_pt = arcs[current].end;
            let in_cap_axis = caps[arcs[current].cap_idx].axis;
            // Tangent at `end_pt` for the incoming arc. Our arcs go CW
            // around the cap axis (θ negative), so the tangent of motion
            // at any point p on the cap circle is p × ω_cap.
            let t_in = end_pt.cross(&in_cap_axis).normalize();

            let candidates: Vec<usize> = (0..n)
                .filter(|&i| {
                    !visited[i]
                        && !arcs[i].is_full_circle
                        && (arcs[i].start - end_pt).norm_squared() < eps_sq
                })
                .collect();
            let next = match candidates.len() {
                0 => None,
                1 => Some(candidates[0]),
                _ => {
                    // Pick by tangent angle: smallest CCW angle from t_in
                    // to t_out, measured around the outward normal end_pt.
                    let mut best = candidates[0];
                    let mut best_angle = f64::INFINITY;
                    for &c in &candidates {
                        let out_cap_axis = caps[arcs[c].cap_idx].axis;
                        let t_out = end_pt.cross(&out_cap_axis).normalize();
                        let dot = t_in.dot(&t_out).clamp(-1.0, 1.0);
                        let cross = t_in.cross(&t_out);
                        let signed = end_pt.dot(&cross);
                        let mut angle = signed.atan2(dot);
                        if angle <= 1e-9 {
                            angle += two_pi;
                        }
                        if angle < best_angle {
                            best_angle = angle;
                            best = c;
                        }
                    }
                    Some(best)
                }
            };
            match next {
                Some(j) => current = j,
                None => break,
            }
        }
        loops += 1;
    }
    loops
}

/// Loop count via graph connectivity over arc-endpoints. Treats every arc
/// as an undirected edge between its start and end vertex points (with
/// vertices identified by position). Each connected component of the
/// resulting graph is one loop. Mis-fuses loops that share a vertex.
#[allow(dead_code)]
fn count_boundary_loops_graph(arcs: &[BoundaryArc]) -> usize {
    let mut full_circle_loops = 0usize;
    let mut vertex_points: Vec<Vec3> = Vec::new();
    let mut arc_edges: Vec<(usize, usize)> = Vec::new();
    let eps_sq = 1e-10;
    for arc in arcs {
        if arc.is_full_circle {
            full_circle_loops += 1;
            continue;
        }
        let s = find_or_add_point(&mut vertex_points, arc.start, eps_sq);
        let e = find_or_add_point(&mut vertex_points, arc.end, eps_sq);
        arc_edges.push((s, e));
    }
    if vertex_points.is_empty() {
        return full_circle_loops;
    }
    let n = vertex_points.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn root(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    for &(s, e) in &arc_edges {
        let rs = root(&mut parent, s);
        let re = root(&mut parent, e);
        if rs != re {
            parent[rs] = re;
        }
    }
    let mut roots: Vec<usize> = Vec::new();
    for i in 0..n {
        let r = root(&mut parent, i);
        if !roots.contains(&r) {
            roots.push(r);
        }
    }
    full_circle_loops + roots.len()
}

fn find_or_add_point(points: &mut Vec<Vec3>, p: Vec3, eps_sq: f64) -> usize {
    for (i, q) in points.iter().enumerate() {
        if (p - *q).norm_squared() < eps_sq {
            return i;
        }
    }
    points.push(p);
    points.len() - 1
}

/// Loop count via arc traversal: walk arcs, at each step picking the
/// next one by matching (end position, outgoing cap). Stops when no
/// next arc matches, so 3+-way concurrencies with mis-recorded
/// `outgoing_cap` make this over-count L. Currently unused — kept for
/// future half-edge face-traversal work (PSA.1h-followup).
#[allow(dead_code)]
fn count_boundary_loops_walked(arcs: &[BoundaryArc], vertices: &[BoundaryVertex]) -> usize {
    let n = arcs.len();
    let mut visited = vec![false; n];
    let mut loops = 0usize;
    let eps_sq = 1e-10;
    for start in 0..n {
        if visited[start] {
            continue;
        }
        if arcs[start].is_full_circle {
            visited[start] = true;
            loops += 1;
            continue;
        }
        let mut current = start;
        loop {
            visited[current] = true;
            let end_pt = arcs[current].end;
            let outgoing_cap = vertices[current].outgoing_cap;
            let next = (0..n).find(|&i| {
                !visited[i]
                    && !arcs[i].is_full_circle
                    && arcs[i].cap_idx == outgoing_cap
                    && (arcs[i].start - end_pt).norm_squared() < eps_sq
            });
            match next {
                Some(j) => current = j,
                None => break,
            }
        }
        loops += 1;
    }
    loops
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::powersasa::arrangement::find_boundary;
    use approx::assert_relative_eq;

    fn v(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3::new(x, y, z)
    }

    #[test]
    fn isolated_atom_full_sphere_area() {
        let area = accessible_area(2.0, &[], &AtomBoundary::FullyExposed);
        let expected = 4.0 * std::f64::consts::PI * 4.0;
        assert_relative_eq!(area, expected, epsilon = 1e-9);
    }

    #[test]
    fn fully_buried_zero_area() {
        let area = accessible_area(2.0, &[], &AtomBoundary::FullyBuried);
        assert_eq!(area, 0.0);
    }

    #[test]
    fn single_cap_matches_analytical_formula() {
        // Single cap, axis = +z, cos α = 0.3 (α ≈ 72.5°). Analytic:
        // A_accessible = 2π R² (1 + cos α).
        let r = 1.0;
        let cos_a = 0.3;
        let caps = vec![SmallCircle::new(v(0.0, 0.0, 1.0), cos_a)];
        let boundary = find_boundary(&caps);
        let area = accessible_area(r, &caps, &boundary);
        let expected = 2.0 * std::f64::consts::PI * r * r * (1.0 + cos_a);
        assert_relative_eq!(area, expected, epsilon = 1e-9);
    }

    #[test]
    fn two_disjoint_caps_match_sum_formula() {
        // Two disjoint small caps on opposite poles. A = 4πR² − A_cap1 − A_cap2.
        let r = 1.5;
        let cos_a1 = 0.7;
        let cos_a2 = 0.6;
        let caps = vec![
            SmallCircle::new(v(0.0, 0.0, 1.0), cos_a1),
            SmallCircle::new(v(0.0, 0.0, -1.0), cos_a2),
        ];
        let boundary = find_boundary(&caps);
        let area = accessible_area(r, &caps, &boundary);
        let cap1 = 2.0 * std::f64::consts::PI * r * r * (1.0 - cos_a1);
        let cap2 = 2.0 * std::f64::consts::PI * r * r * (1.0 - cos_a2);
        let expected = 4.0 * std::f64::consts::PI * r * r - cap1 - cap2;
        assert_relative_eq!(area, expected, epsilon = 1e-9);
    }

    #[test]
    fn two_intersecting_caps_positive_finite_area() {
        // Two caps at right angles, each with cos α = 0.3 (α ≈ 72°).
        // Quantitatively complex; just check we get a positive finite area
        // bounded above by 4πR² and below by 2πR²(1+cos α) (single-cap area).
        let r = 1.0;
        let caps = vec![
            SmallCircle::new(v(0.0, 0.0, 1.0), 0.3),
            SmallCircle::new(v(1.0, 0.0, 0.0), 0.3),
        ];
        let boundary = find_boundary(&caps);
        let area = accessible_area(r, &caps, &boundary);
        assert!(area > 0.0, "area should be positive, got {}", area);
        assert!(area < 4.0 * std::f64::consts::PI, "area exceeds full sphere");
        // Two intersecting caps cover MORE area than one alone, so the
        // accessible area should be LESS than the single-cap accessible area.
        let single = 2.0 * std::f64::consts::PI * (1.0 + 0.3);
        assert!(area < single, "two caps should bury more than one: {} vs {}", area, single);
    }
}
