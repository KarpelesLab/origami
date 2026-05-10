//! Gauss-Bonnet area integration for the accessible region.
//!
//! For a region M on a sphere of radius R bounded by piecewise small-circle
//! arcs joined at vertices, A satisfies (informally)
//! A/R^2 = 2 pi chi(M) - sum_arcs cos(alpha)*theta - sum_vertices epsilon
//! where chi(M) = 2 minus the number of boundary loops for a simply-bounded
//! region on the sphere; alpha is the arc cone half-angle; theta is the
//! signed central angle (sign by traversal); epsilon is the exterior angle
//! at the vertex (positive = turn toward the accessible region).

use super::arrangement::{AtomBoundary, BoundaryArc, BoundaryVertex};
use super::geometry::SmallCircle;

#[cfg(test)]
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
    // For each boundary loop independently, apply the simply-connected form
    // A_loop = R²(2π − arc_sum_loop − vertex_sum_loop). This treats the
    // loop's "left side" (accessible side, by our boundary-walk convention)
    // as a topological disk.
    //
    // To assemble these into the total accessible area we have two
    // candidate formulas:
    //   - **Disjoint patches**: A = Σ A_loop. Correct when the loops bound
    //     k separate accessible disks (typical for crowded atoms where the
    //     buried region is one connected blob with multiple "windows").
    //   - **Annular interpretation**: A = Σ A_loop − (k−1)·4πR². Correct
    //     when there's one big accessible region with k buried holes
    //     (typical when the buried regions are disjoint disks).
    // Both formulas agree when k = 1.
    //
    // Pick the interpretation whose result is a physically valid area in
    // [0, 4πR²]. For typical protein geometry one of the two is in range.

    let r_sq = radius * radius;
    let four_pi_r2 = 4.0 * std::f64::consts::PI * r_sq;

    let loops = collect_loops(arcs, vertices, caps);
    let k = loops.len();
    if k == 0 {
        return 0.0;
    }
    let mut sum_a_loop = 0.0;
    for la in &loops {
        sum_a_loop += r_sq * (2.0 * std::f64::consts::PI - la.arc_sum - la.vertex_sum);
    }

    // Two candidate interpretations of the loop layout:
    //   - disjoint patches (k accessible disks):  A = Σ A_loop
    //   - annular (one region with k buried holes): A = Σ A_loop − (k−1)·4πR²
    // Both candidates collapse to the same value when k = 1.
    let candidate_disjoint = sum_a_loop;
    let candidate_annular = sum_a_loop - (k as f64 - 1.0) * four_pi_r2;

    let in_range = |a: f64| (-1e-3..=four_pi_r2 + 1e-3).contains(&a);
    let result = match (in_range(candidate_disjoint), in_range(candidate_annular)) {
        (true, true) => candidate_disjoint.min(candidate_annular),
        (true, false) => candidate_disjoint,
        (false, true) => candidate_annular,
        (false, false) => candidate_disjoint.clamp(0.0, four_pi_r2),
    };
    result.clamp(0.0, four_pi_r2)
}

struct LoopSums {
    arc_sum: f64,
    vertex_sum: f64,
}

/// Walk the boundary arcs to identify connected loops, summing arc-cos-θ
/// and vertex-exterior-angle within each loop separately.
fn collect_loops(
    arcs: &[BoundaryArc],
    vertices: &[BoundaryVertex],
    caps: &[SmallCircle],
) -> Vec<LoopSums> {
    let mut loops: Vec<LoopSums> = Vec::new();

    // Full-circle arcs: each is its own loop with zero vertex contribution.
    for (i, arc) in arcs.iter().enumerate() {
        if arc.is_full_circle {
            let cap = caps[arc.cap_idx];
            loops.push(LoopSums {
                arc_sum: cap.cos_alpha * arc.theta,
                vertex_sum: 0.0,
            });
            let _ = i;
        }
    }

    // Vertexed arcs: walk by matching arc.end → next arc.start.
    let n = arcs.len();
    let mut visited = vec![false; n];
    for (i, arc) in arcs.iter().enumerate() {
        if arc.is_full_circle {
            visited[i] = true;
        }
    }
    let close = |a: geom::Vec3, b: geom::Vec3| (a - b).norm() < 1e-6;
    for start in 0..n {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut path = vec![start];
        let mut current = start;
        loop {
            let next = (0..n).find(|&i| {
                !visited[i] && !arcs[i].is_full_circle && close(arcs[i].start, arcs[current].end)
            });
            match next {
                Some(j) => {
                    visited[j] = true;
                    path.push(j);
                    current = j;
                }
                None => break,
            }
        }
        // Sum within this path.
        let mut arc_sum = 0.0;
        let mut vertex_sum = 0.0;
        for &arc_idx in &path {
            let cap = caps[arcs[arc_idx].cap_idx];
            arc_sum += cap.cos_alpha * arcs[arc_idx].theta;
            // The vertex at the end of this arc is `vertices[arc_idx]`.
            let vertex = vertices[arc_idx];
            let v = vertex.point;
            let t_in = v.cross(&caps[vertex.incoming_cap].axis).normalize();
            let t_out = v.cross(&caps[vertex.outgoing_cap].axis).normalize();
            let cos_eps = t_in.dot(&t_out).clamp(-1.0, 1.0);
            let sin_eps = v.dot(&t_in.cross(&t_out));
            vertex_sum += sin_eps.atan2(cos_eps);
        }
        loops.push(LoopSums { arc_sum, vertex_sum });
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
