//! Per-atom arrangement of buried caps and accessible-region boundary.
//!
//! For atom *i*, each neighbour atom whose vdW sphere overlaps *i*'s
//! defines a buried spherical cap on *i*'s sphere. The accessible region is
//! the complement of the union of caps. The accessible region's boundary
//! is a collection of arcs of the cap circles, joined at the intersection
//! points of pairs of cap circles.
//!
//! This module computes that boundary as a list of [`BoundaryArc`]s ready
//! for the Gauss-Bonnet area integration in `area.rs`.

use geom::Vec3;

use super::geometry::{intersect_circles, CircleIntersection, SmallCircle};

/// One arc on the boundary of the accessible region.
#[derive(Debug, Clone, Copy)]
pub struct BoundaryArc {
    /// Which input cap this arc lies on (index into the `caps` vector).
    pub cap_idx: usize,
    /// Start point on the unit sphere (on the cap's small circle).
    pub start: Vec3,
    /// End point on the unit sphere.
    pub end: Vec3,
    /// Signed central angle from start to end around the cap's axis, in
    /// `(-2π, 2π]`. Positive when traversal is CCW looking down the axis;
    /// for a typical boundary arc this is negative (we traverse the
    /// boundary CCW around the *accessible* region, which is CW around the
    /// cap interior).
    pub theta: f64,
    /// `true` if this arc is the full circle (the cap has no
    /// intersections with any other cap).
    pub is_full_circle: bool,
}

/// One vertex at the junction of two boundary arcs.
#[derive(Debug, Clone, Copy)]
pub struct BoundaryVertex {
    /// Point on the unit sphere.
    pub point: Vec3,
    /// Index of the cap whose arc ARRIVES at this vertex.
    pub incoming_cap: usize,
    /// Index of the cap whose arc LEAVES this vertex.
    pub outgoing_cap: usize,
}

/// The accessible-region boundary on an atom's sphere.
#[derive(Debug, Clone)]
pub enum AtomBoundary {
    /// No caps cover any part of the sphere → A = 4πR².
    FullyExposed,
    /// At least one cap covers the entire sphere → A = 0.
    FullyBuried,
    /// Accessible region is one or more components bounded by arcs.
    Bounded {
        arcs: Vec<BoundaryArc>,
        vertices: Vec<BoundaryVertex>,
    },
}

/// Build small-circle caps on atom `i`'s unit sphere from a list of
/// candidate neighbour atoms. Each cap represents the region of *i*'s
/// surface buried by the corresponding neighbour.
///
/// Returns `(caps, neighbour_indices)` — caps and the neighbour atom index
/// each cap was derived from. If atom *i* is fully enclosed in any
/// neighbour, returns `None` and the caller should treat the atom as
/// fully buried.
pub fn build_caps(
    p_i: Vec3,
    r_i: f64,
    neighbours: &[(usize, Vec3, f64)], // (idx, position, radius)
) -> Option<(Vec<SmallCircle>, Vec<usize>)> {
    let mut caps = Vec::new();
    let mut owners = Vec::new();
    for &(idx, p_j, r_j) in neighbours {
        let d = (p_j - p_i).norm();
        // Atom i fully inside atom j → no surface accessible at all.
        if d + r_i <= r_j {
            return None;
        }
        // Atom j fully inside atom i, or non-overlapping → no cap to add.
        if d + r_j <= r_i || d >= r_i + r_j {
            continue;
        }
        let axis = (p_j - p_i) / d;
        let cos_alpha = (d * d + r_i * r_i - r_j * r_j) / (2.0 * d * r_i);
        if !(-1.0..=1.0).contains(&cos_alpha) {
            // Defensive — shouldn't happen given the inequalities above.
            continue;
        }
        caps.push(SmallCircle { axis, cos_alpha });
        owners.push(idx);
    }
    Some((caps, owners))
}

/// Build the per-atom boundary from the caps. Brute-force O(n²) pairwise
/// intersections; n is the per-atom cap count, typically < 30.
pub fn find_boundary(caps: &[SmallCircle]) -> AtomBoundary {
    if caps.is_empty() {
        return AtomBoundary::FullyExposed;
    }

    // For each cap, collect the angular positions of its intersections
    // with every other cap. A vertex is on the *boundary of the accessible
    // region* only if it isn't strictly inside any third cap.
    let n = caps.len();
    // For each cap, an angular-sorted list of intersection points (with the
    // other-cap index) that lie on the accessible-region boundary.
    let mut per_cap_vertices: Vec<Vec<(f64, Vec3, usize)>> = vec![Vec::new(); n];

    for k in 0..n {
        for l in (k + 1)..n {
            let pts = match intersect_circles(caps[k], caps[l]) {
                CircleIntersection::Two(p, q) => vec![p, q],
                CircleIntersection::Tangent(p) => vec![p],
                CircleIntersection::Disjoint | CircleIntersection::Coincident => continue,
            };
            for pt in pts {
                // Reject if pt is strictly inside any third cap.
                let mut inside_third = false;
                for (m, cap_m) in caps.iter().enumerate() {
                    if m == k || m == l {
                        continue;
                    }
                    if cap_m.contains_strict(pt) {
                        inside_third = true;
                        break;
                    }
                }
                if inside_third {
                    continue;
                }
                // Record on both caps. Use the angular position from a
                // chosen reference direction on each circle for sorting.
                per_cap_vertices[k].push((angular_position(pt, caps[k]), pt, l));
                per_cap_vertices[l].push((angular_position(pt, caps[l]), pt, k));
            }
        }
    }

    // For each cap, if it has zero intersection vertices on its circle,
    // it's either a "full closed boundary loop" or fully hidden by other
    // caps. Test one representative point on the circle.
    // If it has vertices, sort them by angular position and connect
    // consecutive ones with arcs (subject to the test that the arc midpoint
    // is also outside every other cap).
    let mut arcs: Vec<BoundaryArc> = Vec::new();
    let mut vertices: Vec<BoundaryVertex> = Vec::new();
    for (k, verts) in per_cap_vertices.iter().enumerate() {
        if verts.is_empty() {
            // Test one representative point on this circle.
            let probe = sample_circle_point(caps[k]);
            let buried = caps.iter().enumerate().any(|(m, cap_m)| {
                m != k && cap_m.contains_strict(probe)
            });
            if !buried {
                // The whole circle is a boundary loop. Traverse CW around the
                // cap (CCW around accessible region) → θ_arc = −2π.
                arcs.push(BoundaryArc {
                    cap_idx: k,
                    start: probe,
                    end: probe,
                    theta: -2.0 * std::f64::consts::PI,
                    is_full_circle: true,
                });
            }
            continue;
        }

        // Sort by angular position around the cap axis.
        let mut sorted: Vec<&(f64, Vec3, usize)> = verts.iter().collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // For m vertices on a circle, there are m candidate arcs (CCW
        // segments between angularly-consecutive vertices). For each, test
        // whether its midpoint is outside every other cap. If so, the
        // segment is on the boundary of the accessible region.
        //
        // The boundary walk traverses the accessible region CCW with the
        // region on its left. On a cap k, that means CW around cap k's
        // axis — opposite the CCW segment we selected. So the arc's
        // boundary-walk endpoints are swapped relative to the CCW pair
        // and its signed θ around the axis is −ccw_norm.
        let m = sorted.len();
        for i in 0..m {
            let (a_angle, a_pt, a_other) = *sorted[i];
            let (b_angle, b_pt, _b_other) = *sorted[(i + 1) % m];
            let ccw_norm = wrap_2pi_positive(b_angle - a_angle);
            let mid = midpoint_on_circle(a_pt, caps[k], ccw_norm / 2.0);
            let mid_ok = !caps.iter().enumerate().any(|(idx, cap_m)| {
                idx != k && cap_m.contains_strict(mid)
            });
            if !mid_ok {
                continue;
            }
            // Boundary walk: enter at b_pt, exit at a_pt, going CW around
            // cap k's axis (negative theta).
            arcs.push(BoundaryArc {
                cap_idx: k,
                start: b_pt,
                end: a_pt,
                theta: -ccw_norm,
                is_full_circle: false,
            });
            // Vertex at the end of this arc is `a_pt`. Its incoming cap is
            // k; its outgoing cap is the other cap that creates this
            // intersection point (`a_other`).
            vertices.push(BoundaryVertex {
                point: a_pt,
                incoming_cap: k,
                outgoing_cap: a_other,
            });
        }
    }

    // Detect fully-buried: every probe point hidden, no arcs survived.
    if arcs.is_empty() && every_point_buried(caps) {
        return AtomBoundary::FullyBuried;
    }

    AtomBoundary::Bounded { arcs, vertices }
}

/// Angular position of a point on a small circle relative to a reference
/// direction in the circle's plane. Used only as a sort key — the absolute
/// reference doesn't matter as long as it's consistent.
fn angular_position(point: Vec3, circle: SmallCircle) -> f64 {
    // Build a reference vector orthogonal to circle.axis. Pick one of the
    // global axes that's least parallel to the circle's axis.
    let abs = circle.axis.abs();
    let helper = if abs.x < abs.y && abs.x < abs.z {
        Vec3::new(1.0, 0.0, 0.0)
    } else if abs.y < abs.z {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };
    let u = (helper - circle.axis * helper.dot(&circle.axis)).normalize();
    let v = circle.axis.cross(&u);
    let proj = point - circle.axis * point.dot(&circle.axis);
    proj.dot(&v).atan2(proj.dot(&u))
}

/// Generate a representative point on a small circle (for "is this circle
/// inside any other cap?" probes when the circle has no intersections).
fn sample_circle_point(circle: SmallCircle) -> Vec3 {
    let abs = circle.axis.abs();
    let helper = if abs.x < abs.y && abs.x < abs.z {
        Vec3::new(1.0, 0.0, 0.0)
    } else if abs.y < abs.z {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };
    let perp = (helper - circle.axis * helper.dot(&circle.axis)).normalize();
    circle.axis * circle.cos_alpha + perp * circle.sin_alpha()
}

/// Rotate a point on a small circle around its axis by `delta` radians.
/// The result stays on the same small circle.
fn midpoint_on_circle(point: Vec3, circle: SmallCircle, delta: f64) -> Vec3 {
    let axial = circle.axis * point.dot(&circle.axis);
    let radial = point - axial;
    let r_norm = radial.norm();
    if r_norm < 1e-12 {
        return point;
    }
    let u = radial / r_norm;
    let v = circle.axis.cross(&u);
    axial + (u * delta.cos() + v * delta.sin()) * r_norm
}

fn wrap_2pi_positive(theta: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut t = theta;
    while t <= 0.0 {
        t += two_pi;
    }
    while t > two_pi {
        t -= two_pi;
    }
    t
}

fn every_point_buried(caps: &[SmallCircle]) -> bool {
    // Brute-force check: sample many directions and verify each is inside
    // at least one cap. Not foolproof but adequate for catching the
    // fully-buried case in protein topology.
    for i in 0..200 {
        let phi = (i as f64) * 0.39 + 0.1;
        let theta = (i as f64) * 0.71 + 0.2;
        let p = Vec3::new(
            phi.sin() * theta.cos(),
            phi.sin() * theta.sin(),
            phi.cos(),
        )
        .normalize();
        if !caps.iter().any(|cap| cap.contains_strict(p)) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn v(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3::new(x, y, z)
    }

    #[test]
    fn zero_caps_fully_exposed() {
        assert!(matches!(find_boundary(&[]), AtomBoundary::FullyExposed));
    }

    #[test]
    fn one_cap_produces_full_circle_arc() {
        // Single cap covering the northern hemisphere (axis = +z, cos α = 0.3).
        let caps = vec![SmallCircle::new(v(0.0, 0.0, 1.0), 0.3)];
        match find_boundary(&caps) {
            AtomBoundary::Bounded { arcs, vertices } => {
                assert_eq!(arcs.len(), 1);
                assert!(arcs[0].is_full_circle);
                assert_relative_eq!(arcs[0].theta, -2.0 * std::f64::consts::PI, epsilon = 1e-12);
                assert!(vertices.is_empty());
            }
            other => panic!("expected Bounded with one full-circle arc, got {:?}", other),
        }
    }

    #[test]
    fn fully_buried_when_axis_aligned_caps_cover_whole_sphere() {
        // Two caps with axes ±z and cos α = -0.5 each cover more than a
        // hemisphere; together they cover the whole sphere.
        let caps = vec![
            SmallCircle::new(v(0.0, 0.0, 1.0), -0.5),
            SmallCircle::new(v(0.0, 0.0, -1.0), -0.5),
        ];
        match find_boundary(&caps) {
            AtomBoundary::FullyBuried => {}
            other => panic!("expected FullyBuried, got {:?}", other),
        }
    }

    #[test]
    fn two_disjoint_caps_each_contribute_a_full_circle() {
        // Two small caps near opposite poles, disjoint.
        let caps = vec![
            SmallCircle::new(v(0.0, 0.0, 1.0), 0.8),
            SmallCircle::new(v(0.0, 0.0, -1.0), 0.8),
        ];
        match find_boundary(&caps) {
            AtomBoundary::Bounded { arcs, vertices } => {
                assert_eq!(arcs.len(), 2);
                assert!(arcs.iter().all(|a| a.is_full_circle));
                assert!(vertices.is_empty());
            }
            other => panic!("expected two full-circle arcs, got {:?}", other),
        }
    }

    #[test]
    fn two_intersecting_caps_produce_two_arcs() {
        // Two caps at right angles, both with cos α = 0.3 (so α ≈ 72°).
        // Each pair of circles intersects in two points → two boundary arcs.
        let caps = vec![
            SmallCircle::new(v(0.0, 0.0, 1.0), 0.3),
            SmallCircle::new(v(1.0, 0.0, 0.0), 0.3),
        ];
        match find_boundary(&caps) {
            AtomBoundary::Bounded { arcs, vertices } => {
                // Each cap contributes one arc between the two intersection points.
                assert_eq!(arcs.len(), 2);
                assert!(arcs.iter().all(|a| !a.is_full_circle));
                assert_eq!(vertices.len(), 2);
            }
            other => panic!("expected two arcs, got {:?}", other),
        }
    }
}
