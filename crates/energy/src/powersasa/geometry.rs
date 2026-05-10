//! Spherical geometry primitives for PowerSasa.
//!
//! All routines work on the **unit sphere**. The caller maps points back to
//! the atom's physical sphere by multiplying by the atomic radius. Working
//! in unit-sphere coordinates keeps the formulas dimensionless and lets the
//! same `SmallCircle` type be reused as the boundary of any spherical cap.

use geom::Vec3;

/// A small circle on the unit sphere — the locus of points x with
/// `x · axis = cos_alpha`. The "cap" associated with the circle is the set
/// of points with `x · axis > cos_alpha` (strict inequality; the circle
/// itself is the boundary).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SmallCircle {
    /// Cone axis (unit vector pointing toward the cap interior).
    pub axis: Vec3,
    /// Cosine of the cone half-angle α ∈ (0, π).
    /// α = π/2 ⇔ cos_alpha = 0 ⇔ great circle.
    /// α near 0 ⇔ cos_alpha near 1 ⇔ small cap, narrow circle.
    /// α near π ⇔ cos_alpha near −1 ⇔ huge cap, narrow circle near opposite pole.
    pub cos_alpha: f64,
}

impl SmallCircle {
    pub fn new(axis: Vec3, cos_alpha: f64) -> Self {
        SmallCircle { axis: axis.normalize(), cos_alpha }
    }

    /// Sine of the cone half-angle (always non-negative for α ∈ [0, π]).
    pub fn sin_alpha(self) -> f64 {
        (1.0 - self.cos_alpha * self.cos_alpha).max(0.0).sqrt()
    }

    /// Is `point` (assumed on the unit sphere) strictly inside the cap?
    /// Points on the boundary circle return `false`.
    pub fn contains_strict(&self, point: Vec3) -> bool {
        point.dot(&self.axis) > self.cos_alpha + 1e-12
    }

    /// Is `point` on (or inside) the cap, with a small tolerance for the
    /// boundary itself.
    pub fn contains_or_on(&self, point: Vec3, eps: f64) -> bool {
        point.dot(&self.axis) >= self.cos_alpha - eps
    }
}

/// Possible outcomes when intersecting two small circles on the unit sphere.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircleIntersection {
    /// Two distinct intersection points.
    Two(Vec3, Vec3),
    /// Tangent — one intersection (the two circles touch at a single point).
    Tangent(Vec3),
    /// The two circles do not intersect.
    Disjoint,
    /// The two circles are identical (coaxial with the same cone angle).
    Coincident,
}

/// Intersect two small circles on the unit sphere.
///
/// Derivation: write any common point as `x = a·n1 + b·n2 + c·(n1×n2)/|n1×n2|`.
/// The plane-membership conditions `x·n1 = k1` and `x·n2 = k2` give a linear
/// system for `(a, b)`; unit length gives `c²`. Two intersections exist iff
/// `c² > 0`.
pub fn intersect_circles(c1: SmallCircle, c2: SmallCircle) -> CircleIntersection {
    let sigma = c1.axis.dot(&c2.axis); // = cos(angle between axes)
    let denom = 1.0 - sigma * sigma;

    if denom.abs() < 1e-12 {
        // Axes are (anti-)parallel — coaxial.
        let on_same_side = (c1.cos_alpha - sigma * c2.cos_alpha).abs() < 1e-9;
        return if on_same_side {
            CircleIntersection::Coincident
        } else {
            CircleIntersection::Disjoint
        };
    }

    let a = (c1.cos_alpha - sigma * c2.cos_alpha) / denom;
    let b = (c2.cos_alpha - sigma * c1.cos_alpha) / denom;
    let c_sq = 1.0 - a * a - b * b - 2.0 * a * b * sigma;

    if c_sq < -1e-9 {
        return CircleIntersection::Disjoint;
    }

    let base = c1.axis * a + c2.axis * b;
    if c_sq.abs() <= 1e-9 {
        return CircleIntersection::Tangent(base);
    }

    let c = c_sq.sqrt();
    let normal_cross = c1.axis.cross(&c2.axis);
    let normal_cross_norm = normal_cross.norm();
    let n_perp = normal_cross / normal_cross_norm;
    let offset = n_perp * c;
    CircleIntersection::Two(base + offset, base - offset)
}

/// Signed central angle for an arc on a small circle going from `start` to
/// `end`. Right-hand-rule sign with respect to the small-circle axis: the
/// returned angle is positive iff the rotation from `start` to `end` is
/// counter-clockwise looking *down* the axis (i.e. from outside the cap
/// toward its interior).
///
/// Both `start` and `end` are assumed to lie on the unit sphere and on the
/// small circle (i.e. `x·axis ≈ cos_alpha`). Returns a value in `(-π, π]`.
pub fn signed_arc_angle(start: Vec3, end: Vec3, circle: SmallCircle) -> f64 {
    let proj1 = start - circle.axis * start.dot(&circle.axis);
    let proj2 = end - circle.axis * end.dot(&circle.axis);
    let n1 = proj1.norm();
    let n2 = proj2.norm();
    if n1 < 1e-12 || n2 < 1e-12 {
        return 0.0;
    }
    let v1 = proj1 / n1;
    let v2 = proj2 / n2;
    let cos_t = v1.dot(&v2).clamp(-1.0, 1.0);
    let sin_t = circle.axis.cross(&v1).dot(&v2);
    sin_t.atan2(cos_t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn v(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3::new(x, y, z)
    }

    #[test]
    fn equatorial_circle_basic() {
        let c = SmallCircle::new(v(0.0, 0.0, 1.0), 0.0);
        // Points on the equator are on the circle (not strictly inside cap).
        assert!(!c.contains_strict(v(1.0, 0.0, 0.0)));
        // Northern hemisphere points are inside the cap (axis = +z).
        assert!(c.contains_strict(v(0.0, 0.0, 1.0)));
        assert!(c.contains_strict(v(0.5, 0.5, 0.707)));
        // Southern hemisphere points are not inside.
        assert!(!c.contains_strict(v(0.0, 0.0, -1.0)));
        assert_relative_eq!(c.sin_alpha(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn small_polar_cap() {
        // A small cap around the north pole: axis = +z, cos_alpha = 0.9
        // (α = arccos 0.9 ≈ 25.8°).
        let c = SmallCircle::new(v(0.0, 0.0, 1.0), 0.9);
        // Pole is well inside.
        assert!(c.contains_strict(v(0.0, 0.0, 1.0)));
        // Equator is well outside.
        assert!(!c.contains_strict(v(1.0, 0.0, 0.0)));
    }

    #[test]
    fn two_orthogonal_great_circles_intersect_at_poles() {
        // Equator (axis +z, cos α = 0) and a meridian (axis +x, cos α = 0).
        let c1 = SmallCircle::new(v(0.0, 0.0, 1.0), 0.0);
        let c2 = SmallCircle::new(v(1.0, 0.0, 0.0), 0.0);
        match intersect_circles(c1, c2) {
            CircleIntersection::Two(p, q) => {
                // Intersections are (0, ±1, 0).
                let on_y_axis = |x: Vec3| x.x.abs() < 1e-9 && x.z.abs() < 1e-9 && x.y.abs() > 0.5;
                assert!(on_y_axis(p) && on_y_axis(q));
                assert!((p + q).norm() < 1e-9); // they're antipodal
            }
            other => panic!("expected Two, got {:?}", other),
        }
    }

    #[test]
    fn disjoint_caps() {
        // Two small caps, one near the north pole, one near the south.
        let c1 = SmallCircle::new(v(0.0, 0.0, 1.0), 0.9);
        let c2 = SmallCircle::new(v(0.0, 0.0, -1.0), 0.9);
        assert!(matches!(intersect_circles(c1, c2), CircleIntersection::Disjoint));
    }

    #[test]
    fn coincident_circles() {
        let c1 = SmallCircle::new(v(0.0, 0.0, 1.0), 0.5);
        let c2 = SmallCircle::new(v(0.0, 0.0, 1.0), 0.5);
        assert!(matches!(intersect_circles(c1, c2), CircleIntersection::Coincident));
    }

    #[test]
    fn antipodal_complementary_circles_are_coincident() {
        // Circles on opposite axes with α_1 + α_2 = π trace the same circle.
        // axis +z with cos α = 0.5 (α = 60°)
        // axis -z with cos α = -0.5 (α = 120° = 180° - 60°)
        // Both circles are the latitude z = 0.5.
        let c1 = SmallCircle::new(v(0.0, 0.0, 1.0), 0.5);
        let c2 = SmallCircle::new(v(0.0, 0.0, -1.0), -0.5);
        assert!(matches!(intersect_circles(c1, c2), CircleIntersection::Coincident));
    }

    #[test]
    fn small_circles_intersect_at_two_points() {
        // Two caps, one over the north pole (axis +z, cos 60° = 0.5) and
        // one tilted to cover the +x direction (axis +x, cos 60° = 0.5).
        // They intersect.
        let c1 = SmallCircle::new(v(0.0, 0.0, 1.0), 0.5);
        let c2 = SmallCircle::new(v(1.0, 0.0, 0.0), 0.5);
        match intersect_circles(c1, c2) {
            CircleIntersection::Two(p, q) => {
                // Both intersections must lie on both circles.
                assert_relative_eq!(p.dot(&c1.axis), 0.5, epsilon = 1e-9);
                assert_relative_eq!(p.dot(&c2.axis), 0.5, epsilon = 1e-9);
                assert_relative_eq!(p.norm(), 1.0, epsilon = 1e-9);
                assert_relative_eq!(q.dot(&c1.axis), 0.5, epsilon = 1e-9);
                assert_relative_eq!(q.dot(&c2.axis), 0.5, epsilon = 1e-9);
                assert_relative_eq!(q.norm(), 1.0, epsilon = 1e-9);
                // They should be distinct.
                assert!((p - q).norm() > 1e-6);
            }
            other => panic!("expected Two, got {:?}", other),
        }
    }

    #[test]
    fn signed_arc_angle_quarter_turn() {
        // Equator, axis = +z. Quarter turn from (1,0,0) to (0,1,0) is +π/2.
        let c = SmallCircle::new(v(0.0, 0.0, 1.0), 0.0);
        let theta = signed_arc_angle(v(1.0, 0.0, 0.0), v(0.0, 1.0, 0.0), c);
        assert_relative_eq!(theta, std::f64::consts::FRAC_PI_2, epsilon = 1e-9);
    }

    #[test]
    fn signed_arc_angle_reverse() {
        // Equator, axis = +z. Going from (0,1,0) back to (1,0,0) is -π/2.
        let c = SmallCircle::new(v(0.0, 0.0, 1.0), 0.0);
        let theta = signed_arc_angle(v(0.0, 1.0, 0.0), v(1.0, 0.0, 0.0), c);
        assert_relative_eq!(theta, -std::f64::consts::FRAC_PI_2, epsilon = 1e-9);
    }

    #[test]
    fn signed_arc_angle_180_degrees() {
        // From (1,0,0) to (-1,0,0): exactly π (or -π, we accept either).
        let c = SmallCircle::new(v(0.0, 0.0, 1.0), 0.0);
        let theta = signed_arc_angle(v(1.0, 0.0, 0.0), v(-1.0, 0.0, 0.0), c);
        assert_relative_eq!(theta.abs(), std::f64::consts::PI, epsilon = 1e-9);
    }
}
