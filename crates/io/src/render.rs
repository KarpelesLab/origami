//! Ball-and-stick PNG renderer.
//!
//! Single-pass ray-caster: for each output pixel, find the closest atom
//! sphere or bond cylinder along the ray and shade it. CPK atom colours,
//! diffuse + ambient lighting from a single directional source. No spatial
//! acceleration structure — fine for proteins under a few thousand atoms,
//! which is everything we currently handle.

use chem::Element;
use geom::{build_topology_graph, Structure, Vec3};
use image::{Rgba, RgbaImage};

#[derive(Debug, Clone, Copy)]
pub struct RenderOptions {
    pub width: u32,
    pub height: u32,
    pub show_hydrogens: bool,
    /// Background colour (R, G, B in 0..=255).
    pub background: [u8; 3],
    /// Atom-sphere radius as a fraction of the atom's vdW radius. Smaller
    /// values give "thinner" balls and emphasise the sticks.
    pub atom_scale: f64,
    /// Bond cylinder radius in Å.
    pub bond_radius_a: f64,
    /// Vertical field-of-view in degrees.
    pub fov_deg: f64,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            show_hydrogens: false,
            background: [25, 25, 30],
            atom_scale: 0.35,
            bond_radius_a: 0.18,
            fov_deg: 30.0,
        }
    }
}

/// Render the structure to an RGBA image.
pub fn render(structure: &Structure, opts: &RenderOptions) -> RgbaImage {
    // 1. Flatten atoms and select which to draw.
    let mut atoms: Vec<RenderAtom> = Vec::new();
    let mut global_to_drawn: Vec<Option<usize>> = Vec::new();
    for residue in &structure.residues {
        for atom in &residue.atoms {
            if !opts.show_hydrogens && atom.element == Element::H {
                global_to_drawn.push(None);
                continue;
            }
            global_to_drawn.push(Some(atoms.len()));
            atoms.push(RenderAtom {
                center: atom.position,
                radius: vdw_radius(atom.element) * opts.atom_scale,
                color: cpk_color(atom.element),
            });
        }
    }
    if atoms.is_empty() {
        return RgbaImage::from_pixel(
            opts.width, opts.height,
            Rgba([opts.background[0], opts.background[1], opts.background[2], 255]),
        );
    }

    // 2. Build bonds from the topology graph, keeping only those between
    //    drawn atoms. Each bond carries both endpoint colours so the
    //    cylinder can be split into half-tinted segments.
    let graph = build_topology_graph(structure);
    let mut bonds: Vec<RenderBond> = Vec::new();
    for b in &graph.bonds {
        let (Some(a_drawn), Some(b_drawn)) = (global_to_drawn[b.a], global_to_drawn[b.b]) else {
            continue;
        };
        bonds.push(RenderBond {
            a: atoms[a_drawn].center,
            b: atoms[b_drawn].center,
            radius: opts.bond_radius_a,
            color_a: atoms[a_drawn].color,
            color_b: atoms[b_drawn].color,
        });
    }

    // 3. Set up the camera. Centre on the structure's centroid, look down
    //    +z (toward the structure from +z). Distance picked so the
    //    bounding sphere fits the vertical FOV with margin.
    let centroid = atoms.iter().fold(Vec3::zeros(), |acc, a| acc + a.center) / atoms.len() as f64;
    let bounding_radius = atoms
        .iter()
        .map(|a| (a.center - centroid).norm() + a.radius)
        .fold(0.0_f64, f64::max);
    let fov = opts.fov_deg.to_radians();
    let cam_dist = (bounding_radius / (fov / 2.0).tan()) * 1.25 + 5.0;
    let camera = Camera::new(
        centroid + Vec3::new(0.0, 0.0, cam_dist),
        centroid,
        Vec3::new(0.0, 1.0, 0.0),
        fov,
        opts.width as f64 / opts.height as f64,
    );

    // 4. Lighting (single directional source from upper-right-front).
    let light_dir = Vec3::new(0.6, 0.7, 0.5).normalize();
    let ambient = 0.18;

    // 5. Per-pixel ray cast.
    let mut img = RgbaImage::new(opts.width, opts.height);
    for j in 0..opts.height {
        for i in 0..opts.width {
            let ray = camera.ray_for_pixel(i, j, opts.width, opts.height);
            let mut best_t = f64::INFINITY;
            let mut best_normal = Vec3::zeros();
            let mut best_color = [0.0_f64; 3];
            // Atom spheres.
            for atom in &atoms {
                if let Some(t) = intersect_sphere(&ray, atom.center, atom.radius) {
                    if t > 1e-4 && t < best_t {
                        best_t = t;
                        let hit = ray.origin + ray.direction * t;
                        best_normal = (hit - atom.center) / atom.radius;
                        best_color = atom.color;
                    }
                }
            }
            // Bond cylinders — half-coloured by which endpoint is closer
            // along the bond axis.
            for bond in &bonds {
                if let Some((t, normal, frac)) = intersect_cylinder(&ray, bond.a, bond.b, bond.radius) {
                    if t > 1e-4 && t < best_t {
                        best_t = t;
                        best_normal = normal;
                        best_color = if frac < 0.5 { bond.color_a } else { bond.color_b };
                    }
                }
            }
            let px = if best_t.is_finite() {
                shade(best_normal, light_dir, -ray.direction, best_color, ambient)
            } else {
                [
                    opts.background[0] as f64 / 255.0,
                    opts.background[1] as f64 / 255.0,
                    opts.background[2] as f64 / 255.0,
                ]
            };
            let r = (px[0] * 255.0).clamp(0.0, 255.0) as u8;
            let g = (px[1] * 255.0).clamp(0.0, 255.0) as u8;
            let b = (px[2] * 255.0).clamp(0.0, 255.0) as u8;
            img.put_pixel(i, j, Rgba([r, g, b, 255]));
        }
    }
    img
}

// ---------- Internals ----------

struct RenderAtom {
    center: Vec3,
    radius: f64,
    color: [f64; 3],
}

struct RenderBond {
    a: Vec3,
    b: Vec3,
    radius: f64,
    color_a: [f64; 3],
    color_b: [f64; 3],
}

struct Ray {
    origin: Vec3,
    direction: Vec3,
}

struct Camera {
    origin: Vec3,
    forward: Vec3,
    right: Vec3,
    up: Vec3,
    /// Tangent of half-vertical-fov.
    half_fov_tan: f64,
    aspect: f64,
}

impl Camera {
    fn new(origin: Vec3, look_at: Vec3, world_up: Vec3, fov: f64, aspect: f64) -> Self {
        let forward = (look_at - origin).normalize();
        let right = forward.cross(&world_up).normalize();
        let up = right.cross(&forward).normalize();
        Self {
            origin,
            forward,
            right,
            up,
            half_fov_tan: (fov / 2.0).tan(),
            aspect,
        }
    }

    fn ray_for_pixel(&self, i: u32, j: u32, width: u32, height: u32) -> Ray {
        // Map (i, j) to normalised screen coords in [-1, 1] × [-1, 1]
        // with y flipped (image rows go top-down, world up is +y).
        let u_ndc = (2.0 * (i as f64 + 0.5) / width as f64) - 1.0;
        let v_ndc = 1.0 - (2.0 * (j as f64 + 0.5) / height as f64);
        let dir =
            self.forward + self.right * (u_ndc * self.half_fov_tan * self.aspect) + self.up * (v_ndc * self.half_fov_tan);
        Ray {
            origin: self.origin,
            direction: dir.normalize(),
        }
    }
}

fn intersect_sphere(ray: &Ray, center: Vec3, radius: f64) -> Option<f64> {
    let oc = ray.origin - center;
    let b = oc.dot(&ray.direction);
    let c = oc.dot(&oc) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let s = disc.sqrt();
    let t1 = -b - s;
    let t2 = -b + s;
    if t1 > 1e-4 {
        Some(t1)
    } else if t2 > 1e-4 {
        Some(t2)
    } else {
        None
    }
}

/// Intersect ray with a finite cylinder (no end caps — they're hidden
/// inside the atom spheres). Returns the closest positive `t`, the
/// surface normal at the hit, and the fractional position along the
/// axis from `a` to `b` (0 at `a`, 1 at `b`) used for half-colour split.
fn intersect_cylinder(ray: &Ray, a: Vec3, b: Vec3, radius: f64) -> Option<(f64, Vec3, f64)> {
    let axis = b - a;
    let axis_len = axis.norm();
    if axis_len < 1e-9 {
        return None;
    }
    let axis_hat = axis / axis_len;
    // Project ray and origin onto the plane perpendicular to axis_hat.
    let d_perp = ray.direction - axis_hat * ray.direction.dot(&axis_hat);
    let oa = ray.origin - a;
    let oa_perp = oa - axis_hat * oa.dot(&axis_hat);
    let a_coef = d_perp.dot(&d_perp);
    if a_coef < 1e-12 {
        return None; // ray parallel to axis
    }
    let b_coef = oa_perp.dot(&d_perp);
    let c_coef = oa_perp.dot(&oa_perp) - radius * radius;
    let disc = b_coef * b_coef - a_coef * c_coef;
    if disc < 0.0 {
        return None;
    }
    let s = disc.sqrt();
    let t1 = (-b_coef - s) / a_coef;
    let t2 = (-b_coef + s) / a_coef;
    for &t in &[t1, t2] {
        if t <= 1e-4 {
            continue;
        }
        let hit = ray.origin + ray.direction * t;
        let along = (hit - a).dot(&axis_hat);
        if along < 0.0 || along > axis_len {
            continue;
        }
        let axis_pt = a + axis_hat * along;
        let normal = (hit - axis_pt).normalize();
        let frac = along / axis_len;
        return Some((t, normal, frac));
    }
    None
}

fn shade(
    normal: Vec3,
    light_dir: Vec3,
    view_dir: Vec3,
    color: [f64; 3],
    ambient: f64,
) -> [f64; 3] {
    let diff = normal.dot(&light_dir).max(0.0);
    let intensity = ambient + (1.0 - ambient) * diff;
    // Blinn-Phong specular: gentle highlight on top of Lambert.
    let half = (light_dir + view_dir).normalize();
    let spec_intensity = normal.dot(&half).max(0.0).powf(32.0) * 0.35;
    [
        (color[0] * intensity + spec_intensity).min(1.0),
        (color[1] * intensity + spec_intensity).min(1.0),
        (color[2] * intensity + spec_intensity).min(1.0),
    ]
}

fn vdw_radius(element: Element) -> f64 {
    match element {
        Element::H => 1.20,
        Element::C => 1.70,
        Element::N => 1.55,
        Element::O => 1.52,
        Element::S => 1.80,
    }
}

fn cpk_color(element: Element) -> [f64; 3] {
    // Standard CPK with a slight desaturation for nicer rendering.
    match element {
        Element::H => [0.95, 0.95, 0.95],
        Element::C => [0.50, 0.50, 0.50],
        Element::N => [0.20, 0.30, 0.85],
        Element::O => [0.85, 0.20, 0.20],
        Element::S => [1.00, 0.85, 0.20],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chem::AminoAcid;
    use geom::build_extended_chain;

    #[test]
    fn renders_ala3_to_nonblank_png() {
        let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala, AminoAcid::Ala]).unwrap();
        let opts = RenderOptions { width: 200, height: 150, ..Default::default() };
        let img = render(&s, &opts);
        // Verify at least one non-background pixel exists.
        let bg = Rgba([opts.background[0], opts.background[1], opts.background[2], 255]);
        let any_drawn = img.pixels().any(|p| *p != bg);
        assert!(any_drawn, "rendered image is entirely background");
    }

    #[test]
    fn hide_vs_show_hydrogens_changes_pixel_count() {
        let s = build_extended_chain(&[AminoAcid::Ala, AminoAcid::Ala]).unwrap();
        let opts_hide = RenderOptions { width: 200, height: 150, show_hydrogens: false, ..Default::default() };
        let opts_show = RenderOptions { width: 200, height: 150, show_hydrogens: true, ..Default::default() };
        let img_hide = render(&s, &opts_hide);
        let img_show = render(&s, &opts_show);
        let bg = Rgba([opts_hide.background[0], opts_hide.background[1], opts_hide.background[2], 255]);
        let drawn_hide = img_hide.pixels().filter(|p| **p != bg).count();
        let drawn_show = img_show.pixels().filter(|p| **p != bg).count();
        assert!(drawn_show > drawn_hide,
            "showing hydrogens should produce more drawn pixels: {} vs {}", drawn_show, drawn_hide);
    }
}
