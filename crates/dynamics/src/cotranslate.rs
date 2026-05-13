//! Co-translational chain growth (M6).
//!
//! Real proteins fold *while* the ribosome is still emitting residues into
//! solvent. By the time the C-terminus is added, the N-terminal portion
//! has already been exploring conformational space for a while inside the
//! ribosome's exit tunnel and just outside it. That's a qualitatively
//! different folding regime from "build the whole chain extended, then
//! minimise" — and the project's central hypothesis is that this physics-
//! plus-timing combination is enough for sensible folds to emerge without
//! ML priors.
//!
//! ## Design
//!
//! The driver alternates between two operations:
//!   1. Ask the [`Ribosome`] for the next residue and append it to the
//!      structure.
//!   2. Run [`crate::run_langevin`] for the time slice up until the
//!      ribosome's next emission.
//!
//! Both pieces are abstract:
//!   - [`Ribosome`] is a trait. [`UniformRibosome`] is the simple
//!     constant-rate implementation; eventually we'll have a codon-paced
//!     one with rare-codon pauses, and (per the long-horizon ambition)
//!     an actually-simulated ribosome whose tunnel and emission timing
//!     fall out of its own dynamics.
//!   - [`ExternalPotential`] is a trait for any extra force field that
//!     should act on every atom each integrator step. [`CylindricalTunnel`]
//!     implements it as a soft cylindrical confinement — a stand-in for
//!     the ribosome exit tunnel. A real-ribosome simulation will replace
//!     this with the ribosome's own atomic structure exerting the same
//!     forces directly.

use chem::{AminoAcid, ForceField};
use energy::{total_force_with_scratch, ForceScratch, DEFAULT_CUTOFF_A};
use geom::{
    append_residue, build_topology_graph, Structure, Vec3, DEFAULT_OMEGA, DEFAULT_PHI, DEFAULT_PSI,
};

use crate::langevin::{
    initialise_velocities_for_new_atoms, kinetic_energy_kj_mol, ACCEL_FACTOR,
    BOLTZMANN_KJ_PER_MOL_K,
};
use crate::rng::Xoshiro256pp;
use crate::{minimize, Algorithm, LangevinOptions, MinimizeOptions};

/// A source of amino-acid residues + emission timings.
///
/// Trait so future implementations can plug in unchanged:
///   - [`UniformRibosome`] today — constant interval per residue.
///   - A `CodonPacedRibosome` next — codon-rarity dependent timing.
///   - Eventually, a `SimulatedRibosome` whose timings emerge from
///     simulating the actual ribosome's catalytic dynamics.
pub trait Ribosome {
    /// The amino-acid sequence to emit.
    fn sequence(&self) -> &[AminoAcid];
    /// The time (in fs) at which residue index `i` (0-indexed) becomes
    /// part of the simulated chain. Must be monotonically non-decreasing.
    /// `emission_time_fs(0)` is typically 0.
    fn emission_time_fs(&self, residue_idx: usize) -> f64;
}

/// Constant-rate ribosome: residue `i` is emitted at `i × interval_fs`.
#[derive(Debug, Clone)]
pub struct UniformRibosome {
    sequence: Vec<AminoAcid>,
    interval_fs: f64,
}

impl UniformRibosome {
    pub fn new(sequence: Vec<AminoAcid>, interval_fs: f64) -> Self {
        Self {
            sequence,
            interval_fs: interval_fs.max(0.0),
        }
    }
}

impl Ribosome for UniformRibosome {
    fn sequence(&self) -> &[AminoAcid] {
        &self.sequence
    }
    fn emission_time_fs(&self, residue_idx: usize) -> f64 {
        residue_idx as f64 * self.interval_fs
    }
}

/// An external force field that adds to the per-atom force buffer each
/// integrator step. Designed to be a swappable "what the chain feels
/// from its environment" — today a parameterised exit-tunnel, eventually
/// the atomic structure of an actually-simulated ribosome.
pub trait ExternalPotential {
    /// Add per-atom forces from this potential to `forces`. Must not
    /// resize the buffer. `forces` is in kJ/mol/Å, the same units as the
    /// internal force terms.
    fn add_force(&self, structure: &Structure, forces: &mut [Vec3]);
}

/// Soft cylindrical confinement — a stand-in for the ribosome exit
/// tunnel. Atoms inside the cylinder feel no force; atoms outside the
/// radial bound feel a harmonic pushback toward the axis. Atoms outside
/// the tunnel's axial extent feel no force (they've "emerged" into
/// solvent).
///
/// Parameters from biology: the ribosomal exit tunnel is ~80 Å long
/// and ~10–20 Å wide. Defaults below match that order.
#[derive(Debug, Clone, Copy)]
pub struct CylindricalTunnel {
    /// One endpoint of the tunnel axis.
    pub axis_origin: Vec3,
    /// Unit vector along the tunnel axis (pointing into solvent —
    /// nascent chain emerges in this direction).
    pub axis_direction: Vec3,
    /// Tunnel radius in Å.
    pub radius_a: f64,
    /// Tunnel length in Å along `axis_direction` from `axis_origin`.
    pub length_a: f64,
    /// Harmonic confinement strength outside the radius (kJ/mol/Å²).
    pub k_confine: f64,
}

impl Default for CylindricalTunnel {
    fn default() -> Self {
        Self {
            axis_origin: Vec3::zeros(),
            axis_direction: Vec3::new(0.0, 0.0, 1.0),
            radius_a: 12.0,
            length_a: 80.0,
            k_confine: 50.0,
        }
    }
}

impl ExternalPotential for CylindricalTunnel {
    fn add_force(&self, structure: &Structure, forces: &mut [Vec3]) {
        let mut idx = 0usize;
        for residue in &structure.residues {
            for atom in &residue.atoms {
                let r = atom.position - self.axis_origin;
                let along = r.dot(&self.axis_direction);
                if along < 0.0 || along > self.length_a {
                    idx += 1;
                    continue;
                }
                let perp = r - self.axis_direction * along;
                let perp_dist = perp.norm();
                if perp_dist > self.radius_a {
                    let overshoot = perp_dist - self.radius_a;
                    let outward = perp / perp_dist;
                    forces[idx] -= outward * (self.k_confine * overshoot);
                }
                idx += 1;
            }
        }
    }
}

/// Snapshot delivered to the cotranslate callback.
pub struct CotranslateFrame<'a> {
    pub time_fs: f64,
    pub residue_count: usize,
    pub structure: &'a Structure,
    pub instantaneous_temperature_k: f64,
}

/// Run a co-translational folding trajectory. The chain starts empty;
/// every emission appends a residue and the integrator runs through the
/// time slice up to the next emission. A trailing slice of length
/// `tail_steps × dt_fs` runs after the last residue is emitted so the
/// completed chain has time to relax.
///
/// The Langevin state (velocities, RNG) persists across slices — new
/// residues' atoms are seeded with Maxwell-Boltzmann velocities so the
/// system stays thermalised.
pub fn run_cotranslate<R, F>(
    ribosome: &R,
    ff: &ForceField,
    opts: LangevinOptions,
    tail_steps: usize,
    external: Option<&dyn ExternalPotential>,
    mut callback: F,
) -> Structure
where
    R: Ribosome + ?Sized,
    F: FnMut(CotranslateFrame<'_>),
{
    let seq = ribosome.sequence();
    if seq.is_empty() {
        return Structure::new();
    }

    let mut structure = Structure::new();
    let mut velocities: Vec<Vec3> = Vec::new();
    let mut rng = Xoshiro256pp::from_seed(opts.seed);
    let mut clock_fs: f64 = 0.0;

    for i in 0..seq.len() {
        // Snap the wall clock to this residue's emission time so the
        // ribosome's schedule (uniform / codon-paced / simulated) is
        // authoritative; the integrator just fills the gap.
        let emit_t = ribosome.emission_time_fs(i);
        clock_fs = clock_fs.max(emit_t);

        // Append the residue.
        append_residue(&mut structure, seq[i], DEFAULT_PHI, DEFAULT_PSI, DEFAULT_OMEGA)
            .expect("chain extension failed");
        // Minimise after each append to relieve clashes between the
        // newly-placed residue (NeRF-built at idealised internal
        // coordinates) and the already-wiggled existing chain. Brief
        // (25 steps, 0.05 Å cap) was too tight: by the time chignolin
        // reached residue 8, the chain had drifted enough that the
        // next residue's NeRF placement could land inside another
        // atom's LJ well, and the constrained minimisation couldn't
        // climb out within its step budget. Langevin then ran on a
        // diverging state and atoms flew to infinity.
        //
        // Wider tolerance and step cap give the minimiser room to
        // resolve those clashes before dynamics resumes.
        let graph_after_append = build_topology_graph(&structure);
        let _ = minimize(
            &mut structure,
            &graph_after_append,
            ff,
            MinimizeOptions {
                algorithm: Algorithm::Lbfgs,
                max_steps: 200,
                gradient_tol: 10.0,
                energy_tol: 0.1,
                max_step_a: 0.1,
                include_sasa: false,
            },
        );
        // Seed velocities for new atoms now that positions are stable.
        initialise_velocities_for_new_atoms(
            &structure,
            &mut velocities,
            opts.temperature_k,
            &mut rng,
        );

        // Slice length: time until the next emission, or the tail slice
        // after the last residue.
        let slice_target_fs = if i + 1 < seq.len() {
            ribosome.emission_time_fs(i + 1)
        } else {
            clock_fs + tail_steps as f64 * opts.dt_fs
        };
        let slice_dt = (slice_target_fs - clock_fs).max(0.0);
        let slice_steps = (slice_dt / opts.dt_fs).round() as usize;

        if slice_steps == 0 {
            // Still notify the callback so the trajectory has a frame
            // recording this emission moment.
            let masses = crate::langevin::collect_masses_pub(&structure);
            let ke = kinetic_energy_kj_mol(&velocities, &masses);
            let dof = (3 * structure.atom_count()) as f64;
            let t_inst = if dof > 0.0 {
                2.0 * ke / (dof * BOLTZMANN_KJ_PER_MOL_K)
            } else {
                0.0
            };
            callback(CotranslateFrame {
                time_fs: clock_fs,
                residue_count: i + 1,
                structure: &structure,
                instantaneous_temperature_k: t_inst,
            });
            continue;
        }

        // Run one Langevin slice in-place. We replicate the BAOAB loop
        // here (rather than calling `run_langevin`) so the velocity
        // buffer and RNG carry across slices and external potentials
        // are picked up by every force eval.
        let slice_opts = LangevinOptions {
            steps: slice_steps,
            save_every: opts.save_every,
            ..opts
        };
        run_slice(
            &mut structure,
            &mut velocities,
            ff,
            &mut rng,
            slice_opts,
            external,
            &mut clock_fs,
            i + 1,
            &mut callback,
        );
    }

    structure
}

#[allow(clippy::too_many_arguments)]
fn run_slice<F>(
    structure: &mut Structure,
    velocities: &mut Vec<Vec3>,
    ff: &ForceField,
    rng: &mut Xoshiro256pp,
    opts: LangevinOptions,
    external: Option<&dyn ExternalPotential>,
    clock_fs: &mut f64,
    residue_count: usize,
    callback: &mut F,
) where
    F: FnMut(CotranslateFrame<'_>),
{
    let masses = crate::langevin::collect_masses_pub(structure);
    let n = structure.atom_count();
    assert_eq!(velocities.len(), n);

    let alpha = (-opts.friction_ps_inv * opts.dt_fs * 1.0e-3).exp();
    let kbt = BOLTZMANN_KJ_PER_MOL_K * opts.temperature_k;
    let half_dt = 0.5 * opts.dt_fs;
    let dof = (3 * n) as f64;

    let graph = build_topology_graph(structure);
    let mut scratch = ForceScratch::new(structure, &graph, ff);
    let mut forces: Vec<Vec3> = Vec::with_capacity(n);
    total_force_with_scratch(
        structure,
        &graph,
        ff,
        DEFAULT_CUTOFF_A,
        opts.include_sasa,
        &mut scratch,
        &mut forces,
    );
    if let Some(ext) = external {
        ext.add_force(structure, &mut forces);
    }

    for step in 1..=opts.steps {
        // B
        for i in 0..n {
            velocities[i] += forces[i] * (ACCEL_FACTOR / masses[i] * half_dt);
        }
        // A
        crate::langevin::apply_velocity_step_pub(structure, velocities, half_dt);
        // O
        let one_minus_alpha2 = 1.0 - alpha * alpha;
        for i in 0..n {
            let sigma = (one_minus_alpha2 * kbt * ACCEL_FACTOR / masses[i]).sqrt();
            velocities[i].x = alpha * velocities[i].x + sigma * rng.gaussian();
            velocities[i].y = alpha * velocities[i].y + sigma * rng.gaussian();
            velocities[i].z = alpha * velocities[i].z + sigma * rng.gaussian();
        }
        // A
        crate::langevin::apply_velocity_step_pub(structure, velocities, half_dt);
        // Force re-eval at new positions.
        total_force_with_scratch(
            structure,
            &graph,
            ff,
            DEFAULT_CUTOFF_A,
            opts.include_sasa,
            &mut scratch,
            &mut forces,
        );
        if let Some(ext) = external {
            ext.add_force(structure, &mut forces);
        }
        // B
        for i in 0..n {
            velocities[i] += forces[i] * (ACCEL_FACTOR / masses[i] * half_dt);
        }

        *clock_fs += opts.dt_fs;
        if opts.save_every > 0 && step % opts.save_every == 0 {
            let ke = kinetic_energy_kj_mol(velocities, &masses);
            let t_inst = if dof > 0.0 {
                2.0 * ke / (dof * BOLTZMANN_KJ_PER_MOL_K)
            } else {
                0.0
            };
            callback(CotranslateFrame {
                time_fs: *clock_fs,
                residue_count,
                structure,
                instantaneous_temperature_k: t_inst,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chem::standard_ff;

    #[test]
    fn uniform_ribosome_emission_times() {
        let r = UniformRibosome::new(
            vec![AminoAcid::Ala, AminoAcid::Gly, AminoAcid::Ser],
            500.0,
        );
        assert_eq!(r.emission_time_fs(0), 0.0);
        assert_eq!(r.emission_time_fs(1), 500.0);
        assert_eq!(r.emission_time_fs(2), 1000.0);
    }

    #[test]
    fn cylindrical_tunnel_pushes_atoms_inward() {
        // Atom at (radius + 1, 0, length/2) — well outside the radius,
        // mid-tunnel along the axis. Force should push toward -x.
        let tunnel = CylindricalTunnel {
            axis_origin: Vec3::zeros(),
            axis_direction: Vec3::new(0.0, 0.0, 1.0),
            radius_a: 5.0,
            length_a: 50.0,
            k_confine: 10.0,
        };
        let s = Structure {
            residues: vec![geom::structure::PlacedResidue {
                aa: AminoAcid::Ala,
                atoms: vec![geom::structure::PlacedAtom {
                    name: "CA",
                    element: chem::Element::C,
                    position: Vec3::new(6.0, 0.0, 25.0),
                }],
            }],
        };
        let mut forces = vec![Vec3::zeros(); 1];
        tunnel.add_force(&s, &mut forces);
        // overshoot = 1.0, force = -k*overshoot in +x direction = -10 in x.
        assert!((forces[0].x - (-10.0)).abs() < 1e-9, "got {:?}", forces[0]);
        assert!(forces[0].y.abs() < 1e-9);
        assert!(forces[0].z.abs() < 1e-9);
    }

    #[test]
    fn cylindrical_tunnel_no_force_inside() {
        let tunnel = CylindricalTunnel::default();
        let s = Structure {
            residues: vec![geom::structure::PlacedResidue {
                aa: AminoAcid::Ala,
                atoms: vec![geom::structure::PlacedAtom {
                    name: "CA",
                    element: chem::Element::C,
                    position: Vec3::new(2.0, 0.0, 40.0),
                }],
            }],
        };
        let mut forces = vec![Vec3::zeros(); 1];
        tunnel.add_force(&s, &mut forces);
        assert_eq!(forces[0], Vec3::zeros());
    }

    #[test]
    fn cotranslate_ala3_grows_to_three_residues() {
        let r = UniformRibosome::new(
            vec![AminoAcid::Ala, AminoAcid::Ala, AminoAcid::Ala],
            50.0, // 50 fs per residue
        );
        let opts = LangevinOptions {
            dt_fs: 1.0,
            temperature_k: 310.0,
            friction_ps_inv: 2.0,
            steps: 0, // overridden per slice
            save_every: 10,
            seed: 1,
            randomise_initial_velocities: true,
            include_sasa: false,
        };
        let ff = standard_ff();
        let mut frames = 0usize;
        let final_s = run_cotranslate(&r, ff, opts, 50, None, |frame| {
            frames += 1;
            assert!(frame.residue_count <= 3);
        });
        assert_eq!(final_s.residues.len(), 3);
        assert!(frames > 5, "expected several frames, got {}", frames);
        for resi in &final_s.residues {
            for atom in &resi.atoms {
                assert!(atom.position.x.is_finite());
            }
        }
    }
}
