//! Energy terms for origami's force field.
//!
//! Built on top of `chem` (atom types, parameter tables, partial charges)
//! and `geom` (Structure, topology graph, distance / angle / dihedral
//! measurement). All public functions return energies in **kJ/mol** —
//! CHARMM stores values in kcal/mol, we convert at the leaves.

pub mod bonded;
pub mod cmap;
pub mod forces;
pub mod forces_bonded;
pub mod forces_gb;
pub mod forces_nonbonded;
pub mod forces_sasa;
pub mod gb;
pub mod nonbonded;
pub mod powersasa;
pub mod sasa;
pub mod scratch;
pub mod units;

pub use bonded::{BondedBreakdown, angle_energy, bond_energy, dihedral_energy, improper_energy};
pub use cmap::{add_cmap_forces, cmap_energy};
pub use forces::{
    total_force, total_force_with_cutoff, total_force_with_options, total_force_with_scratch,
};
pub use gb::{GbBreakdown, gb_energy};
pub use nonbonded::{DEFAULT_CUTOFF_A, NonbondedBreakdown, nonbonded_energy};
pub use powersasa::{PowerSasaResult, default_sasa_gammas, powersasa_energy, surface_tension_kcal};
pub use sasa::{SasaBreakdown, sasa_energy, sasa_energy_with_dots};
pub use scratch::ForceScratch;

/// Convenience aggregator returned by the bonded-energy entry point.
#[derive(Debug, Default, Clone, Copy)]
pub struct EnergyBreakdown {
    pub bond_kj_mol: f64,
    pub angle_kj_mol: f64,
    pub dihedral_kj_mol: f64,
    pub improper_kj_mol: f64,
    pub lj_kj_mol: f64,
    pub coulomb_kj_mol: f64,
    pub gb_kj_mol: f64,
    pub sasa_kj_mol: f64,
}

impl EnergyBreakdown {
    pub fn total_kj_mol(&self) -> f64 {
        self.bond_kj_mol
            + self.angle_kj_mol
            + self.dihedral_kj_mol
            + self.improper_kj_mol
            + self.lj_kj_mol
            + self.coulomb_kj_mol
            + self.gb_kj_mol
            + self.sasa_kj_mol
    }
}
