//! Energy terms for origami's force field.
//!
//! Built on top of `chem` (atom types, parameter tables, partial charges)
//! and `geom` (Structure, topology graph, distance / angle / dihedral
//! measurement). All public functions return energies in **kJ/mol** —
//! CHARMM stores values in kcal/mol, we convert at the leaves.

pub mod bonded;
pub mod nonbonded;
pub mod units;

pub use bonded::{
    angle_energy, bond_energy, dihedral_energy, improper_energy, BondedBreakdown,
};
pub use nonbonded::{nonbonded_energy, NonbondedBreakdown, DEFAULT_CUTOFF_A};

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
