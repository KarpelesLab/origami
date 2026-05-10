pub mod builder;
pub mod measure;
pub mod neighbours;
pub mod nerf;
pub mod structure;
pub mod topology_graph;

pub use builder::{build_chain, build_extended_chain, BuildError};
pub use measure::{angle, dihedral, distance};
pub use neighbours::CellList;
pub use nerf::place_atom;
pub use structure::{PlacedAtom, PlacedResidue, Structure};
pub use topology_graph::{build_topology_graph, Angle, Bond, Dihedral, Improper, TopologyGraph};

pub type Vec3 = nalgebra::Vector3<f64>;
