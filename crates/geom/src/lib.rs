pub mod analysis;
pub mod builder;
pub mod cluster;
pub mod dssp;
pub mod measure;
pub mod neighbours;
pub mod nerf;
pub mod rmsd;
pub mod secondary_structure;
pub mod structure;
pub mod topology_graph;

pub use analysis::{
    contact_map_ca, end_to_end_ca, radius_of_gyration_ca, radius_of_gyration_points,
};
pub use cluster::{cluster_sizes, cluster_trajectory};
pub use dssp::{
    assign_dssp, dssp_counts, dssp_string, find_hbonds, DsspType, HBondTable,
};
pub use secondary_structure::{
    classify as classify_phi_psi, phi, psi, secondary_structure_string, ss_counts, SsType,
};
pub use builder::{
    append_residue, build_chain, build_extended_chain, BuildError, DEFAULT_OMEGA, DEFAULT_PHI,
    DEFAULT_PSI,
};
pub use measure::{angle, dihedral, distance};
pub use neighbours::CellList;
pub use nerf::place_atom;
pub use rmsd::{rmsd_ca, rmsd_points};
pub use structure::{PlacedAtom, PlacedResidue, Structure};
pub use topology_graph::{build_topology_graph, Angle, Bond, Dihedral, Improper, TopologyGraph};

pub type Vec3 = nalgebra::Vector3<f64>;
