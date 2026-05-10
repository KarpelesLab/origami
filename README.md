# origami

Experimental, deterministic, first-principles protein folding.

The goal is a pipeline that goes **mRNA sequence → amino-acid chain → 3D folded
structure**, simulated as the protein co-translationally emerges from a
ribosome under thermal motion at body temperature. Folding is driven only by
the physics of the chain — charge, hydrophobicity, hydrogen bonding, sterics,
dihedral preferences, codon-rarity translation timing, Brownian motion — with
no learned priors from structural databases.

It may not work. That is the experiment.

## The pipeline so far

Take the 20-residue Trp-cage mini-protein, `NLYIQWLKDGGPSSGRPPPS`.

**1. Build an all-atom 3D structure from the sequence (`origami build`).**
Every backbone atom and every side-chain atom is placed via NeRF from
idealised internal coordinates, in an extended β-strand conformation. This is
what `origami` starts with — no fold yet.

![Extended Trp-cage chain](docs/images/trp_cage_extended.png)

**2. Compare against the experimentally-determined native fold.** PDB 1L2Y
(NMR, Neidigh et al. 2002) is the reference. The compact tertiary structure
buries the central tryptophan (centre of the image) inside a cage of polyproline
helix and α-helix.

![Native Trp-cage from PDB 1L2Y](docs/images/trp_cage_native.png)

**3. Score and relax (`origami energy`, `origami minimize`).** The hand-built
CHARMM36-derived force field assigns native 1L2Y an energy 49,170 kJ/mol below
the extended chain, confirming the physics has the right direction. L-BFGS
minimisation on the extended chain drops the energy from +47,506 kJ/mol to
−1,789 kJ/mol in 115 steps — it relaxes local strain but cannot cross the
barriers between conformations, so the chain doesn't fold:

![L-BFGS-minimised extended chain](docs/images/trp_cage_minimized.png)

The missing step is **molecular dynamics at body temperature** — Langevin
integration that gives the chain enough thermal energy to escape local minima
and explore conformational space. That's the next milestone.

## Status

Done so far: translation (M1), all-atom chain building (M2), energy evaluation
with CHARMM36-borrowed constants and GB OBC II implicit solvent (M3), and
energy minimisation with L-BFGS (M4). Approximate exact-analytical SASA via
spherical Gauss-Bonnet is partially landed and being debugged for crowded
geometries.

In progress: Langevin dynamics at 310 K (M5), co-translational chain growth
with the ribosome exit tunnel (M6), and end-to-end validation against
chignolin, Trp-cage, and villin headpiece (M7).

## Build

```sh
cargo build --workspace
cargo test --workspace
```

## CLI quick reference

```sh
# mRNA FASTA → amino-acid sequence
origami translate examples/insulin.fasta

# Sequence → all-atom PDB (extended chain)
origami build --seq NLYIQWLKDGGPSSGRPPPS --output trp_cage.pdb

# Energy of a structure with per-term breakdown
origami energy trp_cage.pdb

# L-BFGS or steepest-descent minimisation
origami minimize trp_cage.pdb --output trp_cage_min.pdb --algorithm lbfgs

# Render to PNG (ball-and-stick, CPK colours, hydrogens hidden)
origami render trp_cage.pdb --output trp_cage.png --width 800 --height 600
```

## Layout

```
crates/
  chem/       — atom/AA/codon data, CHARMM36 parameter loader, atom typing
  translate/  — mRNA → amino-acid chain
  geom/       — 3D math, NeRF, all-atom chain builder, topology graph, cell list
  io/         — PDB writer + reader, PNG renderer
  energy/     — bonded + LJ + Coulomb + GB OBC II + SASA, plus analytical forces
  dynamics/   — backtracking line search, steepest descent, L-BFGS minimisation
  cli/        — `origami` binary
data/charmm36 — vendored CHARMM36m parameter and topology files
```

## License

MIT. CHARMM36 parameter files vendored under `data/charmm36/` are
redistributed for academic use; see the headers inside those files for
attribution.
