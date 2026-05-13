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

**4. Heat it up (`origami dynamics`).** BAOAB Langevin integration at 310 K
gives the chain thermal energy to wiggle, explore conformations, and (in
principle) cross barriers between minima. The frame below is a snapshot
from 3 ps of dynamics started from the native fold — the cage stays put but
visibly fluctuates, which is what implicit-solvent MD at body temperature
should look like:

![Trp-cage during Langevin dynamics at 310 K](docs/images/trp_cage_dynamics.png)

**5. Grow it co-translationally (`origami cotranslate`).** A real chain
doesn't appear all at once — the ribosome emits residues N-to-C, and the
N-terminal portion has been folding for a while by the time the C-terminus
arrives. The cotranslate command alternates between appending one residue
and running Langevin dynamics for the time slice up to the next emission.
An optional cylindrical exit-tunnel constraint keeps the nascent chain
inside a confined region, mimicking the ribosomal tunnel.

Combined with `--with-sasa`, the hydrophobic-collapse term drives the
nascent chain as soon as enough side chains are present to cluster.
Chignolin (`GYDPETGTWG`, 10 residues, PDB 1UAO), one residue per 0.5 ps,
100 ps tail of Langevin at 310 K, γ = 2 ps⁻¹, hydrophobic γ-scale 0.25:

![Cotranslational chignolin growth + hydrophobic collapse](docs/animations/chignolin_cotsasa.gif)

[Full quality MP4](docs/animations/chignolin_cotsasa.mp4) ·
1 ps of simulated time per ~83 ms of video at 30 fps.

The Cα RMSD vs the 1UAO native fold over the 100 ps tail:

| time after emergence | Cα RMSD vs 1UAO native |
|---:|---:|
| ~4 ps (chain just complete) | 7.32 Å |
| 20 ps | 6.73 Å |
| 40 ps | 4.05 Å |
| **44 ps** (`docs/images/chignolin_cotsasa_43ps.png`) | **2.94 Å** (minimum) |
| 100 ps | 3.06 Å |

Full trace: [docs/data/chignolin_cotsasa_rmsd.tsv](docs/data/chignolin_cotsasa_rmsd.tsv).
The chain compacts from extended (7.3 Å) into a sub-3 Å native-like
basin during the tail. Compared to the pre-folded baseline below
(1.82 Å from a pre-minimised extended chain), the cotranslational
version reaches a slightly higher RMSD floor — the chain spends its
first few ps growing rather than folding, so it has less wall-clock
time available to relax. But the qualitative behaviour is the same:
emerge, collapse, hover in a compact basin.

**6. Actually fold something.** Start from a minimised extended chignolin
(GYDPETGTWG, 10 residues) and run Langevin dynamics at 310 K. With just
LJ + Coulomb + GB (no hydrophobic forces) over 500 ps:

| frame (× 10 ps) | Cα RMSD vs 1UAO native |
|---:|---|
| 0 (start) | 8.76 Å |
| 6 | 3.66 Å |
| 10 | 2.92 Å |
| 14 | 2.09 Å |
| **16** | **1.82 Å** — within NMR experimental uncertainty |
| 17–18 | 1.88, 2.01 Å |

![Simulated chignolin at 160 ps (1.82 Å RMSD vs native)](docs/images/chignolin_folded_160ps.png)
![Native chignolin (1UAO)](docs/images/chignolin_native.png)

Adding the analytical hydrophobic-SASA forces (PSA.2) accelerates
collapse. A γ-scaling sweep on the same starting structure and seed
shows the trade-off:

| γ scale | Min Cα RMSD vs 1UAO | Time to min | Sim length |
|---:|---:|---:|---:|
| 0.0 (no SASA) | **1.82 Å** | 160 ps | 500 ps |
| 0.25 | 2.04 Å | 100 ps | 200 ps |
| 0.5 | 2.37 Å | 48 ps | 200 ps |
| 1.0 (full literature γ) | 2.82 Å | 64 ps | 200 ps |

Full RMSD traces:
[no-SASA](docs/data/chignolin_rmsd.tsv) ·
[γ=0.25](docs/data/chignolin_sasa_g025_rmsd.tsv) ·
[γ=0.5](docs/data/chignolin_sasa_g050_rmsd.tsv) ·
[γ=1.0](docs/data/chignolin_sasa_rmsd.tsv).

Lower γ → tighter native fit (less molten-globule lock-in). The
γ=0.25 fold at 100 ps:

![γ=0.25 SASA fold at 100 ps (2.04 Å)](docs/images/chignolin_sasa_g025_100ps.png)

**Replica-exchange MD on chignolin: tightest fold so far at 1.43 Å.**
With REMD doing 8 replicas in parallel (T ladder 280-395 K, geometric
spacing, swap every 1 ps, dt = 2 fs + SHAKE), the production
trajectory (lowest-T replica at 280 K) reaches sub-2 Å within
57 ps and sub-1.6 Å within 212 ps; the global minimum is **1.43 Å
at frame 1879 (1.88 ns)** — tighter than the 1.82 Å straight-Langevin
baseline and substantially tighter than the SASA-driven runs above.
99 % of frames cluster into a single basin at 1.5 Å RMSD-linkage
cutoff. Swap acceptance 55-65 % across all 7 adjacent pairs:

![Chignolin REMD fold trajectory, 2 ns × 8 replicas](docs/animations/chignolin_remd.gif)

[Full quality MP4](docs/animations/chignolin_remd.mp4) ·
[RMSD trace](docs/data/chignolin_remd_rmsd.tsv)

The sweet spot looks like γ ∈ [0.25, 0.5]: enough hydrophobic drive
to compact the chain ~2× faster than LJ+GB-only, without
over-stabilising the first compact state it finds. The literature γ
of 5 cal/mol/Å² (our γ=1.0 baseline) appears to be too aggressive
for our combined CHARMM36 + OBC-GB force field. Tunable via the
`ORIGAMI_SASA_GAMMA_SCALE` environment variable.

Either way: the central hypothesis — that hand-built physics
produces reasonable folds without ML priors — is at least true for
the smallest known fold, both with and without hydrophobic forces.

**Trp-cage fold trial.** Same setup on the 20-residue Trp-cage
(NLYIQWLKDGGPSSGRPPPS, starting Cα RMSD 16.66 Å vs the 1L2Y NMR
structure), 300 ps Langevin at γ=0.25. The chain compacts steadily
from 16.66 Å → 4.20 Å (frame 35 / 210 ps) and stabilises in a
~4.2 Å plateau through to 300 ps:

![Trp-cage molten globule at 210 ps (4.20 Å)](docs/images/trpcage_sasa_g025_210ps.png)

Trace: [docs/data/trpcage_sasa_g025_rmsd.tsv](docs/data/trpcage_sasa_g025_rmsd.tsv).

Not the native fold — Trp-cage folds in ~5 μs experimentally, and
our 300 ps run is ~16 000× short of that — but a clear hydrophobic
collapse to a compact molten globule. The chain didn't diverge,
didn't get stuck extended, and didn't blow through the native basin.
To reach the actual 1L2Y fold would need a much longer trajectory
(milliseconds of simulated time, or replica exchange).

**Disulfide bonds.** Crambin (PDB 1CRN, 46 residues, plant peptide)
ships under `crates/io/tests/fixtures/1CRN_crambin.pdb` to exercise
the S-S detection path. `geom::build_topology_graph` finds all three
disulfide bridges geometrically — Cys3-Cys40, Cys4-Cys32, Cys16-Cys26 —
from each SG-SG pair's distance (< 2.5 Å threshold), matching the PDB's
declared SSBOND records exactly. A 1 ps native MD test asserts the
bonded topology with disulfides keeps the chain within 4 Å Cα RMSD;
without the S-S bonds the chain would unfold. The six yellow sulfurs
(three pairs) are visible in the centre of the fold:

![Crambin native (PDB 1CRN), 3 disulfide bridges](docs/images/crambin_native.png)

**Multi-chain proteins.** Insulin (PDB 2HIU, 51 residues across chains
A and B) is the smallest standard test for multi-chain support. The
two-chain `Structure` representation, chain-aware PDB I/O, and the
two-chain-aware peptide-bond auto-detection all ship in
`crates/io/tests/fixtures/2HIU_insulin.pdb`. Disulfide detection is
purely geometric, so the two inter-chain bridges (A7-B7 and A20-B19)
fall out of the same logic that catches crambin's intra-chain
disulfides. The yellow sulfurs split into three pairs:

![Insulin (PDB 2HIU), chains A+B linked by 3 disulfide bridges](docs/images/insulin_native.png)

A 1 ps native MD test asserts Cα RMSD stays under 1 Å.

## Status

Done so far: translation (M1), all-atom chain building (M2), energy evaluation
with CHARMM36-borrowed constants and GB OBC II implicit solvent (M3), energy
minimisation with L-BFGS (M4), BAOAB Langevin dynamics with trajectory
rendering (M5), exact analytical SASA via spherical Gauss-Bonnet (PSA.1, ~1%
match to Shrake-Rupley), analytical SASA forces in the gradient (PSA.2),
co-translational chain growth with optional exit-tunnel constraint (M6),
validation against three small folds (M7): chignolin (1UAO), Trp-cage (1L2Y),
and villin headpiece HP-35 (2F4K). For each, the native fold scores at least
30 000 kJ/mol below the same sequence built as an extended chain, and 2 ps of
Langevin dynamics from the Trp-cage native conformation keeps Cα RMSD under
1 Å.

Bonded-topology features:
- **Disulfide bonds** auto-detected via geometric SG-SG distance check
  (intra- and inter-chain; matches PDB SSBOND records on crambin and insulin)
- **Multi-chain `Structure`** with chain-aware peptide-bond auto-detection
  (no phantom C(A21)-N(B1) bond across insulin's chain break)
- **Monomer enum** holding either an `AminoAcid` (protein) or a
  `Nucleotide` (RNA) — `chem::Nucleotide` is wired through but the
  dynamics path stays protein-only for now (the PDB reader silently
  skips ribonucleotide residues; future work promotes them to full
  inclusion as part of the long-horizon ribosome goal)

Integrator and physics:
- **Reaction-field Coulomb** at the 10 Å cutoff (Tironi ε_RF → ∞);
  both V and F vanish smoothly at the cutoff
- **SHAKE** on X-H bonds → integrator stable at dt = 2 fs → 2× wall-clock
  throughput per simulated picosecond
- **SoA + rayon-parallel** nonbonded pair loop (~3.3× kernel-level
  speedup on Trp-cage; threshold-gated parallelism so smaller proteins
  use the no-overhead serial path)
- **SoA Born-radii** for GB with parallel descreening sum
- **REMD** (parallel tempering) — N replicas at increasing temperatures
  with Metropolis swaps; production trajectory is the lowest-T replica

Analysis:
- **`origami analyze`** — per-frame Cα RMSD vs reference + radius of
  gyration + end-to-end distance + Kabsch-Sander DSSP secondary
  structure (with Ramachandran fallback for heavy-atom-only PDBs) +
  optional residue-residue contact frequency map
- **Trajectory animation** — `origami render --output-dir frames/`
  emits per-frame PNGs with a stable camera locked across frames
  (so molecules visually grow / fold in place); optional
  `--frame-dt-fs` stamps `t = N.NN ps` in the top-left corner

Performance benchmarks (release build, Apple Silicon, M3 Pro):

| protein         | atoms | pairs (10 Å) | total_force / step | sim/wall (dt=2 + SHAKE) |
|-----------------|------:|-------------:|-------------------:|------------------------:|
| chignolin (1UAO)  |   134 |  7 470 |   0.27 ms |       ~500 ns / day |
| Trp-cage  (1L2Y)  |   300 | 24 193 |   1.12 ms |       ~150 ns / day |
| villin    (2F4K)  |   520 | 48 434 |   2.76 ms |        ~60 ns / day |

With analytical SASA forces on: ~30× slower (still bounded by the
SASA cost, which is ~30 ms/step on Trp-cage — the SoA / parallel /
SHAKE work didn't move that dial). Numerical-vs-analytical SASA
agreement: max 4.5 × 10⁻⁹ on the gradient.

Up next: NeRF placement for ribonucleotides and a Monomer-aware
topology graph so RNA chains can actually be built and dynamics'd.

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

# BAOAB Langevin dynamics at 310 K — writes a multi-MODEL trajectory PDB
origami dynamics trp_cage_min.pdb --output-trajectory traj.pdb \
    --steps 3000 --save-every 100 --temperature 310 --friction 5.0

# Same, with SHAKE-constrained X-H bonds and dt = 2 fs (≈ 2× faster)
origami dynamics trp_cage_min.pdb --output-trajectory traj.pdb \
    --steps 1500 --save-every 50 --dt 2.0 --shake-h

# Replica-exchange MD — 8 replicas on a geometric T ladder, swap every 0.5 ps
origami remd trp_cage_min.pdb --output-trajectory remd.pdb \
    --temperatures 300,310,321,333,346,360,375,391 --time-ps 50 \
    --swap-interval-ps 0.5 --dt 2.0 --shake-h

# Co-translational chain growth — append one residue, then run dynamics
# until the ribosome emits the next residue. Optional cylindrical exit
# tunnel mimics the ribosomal tunnel.
origami cotranslate --seq NLYIQWLKDGGPSSGRPPPS --output-trajectory cotrans.pdb \
    --interval 500 --tail 5000 --save-every 50 --with-tunnel

# Render single-frame or trajectory (multi-MODEL → frame_NNNN.png per model).
# `--frame-dt-fs` adds a `t = N.NN ps` overlay in the top-left of each frame.
origami render trp_cage.pdb --output trp_cage.png --width 800 --height 600
origami render traj.pdb --output-dir frames/ --width 800 --height 600 \
    --frame-dt-fs 100

# Trajectory analysis: per-frame Cα RMSD, Rg, end-to-end, DSSP secondary
# structure (H/E/C string + helix/strand %), optional residue-residue
# contact-frequency map, optional RMSD-clustering of frames into fold
# basins.
origami analyze cotrans.pdb \
    --reference crates/io/tests/fixtures/1UAO_chignolin.pdb \
    --output metrics.tsv \
    --contact-map contacts.tsv --contact-cutoff 8.0 \
    --cluster-cutoff 1.5
```

## Layout

```
crates/
  chem/       — atom/AA/codon data, CHARMM36 parameter loader, atom typing
  translate/  — mRNA → amino-acid chain
  geom/       — 3D math, NeRF, all-atom chain builder, topology graph, cell list
  io/         — PDB writer + reader, PNG renderer
  energy/     — bonded + LJ + Coulomb + GB OBC II + SASA, plus analytical forces
  dynamics/   — backtracking line search, steepest descent, L-BFGS minimisation,
                BAOAB Langevin integrator + xoshiro256++ PRNG
  cli/        — `origami` binary
data/charmm36 — vendored CHARMM36m parameter and topology files
```

## License

MIT. CHARMM36 parameter files vendored under `data/charmm36/` are
redistributed for academic use; see the headers inside those files for
attribution.
