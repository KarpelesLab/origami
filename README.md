# origami

Experimental, deterministic, first-principles protein folding.

The goal is a pipeline that goes **mRNA sequence → amino-acid chain → 3D folded
structure**, simulated as the protein co-translationally emerges from a
ribosome under thermal motion at body temperature. Folding is driven only by
the physics of the chain — charge, hydrophobicity, hydrogen bonding, sterics,
dihedral preferences, codon-rarity translation timing, Brownian motion — with
no learned priors from structural databases.

It may not work. That is the experiment.

## Status

Early development. See `/Users/magicaltux/.claude/plans/scalable-wiggling-abelson.md`
for the milestone roadmap.

## Build

```sh
cargo build --workspace
cargo test --workspace
```

## Layout

```
crates/
  chem/       — atoms, amino acids, codons, properties
  translate/  — mRNA → amino-acid chain
  cli/        — `origami` binary
data/         — codon tables, AA properties, force-field parameters
```

## License

MIT.
