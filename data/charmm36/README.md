# CHARMM36m protein force field

The file `par_all36m_prot.prm` is the CHARMM36m (Huang & MacKerell 2016)
protein parameter file, redistributed for academic use with attribution.

## Files

- `par_all36m_prot.prm` — bonded and non-bonded parameters
- `top_all36_prot.rtf` — residue topology + per-atom partial charges

## Source

- Original: https://mackerell.umaryland.edu/charmm_ff.shtml
- Mirror used: https://github.com/prody/coMD (commit on master, retrieved 2026-05)

## Citation

If origami's energy outputs are used in publication, cite the CHARMM36m
references listed inside the file's preamble (Huang et al. 2017; Best et al.
2012; MacKerell et al. 1998).

## What origami uses from this file

- BONDS: `K_b` and `r_0` per atom-type pair (kcal/mol/Å² and Å).
- ANGLES: `K_θ` and `θ_0` per triple (kcal/mol/rad² and degrees).
  Urey-Bradley terms (the optional `K_UB`, `S_0` columns) are **ignored** —
  origami uses simple harmonic angles only.
- DIHEDRALS: `K_χ`, `n`, `δ` per quad — multiple terms per quad supported.
- IMPROPER: `K_ψ` and `ψ_0` per quad (the central column is unused).
- NONBONDED: `ε`, `Rmin/2` per atom type. The optional 1-4 LJ parameters
  are also read.

## What origami does NOT use

- CMAP (2D φ/ψ correction grid) — out of scope per origami's M3 plan.
- HBOND section — origami uses implicit H-bonding via electrostatics + LJ
  on polar H, no explicit H-bond term.
- Section header line continuations and the `nbxmod` switch settings on the
  NONBONDED line — origami fixes its own cutoff and combining rules.

## License

CHARMM force-field files are distributed by the MacKerell lab for academic
use. See https://mackerell.umaryland.edu/charmm_ff.shtml for the licence
terms. Redistribution with attribution and unchanged-content is permitted.
