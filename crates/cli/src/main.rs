use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use chem::{standard_ff, AminoAcid};
use clap::{Parser, Subcommand, ValueEnum};
use dynamics::{minimize, Algorithm, MinimizeOptions};
use energy::{
    bonded::bonded_energy, gb_energy, nonbonded_energy, sasa_energy, DEFAULT_CUTOFF_A,
};
use geom::{build_extended_chain, build_topology_graph};
use io::{read_pdb, render, write_pdb, RenderOptions};
use translate::{find_orfs, parse_fasta, translate_codons};
use translate::translate::{one_letter_string, three_letter_string};

#[derive(Debug, Parser)]
#[command(name = "origami", version, about = "Experimental physics-based protein folding")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Translate an mRNA FASTA file (or stdin) to amino-acid sequence(s).
    Translate {
        /// Input FASTA file. Use `-` (or omit) for stdin.
        #[arg(default_value = "-")]
        input: String,

        /// Find all ORFs across three forward frames instead of translating
        /// each record from position 0.
        #[arg(long)]
        orfs: bool,

        /// Minimum ORF length in amino acids when --orfs is set.
        #[arg(long, default_value_t = 30)]
        min_aa: usize,

        /// Use three-letter amino-acid codes (Met-Ala-…) instead of one-letter (MA…).
        #[arg(long)]
        three_letter: bool,
    },
    /// Build an all-atom 3D structure for an amino-acid sequence and write a PDB file.
    Build {
        /// Amino-acid sequence (one-letter codes, e.g. "MAW").
        #[arg(long, conflicts_with = "from_fasta")]
        seq: Option<String>,

        /// Read the amino-acid sequence from a protein FASTA file (one-letter codes).
        #[arg(long, conflicts_with = "seq")]
        from_fasta: Option<String>,

        /// Output PDB path. Defaults to stdout.
        #[arg(long, short)]
        output: Option<PathBuf>,
    },
    /// Compute the total potential energy of a PDB structure with a per-term breakdown.
    Energy {
        /// Path to the PDB file.
        input: PathBuf,
        /// Skip the SASA term (slow on large structures).
        #[arg(long)]
        skip_sasa: bool,
    },
    /// Render a PDB structure to a PNG image (ball-and-stick).
    Render {
        /// Input PDB.
        input: PathBuf,
        /// Output PNG path.
        #[arg(long, short)]
        output: PathBuf,
        /// Image width in pixels.
        #[arg(long, default_value_t = 800)]
        width: u32,
        /// Image height in pixels.
        #[arg(long, default_value_t = 600)]
        height: u32,
        /// Include hydrogen atoms (hidden by default).
        #[arg(long)]
        show_hydrogens: bool,
    },
    /// Minimize a PDB structure (energy gradient descent).
    Minimize {
        /// Input PDB.
        input: PathBuf,
        /// Output (minimized) PDB.
        #[arg(long, short)]
        output: PathBuf,
        /// Optimization algorithm.
        #[arg(long, value_enum, default_value_t = AlgoFlag::Lbfgs)]
        algorithm: AlgoFlag,
        /// Maximum optimization steps.
        #[arg(long, default_value_t = 500)]
        max_steps: usize,
        /// Convergence threshold on max gradient component (kJ/mol/Å).
        #[arg(long, default_value_t = 1.0)]
        tol: f64,
        /// Maximum atom displacement per step (Å).
        #[arg(long, default_value_t = 0.1)]
        max_step: f64,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum AlgoFlag {
    Sd,
    Lbfgs,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Translate { input, orfs, min_aa, three_letter } => {
            run_translate(&input, orfs, min_aa, three_letter)
        }
        Command::Build { seq, from_fasta, output } => {
            run_build(seq.as_deref(), from_fasta.as_deref(), output.as_deref())
        }
        Command::Energy { input, skip_sasa } => run_energy(&input, skip_sasa),
        Command::Minimize { input, output, algorithm, max_steps, tol, max_step } => {
            run_minimize(&input, &output, algorithm, max_steps, tol, max_step)
        }
        Command::Render { input, output, width, height, show_hydrogens } => {
            run_render(&input, &output, width, height, show_hydrogens)
        }
    }
}

fn run_render(
    input: &Path,
    output: &Path,
    width: u32,
    height: u32,
    show_hydrogens: bool,
) -> Result<()> {
    let file = fs::File::open(input)
        .with_context(|| format!("opening {}", input.display()))?;
    let structure = read_pdb(file)
        .with_context(|| format!("reading {}", input.display()))?;
    let opts = RenderOptions {
        width,
        height,
        show_hydrogens,
        ..Default::default()
    };
    let img = render(&structure, &opts);
    img.save(output)
        .with_context(|| format!("writing {}", output.display()))?;
    println!(
        "Rendered {} atoms ({}×{}) → {}",
        structure.atom_count(),
        width, height, output.display(),
    );
    Ok(())
}

fn run_minimize(
    input: &Path,
    output: &Path,
    algorithm: AlgoFlag,
    max_steps: usize,
    tol: f64,
    max_step_a: f64,
) -> Result<()> {
    let file = fs::File::open(input)
        .with_context(|| format!("opening {}", input.display()))?;
    let mut structure = read_pdb(file)
        .with_context(|| format!("reading {}", input.display()))?;
    let graph = build_topology_graph(&structure);
    let ff = standard_ff();
    let opts = MinimizeOptions {
        algorithm: match algorithm {
            AlgoFlag::Sd => Algorithm::SteepestDescent,
            AlgoFlag::Lbfgs => Algorithm::Lbfgs,
        },
        max_steps,
        gradient_tol: tol,
        max_step_a,
        ..Default::default()
    };
    let result = minimize(&mut structure, &graph, ff, opts);
    println!("Minimization result:");
    println!("  algorithm:      {:?}", result.algorithm);
    println!("  steps:          {}", result.steps);
    println!("  initial energy: {:>12.2} kJ/mol", result.initial_energy);
    println!("  final energy:   {:>12.2} kJ/mol", result.final_energy);
    println!("  max force:      {:>12.4} kJ/mol/Å", result.max_force);
    println!("  converged:      {}", result.converged);
    let mut out_file = fs::File::create(output)
        .with_context(|| format!("creating {}", output.display()))?;
    let title = format!("minimized from {}", input.display());
    write_pdb(&mut out_file, &structure, &title).context("writing minimized PDB")?;
    Ok(())
}

fn run_energy(input: &Path, skip_sasa: bool) -> Result<()> {
    let file = fs::File::open(input)
        .with_context(|| format!("opening {}", input.display()))?;
    let structure = read_pdb(file)
        .with_context(|| format!("reading {}", input.display()))?;
    let graph = build_topology_graph(&structure);
    let ff = standard_ff();

    let bonded = bonded_energy(&structure, &graph, ff);
    let nb = nonbonded_energy(&structure, &graph, ff, DEFAULT_CUTOFF_A);
    let gb = gb_energy(&structure, ff);
    let sasa = if skip_sasa {
        energy::SasaBreakdown::default()
    } else {
        sasa_energy(&structure, ff)
    };

    let total =
        bonded.total_kj_mol() + nb.lj_kj_mol + nb.coulomb_kj_mol + gb.gb_kj_mol + sasa.sasa_kj_mol;

    println!("origami energy report — {}", input.display());
    println!("  residues: {}", structure.residues.len());
    println!("  atoms:    {}", structure.atom_count());
    println!();
    println!("Total: {:>11.2} kJ/mol", total);
    println!();
    println!(
        "  Bond:      {:>11.2}   ({} bonds)",
        bonded.bond_kj_mol, bonded.bond_count
    );
    println!(
        "  Angle:     {:>11.2}   ({} angles)",
        bonded.angle_kj_mol, bonded.angle_count
    );
    println!(
        "  Dihedral:  {:>11.2}   ({} dihedrals)",
        bonded.dihedral_kj_mol, bonded.dihedral_count
    );
    println!(
        "  Improper:  {:>11.2}   ({} impropers)",
        bonded.improper_kj_mol, bonded.improper_count
    );
    println!(
        "  LJ:        {:>11.2}   ({} pairs, {} 1-4)",
        nb.lj_kj_mol, nb.pair_count, nb.one_four_count
    );
    println!("  Coulomb:   {:>11.2}", nb.coulomb_kj_mol);
    println!("  GB:        {:>11.2}   (self {:.2}, cross {:.2})", gb.gb_kj_mol, gb.self_kj_mol, gb.pair_kj_mol);
    if !skip_sasa {
        println!(
            "  SASA:      {:>11.2}   ({:.0} Å² total)",
            sasa.sasa_kj_mol, sasa.total_area_a2
        );
    } else {
        println!("  SASA:      (skipped)");
    }
    if bonded.missing_count > 0 || nb.missing_count > 0 {
        eprintln!(
            "warning: {} bonded + {} nonbonded parameter lookups failed",
            bonded.missing_count, nb.missing_count
        );
    }
    if gb.clamped_count > 0 {
        eprintln!(
            "warning: {} atoms had their effective Born radius clamped",
            gb.clamped_count
        );
    }
    Ok(())
}

fn run_translate(input: &str, orfs: bool, min_aa: usize, three_letter: bool) -> Result<()> {
    let raw = read_input(input)?;
    let records = parse_fasta(&raw).context("parsing FASTA input")?;
    for record in records {
        if orfs {
            let found = find_orfs(&record.sequence, min_aa);
            if found.is_empty() {
                println!("# {} (no ORFs ≥ {} aa)", record.id, min_aa);
                continue;
            }
            for (idx, orf) in found.iter().enumerate() {
                let label = format!(
                    "{}.orf{} frame={} start={} end={} aa={} {}",
                    record.id,
                    idx + 1,
                    orf.frame.label(),
                    orf.start,
                    orf.end,
                    orf.protein.len(),
                    if orf.terminated { "stop=yes" } else { "stop=no" },
                );
                let seq_str = if three_letter {
                    three_letter_string(&orf.protein)
                } else {
                    one_letter_string(&orf.protein)
                };
                println!(">{}", label);
                println!("{}", seq_str);
            }
        } else {
            let outcome = translate_codons(&record.sequence)
                .with_context(|| format!("translating record {:?}", record.id))?;
            let header = if record.description.is_empty() {
                format!(">{} aa={} stop={}", record.id, outcome.protein.len(),
                    if outcome.terminated { "yes" } else { "no" })
            } else {
                format!(">{} {} aa={} stop={}", record.id, record.description,
                    outcome.protein.len(),
                    if outcome.terminated { "yes" } else { "no" })
            };
            println!("{}", header);
            let seq_str = if three_letter {
                three_letter_string(&outcome.protein)
            } else {
                one_letter_string(&outcome.protein)
            };
            println!("{}", seq_str);
        }
    }
    Ok(())
}

fn read_input(input: &str) -> Result<String> {
    if input == "-" {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .context("reading stdin")?;
        Ok(buf)
    } else {
        fs::read_to_string(PathBuf::from(input))
            .with_context(|| format!("reading {input}"))
    }
}

fn run_build(seq: Option<&str>, from_fasta: Option<&str>, output: Option<&std::path::Path>) -> Result<()> {
    let (sequence, title) = if let Some(s) = seq {
        (parse_aa_seq(s)?, format!("seq={}", s))
    } else if let Some(path) = from_fasta {
        let raw = fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
        let (header, body) = parse_protein_fasta(&raw)?;
        (parse_aa_seq(&body)?, header)
    } else {
        return Err(anyhow!("either --seq or --from-fasta is required"));
    };

    let structure = build_extended_chain(&sequence)
        .map_err(|e| anyhow!("chain build failed: {e}"))?;

    if let Some(path) = output {
        let mut file = fs::File::create(path)
            .with_context(|| format!("creating {}", path.display()))?;
        write_pdb(&mut file, &structure, &title).context("writing PDB")?;
    } else {
        let stdout = std::io::stdout();
        let mut handle = stdout.lock();
        write_pdb(&mut handle, &structure, &title).context("writing PDB")?;
        handle.flush().ok();
    }
    Ok(())
}

/// Read a FASTA file containing one or more protein sequences and return the
/// (header, sequence) of the first record. Sequence is the raw one-letter
/// codes with whitespace stripped.
fn parse_protein_fasta(input: &str) -> Result<(String, String)> {
    let mut header = String::new();
    let mut body = String::new();
    let mut started = false;
    for line in input.lines() {
        let line = line.trim_end();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix('>') {
            if started {
                break; // only first record
            }
            header = rest.trim().to_owned();
            started = true;
        } else if started {
            body.extend(line.chars().filter(|c| !c.is_ascii_whitespace()));
        } else {
            return Err(anyhow!("FASTA sequence appears before any > header"));
        }
    }
    if !started {
        return Err(anyhow!("no records in FASTA"));
    }
    Ok((header, body))
}

fn parse_aa_seq(s: &str) -> Result<Vec<AminoAcid>> {
    let mut out = Vec::with_capacity(s.len());
    for (i, ch) in s.chars().enumerate() {
        if ch.is_ascii_whitespace() {
            continue;
        }
        let aa = AminoAcid::from_one_letter(ch).ok_or_else(|| {
            anyhow!("position {i}: {ch:?} is not a valid one-letter amino-acid code")
        })?;
        out.push(aa);
    }
    if out.is_empty() {
        return Err(anyhow!("amino-acid sequence is empty"));
    }
    Ok(out)
}
