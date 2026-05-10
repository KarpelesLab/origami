use std::fs;
use std::io::{Read, Write};
use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use chem::AminoAcid;
use clap::{Parser, Subcommand};
use geom::build_extended_chain;
use io::write_pdb;
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
    }
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
