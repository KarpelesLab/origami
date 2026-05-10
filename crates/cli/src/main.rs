use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
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
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Translate { input, orfs, min_aa, three_letter } => {
            run_translate(&input, orfs, min_aa, three_letter)
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
        io::stdin()
            .read_to_string(&mut buf)
            .context("reading stdin")?;
        Ok(buf)
    } else {
        fs::read_to_string(PathBuf::from(input))
            .with_context(|| format!("reading {input}"))
    }
}
