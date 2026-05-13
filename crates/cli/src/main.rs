use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use chem::{standard_ff, AminoAcid};
use clap::{Parser, Subcommand, ValueEnum};
use dynamics::{
    minimize, run_cotranslate, run_langevin, Algorithm, CylindricalTunnel, LangevinOptions,
    MinimizeOptions, UniformRibosome,
};
use geom::Vec3;
use energy::{
    bonded::bonded_energy, gb_energy, nonbonded_energy, sasa_energy, DEFAULT_CUTOFF_A,
};
use geom::{build_extended_chain, build_topology_graph};
use io::{
    read_pdb, read_pdb_trajectory, render, structure_bounds, write_pdb, write_pdb_trajectory,
    RenderOptions,
};
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
    /// Render a PDB structure (single-MODEL or multi-MODEL trajectory) to PNG.
    Render {
        /// Input PDB. Multi-MODEL files are rendered per-frame when
        /// `--output-dir` is given.
        input: PathBuf,
        /// Output PNG path (for single-frame inputs).
        #[arg(long, short, conflicts_with = "output_dir")]
        output: Option<PathBuf>,
        /// Output directory for trajectory frames (one PNG per MODEL,
        /// named `frame_NNNN.png`).
        #[arg(long, conflicts_with = "output")]
        output_dir: Option<PathBuf>,
        /// Image width in pixels.
        #[arg(long, default_value_t = 800)]
        width: u32,
        /// Image height in pixels.
        #[arg(long, default_value_t = 600)]
        height: u32,
        /// Include hydrogen atoms (hidden by default).
        #[arg(long)]
        show_hydrogens: bool,
        /// Simulation time per saved frame in femtoseconds. When set
        /// along with `--output-dir`, each frame is stamped with a
        /// `t = N.NN ps` overlay in the top-left corner (units auto-
        /// pick: fs / ps / ns based on magnitude). Compute as
        /// `dt × save_every` from the dynamics / cotranslate run.
        #[arg(long)]
        frame_dt_fs: Option<f64>,
    },
    /// Co-translational chain growth: assemble the chain residue-by-
    /// residue at a constant codon rate while Langevin dynamics relaxes
    /// the existing chain between emissions. Optional cylindrical
    /// exit-tunnel constraint.
    Cotranslate {
        /// Amino-acid sequence (one-letter codes, e.g. "MAGW").
        #[arg(long)]
        seq: String,
        /// Output trajectory PDB (multi-MODEL).
        #[arg(long)]
        output_trajectory: PathBuf,
        /// Per-residue emission interval in femtoseconds. With dt = 1 fs,
        /// `interval = 1000` runs 1 ps of dynamics between residues.
        #[arg(long, default_value_t = 1000.0)]
        interval: f64,
        /// Extra fs of dynamics after the last residue is emitted (lets
        /// the completed chain relax).
        #[arg(long, default_value_t = 5000.0)]
        tail: f64,
        /// Save a frame every N integrator steps.
        #[arg(long, default_value_t = 25)]
        save_every: usize,
        /// Integration timestep in fs.
        #[arg(long, default_value_t = 1.0)]
        dt: f64,
        /// Target temperature (K).
        #[arg(long, default_value_t = 310.0)]
        temperature: f64,
        /// Friction γ in ps⁻¹.
        #[arg(long, default_value_t = 2.0)]
        friction: f64,
        /// PRNG seed.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Enable the cylindrical exit-tunnel constraint.
        #[arg(long)]
        with_tunnel: bool,
        /// Tunnel radius in Å (only with --with-tunnel).
        #[arg(long, default_value_t = 12.0)]
        tunnel_radius: f64,
        /// Tunnel length in Å.
        #[arg(long, default_value_t = 80.0)]
        tunnel_length: f64,
        /// Include the SASA hydrophobic term in the Langevin forces.
        /// Drives hydrophobic collapse as the chain emerges; matches the
        /// `--with-sasa` flag on `origami dynamics`. Off by default.
        #[arg(long)]
        with_sasa: bool,
    },
    /// Run Langevin molecular dynamics at constant temperature, writing
    /// a multi-MODEL trajectory PDB.
    Dynamics {
        /// Input PDB (starting configuration).
        input: PathBuf,
        /// Output trajectory PDB (multi-MODEL).
        #[arg(long)]
        output_trajectory: PathBuf,
        /// Number of integrator steps to run.
        #[arg(long, default_value_t = 1000)]
        steps: usize,
        /// Save a frame every N steps.
        #[arg(long, default_value_t = 10)]
        save_every: usize,
        /// Integration timestep in femtoseconds.
        #[arg(long, default_value_t = 1.0)]
        dt: f64,
        /// Target temperature in Kelvin.
        #[arg(long, default_value_t = 310.0)]
        temperature: f64,
        /// Friction coefficient γ in ps⁻¹.
        #[arg(long, default_value_t = 1.0)]
        friction: f64,
        /// PRNG seed (deterministic).
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Skip Maxwell-Boltzmann initial velocity sampling.
        #[arg(long)]
        zero_initial_velocity: bool,
        /// Include the SASA (hydrophobic) term in the forces. Slow
        /// (~100 ms/step on Trp-cage) but adds the only solvation
        /// contribution PSA.2 currently provides.
        #[arg(long)]
        with_sasa: bool,
        /// Constrain every X-H bond length with SHAKE. Removes the
        /// hydrogen-stretch high-frequency mode that otherwise forces
        /// dt ≤ 1 fs; combine with `--dt 2.0` for a 2× longer
        /// trajectory per wall-second.
        #[arg(long)]
        shake_h: bool,
    },
    /// Replica-exchange molecular dynamics. Runs N Langevin trajectories
    /// at different temperatures with periodic Metropolis swaps between
    /// adjacent pairs, accelerating conformational sampling. The lowest-
    /// T replica is the production trajectory.
    Remd {
        /// Input PDB (starting configuration; shared across all replicas).
        input: PathBuf,
        /// Output trajectory PDB for replica 0 (the production / lowest-T
        /// run). Multi-MODEL.
        #[arg(long)]
        output_trajectory: PathBuf,
        /// Comma-separated list of temperatures (K). Replica 0 = lowest.
        /// A typical 4-replica ladder for protein folding: `300,360,430,520`.
        #[arg(long, default_value = "300,360,430,520", value_delimiter = ',')]
        temperatures: Vec<f64>,
        /// Total simulated time per replica (ps).
        #[arg(long, default_value_t = 5.0)]
        time_ps: f64,
        /// Swap attempt interval (ps).
        #[arg(long, default_value_t = 0.5)]
        swap_interval_ps: f64,
        /// Integration timestep (fs).
        #[arg(long, default_value_t = 1.0)]
        dt: f64,
        /// Save a frame every N integrator steps (production replica only).
        #[arg(long, default_value_t = 100)]
        save_every: usize,
        /// Friction γ (ps⁻¹).
        #[arg(long, default_value_t = 2.0)]
        friction: f64,
        /// PRNG seed.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Include SASA (hydrophobic) forces.
        #[arg(long)]
        with_sasa: bool,
        /// SHAKE the X-H bonds (enables dt = 2 fs).
        #[arg(long)]
        shake_h: bool,
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
        /// Include the SASA (hydrophobic) term in the gradient (PSA.2).
        /// Slow; off by default.
        #[arg(long)]
        with_sasa: bool,
    },
    /// Per-frame trajectory analysis: Cα RMSD vs reference, radius of
    /// gyration, end-to-end distance; optional residue-residue contact
    /// frequency map. Reads any multi-MODEL PDB trajectory.
    Analyze {
        /// Input trajectory PDB (single-MODEL works too — produces one
        /// row of metrics).
        input: PathBuf,
        /// Reference PDB for the RMSD column. Must have the same
        /// residue sequence as each trajectory frame; otherwise RMSD
        /// is reported as NaN.
        #[arg(long)]
        reference: Option<PathBuf>,
        /// Write per-frame metrics TSV here. If omitted, prints to
        /// stdout.
        #[arg(long, short)]
        output: Option<PathBuf>,
        /// Write the residue-residue contact-frequency matrix to this
        /// path (tall TSV: `res_i  res_j  frequency`). Skipped if
        /// omitted.
        #[arg(long)]
        contact_map: Option<PathBuf>,
        /// Heavy-atom distance threshold for the contact map (Å).
        #[arg(long, default_value_t = 8.0)]
        contact_cutoff: f64,
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
        Command::Minimize { input, output, algorithm, max_steps, tol, max_step, with_sasa } => {
            run_minimize(&input, &output, algorithm, max_steps, tol, max_step, with_sasa)
        }
        Command::Render { input, output, output_dir, width, height, show_hydrogens, frame_dt_fs } => {
            run_render(&input, output.as_deref(), output_dir.as_deref(), width, height, show_hydrogens, frame_dt_fs)
        }
        Command::Cotranslate {
            seq,
            output_trajectory,
            interval,
            tail,
            save_every,
            dt,
            temperature,
            friction,
            seed,
            with_tunnel,
            tunnel_radius,
            tunnel_length,
            with_sasa,
        } => run_cotranslate_cmd(
            &seq,
            &output_trajectory,
            interval,
            tail,
            save_every,
            dt,
            temperature,
            friction,
            seed,
            with_tunnel,
            tunnel_radius,
            tunnel_length,
            with_sasa,
        ),
        Command::Dynamics {
            input,
            output_trajectory,
            steps,
            save_every,
            dt,
            temperature,
            friction,
            seed,
            zero_initial_velocity,
            with_sasa,
            shake_h,
        } => run_dynamics(
            &input,
            &output_trajectory,
            steps,
            save_every,
            dt,
            temperature,
            friction,
            seed,
            !zero_initial_velocity,
            with_sasa,
            shake_h,
        ),
        Command::Remd {
            input,
            output_trajectory,
            temperatures,
            time_ps,
            swap_interval_ps,
            dt,
            save_every,
            friction,
            seed,
            with_sasa,
            shake_h,
        } => run_remd_cmd(
            &input,
            &output_trajectory,
            temperatures,
            time_ps,
            swap_interval_ps,
            dt,
            save_every,
            friction,
            seed,
            with_sasa,
            shake_h,
        ),
        Command::Analyze {
            input,
            reference,
            output,
            contact_map,
            contact_cutoff,
        } => run_analyze(
            &input,
            reference.as_deref(),
            output.as_deref(),
            contact_map.as_deref(),
            contact_cutoff,
        ),
    }
}

fn run_render(
    input: &Path,
    output: Option<&Path>,
    output_dir: Option<&Path>,
    width: u32,
    height: u32,
    show_hydrogens: bool,
    frame_dt_fs: Option<f64>,
) -> Result<()> {
    let file = fs::File::open(input)
        .with_context(|| format!("opening {}", input.display()))?;
    let opts = RenderOptions {
        width,
        height,
        show_hydrogens,
        ..Default::default()
    };
    if let Some(dir) = output_dir {
        let frames = read_pdb_trajectory(file)
            .with_context(|| format!("reading {}", input.display()))?;
        fs::create_dir_all(dir)
            .with_context(|| format!("creating {}", dir.display()))?;

        // Lock the camera across all frames using the union extents of
        // the trajectory, so the molecule appears to grow / fold in
        // place instead of bouncing around as the per-frame centroid
        // drifts. Centroid = mean of per-frame centroids; bounding
        // radius = max over frames of (per-frame centroid offset from
        // global + per-frame bounding radius), so the camera always
        // contains every frame.
        let mut centroid_sum = Vec3::zeros();
        let mut per_frame: Vec<(Vec3, f64)> = Vec::with_capacity(frames.len());
        for s in &frames {
            let b = structure_bounds(s, show_hydrogens, opts.atom_scale);
            centroid_sum = centroid_sum + b.0;
            per_frame.push(b);
        }
        let global_centroid = centroid_sum / frames.len().max(1) as f64;
        let global_radius = per_frame
            .iter()
            .map(|(c, r)| (*c - global_centroid).norm() + *r)
            .fold(0.0_f64, f64::max);
        let base_frame_opts = RenderOptions {
            fixed_centroid: Some(global_centroid),
            fixed_bounding_radius: Some(global_radius),
            ..opts
        };

        for (idx, structure) in frames.iter().enumerate() {
            let mut frame_opts = base_frame_opts.clone();
            if let Some(dt) = frame_dt_fs {
                frame_opts.overlay_text = Some(format_sim_time(idx as f64 * dt));
            }
            let img = render(structure, &frame_opts);
            let path = dir.join(format!("frame_{:04}.png", idx + 1));
            img.save(&path)
                .with_context(|| format!("writing {}", path.display()))?;
        }
        println!(
            "Rendered {} frames ({}×{}) → {}",
            frames.len(),
            width,
            height,
            dir.display(),
        );
    } else {
        let out_path = output
            .ok_or_else(|| anyhow!("either --output or --output-dir is required"))?;
        let structure = read_pdb(file)
            .with_context(|| format!("reading {}", input.display()))?;
        let img = render(&structure, &opts);
        img.save(out_path)
            .with_context(|| format!("writing {}", out_path.display()))?;
        println!(
            "Rendered {} atoms ({}×{}) → {}",
            structure.atom_count(),
            width,
            height,
            out_path.display(),
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_remd_cmd(
    input: &Path,
    output_traj: &Path,
    temperatures: Vec<f64>,
    time_ps: f64,
    swap_interval_ps: f64,
    dt_fs: f64,
    save_every: usize,
    friction_ps_inv: f64,
    seed: u64,
    with_sasa: bool,
    shake_h: bool,
) -> Result<()> {
    if temperatures.len() < 2 {
        return Err(anyhow!(
            "remd needs at least 2 temperatures, got {}",
            temperatures.len()
        ));
    }
    let file = fs::File::open(input)
        .with_context(|| format!("opening {}", input.display()))?;
    let structure = read_pdb(file)
        .with_context(|| format!("reading {}", input.display()))?;
    let graph = build_topology_graph(&structure);
    let ff = standard_ff();

    let opts = dynamics::remd::RemdOptions {
        temperatures_k: temperatures.clone(),
        dt_fs,
        friction_ps_inv,
        total_time_fs: time_ps * 1000.0,
        swap_interval_fs: swap_interval_ps * 1000.0,
        save_every,
        seed,
        include_sasa: with_sasa,
        constrain_h_bonds: shake_h,
    };

    let mut sorted_temps = temperatures.clone();
    sorted_temps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let production_t = sorted_temps[0];

    eprintln!(
        "origami remd: {} atoms, {} residues, {} replicas at T={:?} K, dt={} fs, total {} ps, swap every {} ps{}{}",
        structure.atom_count(),
        structure.residues.len(),
        sorted_temps.len(),
        sorted_temps,
        dt_fs,
        time_ps,
        swap_interval_ps,
        if with_sasa { ", +SASA" } else { "" },
        if shake_h { ", +SHAKE" } else { "" },
    );

    // Collect frames from replica 0 (production / lowest-T) into a
    // trajectory PDB. Other replicas' frames are accepted by the
    // callback but not saved (a future flag could control this).
    let mut frames: Vec<geom::Structure> = Vec::new();
    let mut last_report_time_fs = 0.0f64;
    let summary = dynamics::remd::run_remd(&structure, &graph, ff, opts, |frame| {
        if frame.replica_idx == 0 {
            frames.push(frame.structure.clone());
            if frame.time_fs - last_report_time_fs >= 1000.0 || frame.step == 0 {
                eprintln!(
                    "  T={:.0}K  step={:>6} t={:>8.1} fs   T_inst={:.1} K",
                    frame.temperature_k,
                    frame.step,
                    frame.time_fs,
                    frame.instantaneous_temperature_k,
                );
                last_report_time_fs = frame.time_fs;
            }
        }
    });

    let title = format!(
        "REMD production T={production_t}K, {} replicas, {} ps",
        sorted_temps.len(),
        time_ps
    );
    let mut out = fs::File::create(output_traj)
        .with_context(|| format!("creating {}", output_traj.display()))?;
    write_pdb_trajectory(&mut out, &title, frames.iter())
        .context("writing trajectory PDB")?;
    eprintln!(
        "wrote {} frames from replica 0 → {}",
        frames.len(),
        output_traj.display()
    );

    let ratios = summary.acceptance_ratios();
    eprintln!();
    eprintln!("REMD summary:");
    eprintln!("  atoms: {}, replicas: {}", summary.atoms_count, summary.n_replicas);
    for (i, r) in summary.per_replica.iter().enumerate() {
        eprintln!(
            "  replica {} @ T={:.0} K: PE={:.1} kJ/mol, KE={:.1} kJ/mol, {}{}",
            i,
            r.temperature_k,
            r.final_potential_energy_kj_mol,
            r.final_kinetic_energy_kj_mol,
            if r.diverged { "DIVERGED" } else { "ok" },
            if r.shake_failures > 0 {
                format!(" ({} SHAKE failures)", r.shake_failures)
            } else {
                String::new()
            },
        );
    }
    eprintln!("  swap pairs (i ↔ i+1):");
    for (i, &att) in summary.swap_attempts.iter().enumerate() {
        let acc = summary.swap_accepts[i];
        eprintln!(
            "    {}↔{}: {}/{} accepts ({:.1}%)",
            i,
            i + 1,
            acc,
            att,
            100.0 * ratios[i],
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_dynamics(
    input: &Path,
    output_traj: &Path,
    steps: usize,
    save_every: usize,
    dt_fs: f64,
    temperature_k: f64,
    friction_ps_inv: f64,
    seed: u64,
    randomise_initial_velocities: bool,
    include_sasa: bool,
    shake_h: bool,
) -> Result<()> {
    let file = fs::File::open(input)
        .with_context(|| format!("opening {}", input.display()))?;
    let mut structure = read_pdb(file)
        .with_context(|| format!("reading {}", input.display()))?;
    let graph = build_topology_graph(&structure);
    let ff = standard_ff();

    let opts = LangevinOptions {
        dt_fs,
        temperature_k,
        friction_ps_inv,
        steps,
        save_every,
        seed,
        randomise_initial_velocities,
        include_sasa,
        constrain_h_bonds: shake_h,
    };

    eprintln!(
        "origami dynamics: {} atoms, {} residues, T={} K, γ={} ps⁻¹, dt={} fs, {} steps (save every {})",
        structure.atom_count(),
        structure.residues.len(),
        temperature_k,
        friction_ps_inv,
        dt_fs,
        steps,
        save_every,
    );

    let mut frames: Vec<geom::Structure> = Vec::new();
    let summary = run_langevin(&mut structure, &graph, ff, opts, |frame| {
        eprintln!(
            "  step {:>6} t={:>8.1} fs   T={:>7.1} K   KE={:>9.2} kJ/mol",
            frame.step, frame.time_fs, frame.instantaneous_temperature_k, frame.kinetic_energy_kj_mol,
        );
        frames.push(frame.structure.clone());
    });

    eprintln!(
        "\ntemperature mean = {:.1} K ± {:.1} K; equipartition ratio = {:.3}; diverged = {}",
        summary.temperature_mean_k,
        summary.temperature_stddev_k,
        summary.equipartition_ratio,
        summary.diverged,
    );

    let title = format!(
        "Langevin T={} K dt={} fs steps={} from {}",
        temperature_k,
        dt_fs,
        steps,
        input.display(),
    );
    let mut out = fs::File::create(output_traj)
        .with_context(|| format!("creating {}", output_traj.display()))?;
    write_pdb_trajectory(&mut out, &title, frames.iter()).context("writing trajectory PDB")?;
    eprintln!(
        "wrote {} frames → {}",
        frames.len(),
        output_traj.display(),
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_cotranslate_cmd(
    seq_str: &str,
    output_traj: &Path,
    interval_fs: f64,
    tail_fs: f64,
    save_every: usize,
    dt_fs: f64,
    temperature_k: f64,
    friction_ps_inv: f64,
    seed: u64,
    with_tunnel: bool,
    tunnel_radius_a: f64,
    tunnel_length_a: f64,
    with_sasa: bool,
) -> Result<()> {
    let sequence = parse_aa_seq(seq_str)?;
    let ribosome = UniformRibosome::new(sequence.clone(), interval_fs);
    let ff = standard_ff();

    let opts = LangevinOptions {
        dt_fs,
        temperature_k,
        friction_ps_inv,
        steps: 0, // overridden per slice by run_cotranslate
        save_every,
        seed,
        randomise_initial_velocities: true,
        include_sasa: with_sasa,
        constrain_h_bonds: false,
    };

    let tunnel = if with_tunnel {
        Some(CylindricalTunnel {
            axis_origin: Vec3::zeros(),
            axis_direction: Vec3::new(0.0, 0.0, 1.0),
            radius_a: tunnel_radius_a,
            length_a: tunnel_length_a,
            k_confine: 50.0,
        })
    } else {
        None
    };
    let external: Option<&dyn dynamics::ExternalPotential> =
        tunnel.as_ref().map(|t| t as &dyn dynamics::ExternalPotential);

    eprintln!(
        "origami cotranslate: {} residues, interval={} fs, dt={} fs, T={} K, γ={} ps⁻¹{}{}",
        sequence.len(),
        interval_fs,
        dt_fs,
        temperature_k,
        friction_ps_inv,
        if with_tunnel {
            format!(
                ", tunnel(R={} Å L={} Å)",
                tunnel_radius_a, tunnel_length_a
            )
        } else {
            String::new()
        },
        if with_sasa { ", +SASA" } else { "" },
    );

    let tail_steps = (tail_fs / dt_fs).round() as usize;
    let mut frames: Vec<geom::Structure> = Vec::new();
    let mut last_residue = 0usize;
    let final_struct = run_cotranslate(&ribosome, ff, opts, tail_steps, external, |frame| {
        if frame.residue_count != last_residue {
            eprintln!(
                "  residue {:>2}/{:<2} appended at t={:>8.1} fs (chain has {} atoms)",
                frame.residue_count,
                sequence.len(),
                frame.time_fs,
                frame.structure.atom_count(),
            );
            last_residue = frame.residue_count;
        }
        frames.push(frame.structure.clone());
    });

    let title = format!(
        "Cotranslate seq={} interval={}fs dt={}fs T={}K{}",
        seq_str,
        interval_fs,
        dt_fs,
        temperature_k,
        if with_tunnel { " +tunnel" } else { "" }
    );
    let mut out = fs::File::create(output_traj)
        .with_context(|| format!("creating {}", output_traj.display()))?;
    write_pdb_trajectory(&mut out, &title, frames.iter()).context("writing trajectory PDB")?;
    eprintln!(
        "wrote {} frames → {} (final: {} residues, {} atoms)",
        frames.len(),
        output_traj.display(),
        final_struct.residues.len(),
        final_struct.atom_count(),
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_minimize(
    input: &Path,
    output: &Path,
    algorithm: AlgoFlag,
    max_steps: usize,
    tol: f64,
    max_step_a: f64,
    include_sasa: bool,
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
        include_sasa,
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

fn run_analyze(
    input: &Path,
    reference: Option<&Path>,
    output: Option<&Path>,
    contact_map: Option<&Path>,
    contact_cutoff_a: f64,
) -> Result<()> {
    let traj_bytes = fs::read(input)
        .with_context(|| format!("reading {}", input.display()))?;
    let frames = read_pdb_trajectory(traj_bytes.as_slice())
        .with_context(|| format!("parsing trajectory {}", input.display()))?;
    if frames.is_empty() {
        return Err(anyhow!("trajectory {} has no frames", input.display()));
    }
    let reference = match reference {
        Some(p) => {
            let bytes = fs::read(p).with_context(|| format!("reading {}", p.display()))?;
            Some(
                read_pdb(bytes.as_slice())
                    .with_context(|| format!("parsing reference {}", p.display()))?,
            )
        }
        None => None,
    };

    let mut sink: Box<dyn std::io::Write> = match output {
        Some(p) => Box::new(
            fs::File::create(p).with_context(|| format!("creating {}", p.display()))?,
        ),
        None => Box::new(std::io::stdout()),
    };
    writeln!(
        sink,
        "# frame\trmsd_ca_A\trg_ca_A\tend_to_end_A\tn_residues\tn_atoms\tpct_helix\tpct_strand\tss_string"
    )?;
    let mut min_rmsd: f64 = f64::INFINITY;
    let mut min_rmsd_idx: usize = 0;
    let mut last_rmsd: Option<f64> = None;
    for (idx, frame) in frames.iter().enumerate() {
        let rmsd = reference
            .as_ref()
            .and_then(|r| geom::rmsd_ca(r, frame))
            .map(|v| {
                if v < min_rmsd {
                    min_rmsd = v;
                    min_rmsd_idx = idx;
                }
                v
            });
        let rg = geom::radius_of_gyration_ca(frame);
        let e2e = geom::end_to_end_ca(frame);
        last_rmsd = rmsd;
        let rmsd_s = rmsd.map(|v| format!("{v:.3}")).unwrap_or_else(|| "NaN".into());
        let rg_s = rg.map(|v| format!("{v:.3}")).unwrap_or_else(|| "NaN".into());
        let e2e_s = e2e.map(|v| format!("{v:.3}")).unwrap_or_else(|| "NaN".into());
        // DSSP-based secondary structure (Kabsch-Sander H-bond
        // detection). If the structure has no explicit `H` atoms
        // (e.g. heavy-atom-only PDB without our chain builder's
        // hydrogens), the H-bond detector finds nothing and the
        // string is all-`C` — fall back to Ramachandran in that case.
        let n_res = frame.residues.len();
        let mut ss_string = geom::dssp_string(frame);
        let mut counts = geom::dssp_counts(frame);
        if counts.0 == 0 && counts.1 == 0 && n_res >= 4 {
            ss_string = geom::secondary_structure_string(frame);
            counts = geom::ss_counts(frame);
        }
        let (n_h, n_e, _n_c) = counts;
        let pct_helix = if n_res > 0 {
            100.0 * n_h as f64 / n_res as f64
        } else {
            0.0
        };
        let pct_strand = if n_res > 0 {
            100.0 * n_e as f64 / n_res as f64
        } else {
            0.0
        };
        writeln!(
            sink,
            "{}\t{}\t{}\t{}\t{}\t{}\t{:.1}\t{:.1}\t{}",
            idx,
            rmsd_s,
            rg_s,
            e2e_s,
            n_res,
            frame.atom_count(),
            pct_helix,
            pct_strand,
            ss_string,
        )?;
    }
    let _ = last_rmsd;

    let total_frames = frames.len();
    if let Some(_r) = reference.as_ref() {
        if min_rmsd.is_finite() {
            eprintln!(
                "min RMSD over {} frame{}: {:.3} Å at frame {}",
                total_frames,
                if total_frames == 1 { "" } else { "s" },
                min_rmsd,
                min_rmsd_idx,
            );
        } else {
            eprintln!(
                "RMSD vs reference was NaN on every frame — sequences may not match"
            );
        }
    }
    if let Some(path) = contact_map {
        // A cotranslate trajectory contains growth frames with varying
        // residue counts. The contact map only makes sense on a fixed-
        // size chain, so pick the most common residue count (= the
        // fully-grown chain for cotranslate, or the only count for a
        // plain dynamics trajectory) and filter to those frames.
        let target_n = mode_residue_count(&frames);
        let kept: Vec<geom::Structure> = frames
            .iter()
            .filter(|f| f.residues.len() == target_n)
            .cloned()
            .collect();
        let dropped = total_frames - kept.len();
        if dropped > 0 {
            eprintln!(
                "contact map: using {} frames of {} (dropped {} frames whose residue count ≠ {})",
                kept.len(),
                total_frames,
                dropped,
                target_n,
            );
        }
        let map = geom::contact_map_ca(&kept, contact_cutoff_a)
            .ok_or_else(|| anyhow!("contact map needs ≥1 frame with consistent residue counts"))?;
        let mut f = fs::File::create(path)
            .with_context(|| format!("creating {}", path.display()))?;
        writeln!(
            f,
            "# res_i\tres_j\tfreq\t# {} fully-grown frames, cutoff {} Å",
            kept.len(),
            contact_cutoff_a
        )?;
        for i in 0..map.len() {
            for j in 0..map[i].len() {
                writeln!(f, "{}\t{}\t{:.4}", i + 1, j + 1, map[i][j])?;
            }
        }
        eprintln!(
            "wrote contact map ({} × {}) → {}",
            map.len(),
            map.len(),
            path.display(),
        );
    }
    Ok(())
}

/// Format a simulation time in femtoseconds as a short string suitable
/// for the trajectory-frame overlay. The font only has digits + `.`,
/// space, and the unit letters `f p n s`, so the output is restricted
/// to those characters.
fn format_sim_time(t_fs: f64) -> String {
    if t_fs < 1000.0 {
        format!("t = {:.0} fs", t_fs)
    } else if t_fs < 1_000_000.0 {
        format!("t = {:.2} ps", t_fs / 1000.0)
    } else {
        format!("t = {:.3} ns", t_fs / 1_000_000.0)
    }
}

fn mode_residue_count(frames: &[geom::Structure]) -> usize {
    let mut counts: std::collections::BTreeMap<usize, usize> = Default::default();
    for f in frames {
        *counts.entry(f.residues.len()).or_default() += 1;
    }
    // BTreeMap iteration is ascending; tie-break on higher residue count
    // (typical for cotranslate — the fully-grown tail dominates).
    counts
        .into_iter()
        .max_by_key(|(n, c)| (*c, *n))
        .map(|(n, _)| n)
        .unwrap_or(0)
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
