//! `loadmaster` CLI. Subcommands delegate to the library crates.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(name = "loadmaster", version, about, propagate_version = true)]
struct Cli {
    /// Increase log verbosity (logs go to stderr).
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Index local model stores (HF cache, Ollama, LM Studio, loose dirs). Read-only.
    Scan {
        /// Extra directories to scan for loose GGUF files.
        #[arg(long)]
        path: Vec<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Detect GPUs and usable memory.
    Gpu {
        #[arg(long)]
        json: bool,
    },
    /// Compute the memory budget and the exact launch command for a model.
    Plan {
        /// Model path, `hf://repo/file`, or a name from `scan`.
        model: String,
        /// `auto` or a GPU name from `loadmaster gpu`.
        #[arg(long, default_value = "auto")]
        gpu: String,
        /// Context length in tokens.
        #[arg(long)]
        ctx: Option<u32>,
        #[arg(long, value_enum, default_value_t = Runtime::LlamaCpp)]
        runtime: Runtime,
        /// Print every term of the footprint and why each choice was made.
        #[arg(long)]
        explain: bool,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum Runtime {
    #[value(name = "llama.cpp")]
    LlamaCpp,
    Vllm,
    Exl3,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    init_tracing(cli.verbose);
    match run(cli) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

fn init_tracing(verbose: u8) {
    let level = match verbose {
        0 => "warn",
        1 => "debug",
        _ => "trace",
    };
    let filter = tracing_subscriber::EnvFilter::try_from_env("LOADMASTER_LOG")
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(level));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();
}

fn run(cli: Cli) -> anyhow::Result<ExitCode> {
    match cli.command {
        Command::Scan { .. } | Command::Gpu { .. } | Command::Plan { .. } => {
            anyhow::bail!("not implemented yet; see CHANGELOG.md for what has shipped")
        }
    }
}
