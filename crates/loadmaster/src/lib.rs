//! `loadmaster` CLI (also installed as `lm`). Subcommands delegate to the library crates.

mod cmd;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand, ValueEnum};

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
        /// Extra directories to scan for GGUF files and HF model directories.
        #[arg(long)]
        path: Vec<PathBuf>,
        #[arg(long)]
        json: bool,
        #[command(flatten)]
        hw: HwArgs,
    },
    /// Detect GPUs and usable memory.
    Gpu {
        #[arg(long)]
        json: bool,
        #[command(flatten)]
        hw: HwArgs,
    },
    /// Compute the memory budget and the exact launch command for a model.
    Plan(PlanArgs),
}

#[derive(Args)]
struct HwArgs {
    /// Read the hardware snapshot from a file (as written by `loadmaster gpu --json`)
    /// instead of detecting it. For reproducing plans and bug reports.
    #[arg(long, hide = true, global = true)]
    hw_file: Option<PathBuf>,
}

#[derive(Args)]
struct PlanArgs {
    /// A .gguf file (any part of a split model), a Hugging Face model directory, or a name
    /// from `loadmaster scan` (`llama3.2:1b`, `org/repo`, `hf://org/repo/file.gguf`).
    model: String,
    /// `auto`, a GPU index, or part of a GPU name from `loadmaster gpu`.
    #[arg(long, default_value = "auto")]
    gpu: String,
    /// Context length in tokens. Default: the largest that fits fully on the GPU.
    #[arg(long)]
    ctx: Option<u32>,
    #[arg(long, value_enum, default_value_t = Runtime::LlamaCpp)]
    runtime: Runtime,
    /// Print every term of the footprint and why each choice was made.
    #[arg(long)]
    explain: bool,
    #[arg(long, conflicts_with = "explain")]
    json: bool,
    #[command(flatten)]
    hw: HwArgs,
}

#[derive(Clone, Copy, ValueEnum)]
enum Runtime {
    #[value(name = "llama.cpp")]
    LlamaCpp,
    Vllm,
    Exl3,
}

impl From<Runtime> for loadmaster_plan::Runtime {
    fn from(r: Runtime) -> Self {
        match r {
            Runtime::LlamaCpp => loadmaster_plan::Runtime::LlamaCpp,
            Runtime::Vllm => loadmaster_plan::Runtime::Vllm,
            Runtime::Exl3 => loadmaster_plan::Runtime::Exl3,
        }
    }
}

/// Entry point shared by the `loadmaster` and `lm` binaries.
pub fn main() -> ExitCode {
    let cli = Cli::parse();
    init_tracing(cli.verbose);
    match run(cli) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::from(2)
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
        Command::Scan { path, json, hw } => cmd::scan::run(path, json, hw.hw_file.as_deref()),
        Command::Gpu { json, hw } => cmd::gpu::run(json, hw.hw_file.as_deref()),
        Command::Plan(args) => cmd::plan::run(args),
    }
}
