//! `loadmaster` CLI (also installed as `lm`). Subcommands delegate to the library crates.

mod cmd;
mod ui;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(name = "loadmaster", version, about, propagate_version = true)]
struct Cli {
    /// Increase log verbosity (logs go to stderr).
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Colour: `auto` (only on a terminal, and not under NO_COLOR), `always`, `never`.
    #[arg(long, global = true, value_enum, default_value_t = Color::Auto)]
    color: Color,

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
        /// Also show architecture, training context, projector and chat template,
        /// and list every directory that was skipped.
        #[arg(long)]
        wide: bool,
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
    /// Search the Hugging Face Hub for models that run on this machine, best fit first.
    Suggest(SuggestArgs),
    /// Scan, plan the chosen models, and write models.lock.
    Init(InitArgs),
    /// Plan one more model and add it to models.lock.
    Add(AddArgs),
    /// Serve the models in models.lock on one OpenAI-compatible port.
    Up(UpArgs),
    /// Stop a running `loadmaster up`.
    Down(LockArgs),
    /// Load a model now, unloading others if its footprint needs the room.
    Swap {
        name: String,
        #[command(flatten)]
        lock: LockArgs,
    },
    /// Show what the running supervisor has loaded.
    Status {
        #[arg(long)]
        json: bool,
        #[command(flatten)]
        lock: LockArgs,
    },
    /// Convert another tool's config into models.lock.
    #[command(subcommand)]
    Import(ImportCmd),
    /// Does tool calling and structured output work on this endpoint? Exit 0 pass,
    /// 1 partial, 2 fail.
    Probe(ProbeArgs),
    /// Run a model once and write what it really used (and how fast it was) next to the
    /// prediction, as a validation case you can review and share. Nothing is uploaded.
    Report(ReportArgs),
    /// A fake OpenAI-compatible runtime for tests.
    #[command(name = "__fake-runtime", hide = true)]
    FakeRuntime(cmd::fake::FakeArgs),
}

#[derive(Subcommand)]
enum ImportCmd {
    /// Import a llama-swap config.yaml.
    LlamaSwap {
        config: PathBuf,
        /// Overwrite an existing models.lock.
        #[arg(long)]
        force: bool,
        #[command(flatten)]
        lock: LockArgs,
        #[command(flatten)]
        hw: HwArgs,
    },
}

#[derive(Args)]
#[command(args_conflicts_with_subcommands = true, subcommand_negates_reqs = true)]
struct ProbeArgs {
    /// OpenAI-compatible base URL, e.g. http://127.0.0.1:8080
    #[arg(required = true)]
    url: Option<String>,
    /// Model name to send.
    #[arg(long, required = true)]
    model: Option<String>,
    #[arg(long)]
    json: bool,
    /// Run the fixtures in this directory instead of the built-in ones.
    #[arg(long)]
    fixtures: Option<PathBuf>,
    /// Only these fixture ids (comma-separated).
    #[arg(long, value_delimiter = ',')]
    only: Vec<String>,
    /// Runs per fixture (default 3; all must agree).
    #[arg(long)]
    repeat: Option<u32>,
    /// blake3 of the chat template the endpoint should be using.
    #[arg(long)]
    expect_template: Option<String>,
    /// Extra known-issue signatures (TOML), on top of the built-in list.
    #[arg(long)]
    known_issues: Option<PathBuf>,
    /// Environment variable holding an API key to send as a bearer token.
    #[arg(long)]
    api_key_env: Option<String>,
    /// Check an embedding model (`/v1/embeddings`) instead of chat and tool calling.
    #[arg(long, conflicts_with_all = ["fixtures", "only", "expect_template"])]
    embeddings: bool,
    #[command(subcommand)]
    sub: Option<ProbeCmd>,
}

#[derive(Subcommand)]
enum ProbeCmd {
    /// Run a JSONL prompts file against models; results go to ~/.loadmaster/eval.db.
    Eval {
        prompts: PathBuf,
        /// Models to run (repeatable).
        #[arg(long = "model", required = true)]
        models: Vec<String>,
        /// Endpoint (default: the proxy in models.lock).
        #[arg(long)]
        url: Option<String>,
        #[arg(long, default_value = loadmaster_lock::FILE_NAME)]
        lock: PathBuf,
        #[arg(long)]
        api_key_env: Option<String>,
    },
    /// Compare two eval runs by id.
    Compare { a: i64, b: i64 },
}

#[derive(Args, Clone)]
struct LockArgs {
    /// The lockfile.
    #[arg(long, default_value = loadmaster_lock::FILE_NAME)]
    lock: PathBuf,
}

#[derive(Args)]
struct InitArgs {
    /// Models to include, by name from `loadmaster scan`.
    names: Vec<String>,
    /// Include every model found.
    #[arg(long, conflicts_with = "names")]
    all: bool,
    /// Context length for every model (default: the largest that fits on the GPU).
    #[arg(long)]
    ctx: Option<u32>,
    /// Address the proxy listens on.
    #[arg(long, default_value = loadmaster_lock::DEFAULT_LISTEN)]
    listen: String,
    /// Overwrite an existing models.lock.
    #[arg(long)]
    force: bool,
    /// Skip hashing model files that the store does not already name by hash.
    #[arg(long)]
    no_hash: bool,
    #[command(flatten)]
    lock: LockArgs,
    #[command(flatten)]
    hw: HwArgs,
}

#[derive(Args)]
struct AddArgs {
    /// Model path or name from `loadmaster scan`.
    model: String,
    /// Name to serve it as (default: derived from the model).
    #[arg(long = "as")]
    name: Option<String>,
    #[arg(long)]
    ctx: Option<u32>,
    #[arg(long, value_enum)]
    runtime: Option<Runtime>,
    /// Extra names the proxy maps to this model.
    #[arg(long = "alias")]
    aliases: Vec<String>,
    /// Never unload for idleness, and evict last.
    #[arg(long)]
    keep_warm: bool,
    #[arg(long)]
    no_hash: bool,
    #[command(flatten)]
    lock: LockArgs,
    #[command(flatten)]
    hw: HwArgs,
}

#[derive(Args)]
struct UpArgs {
    /// After each model is ready, run the conformance probe and record verdicts in
    /// models.lock.
    #[arg(long)]
    verify: bool,
    #[command(flatten)]
    lock: LockArgs,
    #[command(flatten)]
    hw: HwArgs,
}

#[derive(Args)]
struct ReportArgs {
    /// A .gguf file or a name from `loadmaster scan`.
    #[arg(required_unless_present = "from_log")]
    model: Option<String>,
    /// Context length (default: what `plan` picks).
    #[arg(long)]
    ctx: Option<u32>,
    /// Report from the last load of NAME in `loadmaster up`'s log instead of a new run.
    #[arg(long, value_name = "NAME", conflicts_with_all = ["model", "ctx"])]
    from_log: Option<String>,
    /// The llama-server to run.
    #[arg(long, default_value = "llama-server")]
    runtime_bin: String,
    /// Start without asking.
    #[arg(long, short)]
    yes: bool,
    /// On unified memory (Apple Silicon), also run plans under the free-RAM safety margin.
    #[arg(long)]
    unsafe_unified: bool,
    /// On a dedicated GPU, run every layer on the GPU even when the plan says it won't fit,
    /// to check that prediction. Refused on unified memory.
    #[arg(long)]
    try_anyway: bool,
    /// Seconds to wait for the model to load.
    #[arg(long, default_value_t = 600)]
    timeout: u64,
    /// Directory for the report file.
    #[arg(long, default_value = ".")]
    out: PathBuf,
    /// Also print the whole report file (it is always written, and shown in full on the
    /// GitHub issue page before you submit).
    #[arg(long)]
    show: bool,
    #[command(flatten)]
    lock: LockArgs,
    #[command(flatten)]
    hw: HwArgs,
}

#[derive(Args)]
pub(crate) struct HwArgs {
    /// Plan for hardware you don't have: a card (`rtx 4090`), a Mac (`m4 pro 48gb`) or a
    /// size (`24GB`, `48GB-unified`), optionally with host RAM (`rtx 3090+128GB`).
    /// `loadmaster gpu --pretend list` shows the catalog.
    #[arg(long, global = true, conflicts_with = "hw_file")]
    pub(crate) pretend: Option<String>,
    /// Read the hardware snapshot from a file (as written by `loadmaster gpu --json`)
    /// instead of detecting it. For reproducing plans and bug reports.
    #[arg(long, hide = true, global = true)]
    pub(crate) hw_file: Option<PathBuf>,
}

#[derive(Args)]
struct PlanArgs {
    /// A .gguf file (any part of a split model), a Hugging Face model directory, a name
    /// from `loadmaster scan` (`llama3.2:1b`, `org/repo`), or a Hub model:
    /// `hf://org/repo/file.gguf` plans one file, `hf://org/repo` compares a repo's quants.
    /// Hub models not in the local cache are planned from their headers (a few MB), not
    /// downloaded.
    model: String,
    /// Use only Hub headers already cached; never touch the network.
    #[arg(long)]
    offline: bool,
    /// `auto` (every NVIDIA GPU, as llama.cpp splits by default; else the largest), `all`,
    /// indices `0,1`, or one GPU by index or part of its name from `loadmaster gpu`.
    #[arg(long, default_value = "auto")]
    gpu: String,
    /// Context length in tokens. Default: the largest that fits fully on the GPU.
    #[arg(long)]
    ctx: Option<u32>,
    #[arg(long, value_enum, default_value_t = Runtime::LlamaCpp)]
    runtime: Runtime,
    /// Keep a vision or audio model's projector on the CPU (llama.cpp `--no-mmproj-offload`).
    #[arg(long)]
    no_mmproj_offload: bool,
    /// Print every term of the footprint and why each choice was made.
    #[arg(long)]
    explain: bool,
    #[arg(long, conflicts_with = "explain")]
    json: bool,
    /// Print only the launch command, on one line (nothing when the model does not fit).
    #[arg(long, conflicts_with_all = ["explain", "json"])]
    command: bool,
    /// Re-plan the models.lock entry named MODEL and write the new derived fields.
    #[arg(long)]
    update: bool,
    #[arg(long, default_value = loadmaster_lock::FILE_NAME)]
    lock: PathBuf,
    #[command(flatten)]
    hw: HwArgs,
}

#[derive(Args)]
struct SuggestArgs {
    /// Search term, matched against repo names (e.g. `qwen`, `gemma 12b`).
    query: Option<String>,
    /// `chat` (text generation, vision and audio models included), `vision`, `audio`
    /// (speech input) or `embed`.
    #[arg(long, value_enum, default_value_t = SuggestKind::Chat)]
    kind: SuggestKind,
    /// llama.cpp searches GGUF repos and compares every quant; vllm searches safetensors.
    #[arg(long, value_enum, default_value_t = Runtime::LlamaCpp)]
    runtime: Runtime,
    /// Context length to plan at. Default: the largest that fits.
    #[arg(long)]
    ctx: Option<u32>,
    /// GPUs to plan on, as for `plan`.
    #[arg(long, default_value = "auto")]
    gpu: String,
    /// Which models the Hub lists first.
    #[arg(long, value_enum, default_value_t = SuggestSort::Trending)]
    sort: SuggestSort,
    /// Rows to show.
    #[arg(long, default_value_t = 10)]
    limit: usize,
    /// Repos to consider from the Hub (each costs a listing and, if it may fit, a header).
    #[arg(long, default_value_t = 40, value_parser = clap::value_parser!(u32).range(1..=100))]
    candidates: u32,
    /// Also list models that do not fit.
    #[arg(long)]
    all: bool,
    /// Include repos by anyone, not just known quantizers (ggml-org, unsloth, bartowski,
    /// lmstudio-community) and the model's own maker.
    #[arg(long)]
    any_publisher: bool,
    /// Use only cached searches and headers; never touch the network.
    #[arg(long)]
    offline: bool,
    #[arg(long)]
    json: bool,
    #[command(flatten)]
    hw: HwArgs,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SuggestKind {
    Chat,
    Vision,
    Audio,
    Embed,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SuggestSort {
    Trending,
    Downloads,
    New,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Color {
    Auto,
    Always,
    Never,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
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
    match cli.color {
        Color::Auto => anstream::ColorChoice::Auto,
        Color::Always => anstream::ColorChoice::Always,
        Color::Never => anstream::ColorChoice::Never,
    }
    .write_global();
    match run(cli) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("error: {err:#}");
            // 0-2 are results (fits/partial/fail); 3 is "could not answer".
            ExitCode::from(3)
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
        Command::Scan {
            path,
            json,
            wide,
            hw,
        } => cmd::scan::run(path, json, wide, &hw),
        Command::Gpu { json, hw } => cmd::gpu::run(json, &hw),
        Command::Plan(args) if args.update => cmd::lock::update(args),
        Command::Plan(args) => cmd::plan::run(args),
        Command::Suggest(args) => cmd::suggest::run(args),
        Command::Init(args) => cmd::lock::init(args),
        Command::Add(args) => cmd::lock::add(args),
        Command::Up(args) => cmd::serve::up(args),
        Command::Down(lock) => cmd::serve::down(&lock.lock),
        Command::Swap { name, lock } => cmd::serve::swap(&lock.lock, &name),
        Command::Status { json, lock } => cmd::serve::status(&lock.lock, json),
        Command::Import(ImportCmd::LlamaSwap {
            config,
            force,
            lock,
            hw,
        }) => cmd::import::llama_swap(&config, &lock.lock, force, &hw),
        Command::Probe(args) => cmd::probe::run(args),
        Command::Report(r) => cmd::report::run(cmd::report::Args {
            model: r.model,
            from_log: r.from_log,
            ctx: r.ctx,
            runtime_bin: r.runtime_bin,
            yes: r.yes,
            unsafe_unified: r.unsafe_unified,
            try_anyway: r.try_anyway,
            timeout_s: r.timeout,
            out: r.out,
            show: r.show,
            lock: r.lock.lock,
            hw: r.hw,
        }),
        Command::FakeRuntime(args) => cmd::fake::run(args),
    }
}
