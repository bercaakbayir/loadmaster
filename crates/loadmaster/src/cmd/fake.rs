//! `loadmaster __fake-runtime`: a fake OpenAI-compatible server for tests. It picks out its
//! own options and ignores everything else, so it can stand in for llama-server, vllm or
//! tabbyAPI in a lockfile.

use std::collections::HashSet;
use std::process::ExitCode;

use anyhow::{Context, bail};
use clap::Args;
use loadmaster_probe::fake::{Config, Impersonate, Misbehavior};

#[derive(Args)]
pub struct FakeArgs {
    /// `--port N` (required), `--host H`, `--name ID`, `--misbehave a,b`,
    /// `--impersonate llama.cpp|vllm|ollama|lmstudio|sglang|tabbyapi`, `--load-delay SECS`.
    /// `-m PATH` sets the reported chat template from that model file. Anything else
    /// (runtime flags such as `-c`) is ignored.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    args: Vec<String>,
}

pub fn run(a: FakeArgs) -> anyhow::Result<ExitCode> {
    let value = |flag: &str| -> Option<&str> {
        a.args
            .iter()
            .position(|x| x == flag)
            .and_then(|i| a.args.get(i + 1))
            .map(String::as_str)
    };
    let Some(port) = value("--port") else {
        bail!("--port is required");
    };
    let port: u16 = port.parse().context("--port")?;
    let host = value("--host").unwrap_or("127.0.0.1").to_owned();
    let mut misbehave = HashSet::new();
    for m in value("--misbehave")
        .unwrap_or("")
        .split(',')
        .filter(|s| !s.is_empty())
    {
        misbehave
            .insert(Misbehavior::parse(m).with_context(|| format!("unknown misbehaviour `{m}`"))?);
    }
    let mut cfg = Config {
        model: value("--name").unwrap_or("fake-model").to_owned(),
        misbehave,
        impersonate: Impersonate::parse(value("--impersonate").unwrap_or("llama.cpp"))
            .context("unknown runtime to impersonate")?,
        ..Config::default()
    };
    // Like llama-server, report the chat template of the model file it was given.
    if let Some(path) = value("-m").or(value("--model"))
        && let Ok(info) = loadmaster_model::ModelInfo::open(path)
        && let Some(t) = info.chat_template
    {
        cfg.chat_template = t.source;
    }
    let delay: f64 = value("--load-delay")
        .unwrap_or("0")
        .parse()
        .context("--load-delay")?;
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(async move {
        if delay > 0.0 {
            tokio::time::sleep(std::time::Duration::from_secs_f64(delay)).await;
        }
        let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await?;
        loadmaster_probe::fake::serve(listener, cfg).await
    })?;
    Ok(ExitCode::SUCCESS)
}
