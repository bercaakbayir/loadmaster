use std::process::ExitCode;

use anyhow::Context;
use loadmaster_plan::{Error, PlanRequest, explain};
use serde::Serialize;

use crate::PlanArgs;

/// `--json` output: one object whatever the outcome.
#[derive(Serialize)]
#[serde(tag = "result", rename_all = "snake_case")]
enum JsonOut<'a> {
    Fits { plan: &'a loadmaster_plan::Plan },
    CannotFit { no_fit: &'a loadmaster_plan::NoFit },
}

pub fn run(args: PlanArgs) -> anyhow::Result<ExitCode> {
    let model = match super::open_target(&args.model, args.offline)? {
        super::Target::Model(m) => *m,
        super::Target::Repo { spec, files } => {
            let ggufs: Vec<&str> = files
                .iter()
                .map(|f| f.path.as_str())
                .filter(|p| p.ends_with(".gguf"))
                .collect();
            anyhow::bail!(
                "hf://{}: name a file, e.g. hf://{}/{}",
                spec.repo,
                spec.repo,
                ggufs.first().copied().unwrap_or("<file>.gguf")
            );
        }
    };
    let snap = super::snapshot(&args.hw)?;
    let gpu =
        match args.gpu.as_str() {
            "auto" => snap.best_gpu(),
            q => Some(snap.find_gpu(q).with_context(|| {
                format!("no GPU matches `{q}`; run `loadmaster gpu` to list them")
            })?),
        }
        .ok_or(Error::NoGpu)?
        .clone();

    let req = PlanRequest {
        ctx: args.ctx,
        runtime: args.runtime.into(),
        ..PlanRequest::default()
    };
    match loadmaster_plan::plan(&model, &snap, &gpu, &req) {
        Ok(p) => {
            if args.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&JsonOut::Fits { plan: &p })?
                );
            } else if args.explain {
                print!("{}", explain::explain(&p));
            } else {
                print!("{}", explain::summary(&p));
            }
            Ok(ExitCode::SUCCESS)
        }
        Err(Error::CannotFit(n)) => {
            if args.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&JsonOut::CannotFit { no_fit: &n })?
                );
            } else {
                print!("{}", explain::no_fit(&n, args.explain));
            }
            Ok(ExitCode::from(1))
        }
        Err(e) => Err(e.into()),
    }
}
