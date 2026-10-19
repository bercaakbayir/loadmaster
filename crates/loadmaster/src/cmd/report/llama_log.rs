//! Reads what a `llama-server -lv 4` run allocated and how it ended, from its stderr. The
//! formats are llama.cpp's (commit f174553; see `fixtures/llama-log/README.md`); buffer
//! lines are only printed at verbosity 4 or above.

use serde::Serialize;

/// What a buffer holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Model,
    Kv,
    /// Recurrent state (Mamba-style layers).
    Rs,
    Compute,
    Output,
    /// The vision projector's compute buffer (`reserve_compute_meta` in clip.cpp).
    VisionCompute,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Buffer {
    /// The ggml buffer name: `CUDA0`, `MTL0_Mapped`, `CPU`, `CUDA_Host`, ...
    pub buffer: String,
    pub kind: Kind,
    pub mib: f64,
}

impl Buffer {
    /// The device a buffer lives on: `CUDA0`, `MTL0`, `ROCm1`; host buffers (`CPU*`,
    /// `*_Host`) are `CPU`.
    pub fn device(&self) -> &str {
        let b = self.buffer.as_str();
        if b.starts_with("CPU") || b.ends_with("_Host") {
            return "CPU";
        }
        b.split('_').next().unwrap_or(b)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "result", content = "reason")]
pub enum Outcome {
    /// `model loaded` / `listening on` appeared.
    Ready,
    /// An allocation failed.
    Oom(String),
    /// The load failed for another reason.
    Failed(String),
    /// Neither: the log ended (or the run was stopped) before either.
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Default)]
pub struct LoadLog {
    pub buffers: Vec<Buffer>,
    /// clip.cpp's `model size:` for the vision projector, MiB.
    pub vision_model_mib: Option<f64>,
    pub projector: Option<String>,
    /// `build N (commit)` when the trace-level line is present.
    pub build: Option<String>,
    pub outcome: Option<Outcome>,
}

/// `   CUDA0 model buffer size =  4455.35 MiB` after the `func:` prefix.
fn buffer_line(func: &str, rest: &str) -> Option<Buffer> {
    let (head, tail) = rest.split_once(" buffer size = ")?;
    let mib: f64 = tail.trim().strip_suffix("MiB")?.trim().parse().ok()?;
    let mut words = head.split_whitespace();
    let buffer = words.next()?.to_owned();
    let kind = match (words.next()?, func) {
        ("compute", "reserve_compute_meta") => Kind::VisionCompute,
        ("model", _) => Kind::Model,
        ("KV", _) => Kind::Kv,
        ("RS", _) => Kind::Rs,
        ("compute", _) => Kind::Compute,
        ("output", _) => Kind::Output,
        _ => return None,
    };
    words
        .next()
        .is_none()
        .then_some(Buffer { buffer, kind, mib })
}

pub fn parse(log: &str) -> LoadLog {
    let mut out = LoadLog::default();
    for raw in log.lines() {
        // A timestamp or `--log-prefix` may precede the `func:` prefix.
        let line = raw.trim();
        let Some((func, rest)) = line.split_once(": ") else {
            continue;
        };
        let func = func.rsplit(' ').next().unwrap_or(func);
        if let Some(b) = buffer_line(func, rest) {
            out.buffers.push(b);
            continue;
        }
        let rest = rest.trim();
        if func == "load_hparams"
            && let Some(v) = rest.strip_prefix("model size:")
        {
            out.vision_model_mib = v
                .trim()
                .strip_suffix("MiB")
                .and_then(|n| n.trim().parse().ok());
        } else if let Some(p) = rest.strip_prefix("projector:") {
            out.projector = Some(p.trim().to_owned());
        } else if let Some(b) = rest.strip_prefix("build ") {
            // `build 6700 (abc1234) with cc ... for ...`
            out.build = b.split(" with ").next().map(str::to_owned);
        }
        // Allocation failures win over the generic exit line that follows them.
        let oom = ["cudaMalloc failed", "failed to allocate", "out of memory"];
        if oom.iter().any(|p| rest.contains(p)) {
            if !matches!(out.outcome, Some(Outcome::Oom(_))) {
                out.outcome = Some(Outcome::Oom(rest.to_owned()));
            }
        } else if rest.contains("exiting due to model loading error")
            || rest.starts_with("failed to load model")
        {
            if out.outcome.is_none() {
                out.outcome = Some(Outcome::Failed(rest.to_owned()));
            }
        } else if (rest == "model loaded" || rest.starts_with("listening on "))
            && out.outcome.is_none()
        {
            out.outcome = Some(Outcome::Ready);
        }
    }
    out
}

impl LoadLog {
    pub fn outcome(&self) -> Outcome {
        self.outcome.clone().unwrap_or(Outcome::Unknown)
    }

    /// MiB of one kind on one device (`CPU` for host buffers).
    pub fn mib(&self, device: &str, kind: Kind) -> f64 {
        // `+ 0.0` turns the empty sum's -0.0 into 0.0.
        self.buffers
            .iter()
            .filter(|b| b.kind == kind && b.device() == device)
            .map(|b| b.mib)
            .sum::<f64>()
            + 0.0
    }

    /// Accelerator devices that hold any buffer, in first-seen order.
    pub fn devices(&self) -> Vec<String> {
        let mut d: Vec<String> = Vec::new();
        for b in &self.buffers {
            let dev = b.device();
            if dev != "CPU" && !d.iter().any(|x| x == dev) {
                d.push(dev.to_owned());
            }
        }
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> LoadLog {
        let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/llama-log/");
        parse(&std::fs::read_to_string(format!("{p}{name}")).unwrap())
    }

    #[test]
    fn cuda_run_that_fits() {
        let l = fixture("cuda-fits.log");
        assert_eq!(l.outcome(), Outcome::Ready);
        assert_eq!(l.devices(), ["CUDA0"]);
        assert_eq!(l.mib("CUDA0", Kind::Model), 4455.35);
        assert_eq!(l.mib("CUDA0", Kind::Kv), 4608.0);
        assert_eq!(l.mib("CUDA0", Kind::Compute), 304.75);
        // CPU_Mapped weights and CUDA_Host buffers are host memory.
        assert_eq!(l.mib("CPU", Kind::Model), 333.84);
        assert_eq!(l.mib("CPU", Kind::Compute), 72.01);
        assert_eq!(l.mib("CPU", Kind::Output), 0.58);
        assert_eq!(l.build.as_deref(), Some("6700 (abc1234)"));
    }

    #[test]
    fn allocation_failures_are_oom() {
        for f in ["cuda-oom.log", "metal-oom.log"] {
            let l = fixture(f);
            assert!(
                matches!(l.outcome(), Outcome::Oom(_)),
                "{f}: {:?}",
                l.outcome
            );
        }
        let l = fixture("cuda-oom.log");
        let Outcome::Oom(why) = l.outcome() else {
            unreachable!()
        };
        assert!(why.contains("cudaMalloc failed: out of memory"), "{why}");
    }

    #[test]
    fn vision_projector_buffers_are_separate() {
        let l = fixture("metal-vision.log");
        assert_eq!(l.outcome(), Outcome::Ready);
        assert_eq!(l.devices(), ["MTL0"]);
        assert_eq!(l.mib("MTL0", Kind::Compute), 519.62);
        assert_eq!(l.mib("MTL0", Kind::VisionCompute), 1132.34);
        assert_eq!(l.vision_model_mib, Some(814.6));
        assert_eq!(l.projector.as_deref(), Some("gemma3"));
    }

    #[test]
    fn tolerates_prefixes_and_noise() {
        let l = parse(
            "0.01.234.567 I load_tensors:        CUDA1 model buffer size =   12.50 MiB\n\
             garbage line\nllama_kv_cache: size = 1.00 MiB (  256 cells)\n",
        );
        assert_eq!(l.devices(), ["CUDA1"]);
        assert_eq!(l.outcome(), Outcome::Unknown);
    }
}
