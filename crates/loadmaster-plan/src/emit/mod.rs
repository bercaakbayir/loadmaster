//! Runtime flag emitters. Each is a pure function of a [`Plan`](crate::Plan).

pub mod exl3;
pub mod llama_cpp;
pub mod vllm;

use crate::{Plan, Runtime};

/// The launch command for a plan, as one shell line; for exl3 the config file follows.
pub fn command(plan: &Plan) -> String {
    let line = shell_join(&plan.argv);
    match (&plan.runtime, &plan.config_file) {
        (Runtime::Exl3, Some(f)) => format!("{line}\n\n# {}\n{}", f.name, f.contents.trim_end()),
        _ => line,
    }
}

/// Join argv into one line a POSIX shell reads back as the same argv.
pub fn shell_join(argv: &[String]) -> String {
    argv.iter()
        .map(|a| shell_quote(a))
        .collect::<Vec<_>>()
        .join(" ")
}

fn shell_quote(arg: &str) -> String {
    let safe = !arg.is_empty()
        && arg
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_-./:=@%+,".contains(&b));
    if safe {
        arg.to_owned()
    } else {
        format!("'{}'", arg.replace('\'', r"'\''"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quotes_only_when_needed() {
        let argv = [
            "llama-server",
            "-m",
            "/models/My Model's.gguf",
            "-c",
            "8192",
        ]
        .map(String::from);
        assert_eq!(
            shell_join(&argv),
            r"llama-server -m '/models/My Model'\''s.gguf' -c 8192"
        );
    }
}
