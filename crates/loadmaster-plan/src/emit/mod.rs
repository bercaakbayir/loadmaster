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

/// [`shell_join`] over several lines of at most `width` columns, each but the last ending
/// in ` \` and the rest indented two spaces. Lines break only before a flag, so a flag
/// stays with its values; one longer than `width` gets a line of its own.
pub fn shell_wrap(argv: &[String], width: usize) -> Vec<String> {
    let mut groups: Vec<String> = Vec::new();
    for a in argv {
        let q = shell_quote(a);
        match groups.last_mut() {
            Some(g) if !a.starts_with('-') => {
                g.push(' ');
                g.push_str(&q);
            }
            _ => groups.push(q),
        }
    }
    let mut lines: Vec<String> = Vec::new();
    for g in groups {
        match lines.last_mut() {
            Some(l) if l.len() + 1 + g.len() + 2 <= width => {
                l.push(' ');
                l.push_str(&g);
            }
            _ => lines.push(if lines.is_empty() {
                g
            } else {
                format!("  {g}")
            }),
        }
    }
    let n = lines.len();
    for l in &mut lines[..n.saturating_sub(1)] {
        l.push_str(" \\");
    }
    lines
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
    fn wrapping_keeps_flags_with_values_and_reads_back_the_same() {
        let argv = "llama-server --hf-repo bartowski/Llama-3.3-70B-Instruct-GGUF --hf-file Llama-3.3-70B-Instruct-Q4_K_M.gguf -c 32768 -ngl 81 --device CUDA0,CUDA1 --split-mode layer --tensor-split 41,40 -fa on -ctk q8_0 -ctv q8_0 -np 1 --jinja"
            .split(' ')
            .map(String::from)
            .collect::<Vec<_>>();
        let lines = shell_wrap(&argv, 60);
        assert!(lines.len() > 2, "{lines:?}");
        assert!(lines.iter().all(|l| l.len() <= 60), "{lines:?}");
        assert!(
            lines.iter().any(|l| l.starts_with("  --hf-file Llama")),
            "a flag keeps its value: {lines:?}"
        );
        assert!(!lines.last().unwrap().ends_with('\\'));
        let back = lines
            .iter()
            .map(|l| l.trim_start().trim_end_matches(" \\"))
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(back, shell_join(&argv));
        assert_eq!(shell_wrap(&argv[..1], 60), ["llama-server"]);
    }

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
