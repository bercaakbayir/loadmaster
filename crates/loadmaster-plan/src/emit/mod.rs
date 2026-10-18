//! Runtime flag emitters. Each is a pure function of a [`Plan`](crate::Plan).

pub mod llama_cpp;

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
