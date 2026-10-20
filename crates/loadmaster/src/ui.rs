//! Terminal output. Styled text goes through `anstream`, which drops the styles when
//! stdout is not a colour terminal, under `NO_COLOR` or `CLICOLOR=0`, and with
//! `--color never`; the launch command is wrapped only when stdout is a terminal.

use std::io::{IsTerminal, Write};

pub use loadmaster_plan::explain::Style;

/// The style for stdout.
pub fn style() -> Style {
    Style {
        color: true,
        wrap: std::io::stdout().is_terminal(),
    }
}

/// Print to stdout, keeping styles only where they display.
pub fn print(s: &str) {
    let mut out = anstream::stdout();
    // A closed pipe (`lm plan … | head`) is not an error worth reporting.
    let _ = out.write_all(s.as_bytes());
}
