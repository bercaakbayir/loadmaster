//! Terminal styling for the human-readable output, as plain ANSI escapes: the binary
//! prints through a stream that strips them when stdout is not a colour terminal.

/// How output is rendered: `color` adds ANSI styles, `wrap` breaks the launch command
/// over several lines (for a terminal; piped output keeps it on one line).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Style {
    pub color: bool,
    pub wrap: bool,
}

impl Style {
    /// No colour, one-line command: tests, logs and piped output.
    pub const PLAIN: Style = Style {
        color: false,
        wrap: false,
    };

    fn paint(self, code: &str, s: &str) -> String {
        if self.color {
            format!("\x1b[{code}m{s}\x1b[0m")
        } else {
            s.to_owned()
        }
    }

    pub fn bold(self, s: &str) -> String {
        self.paint("1", s)
    }

    pub fn dim(self, s: &str) -> String {
        self.paint("2", s)
    }

    pub fn green(self, s: &str) -> String {
        self.paint("1;32", s)
    }

    pub fn yellow(self, s: &str) -> String {
        self.paint("1;33", s)
    }

    pub fn red(self, s: &str) -> String {
        self.paint("1;31", s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_adds_nothing() {
        assert_eq!(Style::PLAIN.green("✓ Fits"), "✓ Fits");
        let c = Style {
            color: true,
            wrap: false,
        };
        assert_eq!(c.green("✓"), "\x1b[1;32m✓\x1b[0m");
    }
}
