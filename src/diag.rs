//! Shared diagnostics: source spans and caret-style error rendering.
//!
//! Every front-end error (lex, parse, sema) carries a [`Span`]; the CLI
//! renders it against the source text:
//!
//! ```text
//! return type mismatch: function returns int but got str
//!   --> examples/bad.xia:2:12
//!    |
//!  2 |     return "oops"
//!    |            ^^^^^^
//! ```

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Span {
    /// 1-based source line.
    pub line: usize,
    /// 1-based column; 0 means "line precision only" (no caret is drawn).
    pub col: usize,
    /// Width of the underlined region in bytes (clamped to at least 1).
    pub len: usize,
}

impl Span {
    pub fn new(line: usize, col: usize, len: usize) -> Self {
        Span { line, col, len }
    }

    pub fn line_only(line: usize) -> Self {
        Span { line, col: 0, len: 0 }
    }
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub phase: &'static str,
    pub span: Span,
    pub message: String,
}

impl Diagnostic {
    /// Render the message followed by the source line and a caret underline.
    pub fn render(&self, path: &str, source: &str) -> String {
        let mut out = format!("{}\n  --> {}:{}", self.message, path, self.span.line);
        if self.span.col > 0 {
            out += &format!(":{}", self.span.col);
        }
        out.push('\n');
        if let Some(text) = source.lines().nth(self.span.line.saturating_sub(1)) {
            let n = self.span.line.to_string();
            let pad = " ".repeat(n.len());
            out += &format!("{pad} |\n{n} | {text}\n");
            if self.span.col > 0 && self.span.col <= text.len() + 1 {
                let carets = "^".repeat(self.span.len.max(1));
                out += &format!("{pad} | {}{carets}\n", " ".repeat(self.span.col - 1));
            }
        }
        out
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} error (line {}): {}",
            self.phase, self.span.line, self.message
        )
    }
}

impl From<crate::lexer::LexError> for Diagnostic {
    fn from(e: crate::lexer::LexError) -> Self {
        Diagnostic { phase: "lex", span: e.span, message: e.message }
    }
}

impl From<crate::parser::ParseError> for Diagnostic {
    fn from(e: crate::parser::ParseError) -> Self {
        Diagnostic { phase: "parse", span: e.span, message: e.message }
    }
}

impl From<crate::sema::SemaError> for Diagnostic {
    fn from(e: crate::sema::SemaError) -> Self {
        Diagnostic { phase: "semantic", span: e.span, message: e.message }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_caret_under_the_span() {
        let src = "fn main() -> int:\n    return \"oops\"\n";
        let d = Diagnostic {
            phase: "semantic",
            span: Span::new(2, 12, 6),
            message: "return type mismatch".into(),
        };
        let out = d.render("test.xia", src);
        assert!(out.contains("--> test.xia:2:12"));
        assert!(out.contains("2 |     return \"oops\""));
        assert!(out.contains("|            ^^^^^^"));
    }

    #[test]
    fn line_only_span_omits_caret() {
        let d = Diagnostic {
            phase: "parse",
            span: Span::line_only(1),
            message: "oops".into(),
        };
        let out = d.render("t.xia", "let x = 1\n");
        assert!(out.contains("--> t.xia:1\n"));
        assert!(!out.contains('^'));
    }
}
