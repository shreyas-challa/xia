//! Phase 1: Lexical analysis.
//!
//! A `logos`-generated lexer handles the within-line tokens; a wrapper layer
//! tracks an indentation stack (`Vec<usize>`) and emits `Indent` / `Dedent`
//! tokens Python-style. Newlines inside parentheses or brackets are joined
//! implicitly.

use logos::Logos;
use std::fmt;

fn unescape(s: &str) -> String {
    // s includes the surrounding quotes
    let inner = &s[1..s.len() - 1];
    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('0') => out.push('\0'),
                Some('\\') => out.push('\\'),
                Some('"') => out.push('"'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t]+")]
#[logos(skip r"#[^\n]*")]
pub enum TokKind {
    // Keywords
    #[token("fn")]
    Fn,
    #[token("extern")]
    Extern,
    #[token("let")]
    Let,
    #[token("return")]
    Return,
    #[token("if")]
    If,
    #[token("elif")]
    Elif,
    #[token("else")]
    Else,
    #[token("while")]
    While,
    #[token("for")]
    For,
    #[token("in")]
    In,
    #[token("break")]
    Break,
    #[token("continue")]
    Continue,
    #[token("true")]
    True,
    #[token("false")]
    False,
    #[token("and")]
    And,
    #[token("or")]
    Or,
    #[token("not")]
    Not,

    // Literals
    #[regex(r"[0-9]+\.[0-9]+", |lex| lex.slice().parse::<f64>().ok())]
    Float(f64),
    #[regex(r"[0-9]+", |lex| lex.slice().parse::<i64>().ok())]
    Int(i64),
    #[regex(r#""([^"\\\n]|\\.)*""#, |lex| unescape(lex.slice()))]
    Str(String),
    #[regex(r"[A-Za-z_][A-Za-z0-9_]*", |lex| lex.slice().to_string())]
    Ident(String),

    // Operators and punctuation
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("==")]
    EqEq,
    #[token("!=")]
    NotEq,
    #[token("<=")]
    Le,
    #[token(">=")]
    Ge,
    #[token("<")]
    Lt,
    #[token(">")]
    Gt,
    #[token("=")]
    Assign,
    #[token("->")]
    Arrow,
    #[token(":")]
    Colon,
    #[token(",")]
    Comma,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token("...")]
    Ellipsis,

    // Structural tokens emitted by the indentation layer, never by logos.
    Newline,
    Indent,
    Dedent,
    Eof,
}

impl fmt::Display for TokKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokKind::Ident(s) => write!(f, "identifier `{s}`"),
            TokKind::Int(n) => write!(f, "integer `{n}`"),
            TokKind::Float(x) => write!(f, "float `{x}`"),
            TokKind::Str(_) => write!(f, "string literal"),
            TokKind::Newline => write!(f, "newline"),
            TokKind::Indent => write!(f, "indent"),
            TokKind::Dedent => write!(f, "dedent"),
            TokKind::Eof => write!(f, "end of file"),
            other => write!(f, "`{other:?}`"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokKind,
    pub line: usize,
}

#[derive(Debug, Clone)]
pub struct LexError {
    pub line: usize,
    pub message: String,
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "lex error (line {}): {}", self.line, self.message)
    }
}

impl std::error::Error for LexError {}

/// Tokenize a whole source file, producing a flat token stream that includes
/// `Newline`, `Indent`, `Dedent` and a final `Eof`.
pub fn lex(source: &str) -> Result<Vec<Token>, LexError> {
    let mut tokens = Vec::new();
    // The indentation stack. Always starts at column 0.
    let mut indents: Vec<usize> = vec![0];
    // Depth of open ( or [ — newlines inside are joined implicitly.
    let mut paren_depth: usize = 0;

    for (idx, raw_line) in source.lines().enumerate() {
        let line_no = idx + 1;

        // Strip trailing comment-only / blank lines early (outside brackets).
        let trimmed = raw_line.trim_start_matches(' ');
        if paren_depth == 0 {
            if trimmed.starts_with('\t') {
                return Err(LexError {
                    line: line_no,
                    message: "tabs are not allowed in indentation; use spaces".into(),
                });
            }
            let is_blank = trimmed.is_empty() || trimmed.starts_with('#');
            if is_blank {
                continue;
            }

            let indent = raw_line.len() - trimmed.len();
            let current = *indents.last().unwrap();
            if indent > current {
                indents.push(indent);
                tokens.push(Token { kind: TokKind::Indent, line: line_no });
            } else if indent < current {
                while *indents.last().unwrap() > indent {
                    indents.pop();
                    tokens.push(Token { kind: TokKind::Dedent, line: line_no });
                }
                if *indents.last().unwrap() != indent {
                    return Err(LexError {
                        line: line_no,
                        message: format!(
                            "unindent to column {indent} does not match any outer indentation level"
                        ),
                    });
                }
            }
        }

        let mut lexer = TokKind::lexer(raw_line);
        let mut emitted_any = false;
        while let Some(result) = lexer.next() {
            match result {
                Ok(kind) => {
                    match kind {
                        TokKind::LParen | TokKind::LBracket => paren_depth += 1,
                        TokKind::RParen | TokKind::RBracket => {
                            paren_depth = paren_depth.saturating_sub(1)
                        }
                        _ => {}
                    }
                    tokens.push(Token { kind, line: line_no });
                    emitted_any = true;
                }
                Err(()) => {
                    return Err(LexError {
                        line: line_no,
                        message: format!("unexpected character `{}`", lexer.slice()),
                    });
                }
            }
        }

        if paren_depth == 0 && emitted_any {
            tokens.push(Token { kind: TokKind::Newline, line: line_no });
        }
    }

    let last_line = source.lines().count().max(1);
    while indents.len() > 1 {
        indents.pop();
        tokens.push(Token { kind: TokKind::Dedent, line: last_line });
    }
    tokens.push(Token { kind: TokKind::Eof, line: last_line });
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<TokKind> {
        lex(src).unwrap().into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn simple_tokens() {
        use TokKind::*;
        assert_eq!(
            kinds("let x = 1 + 2.5"),
            vec![
                Let,
                Ident("x".into()),
                Assign,
                Int(1),
                Plus,
                Float(2.5),
                Newline,
                Eof
            ]
        );
    }

    #[test]
    fn indent_dedent() {
        use TokKind::*;
        let src = "if x:\n    y = 1\nz = 2\n";
        assert_eq!(
            kinds(src),
            vec![
                If,
                Ident("x".into()),
                Colon,
                Newline,
                Indent,
                Ident("y".into()),
                Assign,
                Int(1),
                Newline,
                Dedent,
                Ident("z".into()),
                Assign,
                Int(2),
                Newline,
                Eof
            ]
        );
    }

    #[test]
    fn nested_dedents_at_eof() {
        let src = "if a:\n    if b:\n        x = 1\n";
        let toks = kinds(src);
        let dedents = toks.iter().filter(|t| **t == TokKind::Dedent).count();
        assert_eq!(dedents, 2);
        assert_eq!(*toks.last().unwrap(), TokKind::Eof);
    }

    #[test]
    fn implicit_line_joining_in_parens() {
        let src = "f(1,\n  2)\n";
        let toks = kinds(src);
        let newlines = toks.iter().filter(|t| **t == TokKind::Newline).count();
        assert_eq!(newlines, 1, "newline inside parens must be joined");
    }

    #[test]
    fn comments_and_blank_lines_ignored() {
        use TokKind::*;
        let src = "# header\n\nx = 1  # trailing\n";
        assert_eq!(
            kinds(src),
            vec![Ident("x".into()), Assign, Int(1), Newline, Eof]
        );
    }

    #[test]
    fn string_escapes() {
        let toks = kinds(r#"s = "a\n\"b\"""#);
        assert!(matches!(&toks[2], TokKind::Str(s) if s == "a\n\"b\""));
    }

    #[test]
    fn for_in_keywords() {
        use TokKind::*;
        assert_eq!(
            kinds("for i in range(3)"),
            vec![
                For,
                Ident("i".into()),
                In,
                Ident("range".into()),
                LParen,
                Int(3),
                RParen,
                Newline,
                Eof
            ]
        );
    }

    #[test]
    fn bad_unindent_is_error() {
        let src = "if a:\n        x = 1\n    y = 2\n";
        assert!(lex(src).is_err());
    }
}
