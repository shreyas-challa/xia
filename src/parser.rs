//! Phase 2: hand-rolled recursive descent parser.
//!
//! Grammar sketch (indentation-sensitive):
//!
//! ```text
//! program    := (extern_decl | function)*
//! extern_decl:= "extern" "fn" IDENT "(" [param_tys] ["," "..."] ")" ["->" type] NEWLINE
//! function   := "fn" IDENT "(" [params] ")" ["->" type] ":" block
//! block      := NEWLINE INDENT statement+ DEDENT
//! statement  := let | assign | if | while | for | return | break | continue | expr NEWLINE
//! for        := "for" IDENT "in" ("range" "(" expr ["," expr] ")" | expr) ":" block
//! expr       := or_expr
//! or         := and ("or" and)*
//! and        := not ("and" not)*
//! not        := "not" not | comparison
//! comparison := additive (("=="|"!="|"<"|"<="|">"|">=") additive)?
//! additive   := term (("+"|"-") term)*
//! term       := unary (("*"|"/"|"%") unary)*
//! unary      := "-" unary | primary
//! primary    := atom ("[" expr "]")*
//! atom       := INT | FLOAT | STR | "true" | "false" | IDENT ["(" args ")"]
//!             | "(" expr ")" | "[" [expr ("," expr)*] "]"
//! type       := "int" | "float" | "bool" | "str" | "[" type "]"
//! ```

use crate::ast::*;
use crate::diag::Span;
use crate::lexer::{TokKind, Token};
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone)]
pub struct ParseError {
    pub span: Span,
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "parse error (line {}): {}", self.span.line, self.message)
    }
}

impl std::error::Error for ParseError {}

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    /// Struct name -> interned id, collected by a pre-scan so a struct can
    /// be referenced before its declaration.
    struct_ids: HashMap<String, u32>,
}

type PResult<T> = Result<T, ParseError>;

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        let mut struct_ids = HashMap::new();
        for pair in tokens.windows(2) {
            if pair[0].kind == TokKind::Struct {
                if let TokKind::Ident(name) = &pair[1].kind {
                    if !struct_ids.contains_key(name) {
                        struct_ids.insert(name.clone(), struct_ids.len() as u32);
                    }
                }
            }
        }
        Parser { tokens, pos: 0, struct_ids }
    }

    fn peek(&self) -> &TokKind {
        &self.tokens[self.pos.min(self.tokens.len() - 1)].kind
    }

    fn span(&self) -> Span {
        self.tokens[self.pos.min(self.tokens.len() - 1)].span
    }

    fn line(&self) -> usize {
        self.span().line
    }

    fn advance(&mut self) -> TokKind {
        let tok = self.tokens[self.pos.min(self.tokens.len() - 1)].kind.clone();
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
        tok
    }

    fn check(&mut self, kind: &TokKind) -> bool {
        if self.peek() == kind {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, kind: TokKind) -> PResult<()> {
        if self.peek() == &kind {
            self.advance();
            Ok(())
        } else {
            Err(self.error(format!("expected {kind}, found {}", self.peek())))
        }
    }

    fn error(&self, message: String) -> ParseError {
        ParseError { span: self.span(), message }
    }

    fn expect_ident(&mut self) -> PResult<String> {
        match self.peek().clone() {
            TokKind::Ident(name) => {
                self.advance();
                Ok(name)
            }
            other => Err(self.error(format!("expected identifier, found {other}"))),
        }
    }

    // ----- top level ---------------------------------------------------

    pub fn parse_program(&mut self) -> PResult<Program> {
        let mut program = Program::default();
        loop {
            match self.peek() {
                TokKind::Eof => break,
                TokKind::Extern => program.externs.push(self.parse_extern()?),
                TokKind::Fn => program.functions.push(self.parse_function()?),
                TokKind::Struct => {
                    let def = self.parse_struct()?;
                    if program.structs.iter().any(|s| s.name == def.name) {
                        return Err(self.error(format!("duplicate struct `{}`", def.name)));
                    }
                    debug_assert_eq!(
                        self.struct_ids[&def.name] as usize,
                        program.structs.len(),
                        "struct ids must match declaration order"
                    );
                    program.structs.push(def);
                }
                other => {
                    return Err(self.error(format!(
                        "expected `fn`, `struct` or `extern` at top level, found {other}"
                    )));
                }
            }
        }
        Ok(program)
    }

    fn parse_type(&mut self) -> PResult<Type> {
        if self.check(&TokKind::LBracket) {
            let elem = self.parse_type()?;
            self.expect(TokKind::RBracket)?;
            let Some(elem) = ElemType::from_type(elem) else {
                return Err(self.error(format!(
                    "`[{elem}]` is not a valid array type (nested arrays are not supported yet)"
                )));
            };
            return Ok(Type::Array(elem));
        }
        let name = self.expect_ident()?;
        match name.as_str() {
            "int" => Ok(Type::Int),
            "float" => Ok(Type::Float),
            "bool" => Ok(Type::Bool),
            "str" => Ok(Type::Str),
            other => match self.struct_ids.get(other) {
                Some(id) => Ok(Type::Struct(*id)),
                None => Err(self.error(format!("unknown type `{other}`"))),
            },
        }
    }

    /// `struct Name:` NEWLINE INDENT (`field: type` NEWLINE)+ DEDENT
    fn parse_struct(&mut self) -> PResult<StructDef> {
        let line = self.line();
        self.expect(TokKind::Struct)?;
        let name = self.expect_ident()?;
        self.expect(TokKind::Colon)?;
        self.expect(TokKind::Newline)?;
        self.expect(TokKind::Indent)?;
        let mut fields: Vec<Param> = Vec::new();
        while !self.check(&TokKind::Dedent) {
            let fname = self.expect_ident()?;
            if fields.iter().any(|f| f.name == fname) {
                return Err(self.error(format!("duplicate field `{fname}` in `{name}`")));
            }
            self.expect(TokKind::Colon)?;
            let ty = self.parse_type()?;
            self.expect(TokKind::Newline)?;
            fields.push(Param { name: fname, ty });
        }
        if fields.is_empty() {
            return Err(self.error(format!("struct `{name}` must have at least one field")));
        }
        Ok(StructDef { name, fields, line })
    }

    fn parse_return_type(&mut self) -> PResult<Type> {
        if self.check(&TokKind::Arrow) {
            self.parse_type()
        } else {
            Ok(Type::Unit)
        }
    }

    /// `extern fn printf(str, ...) -> int`
    /// Parameter names are allowed but optional: `extern fn puts(s: str) -> int`.
    fn parse_extern(&mut self) -> PResult<ExternFn> {
        let line = self.line();
        self.expect(TokKind::Extern)?;
        self.expect(TokKind::Fn)?;
        let name = self.expect_ident()?;
        self.expect(TokKind::LParen)?;
        let mut params = Vec::new();
        let mut varargs = false;
        while !self.check(&TokKind::RParen) {
            if self.check(&TokKind::Ellipsis) {
                varargs = true;
                self.expect(TokKind::RParen)?;
                break;
            }
            // Either `name: type` or a bare `type`.
            let first = self.expect_ident()?;
            let ty = if self.check(&TokKind::Colon) {
                self.parse_type()?
            } else {
                match first.as_str() {
                    "int" => Type::Int,
                    "float" => Type::Float,
                    "bool" => Type::Bool,
                    "str" => Type::Str,
                    other => {
                        return Err(self.error(format!("unknown type `{other}`")));
                    }
                }
            };
            params.push(ty);
            if !self.check(&TokKind::Comma) {
                self.expect(TokKind::RParen)?;
                break;
            }
        }
        let ret = self.parse_return_type()?;
        self.expect(TokKind::Newline)?;
        Ok(ExternFn { name, params, varargs, ret, line })
    }

    fn parse_function(&mut self) -> PResult<Function> {
        let line = self.line();
        self.expect(TokKind::Fn)?;
        let name = self.expect_ident()?;
        self.expect(TokKind::LParen)?;
        let mut params = Vec::new();
        while !self.check(&TokKind::RParen) {
            let pname = self.expect_ident()?;
            self.expect(TokKind::Colon)?;
            let ty = self.parse_type()?;
            params.push(Param { name: pname, ty });
            if !self.check(&TokKind::Comma) {
                self.expect(TokKind::RParen)?;
                break;
            }
        }
        let ret = self.parse_return_type()?;
        self.expect(TokKind::Colon)?;
        let body = self.parse_block()?;
        Ok(Function { name, params, ret, body, line })
    }

    // ----- statements ---------------------------------------------------

    /// A block is NEWLINE INDENT stmt+ DEDENT.
    fn parse_block(&mut self) -> PResult<Block> {
        self.expect(TokKind::Newline)?;
        self.expect(TokKind::Indent)?;
        let mut stmts = Vec::new();
        while !self.check(&TokKind::Dedent) {
            if self.peek() == &TokKind::Eof {
                return Err(self.error("unexpected end of file inside block".into()));
            }
            stmts.push(self.parse_statement()?);
        }
        if stmts.is_empty() {
            return Err(self.error("block cannot be empty".into()));
        }
        Ok(stmts)
    }

    fn parse_statement(&mut self) -> PResult<Stmt> {
        let line = self.line();
        match self.peek() {
            TokKind::Let => {
                self.advance();
                let name = self.expect_ident()?;
                let ty = if self.check(&TokKind::Colon) {
                    Some(self.parse_type()?)
                } else {
                    None
                };
                self.expect(TokKind::Assign)?;
                let value = self.parse_expression()?;
                self.expect(TokKind::Newline)?;
                Ok(Stmt::Let { name, ty, value, line })
            }
            TokKind::Return => {
                self.advance();
                let value = if self.peek() == &TokKind::Newline {
                    None
                } else {
                    Some(self.parse_expression()?)
                };
                self.expect(TokKind::Newline)?;
                Ok(Stmt::Return { value, line })
            }
            TokKind::Break => {
                self.advance();
                self.expect(TokKind::Newline)?;
                Ok(Stmt::Break { line })
            }
            TokKind::Continue => {
                self.advance();
                self.expect(TokKind::Newline)?;
                Ok(Stmt::Continue { line })
            }
            TokKind::If => self.parse_if(),
            TokKind::For => self.parse_for(),
            TokKind::While => {
                self.advance();
                let cond = self.parse_expression()?;
                self.expect(TokKind::Colon)?;
                let body = self.parse_block()?;
                Ok(Stmt::While { cond, body })
            }
            // Assignment or bare expression: parse the expression first, then
            // a trailing `=` decides — the target must be a name or an index.
            TokKind::Ident(_) => {
                let expr = self.parse_expression()?;
                if self.check(&TokKind::Assign) {
                    let value = self.parse_expression()?;
                    self.expect(TokKind::Newline)?;
                    match expr.kind {
                        ExprKind::Var(name) => Ok(Stmt::Assign { name, value, line }),
                        ExprKind::Index(target, index) => Ok(Stmt::IndexAssign {
                            target: *target,
                            index: *index,
                            value,
                            line,
                        }),
                        ExprKind::Field(target, field) => Ok(Stmt::FieldAssign {
                            target: *target,
                            field,
                            value,
                            line,
                        }),
                        _ => Err(ParseError {
                            span: expr.span,
                            message: "invalid assignment target".into(),
                        }),
                    }
                } else {
                    self.expect(TokKind::Newline)?;
                    Ok(Stmt::Expr(expr))
                }
            }
            other => Err(self.error(format!("unexpected {other} at start of statement"))),
        }
    }

    /// `for i in range(end):`, `for i in range(start, end):`, or
    /// `for x in <array expr>:`. `range` is loop syntax, not a function —
    /// it never escapes a `for`.
    fn parse_for(&mut self) -> PResult<Stmt> {
        let span = self.span();
        let line = span.line;
        self.expect(TokKind::For)?;
        let var = self.expect_ident()?;
        self.expect(TokKind::In)?;
        let iterable = self.parse_expression()?;
        self.expect(TokKind::Colon)?;
        let body = self.parse_block()?;
        if let ExprKind::Call(name, _) = &iterable.kind {
            if name == "range" {
                let range_span = iterable.span;
                let ExprKind::Call(_, mut args) = iterable.kind else {
                    unreachable!();
                };
                let (start, end) = match args.len() {
                    1 => (Expr::new(ExprKind::Int(0), range_span), args.pop().unwrap()),
                    2 => {
                        let end = args.pop().unwrap();
                        (args.pop().unwrap(), end)
                    }
                    n => {
                        return Err(ParseError {
                            span: range_span,
                            message: format!("range takes 1 or 2 arguments, got {n}"),
                        });
                    }
                };
                return Ok(Stmt::For { var, start, end, body, line });
            }
        }
        Ok(Stmt::ForEach { var, iterable, body, line })
    }

    /// `elif` chains desugar to nested if/else.
    fn parse_if(&mut self) -> PResult<Stmt> {
        self.expect(TokKind::If)?;
        let cond = self.parse_expression()?;
        self.expect(TokKind::Colon)?;
        let then_block = self.parse_block()?;
        let else_block = match self.peek() {
            TokKind::Elif => Some(vec![self.parse_if_from_elif()?]),
            TokKind::Else => {
                self.advance();
                self.expect(TokKind::Colon)?;
                Some(self.parse_block()?)
            }
            _ => None,
        };
        Ok(Stmt::If { cond, then_block, else_block })
    }

    fn parse_if_from_elif(&mut self) -> PResult<Stmt> {
        self.expect(TokKind::Elif)?;
        let cond = self.parse_expression()?;
        self.expect(TokKind::Colon)?;
        let then_block = self.parse_block()?;
        let else_block = match self.peek() {
            TokKind::Elif => Some(vec![self.parse_if_from_elif()?]),
            TokKind::Else => {
                self.advance();
                self.expect(TokKind::Colon)?;
                Some(self.parse_block()?)
            }
            _ => None,
        };
        Ok(Stmt::If { cond, then_block, else_block })
    }

    // ----- expressions ----------------------------------------------------

    pub fn parse_expression(&mut self) -> PResult<Expr> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> PResult<Expr> {
        let mut lhs = self.parse_and()?;
        while self.peek() == &TokKind::Or {
            let span = self.span();
            self.advance();
            let rhs = self.parse_and()?;
            lhs = Expr::new(
                ExprKind::Binary(Box::new(lhs), BinOp::Or, Box::new(rhs)),
                span,
            );
        }
        Ok(lhs)
    }

    fn parse_and(&mut self) -> PResult<Expr> {
        let mut lhs = self.parse_not()?;
        while self.peek() == &TokKind::And {
            let span = self.span();
            self.advance();
            let rhs = self.parse_not()?;
            lhs = Expr::new(
                ExprKind::Binary(Box::new(lhs), BinOp::And, Box::new(rhs)),
                span,
            );
        }
        Ok(lhs)
    }

    fn parse_not(&mut self) -> PResult<Expr> {
        if self.peek() == &TokKind::Not {
            let span = self.span();
            self.advance();
            let operand = self.parse_not()?;
            Ok(Expr::new(ExprKind::Unary(UnOp::Not, Box::new(operand)), span))
        } else {
            self.parse_comparison()
        }
    }

    fn parse_comparison(&mut self) -> PResult<Expr> {
        let lhs = self.parse_additive()?;
        let op = match self.peek() {
            TokKind::EqEq => BinOp::Eq,
            TokKind::NotEq => BinOp::Ne,
            TokKind::Lt => BinOp::Lt,
            TokKind::Le => BinOp::Le,
            TokKind::Gt => BinOp::Gt,
            TokKind::Ge => BinOp::Ge,
            _ => return Ok(lhs),
        };
        let span = self.span();
        self.advance();
        let rhs = self.parse_additive()?;
        Ok(Expr::new(
            ExprKind::Binary(Box::new(lhs), op, Box::new(rhs)),
            span,
        ))
    }

    fn parse_additive(&mut self) -> PResult<Expr> {
        let mut lhs = self.parse_term()?;
        loop {
            let op = match self.peek() {
                TokKind::Plus => BinOp::Add,
                TokKind::Minus => BinOp::Sub,
                _ => break,
            };
            let span = self.span();
            self.advance();
            let rhs = self.parse_term()?;
            lhs = Expr::new(ExprKind::Binary(Box::new(lhs), op, Box::new(rhs)), span);
        }
        Ok(lhs)
    }

    fn parse_term(&mut self) -> PResult<Expr> {
        let mut lhs = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                TokKind::Star => BinOp::Mul,
                TokKind::Slash => BinOp::Div,
                TokKind::Percent => BinOp::Rem,
                _ => break,
            };
            let span = self.span();
            self.advance();
            let rhs = self.parse_unary()?;
            lhs = Expr::new(ExprKind::Binary(Box::new(lhs), op, Box::new(rhs)), span);
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> PResult<Expr> {
        if self.peek() == &TokKind::Minus {
            let span = self.span();
            self.advance();
            let operand = self.parse_unary()?;
            Ok(Expr::new(ExprKind::Unary(UnOp::Neg, Box::new(operand)), span))
        } else {
            self.parse_primary()
        }
    }

    /// An atom followed by any number of `[index]` / `.field` suffixes.
    fn parse_primary(&mut self) -> PResult<Expr> {
        let mut expr = self.parse_atom()?;
        loop {
            let span = self.span();
            match self.peek() {
                TokKind::LBracket => {
                    self.advance();
                    let idx = self.parse_expression()?;
                    self.expect(TokKind::RBracket)?;
                    expr = Expr::new(ExprKind::Index(Box::new(expr), Box::new(idx)), span);
                }
                TokKind::Dot => {
                    self.advance();
                    let field = self.expect_ident()?;
                    expr = Expr::new(ExprKind::Field(Box::new(expr), field), span);
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_atom(&mut self) -> PResult<Expr> {
        let span = self.span();
        match self.peek().clone() {
            TokKind::Int(n) => {
                self.advance();
                Ok(Expr::new(ExprKind::Int(n), span))
            }
            TokKind::Float(x) => {
                self.advance();
                Ok(Expr::new(ExprKind::Float(x), span))
            }
            TokKind::Str(s) => {
                self.advance();
                Ok(Expr::new(ExprKind::Str(s), span))
            }
            TokKind::FStr(s) => {
                self.advance();
                desugar_fstring(&s, span)
            }
            TokKind::True => {
                self.advance();
                Ok(Expr::new(ExprKind::Bool(true), span))
            }
            TokKind::False => {
                self.advance();
                Ok(Expr::new(ExprKind::Bool(false), span))
            }
            TokKind::LParen => {
                self.advance();
                let inner = self.parse_expression()?;
                self.expect(TokKind::RParen)?;
                Ok(inner)
            }
            TokKind::LBracket => {
                self.advance();
                let mut elems = Vec::new();
                while !self.check(&TokKind::RBracket) {
                    elems.push(self.parse_expression()?);
                    if !self.check(&TokKind::Comma) {
                        self.expect(TokKind::RBracket)?;
                        break;
                    }
                }
                Ok(Expr::new(ExprKind::ArrayLit(elems), span))
            }
            TokKind::Ident(name) => {
                self.advance();
                if self.check(&TokKind::LParen) {
                    let mut args = Vec::new();
                    while !self.check(&TokKind::RParen) {
                        args.push(self.parse_expression()?);
                        if !self.check(&TokKind::Comma) {
                            self.expect(TokKind::RParen)?;
                            break;
                        }
                    }
                    Ok(Expr::new(ExprKind::Call(name, args), span))
                } else {
                    Ok(Expr::new(ExprKind::Var(name), span))
                }
            }
            other => Err(self.error(format!("unexpected {other} in expression"))),
        }
    }
}

/// Desugar `f"a {x} b"` into `"a " + str(x) + " b"`. `{{` / `}}` escape
/// literal braces. Each `{...}` is re-lexed and parsed as an expression;
/// quotes inside must be escaped (`\"`) because the f-string body ends at
/// the first bare quote.
fn desugar_fstring(s: &str, span: Span) -> PResult<Expr> {
    let in_fstring = |message: String| ParseError {
        span,
        message: format!("in f-string: {message}"),
    };
    let mut parts: Vec<Expr> = Vec::new();
    let mut lit = String::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' if chars.peek() == Some(&'{') => {
                chars.next();
                lit.push('{');
            }
            '}' if chars.peek() == Some(&'}') => {
                chars.next();
                lit.push('}');
            }
            '{' => {
                let mut inner = String::new();
                loop {
                    match chars.next() {
                        Some('}') => break,
                        Some(ch) => inner.push(ch),
                        None => {
                            return Err(in_fstring("unterminated `{`".into()));
                        }
                    }
                }
                if inner.trim().is_empty() {
                    return Err(in_fstring("empty `{}` interpolation".into()));
                }
                if !lit.is_empty() {
                    parts.push(Expr::new(ExprKind::Str(std::mem::take(&mut lit)), span));
                }
                let tokens =
                    crate::lexer::lex(&inner).map_err(|e| in_fstring(e.message))?;
                let mut sub = Parser::new(tokens);
                let mut e = sub
                    .parse_expression()
                    .map_err(|e| in_fstring(e.message))?;
                if !matches!(sub.peek(), TokKind::Newline | TokKind::Eof) {
                    return Err(in_fstring(format!(
                        "unexpected {} after expression",
                        sub.peek()
                    )));
                }
                // The snippet was lexed standalone; point its spans at the
                // f-string token so later errors land on the right source.
                respan(&mut e, span);
                parts.push(Expr::new(ExprKind::Call("str".into(), vec![e]), span));
            }
            '}' => return Err(in_fstring("unmatched `}` (use `}}` for a literal)".into())),
            other => lit.push(other),
        }
    }
    if !lit.is_empty() || parts.is_empty() {
        parts.push(Expr::new(ExprKind::Str(lit), span));
    }
    let mut iter = parts.into_iter();
    let mut acc = iter.next().unwrap();
    for p in iter {
        acc = Expr::new(
            ExprKind::Binary(Box::new(acc), BinOp::Add, Box::new(p)),
            span,
        );
    }
    Ok(acc)
}

fn respan(e: &mut Expr, span: Span) {
    e.span = span;
    match &mut e.kind {
        ExprKind::Unary(_, a) => respan(a, span),
        ExprKind::Binary(a, _, b) => {
            respan(a, span);
            respan(b, span);
        }
        ExprKind::Call(_, args) | ExprKind::ArrayLit(args) => {
            for a in args {
                respan(a, span);
            }
        }
        ExprKind::Index(a, b) => {
            respan(a, span);
            respan(b, span);
        }
        ExprKind::Field(a, _) => respan(a, span),
        _ => {}
    }
}

/// Convenience for tests: lex + parse a source string.
#[cfg(test)]
pub fn parse(source: &str) -> Result<Program, Box<dyn std::error::Error>> {
    let tokens = crate::lexer::lex(source)?;
    let mut parser = Parser::new(tokens);
    Ok(parser.parse_program()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_function() {
        let src = "fn main() -> int:\n    return 0\n";
        let prog = parse(src).unwrap();
        assert_eq!(prog.functions.len(), 1);
        let f = &prog.functions[0];
        assert_eq!(f.name, "main");
        assert_eq!(f.ret, Type::Int);
        assert!(matches!(f.body[0], Stmt::Return { .. }));
    }

    #[test]
    fn parses_params_and_let() {
        let src = "fn add(a: int, b: int) -> int:\n    let c = a + b\n    return c\n";
        let prog = parse(src).unwrap();
        let f = &prog.functions[0];
        assert_eq!(f.params.len(), 2);
        assert!(matches!(&f.body[0], Stmt::Let { name, .. } if name == "c"));
    }

    #[test]
    fn precedence_mul_over_add() {
        let src = "fn f() -> int:\n    return 1 + 2 * 3\n";
        let prog = parse(src).unwrap();
        let Stmt::Return { value: Some(e), .. } = &prog.functions[0].body[0] else {
            panic!("expected return");
        };
        // Must parse as 1 + (2 * 3)
        let ExprKind::Binary(_, BinOp::Add, rhs) = &e.kind else {
            panic!("expected top-level add, got {:?}", e.kind);
        };
        assert!(matches!(&rhs.kind, ExprKind::Binary(_, BinOp::Mul, _)));
    }

    #[test]
    fn parses_if_elif_else() {
        let src = "fn f(x: int) -> int:\n    if x > 0:\n        return 1\n    elif x < 0:\n        return -1\n    else:\n        return 0\n";
        let prog = parse(src).unwrap();
        let Stmt::If { else_block: Some(else_b), .. } = &prog.functions[0].body[0] else {
            panic!("expected if with else");
        };
        // elif desugars to a nested If inside else_block
        assert!(matches!(&else_b[0], Stmt::If { .. }));
    }

    #[test]
    fn parses_while_with_nested_block() {
        let src = "fn f() :\n    while true:\n        if false:\n            break\n        continue\n";
        let prog = parse(src).unwrap();
        let Stmt::While { body, .. } = &prog.functions[0].body[0] else {
            panic!("expected while");
        };
        assert_eq!(body.len(), 2);
    }

    #[test]
    fn parses_extern_varargs() {
        let src = "extern fn printf(fmt: str, ...) -> int\nfn main() -> int:\n    printf(\"hi\\n\")\n    return 0\n";
        let prog = parse(src).unwrap();
        let e = &prog.externs[0];
        assert_eq!(e.name, "printf");
        assert!(e.varargs);
        assert_eq!(e.params, vec![Type::Str]);
        assert_eq!(e.ret, Type::Int);
    }

    #[test]
    fn parses_call_with_args() {
        let src = "fn main():\n    let x = add(1, mul(2, 3))\n";
        let prog = parse(src).unwrap();
        let Stmt::Let { value, .. } = &prog.functions[0].body[0] else {
            panic!("expected let");
        };
        let ExprKind::Call(name, args) = &value.kind else {
            panic!("expected call");
        };
        assert_eq!(name, "add");
        assert_eq!(args.len(), 2);
        assert!(matches!(&args[1].kind, ExprKind::Call(n, _) if n == "mul"));
    }

    #[test]
    fn parses_array_literal_type_and_indexing() {
        let src = "fn f():\n    let xs: [int] = [1, 2, 3]\n    let y = xs[0]\n";
        let prog = parse(src).unwrap();
        let Stmt::Let { ty, value, .. } = &prog.functions[0].body[0] else {
            panic!("expected let");
        };
        assert_eq!(*ty, Some(Type::Array(ElemType::Int)));
        let ExprKind::ArrayLit(elems) = &value.kind else {
            panic!("expected array literal");
        };
        assert_eq!(elems.len(), 3);
        let Stmt::Let { value, .. } = &prog.functions[0].body[1] else {
            panic!("expected let");
        };
        assert!(matches!(&value.kind, ExprKind::Index(..)));
    }

    #[test]
    fn parses_index_assignment() {
        let src = "fn f(xs: [str]):\n    xs[1] = \"two\"\n";
        let prog = parse(src).unwrap();
        assert_eq!(prog.functions[0].params[0].ty, Type::Array(ElemType::Str));
        assert!(matches!(
            &prog.functions[0].body[0],
            Stmt::IndexAssign { .. }
        ));
    }

    #[test]
    fn rejects_nested_array_type() {
        assert!(parse("fn f(xs: [[int]]):\n    return\n").is_err());
    }

    #[test]
    fn rejects_bad_assignment_target() {
        assert!(parse("fn f():\n    f() = 1\n").is_err());
    }

    #[test]
    fn fstring_desugars_to_concat_with_str_calls() {
        let src = "fn f(x: int):\n    let s = f\"a {x}!\"\n    print(s)\n";
        let prog = parse(src).unwrap();
        let Stmt::Let { value, .. } = &prog.functions[0].body[0] else {
            panic!("expected let");
        };
        // ("a " + str(x)) + "!"
        let ExprKind::Binary(lhs, BinOp::Add, rhs) = &value.kind else {
            panic!("expected concat, got {:?}", value.kind);
        };
        assert!(matches!(&rhs.kind, ExprKind::Str(s) if s == "!"));
        let ExprKind::Binary(a, BinOp::Add, b) = &lhs.kind else {
            panic!("expected nested concat");
        };
        assert!(matches!(&a.kind, ExprKind::Str(s) if s == "a "));
        assert!(matches!(&b.kind, ExprKind::Call(n, _) if n == "str"));
    }

    #[test]
    fn fstring_brace_escapes_and_errors() {
        let src = "fn f():\n    print(f\"{{x}}\")\n";
        let prog = parse(src).unwrap();
        let Stmt::Expr(e) = &prog.functions[0].body[0] else { panic!() };
        let ExprKind::Call(_, args) = &e.kind else { panic!() };
        assert!(matches!(&args[0].kind, ExprKind::Str(s) if s == "{x}"));

        assert!(parse("fn f():\n    print(f\"{}\")\n").is_err());
        assert!(parse("fn f():\n    print(f\"{x\")\n").is_err());
        assert!(parse("fn f():\n    print(f\"x}\")\n").is_err());
    }

    #[test]
    fn parses_struct_decl_and_field_access() {
        let src = "struct Point:\n    x: int\n    y: int\nfn f(p: Point) -> int:\n    p.x = 5\n    return p.x + p.y\n";
        let prog = parse(src).unwrap();
        assert_eq!(prog.structs.len(), 1);
        assert_eq!(prog.structs[0].name, "Point");
        assert_eq!(prog.structs[0].fields.len(), 2);
        assert_eq!(prog.functions[0].params[0].ty, Type::Struct(0));
        assert!(matches!(
            &prog.functions[0].body[0],
            Stmt::FieldAssign { field, .. } if field == "x"
        ));
    }

    #[test]
    fn struct_types_resolve_in_any_order() {
        let src = "fn f(p: Point) -> int:\n    return p.x\nstruct Point:\n    x: int\n";
        assert!(parse(src).is_ok());
    }

    #[test]
    fn struct_errors() {
        assert!(parse("struct P:\n    x: int\n    x: int\n").is_err(), "dup field");
        assert!(parse("struct P:\n    x: int\nstruct P:\n    y: int\n").is_err(), "dup struct");
        assert!(parse("fn f(p: Nope) -> int:\n    return 0\n").is_err(), "unknown type");
    }

    #[test]
    fn parses_for_range_two_args() {
        let src = "fn f():\n    for i in range(1, 10):\n        print(i)\n";
        let prog = parse(src).unwrap();
        let Stmt::For { var, start, end, body, .. } = &prog.functions[0].body[0] else {
            panic!("expected for");
        };
        assert_eq!(var, "i");
        assert!(matches!(start.kind, ExprKind::Int(1)));
        assert!(matches!(end.kind, ExprKind::Int(10)));
        assert_eq!(body.len(), 1);
    }

    #[test]
    fn for_range_one_arg_starts_at_zero() {
        let src = "fn f():\n    for i in range(5):\n        print(i)\n";
        let prog = parse(src).unwrap();
        let Stmt::For { start, end, .. } = &prog.functions[0].body[0] else {
            panic!("expected for");
        };
        assert!(matches!(start.kind, ExprKind::Int(0)));
        assert!(matches!(end.kind, ExprKind::Int(5)));
    }

    #[test]
    fn for_over_array_parses_as_foreach() {
        let src = "fn f(items: [str]):\n    for x in items:\n        print(x)\n";
        let prog = parse(src).unwrap();
        let Stmt::ForEach { var, iterable, .. } = &prog.functions[0].body[0] else {
            panic!("expected foreach");
        };
        assert_eq!(var, "x");
        assert!(matches!(&iterable.kind, ExprKind::Var(n) if n == "items"));
    }

    #[test]
    fn range_arity_checked() {
        assert!(parse("fn f():\n    for i in range(1, 2, 3):\n        print(i)\n").is_err());
    }

    #[test]
    fn error_on_missing_colon() {
        let src = "fn main()\n    return 0\n";
        assert!(parse(src).is_err());
    }

    #[test]
    fn error_on_statement_at_top_level() {
        let src = "let x = 1\n";
        assert!(parse(src).is_err());
    }
}
