//! Phase 2: hand-rolled recursive descent parser.
//!
//! Grammar sketch (indentation-sensitive):
//!
//! ```text
//! program    := (extern_decl | function)*
//! extern_decl:= "extern" "fn" IDENT "(" [param_tys] ["," "..."] ")" ["->" type] NEWLINE
//! function   := "fn" IDENT "(" [params] ")" ["->" type] ":" block
//! block      := NEWLINE INDENT statement+ DEDENT
//! statement  := let | assign | if | while | return | break | continue | expr NEWLINE
//! expr       := or_expr
//! or         := and ("or" and)*
//! and        := not ("and" not)*
//! not        := "not" not | comparison
//! comparison := additive (("=="|"!="|"<"|"<="|">"|">=") additive)?
//! additive   := term (("+"|"-") term)*
//! term       := unary (("*"|"/"|"%") unary)*
//! unary      := "-" unary | primary
//! primary    := INT | FLOAT | STR | "true" | "false" | IDENT ["(" args ")"] | "(" expr ")"
//! ```

use crate::ast::*;
use crate::lexer::{TokKind, Token};
use std::fmt;

#[derive(Debug, Clone)]
pub struct ParseError {
    pub line: usize,
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "parse error (line {}): {}", self.line, self.message)
    }
}

impl std::error::Error for ParseError {}

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

type PResult<T> = Result<T, ParseError>;

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, pos: 0 }
    }

    fn peek(&self) -> &TokKind {
        &self.tokens[self.pos.min(self.tokens.len() - 1)].kind
    }

    fn line(&self) -> usize {
        self.tokens[self.pos.min(self.tokens.len() - 1)].line
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
        ParseError { line: self.line(), message }
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
                other => {
                    return Err(self.error(format!(
                        "expected `fn` or `extern` at top level, found {other}"
                    )));
                }
            }
        }
        Ok(program)
    }

    fn parse_type(&mut self) -> PResult<Type> {
        let name = self.expect_ident()?;
        match name.as_str() {
            "int" => Ok(Type::Int),
            "float" => Ok(Type::Float),
            "bool" => Ok(Type::Bool),
            "str" => Ok(Type::Str),
            other => Err(self.error(format!("unknown type `{other}`"))),
        }
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
            TokKind::While => {
                self.advance();
                let cond = self.parse_expression()?;
                self.expect(TokKind::Colon)?;
                let body = self.parse_block()?;
                Ok(Stmt::While { cond, body })
            }
            // Assignment or bare expression — disambiguate by lookahead.
            TokKind::Ident(_) => {
                if let TokKind::Assign = self.tokens[self.pos + 1].kind {
                    let name = self.expect_ident()?;
                    self.advance(); // `=`
                    let value = self.parse_expression()?;
                    self.expect(TokKind::Newline)?;
                    Ok(Stmt::Assign { name, value, line })
                } else {
                    let expr = self.parse_expression()?;
                    self.expect(TokKind::Newline)?;
                    Ok(Stmt::Expr(expr))
                }
            }
            other => Err(self.error(format!("unexpected {other} at start of statement"))),
        }
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
            let line = self.line();
            self.advance();
            let rhs = self.parse_and()?;
            lhs = Expr::new(
                ExprKind::Binary(Box::new(lhs), BinOp::Or, Box::new(rhs)),
                line,
            );
        }
        Ok(lhs)
    }

    fn parse_and(&mut self) -> PResult<Expr> {
        let mut lhs = self.parse_not()?;
        while self.peek() == &TokKind::And {
            let line = self.line();
            self.advance();
            let rhs = self.parse_not()?;
            lhs = Expr::new(
                ExprKind::Binary(Box::new(lhs), BinOp::And, Box::new(rhs)),
                line,
            );
        }
        Ok(lhs)
    }

    fn parse_not(&mut self) -> PResult<Expr> {
        if self.peek() == &TokKind::Not {
            let line = self.line();
            self.advance();
            let operand = self.parse_not()?;
            Ok(Expr::new(ExprKind::Unary(UnOp::Not, Box::new(operand)), line))
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
        let line = self.line();
        self.advance();
        let rhs = self.parse_additive()?;
        Ok(Expr::new(
            ExprKind::Binary(Box::new(lhs), op, Box::new(rhs)),
            line,
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
            let line = self.line();
            self.advance();
            let rhs = self.parse_term()?;
            lhs = Expr::new(ExprKind::Binary(Box::new(lhs), op, Box::new(rhs)), line);
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
            let line = self.line();
            self.advance();
            let rhs = self.parse_unary()?;
            lhs = Expr::new(ExprKind::Binary(Box::new(lhs), op, Box::new(rhs)), line);
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> PResult<Expr> {
        if self.peek() == &TokKind::Minus {
            let line = self.line();
            self.advance();
            let operand = self.parse_unary()?;
            Ok(Expr::new(ExprKind::Unary(UnOp::Neg, Box::new(operand)), line))
        } else {
            self.parse_primary()
        }
    }

    fn parse_primary(&mut self) -> PResult<Expr> {
        let line = self.line();
        match self.peek().clone() {
            TokKind::Int(n) => {
                self.advance();
                Ok(Expr::new(ExprKind::Int(n), line))
            }
            TokKind::Float(x) => {
                self.advance();
                Ok(Expr::new(ExprKind::Float(x), line))
            }
            TokKind::Str(s) => {
                self.advance();
                Ok(Expr::new(ExprKind::Str(s), line))
            }
            TokKind::True => {
                self.advance();
                Ok(Expr::new(ExprKind::Bool(true), line))
            }
            TokKind::False => {
                self.advance();
                Ok(Expr::new(ExprKind::Bool(false), line))
            }
            TokKind::LParen => {
                self.advance();
                let inner = self.parse_expression()?;
                self.expect(TokKind::RParen)?;
                Ok(inner)
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
                    Ok(Expr::new(ExprKind::Call(name, args), line))
                } else {
                    Ok(Expr::new(ExprKind::Var(name), line))
                }
            }
            other => Err(self.error(format!("unexpected {other} in expression"))),
        }
    }
}

/// Convenience: lex + parse a source string.
pub fn parse(source: &str) -> Result<Program, Box<dyn std::error::Error>> {
    let tokens = crate::lexer::lex(source)?;
    let mut parser = Parser::new(tokens);
    Ok(parser.parse_program()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;

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
