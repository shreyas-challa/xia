//! Phase 3: semantic analysis.
//!
//! - Scoped symbol tables (a stack of `HashMap`s) track variables and types.
//! - Expression types are inferred bottom-up and annotated onto the AST.
//! - `extern fn` declarations are registered as foreign symbols; calls to
//!   them type-check like normal calls (varargs allowed).

use crate::ast::*;
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone)]
pub struct SemaError {
    pub line: usize,
    pub message: String,
}

impl fmt::Display for SemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "semantic error (line {}): {}", self.line, self.message)
    }
}

impl std::error::Error for SemaError {}

type SResult<T> = Result<T, SemaError>;

#[derive(Debug, Clone)]
pub struct FnSig {
    pub params: Vec<Type>,
    pub varargs: bool,
    pub ret: Type,
    pub is_extern: bool,
}

/// A stack of lexical scopes mapping variable name -> type.
struct SymbolTable {
    scopes: Vec<HashMap<String, Type>>,
}

impl SymbolTable {
    fn new() -> Self {
        SymbolTable { scopes: vec![HashMap::new()] }
    }

    fn push(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop(&mut self) {
        self.scopes.pop();
    }

    fn declare(&mut self, name: &str, ty: Type) -> bool {
        self.scopes
            .last_mut()
            .unwrap()
            .insert(name.to_string(), ty)
            .is_none()
    }

    fn lookup(&self, name: &str) -> Option<Type> {
        self.scopes.iter().rev().find_map(|s| s.get(name)).copied()
    }
}

pub struct Analyzer {
    pub functions: HashMap<String, FnSig>,
    symbols: SymbolTable,
    current_ret: Type,
    loop_depth: usize,
}

impl Analyzer {
    pub fn new() -> Self {
        Analyzer {
            functions: HashMap::new(),
            symbols: SymbolTable::new(),
            current_ret: Type::Unit,
            loop_depth: 0,
        }
    }

    /// Type-check the whole program, annotating expression types in place.
    pub fn analyze(&mut self, program: &mut Program) -> SResult<()> {
        // Register every signature first so call order doesn't matter.
        for e in &program.externs {
            let sig = FnSig {
                params: e.params.clone(),
                varargs: e.varargs,
                ret: e.ret,
                is_extern: true,
            };
            if self.functions.insert(e.name.clone(), sig).is_some() {
                return Err(SemaError {
                    line: e.line,
                    message: format!("duplicate declaration of `{}`", e.name),
                });
            }
        }
        for f in &program.functions {
            let sig = FnSig {
                params: f.params.iter().map(|p| p.ty).collect(),
                varargs: false,
                ret: f.ret,
                is_extern: false,
            };
            if self.functions.insert(f.name.clone(), sig).is_some() {
                return Err(SemaError {
                    line: f.line,
                    message: format!("duplicate definition of `{}`", f.name),
                });
            }
        }

        for f in &mut program.functions {
            self.check_function(f)?;
        }
        Ok(())
    }

    fn check_function(&mut self, f: &mut Function) -> SResult<()> {
        self.current_ret = f.ret;
        self.symbols = SymbolTable::new();
        for p in &f.params {
            if !self.symbols.declare(&p.name, p.ty) {
                return Err(SemaError {
                    line: f.line,
                    message: format!("duplicate parameter `{}` in `{}`", p.name, f.name),
                });
            }
        }
        self.check_block(&mut f.body)?;
        Ok(())
    }

    fn check_block(&mut self, block: &mut Block) -> SResult<()> {
        self.symbols.push();
        for stmt in block.iter_mut() {
            self.check_stmt(stmt)?;
        }
        self.symbols.pop();
        Ok(())
    }

    fn check_stmt(&mut self, stmt: &mut Stmt) -> SResult<()> {
        match stmt {
            Stmt::Let { name, ty, value, line } => {
                let inferred = self.check_expr(value)?;
                if inferred == Type::Unit {
                    return Err(SemaError {
                        line: *line,
                        message: format!("cannot bind `{name}` to a unit (no-value) expression"),
                    });
                }
                if let Some(annotated) = ty {
                    if *annotated != inferred {
                        return Err(SemaError {
                            line: *line,
                            message: format!(
                                "type mismatch: `{name}` declared as {annotated} but initialized with {inferred}"
                            ),
                        });
                    }
                }
                *ty = Some(inferred);
                if !self.symbols.declare(name, inferred) {
                    return Err(SemaError {
                        line: *line,
                        message: format!("`{name}` is already defined in this scope"),
                    });
                }
                Ok(())
            }
            Stmt::Assign { name, value, line } => {
                let Some(var_ty) = self.symbols.lookup(name) else {
                    return Err(SemaError {
                        line: *line,
                        message: format!("assignment to undefined variable `{name}`"),
                    });
                };
                let val_ty = self.check_expr(value)?;
                if val_ty != var_ty {
                    return Err(SemaError {
                        line: *line,
                        message: format!(
                            "type mismatch: cannot assign {val_ty} to `{name}` of type {var_ty}"
                        ),
                    });
                }
                Ok(())
            }
            Stmt::Expr(e) => {
                self.check_expr(e)?;
                Ok(())
            }
            Stmt::Return { value, line } => {
                let ty = match value {
                    Some(e) => self.check_expr(e)?,
                    None => Type::Unit,
                };
                if ty != self.current_ret {
                    return Err(SemaError {
                        line: *line,
                        message: format!(
                            "return type mismatch: function returns {} but got {ty}",
                            self.current_ret
                        ),
                    });
                }
                Ok(())
            }
            Stmt::If { cond, then_block, else_block } => {
                let cond_ty = self.check_expr(cond)?;
                if cond_ty != Type::Bool {
                    return Err(SemaError {
                        line: cond.line,
                        message: format!("if condition must be bool, found {cond_ty}"),
                    });
                }
                self.check_block(then_block)?;
                if let Some(else_b) = else_block {
                    self.check_block(else_b)?;
                }
                Ok(())
            }
            Stmt::While { cond, body } => {
                let cond_ty = self.check_expr(cond)?;
                if cond_ty != Type::Bool {
                    return Err(SemaError {
                        line: cond.line,
                        message: format!("while condition must be bool, found {cond_ty}"),
                    });
                }
                self.loop_depth += 1;
                self.check_block(body)?;
                self.loop_depth -= 1;
                Ok(())
            }
            Stmt::Break { line } | Stmt::Continue { line } => {
                if self.loop_depth == 0 {
                    return Err(SemaError {
                        line: *line,
                        message: "`break`/`continue` outside of a loop".into(),
                    });
                }
                Ok(())
            }
            Stmt::Retain(_) | Stmt::Release(_) => Ok(()),
        }
    }

    fn check_expr(&mut self, expr: &mut Expr) -> SResult<Type> {
        let ty = match &mut expr.kind {
            ExprKind::Int(_) => Type::Int,
            ExprKind::Float(_) => Type::Float,
            ExprKind::Bool(_) => Type::Bool,
            ExprKind::Str(_) => Type::Str,
            ExprKind::Var(name) => self.symbols.lookup(name).ok_or_else(|| SemaError {
                line: expr.line,
                message: format!("undefined variable `{name}`"),
            })?,
            ExprKind::Unary(op, operand) => {
                let line = expr.line;
                let t = self.check_expr(operand)?;
                match op {
                    UnOp::Neg if t == Type::Int || t == Type::Float => t,
                    UnOp::Neg => {
                        return Err(SemaError {
                            line,
                            message: format!("cannot negate a value of type {t}"),
                        });
                    }
                    UnOp::Not if t == Type::Bool => Type::Bool,
                    UnOp::Not => {
                        return Err(SemaError {
                            line,
                            message: format!("`not` requires bool, found {t}"),
                        });
                    }
                }
            }
            ExprKind::Binary(lhs, op, rhs) => {
                let line = expr.line;
                let lt = self.check_expr(lhs)?;
                let rt = self.check_expr(rhs)?;
                if lt != rt {
                    return Err(SemaError {
                        line,
                        message: format!("operand type mismatch: {lt} vs {rt}"),
                    });
                }
                match op {
                    BinOp::Add if lt == Type::Str => Type::Str, // concatenation
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem => {
                        if lt == Type::Int || lt == Type::Float {
                            lt
                        } else {
                            return Err(SemaError {
                                line,
                                message: format!("arithmetic requires int or float, found {lt}"),
                            });
                        }
                    }
                    BinOp::Eq | BinOp::Ne => {
                        if lt == Type::Unit {
                            return Err(SemaError {
                                line,
                                message: "cannot compare unit values".into(),
                            });
                        }
                        Type::Bool
                    }
                    BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                        if lt == Type::Int || lt == Type::Float {
                            Type::Bool
                        } else {
                            return Err(SemaError {
                                line,
                                message: format!("ordering comparison requires int or float, found {lt}"),
                            });
                        }
                    }
                    BinOp::And | BinOp::Or => {
                        if lt == Type::Bool {
                            Type::Bool
                        } else {
                            return Err(SemaError {
                                line,
                                message: format!("`and`/`or` require bool, found {lt}"),
                            });
                        }
                    }
                }
            }
            ExprKind::Call(name, args) => {
                let line = expr.line;
                // `print` is a compiler builtin, polymorphic over printable types.
                if name == "print" {
                    if args.len() != 1 {
                        return Err(SemaError {
                            line,
                            message: format!("print takes exactly 1 argument, got {}", args.len()),
                        });
                    }
                    let t = self.check_expr(&mut args[0])?;
                    if t == Type::Unit {
                        return Err(SemaError {
                            line,
                            message: "cannot print a unit value".into(),
                        });
                    }
                    Type::Unit
                } else {
                    let sig = self
                        .functions
                        .get(name)
                        .cloned()
                        .ok_or_else(|| SemaError {
                            line,
                            message: format!("call to undefined function `{name}`"),
                        })?;
                    if args.len() < sig.params.len()
                        || (!sig.varargs && args.len() != sig.params.len())
                    {
                        return Err(SemaError {
                            line,
                            message: format!(
                                "`{name}` expects {}{} argument(s), got {}",
                                if sig.varargs { "at least " } else { "" },
                                sig.params.len(),
                                args.len()
                            ),
                        });
                    }
                    for (i, arg) in args.iter_mut().enumerate() {
                        let at = self.check_expr(arg)?;
                        if let Some(expected) = sig.params.get(i) {
                            if at != *expected {
                                return Err(SemaError {
                                    line,
                                    message: format!(
                                        "argument {} of `{name}`: expected {expected}, found {at}",
                                        i + 1
                                    ),
                                });
                            }
                        } else if at == Type::Unit {
                            return Err(SemaError {
                                line,
                                message: "cannot pass a unit value as a vararg".into(),
                            });
                        }
                    }
                    sig.ret
                }
            }
        };
        expr.ty = Some(ty);
        Ok(ty)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    fn analyze(src: &str) -> Result<Program, String> {
        let mut prog = parse(src).map_err(|e| e.to_string())?;
        Analyzer::new()
            .analyze(&mut prog)
            .map_err(|e| e.to_string())?;
        Ok(prog)
    }

    #[test]
    fn infers_let_types() {
        let prog = analyze("fn main() -> int:\n    let x = 1 + 2\n    return x\n").unwrap();
        let Stmt::Let { ty, .. } = &prog.functions[0].body[0] else {
            panic!()
        };
        assert_eq!(*ty, Some(Type::Int));
    }

    #[test]
    fn rejects_type_mismatch() {
        assert!(analyze("fn main():\n    let x = 1 + 2.0\n").is_err());
        assert!(analyze("fn main():\n    let x: float = 1\n").is_err());
        assert!(analyze("fn main() -> int:\n    return true\n").is_err());
    }

    #[test]
    fn rejects_undefined_names() {
        assert!(analyze("fn main():\n    let x = y\n").is_err());
        assert!(analyze("fn main():\n    foo()\n").is_err());
    }

    #[test]
    fn rejects_non_bool_condition() {
        assert!(analyze("fn main():\n    if 1:\n        return\n").is_err());
        assert!(analyze("fn main():\n    while 1.5:\n        return\n").is_err());
    }

    #[test]
    fn scoping_blocks_shadow_and_expire() {
        // y defined in if-block is not visible after it
        assert!(
            analyze("fn main():\n    if true:\n        let y = 1\n    let z = y\n").is_err()
        );
    }

    #[test]
    fn checks_extern_calls() {
        let src = "extern fn printf(fmt: str, ...) -> int\nfn main() -> int:\n    printf(\"%d\\n\", 42)\n    return 0\n";
        assert!(analyze(src).is_ok());
        // wrong first arg type
        let bad = "extern fn printf(fmt: str, ...) -> int\nfn main() -> int:\n    printf(1)\n    return 0\n";
        assert!(analyze(bad).is_err());
    }

    #[test]
    fn break_outside_loop_rejected() {
        assert!(analyze("fn main():\n    break\n").is_err());
    }

    #[test]
    fn string_concat_types_as_str() {
        let prog =
            analyze("fn main():\n    let s = \"a\" + \"b\"\n    print(s)\n").unwrap();
        let Stmt::Let { ty, .. } = &prog.functions[0].body[0] else {
            panic!()
        };
        assert_eq!(*ty, Some(Type::Str));
    }
}
