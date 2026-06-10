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
                // An empty array literal has no type of its own; it takes the
                // annotated one (`let xs: [int] = []`).
                if let ExprKind::ArrayLit(elems) = &value.kind {
                    if elems.is_empty() {
                        let Some(annotated @ Type::Array(_)) = ty else {
                            return Err(SemaError {
                                line: *line,
                                message: format!(
                                    "cannot infer the element type of `[]`; annotate the binding, e.g. `let {name}: [int] = []`"
                                ),
                            });
                        };
                        value.ty = Some(*annotated);
                        if !self.symbols.declare(name, *annotated) {
                            return Err(SemaError {
                                line: *line,
                                message: format!("`{name}` is already defined in this scope"),
                            });
                        }
                        return Ok(());
                    }
                }
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
            Stmt::IndexAssign { target, index, value, line } => {
                let target_ty = self.check_expr(target)?;
                let Type::Array(elem) = target_ty else {
                    return Err(SemaError {
                        line: *line,
                        message: format!("cannot index-assign into a value of type {target_ty}"),
                    });
                };
                let idx_ty = self.check_expr(index)?;
                if idx_ty != Type::Int {
                    return Err(SemaError {
                        line: *line,
                        message: format!("array index must be int, found {idx_ty}"),
                    });
                }
                let val_ty = self.check_expr(value)?;
                if val_ty != elem.to_type() {
                    return Err(SemaError {
                        line: *line,
                        message: format!(
                            "type mismatch: cannot store {val_ty} in an array of {}",
                            elem.to_type()
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
            Stmt::For { var, start, end, body, line } => {
                for (label, e) in [("start", &mut *start), ("end", &mut *end)] {
                    let t = self.check_expr(e)?;
                    if t != Type::Int {
                        return Err(SemaError {
                            line: *line,
                            message: format!("range {label} must be int, found {t}"),
                        });
                    }
                }
                // The loop variable lives in its own scope wrapping the body.
                self.symbols.push();
                self.symbols.declare(var, Type::Int);
                self.loop_depth += 1;
                self.check_block(body)?;
                self.loop_depth -= 1;
                self.symbols.pop();
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
                        if lt == Type::Unit || matches!(lt, Type::Array(_)) {
                            return Err(SemaError {
                                line,
                                message: format!("cannot compare values of type {lt}"),
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
                    if t == Type::Unit || matches!(t, Type::Array(_)) {
                        return Err(SemaError {
                            line,
                            message: format!("cannot print a value of type {t}"),
                        });
                    }
                    Type::Unit
                } else if name == "len" {
                    if args.len() != 1 {
                        return Err(SemaError {
                            line,
                            message: format!("len takes exactly 1 argument, got {}", args.len()),
                        });
                    }
                    let t = self.check_expr(&mut args[0])?;
                    if t != Type::Str && !matches!(t, Type::Array(_)) {
                        return Err(SemaError {
                            line,
                            message: format!("len requires a str or array, found {t}"),
                        });
                    }
                    Type::Int
                } else if name == "push" {
                    if args.len() != 2 {
                        return Err(SemaError {
                            line,
                            message: format!("push takes exactly 2 arguments, got {}", args.len()),
                        });
                    }
                    let arr_ty = self.check_expr(&mut args[0])?;
                    let Type::Array(elem) = arr_ty else {
                        return Err(SemaError {
                            line,
                            message: format!("push requires an array, found {arr_ty}"),
                        });
                    };
                    let val_ty = self.check_expr(&mut args[1])?;
                    if val_ty != elem.to_type() {
                        return Err(SemaError {
                            line,
                            message: format!(
                                "cannot push {val_ty} onto an array of {}",
                                elem.to_type()
                            ),
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
            ExprKind::ArrayLit(elems) => {
                let line = expr.line;
                if elems.is_empty() {
                    return Err(SemaError {
                        line,
                        message: "cannot infer the element type of `[]` here".into(),
                    });
                }
                let first = self.check_expr(&mut elems[0])?;
                let Some(elem) = ElemType::from_type(first) else {
                    return Err(SemaError {
                        line,
                        message: format!("arrays of {first} are not supported"),
                    });
                };
                for e in elems.iter_mut().skip(1) {
                    let t = self.check_expr(e)?;
                    if t != first {
                        return Err(SemaError {
                            line: e.line,
                            message: format!(
                                "array elements must all have the same type: expected {first}, found {t}"
                            ),
                        });
                    }
                }
                Type::Array(elem)
            }
            ExprKind::Index(base, index) => {
                let line = expr.line;
                let base_ty = self.check_expr(base)?;
                let Type::Array(elem) = base_ty else {
                    return Err(SemaError {
                        line,
                        message: format!("cannot index a value of type {base_ty}"),
                    });
                };
                let idx_ty = self.check_expr(index)?;
                if idx_ty != Type::Int {
                    return Err(SemaError {
                        line,
                        message: format!("array index must be int, found {idx_ty}"),
                    });
                }
                elem.to_type()
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
    fn for_range_bounds_must_be_int() {
        assert!(analyze("fn main():\n    for i in range(1.5):\n        print(i)\n").is_err());
        assert!(analyze("fn main():\n    for i in range(0, true):\n        print(i)\n").is_err());
    }

    #[test]
    fn for_var_scoped_to_loop_and_allows_break() {
        assert!(analyze("fn main():\n    for i in range(3):\n        print(i)\n    print(i)\n").is_err());
        assert!(analyze("fn main():\n    for i in range(3):\n        break\n").is_ok());
    }

    #[test]
    fn break_outside_loop_rejected() {
        assert!(analyze("fn main():\n    break\n").is_err());
    }

    #[test]
    fn array_literal_infers_and_checks() {
        let prog =
            analyze("fn main():\n    let xs = [1, 2, 3]\n    print(xs[0] + 1)\n").unwrap();
        let Stmt::Let { ty, .. } = &prog.functions[0].body[0] else {
            panic!()
        };
        assert_eq!(*ty, Some(Type::Array(ElemType::Int)));
        assert!(analyze("fn main():\n    let xs = [1, true]\n").is_err());
        assert!(analyze("fn main():\n    let xs = [1]\n    let y = xs[true]\n").is_err());
    }

    #[test]
    fn empty_array_needs_annotation() {
        assert!(analyze("fn main():\n    let xs = []\n").is_err());
        assert!(
            analyze("fn main():\n    let xs: [str] = []\n    push(xs, \"a\")\n    print(len(xs))\n")
                .is_ok()
        );
    }

    #[test]
    fn push_and_len_builtins_type_check() {
        assert!(analyze("fn main():\n    let xs = [1]\n    push(xs, \"a\")\n").is_err());
        assert!(analyze("fn main():\n    push(1, 2)\n").is_err());
        assert!(analyze("fn main():\n    print(len(\"abc\"))\n").is_ok());
        assert!(analyze("fn main():\n    print(len(5))\n").is_err());
    }

    #[test]
    fn arrays_not_printable_or_comparable() {
        assert!(analyze("fn main():\n    let xs = [1]\n    print(xs)\n").is_err());
        assert!(
            analyze("fn main():\n    let a = [1]\n    let b = [1]\n    let e = a == b\n").is_err()
        );
    }

    #[test]
    fn index_assign_checks_types() {
        assert!(analyze("fn main():\n    let xs = [1]\n    xs[0] = 2\n").is_ok());
        assert!(analyze("fn main():\n    let xs = [1]\n    xs[0] = \"s\"\n").is_err());
        assert!(analyze("fn main():\n    let s = \"x\"\n    s[0] = \"y\"\n").is_err());
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
