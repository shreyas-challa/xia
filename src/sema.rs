//! Phase 3: semantic analysis.
//!
//! - Scoped symbol tables (a stack of `HashMap`s) track variables and types.
//! - Expression types are inferred bottom-up and annotated onto the AST.
//! - `extern fn` declarations are registered as foreign symbols; calls to
//!   them type-check like normal calls (varargs allowed).

use crate::ast::*;
use crate::diag::Span;
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone)]
pub struct SemaError {
    pub span: Span,
    pub message: String,
}

impl fmt::Display for SemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "semantic error (line {}): {}", self.span.line, self.message)
    }
}

impl std::error::Error for SemaError {}

type SResult<T> = Result<T, SemaError>;

#[derive(Debug, Clone)]
pub struct FnSig {
    pub params: Vec<Type>,
    pub varargs: bool,
    pub ret: Type,
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
    /// Struct methods, keyed by (struct id, method name). The signature's
    /// `params` exclude the implicit receiver.
    methods: HashMap<(u32, String), FnSig>,
    structs: Vec<StructDef>,
    struct_ids: HashMap<String, u32>,
    enums: Vec<EnumDef>,
    enum_ids: HashMap<String, u32>,
    /// Variant name -> (enum id, variant index). Variant names are global and
    /// unique, so a bare `Some(x)` / `None` resolves without qualification.
    variants: HashMap<String, (u32, u32)>,
    symbols: SymbolTable,
    current_ret: Type,
    loop_depth: usize,
}

const BUILTINS: &[&str] = &["print", "str", "len", "push", "pop", "find"];

impl Analyzer {
    pub fn new() -> Self {
        Analyzer {
            functions: HashMap::new(),
            methods: HashMap::new(),
            structs: Vec::new(),
            struct_ids: HashMap::new(),
            enums: Vec::new(),
            enum_ids: HashMap::new(),
            variants: HashMap::new(),
            symbols: SymbolTable::new(),
            current_ret: Type::Unit,
            loop_depth: 0,
        }
    }

    /// Resolve a type to a user-facing name (struct/enum ids -> their names).
    fn type_name(&self, t: Type) -> String {
        match t {
            Type::Struct(id) => self.structs[id as usize].name.clone(),
            Type::Enum(id) => self.enums[id as usize].name.clone(),
            Type::Array(..) => format!("[{}]", self.type_name(t.array_elem().unwrap())),
            other => other.to_string(),
        }
    }

    /// Type-check the whole program, annotating expression types in place.
    pub fn analyze(&mut self, program: &mut Program) -> SResult<()> {
        self.structs = program.structs.clone();
        for (i, s) in self.structs.iter().enumerate() {
            self.struct_ids.insert(s.name.clone(), i as u32);
            if BUILTINS.contains(&s.name.as_str()) {
                return Err(SemaError {
                    span: Span::line_only(s.line),
                    message: format!("struct `{}` collides with a builtin", s.name),
                });
            }
        }
        self.enums = program.enums.clone();
        for (i, e) in self.enums.iter().enumerate() {
            if BUILTINS.contains(&e.name.as_str()) {
                return Err(SemaError {
                    span: Span::line_only(e.line),
                    message: format!("enum `{}` collides with a builtin", e.name),
                });
            }
            if self.struct_ids.contains_key(&e.name) {
                return Err(SemaError {
                    span: Span::line_only(e.line),
                    message: format!("`{}` is declared as both a struct and an enum", e.name),
                });
            }
            self.enum_ids.insert(e.name.clone(), i as u32);
        }
        // Variant names are global and unique; they must not collide with a
        // builtin, struct constructor, or another variant.
        for (i, e) in self.enums.iter().enumerate() {
            for (j, v) in e.variants.iter().enumerate() {
                if BUILTINS.contains(&v.name.as_str()) || self.struct_ids.contains_key(&v.name) {
                    return Err(SemaError {
                        span: Span::line_only(e.line),
                        message: format!(
                            "variant `{}` collides with a builtin or struct of the same name",
                            v.name
                        ),
                    });
                }
                if self.variants.insert(v.name.clone(), (i as u32, j as u32)).is_some() {
                    return Err(SemaError {
                        span: Span::line_only(e.line),
                        message: format!(
                            "variant `{}` is defined in more than one enum (names are global)",
                            v.name
                        ),
                    });
                }
            }
        }
        // Register every signature first so call order doesn't matter.
        for e in &program.externs {
            let sig = FnSig {
                params: e.params.clone(),
                varargs: e.varargs,
                ret: e.ret,
            };
            if self.functions.insert(e.name.clone(), sig).is_some() {
                return Err(SemaError {
                    span: Span::line_only(e.line),
                    message: format!("duplicate declaration of `{}`", e.name),
                });
            }
        }
        for f in &program.functions {
            let sig = FnSig {
                params: f.params.iter().map(|p| p.ty).collect(),
                varargs: false,
                ret: f.ret,
            };
            if let Some(recv) = &f.recv {
                // A method: keyed by (receiver struct, method name), separate
                // from the free-function namespace.
                let Type::Struct(id) = recv.ty else {
                    return Err(SemaError {
                        span: Span::line_only(f.line),
                        message: format!(
                            "method receiver `{}` must be a struct, found {}",
                            recv.name,
                            self.type_name(recv.ty)
                        ),
                    });
                };
                if self.methods.insert((id, f.name.clone()), sig).is_some() {
                    return Err(SemaError {
                        span: Span::line_only(f.line),
                        message: format!(
                            "duplicate method `{}` on `{}`",
                            f.name, self.structs[id as usize].name
                        ),
                    });
                }
                continue;
            }
            if self.functions.insert(f.name.clone(), sig).is_some() {
                return Err(SemaError {
                    span: Span::line_only(f.line),
                    message: format!("duplicate definition of `{}`", f.name),
                });
            }
            if self.struct_ids.contains_key(&f.name) {
                return Err(SemaError {
                    span: Span::line_only(f.line),
                    message: format!("`{}` is defined as both a struct and a function", f.name),
                });
            }
            if self.enum_ids.contains_key(&f.name) || self.variants.contains_key(&f.name) {
                return Err(SemaError {
                    span: Span::line_only(f.line),
                    message: format!("`{}` collides with an enum or variant of the same name", f.name),
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
        if let Some(recv) = &f.recv {
            self.symbols.declare(&recv.name, recv.ty);
        }
        for p in &f.params {
            if !self.symbols.declare(&p.name, p.ty) {
                return Err(SemaError {
                    span: Span::line_only(f.line),
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
            Stmt::Let { name, ty, value, line: _ } => {
                // An empty array literal has no type of its own; it takes the
                // annotated one (`let xs: [int] = []`).
                if let ExprKind::ArrayLit(elems) = &value.kind {
                    if elems.is_empty() {
                        let Some(annotated @ Type::Array(..)) = ty else {
                            return Err(SemaError {
                                span: value.span,
                                message: format!(
                                    "cannot infer the element type of `[]`; annotate the binding, e.g. `let {name}: [int] = []`"
                                ),
                            });
                        };
                        value.ty = Some(*annotated);
                        if !self.symbols.declare(name, *annotated) {
                            return Err(SemaError {
                                span: value.span,
                                message: format!("`{name}` is already defined in this scope"),
                            });
                        }
                        return Ok(());
                    }
                }
                let inferred = self.check_expr(value)?;
                if inferred == Type::Unit {
                    return Err(SemaError {
                        span: value.span,
                        message: format!("cannot bind `{name}` to a unit (no-value) expression"),
                    });
                }
                if let Some(annotated) = ty {
                    if *annotated != inferred {
                        return Err(SemaError {
                            span: value.span,
                            message: format!(
                                "type mismatch: `{name}` declared as {annotated} but initialized with {inferred}"
                            ),
                        });
                    }
                }
                *ty = Some(inferred);
                if !self.symbols.declare(name, inferred) {
                    return Err(SemaError {
                        span: value.span,
                        message: format!("`{name}` is already defined in this scope"),
                    });
                }
                Ok(())
            }
            Stmt::Assign { name, value, line } => {
                let Some(var_ty) = self.symbols.lookup(name) else {
                    return Err(SemaError {
                        span: Span::line_only(*line),
                        message: format!("assignment to undefined variable `{name}`"),
                    });
                };
                let val_ty = self.check_expr(value)?;
                if val_ty != var_ty {
                    return Err(SemaError {
                        span: value.span,
                        message: format!(
                            "type mismatch: cannot assign {val_ty} to `{name}` of type {var_ty}"
                        ),
                    });
                }
                Ok(())
            }
            Stmt::IndexAssign { target, index, value, line: _ } => {
                let target_ty = self.check_expr(target)?;
                let Some(elem_ty) = target_ty.array_elem() else {
                    return Err(SemaError {
                        span: target.span,
                        message: format!("cannot index-assign into a value of type {target_ty}"),
                    });
                };
                let idx_ty = self.check_expr(index)?;
                if idx_ty != Type::Int {
                    return Err(SemaError {
                        span: index.span,
                        message: format!("array index must be int, found {idx_ty}"),
                    });
                }
                let val_ty = self.check_expr(value)?;
                if val_ty != elem_ty {
                    return Err(SemaError {
                        span: value.span,
                        message: format!(
                            "type mismatch: cannot store {val_ty} in an array of {elem_ty}"
                        ),
                    });
                }
                Ok(())
            }
            Stmt::FieldAssign { target, field, value, line: _ } => {
                let bt = self.check_expr(target)?;
                let Type::Struct(id) = bt else {
                    return Err(SemaError {
                        span: target.span,
                        message: format!(
                            "type {} has no fields to assign",
                            self.type_name(bt)
                        ),
                    });
                };
                let def_name = self.structs[id as usize].name.clone();
                let Some((_, fty)) = self.structs[id as usize].field(field) else {
                    return Err(SemaError {
                        span: target.span,
                        message: format!("`{def_name}` has no field `{field}`"),
                    });
                };
                let vt = self.check_expr(value)?;
                if vt != fty {
                    return Err(SemaError {
                        span: value.span,
                        message: format!(
                            "type mismatch: field `{field}` of `{def_name}` is {}, found {}",
                            self.type_name(fty),
                            self.type_name(vt)
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
                        span: value.as_ref().map(|e| e.span).unwrap_or(Span::line_only(*line)),
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
                        span: cond.span,
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
                        span: cond.span,
                        message: format!("while condition must be bool, found {cond_ty}"),
                    });
                }
                self.loop_depth += 1;
                self.check_block(body)?;
                self.loop_depth -= 1;
                Ok(())
            }
            Stmt::For { var, start, end, body, line: _ } => {
                for (label, e) in [("start", &mut *start), ("end", &mut *end)] {
                    let t = self.check_expr(e)?;
                    if t != Type::Int {
                        return Err(SemaError {
                            span: e.span,
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
            Stmt::ForEach { var, iterable, body, line: _ } => {
                let t = self.check_expr(iterable)?;
                let Some(elem_ty) = t.array_elem() else {
                    return Err(SemaError {
                        span: iterable.span,
                        message: format!("for-in iterates over an array, found {t}"),
                    });
                };
                self.symbols.push();
                self.symbols.declare(var, elem_ty);
                self.loop_depth += 1;
                self.check_block(body)?;
                self.loop_depth -= 1;
                self.symbols.pop();
                Ok(())
            }
            Stmt::Match { scrutinee, arms, line } => {
                let sty = self.check_expr(scrutinee)?;
                let Type::Enum(id) = sty else {
                    return Err(SemaError {
                        span: scrutinee.span,
                        message: format!(
                            "match scrutinee must be an enum, found {}",
                            self.type_name(sty)
                        ),
                    });
                };
                let def = self.enums[id as usize].clone();
                let mut covered = vec![false; def.variants.len()];
                let mut has_wildcard = false;
                for arm in arms.iter_mut() {
                    match &mut arm.pattern {
                        Pattern::Wildcard => {
                            if has_wildcard {
                                return Err(SemaError {
                                    span: Span::line_only(*line),
                                    message: "duplicate `_` arm in match".into(),
                                });
                            }
                            has_wildcard = true;
                            self.check_block(&mut arm.body)?;
                        }
                        Pattern::Variant { name, bindings, types } => {
                            let Some((vidx, fields)) =
                                def.variant(name).map(|(i, v)| (i, v.fields.clone()))
                            else {
                                return Err(SemaError {
                                    span: Span::line_only(*line),
                                    message: format!("`{}` has no variant `{name}`", def.name),
                                });
                            };
                            if covered[vidx] {
                                return Err(SemaError {
                                    span: Span::line_only(*line),
                                    message: format!("duplicate match arm for variant `{name}`"),
                                });
                            }
                            covered[vidx] = true;
                            if bindings.len() != fields.len() {
                                return Err(SemaError {
                                    span: Span::line_only(*line),
                                    message: format!(
                                        "variant `{name}` binds {} value(s) but carries {} field(s)",
                                        bindings.len(),
                                        fields.len()
                                    ),
                                });
                            }
                            *types = fields.clone();
                            self.symbols.push();
                            for (b, t) in bindings.iter().zip(fields.iter()) {
                                if !self.symbols.declare(b, *t) {
                                    self.symbols.pop();
                                    return Err(SemaError {
                                        span: Span::line_only(*line),
                                        message: format!("duplicate binding `{b}` in pattern"),
                                    });
                                }
                            }
                            let r = self.check_block(&mut arm.body);
                            self.symbols.pop();
                            r?;
                        }
                    }
                }
                if !has_wildcard {
                    let missing: Vec<String> = covered
                        .iter()
                        .enumerate()
                        .filter(|(_, c)| !**c)
                        .map(|(i, _)| def.variants[i].name.clone())
                        .collect();
                    if !missing.is_empty() {
                        return Err(SemaError {
                            span: Span::line_only(*line),
                            message: format!(
                                "non-exhaustive match on `{}`: missing {} (add the variant(s) or a `_` arm)",
                                def.name,
                                missing.join(", ")
                            ),
                        });
                    }
                }
                Ok(())
            }
            Stmt::Break { line } | Stmt::Continue { line } => {
                if self.loop_depth == 0 {
                    return Err(SemaError {
                        span: Span::line_only(*line),
                        message: "`break`/`continue` outside of a loop".into(),
                    });
                }
                Ok(())
            }
            Stmt::Retain(_) | Stmt::Release(_) => Ok(()),
        }
    }

    /// A `Call` or `Var` whose name is a known enum variant is an enum
    /// constructor. Rewrite it to `EnumInit`, type-checking the payload, and
    /// return the enum type; otherwise leave it alone.
    fn try_enum_init(&mut self, expr: &mut Expr) -> SResult<Option<Type>> {
        let name = match &expr.kind {
            ExprKind::Call(n, _) => n.clone(),
            // A plain identifier is a nullary variant only if it isn't a local.
            ExprKind::Var(n) if self.symbols.lookup(n).is_none() => n.clone(),
            _ => return Ok(None),
        };
        let Some(&(eid, vidx)) = self.variants.get(&name) else {
            return Ok(None);
        };
        let span = expr.span;
        let mut args = match &mut expr.kind {
            ExprKind::Call(_, a) => std::mem::take(a),
            _ => Vec::new(),
        };
        let fields = self.enums[eid as usize].variants[vidx as usize].fields.clone();
        if args.len() != fields.len() {
            return Err(SemaError {
                span,
                message: format!(
                    "variant `{name}` takes {} payload value(s), got {}",
                    fields.len(),
                    args.len()
                ),
            });
        }
        for (arg, fty) in args.iter_mut().zip(fields.iter()) {
            let at = self.check_expr(arg)?;
            if at != *fty {
                return Err(SemaError {
                    span: arg.span,
                    message: format!(
                        "payload of `{name}`: expected {}, found {}",
                        self.type_name(*fty),
                        self.type_name(at)
                    ),
                });
            }
        }
        expr.kind = ExprKind::EnumInit(eid, vidx, args);
        Ok(Some(Type::Enum(eid)))
    }

    fn check_expr(&mut self, expr: &mut Expr) -> SResult<Type> {
        if let Some(ty) = self.try_enum_init(expr)? {
            expr.ty = Some(ty);
            return Ok(ty);
        }
        let ty = match &mut expr.kind {
            ExprKind::Int(_) => Type::Int,
            ExprKind::Float(_) => Type::Float,
            ExprKind::Bool(_) => Type::Bool,
            ExprKind::Str(_) => Type::Str,
            ExprKind::Var(name) => self.symbols.lookup(name).ok_or_else(|| SemaError {
                span: expr.span,
                message: format!("undefined variable `{name}`"),
            })?,
            ExprKind::Unary(op, operand) => {
                let span = expr.span;
                let t = self.check_expr(operand)?;
                match op {
                    UnOp::Neg if t == Type::Int || t == Type::Float => t,
                    UnOp::Neg => {
                        return Err(SemaError {
                            span,
                            message: format!("cannot negate a value of type {t}"),
                        });
                    }
                    UnOp::Not if t == Type::Bool => Type::Bool,
                    UnOp::Not => {
                        return Err(SemaError {
                            span,
                            message: format!("`not` requires bool, found {t}"),
                        });
                    }
                }
            }
            ExprKind::Binary(lhs, op, rhs) => {
                let span = expr.span;
                let lt = self.check_expr(lhs)?;
                let rt = self.check_expr(rhs)?;
                if lt != rt {
                    return Err(SemaError {
                        span,
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
                                span,
                                message: format!("arithmetic requires int or float, found {lt}"),
                            });
                        }
                    }
                    BinOp::Eq | BinOp::Ne => {
                        if lt == Type::Unit
                            || matches!(lt, Type::Array(..) | Type::Struct(_) | Type::Enum(_))
                        {
                            return Err(SemaError {
                                span,
                                message: format!("cannot compare values of type {}", self.type_name(lt)),
                            });
                        }
                        Type::Bool
                    }
                    BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                        if lt == Type::Int || lt == Type::Float {
                            Type::Bool
                        } else {
                            return Err(SemaError {
                                span,
                                message: format!("ordering comparison requires int or float, found {lt}"),
                            });
                        }
                    }
                    BinOp::And | BinOp::Or => {
                        if lt == Type::Bool {
                            Type::Bool
                        } else {
                            return Err(SemaError {
                                span,
                                message: format!("`and`/`or` require bool, found {lt}"),
                            });
                        }
                    }
                }
            }
            ExprKind::Call(name, args) => {
                let span = expr.span;
                // `print` is a compiler builtin, polymorphic over printable types.
                if name == "print" {
                    if args.len() != 1 {
                        return Err(SemaError {
                            span,
                            message: format!("print takes exactly 1 argument, got {}", args.len()),
                        });
                    }
                    let t = self.check_expr(&mut args[0])?;
                    if t == Type::Unit
                        || matches!(t, Type::Array(..) | Type::Struct(_) | Type::Enum(_))
                    {
                        return Err(SemaError {
                            span,
                            message: format!("cannot print a value of type {}", self.type_name(t)),
                        });
                    }
                    Type::Unit
                } else if name == "str" {
                    if args.len() != 1 {
                        return Err(SemaError {
                            span,
                            message: format!("str takes exactly 1 argument, got {}", args.len()),
                        });
                    }
                    let t = self.check_expr(&mut args[0])?;
                    if t == Type::Unit
                        || matches!(t, Type::Array(..) | Type::Struct(_) | Type::Enum(_))
                    {
                        return Err(SemaError {
                            span,
                            message: format!("cannot convert a value of type {} to str", self.type_name(t)),
                        });
                    }
                    Type::Str
                } else if name == "len" {
                    if args.len() != 1 {
                        return Err(SemaError {
                            span,
                            message: format!("len takes exactly 1 argument, got {}", args.len()),
                        });
                    }
                    let t = self.check_expr(&mut args[0])?;
                    if t != Type::Str && !matches!(t, Type::Array(..)) {
                        return Err(SemaError {
                            span,
                            message: format!("len requires a str or array, found {t}"),
                        });
                    }
                    Type::Int
                } else if name == "pop" {
                    if args.len() != 1 {
                        return Err(SemaError {
                            span,
                            message: format!("pop takes exactly 1 argument, got {}", args.len()),
                        });
                    }
                    let t = self.check_expr(&mut args[0])?;
                    let Some(elem_ty) = t.array_elem() else {
                        return Err(SemaError {
                            span,
                            message: format!("pop requires an array, found {t}"),
                        });
                    };
                    elem_ty
                } else if name == "find" {
                    if args.len() != 2 {
                        return Err(SemaError {
                            span,
                            message: format!("find takes exactly 2 arguments, got {}", args.len()),
                        });
                    }
                    for arg in args.iter_mut() {
                        let t = self.check_expr(arg)?;
                        if t != Type::Str {
                            return Err(SemaError {
                                span: arg.span,
                                message: format!("find requires str arguments, found {t}"),
                            });
                        }
                    }
                    Type::Int
                } else if name == "push" {
                    if args.len() != 2 {
                        return Err(SemaError {
                            span,
                            message: format!("push takes exactly 2 arguments, got {}", args.len()),
                        });
                    }
                    let arr_ty = self.check_expr(&mut args[0])?;
                    let Some(elem_ty) = arr_ty.array_elem() else {
                        return Err(SemaError {
                            span,
                            message: format!("push requires an array, found {arr_ty}"),
                        });
                    };
                    let val_ty = self.check_expr(&mut args[1])?;
                    if val_ty != elem_ty {
                        return Err(SemaError {
                            span,
                            message: format!("cannot push {val_ty} onto an array of {elem_ty}"),
                        });
                    }
                    Type::Unit
                } else if let Some(&id) = self.struct_ids.get(name.as_str()) {
                    // Struct constructor: positional arguments, one per field.
                    let def = self.structs[id as usize].clone();
                    if args.len() != def.fields.len() {
                        return Err(SemaError {
                            span,
                            message: format!(
                                "`{}` has {} field(s), got {} argument(s)",
                                def.name,
                                def.fields.len(),
                                args.len()
                            ),
                        });
                    }
                    for (arg, field) in args.iter_mut().zip(&def.fields) {
                        let at = self.check_expr(arg)?;
                        if at != field.ty {
                            return Err(SemaError {
                                span: arg.span,
                                message: format!(
                                    "field `{}` of `{}` is {}, found {}",
                                    field.name,
                                    def.name,
                                    self.type_name(field.ty),
                                    self.type_name(at)
                                ),
                            });
                        }
                    }
                    Type::Struct(id)
                } else {
                    let sig = self
                        .functions
                        .get(name)
                        .cloned()
                        .ok_or_else(|| SemaError {
                            span,
                            message: format!("call to undefined function `{name}`"),
                        })?;
                    if args.len() < sig.params.len()
                        || (!sig.varargs && args.len() != sig.params.len())
                    {
                        return Err(SemaError {
                            span,
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
                                    span,
                                    message: format!(
                                        "argument {} of `{name}`: expected {expected}, found {at}",
                                        i + 1
                                    ),
                                });
                            }
                        } else if at == Type::Unit {
                            return Err(SemaError {
                                span,
                                message: "cannot pass a unit value as a vararg".into(),
                            });
                        }
                    }
                    sig.ret
                }
            }
            ExprKind::MethodCall(recv, method, args) => {
                let span = expr.span;
                let recv_ty = self.check_expr(recv)?;
                let Type::Struct(id) = recv_ty else {
                    return Err(SemaError {
                        span: recv.span,
                        message: format!(
                            "type {} has no methods to call",
                            self.type_name(recv_ty)
                        ),
                    });
                };
                let Some(sig) = self.methods.get(&(id, method.clone())).cloned() else {
                    return Err(SemaError {
                        span,
                        message: format!(
                            "`{}` has no method `{method}`",
                            self.structs[id as usize].name
                        ),
                    });
                };
                if args.len() != sig.params.len() {
                    return Err(SemaError {
                        span,
                        message: format!(
                            "method `{method}` of `{}` expects {} argument(s), got {}",
                            self.structs[id as usize].name,
                            sig.params.len(),
                            args.len()
                        ),
                    });
                }
                for (i, arg) in args.iter_mut().enumerate() {
                    let at = self.check_expr(arg)?;
                    let expected = sig.params[i];
                    if at != expected {
                        return Err(SemaError {
                            span: arg.span,
                            message: format!(
                                "argument {} of `{method}`: expected {}, found {}",
                                i + 1,
                                self.type_name(expected),
                                self.type_name(at)
                            ),
                        });
                    }
                }
                sig.ret
            }
            ExprKind::ArrayLit(elems) => {
                let span = expr.span;
                if elems.is_empty() {
                    return Err(SemaError {
                        span,
                        message: "cannot infer the element type of `[]` here".into(),
                    });
                }
                let first = self.check_expr(&mut elems[0])?;
                let Some(arr_ty) = Type::array_of(first) else {
                    return Err(SemaError {
                        span,
                        message: format!("arrays of {first} are not supported"),
                    });
                };
                for e in elems.iter_mut().skip(1) {
                    let t = self.check_expr(e)?;
                    if t != first {
                        return Err(SemaError {
                            span: e.span,
                            message: format!(
                                "array elements must all have the same type: expected {first}, found {t}"
                            ),
                        });
                    }
                }
                arr_ty
            }
            ExprKind::Index(base, index) => {
                let span = expr.span;
                let base_ty = self.check_expr(base)?;
                let elem_ty = match base_ty {
                    Type::Array(..) => base_ty.array_elem().unwrap(),
                    // `s[i]` yields the character at byte i as a 1-char str.
                    Type::Str => Type::Str,
                    other => {
                        return Err(SemaError {
                            span,
                            message: format!("cannot index a value of type {other}"),
                        });
                    }
                };
                let idx_ty = self.check_expr(index)?;
                if idx_ty != Type::Int {
                    return Err(SemaError {
                        span,
                        message: format!("index must be int, found {idx_ty}"),
                    });
                }
                elem_ty
            }
            // Already rewritten by `try_enum_init` (e.g. on a re-check); its
            // payload was validated when it was produced.
            ExprKind::EnumInit(eid, _, _) => Type::Enum(*eid),
            ExprKind::Field(base, fname) => {
                let span = expr.span;
                let bt = self.check_expr(base)?;
                let Type::Struct(id) = bt else {
                    return Err(SemaError {
                        span,
                        message: format!("type {} has no fields", self.type_name(bt)),
                    });
                };
                let def = &self.structs[id as usize];
                let Some((_, fty)) = def.field(fname) else {
                    return Err(SemaError {
                        span,
                        message: format!("`{}` has no field `{fname}`", def.name),
                    });
                };
                fty
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
    fn foreach_var_has_element_type() {
        let prog = analyze(
            "fn main():\n    let xs = [\"a\", \"b\"]\n    for s in xs:\n        print(s + \"!\")\n",
        )
        .unwrap();
        let Stmt::ForEach { iterable, .. } = &prog.functions[0].body[1] else {
            panic!()
        };
        assert_eq!(iterable.ty, Some(Type::Array(ElemType::Str, 1)));
        // iterating a non-array is rejected
        assert!(analyze("fn main():\n    for x in 5:\n        print(x)\n").is_err());
        assert!(analyze("fn main():\n    for c in \"abc\":\n        print(c)\n").is_err());
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
        assert_eq!(*ty, Some(Type::Array(ElemType::Int, 1)));
        assert!(analyze("fn main():\n    let xs = [1, true]\n").is_err());
        assert!(analyze("fn main():\n    let xs = [1]\n    let y = xs[true]\n").is_err());
    }

    #[test]
    fn nested_arrays_infer_and_peel_dimensions() {
        // A literal of arrays infers an extra dimension.
        let prog = analyze(
            "fn main():\n    let g = [[1, 2], [3, 4]]\n    print(g[0][1])\n",
        )
        .unwrap();
        let Stmt::Let { ty, .. } = &prog.functions[0].body[0] else {
            panic!()
        };
        assert_eq!(*ty, Some(Type::Array(ElemType::Int, 2)));
        // indexing once yields a [int], twice an int
        assert!(
            analyze("fn main():\n    let g = [[1]]\n    let r: [int] = g[0]\n    print(r[0])\n")
                .is_ok()
        );
        // a row of the wrong base type is rejected
        assert!(analyze("fn main():\n    let g = [[1], [\"a\"]]\n").is_err());
        // pushing a mismatched row is rejected
        assert!(
            analyze("fn main():\n    let g: [[int]] = []\n    push(g, [\"a\"])\n").is_err()
        );
        // a whole nested array still isn't printable
        assert!(analyze("fn main():\n    let g = [[1]]\n    print(g)\n").is_err());
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
    fn pop_find_and_string_indexing() {
        let prog = analyze(
            "fn main():\n    let s = \"abc\"\n    print(s[0])\n    print(find(s, \"b\"))\n    let xs = [1, 2]\n    print(pop(xs))\n",
        )
        .unwrap();
        let Stmt::Expr(e) = &prog.functions[0].body[1] else { panic!() };
        let ExprKind::Call(_, args) = &e.kind else { panic!() };
        assert_eq!(args[0].ty, Some(Type::Str), "s[i] is a str");
        // rejections
        assert!(analyze("fn main():\n    print(pop(\"abc\"))\n").is_err());
        assert!(analyze("fn main():\n    print(find(1, \"a\"))\n").is_err());
        assert!(analyze("fn main():\n    let s = \"abc\"\n    s[0] = \"x\"\n").is_err());
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
    fn str_builtin_and_fstrings_type_check() {
        assert!(analyze("fn main():\n    print(str(42) + str(1.5) + str(true))\n").is_ok());
        let prog = analyze("fn main():\n    let s = f\"n = {1 + 2}\"\n    print(s)\n").unwrap();
        let Stmt::Let { ty, .. } = &prog.functions[0].body[0] else {
            panic!()
        };
        assert_eq!(*ty, Some(Type::Str));
        // arrays can't be converted/interpolated
        assert!(analyze("fn main():\n    let xs = [1]\n    print(f\"{xs}\")\n").is_err());
        assert!(analyze("fn f():\n    return\nfn main():\n    print(str(f()))\n").is_err());
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

    #[test]
    fn struct_constructor_and_fields_type_check() {
        let src = "struct Point:\n    x: int\n    y: int\nfn main():\n    let p = Point(1, 2)\n    print(p.x + p.y)\n";
        let prog = analyze(src).unwrap();
        let Stmt::Let { ty, .. } = &prog.functions[0].body[0] else {
            panic!()
        };
        assert_eq!(*ty, Some(Type::Struct(0)));

        // wrong arity
        assert!(
            analyze("struct P:\n    x: int\n    y: int\nfn main():\n    let p = P(1)\n").is_err(),
            "too few fields"
        );
        // wrong field type
        assert!(
            analyze("struct P:\n    x: int\nfn main():\n    let p = P(true)\n").is_err(),
            "field type mismatch"
        );
        // unknown field access
        assert!(
            analyze("struct P:\n    x: int\nfn main():\n    let p = P(1)\n    print(p.z)\n")
                .is_err(),
            "unknown field"
        );
        // field assignment is type-checked
        assert!(
            analyze("struct P:\n    x: int\nfn main():\n    let p = P(1)\n    p.x = \"s\"\n")
                .is_err(),
            "field assign type mismatch"
        );
    }

    #[test]
    fn structs_not_printable_or_comparable() {
        assert!(
            analyze("struct P:\n    x: int\nfn main():\n    let p = P(1)\n    print(p)\n").is_err(),
            "struct not printable"
        );
        assert!(
            analyze(
                "struct P:\n    x: int\nfn main():\n    let a = P(1)\n    let b = P(1)\n    let e = a == b\n"
            )
            .is_err(),
            "struct not comparable"
        );
        // and a struct can't be str()'d or interpolated
        assert!(
            analyze("struct P:\n    x: int\nfn main():\n    let p = P(1)\n    print(str(p))\n")
                .is_err()
        );
    }

    #[test]
    fn methods_type_check_and_dispatch_on_receiver() {
        let src = "struct Point:\n    x: int\n    y: int\nfn (p: Point) area() -> int:\n    return p.x * p.y\nfn main() -> int:\n    let p = Point(3, 4)\n    return p.area()\n";
        let prog = analyze(src).unwrap();
        // the call resolves to int
        let Stmt::Return { value: Some(e), .. } = &prog.functions[1].body[1] else {
            panic!()
        };
        assert_eq!(e.ty, Some(Type::Int));

        // method args are checked
        let with_arg = "struct V:\n    n: int\nfn (v: V) plus(k: int) -> int:\n    return v.n + k\nfn main() -> int:\n    let v = V(1)\n    return v.plus(2)\n";
        assert!(analyze(with_arg).is_ok());
        // wrong arg type
        assert!(
            analyze("struct V:\n    n: int\nfn (v: V) plus(k: int) -> int:\n    return v.n + k\nfn main() -> int:\n    let v = V(1)\n    return v.plus(true)\n")
                .is_err(),
            "arg type mismatch"
        );
        // wrong arity
        assert!(
            analyze("struct V:\n    n: int\nfn (v: V) plus(k: int) -> int:\n    return v.n + k\nfn main() -> int:\n    let v = V(1)\n    return v.plus()\n")
                .is_err(),
            "arity mismatch"
        );
    }

    #[test]
    fn calling_unknown_method_or_on_non_struct_rejected() {
        // unknown method on a struct
        assert!(
            analyze("struct P:\n    x: int\nfn main():\n    let p = P(1)\n    p.nope()\n").is_err(),
            "unknown method"
        );
        // method call on a non-struct
        assert!(
            analyze("fn main():\n    let n = 5\n    n.area()\n").is_err(),
            "non-struct receiver"
        );
    }

    #[test]
    fn same_method_name_on_different_structs_is_fine() {
        let src = "struct A:\n    n: int\nstruct B:\n    n: int\nfn (a: A) get() -> int:\n    return a.n\nfn (b: B) get() -> int:\n    return b.n + 1\nfn main() -> int:\n    let a = A(1)\n    let b = B(1)\n    return a.get() + b.get()\n";
        assert!(analyze(src).is_ok());
        // a duplicate method on the same struct is rejected
        assert!(
            analyze("struct A:\n    n: int\nfn (a: A) get() -> int:\n    return a.n\nfn (a: A) get() -> int:\n    return 0\nfn main():\n    return\n")
                .is_err(),
            "duplicate method"
        );
    }

    #[test]
    fn method_receiver_must_be_a_struct() {
        // a method can't be declared on a builtin type
        assert!(parse("fn (x: int) f() -> int:\n    return x\n").is_ok(), "parses fine");
        assert!(
            analyze("fn (x: int) f() -> int:\n    return x\nfn main():\n    return\n").is_err(),
            "non-struct receiver rejected by sema"
        );
    }

    #[test]
    fn enum_construction_and_match_type_check() {
        let src = "enum Opt:\n    Some(int)\n    None\nfn unwrap(o: Opt) -> int:\n    match o:\n        Some(n):\n            return n\n        None:\n            return -1\nfn main() -> int:\n    return unwrap(Some(7)) + unwrap(None)\n";
        let prog = analyze(src).unwrap();
        // `Some(7)` is rewritten to an EnumInit with the enum's type.
        let Stmt::Return { value: Some(e), .. } = &prog.functions[1].body[0] else {
            panic!("expected return");
        };
        let ExprKind::Binary(lhs, _, _) = &e.kind else { panic!() };
        let ExprKind::Call(_, args) = &lhs.kind else { panic!("expected call to unwrap") };
        assert!(matches!(&args[0].kind, ExprKind::EnumInit(_, _, a) if a.len() == 1));
        assert_eq!(args[0].ty, Some(Type::Enum(0)));
    }

    #[test]
    fn match_arm_bindings_have_payload_types() {
        // The binding `n` from `Some(n)` is an int inside the arm.
        let src = "enum Box:\n    Val(str)\n    Empty\nfn show(b: Box):\n    match b:\n        Val(s):\n            print(s + \"!\")\n        Empty:\n            print(\"empty\")\nfn main():\n    return\n";
        assert!(analyze(src).is_ok());
        // wrong use of the binding's type is rejected
        let bad = "enum Box:\n    Val(str)\n    Empty\nfn f(b: Box):\n    match b:\n        Val(s):\n            print(s + 1)\n        Empty:\n            return\nfn main():\n    return\n";
        assert!(analyze(bad).is_err(), "binding s is a str, not int");
    }

    #[test]
    fn match_must_be_exhaustive_or_have_wildcard() {
        let missing = "enum E:\n    A\n    B\n    C\nfn f(e: E) -> int:\n    match e:\n        A:\n            return 1\n        B:\n            return 2\n";
        assert!(analyze(missing).is_err(), "C is not covered");
        let wild = "enum E:\n    A\n    B\n    C\nfn f(e: E) -> int:\n    match e:\n        A:\n            return 1\n        _:\n            return 0\n";
        assert!(analyze(wild).is_ok(), "wildcard makes it exhaustive");
        let full = "enum E:\n    A\n    B\nfn f(e: E) -> int:\n    match e:\n        A:\n            return 1\n        B:\n            return 2\n";
        assert!(analyze(full).is_ok());
    }

    #[test]
    fn match_rejects_bad_arms() {
        // unknown variant
        assert!(
            analyze("enum E:\n    A\n    B\nfn f(e: E):\n    match e:\n        A:\n            return\n        Z:\n            return\n").is_err(),
            "unknown variant Z"
        );
        // duplicate arm
        assert!(
            analyze("enum E:\n    A\n    B\nfn f(e: E):\n    match e:\n        A:\n            return\n        A:\n            return\n        B:\n            return\n").is_err(),
            "duplicate A"
        );
        // wrong binding arity
        assert!(
            analyze("enum E:\n    Pair(int, int)\nfn f(e: E):\n    match e:\n        Pair(a):\n            return\n").is_err(),
            "Pair carries two fields"
        );
        // match on a non-enum
        assert!(
            analyze("fn main():\n    let x = 5\n    match x:\n        A:\n            return\n").is_err(),
            "scrutinee must be an enum"
        );
    }

    #[test]
    fn enum_constructor_payload_is_checked() {
        // wrong payload arity
        assert!(
            analyze("enum Opt:\n    Some(int)\n    None\nfn main():\n    let o = Some()\n").is_err(),
            "Some needs one value"
        );
        // wrong payload type
        assert!(
            analyze("enum Opt:\n    Some(int)\n    None\nfn main():\n    let o = Some(true)\n").is_err(),
            "Some takes an int"
        );
        // nullary variant used as a value
        assert!(
            analyze("enum Opt:\n    Some(int)\n    None\nfn use(o: Opt) -> int:\n    return 0\nfn main() -> int:\n    return use(None)\n").is_ok()
        );
    }

    #[test]
    fn enum_and_variant_names_must_not_collide() {
        // two enums sharing a variant name
        assert!(
            analyze("enum A:\n    X\n    Y\nenum B:\n    X\n    Z\nfn main():\n    return\n").is_err(),
            "variant X defined twice"
        );
        // enum name colliding with a struct
        assert!(
            analyze("struct P:\n    x: int\nenum P:\n    A\nfn main():\n    return\n").is_err(),
            "P is both a struct and an enum"
        );
        // variant colliding with a struct constructor
        assert!(
            analyze("struct Pair:\n    a: int\nenum E:\n    Pair(int)\n    Empty\nfn main():\n    return\n").is_err(),
            "variant Pair collides with struct Pair"
        );
    }

    #[test]
    fn enums_not_printable_or_comparable() {
        assert!(
            analyze("enum E:\n    A\n    B\nfn f(e: E):\n    print(e)\n").is_err(),
            "enum not printable"
        );
        assert!(
            analyze("enum E:\n    A\n    B\nfn f(a: E, b: E) -> bool:\n    return a == b\n").is_err(),
            "enum not comparable"
        );
    }

    #[test]
    fn struct_name_cannot_collide_with_builtin_or_fn() {
        assert!(
            analyze("struct print:\n    x: int\nfn main():\n    return\n").is_err(),
            "struct shadowing a builtin is rejected"
        );
    }
}
