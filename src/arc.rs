//! Phase 3 (continued): Automatic Reference Counting.
//!
//! Xia has no garbage collector. After type checking, this pass walks the AST
//! and inserts `Stmt::Retain` / `Stmt::Release` so every heap value's
//! refcount is balanced at scope boundaries.
//!
//! Ownership model:
//! - A `let` binding owns one reference. Binding from an existing variable
//!   (`let b = a`) aliases it, so a `Retain` is inserted; binding a fresh
//!   value (literal, concat, call result) takes over the +1 it was born with.
//! - Function arguments are **borrowed** from the caller. To keep mutation of
//!   heap-typed parameters uniform, the callee retains each heap parameter on
//!   entry and releases it on exit.
//! - Function return values are **owned (+1)** by the caller. `return e`
//!   is rewritten to `let $ret = e` (retaining if it aliases), followed by
//!   releases of every live local, followed by `return $ret`.
//! - A heap value used as a bare expression statement is bound to a
//!   synthetic temporary and released immediately.
//! - `break` / `continue` release the locals of every scope between the
//!   statement and the loop body, inclusive.
//!
//! Reassignment (`s = <expr>`) is handled in codegen, which must evaluate the
//! right-hand side *before* releasing the old value (consider `s = s + "x"`).

use crate::ast::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScopeKind {
    Function,
    Block,
    Loop,
}

struct Scope {
    kind: ScopeKind,
    /// Heap-typed locals owned by this scope, in declaration order.
    owned: Vec<String>,
}

pub struct ArcInserter {
    scopes: Vec<Scope>,
    tmp_counter: usize,
}

fn is_terminator(stmt: &Stmt) -> bool {
    matches!(
        stmt,
        Stmt::Return { .. } | Stmt::Break { .. } | Stmt::Continue { .. }
    )
}

impl ArcInserter {
    pub fn new() -> Self {
        ArcInserter { scopes: Vec::new(), tmp_counter: 0 }
    }

    pub fn run(&mut self, program: &mut Program) {
        for f in &mut program.functions {
            self.process_function(f);
        }
    }

    fn process_function(&mut self, f: &mut Function) {
        self.scopes.clear();
        let mut fn_scope = Scope { kind: ScopeKind::Function, owned: Vec::new() };
        let mut prologue = Vec::new();
        for p in &f.params {
            if p.ty.is_heap() {
                prologue.push(Stmt::Retain(p.name.clone()));
                fn_scope.owned.push(p.name.clone());
            }
        }
        self.scopes.push(fn_scope);

        let body = std::mem::take(&mut f.body);
        let mut processed = self.process_block(body, ScopeKind::Block);

        // A unit function may fall off the end without `return`; the
        // retained parameters still need their releases.
        if !processed.last().map(is_terminator).unwrap_or(false) {
            let fn_scope = self.scopes.last().unwrap();
            for name in fn_scope.owned.iter().rev() {
                processed.push(Stmt::Release(name.clone()));
            }
        }

        let mut new_body = prologue;
        new_body.extend(processed);
        f.body = new_body;
        self.scopes.pop();
    }

    fn process_block(&mut self, block: Block, kind: ScopeKind) -> Block {
        self.scopes.push(Scope { kind, owned: Vec::new() });
        let mut out: Vec<Stmt> = Vec::new();

        for stmt in block {
            match stmt {
                Stmt::Let { name, ty, value, line } => {
                    let is_heap = ty.map(Type::is_heap).unwrap_or(false);
                    let aliases = matches!(value.kind, ExprKind::Var(_));
                    out.push(Stmt::Let { name: name.clone(), ty, value, line });
                    if is_heap {
                        if aliases {
                            out.push(Stmt::Retain(name.clone()));
                        }
                        self.scopes.last_mut().unwrap().owned.push(name);
                    }
                }
                Stmt::Expr(e) => {
                    let is_heap = e.ty.map(Type::is_heap).unwrap_or(false);
                    if is_heap {
                        // Bind the discarded value so it can be released.
                        let tmp = self.fresh_tmp();
                        let line = e.line;
                        out.push(Stmt::Let {
                            name: tmp.clone(),
                            ty: e.ty,
                            value: e,
                            line,
                        });
                        out.push(Stmt::Release(tmp));
                    } else {
                        out.push(Stmt::Expr(e));
                    }
                }
                Stmt::Return { value, line } => {
                    self.lower_return(value, line, &mut out);
                }
                Stmt::Break { line } => {
                    self.release_through_loop(&mut out);
                    out.push(Stmt::Break { line });
                }
                Stmt::Continue { line } => {
                    self.release_through_loop(&mut out);
                    out.push(Stmt::Continue { line });
                }
                Stmt::If { cond, then_block, else_block } => {
                    let then_block = self.process_block(then_block, ScopeKind::Block);
                    let else_block =
                        else_block.map(|b| self.process_block(b, ScopeKind::Block));
                    out.push(Stmt::If { cond, then_block, else_block });
                }
                Stmt::While { cond, body } => {
                    let body = self.process_block(body, ScopeKind::Loop);
                    out.push(Stmt::While { cond, body });
                }
                Stmt::For { var, start, end, body, line } => {
                    // The loop variable is an int — never heap — so only the
                    // body needs a loop scope for break/continue releases.
                    let body = self.process_block(body, ScopeKind::Loop);
                    out.push(Stmt::For { var, start, end, body, line });
                }
                other @ (Stmt::Assign { .. }
                | Stmt::IndexAssign { .. }
                | Stmt::Retain(_)
                | Stmt::Release(_)) => {
                    out.push(other);
                }
            }
        }

        // Scope epilogue: release everything this scope owns, unless control
        // already left via return/break/continue (those released eagerly).
        if !out.last().map(is_terminator).unwrap_or(false) {
            let scope = self.scopes.last().unwrap();
            for name in scope.owned.iter().rev() {
                out.push(Stmt::Release(name.clone()));
            }
        }

        self.scopes.pop();
        out
    }

    /// `return e` → `let $ret = e` (+ retain if aliasing), release all live
    /// locals, `return $ret`. Ownership of the result transfers to the caller.
    fn lower_return(&mut self, value: Option<Expr>, line: usize, out: &mut Vec<Stmt>) {
        let any_owned = self.scopes.iter().any(|s| !s.owned.is_empty());
        match value {
            Some(e) if any_owned => {
                let is_heap = e.ty.map(Type::is_heap).unwrap_or(false);
                let aliases = matches!(e.kind, ExprKind::Var(_));
                let ty = e.ty;
                let tmp = self.fresh_tmp();
                out.push(Stmt::Let { name: tmp.clone(), ty, value: e, line });
                if is_heap && aliases {
                    out.push(Stmt::Retain(tmp.clone()));
                }
                self.release_all(out);
                let mut ret_expr = Expr::new(ExprKind::Var(tmp), line);
                ret_expr.ty = ty;
                out.push(Stmt::Return { value: Some(ret_expr), line });
            }
            None if any_owned => {
                self.release_all(out);
                out.push(Stmt::Return { value: None, line });
            }
            value => out.push(Stmt::Return { value, line }),
        }
    }

    fn release_all(&self, out: &mut Vec<Stmt>) {
        for scope in self.scopes.iter().rev() {
            for name in scope.owned.iter().rev() {
                out.push(Stmt::Release(name.clone()));
            }
        }
    }

    /// Release owned locals of every scope from the innermost out to and
    /// including the nearest loop body.
    fn release_through_loop(&self, out: &mut Vec<Stmt>) {
        for scope in self.scopes.iter().rev() {
            for name in scope.owned.iter().rev() {
                out.push(Stmt::Release(name.clone()));
            }
            if scope.kind == ScopeKind::Loop {
                break;
            }
        }
    }

    fn fresh_tmp(&mut self) -> String {
        let n = self.tmp_counter;
        self.tmp_counter += 1;
        format!("$arc{n}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;
    use crate::sema::Analyzer;

    fn lower(src: &str) -> Program {
        let mut prog = parse(src).unwrap();
        Analyzer::new().analyze(&mut prog).unwrap();
        ArcInserter::new().run(&mut prog);
        prog
    }

    fn count(stmts: &[Stmt], retain: &mut usize, release: &mut usize) {
        for s in stmts {
            match s {
                Stmt::Retain(_) => *retain += 1,
                Stmt::Release(_) => *release += 1,
                Stmt::If { then_block, else_block, .. } => {
                    count(then_block, retain, release);
                    if let Some(b) = else_block {
                        count(b, retain, release);
                    }
                }
                Stmt::While { body, .. } | Stmt::For { body, .. } => {
                    count(body, retain, release)
                }
                _ => {}
            }
        }
    }

    fn balance(prog: &Program) -> (usize, usize) {
        let (mut retain, mut release) = (0, 0);
        for f in &prog.functions {
            count(&f.body, &mut retain, &mut release);
        }
        (retain, release)
    }

    #[test]
    fn fresh_string_released_at_scope_end() {
        let prog = lower("fn main():\n    let s = \"hi\"\n    print(s)\n");
        let (retain, release) = balance(&prog);
        assert_eq!(retain, 0, "fresh value needs no retain");
        assert_eq!(release, 1, "owned local released at scope end");
    }

    #[test]
    fn alias_gets_retained() {
        let prog = lower("fn main():\n    let a = \"x\"\n    let b = a\n    print(b)\n");
        let (retain, release) = balance(&prog);
        assert_eq!(retain, 1, "alias `let b = a` retains");
        assert_eq!(release, 2, "both bindings release");
    }

    #[test]
    fn heap_params_retained_and_released() {
        let prog = lower("fn f(s: str):\n    print(s)\n");
        let (retain, release) = balance(&prog);
        assert_eq!((retain, release), (1, 1));
    }

    #[test]
    fn return_transfers_ownership() {
        let src = "fn f() -> str:\n    let s = \"hi\"\n    return s\n";
        let prog = lower(src);
        let body = &prog.functions[0].body;
        // let s / let $ret = s / retain $ret / release s / return $ret
        assert!(matches!(&body[2], Stmt::Retain(n) if n.starts_with("$arc")));
        assert!(matches!(&body[3], Stmt::Release(n) if n == "s"));
        assert!(matches!(&body[4], Stmt::Return { .. }));
    }

    #[test]
    fn discarded_heap_expr_bound_and_released() {
        let src = "fn f() -> str:\n    return \"x\"\nfn main():\n    f()\n";
        let prog = lower(src);
        let main = prog.functions.iter().find(|f| f.name == "main").unwrap();
        assert!(matches!(&main.body[0], Stmt::Let { name, .. } if name.starts_with("$arc")));
        assert!(matches!(&main.body[1], Stmt::Release(_)));
    }

    #[test]
    fn break_releases_loop_locals() {
        let src = "fn main():\n    while true:\n        let s = \"x\"\n        break\n";
        let prog = lower(src);
        let Stmt::While { body, .. } = &prog.functions[0].body[0] else {
            panic!()
        };
        let release_before_break = body
            .iter()
            .position(|s| matches!(s, Stmt::Release(n) if n == "s"))
            .unwrap();
        let break_pos = body
            .iter()
            .position(|s| matches!(s, Stmt::Break { .. }))
            .unwrap();
        assert!(release_before_break < break_pos);
    }

    #[test]
    fn inner_scope_released_at_block_end() {
        let src = "fn main():\n    if true:\n        let s = \"x\"\n        print(s)\n";
        let prog = lower(src);
        let Stmt::If { then_block, .. } = &prog.functions[0].body[0] else {
            panic!()
        };
        assert!(matches!(then_block.last().unwrap(), Stmt::Release(n) if n == "s"));
    }

    #[test]
    fn arrays_are_arc_managed_like_strings() {
        let prog = lower("fn main():\n    let xs = [1, 2]\n    print(len(xs))\n");
        assert_eq!(balance(&prog), (0, 1), "array binding released at scope end");
        let prog =
            lower("fn main():\n    let xs = [\"a\"]\n    let ys = xs\n    print(len(ys))\n");
        assert_eq!(balance(&prog), (1, 2), "alias retains; both bindings release");
    }

    #[test]
    fn non_heap_code_untouched() {
        let prog = lower("fn main() -> int:\n    let x = 1\n    return x + 1\n");
        assert_eq!(balance(&prog), (0, 0));
    }
}
