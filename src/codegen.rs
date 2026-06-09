//! Phase 4: LLVM code generation via inkwell.
//!
//! A visitor walks the (type-annotated, ARC-rewritten) AST and lowers it to
//! LLVM IR. Type mapping:
//!
//! | Xia    | LLVM  |
//! |--------|-------|
//! | int    | i64   |
//! | float  | f64   |
//! | bool   | i1    |
//! | str    | ptr   |
//! | unit   | void  |
//!
//! `extern fn` declarations become plain function declarations — the system
//! linker resolves them against libc (or anything else), with zero wrapper
//! overhead.
//!
//! The user's `main` is compiled as `xia_main`; a synthetic C `main`
//! truncates its result to i32 so the CRT entry point links cleanly.

use crate::ast::*;
use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum};
use inkwell::values::{BasicMetadataValueEnum, BasicValueEnum, FunctionValue, PointerValue};
use inkwell::{AddressSpace, FloatPredicate, IntPredicate};
use std::collections::HashMap;

pub struct CodeGen<'ctx> {
    context: &'ctx Context,
    pub module: Module<'ctx>,
    builder: Builder<'ctx>,
    /// Scope stack mapping variable name -> (stack slot, type).
    variables: Vec<HashMap<String, (PointerValue<'ctx>, Type)>>,
    /// (continue target, break target) for each enclosing loop.
    loop_stack: Vec<(BasicBlock<'ctx>, BasicBlock<'ctx>)>,
}

type CResult<T> = Result<T, String>;

impl<'ctx> CodeGen<'ctx> {
    pub fn new(context: &'ctx Context) -> Self {
        let module = context.create_module("xia");
        let builder = context.create_builder();
        CodeGen {
            context,
            module,
            builder,
            variables: Vec::new(),
            loop_stack: Vec::new(),
        }
    }

    fn basic_type(&self, ty: Type) -> BasicTypeEnum<'ctx> {
        match ty {
            Type::Int => self.context.i64_type().into(),
            Type::Float => self.context.f64_type().into(),
            Type::Bool => self.context.bool_type().into(),
            Type::Str => self.context.ptr_type(AddressSpace::default()).into(),
            Type::Unit => unreachable!("unit has no basic type"),
        }
    }

    fn fn_type(
        &self,
        params: &[Type],
        ret: Type,
        varargs: bool,
    ) -> inkwell::types::FunctionType<'ctx> {
        let param_tys: Vec<BasicMetadataTypeEnum> =
            params.iter().map(|t| self.basic_type(*t).into()).collect();
        match ret {
            Type::Unit => self.context.void_type().fn_type(&param_tys, varargs),
            other => self.basic_type(other).fn_type(&param_tys, varargs),
        }
    }

    // ----- top level -------------------------------------------------------

    pub fn compile(&mut self, program: &Program) -> CResult<()> {
        for e in &program.externs {
            self.module
                .add_function(&e.name, self.fn_type(&e.params, e.ret, e.varargs), None);
        }

        // Declare all user functions first so call order doesn't matter.
        for f in &program.functions {
            let name = if f.name == "main" { "xia_main" } else { &f.name };
            let params: Vec<Type> = f.params.iter().map(|p| p.ty).collect();
            self.module
                .add_function(name, self.fn_type(&params, f.ret, false), None);
        }

        for f in &program.functions {
            self.compile_function(f)?;
        }

        if let Some(user_main) = self.module.get_function("xia_main") {
            self.emit_c_main(user_main)?;
        }

        self.module
            .verify()
            .map_err(|e| format!("LLVM module verification failed:\n{}", e.to_string()))
    }

    /// `int main()` shim: calls `xia_main`, truncating int results to i32.
    fn emit_c_main(&mut self, user_main: FunctionValue<'ctx>) -> CResult<()> {
        let i32_ty = self.context.i32_type();
        let c_main = self
            .module
            .add_function("main", i32_ty.fn_type(&[], false), None);
        let entry = self.context.append_basic_block(c_main, "entry");
        self.builder.position_at_end(entry);
        let call = self
            .builder
            .build_call(user_main, &[], "ret")
            .map_err(err)?;
        let exit_code = match call.try_as_basic_value().left() {
            Some(BasicValueEnum::IntValue(v)) if v.get_type().get_bit_width() == 64 => self
                .builder
                .build_int_truncate(v, i32_ty, "code")
                .map_err(err)?,
            _ => i32_ty.const_zero(),
        };
        self.builder.build_return(Some(&exit_code)).map_err(err)?;
        Ok(())
    }

    fn compile_function(&mut self, f: &Function) -> CResult<()> {
        let name = if f.name == "main" { "xia_main" } else { &f.name };
        let function = self.module.get_function(name).unwrap();
        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        self.variables.clear();
        self.variables.push(HashMap::new());
        self.loop_stack.clear();

        for (i, p) in f.params.iter().enumerate() {
            let slot = self
                .builder
                .build_alloca(self.basic_type(p.ty), &p.name)
                .map_err(err)?;
            self.builder
                .build_store(slot, function.get_nth_param(i as u32).unwrap())
                .map_err(err)?;
            self.variables
                .last_mut()
                .unwrap()
                .insert(p.name.clone(), (slot, p.ty));
        }

        self.compile_block(&f.body)?;

        // Implicit return for unit functions that fall off the end.
        if !self.block_terminated() {
            if f.ret == Type::Unit {
                self.builder.build_return(None).map_err(err)?;
            } else {
                return Err(format!(
                    "function `{}` may end without returning a {} value",
                    f.name, f.ret
                ));
            }
        }
        Ok(())
    }

    // ----- statements --------------------------------------------------

    fn block_terminated(&self) -> bool {
        self.builder
            .get_insert_block()
            .and_then(|b| b.get_terminator())
            .is_some()
    }

    fn lookup(&self, name: &str) -> Option<(PointerValue<'ctx>, Type)> {
        self.variables
            .iter()
            .rev()
            .find_map(|scope| scope.get(name))
            .copied()
    }

    fn compile_block(&mut self, block: &Block) -> CResult<()> {
        self.variables.push(HashMap::new());
        for stmt in block {
            if self.block_terminated() {
                break; // dead code after return/break/continue
            }
            self.compile_stmt(stmt)?;
        }
        self.variables.pop();
        Ok(())
    }

    fn compile_stmt(&mut self, stmt: &Stmt) -> CResult<()> {
        match stmt {
            Stmt::Let { name, ty, value, .. } => {
                let ty = ty.expect("sema must annotate let types");
                let val = self.compile_expr(value)?;
                let slot = self
                    .builder
                    .build_alloca(self.basic_type(ty), name)
                    .map_err(err)?;
                self.builder.build_store(slot, val).map_err(err)?;
                self.variables
                    .last_mut()
                    .unwrap()
                    .insert(name.clone(), (slot, ty));
                Ok(())
            }
            Stmt::Assign { name, value, .. } => {
                let (slot, ty) = self
                    .lookup(name)
                    .ok_or_else(|| format!("codegen: unknown variable `{name}`"))?;
                let new_val = self.compile_expr(value)?;
                if ty.is_heap() {
                    // ARC: evaluate first, retain if aliasing another binding,
                    // then release the old value (order matters: `s = s + "x"`).
                    if matches!(value.kind, ExprKind::Var(_)) {
                        self.emit_retain(new_val.into_pointer_value())?;
                    }
                    let old = self
                        .builder
                        .build_load(self.basic_type(ty), slot, "old")
                        .map_err(err)?;
                    self.builder.build_store(slot, new_val).map_err(err)?;
                    self.emit_release(old.into_pointer_value())?;
                } else {
                    self.builder.build_store(slot, new_val).map_err(err)?;
                }
                Ok(())
            }
            Stmt::Expr(e) => {
                self.compile_expr_or_unit(e)?;
                Ok(())
            }
            Stmt::Return { value, .. } => {
                match value {
                    Some(e) => {
                        let v = self.compile_expr(e)?;
                        self.builder.build_return(Some(&v)).map_err(err)?;
                    }
                    None => {
                        self.builder.build_return(None).map_err(err)?;
                    }
                }
                Ok(())
            }
            Stmt::If { cond, then_block, else_block } => {
                let function = self.current_function();
                let cond_v = self.compile_expr(cond)?.into_int_value();
                let then_bb = self.context.append_basic_block(function, "then");
                let merge_bb = self.context.append_basic_block(function, "endif");
                let else_bb = match else_block {
                    Some(_) => self.context.append_basic_block(function, "else"),
                    None => merge_bb,
                };
                self.builder
                    .build_conditional_branch(cond_v, then_bb, else_bb)
                    .map_err(err)?;

                self.builder.position_at_end(then_bb);
                self.compile_block(then_block)?;
                if !self.block_terminated() {
                    self.builder
                        .build_unconditional_branch(merge_bb)
                        .map_err(err)?;
                }

                if let Some(else_block) = else_block {
                    self.builder.position_at_end(else_bb);
                    self.compile_block(else_block)?;
                    if !self.block_terminated() {
                        self.builder
                            .build_unconditional_branch(merge_bb)
                            .map_err(err)?;
                    }
                }

                self.builder.position_at_end(merge_bb);
                Ok(())
            }
            Stmt::While { cond, body } => {
                let function = self.current_function();
                let cond_bb = self.context.append_basic_block(function, "loop.cond");
                let body_bb = self.context.append_basic_block(function, "loop.body");
                let after_bb = self.context.append_basic_block(function, "loop.end");

                self.builder.build_unconditional_branch(cond_bb).map_err(err)?;
                self.builder.position_at_end(cond_bb);
                let cond_v = self.compile_expr(cond)?.into_int_value();
                self.builder
                    .build_conditional_branch(cond_v, body_bb, after_bb)
                    .map_err(err)?;

                self.builder.position_at_end(body_bb);
                self.loop_stack.push((cond_bb, after_bb));
                self.compile_block(body)?;
                self.loop_stack.pop();
                if !self.block_terminated() {
                    self.builder
                        .build_unconditional_branch(cond_bb)
                        .map_err(err)?;
                }

                self.builder.position_at_end(after_bb);
                Ok(())
            }
            Stmt::Break { .. } => {
                let (_, after) = *self
                    .loop_stack
                    .last()
                    .ok_or("codegen: break outside loop")?;
                self.builder.build_unconditional_branch(after).map_err(err)?;
                Ok(())
            }
            Stmt::Continue { .. } => {
                let (cond, _) = *self
                    .loop_stack
                    .last()
                    .ok_or("codegen: continue outside loop")?;
                self.builder.build_unconditional_branch(cond).map_err(err)?;
                Ok(())
            }
            Stmt::Retain(name) => {
                let (slot, ty) = self
                    .lookup(name)
                    .ok_or_else(|| format!("codegen: unknown variable `{name}`"))?;
                let v = self
                    .builder
                    .build_load(self.basic_type(ty), slot, name)
                    .map_err(err)?;
                self.emit_retain(v.into_pointer_value())
            }
            Stmt::Release(name) => {
                let (slot, ty) = self
                    .lookup(name)
                    .ok_or_else(|| format!("codegen: unknown variable `{name}`"))?;
                let v = self
                    .builder
                    .build_load(self.basic_type(ty), slot, name)
                    .map_err(err)?;
                self.emit_release(v.into_pointer_value())
            }
        }
    }

    fn current_function(&self) -> FunctionValue<'ctx> {
        self.builder
            .get_insert_block()
            .unwrap()
            .get_parent()
            .unwrap()
    }

    // ----- expressions ----------------------------------------------------

    fn compile_expr_or_unit(&mut self, e: &Expr) -> CResult<Option<BasicValueEnum<'ctx>>> {
        match (&e.kind, e.ty) {
            (ExprKind::Call(name, args), Some(Type::Unit)) => {
                self.compile_call(name, args, e.line)?;
                Ok(None)
            }
            _ => Ok(Some(self.compile_expr(e)?)),
        }
    }

    fn compile_expr(&mut self, e: &Expr) -> CResult<BasicValueEnum<'ctx>> {
        match &e.kind {
            ExprKind::Int(n) => Ok(self.context.i64_type().const_int(*n as u64, true).into()),
            ExprKind::Float(x) => Ok(self.context.f64_type().const_float(*x).into()),
            ExprKind::Bool(b) => Ok(self.context.bool_type().const_int(*b as u64, false).into()),
            ExprKind::Str(_) => Err("codegen: string support lands with the ARC runtime".into()),
            ExprKind::Var(name) => {
                let (slot, ty) = self
                    .lookup(name)
                    .ok_or_else(|| format!("codegen: unknown variable `{name}`"))?;
                self.builder
                    .build_load(self.basic_type(ty), slot, name)
                    .map_err(err)
            }
            ExprKind::Unary(op, operand) => {
                let v = self.compile_expr(operand)?;
                match (op, v) {
                    (UnOp::Neg, BasicValueEnum::IntValue(i)) => {
                        Ok(self.builder.build_int_neg(i, "neg").map_err(err)?.into())
                    }
                    (UnOp::Neg, BasicValueEnum::FloatValue(f)) => {
                        Ok(self.builder.build_float_neg(f, "neg").map_err(err)?.into())
                    }
                    (UnOp::Not, BasicValueEnum::IntValue(b)) => {
                        Ok(self.builder.build_not(b, "not").map_err(err)?.into())
                    }
                    _ => Err("codegen: invalid unary operand".into()),
                }
            }
            ExprKind::Binary(lhs, op, rhs) => self.compile_binary(lhs, *op, rhs),
            ExprKind::Call(name, args) => {
                self.compile_call(name, args, e.line)?
                    .ok_or_else(|| format!("codegen: `{name}` returns no value (line {})", e.line))
            }
        }
    }

    fn compile_binary(
        &mut self,
        lhs: &Expr,
        op: BinOp,
        rhs: &Expr,
    ) -> CResult<BasicValueEnum<'ctx>> {
        // Short-circuit `and` / `or` need control flow, not plain instructions.
        if matches!(op, BinOp::And | BinOp::Or) {
            return self.compile_short_circuit(lhs, op, rhs);
        }

        let l = self.compile_expr(lhs)?;
        let r = self.compile_expr(rhs)?;
        let b = &self.builder;

        match (l, r) {
            (BasicValueEnum::IntValue(l), BasicValueEnum::IntValue(r)) => {
                let v: BasicValueEnum = match op {
                    BinOp::Add => b.build_int_add(l, r, "add").map_err(err)?.into(),
                    BinOp::Sub => b.build_int_sub(l, r, "sub").map_err(err)?.into(),
                    BinOp::Mul => b.build_int_mul(l, r, "mul").map_err(err)?.into(),
                    BinOp::Div => b.build_int_signed_div(l, r, "div").map_err(err)?.into(),
                    BinOp::Rem => b.build_int_signed_rem(l, r, "rem").map_err(err)?.into(),
                    BinOp::Eq => b
                        .build_int_compare(IntPredicate::EQ, l, r, "eq")
                        .map_err(err)?
                        .into(),
                    BinOp::Ne => b
                        .build_int_compare(IntPredicate::NE, l, r, "ne")
                        .map_err(err)?
                        .into(),
                    BinOp::Lt => b
                        .build_int_compare(IntPredicate::SLT, l, r, "lt")
                        .map_err(err)?
                        .into(),
                    BinOp::Le => b
                        .build_int_compare(IntPredicate::SLE, l, r, "le")
                        .map_err(err)?
                        .into(),
                    BinOp::Gt => b
                        .build_int_compare(IntPredicate::SGT, l, r, "gt")
                        .map_err(err)?
                        .into(),
                    BinOp::Ge => b
                        .build_int_compare(IntPredicate::SGE, l, r, "ge")
                        .map_err(err)?
                        .into(),
                    BinOp::And | BinOp::Or => unreachable!(),
                };
                Ok(v)
            }
            (BasicValueEnum::FloatValue(l), BasicValueEnum::FloatValue(r)) => {
                let v: BasicValueEnum = match op {
                    BinOp::Add => b.build_float_add(l, r, "fadd").map_err(err)?.into(),
                    BinOp::Sub => b.build_float_sub(l, r, "fsub").map_err(err)?.into(),
                    BinOp::Mul => b.build_float_mul(l, r, "fmul").map_err(err)?.into(),
                    BinOp::Div => b.build_float_div(l, r, "fdiv").map_err(err)?.into(),
                    BinOp::Rem => b.build_float_rem(l, r, "frem").map_err(err)?.into(),
                    BinOp::Eq => b
                        .build_float_compare(FloatPredicate::OEQ, l, r, "feq")
                        .map_err(err)?
                        .into(),
                    BinOp::Ne => b
                        .build_float_compare(FloatPredicate::ONE, l, r, "fne")
                        .map_err(err)?
                        .into(),
                    BinOp::Lt => b
                        .build_float_compare(FloatPredicate::OLT, l, r, "flt")
                        .map_err(err)?
                        .into(),
                    BinOp::Le => b
                        .build_float_compare(FloatPredicate::OLE, l, r, "fle")
                        .map_err(err)?
                        .into(),
                    BinOp::Gt => b
                        .build_float_compare(FloatPredicate::OGT, l, r, "fgt")
                        .map_err(err)?
                        .into(),
                    BinOp::Ge => b
                        .build_float_compare(FloatPredicate::OGE, l, r, "fge")
                        .map_err(err)?
                        .into(),
                    BinOp::And | BinOp::Or => unreachable!(),
                };
                Ok(v)
            }
            _ => Err("codegen: string operations land with the ARC runtime".into()),
        }
    }

    fn compile_short_circuit(
        &mut self,
        lhs: &Expr,
        op: BinOp,
        rhs: &Expr,
    ) -> CResult<BasicValueEnum<'ctx>> {
        let function = self.current_function();
        let l = self.compile_expr(lhs)?.into_int_value();
        let lhs_bb = self.builder.get_insert_block().unwrap();
        let rhs_bb = self.context.append_basic_block(function, "sc.rhs");
        let merge_bb = self.context.append_basic_block(function, "sc.end");

        match op {
            BinOp::And => self
                .builder
                .build_conditional_branch(l, rhs_bb, merge_bb)
                .map_err(err)?,
            BinOp::Or => self
                .builder
                .build_conditional_branch(l, merge_bb, rhs_bb)
                .map_err(err)?,
            _ => unreachable!(),
        };

        self.builder.position_at_end(rhs_bb);
        let r = self.compile_expr(rhs)?.into_int_value();
        let rhs_end_bb = self.builder.get_insert_block().unwrap();
        self.builder
            .build_unconditional_branch(merge_bb)
            .map_err(err)?;

        self.builder.position_at_end(merge_bb);
        let phi = self
            .builder
            .build_phi(self.context.bool_type(), "sc")
            .map_err(err)?;
        let short_val = self
            .context
            .bool_type()
            .const_int((op == BinOp::Or) as u64, false);
        phi.add_incoming(&[(&short_val, lhs_bb), (&r, rhs_end_bb)]);
        Ok(phi.as_basic_value())
    }

    fn compile_call(
        &mut self,
        name: &str,
        args: &[Expr],
        line: usize,
    ) -> CResult<Option<BasicValueEnum<'ctx>>> {
        if name == "print" {
            return self.compile_print(&args[0]).map(|_| None);
        }
        let callee_name = if name == "main" { "xia_main" } else { name };
        let callee = self
            .module
            .get_function(callee_name)
            .ok_or_else(|| format!("codegen: unknown function `{name}` (line {line})"))?;
        let mut compiled: Vec<BasicMetadataValueEnum> = Vec::with_capacity(args.len());
        for a in args {
            compiled.push(self.compile_expr(a)?.into());
        }
        let call = self
            .builder
            .build_call(callee, &compiled, "")
            .map_err(err)?;
        Ok(call.try_as_basic_value().left())
    }

    fn compile_print(&mut self, arg: &Expr) -> CResult<()> {
        Err(format!(
            "codegen: print lands with the ARC runtime (line {})",
            arg.line
        ))
    }

    // ----- ARC runtime hooks (next commit) ---------------------------------

    fn emit_retain(&mut self, _val: PointerValue<'ctx>) -> CResult<()> {
        Err("codegen: ARC runtime not yet lowered".into())
    }

    fn emit_release(&mut self, _val: PointerValue<'ctx>) -> CResult<()> {
        Err("codegen: ARC runtime not yet lowered".into())
    }
}

fn err(e: inkwell::builder::BuilderError) -> String {
    format!("LLVM builder error: {e}")
}

/// Run the full front-end + codegen pipeline, returning the IR as text.
#[cfg(test)]
pub fn compile_to_ir(source: &str) -> Result<String, String> {
    let mut program = crate::parser::parse(source).map_err(|e| e.to_string())?;
    crate::sema::Analyzer::new()
        .analyze(&mut program)
        .map_err(|e| e.to_string())?;
    crate::arc::ArcInserter::new().run(&mut program);
    let context = Context::create();
    let mut cg = CodeGen::new(&context);
    cg.compile(&program)?;
    Ok(cg.module.print_to_string().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_arithmetic_function() {
        let ir = compile_to_ir("fn add(a: int, b: int) -> int:\n    return a + b\n").unwrap();
        assert!(ir.contains("define i64 @add"));
        assert!(ir.contains("add i64"));
    }

    #[test]
    fn emits_c_main_shim() {
        let ir = compile_to_ir("fn main() -> int:\n    return 7\n").unwrap();
        assert!(ir.contains("define i64 @xia_main"));
        assert!(ir.contains("define i32 @main"));
        assert!(ir.contains("call i64 @xia_main"));
    }

    #[test]
    fn compiles_fib_with_control_flow() {
        let src = "fn fib(n: int) -> int:\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n";
        let ir = compile_to_ir(src).unwrap();
        assert!(ir.contains("define i64 @fib"));
        assert!(ir.contains("br i1"));
        assert!(ir.contains("call i64 @fib"));
    }

    #[test]
    fn compiles_while_loop_with_break() {
        let src = "fn main() -> int:\n    let i = 0\n    while true:\n        i = i + 1\n        if i > 10:\n            break\n    return i\n";
        let ir = compile_to_ir(src).unwrap();
        assert!(ir.contains("loop.cond"));
        assert!(ir.contains("loop.end"));
    }

    #[test]
    fn declares_externs_without_bodies() {
        let src = "extern fn putchar(c: int) -> int\nfn main() -> int:\n    putchar(65)\n    return 0\n";
        let ir = compile_to_ir(src).unwrap();
        assert!(ir.contains("declare i64 @putchar(i64)"));
    }

    #[test]
    fn short_circuit_uses_phi() {
        let src = "fn f(a: bool, b: bool) -> bool:\n    return a and b\n";
        let ir = compile_to_ir(src).unwrap();
        assert!(ir.contains("phi i1"));
    }

    #[test]
    fn float_ops_lower_to_float_instructions() {
        let src = "fn f(x: float) -> float:\n    return x * 2.0 + 0.5\n";
        let ir = compile_to_ir(src).unwrap();
        assert!(ir.contains("fmul double"));
        assert!(ir.contains("fadd double"));
    }
}
