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
use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum, FunctionType};
use inkwell::values::{BasicMetadataValueEnum, BasicValueEnum, FunctionValue, PointerValue};
use inkwell::{AddressSpace, FloatPredicate, IntPredicate};
use std::collections::HashMap;

/// String memory layout: a heap block is `[i64 refcount][bytes...][NUL]` and
/// the `str` value points at the bytes, so it doubles as a `char*` for the C
/// FFI. A negative refcount marks an immortal value (string literals live in
/// constant globals and are never freed). `xia_retain`/`xia_release` are
/// null-safe and skip immortals.
const RC_OFFSET: i64 = 8;

pub struct CodeGen<'ctx> {
    context: &'ctx Context,
    pub module: Module<'ctx>,
    builder: Builder<'ctx>,
    /// Scope stack mapping variable name -> (stack slot, type).
    variables: Vec<HashMap<String, (PointerValue<'ctx>, Type)>>,
    /// (continue target, break target) for each enclosing loop.
    loop_stack: Vec<(BasicBlock<'ctx>, BasicBlock<'ctx>)>,
    /// Function name -> (is_extern, return type), for ARC at call sites.
    sigs: HashMap<String, (bool, Type)>,
    /// Fresh (+1) heap values produced while compiling the current statement;
    /// released once the statement completes unless ownership transferred.
    stmt_temps: Vec<PointerValue<'ctx>>,
    /// Interned string literal globals.
    str_literals: HashMap<String, PointerValue<'ctx>>,
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
            sigs: HashMap::new(),
            stmt_temps: Vec::new(),
            str_literals: HashMap::new(),
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
            self.sigs.insert(e.name.clone(), (true, e.ret));
            self.module
                .add_function(&e.name, self.fn_type(&e.params, e.ret, e.varargs), None);
        }
        for f in &program.functions {
            self.sigs.insert(f.name.clone(), (false, f.ret));
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
                if ty.is_heap() {
                    // Ownership of a fresh value moves into the binding.
                    self.consume_temp(val.into_pointer_value());
                }
                self.flush_temps(0)
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
                    self.consume_temp(new_val.into_pointer_value());
                } else {
                    self.builder.build_store(slot, new_val).map_err(err)?;
                }
                self.flush_temps(0)
            }
            Stmt::Expr(e) => {
                self.compile_expr_or_unit(e)?;
                self.flush_temps(0)
            }
            Stmt::Return { value, .. } => {
                match value {
                    Some(e) => {
                        let v = self.compile_expr(e)?;
                        if e.ty.map(Type::is_heap).unwrap_or(false) {
                            // The caller owns the result; don't release it here.
                            self.consume_temp(v.into_pointer_value());
                        }
                        self.flush_temps(0)?;
                        self.builder.build_return(Some(&v)).map_err(err)?;
                    }
                    None => {
                        self.flush_temps(0)?;
                        self.builder.build_return(None).map_err(err)?;
                    }
                }
                Ok(())
            }
            Stmt::If { cond, then_block, else_block } => {
                let function = self.current_function();
                let cond_v = self.compile_expr(cond)?.into_int_value();
                self.flush_temps(0)?;
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
                self.flush_temps(0)?;
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
            ExprKind::Str(s) => Ok(self.str_literal(s)?.into()),
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
            (BasicValueEnum::PointerValue(l), BasicValueEnum::PointerValue(r)) => {
                match op {
                    BinOp::Add => {
                        let concat = self.get_or_build_concat()?;
                        let out = self
                            .builder
                            .build_call(concat, &[l.into(), r.into()], "concat")
                            .map_err(err)?
                            .try_as_basic_value()
                            .left()
                            .unwrap();
                        self.stmt_temps.push(out.into_pointer_value());
                        Ok(out)
                    }
                    BinOp::Eq | BinOp::Ne => {
                        let eq_fn = self.get_or_build_str_eq()?;
                        let eq = self
                            .builder
                            .build_call(eq_fn, &[l.into(), r.into()], "streq")
                            .map_err(err)?
                            .try_as_basic_value()
                            .left()
                            .unwrap()
                            .into_int_value();
                        let v = if op == BinOp::Ne {
                            self.builder.build_not(eq, "strne").map_err(err)?
                        } else {
                            eq
                        };
                        Ok(v.into())
                    }
                    _ => Err("codegen: unsupported string operation".into()),
                }
            }
            _ => Err("codegen: mismatched binary operand types".into()),
        }
    }

    fn compile_short_circuit(
        &mut self,
        lhs: &Expr,
        op: BinOp,
        rhs: &Expr,
    ) -> CResult<BasicValueEnum<'ctx>> {
        let function = self.current_function();
        // Release only temps created inside this short-circuit: both operands
        // are bool, so any string temps are dead once the i1 is computed.
        let checkpoint = self.stmt_temps.len();
        let l = self.compile_expr(lhs)?.into_int_value();
        self.flush_temps(checkpoint)?;
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

        // String temps created on a conditional path must be released on that
        // path — their SSA values don't dominate the merge block.
        self.builder.position_at_end(rhs_bb);
        let r = self.compile_expr(rhs)?.into_int_value();
        self.flush_temps(checkpoint)?;
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
        let mut result = call.try_as_basic_value().left();

        // ARC at the call boundary: Xia functions return +1 owned values; a
        // `str` from an extern C function is foreign memory with no refcount
        // header, so copy it into a fresh Xia string (null-safe).
        if let Some((is_extern, Type::Str)) = self.sigs.get(name).copied() {
            let mut ptr = result.unwrap().into_pointer_value();
            if is_extern {
                let dup = self.get_or_build_str_dup()?;
                ptr = self
                    .builder
                    .build_call(dup, &[ptr.into()], "dup")
                    .map_err(err)?
                    .try_as_basic_value()
                    .left()
                    .unwrap()
                    .into_pointer_value();
                result = Some(ptr.into());
            }
            self.stmt_temps.push(ptr);
        }
        Ok(result)
    }

    /// Compiler builtin: `print(x)` lowers to a libc `printf` call.
    fn compile_print(&mut self, arg: &Expr) -> CResult<()> {
        let v = self.compile_expr(arg)?;
        let (fmt, value): (&str, BasicMetadataValueEnum) = match arg.ty.unwrap() {
            Type::Int => ("%lld\n", v.into()),
            Type::Float => ("%g\n", v.into()),
            Type::Str => ("%s\n", v.into()),
            Type::Bool => {
                let t = self.str_literal("true")?;
                let f = self.str_literal("false")?;
                let sel = self
                    .builder
                    .build_select(v.into_int_value(), t, f, "boolstr")
                    .map_err(err)?;
                ("%s\n", sel.into())
            }
            Type::Unit => return Err("codegen: cannot print unit".into()),
        };
        let fmt_ptr = self.str_literal(fmt)?;
        let printf = self.libc_printf();
        self.builder
            .build_indirect_call(
                printf.1,
                printf.0,
                &[fmt_ptr.into(), value],
                "",
            )
            .map_err(err)?;
        Ok(())
    }

    // ----- statement temporaries ------------------------------------------

    /// Remove a heap value from the pending-release list because its
    /// ownership was transferred (into a binding or to the caller).
    fn consume_temp(&mut self, val: PointerValue<'ctx>) {
        if let Some(pos) = self.stmt_temps.iter().rposition(|t| *t == val) {
            self.stmt_temps.remove(pos);
        }
    }

    /// Release every pending temp created since `from` (a checkpoint index).
    fn flush_temps(&mut self, from: usize) -> CResult<()> {
        while self.stmt_temps.len() > from {
            let t = self.stmt_temps.pop().unwrap();
            self.emit_release(t)?;
        }
        Ok(())
    }

    // ----- ARC / string runtime, built directly in IR ----------------------

    fn ptr_ty(&self) -> inkwell::types::PointerType<'ctx> {
        self.context.ptr_type(AddressSpace::default())
    }

    /// Get-or-declare a libc function. If the user already declared the same
    /// symbol via `extern fn` with a different signature, calls go through a
    /// function pointer with our types so the two coexist at the IR level.
    fn libc(&self, name: &str, ty: FunctionType<'ctx>) -> (PointerValue<'ctx>, FunctionType<'ctx>) {
        let f = self
            .module
            .get_function(name)
            .unwrap_or_else(|| self.module.add_function(name, ty, None));
        (f.as_global_value().as_pointer_value(), ty)
    }

    fn libc_printf(&self) -> (PointerValue<'ctx>, FunctionType<'ctx>) {
        let ty = self
            .context
            .i32_type()
            .fn_type(&[self.ptr_ty().into()], true);
        self.libc("printf", ty)
    }

    fn emit_retain(&mut self, val: PointerValue<'ctx>) -> CResult<()> {
        let f = self.get_or_build_retain()?;
        self.builder.build_call(f, &[val.into()], "").map_err(err)?;
        Ok(())
    }

    fn emit_release(&mut self, val: PointerValue<'ctx>) -> CResult<()> {
        let f = self.get_or_build_release()?;
        self.builder.build_call(f, &[val.into()], "").map_err(err)?;
        Ok(())
    }

    /// Interned immortal string literal: a private constant global laid out
    /// as `{ i64 -1, [n+1 x i8] }` whose value pointer targets the bytes.
    fn str_literal(&mut self, s: &str) -> CResult<PointerValue<'ctx>> {
        if let Some(p) = self.str_literals.get(s) {
            return Ok(*p);
        }
        let i64_ty = self.context.i64_type();
        let data = self.context.const_string(s.as_bytes(), true);
        let init = self
            .context
            .const_struct(&[i64_ty.const_int(u64::MAX, true).into(), data.into()], false);
        let global = self.module.add_global(init.get_type(), None, "str.lit");
        global.set_initializer(&init);
        global.set_constant(true);
        global.set_linkage(inkwell::module::Linkage::Private);
        let zero = self.context.i32_type().const_zero();
        let one = self.context.i32_type().const_int(1, false);
        let ptr = unsafe {
            global
                .as_pointer_value()
                .const_gep(init.get_type(), &[zero, one])
        };
        self.str_literals.insert(s.to_string(), ptr);
        Ok(ptr)
    }

    /// Position-preserving helper: create function `name`, build its body
    /// with a dedicated builder, and return it.
    fn build_runtime_fn(
        &self,
        name: &str,
        ty: FunctionType<'ctx>,
        build: impl FnOnce(&Builder<'ctx>, FunctionValue<'ctx>) -> CResult<()>,
    ) -> CResult<FunctionValue<'ctx>> {
        if let Some(f) = self.module.get_function(name) {
            return Ok(f);
        }
        let f = self.module.add_function(name, ty, None);
        f.set_linkage(inkwell::module::Linkage::Internal);
        let b = self.context.create_builder();
        build(&b, f)?;
        Ok(f)
    }

    /// Pointer to the refcount header, `val - 8`.
    fn rc_ptr(
        &self,
        b: &Builder<'ctx>,
        val: PointerValue<'ctx>,
    ) -> CResult<PointerValue<'ctx>> {
        let off = self.context.i64_type().const_int((-RC_OFFSET) as u64, true);
        unsafe {
            b.build_gep(self.context.i8_type(), val, &[off], "rc.ptr")
                .map_err(err)
        }
    }

    fn get_or_build_retain(&mut self) -> CResult<FunctionValue<'ctx>> {
        let void = self.context.void_type();
        let ty = void.fn_type(&[self.ptr_ty().into()], false);
        let i64_ty = self.context.i64_type();
        let ctx = self.context;
        self.build_runtime_fn("xia_retain", ty, |b, f| {
            let p = f.get_nth_param(0).unwrap().into_pointer_value();
            let entry = ctx.append_basic_block(f, "entry");
            let check = ctx.append_basic_block(f, "check");
            let inc = ctx.append_basic_block(f, "inc");
            let done = ctx.append_basic_block(f, "done");

            b.position_at_end(entry);
            let is_null = b
                .build_is_null(p, "is_null")
                .map_err(err)?;
            b.build_conditional_branch(is_null, done, check).map_err(err)?;

            b.position_at_end(check);
            let rc_ptr = self.rc_ptr(b, p)?;
            let rc = b.build_load(i64_ty, rc_ptr, "rc").map_err(err)?.into_int_value();
            // Negative refcount = immortal literal; leave it alone.
            let immortal = b
                .build_int_compare(IntPredicate::SLT, rc, i64_ty.const_zero(), "imm")
                .map_err(err)?;
            b.build_conditional_branch(immortal, done, inc).map_err(err)?;

            b.position_at_end(inc);
            let bumped = b
                .build_int_add(rc, i64_ty.const_int(1, false), "rc.inc")
                .map_err(err)?;
            b.build_store(rc_ptr, bumped).map_err(err)?;
            b.build_unconditional_branch(done).map_err(err)?;

            b.position_at_end(done);
            b.build_return(None).map_err(err)?;
            Ok(())
        })
    }

    fn get_or_build_release(&mut self) -> CResult<FunctionValue<'ctx>> {
        let void = self.context.void_type();
        let ty = void.fn_type(&[self.ptr_ty().into()], false);
        let i64_ty = self.context.i64_type();
        let ctx = self.context;
        let free_ty = void.fn_type(&[self.ptr_ty().into()], false);
        let (free_ptr, free_fnty) = self.libc("free", free_ty);
        self.build_runtime_fn("xia_release", ty, |b, f| {
            let p = f.get_nth_param(0).unwrap().into_pointer_value();
            let entry = ctx.append_basic_block(f, "entry");
            let check = ctx.append_basic_block(f, "check");
            let dec = ctx.append_basic_block(f, "dec");
            let dead = ctx.append_basic_block(f, "free");
            let store = ctx.append_basic_block(f, "store");
            let done = ctx.append_basic_block(f, "done");

            b.position_at_end(entry);
            let is_null = b.build_is_null(p, "is_null").map_err(err)?;
            b.build_conditional_branch(is_null, done, check).map_err(err)?;

            b.position_at_end(check);
            let rc_ptr = self.rc_ptr(b, p)?;
            let rc = b.build_load(i64_ty, rc_ptr, "rc").map_err(err)?.into_int_value();
            let immortal = b
                .build_int_compare(IntPredicate::SLT, rc, i64_ty.const_zero(), "imm")
                .map_err(err)?;
            b.build_conditional_branch(immortal, done, dec).map_err(err)?;

            b.position_at_end(dec);
            let dropped = b
                .build_int_sub(rc, i64_ty.const_int(1, false), "rc.dec")
                .map_err(err)?;
            let is_zero = b
                .build_int_compare(IntPredicate::EQ, dropped, i64_ty.const_zero(), "zero")
                .map_err(err)?;
            b.build_conditional_branch(is_zero, dead, store).map_err(err)?;

            b.position_at_end(dead);
            // The malloc'd block starts at the refcount header.
            b.build_indirect_call(free_fnty, free_ptr, &[rc_ptr.into()], "")
                .map_err(err)?;
            b.build_unconditional_branch(done).map_err(err)?;

            b.position_at_end(store);
            b.build_store(rc_ptr, dropped).map_err(err)?;
            b.build_unconditional_branch(done).map_err(err)?;

            b.position_at_end(done);
            b.build_return(None).map_err(err)?;
            Ok(())
        })
    }

    /// `xia_alloc_str(len) -> ptr`: malloc a block for `len` bytes of payload
    /// (+ header + NUL), set refcount 1, return the payload pointer.
    fn get_or_build_alloc(&mut self) -> CResult<FunctionValue<'ctx>> {
        let i64_ty = self.context.i64_type();
        let ty = self.ptr_ty().fn_type(&[i64_ty.into()], false);
        let ctx = self.context;
        let malloc_ty = self.ptr_ty().fn_type(&[i64_ty.into()], false);
        let (malloc_ptr, malloc_fnty) = self.libc("malloc", malloc_ty);
        self.build_runtime_fn("xia_alloc_str", ty, |b, f| {
            let len = f.get_nth_param(0).unwrap().into_int_value();
            let entry = ctx.append_basic_block(f, "entry");
            b.position_at_end(entry);
            let header = i64_ty.const_int((RC_OFFSET + 1) as u64, false);
            let size = b.build_int_add(len, header, "size").map_err(err)?;
            let block = b
                .build_indirect_call(malloc_fnty, malloc_ptr, &[size.into()], "block")
                .map_err(err)?
                .try_as_basic_value()
                .left()
                .unwrap()
                .into_pointer_value();
            b.build_store(block, i64_ty.const_int(1, false)).map_err(err)?;
            let data = unsafe {
                b.build_gep(
                    ctx.i8_type(),
                    block,
                    &[i64_ty.const_int(RC_OFFSET as u64, false)],
                    "data",
                )
                .map_err(err)?
            };
            b.build_return(Some(&data)).map_err(err)?;
            Ok(())
        })
    }

    fn get_or_build_concat(&mut self) -> CResult<FunctionValue<'ctx>> {
        let i64_ty = self.context.i64_type();
        let i8_ty = self.context.i8_type();
        let ptr = self.ptr_ty();
        let ty = ptr.fn_type(&[ptr.into(), ptr.into()], false);
        let ctx = self.context;
        let strlen_ty = i64_ty.fn_type(&[ptr.into()], false);
        let (strlen_ptr, strlen_fnty) = self.libc("strlen", strlen_ty);
        let alloc = self.get_or_build_alloc()?;
        self.build_runtime_fn("xia_str_concat", ty, |b, f| {
            let a = f.get_nth_param(0).unwrap().into_pointer_value();
            let c = f.get_nth_param(1).unwrap().into_pointer_value();
            let entry = ctx.append_basic_block(f, "entry");
            b.position_at_end(entry);
            let la = b
                .build_indirect_call(strlen_fnty, strlen_ptr, &[a.into()], "la")
                .map_err(err)?
                .try_as_basic_value()
                .left()
                .unwrap()
                .into_int_value();
            let lc = b
                .build_indirect_call(strlen_fnty, strlen_ptr, &[c.into()], "lc")
                .map_err(err)?
                .try_as_basic_value()
                .left()
                .unwrap()
                .into_int_value();
            let total = b.build_int_add(la, lc, "total").map_err(err)?;
            let data = b
                .build_call(alloc, &[total.into()], "data")
                .map_err(err)?
                .try_as_basic_value()
                .left()
                .unwrap()
                .into_pointer_value();
            b.build_memcpy(data, 1, a, 1, la).map_err(err)?;
            let second = unsafe {
                b.build_gep(i8_ty, data, &[la], "second").map_err(err)?
            };
            b.build_memcpy(second, 1, c, 1, lc).map_err(err)?;
            let end = unsafe {
                b.build_gep(i8_ty, second, &[lc], "end").map_err(err)?
            };
            b.build_store(end, i8_ty.const_zero()).map_err(err)?;
            b.build_return(Some(&data)).map_err(err)?;
            Ok(())
        })
    }

    /// Copy a foreign C string into a fresh Xia string (null-safe).
    fn get_or_build_str_dup(&mut self) -> CResult<FunctionValue<'ctx>> {
        let i64_ty = self.context.i64_type();
        let ptr = self.ptr_ty();
        let ty = ptr.fn_type(&[ptr.into()], false);
        let ctx = self.context;
        let strlen_ty = i64_ty.fn_type(&[ptr.into()], false);
        let (strlen_ptr, strlen_fnty) = self.libc("strlen", strlen_ty);
        let alloc = self.get_or_build_alloc()?;
        self.build_runtime_fn("xia_str_dup", ty, |b, f| {
            let p = f.get_nth_param(0).unwrap().into_pointer_value();
            let entry = ctx.append_basic_block(f, "entry");
            let copy = ctx.append_basic_block(f, "copy");
            let null = ctx.append_basic_block(f, "null");
            b.position_at_end(entry);
            let is_null = b.build_is_null(p, "is_null").map_err(err)?;
            b.build_conditional_branch(is_null, null, copy).map_err(err)?;

            b.position_at_end(null);
            b.build_return(Some(&ptr.const_null())).map_err(err)?;

            b.position_at_end(copy);
            let len = b
                .build_indirect_call(strlen_fnty, strlen_ptr, &[p.into()], "len")
                .map_err(err)?
                .try_as_basic_value()
                .left()
                .unwrap()
                .into_int_value();
            let data = b
                .build_call(alloc, &[len.into()], "data")
                .map_err(err)?
                .try_as_basic_value()
                .left()
                .unwrap()
                .into_pointer_value();
            let with_nul = b
                .build_int_add(len, i64_ty.const_int(1, false), "with_nul")
                .map_err(err)?;
            b.build_memcpy(data, 1, p, 1, with_nul).map_err(err)?;
            b.build_return(Some(&data)).map_err(err)?;
            Ok(())
        })
    }

    fn get_or_build_str_eq(&mut self) -> CResult<FunctionValue<'ctx>> {
        let ptr = self.ptr_ty();
        let ty = self.context.bool_type().fn_type(&[ptr.into(), ptr.into()], false);
        let ctx = self.context;
        let strcmp_ty = self.context.i32_type().fn_type(&[ptr.into(), ptr.into()], false);
        let (strcmp_ptr, strcmp_fnty) = self.libc("strcmp", strcmp_ty);
        self.build_runtime_fn("xia_str_eq", ty, |b, f| {
            let a = f.get_nth_param(0).unwrap().into_pointer_value();
            let c = f.get_nth_param(1).unwrap().into_pointer_value();
            let entry = ctx.append_basic_block(f, "entry");
            b.position_at_end(entry);
            let cmp = b
                .build_indirect_call(strcmp_fnty, strcmp_ptr, &[a.into(), c.into()], "cmp")
                .map_err(err)?
                .try_as_basic_value()
                .left()
                .unwrap()
                .into_int_value();
            let eq = b
                .build_int_compare(IntPredicate::EQ, cmp, ctx.i32_type().const_zero(), "eq")
                .map_err(err)?;
            b.build_return(Some(&eq)).map_err(err)?;
            Ok(())
        })
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
    fn string_program_emits_arc_runtime() {
        let src = "fn greet(name: str) -> str:\n    return \"hello, \" + name\nfn main() -> int:\n    let g = greet(\"world\")\n    print(g)\n    return 0\n";
        let ir = compile_to_ir(src).unwrap();
        assert!(ir.contains("define internal void @xia_retain"));
        assert!(ir.contains("define internal void @xia_release"));
        assert!(ir.contains("define internal ptr @xia_str_concat"));
        assert!(ir.contains("declare ptr @malloc"));
        assert!(ir.contains("declare void @free"));
        assert!(ir.contains("@printf"));
    }

    #[test]
    fn string_literals_are_immortal_constants() {
        let ir = compile_to_ir("fn main():\n    print(\"hi\")\n").unwrap();
        assert!(ir.contains("i64 -1"), "literal header must be -1 (immortal)");
        assert!(ir.contains("c\"hi\\00\""));
    }

    #[test]
    fn str_equality_uses_strcmp() {
        let src = "fn f(a: str, b: str) -> bool:\n    return a == b\n";
        let ir = compile_to_ir(src).unwrap();
        assert!(ir.contains("xia_str_eq"));
        assert!(ir.contains("@strcmp"));
    }

    #[test]
    fn discarded_concat_is_released() {
        let src = "fn main():\n    let a = \"x\"\n    let b = a + \"y\"\n    print(b)\n";
        let ir = compile_to_ir(src).unwrap();
        assert!(ir.contains("call void @xia_release"));
    }

    #[test]
    fn extern_str_result_is_duplicated() {
        let src = "extern fn getenv(name: str) -> str\nfn main():\n    let p = getenv(\"PATH\")\n    print(p)\n";
        let ir = compile_to_ir(src).unwrap();
        assert!(ir.contains("xia_str_dup"));
    }

    #[test]
    fn float_ops_lower_to_float_instructions() {
        let src = "fn f(x: float) -> float:\n    return x * 2.0 + 0.5\n";
        let ir = compile_to_ir(src).unwrap();
        assert!(ir.contains("fmul double"));
        assert!(ir.contains("fadd double"));
    }
}
