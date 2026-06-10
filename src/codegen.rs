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

/// Heap memory layout. Every reference-counted block carries a 16-byte
/// header `[i64 kind][i64 refcount]` and the value points just past it:
///
/// - string (kind 0): `[kind][rc][bytes...][NUL]` — the value points at the
///   bytes, so every Xia `str` doubles as a `char*` for the C FFI.
/// - array (kind 1 plain / kind 2 heap elements):
///   `[kind][rc][i64 len][i64 cap][ptr data]` — the value points at `len`;
///   elements live in a separately allocated buffer of 8-byte words.
///
/// A negative refcount marks an immortal value (string literals live in
/// constant globals and are never freed). `xia_retain`/`xia_release` are
/// null-safe and skip immortals. Releasing a kind-2 array to zero releases
/// each element before freeing the buffer and the block.
const RC_OFFSET: i64 = 8;
const HEADER_SIZE: i64 = 16;
const KIND_STR: u64 = 0;
const KIND_ARR: u64 = 1;
const KIND_ARR_HEAP: u64 = 2;
/// Array field offsets relative to the value pointer.
const ARR_CAP_OFFSET: u64 = 8;
const ARR_DATA_OFFSET: u64 = 16;

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
            Type::Array(_) => self.context.ptr_type(AddressSpace::default()).into(),
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
            Stmt::IndexAssign { target, index, value, .. } => {
                let Some(Type::Array(elem)) = target.ty else {
                    return Err("codegen: index-assign target is not an array".into());
                };
                let arr = self.compile_expr(target)?.into_pointer_value();
                let idx = self.compile_expr(index)?.into_int_value();
                let v = self.compile_expr(value)?;
                let w = self.to_word(v, elem)?;
                // xia_arr_set retains the new element / releases the old one.
                let set_fn = self.get_or_build_arr_set()?;
                self.builder
                    .build_call(set_fn, &[arr.into(), idx.into(), w.into()], "")
                    .map_err(err)?;
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
            Stmt::For { var, start, end, body, .. } => {
                let function = self.current_function();
                let i64_ty = self.context.i64_type();

                // Bounds are evaluated once, before the loop.
                let start_v = self.compile_expr(start)?.into_int_value();
                let end_v = self.compile_expr(end)?.into_int_value();
                self.flush_temps(0)?;

                let slot = self.builder.build_alloca(i64_ty, var).map_err(err)?;
                self.builder.build_store(slot, start_v).map_err(err)?;
                self.variables.push(HashMap::new());
                self.variables
                    .last_mut()
                    .unwrap()
                    .insert(var.clone(), (slot, Type::Int));

                let cond_bb = self.context.append_basic_block(function, "for.cond");
                let body_bb = self.context.append_basic_block(function, "for.body");
                let inc_bb = self.context.append_basic_block(function, "for.inc");
                let after_bb = self.context.append_basic_block(function, "for.end");

                self.builder.build_unconditional_branch(cond_bb).map_err(err)?;
                self.builder.position_at_end(cond_bb);
                let cur = self
                    .builder
                    .build_load(i64_ty, slot, var)
                    .map_err(err)?
                    .into_int_value();
                let in_range = self
                    .builder
                    .build_int_compare(IntPredicate::SLT, cur, end_v, "for.lt")
                    .map_err(err)?;
                self.builder
                    .build_conditional_branch(in_range, body_bb, after_bb)
                    .map_err(err)?;

                // `continue` must run the increment, so it targets inc_bb.
                self.builder.position_at_end(body_bb);
                self.loop_stack.push((inc_bb, after_bb));
                self.compile_block(body)?;
                self.loop_stack.pop();
                if !self.block_terminated() {
                    self.builder.build_unconditional_branch(inc_bb).map_err(err)?;
                }

                self.builder.position_at_end(inc_bb);
                let cur = self
                    .builder
                    .build_load(i64_ty, slot, var)
                    .map_err(err)?
                    .into_int_value();
                let next = self
                    .builder
                    .build_int_add(cur, i64_ty.const_int(1, false), "for.next")
                    .map_err(err)?;
                self.builder.build_store(slot, next).map_err(err)?;
                self.builder.build_unconditional_branch(cond_bb).map_err(err)?;

                self.variables.pop();
                self.builder.position_at_end(after_bb);
                Ok(())
            }
            Stmt::ForEach { var, iterable, body, .. } => {
                let Some(Type::Array(elem)) = iterable.ty else {
                    return Err("codegen: for-in over a non-array".into());
                };
                let function = self.current_function();
                let i64_ty = self.context.i64_type();

                // A fresh iterable (literal, call result) must stay alive for
                // the whole loop; pull it out of the statement temps and
                // release it after the loop instead.
                let checkpoint = self.stmt_temps.len();
                let arr = self.compile_expr(iterable)?.into_pointer_value();
                let arr_is_temp = self.stmt_temps.last() == Some(&arr);
                if arr_is_temp {
                    self.consume_temp(arr);
                }
                self.flush_temps(checkpoint)?;

                let idx_slot = self.builder.build_alloca(i64_ty, "foreach.idx").map_err(err)?;
                self.builder.build_store(idx_slot, i64_ty.const_zero()).map_err(err)?;
                let var_ty = elem.to_type();
                let var_slot = self
                    .builder
                    .build_alloca(self.basic_type(var_ty), var)
                    .map_err(err)?;
                self.variables.push(HashMap::new());
                self.variables
                    .last_mut()
                    .unwrap()
                    .insert(var.clone(), (var_slot, var_ty));

                let cond_bb = self.context.append_basic_block(function, "foreach.cond");
                let body_bb = self.context.append_basic_block(function, "foreach.body");
                let inc_bb = self.context.append_basic_block(function, "foreach.inc");
                let after_bb = self.context.append_basic_block(function, "foreach.end");

                self.builder.build_unconditional_branch(cond_bb).map_err(err)?;
                self.builder.position_at_end(cond_bb);
                let idx = self
                    .builder
                    .build_load(i64_ty, idx_slot, "idx")
                    .map_err(err)?
                    .into_int_value();
                // Reload the length every iteration: the body may push.
                let len = self
                    .builder
                    .build_load(i64_ty, arr, "len")
                    .map_err(err)?
                    .into_int_value();
                let more = self
                    .builder
                    .build_int_compare(IntPredicate::SLT, idx, len, "foreach.lt")
                    .map_err(err)?;
                self.builder
                    .build_conditional_branch(more, body_bb, after_bb)
                    .map_err(err)?;

                self.builder.position_at_end(body_bb);
                let idx = self
                    .builder
                    .build_load(i64_ty, idx_slot, "idx")
                    .map_err(err)?
                    .into_int_value();
                let get_fn = self.get_or_build_arr_get()?;
                let raw = self
                    .builder
                    .build_call(get_fn, &[arr.into(), idx.into()], "elem")
                    .map_err(err)?
                    .try_as_basic_value()
                    .left()
                    .unwrap()
                    .into_int_value();
                let v = self.from_word(raw, elem)?;
                self.builder.build_store(var_slot, v).map_err(err)?;
                // The ARC pass owns `var` in the loop scope and releases it
                // at the end of each iteration (and on break/continue/return).
                self.loop_stack.push((inc_bb, after_bb));
                self.compile_block(body)?;
                self.loop_stack.pop();
                if !self.block_terminated() {
                    self.builder.build_unconditional_branch(inc_bb).map_err(err)?;
                }

                self.builder.position_at_end(inc_bb);
                let idx = self
                    .builder
                    .build_load(i64_ty, idx_slot, "idx")
                    .map_err(err)?
                    .into_int_value();
                let next = self
                    .builder
                    .build_int_add(idx, i64_ty.const_int(1, false), "foreach.next")
                    .map_err(err)?;
                self.builder.build_store(idx_slot, next).map_err(err)?;
                self.builder.build_unconditional_branch(cond_bb).map_err(err)?;

                self.variables.pop();
                self.builder.position_at_end(after_bb);
                if arr_is_temp {
                    self.emit_release(arr)?;
                }
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
                self.compile_call(name, args, e.span.line)?;
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
                self.compile_call(name, args, e.span.line)?
                    .ok_or_else(|| format!("codegen: `{name}` returns no value (line {})", e.span.line))
            }
            ExprKind::ArrayLit(elems) => {
                let Some(Type::Array(elem)) = e.ty else {
                    return Err("codegen: array literal without array type".into());
                };
                let i64_ty = self.context.i64_type();
                let kind = if elem.to_type().is_heap() { KIND_ARR_HEAP } else { KIND_ARR };
                let cap = (elems.len() as u64).max(4);
                let new_fn = self.get_or_build_arr_new()?;
                let arr = self
                    .builder
                    .build_call(
                        new_fn,
                        &[
                            i64_ty.const_int(kind, false).into(),
                            i64_ty.const_int(cap, false).into(),
                        ],
                        "arr",
                    )
                    .map_err(err)?
                    .try_as_basic_value()
                    .left()
                    .unwrap()
                    .into_pointer_value();
                let push_fn = self.get_or_build_arr_push()?;
                for el in elems {
                    let v = self.compile_expr(el)?;
                    let w = self.to_word(v, elem)?;
                    self.builder
                        .build_call(push_fn, &[arr.into(), w.into()], "")
                        .map_err(err)?;
                }
                self.stmt_temps.push(arr);
                Ok(arr.into())
            }
            ExprKind::Index(base, index) => {
                let Some(Type::Array(elem)) = base.ty else {
                    return Err("codegen: indexing a non-array".into());
                };
                let arr = self.compile_expr(base)?.into_pointer_value();
                let idx = self.compile_expr(index)?.into_int_value();
                let get_fn = self.get_or_build_arr_get()?;
                let raw = self
                    .builder
                    .build_call(get_fn, &[arr.into(), idx.into()], "elem")
                    .map_err(err)?
                    .try_as_basic_value()
                    .left()
                    .unwrap()
                    .into_int_value();
                let v = self.from_word(raw, elem)?;
                if elem.to_type().is_heap() {
                    // xia_arr_get returns heap elements retained (+1).
                    self.stmt_temps.push(v.into_pointer_value());
                }
                Ok(v)
            }
        }
    }

    /// Pack a value into the 8-byte word stored in an array buffer.
    fn to_word(
        &mut self,
        v: BasicValueEnum<'ctx>,
        elem: ElemType,
    ) -> CResult<inkwell::values::IntValue<'ctx>> {
        let i64_ty = self.context.i64_type();
        match elem {
            ElemType::Int => Ok(v.into_int_value()),
            ElemType::Float => Ok(self
                .builder
                .build_bit_cast(v.into_float_value(), i64_ty, "fbits")
                .map_err(err)?
                .into_int_value()),
            ElemType::Bool => self
                .builder
                .build_int_z_extend(v.into_int_value(), i64_ty, "bword")
                .map_err(err),
            ElemType::Str => self
                .builder
                .build_ptr_to_int(v.into_pointer_value(), i64_ty, "pword")
                .map_err(err),
        }
    }

    /// Unpack an 8-byte array word back into a typed value.
    fn from_word(
        &mut self,
        w: inkwell::values::IntValue<'ctx>,
        elem: ElemType,
    ) -> CResult<BasicValueEnum<'ctx>> {
        match elem {
            ElemType::Int => Ok(w.into()),
            ElemType::Float => Ok(self
                .builder
                .build_bit_cast(w, self.context.f64_type(), "fval")
                .map_err(err)?),
            ElemType::Bool => Ok(self
                .builder
                .build_int_truncate(w, self.context.bool_type(), "bval")
                .map_err(err)?
                .into()),
            ElemType::Str => Ok(self
                .builder
                .build_int_to_ptr(w, self.ptr_ty(), "pval")
                .map_err(err)?
                .into()),
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
        if name == "str" {
            let arg = &args[0];
            let v = self.compile_expr(arg)?;
            let out = match arg.ty.unwrap() {
                Type::Int => {
                    let f = self.get_or_build_to_str(
                        "xia_int_to_str",
                        "%lld",
                        self.context.i64_type().into(),
                    )?;
                    self.builder
                        .build_call(f, &[v.into()], "str")
                        .map_err(err)?
                        .try_as_basic_value()
                        .left()
                        .unwrap()
                        .into_pointer_value()
                }
                Type::Float => {
                    let f = self.get_or_build_to_str(
                        "xia_float_to_str",
                        "%g",
                        self.context.f64_type().into(),
                    )?;
                    self.builder
                        .build_call(f, &[v.into()], "str")
                        .map_err(err)?
                        .try_as_basic_value()
                        .left()
                        .unwrap()
                        .into_pointer_value()
                }
                Type::Bool => {
                    // Immortal literals: retain/release are no-ops on them,
                    // so the select result is safely treated as owned.
                    let t = self.str_literal("true")?;
                    let f = self.str_literal("false")?;
                    self.builder
                        .build_select(v.into_int_value(), t, f, "boolstr")
                        .map_err(err)?
                        .into_pointer_value()
                }
                Type::Str => {
                    let dup = self.get_or_build_str_dup()?;
                    self.builder
                        .build_call(dup, &[v.into()], "str")
                        .map_err(err)?
                        .try_as_basic_value()
                        .left()
                        .unwrap()
                        .into_pointer_value()
                }
                other => return Err(format!("codegen: str() of {other}")),
            };
            self.stmt_temps.push(out);
            return Ok(Some(out.into()));
        }
        if name == "len" {
            let arg = &args[0];
            let v = self.compile_expr(arg)?;
            let out = match arg.ty.unwrap() {
                Type::Str => {
                    let i64_ty = self.context.i64_type();
                    let strlen_ty = i64_ty.fn_type(&[self.ptr_ty().into()], false);
                    let (strlen_ptr, strlen_fnty) = self.libc("strlen", strlen_ty);
                    self.builder
                        .build_indirect_call(strlen_fnty, strlen_ptr, &[v.into()], "len")
                        .map_err(err)?
                        .try_as_basic_value()
                        .left()
                        .unwrap()
                }
                Type::Array(_) => {
                    // The array value points directly at its length field.
                    self.builder
                        .build_load(self.context.i64_type(), v.into_pointer_value(), "len")
                        .map_err(err)?
                }
                other => return Err(format!("codegen: len of {other}")),
            };
            return Ok(Some(out));
        }
        if name == "push" {
            let Some(Type::Array(elem)) = args[0].ty else {
                return Err("codegen: push to a non-array".into());
            };
            let arr = self.compile_expr(&args[0])?.into_pointer_value();
            let v = self.compile_expr(&args[1])?;
            let w = self.to_word(v, elem)?;
            // xia_arr_push retains heap elements; a fresh temp's own +1 is
            // then dropped by the statement flush, leaving the array's ref.
            let push_fn = self.get_or_build_arr_push()?;
            self.builder
                .build_call(push_fn, &[arr.into(), w.into()], "")
                .map_err(err)?;
            return Ok(None);
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
            Type::Array(_) | Type::Unit => {
                return Err("codegen: cannot print this type".into());
            }
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
    /// as `{ i64 kind, i64 -1, [n+1 x i8] }` whose value pointer targets the
    /// bytes.
    fn str_literal(&mut self, s: &str) -> CResult<PointerValue<'ctx>> {
        if let Some(p) = self.str_literals.get(s) {
            return Ok(*p);
        }
        let i64_ty = self.context.i64_type();
        let data = self.context.const_string(s.as_bytes(), true);
        let init = self.context.const_struct(
            &[
                i64_ty.const_int(KIND_STR, false).into(),
                i64_ty.const_int(u64::MAX, true).into(),
                data.into(),
            ],
            false,
        );
        let global = self.module.add_global(init.get_type(), None, "str.lit");
        global.set_initializer(&init);
        global.set_constant(true);
        global.set_linkage(inkwell::module::Linkage::Private);
        let zero = self.context.i32_type().const_zero();
        let two = self.context.i32_type().const_int(2, false);
        let ptr = unsafe {
            global
                .as_pointer_value()
                .const_gep(init.get_type(), &[zero, two])
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

    /// Pointer to the start of the block (the kind word), `val - 16`.
    fn block_ptr(
        &self,
        b: &Builder<'ctx>,
        val: PointerValue<'ctx>,
    ) -> CResult<PointerValue<'ctx>> {
        let off = self
            .context
            .i64_type()
            .const_int((-HEADER_SIZE) as u64, true);
        unsafe {
            b.build_gep(self.context.i8_type(), val, &[off], "block.ptr")
                .map_err(err)
        }
    }

    /// Pointer to a field of an array value at byte offset `off`.
    fn arr_field(
        &self,
        b: &Builder<'ctx>,
        val: PointerValue<'ctx>,
        off: u64,
        name: &str,
    ) -> CResult<PointerValue<'ctx>> {
        let off = self.context.i64_type().const_int(off, false);
        unsafe {
            b.build_gep(self.context.i8_type(), val, &[off], name)
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
        let ptr_ty = self.ptr_ty();
        let ctx = self.context;
        let free_ty = void.fn_type(&[self.ptr_ty().into()], false);
        let (free_ptr, free_fnty) = self.libc("free", free_ty);
        self.build_runtime_fn("xia_release", ty, |b, f| {
            let p = f.get_nth_param(0).unwrap().into_pointer_value();
            let entry = ctx.append_basic_block(f, "entry");
            let check = ctx.append_basic_block(f, "check");
            let dec = ctx.append_basic_block(f, "dec");
            let dead = ctx.append_basic_block(f, "dead");
            let arr = ctx.append_basic_block(f, "arr");
            let elem_loop = ctx.append_basic_block(f, "elem.loop");
            let elem_body = ctx.append_basic_block(f, "elem.body");
            let free_data = ctx.append_basic_block(f, "free.data");
            let free_block = ctx.append_basic_block(f, "free.block");
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

            // Dead value: arrays free their element buffer (releasing each
            // element first if they are heap values); then the block goes.
            b.position_at_end(dead);
            let block = self.block_ptr(b, p)?;
            let kind = b.build_load(i64_ty, block, "kind").map_err(err)?.into_int_value();
            let is_arr = b
                .build_int_compare(
                    IntPredicate::NE,
                    kind,
                    i64_ty.const_int(KIND_STR, false),
                    "is_arr",
                )
                .map_err(err)?;
            b.build_conditional_branch(is_arr, arr, free_block).map_err(err)?;

            b.position_at_end(arr);
            let len = b.build_load(i64_ty, p, "len").map_err(err)?.into_int_value();
            let data_slot = self.arr_field(b, p, ARR_DATA_OFFSET, "data.slot")?;
            let data = b
                .build_load(ptr_ty, data_slot, "data")
                .map_err(err)?
                .into_pointer_value();
            let heap_elems = b
                .build_int_compare(
                    IntPredicate::EQ,
                    kind,
                    i64_ty.const_int(KIND_ARR_HEAP, false),
                    "heap_elems",
                )
                .map_err(err)?;
            b.build_conditional_branch(heap_elems, elem_loop, free_data)
                .map_err(err)?;

            b.position_at_end(elem_loop);
            let i = b.build_phi(i64_ty, "i").map_err(err)?;
            let iv = i.as_basic_value().into_int_value();
            let more = b
                .build_int_compare(IntPredicate::SLT, iv, len, "more")
                .map_err(err)?;
            b.build_conditional_branch(more, elem_body, free_data)
                .map_err(err)?;

            b.position_at_end(elem_body);
            let slot = unsafe {
                b.build_gep(i64_ty, data, &[iv], "elem.slot").map_err(err)?
            };
            let word = b.build_load(i64_ty, slot, "elem").map_err(err)?.into_int_value();
            let elem_ptr = b.build_int_to_ptr(word, ptr_ty, "elem.ptr").map_err(err)?;
            // Recursive: nested heap values release through the same path.
            b.build_call(f, &[elem_ptr.into()], "").map_err(err)?;
            let next = b
                .build_int_add(iv, i64_ty.const_int(1, false), "i.next")
                .map_err(err)?;
            i.add_incoming(&[(&i64_ty.const_zero(), arr), (&next, elem_body)]);
            b.build_unconditional_branch(elem_loop).map_err(err)?;

            b.position_at_end(free_data);
            b.build_indirect_call(free_fnty, free_ptr, &[data.into()], "")
                .map_err(err)?;
            b.build_unconditional_branch(free_block).map_err(err)?;

            b.position_at_end(free_block);
            b.build_indirect_call(free_fnty, free_ptr, &[block.into()], "")
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
            let header = i64_ty.const_int((HEADER_SIZE + 1) as u64, false);
            let size = b.build_int_add(len, header, "size").map_err(err)?;
            let block = b
                .build_indirect_call(malloc_fnty, malloc_ptr, &[size.into()], "block")
                .map_err(err)?
                .try_as_basic_value()
                .left()
                .unwrap()
                .into_pointer_value();
            b.build_store(block, i64_ty.const_int(KIND_STR, false)).map_err(err)?;
            let rc_slot = unsafe {
                b.build_gep(
                    ctx.i8_type(),
                    block,
                    &[i64_ty.const_int(RC_OFFSET as u64, false)],
                    "rc.slot",
                )
                .map_err(err)?
            };
            b.build_store(rc_slot, i64_ty.const_int(1, false)).map_err(err)?;
            let data = unsafe {
                b.build_gep(
                    ctx.i8_type(),
                    block,
                    &[i64_ty.const_int(HEADER_SIZE as u64, false)],
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

    /// Number-to-string via two-pass snprintf: measure, allocate, format.
    fn get_or_build_to_str(
        &mut self,
        fn_name: &str,
        fmt: &str,
        param_ty: BasicMetadataTypeEnum<'ctx>,
    ) -> CResult<FunctionValue<'ctx>> {
        let i64_ty = self.context.i64_type();
        let ptr_ty = self.ptr_ty();
        let ty = ptr_ty.fn_type(&[param_ty], false);
        let ctx = self.context;
        let fmt_ptr = self.str_literal(fmt)?;
        let snprintf_ty = self
            .context
            .i32_type()
            .fn_type(&[ptr_ty.into(), i64_ty.into(), ptr_ty.into()], true);
        let (snp_ptr, snp_fnty) = self.libc("snprintf", snprintf_ty);
        let alloc = self.get_or_build_alloc()?;
        self.build_runtime_fn(fn_name, ty, |b, f| {
            let v = f.get_nth_param(0).unwrap();
            let entry = ctx.append_basic_block(f, "entry");
            b.position_at_end(entry);
            let len32 = b
                .build_indirect_call(
                    snp_fnty,
                    snp_ptr,
                    &[ptr_ty.const_null().into(), i64_ty.const_zero().into(), fmt_ptr.into(), v.into()],
                    "len32",
                )
                .map_err(err)?
                .try_as_basic_value()
                .left()
                .unwrap()
                .into_int_value();
            let len = b.build_int_s_extend(len32, i64_ty, "len").map_err(err)?;
            let data = b
                .build_call(alloc, &[len.into()], "data")
                .map_err(err)?
                .try_as_basic_value()
                .left()
                .unwrap()
                .into_pointer_value();
            let cap = b
                .build_int_add(len, i64_ty.const_int(1, false), "cap")
                .map_err(err)?;
            b.build_indirect_call(
                snp_fnty,
                snp_ptr,
                &[data.into(), cap.into(), fmt_ptr.into(), v.into()],
                "",
            )
            .map_err(err)?;
            b.build_return(Some(&data)).map_err(err)?;
            Ok(())
        })
    }

    // ----- array runtime ----------------------------------------------------

    /// `xia_arr_new(kind, cap) -> ptr`: allocate the handle block and an
    /// element buffer of `cap` 8-byte words; len starts at 0, refcount at 1.
    fn get_or_build_arr_new(&mut self) -> CResult<FunctionValue<'ctx>> {
        let i64_ty = self.context.i64_type();
        let ty = self
            .ptr_ty()
            .fn_type(&[i64_ty.into(), i64_ty.into()], false);
        let ctx = self.context;
        let malloc_ty = self.ptr_ty().fn_type(&[i64_ty.into()], false);
        let (malloc_ptr, malloc_fnty) = self.libc("malloc", malloc_ty);
        self.build_runtime_fn("xia_arr_new", ty, |b, f| {
            let kind = f.get_nth_param(0).unwrap().into_int_value();
            let cap = f.get_nth_param(1).unwrap().into_int_value();
            let entry = ctx.append_basic_block(f, "entry");
            b.position_at_end(entry);
            // [kind][rc][len][cap][data ptr] = 40 bytes
            let block = b
                .build_indirect_call(
                    malloc_fnty,
                    malloc_ptr,
                    &[i64_ty.const_int(40, false).into()],
                    "block",
                )
                .map_err(err)?
                .try_as_basic_value()
                .left()
                .unwrap()
                .into_pointer_value();
            b.build_store(block, kind).map_err(err)?;
            let rc_slot = self.arr_field(b, block, RC_OFFSET as u64, "rc.slot")?;
            b.build_store(rc_slot, i64_ty.const_int(1, false)).map_err(err)?;
            let val = self.arr_field(b, block, HEADER_SIZE as u64, "val")?;
            b.build_store(val, i64_ty.const_zero()).map_err(err)?;
            let cap_slot = self.arr_field(b, val, ARR_CAP_OFFSET, "cap.slot")?;
            b.build_store(cap_slot, cap).map_err(err)?;
            let bytes = b
                .build_int_mul(cap, i64_ty.const_int(8, false), "bytes")
                .map_err(err)?;
            let data = b
                .build_indirect_call(malloc_fnty, malloc_ptr, &[bytes.into()], "data")
                .map_err(err)?
                .try_as_basic_value()
                .left()
                .unwrap()
                .into_pointer_value();
            let data_slot = self.arr_field(b, val, ARR_DATA_OFFSET, "data.slot")?;
            b.build_store(data_slot, data).map_err(err)?;
            b.build_return(Some(&val)).map_err(err)?;
            Ok(())
        })
    }

    /// `xia_arr_push(arr, word)`: append, doubling the buffer when full.
    /// Heap elements are retained — the array owns its own reference.
    fn get_or_build_arr_push(&mut self) -> CResult<FunctionValue<'ctx>> {
        let i64_ty = self.context.i64_type();
        let ptr_ty = self.ptr_ty();
        let void = self.context.void_type();
        let ty = void.fn_type(&[ptr_ty.into(), i64_ty.into()], false);
        let ctx = self.context;
        let malloc_ty = ptr_ty.fn_type(&[i64_ty.into()], false);
        let (malloc_ptr, malloc_fnty) = self.libc("malloc", malloc_ty);
        let free_ty = void.fn_type(&[ptr_ty.into()], false);
        let (free_ptr, free_fnty) = self.libc("free", free_ty);
        let retain = self.get_or_build_retain()?;
        self.build_runtime_fn("xia_arr_push", ty, |b, f| {
            let arr = f.get_nth_param(0).unwrap().into_pointer_value();
            let val = f.get_nth_param(1).unwrap().into_int_value();
            let entry = ctx.append_basic_block(f, "entry");
            let grow = ctx.append_basic_block(f, "grow");
            let store = ctx.append_basic_block(f, "store");

            b.position_at_end(entry);
            let len = b.build_load(i64_ty, arr, "len").map_err(err)?.into_int_value();
            let cap_slot = self.arr_field(b, arr, ARR_CAP_OFFSET, "cap.slot")?;
            let cap = b.build_load(i64_ty, cap_slot, "cap").map_err(err)?.into_int_value();
            let full = b
                .build_int_compare(IntPredicate::EQ, len, cap, "full")
                .map_err(err)?;
            b.build_conditional_branch(full, grow, store).map_err(err)?;

            b.position_at_end(grow);
            let newcap = b
                .build_int_mul(cap, i64_ty.const_int(2, false), "newcap")
                .map_err(err)?;
            let newbytes = b
                .build_int_mul(newcap, i64_ty.const_int(8, false), "newbytes")
                .map_err(err)?;
            let newdata = b
                .build_indirect_call(malloc_fnty, malloc_ptr, &[newbytes.into()], "newdata")
                .map_err(err)?
                .try_as_basic_value()
                .left()
                .unwrap()
                .into_pointer_value();
            let data_slot = self.arr_field(b, arr, ARR_DATA_OFFSET, "data.slot")?;
            let olddata = b
                .build_load(ptr_ty, data_slot, "olddata")
                .map_err(err)?
                .into_pointer_value();
            let used = b
                .build_int_mul(len, i64_ty.const_int(8, false), "used")
                .map_err(err)?;
            b.build_memcpy(newdata, 8, olddata, 8, used).map_err(err)?;
            b.build_indirect_call(free_fnty, free_ptr, &[olddata.into()], "")
                .map_err(err)?;
            b.build_store(data_slot, newdata).map_err(err)?;
            b.build_store(cap_slot, newcap).map_err(err)?;
            b.build_unconditional_branch(store).map_err(err)?;

            b.position_at_end(store);
            let block = self.block_ptr(b, arr)?;
            let kind = b.build_load(i64_ty, block, "kind").map_err(err)?.into_int_value();
            let heap = b
                .build_int_compare(
                    IntPredicate::EQ,
                    kind,
                    i64_ty.const_int(KIND_ARR_HEAP, false),
                    "heap",
                )
                .map_err(err)?;
            let retain_bb = ctx.append_basic_block(f, "retain");
            let write = ctx.append_basic_block(f, "write");
            b.build_conditional_branch(heap, retain_bb, write).map_err(err)?;

            b.position_at_end(retain_bb);
            let as_ptr = b.build_int_to_ptr(val, ptr_ty, "val.ptr").map_err(err)?;
            b.build_call(retain, &[as_ptr.into()], "").map_err(err)?;
            b.build_unconditional_branch(write).map_err(err)?;

            b.position_at_end(write);
            let data_slot = self.arr_field(b, arr, ARR_DATA_OFFSET, "data.slot")?;
            let data = b
                .build_load(ptr_ty, data_slot, "data")
                .map_err(err)?
                .into_pointer_value();
            let slot = unsafe {
                b.build_gep(i64_ty, data, &[len], "slot").map_err(err)?
            };
            b.build_store(slot, val).map_err(err)?;
            let newlen = b
                .build_int_add(len, i64_ty.const_int(1, false), "newlen")
                .map_err(err)?;
            b.build_store(arr, newlen).map_err(err)?;
            b.build_return(None).map_err(err)?;
            Ok(())
        })
    }

    /// `xia_bounds_fail(idx, len)`: print a diagnostic and exit(1).
    fn get_or_build_bounds_fail(&mut self) -> CResult<FunctionValue<'ctx>> {
        let i64_ty = self.context.i64_type();
        let void = self.context.void_type();
        let ty = void.fn_type(&[i64_ty.into(), i64_ty.into()], false);
        let ctx = self.context;
        let fmt = self.str_literal("xia: index %lld out of bounds for array of length %lld\n")?;
        let (printf_ptr, printf_fnty) = self.libc_printf();
        let exit_ty = void.fn_type(&[self.context.i32_type().into()], false);
        let (exit_ptr, exit_fnty) = self.libc("exit", exit_ty);
        self.build_runtime_fn("xia_bounds_fail", ty, |b, f| {
            let idx = f.get_nth_param(0).unwrap();
            let len = f.get_nth_param(1).unwrap();
            let entry = ctx.append_basic_block(f, "entry");
            b.position_at_end(entry);
            b.build_indirect_call(
                printf_fnty,
                printf_ptr,
                &[fmt.into(), idx.into(), len.into()],
                "",
            )
            .map_err(err)?;
            b.build_indirect_call(
                exit_fnty,
                exit_ptr,
                &[ctx.i32_type().const_int(1, false).into()],
                "",
            )
            .map_err(err)?;
            b.build_unreachable().map_err(err)?;
            Ok(())
        })
    }

    /// `xia_arr_get(arr, idx) -> word`: bounds-checked load. Heap elements
    /// come back retained (+1) so the caller owns what it received.
    fn get_or_build_arr_get(&mut self) -> CResult<FunctionValue<'ctx>> {
        let i64_ty = self.context.i64_type();
        let ptr_ty = self.ptr_ty();
        let ty = i64_ty.fn_type(&[ptr_ty.into(), i64_ty.into()], false);
        let ctx = self.context;
        let bounds_fail = self.get_or_build_bounds_fail()?;
        let retain = self.get_or_build_retain()?;
        self.build_runtime_fn("xia_arr_get", ty, |b, f| {
            let arr = f.get_nth_param(0).unwrap().into_pointer_value();
            let idx = f.get_nth_param(1).unwrap().into_int_value();
            let entry = ctx.append_basic_block(f, "entry");
            let trap = ctx.append_basic_block(f, "trap");
            let load = ctx.append_basic_block(f, "load");
            let retain_bb = ctx.append_basic_block(f, "retain");
            let done = ctx.append_basic_block(f, "done");

            b.position_at_end(entry);
            let len = b.build_load(i64_ty, arr, "len").map_err(err)?.into_int_value();
            // Unsigned compare folds the `idx < 0` case in: it wraps huge.
            let ok = b
                .build_int_compare(IntPredicate::ULT, idx, len, "ok")
                .map_err(err)?;
            b.build_conditional_branch(ok, load, trap).map_err(err)?;

            b.position_at_end(trap);
            b.build_call(bounds_fail, &[idx.into(), len.into()], "")
                .map_err(err)?;
            b.build_unreachable().map_err(err)?;

            b.position_at_end(load);
            let data_slot = self.arr_field(b, arr, ARR_DATA_OFFSET, "data.slot")?;
            let data = b
                .build_load(ptr_ty, data_slot, "data")
                .map_err(err)?
                .into_pointer_value();
            let slot = unsafe {
                b.build_gep(i64_ty, data, &[idx], "slot").map_err(err)?
            };
            let word = b.build_load(i64_ty, slot, "word").map_err(err)?.into_int_value();
            let block = self.block_ptr(b, arr)?;
            let kind = b.build_load(i64_ty, block, "kind").map_err(err)?.into_int_value();
            let heap = b
                .build_int_compare(
                    IntPredicate::EQ,
                    kind,
                    i64_ty.const_int(KIND_ARR_HEAP, false),
                    "heap",
                )
                .map_err(err)?;
            b.build_conditional_branch(heap, retain_bb, done).map_err(err)?;

            b.position_at_end(retain_bb);
            let as_ptr = b.build_int_to_ptr(word, ptr_ty, "word.ptr").map_err(err)?;
            b.build_call(retain, &[as_ptr.into()], "").map_err(err)?;
            b.build_unconditional_branch(done).map_err(err)?;

            b.position_at_end(done);
            b.build_return(Some(&word)).map_err(err)?;
            Ok(())
        })
    }

    /// `xia_arr_set(arr, idx, word)`: bounds-checked store. For heap elements
    /// the new value is retained before the old one is released, so
    /// `xs[i] = xs[i]` is safe.
    fn get_or_build_arr_set(&mut self) -> CResult<FunctionValue<'ctx>> {
        let i64_ty = self.context.i64_type();
        let ptr_ty = self.ptr_ty();
        let void = self.context.void_type();
        let ty = void.fn_type(&[ptr_ty.into(), i64_ty.into(), i64_ty.into()], false);
        let ctx = self.context;
        let bounds_fail = self.get_or_build_bounds_fail()?;
        let retain = self.get_or_build_retain()?;
        let release = self.get_or_build_release()?;
        self.build_runtime_fn("xia_arr_set", ty, |b, f| {
            let arr = f.get_nth_param(0).unwrap().into_pointer_value();
            let idx = f.get_nth_param(1).unwrap().into_int_value();
            let val = f.get_nth_param(2).unwrap().into_int_value();
            let entry = ctx.append_basic_block(f, "entry");
            let trap = ctx.append_basic_block(f, "trap");
            let check = ctx.append_basic_block(f, "check");
            let swap = ctx.append_basic_block(f, "swap");
            let write = ctx.append_basic_block(f, "write");

            b.position_at_end(entry);
            let len = b.build_load(i64_ty, arr, "len").map_err(err)?.into_int_value();
            let ok = b
                .build_int_compare(IntPredicate::ULT, idx, len, "ok")
                .map_err(err)?;
            b.build_conditional_branch(ok, check, trap).map_err(err)?;

            b.position_at_end(trap);
            b.build_call(bounds_fail, &[idx.into(), len.into()], "")
                .map_err(err)?;
            b.build_unreachable().map_err(err)?;

            b.position_at_end(check);
            let data_slot = self.arr_field(b, arr, ARR_DATA_OFFSET, "data.slot")?;
            let data = b
                .build_load(ptr_ty, data_slot, "data")
                .map_err(err)?
                .into_pointer_value();
            let slot = unsafe {
                b.build_gep(i64_ty, data, &[idx], "slot").map_err(err)?
            };
            let block = self.block_ptr(b, arr)?;
            let kind = b.build_load(i64_ty, block, "kind").map_err(err)?.into_int_value();
            let heap = b
                .build_int_compare(
                    IntPredicate::EQ,
                    kind,
                    i64_ty.const_int(KIND_ARR_HEAP, false),
                    "heap",
                )
                .map_err(err)?;
            b.build_conditional_branch(heap, swap, write).map_err(err)?;

            b.position_at_end(swap);
            let new_ptr = b.build_int_to_ptr(val, ptr_ty, "new.ptr").map_err(err)?;
            b.build_call(retain, &[new_ptr.into()], "").map_err(err)?;
            let old = b.build_load(i64_ty, slot, "old").map_err(err)?.into_int_value();
            let old_ptr = b.build_int_to_ptr(old, ptr_ty, "old.ptr").map_err(err)?;
            b.build_call(release, &[old_ptr.into()], "").map_err(err)?;
            b.build_unconditional_branch(write).map_err(err)?;

            b.position_at_end(write);
            b.build_store(slot, val).map_err(err)?;
            b.build_return(None).map_err(err)?;
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
    fn for_loop_has_increment_block() {
        let src = "fn main() -> int:\n    let sum = 0\n    for i in range(1, 5):\n        if i == 3:\n            continue\n        sum = sum + i\n    return sum\n";
        let ir = compile_to_ir(src).unwrap();
        assert!(ir.contains("for.cond"));
        assert!(ir.contains("for.inc"));
        assert!(ir.contains("for.end"));
        // `continue` must branch to the increment, never back to the test.
        assert!(ir.contains("br label %for.inc"));
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
    fn arrays_lower_to_runtime_calls() {
        let src = "fn main() -> int:\n    let xs = [10, 20, 30]\n    xs[1] = 5\n    push(xs, 40)\n    return xs[1] + len(xs)\n";
        let ir = compile_to_ir(src).unwrap();
        assert!(ir.contains("xia_arr_new"));
        assert!(ir.contains("xia_arr_push"));
        assert!(ir.contains("xia_arr_get"));
        assert!(ir.contains("xia_arr_set"));
        assert!(ir.contains("xia_bounds_fail"));
    }

    #[test]
    fn str_arrays_use_heap_element_kind() {
        let src = "fn main():\n    let xs = [\"a\", \"b\"]\n    print(xs[0])\n";
        let ir = compile_to_ir(src).unwrap();
        // kind 2 = heap elements; release loops over them when the array dies
        assert!(ir.contains("@xia_arr_new(i64 2,"));
        assert!(ir.contains("elem.loop"));
    }

    #[test]
    fn foreach_lowers_with_per_iteration_get() {
        let src = "fn sum(xs: [int]) -> int:\n    let total = 0\n    for n in xs:\n        total = total + n\n    return total\n";
        let ir = compile_to_ir(src).unwrap();
        assert!(ir.contains("foreach.cond"));
        assert!(ir.contains("foreach.inc"));
        assert!(ir.contains("xia_arr_get"));
    }

    #[test]
    fn foreach_over_fresh_literal_released_after_loop() {
        let src = "fn main():\n    for s in [\"a\", \"b\"]:\n        print(s)\n";
        let ir = compile_to_ir(src).unwrap();
        assert!(ir.contains("foreach.end"));
        assert!(ir.contains("call void @xia_release"));
    }

    #[test]
    fn str_conversion_uses_snprintf() {
        let src = "fn main():\n    print(f\"n = {7} x = {2.5}\")\n";
        let ir = compile_to_ir(src).unwrap();
        assert!(ir.contains("xia_int_to_str"));
        assert!(ir.contains("xia_float_to_str"));
        assert!(ir.contains("@snprintf"));
        assert!(ir.contains("xia_str_concat"));
    }

    #[test]
    fn float_ops_lower_to_float_instructions() {
        let src = "fn f(x: float) -> float:\n    return x * 2.0 + 0.5\n";
        let ir = compile_to_ir(src).unwrap();
        assert!(ir.contains("fmul double"));
        assert!(ir.contains("fadd double"));
    }
}
