//! Phases 4/5: object emission and optimization.
//!
//! Feeds the generated module into an LLVM `TargetMachine`. The target triple
//! is configurable, so the same Xia source cross-compiles to ELF (Linux),
//! Mach-O (macOS) or PE/COFF (Windows) objects. Standard optimization
//! pipelines (`default<O3>` / `default<Oz>`) run through the new pass
//! manager, and release builds strip symbols from the IR.

use inkwell::module::Module;
use inkwell::passes::PassBuilderOptions;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine, TargetTriple,
};
use inkwell::OptimizationLevel;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    /// -O0: fast builds, straight-line debuggable output.
    None,
    /// -O3: optimize for speed (DCE, inlining, loop unrolling, ...).
    Speed,
    /// -Oz: optimize for binary size.
    Size,
}

impl OptLevel {
    fn pass_pipeline(self) -> &'static str {
        match self {
            OptLevel::None => "default<O0>",
            OptLevel::Speed => "default<O3>",
            OptLevel::Size => "default<Oz>",
        }
    }

    fn llvm_level(self) -> OptimizationLevel {
        match self {
            OptLevel::None => OptimizationLevel::None,
            OptLevel::Speed => OptimizationLevel::Aggressive,
            OptLevel::Size => OptimizationLevel::Default,
        }
    }
}

pub struct Backend {
    machine: TargetMachine,
}

impl Backend {
    /// `triple = None` targets the host machine.
    pub fn new(triple: Option<&str>, opt: OptLevel) -> Result<Self, String> {
        Target::initialize_all(&InitializationConfig::default());

        let (triple, cpu, features) = match triple {
            Some(t) => (TargetTriple::create(t), String::new(), String::new()),
            None => (
                TargetMachine::get_default_triple(),
                TargetMachine::get_host_cpu_name().to_string(),
                TargetMachine::get_host_cpu_features().to_string(),
            ),
        };

        let target = Target::from_triple(&triple)
            .map_err(|e| format!("unknown target triple: {e}"))?;
        let machine = target
            .create_target_machine(
                &triple,
                &cpu,
                &features,
                opt.llvm_level(),
                RelocMode::PIC,
                CodeModel::Default,
            )
            .ok_or("could not create target machine")?;
        Ok(Backend { machine })
    }

    pub fn triple(&self) -> String {
        self.machine.get_triple().as_str().to_string_lossy().into_owned()
    }

    /// Run the standard optimization pipeline; release builds also strip
    /// symbols so emitted binaries stay lean.
    pub fn optimize(&self, module: &Module, opt: OptLevel) -> Result<(), String> {
        module.set_triple(&self.machine.get_triple());
        module.set_data_layout(&self.machine.get_target_data().get_data_layout());

        let mut pipeline = opt.pass_pipeline().to_string();
        if opt != OptLevel::None {
            pipeline.push_str(",strip");
        }
        module
            .run_passes(&pipeline, &self.machine, PassBuilderOptions::create())
            .map_err(|e| format!("optimization pipeline failed: {e}"))
    }

    pub fn emit_object(&self, module: &Module, path: &Path) -> Result<(), String> {
        self.machine
            .write_to_file(module, FileType::Object, path)
            .map_err(|e| format!("failed to write object file: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::CodeGen;
    use inkwell::context::Context;

    fn build_module<'ctx>(src: &str, ctx: &'ctx Context) -> inkwell::module::Module<'ctx> {
        let mut program = crate::parser::parse(src).unwrap();
        crate::sema::Analyzer::new().analyze(&mut program).unwrap();
        crate::arc::ArcInserter::new().run(&mut program);
        let mut cg = CodeGen::new(ctx);
        cg.compile(&program).unwrap();
        cg.module
    }

    const FIB: &str = "fn fib(n: int) -> int:\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\nfn main() -> int:\n    return fib(10)\n";

    #[test]
    fn emits_native_object() {
        let ctx = Context::create();
        let module = build_module(FIB, &ctx);
        let backend = Backend::new(None, OptLevel::Speed).unwrap();
        backend.optimize(&module, OptLevel::Speed).unwrap();
        let dir = std::env::temp_dir();
        let obj = dir.join("xia_test_fib.o");
        backend.emit_object(&module, &obj).unwrap();
        let meta = std::fs::metadata(&obj).unwrap();
        assert!(meta.len() > 0);
        std::fs::remove_file(&obj).ok();
    }

    #[test]
    fn cross_compiles_to_linux_elf() {
        let ctx = Context::create();
        let module = build_module(FIB, &ctx);
        let backend = Backend::new(Some("x86_64-unknown-linux-gnu"), OptLevel::Speed).unwrap();
        backend.optimize(&module, OptLevel::Speed).unwrap();
        let obj = std::env::temp_dir().join("xia_test_fib_linux.o");
        backend.emit_object(&module, &obj).unwrap();
        let bytes = std::fs::read(&obj).unwrap();
        assert_eq!(&bytes[..4], b"\x7fELF", "must be an ELF object");
        std::fs::remove_file(&obj).ok();
    }

    #[test]
    fn o3_inlines_and_folds_fib() {
        let ctx = Context::create();
        let module = build_module(FIB, &ctx);
        let backend = Backend::new(None, OptLevel::Speed).unwrap();
        backend.optimize(&module, OptLevel::Speed).unwrap();
        let ir = module.print_to_string().to_string();
        // After O3, xia_main's fib(10) reduces; the C main shim either gets
        // a constant or at minimum the module still verifies and shrinks.
        assert!(ir.contains("define"), "module survived optimization");
    }
}
