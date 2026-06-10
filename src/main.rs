//! Phase 5: the `xia` CLI — an end-to-end pipeline (lex, parse, analyze,
//! compile, link) in the style of `cargo` / `go build`.

mod arc;
mod ast;
mod backend;
mod codegen;
mod diag;
mod lexer;
mod linker;
mod parser;
mod sema;

use backend::{Backend, OptLevel};
use clap::{Parser, Subcommand, ValueEnum};
use inkwell::context::Context;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Parser)]
#[command(name = "xia", version, about = "The Xia programming language compiler")]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Compile a .xia source file to a native executable
    Build {
        file: PathBuf,
        /// Optimize for speed (-O3) and strip symbols
        #[arg(long)]
        release: bool,
        /// Optimize for size (-Oz) instead of speed
        #[arg(long, conflicts_with = "release")]
        opt_size: bool,
        /// LLVM target triple (defaults to the host)
        #[arg(long)]
        target: Option<String>,
        /// What to produce
        #[arg(long, value_enum, default_value_t = Emit::Exe)]
        emit: Emit,
        /// Output path (defaults next to the source file)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Build and immediately run a .xia program
    Run {
        file: PathBuf,
        #[arg(long)]
        release: bool,
    },
    /// Type-check a .xia file without generating code
    Check { file: PathBuf },
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Emit {
    /// Textual LLVM IR (.ll)
    Ir,
    /// A relocatable object file (.o / .obj)
    Obj,
    /// A linked native executable
    Exe,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Cmd::Build { file, release, opt_size, target, emit, output } => {
            let opt = pick_opt(release, opt_size);
            build(&file, opt, target.as_deref(), emit, output).map(|out| {
                println!("{}", out.display());
                ExitCode::SUCCESS
            })
        }
        Cmd::Run { file, release } => {
            let opt = pick_opt(release, false);
            build(&file, opt, None, Emit::Exe, None).and_then(|exe| run(&exe))
        }
        Cmd::Check { file } => check(&file).map(|()| {
            println!("ok");
            ExitCode::SUCCESS
        }),
    };
    result.unwrap_or_else(|e| {
        eprintln!("error: {e}");
        ExitCode::FAILURE
    })
}

fn pick_opt(release: bool, opt_size: bool) -> OptLevel {
    if opt_size {
        OptLevel::Size
    } else if release {
        OptLevel::Speed
    } else {
        OptLevel::None
    }
}

fn frontend(file: &Path) -> Result<ast::Program, String> {
    let source = std::fs::read_to_string(file)
        .map_err(|e| format!("cannot read {}: {e}", file.display()))?;
    let path = file.display().to_string();
    let render = |d: diag::Diagnostic| d.render(&path, &source);
    let tokens = lexer::lex(&source).map_err(|e| render(e.into()))?;
    let mut program = parser::Parser::new(tokens)
        .parse_program()
        .map_err(|e| render(e.into()))?;
    sema::Analyzer::new()
        .analyze(&mut program)
        .map_err(|e| render(e.into()))?;
    Ok(program)
}

fn check(file: &Path) -> Result<(), String> {
    frontend(file).map(|_| ())
}

fn build(
    file: &Path,
    opt: OptLevel,
    target: Option<&str>,
    emit: Emit,
    output: Option<PathBuf>,
) -> Result<PathBuf, String> {
    let mut program = frontend(file)?;
    arc::ArcInserter::new().run(&mut program);

    let context = Context::create();
    let mut cg = codegen::CodeGen::new(&context);
    cg.compile(&program)?;

    let backend = Backend::new(target, opt)?;
    backend.optimize(&cg.module, opt)?;

    let stem = file.file_stem().unwrap_or_default().to_string_lossy();
    let dir = file.parent().unwrap_or(Path::new(".")).to_path_buf();
    let out = |ext: &str| output.clone().unwrap_or_else(|| dir.join(format!("{stem}{ext}")));

    match emit {
        Emit::Ir => {
            let path = out(".ll");
            cg.module
                .print_to_file(&path)
                .map_err(|e| format!("failed to write IR: {e}"))?;
            Ok(path)
        }
        Emit::Obj => {
            let path = out(if cfg!(windows) { ".obj" } else { ".o" });
            backend.emit_object(&cg.module, &path)?;
            Ok(path)
        }
        Emit::Exe => {
            if target.is_some() && backend.triple() != Backend::new(None, opt)?.triple() {
                return Err(
                    "cross-linking is not supported; use --emit obj and link on the target system"
                        .into(),
                );
            }
            let obj = out(if cfg!(windows) { ".obj" } else { ".o" });
            backend.emit_object(&cg.module, &obj)?;
            let exe = out(if cfg!(windows) { ".exe" } else { "" });
            linker::link(&obj, &exe)?;
            std::fs::remove_file(&obj).ok();
            Ok(exe)
        }
    }
}

fn run(exe: &Path) -> Result<ExitCode, String> {
    let status = std::process::Command::new(exe)
        .status()
        .map_err(|e| format!("failed to run {}: {e}", exe.display()))?;
    Ok(ExitCode::from(status.code().unwrap_or(1).clamp(0, 255) as u8))
}
