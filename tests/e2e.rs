//! End-to-end tests: drive the `xia` CLI to compile real programs to native
//! executables, run them, and assert on their output.

use std::path::{Path, PathBuf};
use std::process::Command;

fn xia() -> Command {
    Command::new(env!("CARGO_BIN_EXE_xia"))
}

fn compile_and_run(name: &str, source: &str, args: &[&str]) -> (String, i32) {
    let dir = std::env::temp_dir().join("xia-e2e");
    std::fs::create_dir_all(&dir).unwrap();
    let src_path = dir.join(format!("{name}.xia"));
    std::fs::write(&src_path, source).unwrap();

    let mut build = xia();
    build.arg("build").arg(&src_path).args(args);
    let out = build.output().expect("xia build must run");
    assert!(
        out.status.success(),
        "build failed:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let exe = exe_path(&dir, name);
    let run = Command::new(&exe).output().expect("compiled exe must run");
    (
        String::from_utf8_lossy(&run.stdout).replace("\r\n", "\n"),
        run.status.code().unwrap_or(-1),
    )
}

fn exe_path(dir: &Path, name: &str) -> PathBuf {
    if cfg!(windows) {
        dir.join(format!("{name}.exe"))
    } else {
        dir.join(name)
    }
}

#[test]
fn fib_prints_sequence_and_exits_zero() {
    let src = "fn fib(n: int) -> int:\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\nfn main() -> int:\n    let i = 0\n    while i <= 10:\n        print(fib(i))\n        i = i + 1\n    return 0\n";
    let (stdout, code) = compile_and_run("e2e_fib", src, &[]);
    assert_eq!(stdout, "0\n1\n1\n2\n3\n5\n8\n13\n21\n34\n55\n");
    assert_eq!(code, 0);
}

#[test]
fn exit_code_propagates_from_main() {
    let src = "fn main() -> int:\n    return 42\n";
    let (_, code) = compile_and_run("e2e_exit", src, &[]);
    assert_eq!(code, 42);
}

#[test]
fn strings_arc_and_concat() {
    let src = r#"fn greet(name: str) -> str:
    return "hello, " + name + "!"

fn shadow(s: str) -> str:
    let alias = s
    return alias

fn main() -> int:
    let g = greet("world")
    print(g)
    let a = shadow(g)
    a = a + " bye"
    print(a)
    if g == "hello, world!":
        print("eq works")
    let i = 0
    while i < 3:
        let tmp = g + " loop"
        print(tmp)
        i = i + 1
    return 0
"#;
    let (stdout, code) = compile_and_run("e2e_strings", src, &["--release"]);
    assert_eq!(
        stdout,
        "hello, world!\nhello, world! bye\neq works\nhello, world! loop\nhello, world! loop\nhello, world! loop\n"
    );
    assert_eq!(code, 0);
}

#[test]
fn extern_ffi_calls_libc_directly() {
    let src = "extern fn printf(fmt: str, ...) -> int\nextern fn llabs(n: int) -> int\nfn main() -> int:\n    printf(\"%lld\\n\", llabs(0 - 9))\n    return 0\n";
    let (stdout, code) = compile_and_run("e2e_ffi", src, &[]);
    assert_eq!(stdout, "9\n");
    assert_eq!(code, 0);
}

#[test]
fn floats_and_bools_print() {
    let src = "fn main() -> int:\n    print(2.5 * 2.0)\n    print(true)\n    print(1 < 2 and not false)\n    return 0\n";
    let (stdout, code) = compile_and_run("e2e_misc", src, &["--opt-size"]);
    assert_eq!(stdout, "5\ntrue\ntrue\n");
    assert_eq!(code, 0);
}

#[test]
fn emit_ir_writes_textual_llvm() {
    let dir = std::env::temp_dir().join("xia-e2e");
    std::fs::create_dir_all(&dir).unwrap();
    let src_path = dir.join("e2e_ir.xia");
    std::fs::write(&src_path, "fn main() -> int:\n    return 0\n").unwrap();
    let out = xia()
        .arg("build")
        .arg(&src_path)
        .args(["--emit", "ir"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let ir = std::fs::read_to_string(dir.join("e2e_ir.ll")).unwrap();
    assert!(ir.contains("define"));
}

#[test]
fn check_rejects_bad_programs() {
    let dir = std::env::temp_dir().join("xia-e2e");
    std::fs::create_dir_all(&dir).unwrap();
    let src_path = dir.join("e2e_bad.xia");
    std::fs::write(&src_path, "fn main() -> int:\n    return \"oops\"\n").unwrap();
    let out = xia().arg("check").arg(&src_path).output().unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("return type mismatch"));
}
