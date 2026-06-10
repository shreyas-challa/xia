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
fn for_range_with_break_and_continue() {
    let src = "fn main() -> int:\n    let sum = 0\n    for i in range(10):\n        if i == 2:\n            continue\n        if i == 5:\n            break\n        sum = sum + i\n    for j in range(3, 5):\n        print(j)\n    return sum\n";
    let (stdout, code) = compile_and_run("e2e_for", src, &[]);
    assert_eq!(stdout, "3\n4\n");
    assert_eq!(code, 0 + 1 + 3 + 4);
}

#[test]
fn arrays_index_push_len_and_growth() {
    let src = "fn main() -> int:\n    let xs = [10, 20, 30]\n    xs[1] = 21\n    for i in range(100):\n        push(xs, i)\n    print(len(xs))\n    print(xs[1])\n    print(xs[102])\n    let sum = 0\n    for n in [1, 2, 3, 4]:\n        sum = sum + n\n    print(sum)\n    return 0\n";
    let (stdout, code) = compile_and_run("e2e_arrays", src, &[]);
    assert_eq!(stdout, "103\n21\n99\n10\n");
    assert_eq!(code, 0);
}

#[test]
fn string_arrays_with_arc() {
    let src = r#"fn names() -> [str]:
    let xs = ["ada", "grace"]
    push(xs, "alan" + " turing")
    return xs

fn main() -> int:
    let xs = names()
    xs[0] = xs[0] + "!"
    for name in xs:
        print(name)
    print(len(xs))
    return 0
"#;
    let (stdout, code) = compile_and_run("e2e_str_arrays", src, &["--release"]);
    assert_eq!(stdout, "ada!\ngrace\nalan turing\n3\n");
    assert_eq!(code, 0);
}

#[test]
fn out_of_bounds_index_traps() {
    let src = "fn main() -> int:\n    let xs = [1, 2]\n    print(xs[5])\n    return 0\n";
    let (stdout, code) = compile_and_run("e2e_oob", src, &[]);
    assert_eq!(code, 1, "bounds failure must exit(1)");
    assert!(stdout.contains("out of bounds"), "got: {stdout}");
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
    // Diagnostics point into the source with a caret underline.
    assert!(stderr.contains("e2e_bad.xia:2:12"), "got: {stderr}");
    assert!(stderr.contains("return \"oops\""), "got: {stderr}");
    assert!(stderr.contains("^^^^^^"), "got: {stderr}");
}
