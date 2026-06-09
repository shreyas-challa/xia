# Xia

Xia is an ahead-of-time compiled programming language with Pythonic
indentation-based syntax, automatic reference counting (ARC) instead of a
garbage collector, and a zero-cost C FFI. The compiler is written in Rust and
emits native machine code through LLVM 18.

```
# Strings are heap-allocated and managed by ARC — no GC.
extern fn printf(fmt: str, ...) -> int

fn greet(name: str) -> str:
    return "hello, " + name + "!"

fn fib(n: int) -> int:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

fn main() -> int:
    print(greet("world"))
    printf("fib(10) = %lld\n", fib(10))
    return 0
```

```
$ xia run examples/strings.xia
hello, world!
...
```

## The pipeline

1. **Lexing** (`src/lexer.rs`) — a [`logos`](https://crates.io/crates/logos)
   tokenizer plus an indentation stack (`Vec<usize>`) that emits `INDENT` /
   `DEDENT` tokens, Python-style, with implicit line joining inside brackets.
2. **Parsing** (`src/parser.rs`) — a hand-rolled recursive descent parser
   producing the AST in `src/ast.rs`; `elif` chains desugar to nested if/else.
3. **Semantic analysis** (`src/sema.rs`) — scoped symbol tables (a stack of
   `HashMap`s), bottom-up type inference, and call checking against both Xia
   and `extern` signatures.
4. **ARC insertion** (`src/arc.rs`) — rewrites the typed AST with
   `retain` / `release` so every heap value's refcount balances at scope
   boundaries: aliases retain, returns transfer ownership to the caller,
   `break`/`continue`/`return` release eagerly along their paths.
5. **Code generation** (`src/codegen.rs`) — an
   [`inkwell`](https://crates.io/crates/inkwell) visitor lowers the AST to
   LLVM IR. The ARC/string runtime (`xia_retain`, `xia_release`,
   `xia_str_concat`, `xia_str_dup`, `xia_str_eq`) is built directly in IR; a
   string's refcount header sits at `ptr - 8`, so every Xia `str` doubles as a
   `char*` for the FFI.
6. **Backend & linking** (`src/backend.rs`, `src/linker.rs`) — LLVM
   `TargetMachine` object emission for any target triple, the standard
   `default<O3>` / `default<Oz>` pass pipelines plus symbol stripping for
   release builds, then `lld-link` (Windows) or `cc` (Unix) links directly
   against libc.

## Memory model

- `int` (i64), `float` (f64), `bool` (i1) are plain values.
- `str` is a refcounted heap block `[i64 refcount][bytes][NUL]`; the value
  points at the bytes. A negative refcount marks immortal data (literals live
  in constant globals and are never freed).
- The compiler inserts all retain/release calls; there is nothing to call
  manually and no GC pause. Function arguments are borrowed, returns are +1,
  and `str` results from `extern` functions are copied into Xia-owned memory.

## CLI

```
xia build <file.xia> [--release | --opt-size] [--target <triple>]
                     [--emit ir|obj|exe] [-o <out>]
xia run   <file.xia> [--release]
xia check <file.xia>
```

`--target` accepts any LLVM triple — the same source emits ELF, Mach-O, or
PE/COFF objects (`--emit obj`; cross-linking needs a linker for that target).

## Building the compiler

Requires Rust and an LLVM 18.1 build with `llvm-config` and static libraries.
On Windows, official installers don't ship those; grab a dev package from
[c3lang/win-llvm](https://github.com/c3lang/win-llvm) and point `llvm-sys` at
it:

```powershell
$env:LLVM_SYS_181_PREFIX = "C:\path\to\llvm-18.1.8-windows-amd64-msvc17-msvcrt"
cargo build --release
cargo test        # 47 unit/IR tests + 7 end-to-end binary tests
```

Linking on Windows uses `lld-link` from the same LLVM package against the
MSVC / Windows SDK import libraries (Visual Studio Build Tools required); on
Linux/macOS it uses the system `cc`.

## Language reference (v0.1)

- Types: `int`, `float`, `bool`, `str`; functions may return nothing (unit).
- `let x = expr` (inferred) or `let x: type = expr`; assignment with `=`.
- `if` / `elif` / `else`, `while`, `break`, `continue` — blocks by
  indentation, no braces.
- Operators: `+ - * / %`, comparisons, `and` / `or` / `not` (short-circuit);
  `+` concatenates strings, `==`/`!=` compare them by value.
- `print(x)` builtin for any printable type.
- `extern fn name(types...) -> ret` declares a C symbol; `...` marks varargs
  (e.g. `printf`). Calls have zero wrapper overhead — they are direct calls.
