# Xia

Xia is an ahead-of-time compiled programming language with Pythonic
indentation-based syntax, automatic reference counting (ARC) instead of a
garbage collector, and a zero-cost C FFI. The compiler is written in Rust and
emits native machine code through LLVM 18.

```
# Strings and arrays are heap-allocated and managed by ARC — no GC.
extern fn printf(fmt: str, ...) -> int

fn greet(name: str) -> str:
    return "hello, " + name + "!"

fn fib(n: int) -> int:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

fn main() -> int:
    print(greet("world"))
    for i in range(11):
        printf("fib(%lld) = %lld\n", i, fib(i))

    let langs = ["xia", "c"]
    push(langs, "rust")
    for name in langs:
        print(name)
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
- Every heap block starts with a `[i64 kind][i64 refcount]` header and the
  value points just past it. A negative refcount marks immortal data
  (string literals live in constant globals and are never freed).
- `str` (kind 0) is `[header][bytes][NUL]`; the value points at the bytes,
  so every Xia string doubles as a `char*` for the FFI.
- `[T]` arrays (kind 1, or 2 for heap elements) are
  `[header][len][cap][data ptr]` handles over a growable buffer of 8-byte
  words. Indexing is bounds-checked (out of bounds prints a diagnostic and
  exits with code 1). Arrays retain their heap elements; releasing the last
  reference releases every element before freeing the buffer.
- A `struct` is `[header][field0][field1]...` with one 8-byte word per field.
  Structs with no heap fields use kind 0; structs that own heap data store the
  address of a generated destructor (`xia_drop_<Name>`) in the kind word, so
  `xia_release` dispatches to it and releases each heap field before freeing.
- An `enum` is a tagged union `[header][i64 tag][payload...]`, sized to the
  constructed variant; the value points at the tag and payload field `j` sits
  at `+8 + j*8`. Like structs, a scalar-only enum uses kind 0, while one whose
  variants own heap data stores `xia_drop_<Name>` in the kind word — that
  destructor switches on the tag and releases the live variant's heap fields.
  Self-referential enums (e.g. a cons list) are released recursively.
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
cargo test        # 115 unit/IR tests + 18 end-to-end binary tests
```

On Linux the distro packages work directly (see `.github/workflows/ci.yml`,
which runs the suite on every push):

```bash
sudo apt-get install llvm-18-dev libpolly-18-dev libzstd-dev zlib1g-dev
LLVM_SYS_181_PREFIX=/usr/lib/llvm-18 cargo test
```

Linking on Windows uses `lld-link` from the same LLVM package against the
MSVC / Windows SDK import libraries (Visual Studio Build Tools required); on
Linux/macOS it uses the system `cc`.

## Language reference (v0.2)

- Types: `int`, `float`, `bool`, `str`, `[T]` arrays (including nested
  `[[T]]`), `struct`s, and `enum`s; functions may return nothing (unit).
- `struct Name:` followed by an indented `field: type` per line declares a
  product type. Construct with positional arguments (`Name(a, b)`), read and
  assign fields with `.` (`p.x`, `p.x = 5`). Struct types resolve regardless
  of declaration order. See `examples/structs.xia`.
- Methods take an explicit receiver before the name —
  `fn (p: Point) area() -> int:` — and are called as `p.area(args)`. The
  receiver is borrowed (like any parameter) and the result is owned by the
  caller; calls compile to a direct call to a `Struct.method` symbol with the
  receiver passed as the implicit first argument (no vtables). Different
  structs may share a method name.
- `enum Name:` followed by indented variant lines declares a tagged union;
  a variant is a bare name (`Nil`) or carries a positional payload
  (`Cons(int, IntList)`, `Some(str)`). Variant names are global and unique, so
  they construct without qualification: `Some(x)`, `Nil`. Enum types resolve
  regardless of declaration order and may be self-referential. Take a value
  apart with `match`:

  ```
  match shape:
      Circle(r): return r * r * 3
      Rect(w, h): return w * h
      Nothing: return 0
  ```

  Each arm names a variant and binds its payload positionally (in scope only
  within that arm), or is the catch-all `_`. A `match` must be exhaustive or
  end in a `_` arm; duplicate and unknown-variant arms are rejected. See
  `examples/enums.xia`.
- Functions may be generic: `fn name[T, U](...)` introduces type parameters
  usable anywhere a type is expected in the signature and body (`fn first[T]
  (xs: [T]) -> T:`). A generic function is a template — each call infers its
  type arguments from the argument types and the compiler stamps out a
  concrete copy (monomorphization), so generics have no runtime cost and `T`
  never reaches codegen. Type arguments must be inferable from the arguments;
  a parameter shared by several arguments must agree across them. Methods may
  not be generic. See `examples/generics.xia`.
- `let x = expr` (inferred) or `let x: type = expr`; assignment with `=`.
  An empty array literal needs an annotation: `let xs: [int] = []`.
- `if` / `elif` / `else`, `while`, `break`, `continue` — blocks by
  indentation, no braces.
- `for i in range(end):` / `for i in range(start, end):` counts `start`
  (inclusive) to `end` (exclusive); `for x in xs:` iterates an array's
  elements. `continue` always advances the loop.
- Operators: `+ - * / %`, comparisons, `and` / `or` / `not` (short-circuit);
  `+` concatenates strings, `==`/`!=` compare them by value. `xs[i]` indexes
  (bounds-checked); `xs[i] = v` assigns in place. Indexing chains for nested
  arrays: `grid[r][c]`, `grid[r][c] = v` (see `examples/matrix.xia`).
- Builtins: `print(x)` for any printable type; `len(s)` / `len(xs)`;
  `push(xs, v)` appends (the buffer grows by doubling).
- `extern fn name(types...) -> ret` declares a C symbol; `...` marks varargs
  (e.g. `printf`). Calls have zero wrapper overhead — they are direct calls.
