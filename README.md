# Xia

Xia is an ahead-of-time compiled programming language with Pythonic
indentation-based syntax, automatic reference counting (ARC) instead of a
garbage collector, and a zero-cost C FFI. The compiler is written in Rust and
emits native machine code through LLVM.

## Pipeline

1. **Lexing** — a [`logos`](https://crates.io/crates/logos)-based tokenizer plus
   an indentation stack that emits `INDENT` / `DEDENT` tokens, Python-style.
2. **Parsing** — a hand-rolled recursive descent parser producing a typed AST.
3. **Semantic analysis** — scoped symbol tables, type inference, and automatic
   insertion of `retain` / `release` calls for heap values (ARC).
4. **Code generation** — an AST visitor built on
   [`inkwell`](https://crates.io/crates/inkwell) that lowers to LLVM IR, runs
   standard optimization passes (DCE, inlining, loop unrolling), and emits
   native objects for any LLVM target triple.

## CLI

```
xia build <file.xia> [--release] [--target <triple>] [--emit ir|obj|exe]
xia run   <file.xia>
xia check <file.xia>
```

## Status

Under active development.
