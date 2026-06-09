mod arc;
mod ast;
mod codegen;
mod lexer;
mod parser;
mod sema;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: xia <file.xia>");
        std::process::exit(2);
    });
    let source = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("error: cannot read {path}: {e}");
        std::process::exit(1);
    });
    let mut program = match parser::parse(&source) {
        Ok(program) => program,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    if let Err(e) = sema::Analyzer::new().analyze(&mut program) {
        eprintln!("{e}");
        std::process::exit(1);
    }
    arc::ArcInserter::new().run(&mut program);
    println!("{program:#?}");
}
