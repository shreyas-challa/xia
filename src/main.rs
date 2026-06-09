mod ast;
mod lexer;
mod parser;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: xia <file.xia>");
        std::process::exit(2);
    });
    let source = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("error: cannot read {path}: {e}");
        std::process::exit(1);
    });
    match parser::parse(&source) {
        Ok(program) => println!("{program:#?}"),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
