//! Phase 2: the abstract syntax tree.
//!
//! The parser produces this tree; semantic analysis annotates and rewrites it
//! (notably inserting ARC retain/release), and codegen walks it.

/// The scalar *base* of an array. Keeping it a small enum lets `Type` stay
/// `Copy`; array nesting is encoded by a separate dimension count on
/// `Type::Array`, so `[[int]]` is `Array(Int, 2)` rather than a recursive type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElemType {
    Int,
    Float,
    Bool,
    Str,
    /// Interned struct id — an index into `Program::structs`.
    Struct(u32),
}

impl ElemType {
    pub fn to_type(self) -> Type {
        match self {
            ElemType::Int => Type::Int,
            ElemType::Float => Type::Float,
            ElemType::Bool => Type::Bool,
            ElemType::Str => Type::Str,
            ElemType::Struct(id) => Type::Struct(id),
        }
    }

    /// The base scalar of a non-array type. Returns `None` for arrays and unit.
    pub fn from_type(ty: Type) -> Option<ElemType> {
        match ty {
            Type::Int => Some(ElemType::Int),
            Type::Float => Some(ElemType::Float),
            Type::Bool => Some(ElemType::Bool),
            Type::Str => Some(ElemType::Str),
            Type::Struct(id) => Some(ElemType::Struct(id)),
            Type::Array(..) | Type::Unit => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Int,
    Float,
    Bool,
    Str,
    /// `[T]` — a growable, reference-counted array. The `u8` is the number of
    /// nesting dimensions (≥ 1): `[int]` is `Array(Int, 1)`, `[[int]]` is
    /// `Array(Int, 2)`. Indexing peels off one dimension.
    Array(ElemType, u8),
    /// A user-defined struct, by interned id (index into `Program::structs`).
    Struct(u32),
    /// The "no value" type of statements and functions without `-> T`.
    Unit,
}

impl Type {
    /// Heap-allocated, reference-counted types managed by ARC.
    pub fn is_heap(self) -> bool {
        matches!(self, Type::Str | Type::Array(..) | Type::Struct(_))
    }

    /// The element type produced by indexing an array once. `[[int]]` yields
    /// `[int]`; `[int]` yields `int`. `None` for non-arrays.
    pub fn array_elem(self) -> Option<Type> {
        match self {
            Type::Array(base, 1) => Some(base.to_type()),
            Type::Array(base, dims) => Some(Type::Array(base, dims - 1)),
            _ => None,
        }
    }

    /// Build the type of an array whose elements have type `elem`. Stacking an
    /// array on an array deepens the dimension count; `None` for `unit`.
    pub fn array_of(elem: Type) -> Option<Type> {
        match elem {
            Type::Array(base, dims) => Some(Type::Array(base, dims + 1)),
            Type::Unit => None,
            scalar => Some(Type::Array(ElemType::from_type(scalar)?, 1)),
        }
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Int => write!(f, "int"),
            Type::Float => write!(f, "float"),
            Type::Bool => write!(f, "bool"),
            Type::Str => write!(f, "str"),
            Type::Array(base, dims) => {
                for _ in 0..*dims {
                    write!(f, "[")?;
                }
                write!(f, "{}", base.to_type())?;
                for _ in 0..*dims {
                    write!(f, "]")?;
                }
                Ok(())
            }
            Type::Struct(id) => write!(f, "struct#{id}"),
            Type::Unit => write!(f, "unit"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: crate::diag::Span,
    /// Filled in by semantic analysis; `None` straight out of the parser.
    pub ty: Option<Type>,
}

impl Expr {
    pub fn new(kind: ExprKind, span: crate::diag::Span) -> Self {
        Expr { kind, span, ty: None }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Var(String),
    Unary(UnOp, Box<Expr>),
    Binary(Box<Expr>, BinOp, Box<Expr>),
    Call(String, Vec<Expr>),
    /// `receiver.method(args)` — a struct method call. Lowered to a direct
    /// call passing the receiver as an implicit first argument.
    MethodCall(Box<Expr>, String, Vec<Expr>),
    /// `[e1, e2, ...]` — an empty literal needs a type annotation.
    ArrayLit(Vec<Expr>),
    /// `base[index]`
    Index(Box<Expr>, Box<Expr>),
    /// `base.field`
    Field(Box<Expr>, String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Let {
        name: String,
        ty: Option<Type>,
        value: Expr,
        line: usize,
    },
    Assign {
        name: String,
        value: Expr,
        line: usize,
    },
    /// `target[index] = value`
    IndexAssign {
        target: Expr,
        index: Expr,
        value: Expr,
        line: usize,
    },
    /// `target.field = value`
    FieldAssign {
        target: Expr,
        field: String,
        value: Expr,
        line: usize,
    },
    Expr(Expr),
    Return {
        value: Option<Expr>,
        line: usize,
    },
    If {
        cond: Expr,
        then_block: Block,
        else_block: Option<Block>,
    },
    While {
        cond: Expr,
        body: Block,
    },
    /// `for var in range(start, end):` — counts from `start` (inclusive) to
    /// `end` (exclusive). `continue` jumps to the increment, not the test.
    For {
        var: String,
        start: Expr,
        end: Expr,
        body: Block,
        line: usize,
    },
    /// `for var in xs:` — iterates the elements of an array. `var` is a fresh
    /// binding each iteration; heap elements arrive retained (+1).
    ForEach {
        var: String,
        iterable: Expr,
        body: Block,
        line: usize,
    },
    Break {
        line: usize,
    },
    Continue {
        line: usize,
    },
    /// ARC bookkeeping inserted by semantic analysis — never by the parser.
    Retain(String),
    Release(String),
}

pub type Block = Vec<Stmt>;

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: String,
    /// `Some` for a struct method `fn (p: Point) area():` — the explicit
    /// receiver, which is not part of `params`. Its type must be a struct.
    pub recv: Option<Param>,
    pub params: Vec<Param>,
    pub ret: Type,
    pub body: Block,
    pub line: usize,
}

impl Function {
    /// The mangled symbol name for a method (`Struct.method`), given the
    /// receiver's struct name. Free functions keep their bare name.
    pub fn method_symbol(struct_name: &str, method: &str) -> String {
        format!("{struct_name}.{method}")
    }
}

/// `extern fn name(t1, t2, ...) -> ret` — a zero-cost C FFI declaration.
/// Codegen declares the symbol to LLVM and the system linker resolves it.
#[derive(Debug, Clone, PartialEq)]
pub struct ExternFn {
    pub name: String,
    pub params: Vec<Type>,
    pub varargs: bool,
    pub ret: Type,
    pub line: usize,
}

/// `struct Name:` followed by indented `field: type` lines. Structs are
/// heap-allocated and reference-counted like strings and arrays; releasing
/// the last reference releases any heap-typed fields first.
#[derive(Debug, Clone, PartialEq)]
pub struct StructDef {
    pub name: String,
    pub fields: Vec<Param>,
    pub line: usize,
}

impl StructDef {
    pub fn field(&self, name: &str) -> Option<(usize, Type)> {
        self.fields
            .iter()
            .position(|f| f.name == name)
            .map(|i| (i, self.fields[i].ty))
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Program {
    pub externs: Vec<ExternFn>,
    pub functions: Vec<Function>,
    pub structs: Vec<StructDef>,
}
