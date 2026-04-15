#![cfg_attr(feature = "autodiff", feature(autodiff))]

// Require autodiff unless the `ci` feature is set (for CI without Enzyme toolchain).
#[cfg(all(not(feature = "autodiff"), not(feature = "ci")))]
compile_error!(
    "The `autodiff` feature is required. Finite-difference gradients are not supported. \
     Build with: cargo build --release (autodiff is now the default feature), \
     or explicitly: cargo build --release --features autodiff. \
     Requires the Enzyme Rust toolchain: rustup toolchain install enzyme"
);

pub mod types;
pub mod pk;
pub mod stats;
pub mod estimation;
pub mod io;
pub mod parser;
pub mod api;
pub mod ad;
pub mod ode;

pub use types::*;
pub use api::{fit, fit_from_files, run_from_file, run_model_with_data, run_model_simulate, simulate, simulate_with_seed, predict};
pub use io::datareader::read_nonmem_csv;
pub use parser::model_parser::{parse_model_file, parse_model_string, parse_full_model_file};
