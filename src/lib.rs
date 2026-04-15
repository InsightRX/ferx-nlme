#![cfg_attr(feature = "autodiff", feature(autodiff))]

// Require autodiff unless the `ci` feature is set (for CI without Enzyme toolchain).
#[cfg(all(not(feature = "autodiff"), not(feature = "ci")))]
compile_error!(
    "The `autodiff` feature is required. Finite-difference gradients are not supported. \
     Build with: cargo build --release (autodiff is now the default feature), \
     or explicitly: cargo build --release --features autodiff. \
     Requires the Enzyme Rust toolchain: rustup toolchain install enzyme"
);

pub mod ad;
pub mod api;
pub mod estimation;
pub mod io;
pub mod ode;
pub mod parser;
pub mod pk;
pub mod stats;
pub mod types;

pub use api::{
    fit, fit_from_files, predict, run_from_file, run_model_simulate, run_model_with_data, simulate,
    simulate_with_seed,
};
pub use io::datareader::read_nonmem_csv;
pub use parser::model_parser::{parse_full_model_file, parse_model_file, parse_model_string};
pub use types::*;
