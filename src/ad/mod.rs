#[cfg(feature = "autodiff")]
pub mod ad_gradients;
pub mod dual;

pub use dual::Dual;
