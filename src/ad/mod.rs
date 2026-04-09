pub mod dual;
#[cfg(feature = "autodiff")]
pub mod ad_gradients;

pub use dual::Dual;
