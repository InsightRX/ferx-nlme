//! Cooperative cancellation token for long-running fits.
//!
//! Motivation: when ferx-nlme is called from R (via extendr) or any other
//! FFI host, the host cannot interrupt our Rust hot loops. `CancelFlag` is a
//! cheap shared atomic that callers clone into [`FitOptions`](crate::types::FitOptions);
//! the estimation loops poll it at safe points and return a partial/empty
//! result quickly when set.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Clone-share this into `FitOptions.cancel` and call [`CancelFlag::cancel`]
/// from another thread to request a cooperative abort of an in-flight fit.
#[derive(Debug, Clone, Default)]
pub struct CancelFlag(Arc<AtomicBool>);

impl CancelFlag {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// Cheap helper: returns `true` if the optional flag is set.
///
/// Used inside tight loops to keep the polling call-site a one-liner.
#[inline]
pub fn is_cancelled(flag: &Option<CancelFlag>) -> bool {
    matches!(flag, Some(f) if f.is_cancelled())
}
