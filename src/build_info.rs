use crate::types::{EstimationMethod, Optimizer};

/// Compile-time build metadata embedded by `build.rs`.
pub struct BuildInfo {
    pub variant: &'static str,
    pub profile: &'static str,
    pub ferx_version: &'static str,
    pub rustc_version: &'static str,
    pub build_timestamp: u64,
    pub has_autodiff: bool,
}

pub const BUILD_INFO: BuildInfo = BuildInfo {
    variant: env!("FERX_BUILD_VARIANT"),
    profile: env!("FERX_BUILD_PROFILE"),
    ferx_version: env!("CARGO_PKG_VERSION"),
    rustc_version: env!("FERX_RUSTC_VERSION"),
    build_timestamp: {
        let s = env!("FERX_BUILD_TIMESTAMP");
        let mut n: u64 = 0;
        let b = s.as_bytes();
        let mut i = 0;
        while i < b.len() {
            n = n * 10 + (b[i] - b'0') as u64;
            i += 1;
        }
        n
    },
    has_autodiff: cfg!(feature = "autodiff"),
};

/// Reported gradient method for a fit loop (inner or outer).
///
/// Distinct from [`crate::types::GradientMethod`] which controls _selection_;
/// this enum describes what actually runs and is used only for reporting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GradientMethodKind {
    /// Enzyme automatic differentiation — exact, constant cost in n_eta.
    EnzymeAD,
    /// Central finite differences — cost scales as 2×n_eta per gradient call.
    FiniteDifferences,
    /// Not applicable: derivative-free optimizer or sampling-based step.
    NotApplicable,
}

impl GradientMethodKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::EnzymeAD => "Enzyme AD",
            Self::FiniteDifferences => "finite differences",
            Self::NotApplicable => "N/A",
        }
    }
}

/// Gradient method used in the inner (per-subject EBE) loop.
///
/// Determined solely by the build variant — if autodiff is compiled in, the
/// inner BFGS uses Enzyme AD for analytical models; otherwise finite differences.
pub fn gradient_method_inner(build: &BuildInfo) -> GradientMethodKind {
    if build.has_autodiff {
        GradientMethodKind::EnzymeAD
    } else {
        GradientMethodKind::FiniteDifferences
    }
}

/// Gradient method used in the outer (population parameter) loop.
///
/// Depends on both the build variant and the chosen estimation method/optimizer:
/// - NLopt-based: NLopt uses its own internal FD regardless of build variant.
/// - BOBYQA: derivative-free — no outer gradient at all.
/// - Built-in BFGS/LBFGS: uses AD when available, else FD.
/// - TrustRegion: FD Hessian via the argmin crate.
/// - GN/GnHybrid: BHHH outer approximation — always FD.
/// - SAEM: MH E-step has no gradient; M-step uses NLopt internally.
pub fn gradient_method_outer(
    build: &BuildInfo,
    method: EstimationMethod,
    optimizer: Optimizer,
) -> GradientMethodKind {
    match method {
        EstimationMethod::Saem => GradientMethodKind::NotApplicable,
        EstimationMethod::FoceGn | EstimationMethod::FoceGnHybrid => {
            GradientMethodKind::FiniteDifferences
        }
        EstimationMethod::Foce | EstimationMethod::FoceI => match optimizer {
            Optimizer::Bobyqa => GradientMethodKind::NotApplicable,
            Optimizer::Bfgs | Optimizer::Lbfgs => gradient_method_inner(build),
            Optimizer::Slsqp
            | Optimizer::NloptLbfgs
            | Optimizer::Mma
            | Optimizer::TrustRegion => GradientMethodKind::FiniteDifferences,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ad_build() -> BuildInfo {
        BuildInfo {
            variant: "autodiff",
            profile: "debug",
            ferx_version: "0.1.0",
            rustc_version: "rustc 1.0.0",
            build_timestamp: 0,
            has_autodiff: true,
        }
    }

    fn ci_build() -> BuildInfo {
        BuildInfo {
            variant: "ci",
            profile: "release",
            ferx_version: "0.1.0",
            rustc_version: "rustc 1.0.0",
            build_timestamp: 0,
            has_autodiff: false,
        }
    }

    #[test]
    fn inner_ad_build_returns_enzyme() {
        assert_eq!(
            gradient_method_inner(&ad_build()),
            GradientMethodKind::EnzymeAD
        );
    }

    #[test]
    fn inner_ci_build_returns_fd() {
        assert_eq!(
            gradient_method_inner(&ci_build()),
            GradientMethodKind::FiniteDifferences
        );
    }

    #[test]
    fn outer_nlopt_always_fd() {
        for optimizer in [Optimizer::Slsqp, Optimizer::NloptLbfgs, Optimizer::Mma] {
            for &build in &[&ad_build(), &ci_build()] {
                assert_eq!(
                    gradient_method_outer(build, EstimationMethod::Foce, optimizer),
                    GradientMethodKind::FiniteDifferences,
                    "expected FD for NLopt optimizer {:?}",
                    optimizer
                );
            }
        }
    }

    #[test]
    fn outer_bobyqa_not_applicable() {
        assert_eq!(
            gradient_method_outer(&ad_build(), EstimationMethod::Foce, Optimizer::Bobyqa),
            GradientMethodKind::NotApplicable
        );
    }

    #[test]
    fn outer_bfgs_follows_build() {
        assert_eq!(
            gradient_method_outer(&ad_build(), EstimationMethod::Foce, Optimizer::Bfgs),
            GradientMethodKind::EnzymeAD
        );
        assert_eq!(
            gradient_method_outer(&ci_build(), EstimationMethod::Foce, Optimizer::Bfgs),
            GradientMethodKind::FiniteDifferences
        );
    }

    #[test]
    fn outer_saem_not_applicable() {
        assert_eq!(
            gradient_method_outer(&ad_build(), EstimationMethod::Saem, Optimizer::Bobyqa),
            GradientMethodKind::NotApplicable
        );
    }

    #[test]
    fn outer_gn_always_fd() {
        assert_eq!(
            gradient_method_outer(&ad_build(), EstimationMethod::FoceGn, Optimizer::Bobyqa),
            GradientMethodKind::FiniteDifferences
        );
    }
}
