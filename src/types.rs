use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

/// A single dose event (bolus, infusion, or oral)
#[derive(Debug, Clone)]
pub struct DoseEvent {
    pub time: f64,
    pub amt: f64,
    pub cmt: usize,
    pub rate: f64,
    pub duration: f64,
    pub ss: bool,
    pub ii: f64,
}

impl DoseEvent {
    pub fn new(time: f64, amt: f64, cmt: usize, rate: f64, ss: bool, ii: f64) -> Self {
        let duration = if rate > 0.0 { amt / rate } else { 0.0 };
        Self {
            time,
            amt,
            cmt,
            rate,
            duration,
            ss,
            ii,
        }
    }

    pub fn is_infusion(&self) -> bool {
        self.rate > 0.0
    }
}

/// Fixed-layout PK parameters — replaces HashMap<String, f64> for AD compatibility.
///
/// Index convention:
///   0: CL   (clearance)
///   1: V    (volume, or V1 for 2-cmt)
///   2: Q    (intercompartmental clearance, 2-cmt only)
///   3: V2   (peripheral volume, 2-cmt only)
///   4: KA   (absorption rate constant, oral only)
///   5: F    (bioavailability, default 1.0)
///   6-7: reserved
pub const MAX_PK_PARAMS: usize = 8;

pub const PK_IDX_CL: usize = 0;
pub const PK_IDX_V: usize = 1;
pub const PK_IDX_Q: usize = 2;
pub const PK_IDX_V2: usize = 3;
pub const PK_IDX_KA: usize = 4;
pub const PK_IDX_F: usize = 5;

#[derive(Debug, Clone, Copy)]
pub struct PkParams {
    pub values: [f64; MAX_PK_PARAMS],
}

impl Default for PkParams {
    fn default() -> Self {
        let mut v = [0.0; MAX_PK_PARAMS];
        v[PK_IDX_F] = 1.0; // bioavailability defaults to 1
        Self { values: v }
    }
}

impl PkParams {
    pub fn cl(&self) -> f64 {
        self.values[PK_IDX_CL]
    }
    pub fn v(&self) -> f64 {
        self.values[PK_IDX_V]
    }
    pub fn q(&self) -> f64 {
        self.values[PK_IDX_Q]
    }
    pub fn v2(&self) -> f64 {
        self.values[PK_IDX_V2]
    }
    pub fn ka(&self) -> f64 {
        self.values[PK_IDX_KA]
    }
    pub fn f_bio(&self) -> f64 {
        self.values[PK_IDX_F]
    }

    /// Map a PK parameter name to its index in the fixed-size array.
    pub fn name_to_index(name: &str) -> Option<usize> {
        match name {
            "cl" => Some(PK_IDX_CL),
            "v" | "v1" => Some(PK_IDX_V),
            "q" => Some(PK_IDX_Q),
            "v2" => Some(PK_IDX_V2),
            "ka" => Some(PK_IDX_KA),
            "f" => Some(PK_IDX_F),
            _ => None,
        }
    }

    /// Build from named HashMap (bridge for parser compatibility)
    pub fn from_hashmap(map: &HashMap<String, f64>) -> Self {
        let mut p = Self::default();
        if let Some(&v) = map.get("cl") {
            p.values[PK_IDX_CL] = v;
        }
        if let Some(&v) = map.get("v") {
            p.values[PK_IDX_V] = v;
        }
        if let Some(&v) = map.get("v1") {
            p.values[PK_IDX_V] = v;
        }
        if let Some(&v) = map.get("q") {
            p.values[PK_IDX_Q] = v;
        }
        if let Some(&v) = map.get("v2") {
            p.values[PK_IDX_V2] = v;
        }
        if let Some(&v) = map.get("ka") {
            p.values[PK_IDX_KA] = v;
        }
        if let Some(&v) = map.get("f") {
            p.values[PK_IDX_F] = v;
        }
        p
    }
}

/// A single subject with dosing and observation data
#[derive(Debug, Clone)]
pub struct Subject {
    pub id: String,
    pub doses: Vec<DoseEvent>,
    pub obs_times: Vec<f64>,
    pub observations: Vec<f64>,
    pub obs_cmts: Vec<usize>,
    pub covariates: HashMap<String, f64>,
    pub tvcov: HashMap<String, Vec<f64>>,
}

/// A collection of subjects
#[derive(Debug, Clone)]
pub struct Population {
    pub subjects: Vec<Subject>,
    pub covariate_names: Vec<String>,
    pub dv_column: String,
}

impl Population {
    pub fn n_obs(&self) -> usize {
        self.subjects.iter().map(|s| s.observations.len()).sum()
    }
}

/// Between-subject variability matrix (Omega)
#[derive(Debug, Clone)]
pub struct OmegaMatrix {
    pub matrix: DMatrix<f64>,
    pub chol: DMatrix<f64>,
    pub eta_names: Vec<String>,
    pub diagonal: bool,
}

impl OmegaMatrix {
    pub fn from_matrix(m: DMatrix<f64>, names: Vec<String>, diagonal: bool) -> Self {
        let n = m.nrows();
        let chol = match m.clone().cholesky() {
            Some(c) => c.l(),
            None => {
                let eig = m.clone().symmetric_eigen();
                let min_eig = eig.eigenvalues.min();
                let reg = if min_eig < 0.0 { -min_eig + 1e-8 } else { 1e-8 };
                let m_reg = &m + DMatrix::identity(n, n) * reg;
                m_reg.cholesky().expect("Regularized matrix must be PD").l()
            }
        };
        Self {
            matrix: m,
            chol,
            eta_names: names,
            diagonal,
        }
    }

    pub fn from_diagonal(variances: &[f64], names: Vec<String>) -> Self {
        let n = variances.len();
        let mut m = DMatrix::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = variances[i];
        }
        Self::from_matrix(m, names, true)
    }

    pub fn dim(&self) -> usize {
        self.matrix.nrows()
    }
}

/// Residual error parameters (Sigma)
#[derive(Debug, Clone)]
pub struct SigmaVector {
    pub values: Vec<f64>,
    pub names: Vec<String>,
}

/// Full set of model parameters
#[derive(Debug, Clone)]
pub struct ModelParameters {
    pub theta: Vec<f64>,
    pub theta_names: Vec<String>,
    pub theta_lower: Vec<f64>,
    pub theta_upper: Vec<f64>,
    pub omega: OmegaMatrix,
    pub sigma: SigmaVector,
}

/// Supported PK structural models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PkModel {
    OneCptIvBolus,
    OneCptOral,
    OneCptInfusion,
    TwoCptIvBolus,
    TwoCptOral,
    TwoCptInfusion,
}

/// Supported residual error models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorModel {
    Additive,
    Proportional,
    Combined,
}

/// PK parameter function: maps (theta, eta, covariates) -> PkParams
pub type PkParamFn = Box<dyn Fn(&[f64], &[f64], &HashMap<String, f64>) -> PkParams + Send + Sync>;

/// A compiled model ready for estimation
pub struct CompiledModel {
    pub name: String,
    pub pk_model: PkModel,
    pub error_model: ErrorModel,
    pub pk_param_fn: PkParamFn,
    pub n_theta: usize,
    pub n_eta: usize,
    pub n_epsilon: usize,
    pub theta_names: Vec<String>,
    pub eta_names: Vec<String>,
    pub default_params: ModelParameters,
    /// Computes covariate-adjusted typical values per subject for AD.
    /// Returns `tv[i]` such that `PK_param[i] = tv[i] * exp(eta[i])`.
    /// Covariates and theta are folded in; only eta is differentiated.
    /// When `Some`, enables AD gradient computation in the inner loop.
    /// When `None`, falls back to finite-difference gradients.
    pub tv_fn: Option<Box<dyn Fn(&[f64], &HashMap<String, f64>) -> Vec<f64> + Send + Sync>>,
    /// Maps each individual parameter (eta index) to its PK parameter index.
    /// E.g. for a model with CL, V, KA: [PK_IDX_CL, PK_IDX_V, PK_IDX_KA] = [0, 1, 4].
    /// Used by AD functions to place parameters in the correct PK slots.
    pub pk_indices: Vec<usize>,
    /// ODE specification. When `Some`, predictions use ODE integration instead of
    /// analytical PK equations. The `pk_param_fn` output is flattened and passed
    /// to the ODE RHS function as the parameter vector.
    pub ode_spec: Option<crate::ode::OdeSpec>,
}

impl std::fmt::Debug for CompiledModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledModel")
            .field("name", &self.name)
            .field("pk_model", &self.pk_model)
            .field("error_model", &self.error_model)
            .field("n_theta", &self.n_theta)
            .field("n_eta", &self.n_eta)
            .finish()
    }
}

/// Per-subject estimation results
#[derive(Debug, Clone)]
pub struct SubjectResult {
    pub id: String,
    pub eta: DVector<f64>,
    pub ipred: Vec<f64>,
    pub pred: Vec<f64>,
    pub iwres: Vec<f64>,
    pub cwres: Vec<f64>,
    pub ofv_contribution: f64,
}

/// Full fit result
#[derive(Debug, Clone)]
pub struct FitResult {
    pub method: EstimationMethod,
    pub converged: bool,
    pub ofv: f64,
    pub aic: f64,
    pub bic: f64,
    pub theta: Vec<f64>,
    pub theta_names: Vec<String>,
    pub omega: DMatrix<f64>,
    pub sigma: Vec<f64>,
    pub covariance_matrix: Option<DMatrix<f64>>,
    pub se_theta: Option<Vec<f64>>,
    pub se_omega: Option<Vec<f64>>,
    pub se_sigma: Option<Vec<f64>>,
    pub subjects: Vec<SubjectResult>,
    pub n_obs: usize,
    pub n_subjects: usize,
    pub n_parameters: usize,
    pub n_iterations: usize,
    pub interaction: bool,
    pub warnings: Vec<String>,
}

/// Options for fit()
#[derive(Debug, Clone)]
pub struct FitOptions {
    pub method: EstimationMethod,
    pub outer_maxiter: usize,
    pub outer_gtol: f64,
    pub inner_maxiter: usize,
    pub inner_tol: f64,
    pub run_covariance_step: bool,
    pub interaction: bool,
    pub verbose: bool,
    pub optimizer: Optimizer,
    pub lbfgs_memory: usize,
    /// Run a gradient-free global pre-search (NLopt GN_CRS2_LM) before local optimization.
    pub global_search: bool,
    /// Max evaluations for the global pre-search (0 = auto).
    pub global_maxeval: usize,
    // SAEM-specific options
    pub saem_n_exploration: usize,
    pub saem_n_convergence: usize,
    pub saem_n_mh_steps: usize,
    pub saem_adapt_interval: usize,
    pub saem_seed: Option<u64>,
    /// Levenberg-Marquardt damping factor for Gauss-Newton (0 = pure GN).
    pub gn_lambda: f64,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            method: EstimationMethod::Foce,
            outer_maxiter: 500,
            outer_gtol: 1e-6,
            inner_maxiter: 200,
            inner_tol: 1e-8,
            run_covariance_step: true,
            interaction: false,
            verbose: true,
            optimizer: Optimizer::Slsqp,
            lbfgs_memory: 5,
            global_search: false,
            global_maxeval: 0,
            saem_n_exploration: 150,
            saem_n_convergence: 250,
            saem_n_mh_steps: 3,
            saem_adapt_interval: 50,
            saem_seed: None,
            gn_lambda: 0.01,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Optimizer {
    Bfgs,
    Lbfgs,
    /// NLopt LD_SLSQP — Sequential Least Squares Programming (recommended)
    Slsqp,
    /// NLopt LD_LBFGS
    NloptLbfgs,
    /// NLopt LD_MMA — Method of Moving Asymptotes
    Mma,
}

/// Estimation method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimationMethod {
    Foce,
    FoceI,
    FoceGn,
    FoceGnHybrid,
    Saem,
}

/// Trial design specification parsed from [simulation] block
#[derive(Debug, Clone)]
pub struct SimulationSpec {
    pub n_subjects: usize,
    pub dose_amt: f64,
    pub dose_cmt: usize,
    pub obs_times: Vec<f64>,
    pub seed: u64,
    /// Optional per-subject covariates: (name, values) — length must equal n_subjects
    pub covariates: Vec<(String, Vec<f64>)>,
}

/// Omega initial values: either diagonal variances or lower-triangle of a full matrix.
#[derive(Debug, Clone)]
pub enum OmegaInit {
    /// Diagonal variances only (length = n_eta)
    Diagonal(Vec<f64>),
    /// Lower triangle row-wise (length = n_eta*(n_eta+1)/2)
    LowerTriangle(Vec<f64>),
}

/// Full parsed model including simulation spec, initial values, and fit options
pub struct ParsedModel {
    pub model: CompiledModel,
    pub simulation: Option<SimulationSpec>,
    pub init_theta: Option<Vec<f64>>,
    pub init_omega: Option<OmegaInit>,
    pub init_sigma: Option<Vec<f64>>,
    pub fit_options: FitOptions,
}
