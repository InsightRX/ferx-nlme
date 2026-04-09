/// Forward-mode dual number: val + deriv * ε
///
/// Used for automatic differentiation through the expression evaluator
/// and PK model equations. Each operation propagates the derivative
/// via the chain rule.
#[derive(Debug, Clone, Copy)]
pub struct Dual {
    pub val: f64,
    pub deriv: f64,
}

impl Dual {
    pub fn new(val: f64, deriv: f64) -> Self {
        Self { val, deriv }
    }

    pub fn constant(val: f64) -> Self {
        Self { val, deriv: 0.0 }
    }

    pub fn variable(val: f64) -> Self {
        Self { val, deriv: 1.0 }
    }

    pub fn exp(self) -> Self {
        let e = self.val.exp();
        Self { val: e, deriv: self.deriv * e }
    }

    pub fn ln(self) -> Self {
        Self {
            val: self.val.max(1e-30).ln(),
            deriv: self.deriv / self.val.max(1e-30),
        }
    }

    pub fn sqrt(self) -> Self {
        let s = self.val.max(0.0).sqrt();
        Self {
            val: s,
            deriv: if s > 1e-30 { self.deriv / (2.0 * s) } else { 0.0 },
        }
    }

    pub fn abs(self) -> Self {
        if self.val >= 0.0 {
            self
        } else {
            Self { val: -self.val, deriv: -self.deriv }
        }
    }

    pub fn powf(self, exp: Dual) -> Self {
        if self.val <= 0.0 {
            return Self::constant(0.0);
        }
        let val = self.val.powf(exp.val);
        // d/dx (x^y) = y*x^(y-1)*dx + x^y*ln(x)*dy
        let deriv = val * (exp.val * self.deriv / self.val + exp.deriv * self.val.ln());
        Self { val, deriv }
    }

    pub fn powi(self, n: i32) -> Self {
        let val = self.val.powi(n);
        let deriv = self.deriv * n as f64 * self.val.powi(n - 1);
        Self { val, deriv }
    }

    pub fn max(self, other: f64) -> Self {
        if self.val >= other {
            self
        } else {
            Self::constant(other)
        }
    }
}

impl std::ops::Add for Dual {
    type Output = Dual;
    fn add(self, rhs: Dual) -> Dual {
        Dual { val: self.val + rhs.val, deriv: self.deriv + rhs.deriv }
    }
}

impl std::ops::Sub for Dual {
    type Output = Dual;
    fn sub(self, rhs: Dual) -> Dual {
        Dual { val: self.val - rhs.val, deriv: self.deriv - rhs.deriv }
    }
}

impl std::ops::Mul for Dual {
    type Output = Dual;
    fn mul(self, rhs: Dual) -> Dual {
        Dual {
            val: self.val * rhs.val,
            deriv: self.val * rhs.deriv + self.deriv * rhs.val,
        }
    }
}

impl std::ops::Div for Dual {
    type Output = Dual;
    fn div(self, rhs: Dual) -> Dual {
        if rhs.val.abs() < 1e-30 {
            return Dual::constant(0.0);
        }
        Dual {
            val: self.val / rhs.val,
            deriv: (self.deriv * rhs.val - self.val * rhs.deriv) / (rhs.val * rhs.val),
        }
    }
}

impl std::ops::Neg for Dual {
    type Output = Dual;
    fn neg(self) -> Dual {
        Dual { val: -self.val, deriv: -self.deriv }
    }
}

impl PartialEq for Dual {
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val
    }
}

impl PartialOrd for Dual {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.val.partial_cmp(&other.val)
    }
}
