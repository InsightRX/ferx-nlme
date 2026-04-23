# Frequently Asked Questions

## Do I need to use MU-referencing in my model definitions, like in NONMEM / nlmixr2?

**No.** FeRx does not require MU-referencing.

In NONMEM and nlmixr2, MU-referencing is a convention in which each random effect `ETA(i)` is linearly associated with a single `MU_i` term (typically `MU_i = LOG(THETA(i))`), so individual parameters look like:

```
MU_1 = LOG(THETA(1))
CL = EXP(MU_1 + ETA(1))
```

This structure is required by those tools' SAEM implementations because they rely on conjugate Gibbs updates that are only valid when the `MU_i` → `ETA(i)` relationship is strictly linear (and typically on a log scale). If you deviate — for example by writing `CL = THETA(1) * EXP(ETA(1))` without going through an intermediate `MU_1`, or by mixing multiple etas into one parameter — NONMEM SAEM will either reject the model or silently produce biased estimates.

FeRx's SAEM implementation uses **Metropolis-Hastings sampling** for the E-step rather than Gibbs, which does not require MU-referencing. You can write individual parameters in any form you like:

```
# All of these work fine in ferx:
CL = TVCL * exp(ETA_CL)
CL = TVCL + ETA_CL                              # additive eta
CL = TVCL * (WT/70)^0.75 * exp(ETA_CL)          # with covariates
CL = TVCL * exp(ETA_CL + ETA_CL_OCC)            # multiple etas
VMAX = TVVMAX * exp(ETA_VMAX)
KM   = TVKM                                      # no eta at all
```

The FOCE / FOCEI estimators have no MU-referencing requirement in any NLME tool — they use MAP optimization over etas regardless of parameterization. This is equally true in ferx.

### Performance implication

The main tradeoff is that MH sampling is slightly less efficient per iteration than conjugate Gibbs for MU-referenced models. In practice:

- For models with a few random effects, the difference is negligible
- For models with many (>10) random effects, conjugate Gibbs would converge in fewer iterations — but in exchange you gain flexibility to write models that don't fit the MU-referenced mold

If you have a NONMEM model that uses MU-referencing and want to port it to ferx, you can drop the MU intermediate step and write the individual parameters directly — the results will be equivalent.

## Which outer optimizer should I pick?

`slsqp` (the default) is the right choice for most models — it is fast, handles
box constraints cleanly, and behaves well on the log-transformed parameter
scale that ferx uses internally.

Reach for a different optimizer when SLSQP misbehaves:

- **`bobyqa`** — derivative-free, good when FOCE's FD gradients are noisy and
  SLSQP stalls or oscillates. Slower per iteration on smooth problems, but
  often converges when gradient-based methods give up.
- **`trust_region`** — second-order Newton trust-region. Can be faster near
  convergence because it uses curvature information; tune the CG budget with
  `steihaug_max_iters` (default 50) if you have more than ~50 packed
  parameters.
- **`lbfgs` / `bfgs`** — fall back to these only when NLopt is unavailable.

See [Fit Options](model-file/fit-options.md#optimizer-choices) for the full
list.
