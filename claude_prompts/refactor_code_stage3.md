# Prompt file for ALIS software refactoring -- STAGE 3

> **Fitting-engine improvements.** With the clean core in place (Stage 2), add
> performance and reliability features: profile caching/memoisation, automated
> fit diagnostics, and stronger convergence checks. Results must remain within
> the Stage 0 tolerances (caching in particular must not change the numbers).
> Depends on Stage 2.

## Tasks

> Complete in order; log each in `ALIS/claude_prompts/logs/refactor_code_stage3_log.md`.

**3.1 — Profile caching / memoisation.**
- Cache model components whose parameters are unchanged between iterations
  (fixed and tied components) and recompute only when their inputs change. For
  models with many fixed components this should cut per-iteration cost. Verify
  the cached path is numerically identical to the uncached path.

**3.2 — Fit diagnostics / residual analysis.**
- Add a post-fit "fit quality report": per-region reduced χ², flagging of
  poorly-fit wavelength intervals, and residual summaries (mean, scatter,
  runs/outliers). Complements the convergence checks.

**3.3 — Convergence robustness.**
- Formalise and extend the multi-start / random-restart machinery (currently in
  `alsims.py` and the external Monte-Carlo `newstart` workflow) so fits with
  hundreds of parameters can be shown not to "remember" their starting values.

## Skills to use for this stage

- `profile-fit` — quantify the caching speed-up and locate remaining bottlenecks.
- `convergence-check` — exercise the multi-start robustness checks.
- `check-fit` — build/reuse for the diagnostics report.
- `run-tests` — Stage 0 gate.

## Context

- `alcsmin.py` (minimiser), `alsims.py` (Monte-Carlo), the `sim newstart`
  workflow described in `doc/ALIS_workflow.md` §"Monte Carlo Convergence
  Testing", and the `DH`/`DH_orders` real-world fits (many-parameter, slow).
- Plan "Fitting and uncertainty" goals in `context.md`.

## Queries

**Q3.1 — Caching correctness bar.** Must the cached result be bitwise-identical
to the uncached result, or only within the Stage 0 tolerance? (Bitwise is safer
but constrains the implementation.)

**Response:** It must be bitwise identical. The caching mechanism should not
introduce any numerical differences, ensuring that the results are exactly the
same as if the computations were performed without caching. I previously attempted
an implementation of this in `alis/model_eval.model_func_ddp()` but it didn't quite
work. The goal is that emission components are all additive, and absorption components
are multiplicative, so the cached results should be combined in a way that preserves
the exact numerical outcome. The optimal approach might be to store all of the models
that contribute to a given specid (emission and absorption) and then only alter the
models components that have a parameter that influences that specid during the chi-squared
minimisation.

**Q3.2 — Diagnostics output.** Where should the fit-quality report go — a new
file (e.g. `<model>.report`), extra commented lines in `.mod.out`, or logged to
the console only?

**Response:** The fit-quality report should be store in printed to the console, and
then a separate `.report` file should be generated to store the result in.

**Q3.3 — Convergence approach.** Should Stage 3.3 build on the existing
`sim`/`newstart` mechanism, or is a redesigned convergence framework in scope?
Any specific acceptance criterion for "did not remember the starting values"
(e.g. all restarts agree within X·σ)?

**Response:** Stage 3.3 should build on the existing `sim`/`newstart` mechanism,
but it can be enhanced to improve robustness and scalability. The acceptance criterion
for "did not remember the starting values" should be that all restarts converge to
solutions that agree within a specified tolerance (e.g., within X·σ) for all parameters.
This ensures that the fitting process is not biased by initial conditions and that the
results are reliable. If you can propose an alternative strategy to improve the
convergence checks, that would be welcome, but it should still be compatible with
the existing framework. For example, users could select between several options
for convergence checks, such as a stricter tolerance for agreement or a more
robust statistical test to assess convergence across multiple restarts.

## Prompts

1. Please read this doc, including my responses to your queries, and check if any updates need to be made to this document before commencing. Ask further queries if needed.
