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
  `simulate.py` and the external Monte-Carlo `newstart` workflow) so fits with
  hundreds of parameters can be shown not to "remember" their starting values.

## Skills to use for this stage

- `profile-fit` — quantify the caching speed-up and locate remaining bottlenecks.
- `convergence-check` — exercise the multi-start robustness checks.
- `check-fit` — build/reuse for the diagnostics report.
- `run-tests` — Stage 0 gate.

## Context

- `minimise.py` (minimiser), `simulate.py` (Monte-Carlo), the `sim newstart`
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

**Q3.4 — Task 3.1 caching approach (raised during Prompt 1).** Build on the
existing machinery or design fresh?

**Response:** Build on the existing `emab`/`getemab` machinery. The base call
(`getemab=True`) already returns each specid's emission (additive) and
absorption (multiplicative) component contributions; derivative calls recompute
only the components whose parameter influences that specid (via `_pinfl`) and
recombine from the cached component arrays (no division → bitwise-safe). This
completes/repairs the WIP `model_func_ddp` + the `emab` plumbing already threaded
through `minimise.funcderiv`.

**Q3.5 — Task 3.1 verification + rollout (raised during Prompt 1).** How to
guarantee/verify bitwise-identical results?

**Response:** Toggle + A/B bitwise self-check. Add a setting (e.g. `run cache
True/False`) defaulting to the current uncached behaviour so both paths coexist;
add a self-check that runs cached vs uncached and asserts the model arrays are
bitwise-identical on the examples; flip the default to cached once proven. (The
Stage 0 suite compares against golden files within tolerance, so it cannot by
itself prove an exact cached-vs-uncached match.)

Findings recorded for Task 3.1: the minimiser uses two-sided finite-difference
derivatives; `model_func` already skips components not in `_pinfl[0][sp][sn]`
during a derivative (safe only because two-sided un-influenced regions cancel);
`emab` (cached em/ab components) is returned by the `getemab=True` base call and
passed into each `funcderiv` derivative.

Note: the finer specifics of Task 3.2 (definition of a "region", the
poorly-fit threshold, `.report` layout, always-on vs opt-in) and Task 3.3 (the
`X` in the `X·σ` agreement criterion, the menu of convergence tests) will be
settled at the start of those tasks' prompts.

**Q3.6 — Caching optimisation vs proceed (raised during Prompt 4).** After 3.1,
proceed to 3.2 or first make caching a universal win? Would moving the cached
functions to `model_eval_cached.py` avoid the per-derivative worker pickling?

**Response / findings:**
- Relocating the functions would **not** avoid the pickling. `minimise.funcderiv`
  dispatches each derivative with `apply_async(self.funcderiv, (fcn, fvec,
  functkw, ..., emab, ...))`; the *arguments* (`functkw` with the `FitState`,
  `emab` with the `compcache`) are pickled **by value** per derivative, while
  functions are pickled **by reference** — so module location is irrelevant to
  the payload. (Pickling also happens at `ncpus=1`: a `Pool` is always used.)
- Real optimisations: **(a) subset-pickling** — send only each derivative's
  influenced-sp/sn cache subset (`model_func` already skips the rest), a small
  pickle; **(b) shared memory** for the cache/`FitState`; **(c)** re-enable the
  commented serial path for `ncpus=1`. A `model_eval_cached.py` is fine as
  *organisation* but must accompany (a)/(b) to help performance.
- **Decision: proceed to Task 3.2.** Caching stays the verified opt-in;
  the subset-pickling optimisation (a) is recorded as a scoped follow-up (with
  DH_orders benchmarking) rather than blocking the stage's diagnostics /
  convergence features.

## Prompts

1. Please read this doc, including my responses to your queries, and check if any updates need to be made to this document before commencing. Ask further queries if needed.

2. Please re-read this doc. There are mentions of files that have changed since Stage 2 (e.g. `alcsmin.py`, `alsims.py`). Please check these carefully, and update as needed. If you find any discrepancies, please ask for clarification.

3. Please read this doc, and execute Task 3.1.

4. Considering the result from step 3.1, do you recommend that we proceed to Task 3.2, or pursue the optimization to make caching a universal win (i.e. avoid the per-derivative worker pickling [shared-memory or worker-side cache])? Please consider your answer carefully, and provide a detailed explanation of your reasoning, and any queries you have at this point. My thought is to move the functions in model_eval.py into a new file, model_eval_cached.py, and have the cached versions of the functions there. The original functions would remain in `model_eval.py` to within the `alfit` class in `minimise.py`. Would this avoid the pickling? I'm also open to hear your alternative suggestions.

5. Please read this doc, and execute Task 3.2.