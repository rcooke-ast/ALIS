# Prompt file for ALIS software refactoring -- STAGE 3.5 (minimiser <-> model-eval seam)

> **Preparatory engine refactor for Stage 4 (GPU).** Restructure the
> `minimise.alfit` <-> `model_eval` call chain so the minimiser owns the model
> evaluation and the per-iteration setup is explicit, giving Stage 4 a clean
> `prepare_iteration()` / `eval_derivative(j)` seam to port to the GPU with
> minimal risk. This is a Stage-3-style engine refactor: it must land
> **bitwise-identical** under the Stage 0 gate (as Stage 3.4 did) *before* any
> GPU work begins. Priority is pure structural cleanup that enables the GPU port;
> CPU wins are banked only where they do not complicate the GPU implementation
> (RJC, Prompt 10 responses).
>
> Depends on Stage 3.4 (persistent Pool + chunked, subset-pickled Jacobian).
> Enables Stage 4, especially Task 4.5 (shared-memory read-only arrays).

## Why this is a prerequisite for Stage 4

The minimiser talks to the model through a **generic stateless-function**
interface inherited from MPFIT: `alfit.call()` does `fcn(x, **functkw)`
(`minimise.py:1236-1252`). But ALIS only ever fits its own model -- `fcn` is
*always* `model_eval._minimiser_eval` at all 6 `alfit(...)` call sites (main.py
x3, simulate.py x3), and `functkw` is always `{x, y, err, state}`. The generic
interface forces a "re-derive everything per call" style and hides the natural
work hierarchy:

- **per-fit invariants** (data, atomic data, funcarray, `posnfull`/`wavefull`,
  model structure) -- already shared via `FitState` (Stage 2.4);
- **per-iteration invariants** (change once per accepted step): the influence
  table `_pinfl`, the base-point component cache, and (if `renew_subpix`) the
  base sub-pixel grid -- today computed as *side-effects* of the base
  `model_func` call, with no explicit "iteration" concept;
- **per-derivative work** (per perturbed parameter): recompute only the
  influenced component(s).

A GPU port needs exactly this separation: upload the per-iteration invariants
once (Task 4.5 shared/device memory), then run the derivative kernel many times
with only the perturbed parameters changing. Doing the CPU-side restructuring
first means Stage 4 ports a clean seam instead of the "re-derive per call"
structure.

## Tasks

> Complete in order; log each in
> `ALIS/claude_prompts/logs/refactor_code_stage3p5_minimiser_log.md`.
> Every task is gated by the Stage 0 regression suite and must be
> **bitwise-identical** to the current output.

**3.5.1 -- Absorb `fcn` into `alfit`.**
- `fcn` never varies, so remove it as a constructor/`call()` parameter and have
  the minimiser invoke the model evaluation directly (a method or bound
  reference), owning the `FitState`. Drop the `**functkw` unpacking and the
  generic `call()` indirection at all 6 `alfit(...)` sites. Pure structural,
  bitwise-neutral (identical computation, fewer hops). Keep `damp`/`tie`
  handling intact.

**3.5.2 -- Make the per-iteration setup explicit (`prepare_iteration`).**
- Introduce an `alfit`-owned step run **once per accepted parameter set**
  (before the Jacobian) that computes the per-iteration invariants: the
  influence table `_pinfl`, the base-point component cache (the existing
  `getemab`/`emab` machinery), and -- when `renew_subpix` is on -- the base
  sub-pixel grid. The base residual evaluation and the 2n derivative evaluations
  then consume these. This is **bitwise-neutral by construction**: `_pinfl` is
  *already* computed at the base point (`ddpid is None`) and reused by every
  derivative (see Findings F1); the base cache is already built once
  (Stage 3.1). The task only *relocates* these into an explicit method, turning
  implicit side-effects into a named per-iteration contract.

**3.5.3 -- Establish the `eval_derivative(j)` contract (return, don't mutate).**
- Today each evaluation mutates shared `state` (`state._modfinal/_contfinal/
  _zerofinal/_pinfl`), which forces the per-call `copy.copy(state)` in
  `_minimiser_eval` (2n/iter). Refactor the derivative path so a derivative is a
  pure function of `(per-iteration invariants, perturbed params)` that *returns*
  its residual column instead of writing into shared state. This removes the
  per-call shallow copy and yields the exact shape a GPU kernel needs. Highest
  bitwise-risk task -- do it in small, individually gated steps; the Stage 0
  bitwise gate is the arbiter.

**3.5.4 -- Delete dead code.**
- Remove `model_eval.model_func_ddp` (its body opens with
  `msgs.bug("Shifts not implemented in speed-up model_func_ddp")` and is only
  reachable via an unused `main.py` wrapper) and the wrapper; drop the
  `modelem`/`modelab` (`emab[0]`/`emab[1]`) plumbing if it is unused once
  `model_func_ddp` is gone (the derivative path reads only `emab[2]`, the cache).

**3.5.5 -- (CPU win, bitwise-safe, only if low-risk) `renew_subpix` conditional
recompute.**
- Only relevant when `run renew_subpix True` (default off, but **on** in the
  real helium34 fits). NOTE the trap (Finding F2): the subpix grid depends on
  fitted widths/resolution, so a blanket "compute once per iteration" hoist is
  **not** bitwise-identical -- it would drop a real derivative dependence. The
  bitwise-safe win is a *conditional* recompute: reuse the base grid for
  derivative perturbations of parameters that do not feed the subpix/resolution
  model, and recompute only for those that do. Implement only if it stays simple
  and does not complicate the GPU seam; otherwise leave a clear note and defer.

**3.5.6 -- Unit tests for this stage's stable surface (do last).**
- Per the cross-cutting policy (`claude_prompts/refactor_code_unit_tests.md`),
  add `unit`-marked tests for the new stable surface: the `prepare_iteration`
  invariants, the `eval_derivative` return contract, and (if done) the
  `renew_subpix` conditional-recompute predicate. Keep them fast and isolated;
  the existing `unit` CI job picks them up.

## Findings (investigation results -- authoritative for the refactor)

**F1 -- The influence table `_pinfl` is value-dependent; correct granularity is
once-per-iteration (NOT once-per-fit).**
`load.load_par_influence` (`load.py:1718`) decides, per region, which parameters
influence it by testing whether each component's lines fall inside the region's
window. In `functions/voigt.py:set_vars` (the main absorption function), the
influence list is returned **empty** when the component's *redshifted* lines lie
outside the window:
`nw = where(Wavelength*(1+pt[1]) in [wvmin,wvmax]); if size==0 and getinfl: return [], []`.
So influence depends on parameter values through (a) the component **redshift**
`pt[1]` and (b) the **shift** params (via `wvrng`, the shifted region range).
A line near a region edge can move in/out during the fit, so `_pinfl` **cannot**
be safely hoisted to once-per-fit (a stale table would break bitwise caching
correctness). It is *already* computed once per iteration (base call,
`model_eval.py:120`) and reused by all 2n derivatives -- which is both the
correct minimum granularity and what makes the current caching bitwise-exact.
=> Task 3.5.2 relocates this computation unchanged; there is no CPU win here
(it is 1x/iter vs the Jacobian's ~95% compute), only a structural one.
(The base class `base.py:set_vars` ignores `wvrng` -- influence there is purely
structural -- so the value-dependence comes specifically from the line-bearing
functions: `voigt`, `gaussian`, `lineemission`, `splineabs`.)

**F2 -- `renew_subpix` per-derivative recompute is mostly-redundant but
partially-essential; a blanket hoist is NOT bitwise-safe.**
`model_func` recomputes `load.load_subpixels` at the top of every call when
`run renew_subpix True` (`model_eval.py:115`). In the derivative path the grid
is recomputed from the **perturbed** params `pp`, and `load_subpixels`
(`load.py:1620`) sizes each region's sub-pixel count from the narrowest line
width and the instrumental FWHM -- both fitted. So perturbing a width/resolution
parameter genuinely changes the grid, and that dependence is part of the correct
finite-difference derivative. Recomputing per-derivative is therefore redundant
*only* for parameters that do not feed the subpix model (most of them). The
bitwise-safe optimisation is a conditional recompute keyed on whether the
perturbed parameter affects subpixellation -- see Task 3.5.5.

**F3 -- Already addressed by earlier stages (no action needed, recorded to avoid
re-litigating):** `myfunct`/`model_func` are already standalone functions taking
an explicit `FitState` (Stage 2.3/2.4), not `ClassMain` methods; the
`alfunc` registry `base.call` is already built once via `build_funcarray`
(Stage 2.2), so it is not re-instantiated per iteration; the model is evaluated
2n times/iteration only because that is inherent to the finite-difference
Jacobian (attacked by Stage 3.1 caching + 3.4 parallelism).

## Skills to use for this stage

- `profile-fit` -- confirm bitwise-identity and check for any CPU regression.
- `run-tests` -- Stage 0 gate (must stay green and bitwise) + the `unit` batch.

## Context

- `minimise.py`: `alfit.__init__` (fcn/functkw), `alfit.call` (1236-1252, the
  generic dispatch), `fdjac2` (the Stage 3.4 persistent-Pool chunked Jacobian
  and its worker replicas), `enorm`.
- `model_eval.py`: `_minimiser_eval` (86, the `copy.copy(state)` per call),
  `model_func` (102, the base/derivative branch, cache, renew_subpix, `_pinfl`),
  `model_func_ddp` (427, dead), `myfunct` (615), `FitState` (26).
- `load.py`: `load_par_influence` (1718), `load_subpixels` (1620).
- `functions/base.py:set_vars` (589) and `functions/voigt.py:set_vars` -- the
  `getinfl`/`wvrng` influence logic behind Finding F1.
- `main.py`: `build_funcarray` (29), the `alfit(...)` sites (273/315/361), the
  thin `self.myfunct/model_func` wrappers (151-163) used by non-fit paths
  (initial chi2, plotting, sim setup).
- Stage 3.4 log (`refactor_code_stage3_log.md`) and Prompt 10 analysis in
  `logs/ALIS_v2_code_plan_logs.md`.

## Queries

**Q3.5.1 -- Scope of the doc (RJC, Prompt 10).**
**Response:** Fold this into a dedicated pre-Stage-4 doc (this file). Its vision
is to be an essential preparatory step for Stage 4, structuring the code so the
GPU port lands with minimal risk. Include relevant queries/responses and any
useful refactor information (done: see Findings).

**Q3.5.2 -- Bitwise requirement (RJC, Prompt 10).**
**Response:** Yes -- must stay bitwise-identical under the Stage 0 gate, as
Stage 3.4 did.

**Q3.5.3 -- Structural vs performance appetite (RJC, Prompt 10).**
**Response:** Focus on pure structural cleanup that enables the GPU port
(minimal risk, defer perf to Stage 4); bank CPU wins only where they do not
complicate the GPU implementation.

**Q3.5.4 -- Is `_pinfl` value-independent / hoistable? (RJC, Prompt 10).**
**Response:** Investigated -- see Finding F1. It is value-dependent (via
component redshift and shift), so it is hoistable only to *once-per-iteration*
(already the case), not once-per-fit. Task 3.5.2 relocates it unchanged.

**Q3.5.5 -- Non-fit `self.myfunct` call sites (open).** The initial-chi2
(`main.py:205`), `plot only`, and `sim beginfrom` paths call `self.myfunct`
directly (not through the minimiser). After 3.5.1/3.5.3, should these route
through the same standalone `model_eval` entry point (one evaluation path), or
keep the thin `main.py` wrappers for these non-fit uses? Recommendation: unify
on the standalone path to remove duplication, provided Stage 0 stays bitwise.

**Q3.5.6 -- Risk tolerance for Task 3.5.3 (open).** The return-instead-of-mutate
contract is the highest-value GPU enabler but touches the numerically sensitive
result-passing. Confirm it is in scope for this stage (done in small,
individually gated steps), rather than deferred into Stage 4 with the GPU work.

## Prompts

> RJC will be responsible for writing this section.
