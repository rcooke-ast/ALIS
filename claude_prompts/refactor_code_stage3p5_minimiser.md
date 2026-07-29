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

**3.5.1 -- Absorb `fcn` into `alfit`. [COMPLETE -- bitwise-verified]**
- `fcn` never varies, so remove it as a constructor/`call()` parameter and have
  the minimiser invoke the model evaluation directly (a method or bound
  reference), owning the `FitState`. Drop the `**functkw` unpacking and the
  generic `call()` indirection at all 6 `alfit(...)` sites. Pure structural,
  bitwise-neutral (identical computation, fewer hops). Keep `damp`/`tie`
  handling intact.

**3.5.2 -- Make the per-iteration setup explicit (`prepare_iteration`). [COMPLETE -- bitwise-verified; + post-fit influence check per RJC]**
- Introduce an `alfit`-owned step run **once per accepted parameter set**
  (before the Jacobian) that computes the per-iteration invariants: the
  influence table `_pinfl`, the base-point component cache (the existing
  `getemab`/`emab` machinery), and -- when `renew_subpix` is on -- the base
  sub-pixel grid. The base residual evaluation and the 2n derivative evaluations
  then consume these. This is **bitwise-neutral by construction**: `_pinfl` is
  *already* computed at the base point (`ddpid is None`) and reused by every
  derivative (see Findings F1); the base cache is already built once
  (Stage 3.1). The task only *relocates* these into an explicit method, turning
  implicit side-effects into a named per-iteration contract. **Define a clean
  `eval_derivative(j)` boundary here** (even though it still internally mutates a
  copied `state` for now) so that completing the return-not-mutate contract in
  Stage 4 (deferred Task 3.5.3) is a localized change.

**3.5.3 -- [DEFERRED to Stage 4, Task 4.0] Return-not-mutate evaluation
contract.**
- Deferred to Stage 4 per RJC (see Stage 4 Task 4.0). Rationale: this is the
  highest bitwise-risk change; its only strict *necessity* is the GPU (a kernel
  cannot mutate shared Python `state`), so it is co-designed with the kernel
  port; and its CPU-only benefit is negligible -- it removes the ~2n/iter shallow
  `copy.copy(state)` in `_minimiser_eval`, but that copy only rebinds ~28
  attribute references (the read-only arrays are shared, not duplicated, so it
  does **not** block Task 4.5 shared memory), against a Jacobian that is ~95%
  compute. Note the residual column is *already* returned (`myfunct` returns
  `[status, (y-modf)/err]`); the copy only isolates the per-call *scratch*
  mutations (`state._modfinal/_contfinal/_zerofinal/_pinfl`). **None of the other
  3.5 tasks depend on this**, so it defers cleanly; Task 3.5.2 leaves the clean
  `eval_derivative(j)` boundary that makes the Stage 4 change localized.

**3.5.4 -- Delete dead code. [COMPLETE -- bitwise-verified]**
- Remove `model_eval.model_func_ddp` (its body opens with
  `msgs.bug("Shifts not implemented in speed-up model_func_ddp")` and is only
  reachable via an unused `main.py` wrapper) and the wrapper; drop the
  `modelem`/`modelab` (`emab[0]`/`emab[1]`) plumbing if it is unused once
  `model_func_ddp` is gone (the derivative path reads only `emab[2]`, the cache).

**3.5.5 -- (CPU win, bitwise-safe, only if low-risk) `renew_subpix` conditional
recompute. [DEFERRED -- see log; bounded perf win, non-trivial predicate on the
bitwise-sensitive derivative path]**
- Only relevant when `run renew_subpix True` (default off, but **on** in the
  real helium34 fits). NOTE the trap (Finding F2): the subpix grid depends on
  fitted widths/resolution, so a blanket "compute once per iteration" hoist is
  **not** bitwise-identical -- it would drop a real derivative dependence. The
  bitwise-safe win is a *conditional* recompute: reuse the base grid for
  derivative perturbations of parameters that do not feed the subpix/resolution
  model, and recompute only for those that do. Implement only if it stays simple
  and does not complicate the GPU seam; otherwise leave a clear note and defer.

**3.5.6 -- Unit tests for this stage's stable surface (do last). [COMPLETE]**
- Per the cross-cutting policy (`claude_prompts/refactor_code_unit_tests.md`),
  add `unit`-marked tests for the new stable surface: the `prepare_iteration`
  invariants and (if done) the `renew_subpix` conditional-recompute predicate.
  (The `eval_derivative` return contract is deferred to Stage 4, so its tests go
  there.) Keep them fast and isolated; the existing `unit` CI job picks them up.

## Findings (investigation results -- authoritative for the refactor)

**F1 (CORRECTED during 3.5.1 -- see stage log) -- `_pinfl` is value-dependent
*in principle*, but the fit actually uses a FIXED (p0) table; the per-iteration
recompute is silently discarded.**
`load.load_par_influence` (`load.py:1718`) decides, per region, which parameters
influence it by testing whether each component's lines fall inside the region's
window. In `functions/voigt.py:set_vars` (the main absorption function) the
influence list is returned **empty** when the component's *redshifted* lines lie
outside the window:
`nw = where(Wavelength*(1+pt[1]) in [wvmin,wvmax]); if size==0 and getinfl: return [], []`.
So influence *could* depend on parameter values via the component **redshift**
`pt[1]` and the **shift** params (`wvrng`). (The base class `base.py:set_vars`
ignores `wvrng` -- so this value-dependence is specific to the line functions:
`voigt`, `gaussian`, `lineemission`, `splineabs`.)

**However**, the fit never sees that value-dependence: `_minimiser_eval` does a
shallow `copy.copy(state)` per call, and the per-iteration recompute at
`model_eval.py:120` (base call only) rebinds `_pinfl` on that *throwaway copy*.
The original `functkw['state']._pinfl` -- built once at setup (`main.py:178`, from
`p0`) -- is never updated, and that is the table the derivative workers pickle and
the subset-slicer reads. The base call never *reads* `_pinfl` (the `ddpid not in
_pinfl` skips are derivative-only), so line 120 writes a value that is never read
-> **effectively dead in the fit path** (empirically confirmed: disabling it
leaves metal_line_abs bitwise-identical). Net: `_pinfl` is a **per-fit** invariant
(p0) in practice.
=> Consequences for 3.5.2: (i) `prepare_iteration` can compute `_pinfl` once at
fit start (matching current behaviour) -- it is already effectively per-fit;
(ii) the real per-iteration invariant that must be recomputed each accepted step
is the component **cache** (already flowing via `emab`), not `_pinfl`; (iii) a
line crossing a region edge mid-fit is NOT tracked (the p0 table is frozen) -- a
latent science-correctness question, flagged for RJC (kept as-is preserves
bitwise identity; changing it would need new golden references).

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
directly (not through the minimiser). After 3.5.1, should these route
through the same standalone `model_eval` entry point (one evaluation path), or
keep the thin `main.py` wrappers for these non-fit uses? Recommendation: unify
on the standalone path to remove duplication, provided Stage 0 stays bitwise.

**Response:** Agreed, unify on the standalone path. The thin `main.py` wrappers
are redundant and can be removed, as long as the Stage 0 regression suite confirms
bitwise identity.

**Q3.5.6 -- Risk tolerance for Task 3.5.3 (resolved).** The return-instead-of-
mutate contract is the highest-value GPU enabler but touches the numerically
sensitive result-passing. Confirm it is in scope for this stage (done in small,
individually gated steps), rather than deferred into Stage 4 with the GPU work.

**Response:** Deferred to Stage 4 (new Task 4.0). The other 3.5 tasks do not
depend on it, and its necessity is GPU-specific (a kernel cannot mutate shared
Python state), so it is co-designed with the kernel port. Stage 3.5 keeps the
current `copy.copy(state)` and mutate-based eval internals; Task 3.5.2 defines
the clean `eval_derivative(j)` boundary so the Stage 4 change is localized.

## Prompts

1. Please read this doc, including my responses to your queries. Ask further queries if needed. If you have no further queries, please proceed to implement the tasks in order, logging each in `ALIS/claude_prompts/logs/refactor_code_stage3p5_minimiser_log.md`.  Every task is gated by the Stage 0 regression suite and must be **bitwise-identical** to the current output.

