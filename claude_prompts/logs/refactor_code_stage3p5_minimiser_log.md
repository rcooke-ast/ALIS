# Log -- Stage 3.5 (minimiser <-> model-eval seam)

Plan: `claude_prompts/refactor_code_stage3p5_minimiser.md`.
Bitwise A/B baselines captured from the committed code (in `/tmp/s35base/`):
metal_line_abs full fit (Initial chi2 512.367976) and DH_orders 1-iteration
(Bestfit chi2 92636.619932), volatile header lines stripped.

## Task 3.5.1 -- Absorb `fcn` into `alfit` (COMPLETE, bitwise-verified)

`fcn` was always `model_eval._minimiser_eval` at all 6 `alfit(...)` sites, so it
is no longer threaded as a parameter; the minimiser calls the model evaluation
directly and owns the state (`self.functkw = {x, y, err, FitState}`).
- `minimise.py`: removed `fcn` from `__init__`, `call`, `fdjac2`, `funcderiv`,
  `defiter`, and the pickled worker `state` tuple; `call`/`_worker_call` now call
  `model_eval._minimiser_eval` directly; added `from alis import model_eval` (no
  circular import -- load/save/logger/model_eval do not import minimise); removed
  the obsolete `if fcn is None` guard and the dead `_WORKER`/`_worker_init` (the
  abandoned initializer); updated stale `str(fcn)` messages.
- `main.py` / `simulate.py`: dropped the `model_eval._minimiser_eval` first arg at
  all `alfit(...)` sites (active + commented).
- Verified: `import` OK; **metal_line_abs BITWISE-IDENTICAL**; **DH_orders
  1-iteration BITWISE-IDENTICAL** (parallel chunked Jacobian path) vs baseline.

## Finding F1-corrected -- the base-call `_pinfl` recompute is DISCARDED

While scoping 3.5.2 I found (and empirically confirmed) that the doc's Finding F1
was inaccurate about granularity. `_minimiser_eval` does `copy.copy(state)`
(shallow) per call, and `model_eval.py:120` (`state._pinfl =
load_par_influence(state, p)`, base call only) rebinds the attribute on that
*throwaway copy*. So:
- the original `functkw['state']._pinfl` (built once at setup, `main.py:178`, from
  `p0`) is never updated during the fit;
- the derivative workers pickle `functkw['state']`, so they use the **setup-time
  (p0) `_pinfl`**; the subset-slicing `_param_spsn_map(self.functkw, ...)` also
  reads the p0 table;
- the base call (`ddpid is None`) never *reads* `_pinfl` (the `ddpid not in
  _pinfl` skips are derivative-only), so line 120 writes a value that is never
  read -> effectively **dead in the fit path**.
Empirical proof: gating line 120 behind an env flag and disabling it leaves
metal_line_abs **bitwise-identical**.
Implication: `_pinfl` is effectively a **per-fit** invariant (p0), not a
per-iteration one. Good for the GPU seam (compute once at fit start). But it also
means the value-dependence noted in F1 (redshift/shift moving a line across a
region edge) is **not** tracked during the fit -- the p0 influence table is used
throughout. This is the current (reference-matching) behaviour; "fixing" it to be
per-iteration would change results (non-bitwise) and is a separate decision.
=> Consulting RJC before 3.5.2 on how to treat `_pinfl` and the dead line 120.

## RJC direction (query during 3.5.1) -- influence set-once + post-fit check

RJC: parameter influence must NOT change during the fit ("if it changes, something
has gone wrong"); never update it during the fit; set it before the fit and check
it after, warning if the end-of-fit influence differs from the start. Line 120:
verify other paths then remove.

Verification result: the sim `beginfrom` path (main.py:231/243 -> simulate.py:87
`FitState.from_orchestrator`) relies on line 120 writing `self._pinfl` for the
*loaded* model before the sim snapshot, so removing line 120 is NOT bitwise-safe
there. Decision: LEAVE line 120 (it is already discarded in the fit path -> the
fit never updates influence, satisfying RJC's principle; it is load-bearing only
in the sim setup). Documented; not removed.

## Task 3.5.2 -- prepare_iteration + influence check (COMPLETE, bitwise-verified)

Part A (structural seam): added `alfit.prepare_iteration(params)` -- the explicit
per-iteration step that evaluates the model at the accepted params and stores the
one per-iteration invariant, the component cache, in `self._emab`
(`[modelem, modelab, compcache]`). The two base evaluations (initial + post-step)
now call it; `fdjac2` reads `self._emab` instead of a threaded `emab` param
(dropped from its signature). `_pinfl` is left as the per-fit invariant it already
is (Finding F1-corrected); the sub-pixel grid stays per-call (3.5.5). This is the
CPU/GPU seam (prepare once, derivative kernels consume).
Part B (RJC's diagnostic): added `load.check_par_influence(slf, parin, refpinfl)`
-- recomputes influence at the best-fit params and warns per region if it differs
from the start-of-fit `_pinfl`; called after the standard fit in main.py.

Verified: metal_line_abs and DH_orders 1-iteration **BITWISE-IDENTICAL**; the
influence check logs "Parameter influence is stable" on metal_line_abs and does
not alter `.mod.out` (console diagnostic only).

## Task 3.5.4 -- Delete dead code (COMPLETE, bitwise-verified)

- Deleted `model_eval.model_func_ddp` (188 lines; a dead bug-stub) and its
  `main.py` wrapper. It was the sole reader of `emab[0]`/`emab[1]`
  (modelem/modelab), so `model_func`'s getemab return was trimmed to
  `[None, None, compcache]` (the downstream derivative path + Jacobian slicer read
  only `emab[2]`; the workers already dropped [0]/[1] via `_slice_emab`). Removed
  the now-unused `from alis import save` in model_eval.py.
- Verified: metal_line_abs and DH_orders 1-iteration **BITWISE-IDENTICAL**.

## Task 3.5.5 -- renew_subpix conditional recompute (DEFERRED, with note)

Deferred per RJC's "only if low-risk / defer perf" guidance and Finding F2. A
bitwise-safe win requires a *conditional* recompute keyed on whether the
perturbed parameter feeds the subpix/resolution model (a blanket per-iteration
hoist would drop a real derivative dependence and break bitwise identity). That
predicate is non-trivial and sits on the numerically-sensitive derivative path,
for a bounded CPU gain on the minority of fits that set `run renew_subpix True`
(helium34). Not worth the bitwise risk during this structural stage; left as a
clear note for a future perf pass (or Stage 4, alongside the GPU buffers).

## Task 3.5.6 -- Unit tests (COMPLETE)

The new stable surface (`prepare_iteration`, `check_par_influence`) needs the full
model stack, so the genuinely pure/testable piece -- the influence-table
comparison -- was extracted as `load.pinfl_changed(refpinfl, newpinfl)` (returns
the list of changed `(sp, sn)`), and `check_par_influence` now uses it. Added 4
`unit` tests in `tests/test_load_units.py` (identical -> empty; set/order
insensitivity; added-parameter detection; multi-region reporting). `pytest
tests/test_load_units.py -m unit` -> 16 passed.

## Stage 3.5 -- COMPLETE, Stage 0 gate GREEN

Full suite `pytest --run-slow -k "not (DH and J0814p5029)"`: **141 passed, 1
deselected in 32:17**. Unit batch `pytest -m unit`: **59 passed**. All Stage 3.5
changes are bitwise-identical to the pre-stage baseline (verified per task on
metal_line_abs + DH_orders 1-iteration, and confirmed by the reference gate).
Tasks 3.5.1/3.5.2/3.5.4/3.5.6 done; 3.5.3 deferred to Stage 4 (Task 4.0); 3.5.5
deferred (note above). No reference/golden files changed.
