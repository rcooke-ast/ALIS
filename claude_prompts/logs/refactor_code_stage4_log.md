# Log -- Stage 4 (GPU support and modularity)

Plan: `claude_prompts/refactor_code_stage4.md`.
Bitwise A/B baselines from the committed post-3.5 code (in `/tmp/s40base/`):
metal_line_abs full fit and DH_orders 1-iteration (volatile header lines
stripped).

## Task 4.0 -- Return-not-mutate contract + `_pinfl` freeze (IN PROGRESS)

Completes the deferred Stage 3.5.3. Two bitwise-gated steps:

**Step 1 -- Freeze `_pinfl` (per-fit constant, never recomputed during the fit).**
- `model_eval.model_func`: removed the base-call per-iteration recompute
  (`state._pinfl = load_par_influence(...)`, old line 120). `_pinfl` is now only
  ever set by the callers before evaluation.
- `main.py`: the two sim `beginfrom` branches now set
  `self._pinfl = load.load_par_influence(self, modpass['p0'])` explicitly before
  evaluating/snapshotting the loaded model (they previously relied on the
  now-removed recompute). The standard fit (main.py:178) and iterate branch
  (main.py:289) already set `_pinfl` before their fits.
- Rationale: on the fit path the old recompute landed on the throwaway
  `copy.copy(state)` and was discarded (Stage 3.5 Finding F1, empirically
  confirmed), so removing it is bitwise-neutral -- and it is the prerequisite for
  Step 2 (without the copy, that write would otherwise mutate the shared table
  every iteration).

**Step 1 verified:** metal_line_abs + DH_orders 1-iteration **BITWISE-IDENTICAL**.

**Step 2 -- Drop the per-call `copy.copy(state)` in `_minimiser_eval`.**
- `model_eval._minimiser_eval` now passes `state` straight through (no
  `copy.copy`). Safe because: (i) `_pinfl` is frozen (Step 1), so the base call no
  longer mutates the shared influence table; (ii) the remaining state writes
  (`_modfinal`/`_contfinal`/`_zerofinal`) are per-call scratch rebuilt from
  scratch each call and never read back on the fit path (main-process base call
  mutates the FitState harmlessly; workers hold a private unpickled state); (iii)
  the residual column is already returned, not mutated. This is the return-not-
  mutate seam the GPU kernel path needs; any remaining incidental scratch writes
  in the derivative branch will disappear when that path becomes a device kernel
  (4.2/4.3).
- Verified: metal_line_abs (full multi-iteration fit) + DH_orders 1-iteration
  **BITWISE-IDENTICAL** vs baseline.
- Unit test: `tests/test_model_eval.py::test_minimiser_eval_does_not_copy_state`
  (monkeypatches `myfunct`, asserts the identical `state` object passes through)
  -- the pure-surface check of the 4.0 contract; numerical equivalence is covered
  by the Stage 0 bitwise gate. (Deferred 3.5.6 eval-contract test.)

## Task 4.0 -- COMPLETE, Stage 0 gate GREEN

Full suite `pytest --run-slow -k "not (DH and J0814p5029)"`: **142 passed, 1
deselected in 30:56**. Unit batch `pytest -m unit`: **60 passed**. All changes
bitwise-identical to the pre-4.0 baseline (metal_line_abs full fit + DH_orders
1-iteration verified per step; the full multi-iteration suite confirms). No
reference/golden files changed. Task 4.0 also completes the deferred Stage 3.5.3
(return-not-mutate) and freezes `_pinfl` per RJC's set-once invariant. The
minimiser derivative path is now a return-based evaluation with no per-call
`copy.copy` -- the seam the GPU kernel path (4.2/4.3) will build on.
